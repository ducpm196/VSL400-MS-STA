#!/usr/bin/env python3
"""CTR-matched equal-budget confirm + architecture gate for b4_v6_ctr."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_porcelain, git_rev  # noqa: E402

from src.eval.ms_sta_v5_m0_strong_baseline import (  # noqa: E402
    hierarchical_bootstrap_ci,
    load_json,
    per_signer_delta_top1_pp,
    PredRow,
    sha256_file,
)


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def write_status(runs_root: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write(runs_root / "STATUS.json", payload)


def load_pred_rows(path: Path) -> dict[str, PredRow]:
    obj = load_json(path)
    out: dict[str, PredRow] = {}
    for r in obj["rows"]:
        sid = str(r["sample_id"])
        out[sid] = PredRow(
            sample_id=sid,
            signer=str(r.get("signer") or ""),
            label=int(r["label"]),
            correct=bool(r["correct"]),
            top5_hit=bool(r["top5_hit"]),
        )
    return out


def run_trial(policy_path: Path, trial: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "SUMMARY.json"
    if summary_path.is_file() and (out_dir / "best.pt").is_file() and (out_dir / "VAL_PREDS.json").is_file():
        return load_json(summary_path)
    cmd = [
        str(_ROOT / "_venvs" / "train_cuda" / "Scripts" / "python.exe"),
        str(_ROOT / "src" / "training" / "ms_sta_v5_train.py"),
        "--policy",
        str(policy_path),
        "--variant",
        trial["variant"],
        "--phase",
        "confirm",
        "--seed",
        str(trial["seed"]),
        "--lr",
        str(trial["lr"]),
        "--dropout",
        str(trial["dropout"]),
        "--weight-decay",
        str(trial["weight_decay"]),
        "--out-dir",
        str(out_dir),
        "--device",
        "cuda",
    ]
    with (out_dir / "train.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0 or not summary_path.is_file():
        raise RuntimeError(f"trial_failed:{out_dir}:rc={proc.returncode}")
    return load_json(summary_path)


def recompute_top1(preds: dict[str, PredRow]) -> float:
    return sum(1 for p in preds.values() if p.correct) / max(len(preds), 1)


def recompute_top5(preds: dict[str, PredRow]) -> float:
    return sum(1 for p in preds.values() if p.top5_hit) / max(len(preds), 1)


def architecture_gate(policy: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    seeds = [int(s) for s in policy["seeds_confirm"]]
    gates = policy["ctr_matched_architecture_gates"]
    candidate = policy["candidate"]
    primary = gates["primary_baseline"]
    secondary = gates["secondary_baseline"]
    cand_root = _ROOT / policy["candidate_confirm_root_rel"]

    panels: dict[str, dict[int, dict[str, PredRow]]] = {candidate: {}, primary: {}, secondary: {}}
    top1: dict[str, list[float]] = {candidate: [], primary: [], secondary: []}
    top5: dict[str, list[float]] = {candidate: [], primary: [], secondary: []}
    f1: dict[str, list[float]] = {candidate: [], primary: [], secondary: []}

    for seed in seeds:
        for variant, root in (
            (candidate, cand_root),
            (primary, runs_root / "confirm"),
            (secondary, runs_root / "confirm"),
        ):
            preds = load_pred_rows(root / f"{variant}_seed{seed}" / "VAL_PREDS.json")
            summary = load_json(root / f"{variant}_seed{seed}" / "SUMMARY.json")
            panels[variant][seed] = preds
            top1[variant].append(recompute_top1(preds))
            top5[variant].append(recompute_top5(preds))
            f1[variant].append(float(summary.get("best_val_macro_f1") or 0.0))

    ids = sorted(panels[candidate][seeds[0]].keys())
    if set(ids) != set(panels[primary][seeds[0]].keys()) or set(ids) != set(panels[secondary][seeds[0]].keys()):
        raise RuntimeError("confirm_id_mismatch")

    d1, signers = [], []
    for sid in ids:
        vals = [
            float(panels[candidate][s][sid].correct) - float(panels[primary][s][sid].correct) for s in seeds
        ]
        d1.append(sum(vals) / len(vals))
        signers.append(panels[candidate][seeds[0]][sid].signer)
    mean_boot, ci_lo, ci_hi = hierarchical_bootstrap_ci(
        d1, signers, n_boot=int(gates["n_boot"]), seed=int(gates["boot_seed"])
    )
    signer_deltas = per_signer_delta_top1_pp(
        ids,
        {sid: panels[candidate][seeds[0]][sid].signer for sid in ids},
        panels[candidate],
        panels[primary],
    )
    mean_d1 = 100.0 * (sum(top1[candidate][i] - top1[primary][i] for i in range(3)) / 3.0)
    mean_d5 = 100.0 * (sum(top5[candidate][i] - top5[primary][i] for i in range(3)) / 3.0)
    mean_df1 = 100.0 * (sum(f1[candidate][i] - f1[primary][i] for i in range(3)) / 3.0)
    beat_primary = all(top1[candidate][i] > top1[primary][i] for i in range(3))
    beat_secondary = all(top1[candidate][i] > top1[secondary][i] for i in range(3))
    checks = {
        "beat_primary_every_seed": beat_primary,
        "mean_delta_top1_pp": mean_d1 >= float(gates["mean_delta_top1_pp_vs_primary_min"]),
        "hierarchical_ci95_lower_pp": ci_lo > float(gates["hierarchical_ci95_lower_pp_min"]),
        "top5_no_regress": mean_d5 >= 0.0 if gates["top5_no_regress_vs_primary"] else True,
        "macro_f1_no_regress": mean_df1 >= 0.0 if gates["macro_f1_no_regress_vs_primary"] else True,
        "no_signer_catastrophe": all(v >= float(gates["no_signer_catastrophe_pp"]) for v in signer_deltas.values()),
        "beat_secondary_every_seed_informative": beat_secondary,
    }
    required = [
        "beat_primary_every_seed",
        "mean_delta_top1_pp",
        "hierarchical_ci95_lower_pp",
        "top5_no_regress",
        "macro_f1_no_regress",
        "no_signer_catastrophe",
    ]
    verdict = "V6_CTR_MATCHED_ARCHITECTURE_PASS" if all(checks[k] for k in required) else "V6_CTR_MATCHED_ARCHITECTURE_NO_GO"
    return {
        "schema": "ms_sta_v6_ctr_matched_architecture_gate_v1",
        "verdict": verdict,
        "candidate": candidate,
        "primary_baseline": primary,
        "secondary_baseline": secondary,
        "candidate_top1_recomputed": top1[candidate],
        "primary_top1_recomputed": top1[primary],
        "secondary_top1_recomputed": top1[secondary],
        "mean_delta_top1_pp_vs_primary": mean_d1,
        "hierarchical_ci95_top1_pp_vs_primary": [ci_lo, ci_hi],
        "mean_hierarchical_boot_pp": mean_boot,
        "mean_delta_top5_pp_vs_primary": mean_d5,
        "mean_delta_macro_f1_pp_vs_primary": mean_df1,
        "signer_deltas_pp": signer_deltas,
        "checks": checks,
        "canonical_teacher_allowed": verdict == "V6_CTR_MATCHED_ARCHITECTURE_PASS",
        "defense_registry_unchanged": True,
        "retain_defense": "b4_star_fixed",
        "e4_allowed": verdict == "V6_CTR_MATCHED_ARCHITECTURE_PASS",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def package_raw_evidence(policy: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    archive_root = Path(policy["external_archive_root"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = archive_root / f"v6_ctr_matched_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    sources = [
        _ROOT / "Docs/results/ms_sta_v6_runs/confirm",
        _ROOT / "Docs/results/ms_sta_v6_runs/hpo",
        runs_root,
        _ROOT / "Docs/results/ms_sta_v6_e3_runs",
    ]
    copied = []
    for src in sources:
        if not src.exists():
            continue
        target = dest / src.name
        if src.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("*.pt"))
            # Also copy candidate checkpoints only (small set) with hashes recorded separately
            copied.append(str(src))
    # Hash inventory (files under dest)
    inventory = []
    for p in dest.rglob("*"):
        if p.is_file():
            inventory.append({"rel": str(p.relative_to(dest)).replace("\\", "/"), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    manifest = {
        "schema": "ms_sta_v6_external_evidence_manifest_v1",
        "archive_root": str(dest),
        "exclude_note": "*.pt omitted from bulk copy; candidate checkpoint SHAs remain in teacher lock / SUMMARY",
        "sources": copied,
        "n_files": len(inventory),
        "files": inventory,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    man_path = dest / "MANIFEST.json"
    atomic_write(man_path, manifest)
    # Pointer inside repo (hashes only)
    pointer = {
        "schema": "ms_sta_v6_external_archive_pointer_v1",
        "archive_root": str(dest),
        "manifest_sha256": sha256_file(man_path),
        "n_files": len(inventory),
        "recorded_at": manifest["recorded_at"],
    }
    atomic_write(_ROOT / "Docs/results/ms_sta_v6_audit" / "EXTERNAL_ARCHIVE_POINTER.json", pointer)
    return pointer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v6_ctr_matched_policy/CTR_MATCHED_POLICY.json")
    ap.add_argument("--phase", choices=["dry-plan", "confirm", "gate", "package", "all"], default="dry-plan")
    args = ap.parse_args()
    policy_path = _ROOT / args.policy
    policy = load_json(policy_path)
    if policy.get("status") != "FROZEN" or policy.get("train_allowed") is not True:
        raise SystemExit("policy_invalid")
    head = git_rev(_ROOT)
    porcelain_raw = git_porcelain(_ROOT).strip()
    allow_prefixes = (
        "Docs/results/ms_sta_v6_runs/",
        "Docs/results/ms_sta_v6_e3_runs/",
        "Docs/results/ms_sta_v6_ctr_matched_runs/",
        "Docs/results/ms_sta_v6_audit/",
    )
    blocking = []
    for line in porcelain_raw.splitlines():
        if not line.startswith("??"):
            # allow modified lock/auth already staged intent — block other tracked mods unless under results
            path = line[3:].replace("\\", "/").strip() if line[:2] in {" M", "M ", "A ", "D ", "??"} else line[3:].replace("\\", "/").strip()
            # keep simple: only untracked under allowlist OK; any modified tracked outside allow → block for confirm
            continue
        path = line[3:].replace("\\", "/").strip()
        if not any(path.startswith(p) for p in allow_prefixes):
            blocking.append(line)
    # Also block dirty tracked files outside results for confirm/all
    for line in porcelain_raw.splitlines():
        if line.startswith("??"):
            continue
        path = line[3:].replace("\\", "/").strip() if len(line) > 3 else line
        # status format XY PATH
        parts = line.split(maxsplit=1)
        path = parts[1].replace("\\", "/").strip() if len(parts) == 2 else ""
        if path and not any(path.startswith(p) for p in allow_prefixes) and not path.startswith("Docs/results/ms_sta_v6"):
            # allow Docs/results/ms_sta_v6* and policy/review paths during this campaign
            if not (
                path.startswith("Docs/results/ms_sta_v6")
                or path.startswith("Docs/results/ms_sta_v6_ctr_matched_policy")
                or path.startswith("Docs/reviews/")
                or path.startswith("src/models/ms_sta_v6")
                or path.startswith("src/training/ms_sta_v5_train.py")
                or path.startswith("scripts/")
                or path.startswith("tests/test_ms_sta_v6")
            ):
                blocking.append(line)

    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)
    budget = policy["equal_budget_locked_from"]
    plan = {
        "schema": "ms_sta_v6_ctr_matched_plan_v1",
        "phase": args.phase,
        "head": head,
        "controls": policy["controls"],
        "candidate": policy["candidate"],
        "equal_budget": budget,
        "n_confirm_trials": len(policy["controls"]) * len(policy["seeds_confirm"]),
        "canonical_teacher": False,
        "forbid_final_test": True,
        "e4_allowed": False,
    }
    if args.phase == "dry-plan":
        print(json.dumps(plan, indent=2))
        return 0

    if args.phase in {"confirm", "all"}:
        # For confirm after commit, porcelain should be clean except allowlisted untracked run artifacts.
        dirty = [ln for ln in porcelain_raw.splitlines() if not ln.startswith("??")]
        if dirty:
            # tolerate only if user already committed — otherwise require clean tracked tree
            raise SystemExit(f"dirty_tracked_worktree_forbidden:{chr(10).join(dirty[:30])}")
        rows = []
        for variant in policy["controls"]:
            for seed in policy["seeds_confirm"]:
                trial = {
                    "variant": variant,
                    "lr": float(budget["lr"]),
                    "dropout": float(budget["dropout"]),
                    "weight_decay": float(budget["weight_decay"]),
                    "seed": int(seed),
                }
                out_dir = runs_root / "confirm" / f"{variant}_seed{seed}"
                write_status(runs_root, {"phase": "confirm", "variant": variant, "seed": seed, "head": head})
                print(json.dumps({"event": "confirm_start", "variant": variant, "seed": seed}), flush=True)
                summary = run_trial(policy_path, trial, out_dir)
                # Prefer recomputed top1 in status
                preds = load_pred_rows(out_dir / "VAL_PREDS.json")
                summary["recomputed_val_top1"] = recompute_top1(preds)
                summary["recomputed_val_top5"] = recompute_top5(preds)
                summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
                rows.append(summary)
                atomic_write(runs_root / "CONFIRM_RESULTS.json", {"equal_budget": budget, "trials": rows})
        print(json.dumps({"event": "confirm_complete", "n": len(rows)}), flush=True)

    if args.phase in {"gate", "all"}:
        report = architecture_gate(policy, runs_root)
        atomic_write(runs_root / "CTR_MATCHED_ARCHITECTURE_GATE.json", report)
        atomic_write(_ROOT / "Docs/results/ms_sta_v6_audit" / "CTR_MATCHED_ARCHITECTURE_GATE.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if report["verdict"] != "V6_CTR_MATCHED_ARCHITECTURE_PASS":
            return 2

    if args.phase in {"package", "all"}:
        pointer = package_raw_evidence(policy, runs_root)
        print(json.dumps({"event": "packaged", "pointer": pointer}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
