#!/usr/bin/env python3
"""MS-STA v6 Stage-A orchestrator — HPO / confirm / teacher gate (no test/demo/registry)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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


def hpo_grid(policy: dict[str, Any]) -> list[dict[str, Any]]:
    hpo = policy["hpo"]
    trials = []
    for variant, lr in itertools.product(policy["variants"], hpo["lr"]):
        trials.append(
            {
                "variant": variant,
                "lr": lr,
                "dropout": float(hpo["dropout"]),
                "weight_decay": float(hpo["weight_decay"]),
                "seed": 42,
            }
        )
    if len(trials) > int(hpo["max_runs"]):
        raise RuntimeError(f"hpo_grid_exceeds_max_runs:{len(trials)}")
    return trials


def run_trial(policy_path: Path, trial: dict[str, Any], phase: str, out_dir: Path) -> dict[str, Any]:
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
        phase,
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
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0 or not summary_path.is_file():
        raise RuntimeError(f"trial_failed:{out_dir}:rc={proc.returncode}")
    return load_json(summary_path)


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


def select_per_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        v = r["variant"]
        if v not in best or float(r["best_val_top1"]) > float(best[v]["best_val_top1"]):
            best[v] = r
    return best


def evaluate_teacher_gate(policy: dict[str, Any], runs_root: Path, teacher_variant: str) -> dict[str, Any]:
    seeds = [int(s) for s in policy["seeds_confirm"]]
    gates = policy["teacher_accuracy_gates"]
    roles = {
        "teacher": teacher_variant,
        "rgb": "rgb_strong",
        "mlp": "b3_mlp_strong",
    }
    panels: dict[str, dict[int, dict[str, PredRow]]] = {k: {} for k in roles}
    top1: dict[str, list[float]] = {k: [] for k in roles}
    top5: dict[str, list[float]] = {k: [] for k in roles}
    f1: dict[str, list[float]] = {k: [] for k in roles}
    for role, variant in roles.items():
        for seed in seeds:
            summary = load_json(runs_root / "confirm" / f"{variant}_seed{seed}" / "SUMMARY.json")
            preds = load_pred_rows(runs_root / "confirm" / f"{variant}_seed{seed}" / "VAL_PREDS.json")
            panels[role][seed] = preds
            top1[role].append(float(summary["best_val_top1"]))
            top5[role].append(float(summary.get("best_val_top5") or 0.0))
            f1[role].append(float(summary.get("best_val_macro_f1") or 0.0))

    ids = sorted(panels["teacher"][seeds[0]].keys())
    if set(ids) != set(panels["mlp"][seeds[0]].keys()) or set(ids) != set(panels["rgb"][seeds[0]].keys()):
        raise RuntimeError("confirm_id_mismatch_across_variants")

    d1, signers = [], []
    for sid in ids:
        signer = panels["teacher"][seeds[0]][sid].signer
        vals = []
        for seed in seeds:
            t = panels["teacher"][seed][sid]
            m = panels["mlp"][seed][sid]
            vals.append(float(t.correct) - float(m.correct))
        d1.append(sum(vals) / len(vals))
        signers.append(signer)
    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(
        d1, signers, n_boot=int(gates["n_boot"]), seed=int(gates["boot_seed"])
    )
    signer_map = {sid: panels["teacher"][seeds[0]][sid].signer for sid in ids}
    signer_deltas = per_signer_delta_top1_pp(ids, signer_map, panels["teacher"], panels["mlp"])

    mean_delta_mlp = 100.0 * (sum(top1["teacher"][i] - top1["mlp"][i] for i in range(3)) / 3.0)
    mean_delta_top5 = 100.0 * (sum(top5["teacher"][i] - top5["mlp"][i] for i in range(3)) / 3.0)
    mean_delta_f1 = 100.0 * (sum(f1["teacher"][i] - f1["mlp"][i] for i in range(3)) / 3.0)
    beat_rgb = all(top1["teacher"][i] > top1["rgb"][i] for i in range(3))
    beat_mlp = all(top1["teacher"][i] > top1["mlp"][i] for i in range(3))

    checks = {
        "beat_rgb_strong_every_seed": beat_rgb if gates["beat_rgb_strong_every_seed"] else True,
        "beat_b3_mlp_strong_every_seed": beat_mlp if gates["beat_b3_mlp_strong_every_seed"] else True,
        "mean_delta_top1_pp_vs_b3_mlp": mean_delta_mlp >= float(gates["mean_delta_top1_pp_vs_b3_mlp_min"]),
        "hierarchical_ci95_lower_pp": ci_lo > float(gates["hierarchical_ci95_lower_pp_min"]),
        "top5_no_regress_vs_b3_mlp": mean_delta_top5 >= 0.0 if gates["top5_no_regress_vs_b3_mlp"] else True,
        "macro_f1_no_regress_vs_b3_mlp": mean_delta_f1 >= 0.0 if gates["macro_f1_no_regress_vs_b3_mlp"] else True,
        "no_signer_catastrophe": all(
            v >= float(gates["no_signer_catastrophe_pp"]) for v in signer_deltas.values()
        ),
    }
    verdict = "V6_TEACHER_PASS" if all(checks.values()) else "V6_TEACHER_NO_GO"
    return {
        "schema": "ms_sta_v6_teacher_gate_v1",
        "stage_label": policy.get("stage_label"),
        "verdict": verdict,
        "teacher_variant": teacher_variant,
        "top1": top1,
        "mean_delta_top1_pp_vs_b3_mlp": mean_delta_mlp,
        "hierarchical_ci95_top1_pp_vs_b3_mlp": [ci_lo, ci_hi],
        "mean_delta_top5_pp_vs_b3_mlp": mean_delta_top5,
        "mean_delta_macro_f1_pp_vs_b3_mlp": mean_delta_f1,
        "per_signer_delta_top1_pp_vs_b3_mlp": signer_deltas,
        "checks": checks,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v6_policy/V6_POLICY.json")
    ap.add_argument("--phase", choices=["dry-plan", "hpo", "confirm", "gate", "all"], default="dry-plan")
    args = ap.parse_args()

    policy_path = _ROOT / args.policy
    policy = load_json(policy_path)
    auth = load_json(_ROOT / "Docs/results/ms_sta_v6/V6_AUTHORIZATION.json")
    if policy.get("status") != "FROZEN" or policy.get("train_allowed") is not True:
        raise SystemExit("policy_invalid")
    if policy.get("forbid_final_test") is not True:
        raise SystemExit("policy_must_forbid_final_test")
    if auth.get("status") != "AUTHORIZED" or auth.get("allow_final_test") is not False:
        raise SystemExit("auth_invalid")
    if sha256_file(_ROOT / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json") != policy["bound_validation_lock_v2_sha256"]:
        raise SystemExit("lock_v2_sha_mismatch")

    head = git_rev(_ROOT)
    porcelain_raw = git_porcelain(_ROOT).strip()
    runs_rel = policy["runs_root_rel"].replace("\\", "/").rstrip("/") + "/"
    blocking = []
    for line in porcelain_raw.splitlines():
        path = line[3:].replace("\\", "/").strip()
        if line.startswith("??") and (path.startswith(runs_rel) or path.rstrip("/") == runs_rel.rstrip("/")):
            continue
        blocking.append(line)
    porcelain = "\n".join(blocking).strip()
    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)
    trials = hpo_grid(policy)
    plan = {
        "schema": "ms_sta_v6_orchestrator_plan_v1",
        "phase": args.phase,
        "head": head,
        "porcelain_clean": porcelain == "",
        "n_hpo_trials": len(trials),
        "variants": policy["variants"],
        "stage_label": policy.get("stage_label"),
        "forbid_final_test": True,
        "policy_sha256": sha256_file(policy_path),
    }
    if args.phase == "dry-plan":
        print(json.dumps(plan, indent=2))
        return 0
    if porcelain:
        raise SystemExit(f"dirty_worktree_forbidden:{porcelain[:400]}")

    if args.phase in {"hpo", "all"}:
        rows = []
        for i, trial in enumerate(trials):
            tag = f"{trial['variant']}_lr{trial['lr']}_do{trial['dropout']}_wd{trial['weight_decay']}_s{trial['seed']}"
            out_dir = runs_root / "hpo" / tag
            write_status(runs_root, {"phase": "hpo", "i": i, "n": len(trials), "tag": tag, "head": head})
            print(json.dumps({"event": "hpo_trial_start", "i": i, "n": len(trials), "tag": tag}), flush=True)
            summary = run_trial(policy_path, trial, "hpo", out_dir)
            summary["tag"] = tag
            summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
            rows.append(summary)
            atomic_write(
                runs_root / "HPO_LEADERBOARD.json",
                {"trials": rows, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            )
        selected = select_per_variant(rows)
        # Teacher pick among primary_teacher_candidates by tune top1.
        teacher = max(
            (selected[v] for v in policy["primary_teacher_candidates"] if v in selected),
            key=lambda r: float(r["best_val_top1"]),
        )
        atomic_write(
            runs_root / "HPO_SELECTED.json",
            {
                "selected_per_variant": selected,
                "teacher_candidate": teacher,
                "contrast_required_confirm": ["rgb_strong", "b3_mlp_strong", teacher["variant"]],
            },
        )
        print(json.dumps({"event": "hpo_complete", "teacher": teacher["variant"], "tune_top1": teacher["best_val_top1"]}, indent=2), flush=True)

    if args.phase in {"confirm", "all"}:
        sel = load_json(runs_root / "HPO_SELECTED.json")
        teacher_v = sel["teacher_candidate"]["variant"]
        confirm_variants = sorted(set(sel["contrast_required_confirm"]))
        confirm_rows = []
        for variant in confirm_variants:
            hyp = sel["selected_per_variant"][variant]
            for seed in policy["seeds_confirm"]:
                trial = {
                    "variant": variant,
                    "lr": hyp["lr"],
                    "dropout": hyp["dropout"],
                    "weight_decay": hyp["weight_decay"],
                    "seed": seed,
                }
                out_dir = runs_root / "confirm" / f"{variant}_seed{seed}"
                write_status(runs_root, {"phase": "confirm", "variant": variant, "seed": seed, "head": head})
                print(json.dumps({"event": "confirm_start", "variant": variant, "seed": seed}), flush=True)
                summary = run_trial(policy_path, trial, "confirm", out_dir)
                summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
                confirm_rows.append(summary)
                atomic_write(
                    runs_root / "CONFIRM_RESULTS.json",
                    {"teacher_variant": teacher_v, "trials": confirm_rows},
                )
        print(json.dumps({"event": "confirm_complete", "n": len(confirm_rows)}), flush=True)

    if args.phase in {"gate", "all"}:
        teacher_v = load_json(runs_root / "CONFIRM_RESULTS.json")["teacher_variant"]
        report = evaluate_teacher_gate(policy, runs_root, teacher_v)
        atomic_write(runs_root / "V6_TEACHER_GATE.json", report)
        atomic_write(_ROOT / "Docs/results/ms_sta_v6/V6_TEACHER_GATE.json", report)
        write_status(runs_root, {"phase": "gate", "verdict": report["verdict"], "head": head})
        print(json.dumps(report, indent=2), flush=True)
        if report["verdict"] != "V6_TEACHER_PASS":
            no_go = {
                "schema": "ms_sta_v6_teacher_no_go_v1",
                "status": "V6_TEACHER_NO_GO",
                "retain_defense_model": "b4_star_fixed",
                "forbid_e3": True,
                "gate": report,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_write(_ROOT / "Docs/results/ms_sta_v6/V6_TEACHER_NO_GO.json", no_go)
            return 2
        print(json.dumps({"event": "teacher_pass_e3_authorized_next"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
