#!/usr/bin/env python3
"""MS-STA v5-E orchestrator — HPO/confirm for b4_star_e_stride4 + non-inferiority vs B4*.

Final test remains forbidden. No B4 warm-start.
"""

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
    if b"\r" in raw:
        raise RuntimeError("crlf_forbidden")
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
    grid = policy["hpo"]["grid"]
    trials = []
    for variant, lr, dropout, wd in itertools.product(
        policy["variants"], grid["lr"], grid["dropout"], grid["weight_decay"]
    ):
        trials.append({"variant": variant, "lr": lr, "dropout": dropout, "weight_decay": wd, "seed": 42})
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


def evaluate_non_inferiority(policy: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    seeds = [int(s) for s in policy["seeds_confirm"]]
    gates = policy["accuracy_non_inferiority_vs_b4"]
    b4_root = _ROOT / policy["b4_confirm_root_rel"]
    e_top1 = []
    b4_top1 = []
    e_top5 = []
    b4_top5 = []
    e_f1 = []
    b4_f1 = []
    e_panels: dict[int, dict[str, PredRow]] = {}
    b4_panels: dict[int, dict[str, PredRow]] = {}
    for seed in seeds:
        e_sum = load_json(runs_root / "confirm" / f"b4_star_e_stride4_seed{seed}" / "SUMMARY.json")
        e_preds = load_pred_rows(runs_root / "confirm" / f"b4_star_e_stride4_seed{seed}" / "VAL_PREDS.json")
        b4_sum = load_json(b4_root / f"b4_star_fixed_seed{seed}" / "SUMMARY.json")
        b4_preds = load_pred_rows(b4_root / f"b4_star_fixed_seed{seed}" / "VAL_PREDS.json")
        if set(e_preds) != set(b4_preds):
            raise RuntimeError(f"id_mismatch_seed_{seed}")
        e_top1.append(float(e_sum["best_val_top1"]))
        b4_top1.append(float(b4_sum["best_val_top1"]))
        e_top5.append(float(e_sum.get("best_val_top5") or 0.0))
        b4_top5.append(float(b4_sum.get("best_val_top5") or 0.0))
        e_f1.append(float(e_sum.get("best_val_macro_f1") or 0.0))
        b4_f1.append(float(b4_sum.get("best_val_macro_f1") or 0.0))
        e_panels[seed] = e_preds
        b4_panels[seed] = b4_preds

    ids = sorted(e_panels[seeds[0]].keys())
    # Seed-averaged per-sample deltas (E - B4).
    d1 = []
    d5 = []
    signers = []
    for sid in ids:
        signer = e_panels[seeds[0]][sid].signer
        t1 = []
        t5 = []
        for seed in seeds:
            e = e_panels[seed][sid]
            b = b4_panels[seed][sid]
            if e.signer != signer or b.signer != signer or e.label != b.label:
                raise RuntimeError(f"alignment_fail:{sid}:{seed}")
            t1.append(float(e.correct) - float(b.correct))
            t5.append(float(e.top5_hit) - float(b.top5_hit))
        d1.append(sum(t1) / len(t1))
        d5.append(sum(t5) / len(t5))
        signers.append(signer)

    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(
        d1, signers, n_boot=int(gates["n_boot"]), seed=int(gates["boot_seed"])
    )
    mean5_pp, ci5_lo, ci5_hi = hierarchical_bootstrap_ci(
        d5, signers, n_boot=int(gates["n_boot"]), seed=int(gates["boot_seed"])
    )
    signer_map = {sid: e_panels[seeds[0]][sid].signer for sid in ids}
    signer_deltas = per_signer_delta_top1_pp(ids, signer_map, e_panels, b4_panels)

    deltas = [e_top1[i] - b4_top1[i] for i in range(3)]
    mean_delta_top1_pp = 100.0 * (sum(deltas) / 3.0)
    mean_delta_top5_pp = 100.0 * (sum(e_top5[i] - b4_top5[i] for i in range(3)) / 3.0)
    mean_e_f1 = sum(e_f1) / 3.0
    mean_b4_f1 = sum(b4_f1) / 3.0
    f1_reg_pp = 100.0 * (mean_b4_f1 - mean_e_f1)

    checks = {
        "mean_delta_top1_pp": mean_delta_top1_pp >= float(gates["mean_delta_top1_pp_min"]),
        "hierarchical_ci95_lower_pp": ci_lo >= float(gates["hierarchical_ci95_lower_pp_min"]),
        "mean_delta_top5_pp": mean_delta_top5_pp >= float(gates["mean_delta_top5_pp_min"]),
        "macro_f1_regression_pp": f1_reg_pp <= float(gates["macro_f1_regression_pp_max"]),
        "no_signer_catastrophe": all(
            v >= float(gates["no_signer_catastrophe_pp"]) for v in signer_deltas.values()
        ),
    }
    verdict = "V5_E_ACCURACY_PASS" if all(checks.values()) else "V5_E_NO_GO"
    return {
        "schema": "ms_sta_v5_e_accuracy_non_inferiority_v1",
        "verdict": verdict,
        "comparison": "b4_star_e_stride4_minus_b4_star_fixed",
        "seeds": seeds,
        "e_top1": e_top1,
        "b4_top1": b4_top1,
        "deltas_top1": deltas,
        "mean_delta_top1_pp": mean_delta_top1_pp,
        "e_top5": e_top5,
        "b4_top5": b4_top5,
        "mean_delta_top5_pp": mean_delta_top5_pp,
        "e_macro_f1": e_f1,
        "b4_macro_f1": b4_f1,
        "macro_f1_regression_pp": f1_reg_pp,
        "hierarchical_mean_delta_top1_pp": mean_pp,
        "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
        "hierarchical_mean_delta_top5_pp": mean5_pp,
        "hierarchical_ci95_top5_pp": [ci5_lo, ci5_hi],
        "per_signer_delta_top1_pp": signer_deltas,
        "checks": checks,
        "forbid_final_test": True,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v5_e_policy/V5_E_POLICY.json")
    ap.add_argument(
        "--phase",
        choices=["dry-plan", "hpo", "confirm", "gate", "all"],
        default="dry-plan",
    )
    ap.add_argument("--max-trials", type=int, default=0)
    args = ap.parse_args()

    policy_path = _ROOT / args.policy
    policy = load_json(policy_path)
    auth = load_json(_ROOT / "Docs/results/ms_sta_v5_e/V5_E_AUTHORIZATION.json")
    if policy.get("status") != "FROZEN" or policy.get("forbid_final_test") is not True:
        raise SystemExit("policy_invalid")
    if auth.get("status") != "AUTHORIZED" or auth.get("allow_final_test") is not False:
        raise SystemExit("auth_invalid")
    v2_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json")
    if v2_sha != policy["bound_validation_lock_v2_sha256"]:
        raise SystemExit("lock_v2_sha_mismatch")

    head = git_rev(_ROOT)
    porcelain = git_porcelain(_ROOT).strip()
    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)
    trials = hpo_grid(policy)
    if args.max_trials and args.max_trials > 0:
        trials = trials[: int(args.max_trials)]

    plan = {
        "schema": "ms_sta_v5_e_orchestrator_plan_v1",
        "phase": args.phase,
        "head": head,
        "porcelain_clean": porcelain == "",
        "n_hpo_trials": len(trials),
        "variant": "b4_star_e_stride4",
        "forbid_final_test": True,
        "policy_sha256": sha256_file(policy_path),
        "lock_v2_sha256": v2_sha,
    }
    if args.phase == "dry-plan":
        print(json.dumps(plan, indent=2))
        return 0

    if porcelain:
        raise SystemExit(f"dirty_worktree_forbidden:{porcelain[:200]}")

    if args.phase in {"hpo", "all"}:
        rows = []
        for i, trial in enumerate(trials):
            tag = (
                f"{trial['variant']}_lr{trial['lr']}_do{trial['dropout']}"
                f"_wd{trial['weight_decay']}_s{trial['seed']}"
            )
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
        best = max(rows, key=lambda r: float(r["best_val_top1"]))
        atomic_write(runs_root / "HPO_SELECTED.json", {"selected": {"b4_star_e_stride4": best}})
        print(json.dumps({"event": "hpo_complete", "best_val_top1": best["best_val_top1"]}), flush=True)

    if args.phase in {"confirm", "all"}:
        selected = load_json(runs_root / "HPO_SELECTED.json")["selected"]["b4_star_e_stride4"]
        confirm_rows = []
        for seed in policy["seeds_confirm"]:
            trial = {
                "variant": "b4_star_e_stride4",
                "lr": selected["lr"],
                "dropout": selected["dropout"],
                "weight_decay": selected["weight_decay"],
                "seed": seed,
            }
            out_dir = runs_root / "confirm" / f"b4_star_e_stride4_seed{seed}"
            write_status(runs_root, {"phase": "confirm", "seed": seed, "head": head})
            print(json.dumps({"event": "confirm_start", "variant": "b4_star_e_stride4", "seed": seed}), flush=True)
            summary = run_trial(policy_path, trial, "confirm", out_dir)
            summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
            confirm_rows.append(summary)
            atomic_write(runs_root / "CONFIRM_RESULTS.json", {"trials": confirm_rows})
        print(json.dumps({"event": "confirm_complete", "n": len(confirm_rows)}), flush=True)

    if args.phase in {"gate", "all"}:
        report = evaluate_non_inferiority(policy, runs_root)
        atomic_write(runs_root / "ACCURACY_NONINFERIORITY_GATE.json", report)
        atomic_write(_ROOT / "Docs/results/ms_sta_v5_e/ACCURACY_NONINFERIORITY_GATE.json", report)
        write_status(runs_root, {"phase": "gate", "verdict": report["verdict"], "head": head})
        print(json.dumps(report, indent=2), flush=True)
        if report["verdict"] != "V5_E_ACCURACY_PASS":
            no_go = {
                "schema": "ms_sta_v5_e_no_go_v1",
                "status": "V5_E_NO_GO",
                "retain_final_model": "b4_star_fixed",
                "forbid_final_test_until_reauth": True,
                "gate": report,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_write(_ROOT / "Docs/results/ms_sta_v5_e/V5_E_NO_GO.json", no_go)
            print(json.dumps({"event": "STOP_BEFORE_FINAL_TEST", "reason": "V5_E_NO_GO"}, indent=2), flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
