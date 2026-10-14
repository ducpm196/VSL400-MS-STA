#!/usr/bin/env python3
"""MS-STA v5-E2 orchestrator — pilot HPO / confirm / gates (no test/demo/registry)."""

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

import torch

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
from src.models.ms_sta_v5 import build_ms_sta_v5  # noqa: E402
from src.models.ms_sta_v5_e2 import build_ms_sta_v5_e2  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_split_role_ids  # noqa: E402
from src.training.ms_sta_v5_train import evaluate, load_metadata_by_id  # noqa: E402


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        # allow STATUS overwrite; refuse other create-only artifacts when flagged
        pass
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
    for variant, lr, dw in itertools.product(
        policy["candidates"], hpo["lr"], hpo["distill_weight"]
    ):
        trials.append(
            {
                "variant": variant,
                "lr": lr,
                "dropout": float(hpo["dropout"]),
                "weight_decay": float(hpo["weight_decay"]),
                "distill_weight": dw,
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
        "--distill-weight",
        str(trial.get("distill_weight", 0.0)),
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


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


@torch.no_grad()
def measure_batch1_median(ckpt_path: Path, variant: str, dropout: float, policy: dict, device: torch.device) -> float:
    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    val_ids = load_split_role_ids(split, "val")[:1]
    ds = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    batch = collate_v5([ds[0]])
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if variant.startswith("e2_"):
        model = build_ms_sta_v5_e2(variant, dropout=dropout).to(device)  # type: ignore[arg-type]
    else:
        model = build_ms_sta_v5(
            variant,
            d_model=int(policy["model"]["d_model"]),
            nhead=int(policy["model"]["nhead"]),
            temporal_layers=int(policy["model"]["temporal_layers"]),
            dropout=dropout,
            alpha_max=float(policy["model"]["alpha_max"]),
        ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for _ in range(20):
        _ = model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(80):
        t0 = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return _median(times)


def eval_b4_tune_top1(policy: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Fair pilot baseline: B4* HPO checkpoint trained on hpo_train only (tune held out).

    Do NOT evaluate the confirm_train B4* checkpoint on tune — confirm_train includes
    tune IDs, so that score is in-sample and not comparable to E2 HPO students.
    """
    # Locked M0 HPO winner (seed 42) on the same tune split.
    rel = "Docs/results/ms_sta_v5_m0_runs/hpo/b4_star_fixed_lr0.0001_do0.1_wd0.01_s42/SUMMARY.json"
    summary = load_json(_ROOT / rel)
    top1 = float(summary["best_val_top1"])
    if abs(top1 - 0.6030169242089772) > 1e-9:
        raise RuntimeError(f"b4_hpo_tune_baseline_drift:{top1}")
    return {
        "b4_tune_top1": top1,
        "b4_tune_baseline_rel": rel,
        "b4_tune_baseline_sha256": sha256_file(_ROOT / rel),
        "note": "hpo_matched_hpo_train_only_not_confirm_ckpt_on_tune",
    }


def promote_candidates(policy: dict[str, Any], rows: list[dict[str, Any]], b4_tune: float, b4_lat: float) -> list[dict[str, Any]]:
    req = policy["hpo"]["promote_requires"]
    eligible = []
    for r in rows:
        delta_pp = 100.0 * (float(r["best_val_top1"]) - b4_tune)
        speed = 1.0 - float(r["batch1_median_sec"]) / max(b4_lat, 1e-12)
        r["tune_delta_top1_pp_vs_b4"] = delta_pp
        r["batch1_median_speedup_vs_b4"] = speed
        if delta_pp >= float(req["tune_delta_top1_pp_min"]) and speed >= float(req["batch1_median_speedup_min"]):
            eligible.append(r)
    # Pareto-ish: sort by top1 desc, then speed desc, then variant id
    eligible.sort(
        key=lambda r: (-float(r["best_val_top1"]), -float(r["batch1_median_speedup_vs_b4"]), r["variant"])
    )
    # at most one per variant, max promote_max
    chosen: list[dict[str, Any]] = []
    seen = set()
    for r in eligible:
        if r["variant"] in seen:
            continue
        chosen.append(r)
        seen.add(r["variant"])
        if len(chosen) >= int(policy["hpo"]["promote_max"]):
            break
    return chosen


def evaluate_accuracy_gate(policy: dict[str, Any], runs_root: Path, variant: str) -> dict[str, Any]:
    seeds = [int(s) for s in policy["seeds_confirm"]]
    gates = policy["accuracy_non_inferiority_vs_b4"]
    b4_root = _ROOT / policy["b4_confirm_root_rel"]
    e_top1, b4_top1, e_top5, b4_top5, e_f1, b4_f1 = [], [], [], [], [], []
    e_panels: dict[int, dict[str, PredRow]] = {}
    b4_panels: dict[int, dict[str, PredRow]] = {}
    for seed in seeds:
        e_sum = load_json(runs_root / "confirm" / f"{variant}_seed{seed}" / "SUMMARY.json")
        e_preds = load_pred_rows(runs_root / "confirm" / f"{variant}_seed{seed}" / "VAL_PREDS.json")
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
    d1, d5, signers = [], [], []
    for sid in ids:
        signer = e_panels[seeds[0]][sid].signer
        t1, t5 = [], []
        for seed in seeds:
            e = e_panels[seed][sid]
            b = b4_panels[seed][sid]
            t1.append(float(e.correct) - float(b.correct))
            t5.append(float(e.top5_hit) - float(b.top5_hit))
        d1.append(sum(t1) / len(t1))
        d5.append(sum(t5) / len(t5))
        signers.append(signer)
    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(
        d1, signers, n_boot=int(gates["n_boot"]), seed=int(gates["boot_seed"])
    )
    signer_map = {sid: e_panels[seeds[0]][sid].signer for sid in ids}
    signer_deltas = per_signer_delta_top1_pp(ids, signer_map, e_panels, b4_panels)
    mean_delta_top1_pp = 100.0 * (sum(e_top1[i] - b4_top1[i] for i in range(3)) / 3.0)
    mean_delta_top5_pp = 100.0 * (sum(e_top5[i] - b4_top5[i] for i in range(3)) / 3.0)
    f1_reg_pp = 100.0 * ((sum(b4_f1) / 3.0) - (sum(e_f1) / 3.0))
    checks = {
        "mean_delta_top1_pp": mean_delta_top1_pp >= float(gates["mean_delta_top1_pp_min"]),
        "hierarchical_ci95_lower_pp": ci_lo >= float(gates["hierarchical_ci95_lower_pp_min"]),
        "mean_delta_top5_pp": mean_delta_top5_pp >= float(gates["mean_delta_top5_pp_min"]),
        "macro_f1_regression_pp": f1_reg_pp <= float(gates["macro_f1_regression_pp_max"]),
        "no_signer_catastrophe": all(
            v >= float(gates["no_signer_catastrophe_pp"]) for v in signer_deltas.values()
        ),
    }
    verdict = "E2_ACCURACY_PASS" if all(checks.values()) else "E2_ACCURACY_NO_GO"
    return {
        "schema": "ms_sta_v5_e2_accuracy_gate_v1",
        "verdict": verdict,
        "variant": variant,
        "mean_delta_top1_pp": mean_delta_top1_pp,
        "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
        "mean_delta_top5_pp": mean_delta_top5_pp,
        "macro_f1_regression_pp": f1_reg_pp,
        "per_signer_delta_top1_pp": signer_deltas,
        "checks": checks,
        "e_top1": e_top1,
        "b4_top1": b4_top1,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v5_e2_policy/E2_POLICY.json")
    ap.add_argument(
        "--phase",
        choices=["dry-plan", "hpo", "confirm", "gate", "all"],
        default="dry-plan",
    )
    args = ap.parse_args()

    policy_path = _ROOT / args.policy
    policy = load_json(policy_path)
    auth = load_json(_ROOT / "Docs/results/ms_sta_v5_e2/E2_AUTHORIZATION.json")
    if policy.get("status") != "FROZEN" or policy.get("train_allowed") is not True:
        raise SystemExit("policy_invalid")
    if policy.get("forbid_final_test") is not True:
        raise SystemExit("policy_must_forbid_final_test")
    if auth.get("status") != "AUTHORIZED" or auth.get("allow_final_test") is not False:
        raise SystemExit("auth_invalid")
    v2_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json")
    if v2_sha != policy["bound_validation_lock_v2_sha256"]:
        raise SystemExit("lock_v2_sha_mismatch")
    if sha256_file(_ROOT / policy["teacher_checkpoint_rel"]) != policy["teacher_checkpoint_sha256"]:
        raise SystemExit("teacher_sha_mismatch")
    if sha256_file(_ROOT / policy["profile_report_rel"]) != policy["profile_report_sha256"]:
        raise SystemExit("profile_sha_mismatch")

    head = git_rev(_ROOT)
    porcelain_raw = git_porcelain(_ROOT).strip()
    runs_rel = policy["runs_root_rel"].replace("\\", "/").rstrip("/") + "/"
    blocking = []
    for line in porcelain_raw.splitlines():
        path = line[3:].replace("\\", "/").strip()
        # Allow campaign run artifacts to exist untracked while HPO/confirm execute.
        if line.startswith("??") and (path.startswith(runs_rel) or path.rstrip("/") == runs_rel.rstrip("/")):
            continue
        blocking.append(line)
    porcelain = "\n".join(blocking).strip()
    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)
    trials = hpo_grid(policy)
    plan = {
        "schema": "ms_sta_v5_e2_orchestrator_plan_v1",
        "phase": args.phase,
        "head": head,
        "porcelain_clean": porcelain == "",
        "n_hpo_trials": len(trials),
        "candidates": policy["candidates"],
        "forbid_final_test": True,
        "policy_sha256": sha256_file(policy_path),
    }
    if args.phase == "dry-plan":
        print(json.dumps(plan, indent=2))
        return 0

    if porcelain:
        raise SystemExit(f"dirty_worktree_forbidden:{porcelain[:400]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required")

    if args.phase in {"hpo", "all"}:
        b4_base = eval_b4_tune_top1(policy, device)
        b4_tune = float(b4_base["b4_tune_top1"])
        b4_lat = measure_batch1_median(
            _ROOT / policy["teacher_checkpoint_rel"], "b4_star_fixed", 0.1, policy, device
        )
        rows: list[dict[str, Any]] = []
        for i, trial in enumerate(trials):
            tag = (
                f"{trial['variant']}_lr{trial['lr']}_dw{trial['distill_weight']}"
                f"_do{trial['dropout']}_wd{trial['weight_decay']}_s{trial['seed']}"
            )
            out_dir = runs_root / "hpo" / tag
            write_status(runs_root, {"phase": "hpo", "i": i, "n": len(trials), "tag": tag, "head": head})
            print(json.dumps({"event": "hpo_trial_start", "i": i, "n": len(trials), "tag": tag}), flush=True)
            summary = run_trial(policy_path, trial, "hpo", out_dir)
            lat = measure_batch1_median(
                out_dir / "best.pt", trial["variant"], float(trial["dropout"]), policy, device
            )
            summary["tag"] = tag
            summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
            summary["batch1_median_sec"] = lat
            summary["b4_tune_top1"] = b4_tune
            summary["b4_batch1_median_sec"] = b4_lat
            summary["b4_tune_baseline"] = b4_base
            rows.append(summary)
            atomic_write(
                runs_root / "HPO_LEADERBOARD.json",
                {
                    "b4_tune_top1": b4_tune,
                    "b4_tune_baseline": b4_base,
                    "b4_batch1_median_sec": b4_lat,
                    "trials": rows,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        promoted = promote_candidates(policy, rows, b4_tune, b4_lat)
        atomic_write(
            runs_root / "HPO_SELECTED.json",
            {
                "promoted": promoted,
                "b4_tune_top1": b4_tune,
                "b4_tune_baseline": b4_base,
                "b4_batch1_median_sec": b4_lat,
                "n_eligible_checked": len(rows),
            },
        )
        if not promoted:
            no_go = {
                "schema": "ms_sta_v5_e2_pilot_no_go_v1",
                "status": "E2_PILOT_NO_GO",
                "reason": "no_candidate_met_tune_delta_and_speed_gates",
                "retain_final_model": "b4_star_fixed",
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_write(runs_root / "E2_PILOT_NO_GO.json", no_go)
            atomic_write(_ROOT / "Docs/results/ms_sta_v5_e2/E2_PILOT_NO_GO.json", no_go)
            print(json.dumps({"event": "STOP", "reason": "E2_PILOT_NO_GO"}, indent=2), flush=True)
            return 2
        print(json.dumps({"event": "hpo_complete", "promoted": [p["variant"] for p in promoted]}, indent=2), flush=True)

    if args.phase in {"confirm", "all"}:
        selected = load_json(runs_root / "HPO_SELECTED.json")["promoted"]
        if not selected:
            raise SystemExit("no_promoted_candidates")
        # Confirm only the top promoted candidate (accuracy/efficiency single winner path).
        winner = selected[0]
        confirm_rows = []
        for seed in policy["seeds_confirm"]:
            trial = {
                "variant": winner["variant"],
                "lr": winner["lr"],
                "dropout": winner["dropout"],
                "weight_decay": winner["weight_decay"],
                "distill_weight": winner.get("distill_weight", 0.5),
                "seed": seed,
            }
            out_dir = runs_root / "confirm" / f"{winner['variant']}_seed{seed}"
            write_status(runs_root, {"phase": "confirm", "seed": seed, "variant": winner["variant"], "head": head})
            print(json.dumps({"event": "confirm_start", "variant": winner["variant"], "seed": seed}), flush=True)
            summary = run_trial(policy_path, trial, "confirm", out_dir)
            summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
            confirm_rows.append(summary)
            atomic_write(runs_root / "CONFIRM_RESULTS.json", {"winner": winner["variant"], "trials": confirm_rows})
        print(json.dumps({"event": "confirm_complete", "n": len(confirm_rows)}), flush=True)

    if args.phase in {"gate", "all"}:
        winner = load_json(runs_root / "CONFIRM_RESULTS.json")["winner"]
        report = evaluate_accuracy_gate(policy, runs_root, winner)
        atomic_write(runs_root / "E2_ACCURACY_GATE.json", report)
        atomic_write(_ROOT / "Docs/results/ms_sta_v5_e2/E2_ACCURACY_GATE.json", report)
        write_status(runs_root, {"phase": "gate", "verdict": report["verdict"], "head": head})
        print(json.dumps(report, indent=2), flush=True)
        if report["verdict"] != "E2_ACCURACY_PASS":
            no_go = {
                "schema": "ms_sta_v5_e2_accuracy_no_go_v1",
                "status": "E2_ACCURACY_NO_GO",
                "retain_final_model": "b4_star_fixed",
                "gate": report,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_write(_ROOT / "Docs/results/ms_sta_v5_e2/E2_ACCURACY_NO_GO.json", no_go)
            return 2
        # Efficiency confirm left to dedicated phase script after accuracy PASS.
        print(json.dumps({"event": "accuracy_pass_next_efficiency_confirm"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
