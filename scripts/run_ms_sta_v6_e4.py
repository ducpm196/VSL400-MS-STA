#!/usr/bin/env python3
"""MS-STA v6-E4 orchestrator — width/layer student vs Stage-A teacher."""

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
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.ms_sta_v6_e4 import build_ms_sta_v6_e4  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_split_role_ids  # noqa: E402
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402


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
    for variant, lr, dw in itertools.product(policy["candidates"], hpo["lr"], hpo["distill_weight"]):
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
    with (out_dir / "train.log").open("w", encoding="utf-8") as log:
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
def measure_batch(ckpt: Path, builder, variant: str, policy: dict, device: torch.device, batch_size: int) -> tuple[float, int]:
    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    ids = load_split_role_ids(split, "val")[:batch_size]
    ds = VSL400MSSTAv5Dataset(
        ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    batch = collate_v5([ds[i] for i in range(len(ds))])
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model = builder(variant, dropout=0.0).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    for _ in range(20):
        _ = model(x)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    times = []
    for _ in range(80):
        t0 = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return _median(times), int(torch.cuda.max_memory_allocated())


def promote(policy: dict, rows: list[dict], teacher_tune: float, teacher_lat: float) -> list[dict]:
    req = policy["hpo"]["promote_requires"]
    eligible = []
    for r in rows:
        dpp = 100.0 * (float(r["best_val_top1"]) - teacher_tune)
        sp = 1.0 - float(r["batch1_median_sec"]) / max(teacher_lat, 1e-12)
        r["tune_delta_top1_pp_vs_teacher"] = dpp
        r["batch1_median_speedup_vs_teacher"] = sp
        if dpp >= float(req["tune_delta_top1_pp_min"]) and sp >= float(req["batch1_median_speedup_min"]):
            eligible.append(r)
    eligible.sort(key=lambda r: (-float(r["best_val_top1"]), -float(r["batch1_median_speedup_vs_teacher"]), r["variant"]))
    chosen, seen = [], set()
    for r in eligible:
        if r["variant"] in seen:
            continue
        chosen.append(r)
        seen.add(r["variant"])
        if len(chosen) >= int(policy["hpo"]["promote_max"]):
            break
    return chosen


def accuracy_gate(policy: dict, runs_root: Path, variant: str) -> dict[str, Any]:
    seeds = [int(s) for s in policy["seeds_confirm"]]
    gates = policy["accuracy_non_inferiority_vs_teacher"]
    # Prefer local confirm preds if present; else external archive confirm/
    t_root = _ROOT / policy["teacher_confirm_root_rel"]
    if not (t_root / "b4_v6_ctr_seed42" / "VAL_PREDS.json").is_file():
        arch = Path(load_json(_ROOT / policy["artifact_index_rel"])["raw_evidence_archive"]["root"]) / "confirm"
        t_root = arch
    e_top1, t_top1, e_top5, t_top5 = [], [], [], []
    e_panels, t_panels = {}, {}
    for seed in seeds:
        e_preds = load_pred_rows(runs_root / "confirm" / f"{variant}_seed{seed}" / "VAL_PREDS.json")
        t_preds = load_pred_rows(t_root / f"b4_v6_ctr_seed{seed}" / "VAL_PREDS.json")
        if set(e_preds) != set(t_preds):
            raise RuntimeError(f"id_mismatch_seed_{seed}")
        e_top1.append(sum(1 for p in e_preds.values() if p.correct) / max(len(e_preds), 1))
        t_top1.append(sum(1 for p in t_preds.values() if p.correct) / max(len(t_preds), 1))
        e_top5.append(sum(1 for p in e_preds.values() if p.top5_hit) / max(len(e_preds), 1))
        t_top5.append(sum(1 for p in t_preds.values() if p.top5_hit) / max(len(t_preds), 1))
        e_panels[seed] = e_preds
        t_panels[seed] = t_preds
    ids = sorted(e_panels[seeds[0]].keys())
    d1, signers = [], []
    for sid in ids:
        vals = [float(e_panels[s][sid].correct) - float(t_panels[s][sid].correct) for s in seeds]
        d1.append(sum(vals) / len(vals))
        signers.append(e_panels[seeds[0]][sid].signer)
    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(d1, signers, n_boot=int(gates["n_boot"]), seed=int(gates["boot_seed"]))
    signer_deltas = per_signer_delta_top1_pp(ids, {sid: e_panels[seeds[0]][sid].signer for sid in ids}, e_panels, t_panels)
    mean_d1 = 100.0 * (sum(e_top1[i] - t_top1[i] for i in range(3)) / 3.0)
    mean_d5 = 100.0 * (sum(e_top5[i] - t_top5[i] for i in range(3)) / 3.0)
    checks = {
        "mean_delta_top1_pp": mean_d1 >= float(gates["mean_delta_top1_pp_min"]),
        "hierarchical_ci95_lower_pp": ci_lo >= float(gates["hierarchical_ci95_lower_pp_min"]),
        "mean_delta_top5_pp": mean_d5 >= float(gates["mean_delta_top5_pp_min"]),
        "no_signer_catastrophe": all(v >= float(gates["no_signer_catastrophe_pp"]) for v in signer_deltas.values()),
    }
    verdict = "E4_ACCURACY_PASS" if all(checks.values()) else "E4_ACCURACY_NO_GO"
    return {
        "schema": "ms_sta_v6_e4_accuracy_gate_v1",
        "verdict": verdict,
        "variant": variant,
        "mean_delta_top1_pp": mean_d1,
        "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
        "mean_delta_top5_pp": mean_d5,
        "checks": checks,
        "e_top1": e_top1,
        "teacher_top1": t_top1,
        "signer_deltas_pp": signer_deltas,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def efficiency_gate(policy: dict, runs_root: Path, variant: str, seed: int = 42) -> dict[str, Any]:
    eff = policy["efficiency"]
    device = torch.device("cuda")
    student_ckpt = runs_root / "confirm" / f"{variant}_seed{seed}" / "best.pt"
    teacher_ckpt = Path(policy["teacher_checkpoint_abs"])
    s1, sv = measure_batch(student_ckpt, lambda v, dropout: build_ms_sta_v6_e4(v, dropout=dropout), variant, policy, device, 1)
    s8, _ = measure_batch(student_ckpt, lambda v, dropout: build_ms_sta_v6_e4(v, dropout=dropout), variant, policy, device, 8)
    t1 = float(eff["teacher_batch1_fp32_median_sec"])
    t8 = float(eff["teacher_batch8_fp32_median_sec"])
    tv = int(eff["teacher_batch1_peak_vram_bytes"])
    red1 = 1.0 - s1 / max(t1, 1e-12)
    red8 = 1.0 - s8 / max(t8, 1e-12)
    vram_delta = (sv - tv) / max(tv, 1)
    checks = {
        "batch1_median_reduction": red1 >= float(eff["batch1_median_reduction_min"]),
        "batch8_median_reduction": red8 >= float(eff["batch8_median_reduction_min"]),
        "vram_not_materially_up": vram_delta <= float(eff["vram_increase_max"]),
    }
    verdict = "E4_EFFICIENCY_PASS" if all(checks.values()) else "E4_EFFICIENCY_NO_GO"
    return {
        "schema": "ms_sta_v6_e4_efficiency_gate_v1",
        "verdict": verdict,
        "variant": variant,
        "seed": seed,
        "student_batch1_median_sec": s1,
        "student_batch8_median_sec": s8,
        "teacher_batch1_median_sec": t1,
        "teacher_batch8_median_sec": t8,
        "batch1_reduction": red1,
        "batch8_reduction": red8,
        "student_batch1_peak_vram": sv,
        "teacher_batch1_peak_vram": tv,
        "vram_delta_frac": vram_delta,
        "checks": checks,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v6_e4_policy/E4_POLICY.json")
    ap.add_argument("--phase", choices=["dry-plan", "hpo", "confirm", "gate", "all"], default="dry-plan")
    args = ap.parse_args()
    policy_path = _ROOT / args.policy
    policy = load_json(policy_path)
    if policy.get("status") != "FROZEN" or policy.get("train_allowed") is not True:
        raise SystemExit("policy_invalid")
    if sha256_file(Path(policy["teacher_checkpoint_abs"])) != policy["teacher_checkpoint_sha256"]:
        raise SystemExit("teacher_sha_mismatch")
    head = git_rev(_ROOT)
    porcelain_raw = git_porcelain(_ROOT).strip()
    runs_rel = policy["runs_root_rel"].replace("\\", "/").rstrip("/") + "/"
    blocking = []
    for line in porcelain_raw.splitlines():
        if line.startswith("??"):
            path = line[3:].replace("\\", "/").strip()
            if path.startswith(runs_rel) or path.startswith("Docs/results/ms_sta_v6_e4/"):
                continue
            blocking.append(line)
            continue
        parts = line.split(maxsplit=1)
        path = parts[1].replace("\\", "/").strip() if len(parts) == 2 else ""
        if path and not (
            path.startswith("Docs/results/ms_sta_v6_e4")
            or path.startswith("Docs/results/ms_sta_v6_e4_policy")
            or path.startswith("src/models/ms_sta_v6_e4")
            or path.startswith("src/training/ms_sta_v5_train.py")
            or path.startswith("scripts/run_ms_sta_v6_e4")
            or path.startswith("scripts/profile_ms_sta_v6_e4")
            or path.startswith("tests/test_ms_sta_v6_e4")
        ):
            blocking.append(line)
    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)
    trials = hpo_grid(policy)
    plan = {
        "schema": "ms_sta_v6_e4_plan_v1",
        "phase": args.phase,
        "head": head,
        "n_hpo": len(trials),
        "candidates": policy["candidates"],
        "hypothesis": policy["hypothesis"],
        "forbid_final_test": True,
    }
    if args.phase == "dry-plan":
        print(json.dumps(plan, indent=2))
        return 0
    dirty = [ln for ln in porcelain_raw.splitlines() if not ln.startswith("??")]
    if dirty and args.phase in {"hpo", "confirm", "all"}:
        # allow only e4-related tracked mods before first commit
        bad = [ln for ln in dirty if "ms_sta_v6_e4" not in ln and "ms_sta_v5_train" not in ln and "profile_ms_sta_v6_e4" not in ln and "E4_" not in ln]
        if bad:
            raise SystemExit(f"dirty_tracked_worktree_forbidden:{chr(10).join(bad[:20])}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required")

    if args.phase in {"hpo", "all"}:
        teacher_tune = float(policy.get("teacher_hpo_tune_top1", 0.7273730684326711))
        teacher_lat = float(policy["efficiency"]["teacher_batch1_fp32_median_sec"])
        rows = []
        for i, trial in enumerate(trials):
            tag = f"{trial['variant']}_lr{trial['lr']}_dw{trial['distill_weight']}_do{trial['dropout']}_wd{trial['weight_decay']}_s{trial['seed']}"
            out_dir = runs_root / "hpo" / tag
            write_status(runs_root, {"phase": "hpo", "i": i, "n": len(trials), "tag": tag, "head": head})
            print(json.dumps({"event": "hpo_trial_start", "i": i, "n": len(trials), "tag": tag}), flush=True)
            summary = run_trial(policy_path, trial, "hpo", out_dir)
            lat, _vram = measure_batch(
                out_dir / "best.pt",
                lambda v, dropout: build_ms_sta_v6_e4(v, dropout=dropout),
                trial["variant"],
                policy,
                device,
                1,
            )
            summary.update({"tag": tag, "batch1_median_sec": lat, "teacher_tune_top1": teacher_tune, "teacher_batch1_median_sec": teacher_lat})
            rows.append(summary)
            atomic_write(runs_root / "HPO_LEADERBOARD.json", {"trials": rows, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        promoted = promote(policy, rows, teacher_tune, teacher_lat)
        atomic_write(runs_root / "HPO_SELECTED.json", {"promoted": promoted, "teacher_tune_top1": teacher_tune})
        if not promoted:
            no_go = {"schema": "ms_sta_v6_e4_pilot_no_go_v1", "status": "E4_PILOT_NO_GO", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            atomic_write(_ROOT / "Docs/results/ms_sta_v6_e4/E4_PILOT_NO_GO.json", no_go)
            print(json.dumps({"event": "STOP", "reason": "E4_PILOT_NO_GO"}, indent=2), flush=True)
            return 2
        print(json.dumps({"event": "hpo_complete", "promoted": [p["variant"] for p in promoted]}, indent=2), flush=True)

    if args.phase in {"confirm", "all"}:
        winner = load_json(runs_root / "HPO_SELECTED.json")["promoted"][0]
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
            confirm_rows.append(summary)
            atomic_write(runs_root / "CONFIRM_RESULTS.json", {"winner": winner["variant"], "trials": confirm_rows})
        print(json.dumps({"event": "confirm_complete", "n": len(confirm_rows)}), flush=True)

    if args.phase in {"gate", "all"}:
        winner = load_json(runs_root / "CONFIRM_RESULTS.json")["winner"]
        acc = accuracy_gate(policy, runs_root, winner)
        atomic_write(runs_root / "E4_ACCURACY_GATE.json", acc)
        atomic_write(_ROOT / "Docs/results/ms_sta_v6_e4/E4_ACCURACY_GATE.json", acc)
        print(json.dumps(acc, indent=2), flush=True)
        if acc["verdict"] != "E4_ACCURACY_PASS":
            atomic_write(_ROOT / "Docs/results/ms_sta_v6_e4/E4_NO_GO.json", {"status": "E4_ACCURACY_NO_GO", "gate": acc})
            return 2
        eff = efficiency_gate(policy, runs_root, winner)
        atomic_write(runs_root / "E4_EFFICIENCY_GATE.json", eff)
        atomic_write(_ROOT / "Docs/results/ms_sta_v6_e4/E4_EFFICIENCY_GATE.json", eff)
        print(json.dumps(eff, indent=2), flush=True)
        if eff["verdict"] != "E4_EFFICIENCY_PASS":
            atomic_write(_ROOT / "Docs/results/ms_sta_v6_e4/E4_NO_GO.json", {"status": "E4_EFFICIENCY_NO_GO", "accuracy": acc, "efficiency": eff})
            return 2
        lock = {
            "schema": "ms_sta_v6_e4_validation_efficiency_lock_v1",
            "status": "E4_VALIDATION_EFFICIENCY_PASS",
            "variant": winner,
            "accuracy": acc,
            "efficiency": eff,
            "retain_defense": "b4_star_fixed",
            "retain_stage_a_teacher": "b4_v6_ctr",
            "forbid_final_test": True,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write(_ROOT / "Docs/results/ms_sta_v6_e4/E4_VALIDATION_EFFICIENCY_LOCK.json", lock)
        print(json.dumps({"event": "E4_PASS_STOP_BEFORE_FINAL_TEST", "lock": lock["status"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
