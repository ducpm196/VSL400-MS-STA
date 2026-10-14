#!/usr/bin/env python3
"""E4-R3 Phase-2 feasibility microbench (model-only GPU).

Promote arms with model-forward median ≥8% faster than B4.
If none qualify → write E4_R3_FEASIBILITY_NO_GO and exit 3.
Not efficiency-gate evidence; screening only before HPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters as count_b4  # noqa: E402
from src.models.ms_sta_v6_e4_r3 import build_ms_sta_v6_e4_r3, count_parameters  # noqa: E402
from src.training.ms_sta_v5_data import (  # noqa: E402
    VSL400MSSTAv5Dataset,
    collate_v5,
    load_json,
    load_split_role_ids,
)
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def _p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[max(0, int(0.9 * (len(s) - 1)))]


def _load_batch(policy: dict, batch_size: int) -> dict[str, Any]:
    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    val_ids = load_split_role_ids(split, "val")[:batch_size]
    if set(val_ids) & test_ids:
        raise SystemExit("test_leakage_detected_in_feasibility_batch")
    ds = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    return collate_v5([ds[i] for i in range(len(ds))])


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _peak_vram_mb() -> dict[str, float]:
    return {
        "allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    policy_path = _ROOT / "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_POLICY.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    feas = policy["feasibility"]
    if args.warmup < int(feas["warmup_min"]) or args.trials < int(feas["trials_min"]):
        raise SystemExit("feasibility_warmup_or_trials_below_policy_min")
    speedup_min = float(feas["model_forward_median_speedup_vs_b4_min"])
    max_promote = int(feas["max_arms_promoted_to_hpo"])

    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda_required"}))
        return 2
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    required = (policy.get("environment_lock") or {}).get("required_gpu_name")
    if required and device_name != required:
        raise SystemExit(f"gpu_name_mismatch:{device_name}!={required}")

    batch = _to_device(_load_batch(policy, args.batch_size), device)

    teacher = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = _ROOT / policy["teacher"]["checkpoint_rel"]
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    teacher.load_state_dict(sd, strict=True)
    observed = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if observed != policy["teacher"]["checkpoint_sha256"]:
        raise SystemExit("teacher_checkpoint_sha_mismatch")

    models: dict[str, torch.nn.Module] = {"B4": teacher}
    for arm in ("A", "B", "C"):
        models[arm] = build_ms_sta_v6_e4_r3(arm, dropout=0.0).to(device).eval()

    def make_runner(model: torch.nn.Module) -> Callable[[], None]:
        def _run() -> None:
            with torch.inference_mode():
                model(batch)

        return _run

    runners = {k: make_runner(m) for k, m in models.items()}
    keys = list(runners.keys())

    # Warmup interleaved
    rng = random.Random(args.seed)
    for _ in range(args.warmup):
        k = rng.choice(keys)
        runners[k]()
    torch.cuda.synchronize()

    times: dict[str, list[float]] = {k: [] for k in keys}
    peak_by_key: dict[str, dict[str, float]] = {}

    # Randomized interleaved measurement
    order = [keys[i % len(keys)] for i in range(args.trials * len(keys))]
    rng.shuffle(order)
    for k in order:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runners[k]()
        torch.cuda.synchronize()
        times[k].append(time.perf_counter() - t0)
        peak_by_key[k] = _peak_vram_mb()

    b4_med = _median(times["B4"])
    arms: dict[str, Any] = {}
    for arm in ("A", "B", "C"):
        med = _median(times[arm])
        speedup = (b4_med / med) - 1.0 if med > 0 else float("nan")
        eligible = bool(speedup >= speedup_min)
        arms[arm] = {
            "median_sec": med,
            "p90_sec": _p90(times[arm]),
            "mean_sec": sum(times[arm]) / len(times[arm]),
            "throughput_sps": (args.batch_size / med) if med > 0 else float("nan"),
            "n_trials": len(times[arm]),
            "speedup_vs_b4": speedup,
            "params": count_parameters(models[arm]),
            "peak_vram": peak_by_key.get(arm),
            "latency_eligible": eligible,
            "reject_reason": None if eligible else f"speedup_lt_{speedup_min}",
        }

    ranked = sorted(
        ((k, v) for k, v in arms.items() if v["latency_eligible"]),
        key=lambda kv: (-kv[1]["speedup_vs_b4"], kv[0]),
    )
    promoted = [k for k, _ in ranked[:max_promote]]
    status = "FEASIBILITY_PASS" if promoted else str(feas["fail_status"])

    report = {
        "schema": "ms_sta_v6_e4_r3_feasibility_bench_v1",
        "campaign": "E4-R3",
        "phase": "2_feasibility_microbench",
        "status": status,
        "device": device_name,
        "tier": "model_only_gpu",
        "dtype": "fp32",
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "trials_per_model_approx": args.trials,
        "interleaved": True,
        "randomized": True,
        "seed": args.seed,
        "policy_sha256": policy_sha,
        "teacher_checkpoint_sha256": observed,
        "speedup_min_required": speedup_min,
        "b4": {
            "median_sec": b4_med,
            "p90_sec": _p90(times["B4"]),
            "mean_sec": sum(times["B4"]) / len(times["B4"]),
            "throughput_sps": args.batch_size / b4_med if b4_med > 0 else float("nan"),
            "n_trials": len(times["B4"]),
            "params": count_b4(teacher),
            "peak_vram": peak_by_key.get("B4"),
            "variant": "b4_v6_ctr",
        },
        "arms": arms,
        "promoted_arms": promoted,
        "note": (
            "Screening only. Efficiency gate requires later canonical protocol. "
            "No grayscale. No HPO if empty promoted_arms."
        ),
    }

    out_dir = _ROOT / policy["results_root_rel"]
    out = out_dir / "E4_R3_FEASIBILITY.json"
    sha = atomic_write(out, report)
    print(
        json.dumps(
            {
                "event": "e4_r3_feasibility_written",
                "status": status,
                "sha256": sha,
                "out": str(out.relative_to(_ROOT)).replace("\\", "/"),
                "promoted_arms": promoted,
                "speedups": {k: v["speedup_vs_b4"] for k, v in arms.items()},
                "b4_median_ms": b4_med * 1000.0,
            },
            indent=2,
        )
    )

    if not promoted:
        nogo = {
            "schema": "ms_sta_v6_e4_r3_feasibility_no_go_v1",
            "status": feas["fail_status"],
            "campaign": "E4-R3",
            "reason": "no_arm_met_model_forward_median_speedup_min",
            "speedup_min_required": speedup_min,
            "feasibility_report_rel": str(out.relative_to(_ROOT)).replace("\\", "/"),
            "feasibility_report_sha256": sha,
            "arms": {k: {"speedup_vs_b4": v["speedup_vs_b4"], "median_sec": v["median_sec"]} for k, v in arms.items()},
            "halt": True,
            "forbid_hpo": True,
            "forbid_confirm": True,
            "forbid_test": True,
            "forbid_d4": True,
            "e4_ready_remains": False,
        }
        nogo_path = out_dir / "E4_R3_FEASIBILITY_NO_GO.json"
        nogo_sha = atomic_write(nogo_path, nogo)
        print(json.dumps({"event": "e4_r3_feasibility_no_go", "sha256": nogo_sha}, indent=2))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
