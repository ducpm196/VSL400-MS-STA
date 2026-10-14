#!/usr/bin/env python3
"""E4-R2 Phase-2 pilot microbench: reject arms slower than B4 (model-only GPU).

Not a gate PASS claim — screening only before distillation/HPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters as count_b4  # noqa: E402
from src.models.ms_sta_v6_e4_r2 import build_ms_sta_v6_e4_r2, count_parameters  # noqa: E402
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


def _load_batch(policy: dict, batch_size: int) -> dict[str, Any]:
    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    val_ids = load_split_role_ids(split, "val")[:batch_size]
    ds = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    return collate_v5([ds[i] for i in range(len(ds))])


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _time_fn(fn: Callable[[], Any], warmup: int, trials: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda_required"}))
        return 2

    policy = json.loads((_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_POLICY.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    batch_cpu = _load_batch(policy, args.batch_size)
    batch = _to_device(batch_cpu, device)

    teacher = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = _ROOT / policy["teacher"]["checkpoint_rel"]
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    teacher.load_state_dict(sd, strict=True)

    def run_teacher() -> None:
        with torch.inference_mode():
            teacher(batch)

    b4_times = _time_fn(run_teacher, args.warmup, args.trials)
    b4_med = _median(b4_times)

    arms: dict[str, Any] = {}
    for arm in ("A", "B", "C", "D", "E"):
        m = build_ms_sta_v6_e4_r2(arm, dropout=0.0).to(device).eval()

        def run_m(model=m) -> None:
            with torch.inference_mode():
                model(batch)

        times = _time_fn(run_m, args.warmup, args.trials)
        med = _median(times)
        speedup = (b4_med / med) - 1.0 if med > 0 else float("nan")
        eligible = bool(med < b4_med)
        arms[arm] = {
            "median_sec": med,
            "p90_sec": sorted(times)[max(0, int(0.9 * (len(times) - 1)))],
            "speedup_vs_b4": speedup,
            "params": count_parameters(m),
            "latency_eligible": eligible,
            "reject_reason": None if eligible else "slower_than_b4",
        }
        del m
        torch.cuda.empty_cache()

    report = {
        "schema": "ms_sta_v6_e4_r2_pilot_bench_v1",
        "campaign": "E4-R2",
        "phase": "2_architecture_pilot_microbench",
        "device": device_name,
        "tier": "model_only_gpu",
        "dtype": "fp32",
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "trials": args.trials,
        "b4": {
            "median_sec": b4_med,
            "params": count_b4(teacher),
            "variant": "b4_v6_ctr",
        },
        "arms": arms,
        "latency_promising_arms": [k for k, v in arms.items() if v["latency_eligible"]],
        "note": "Reject slower-than-B4 arms before distillation/HPO. Not efficiency-gate evidence.",
    }
    out = _ROOT / "Docs/results/ms_sta_v6_e4_r2/E4_R2_PILOT_BENCH.json"
    sha = atomic_write(out, report)
    print(json.dumps({"event": "e4_r2_pilot_bench_written", "sha256": sha, "out": str(out.relative_to(_ROOT))}))
    print(json.dumps({"latency_promising_arms": report["latency_promising_arms"], "arms": {k: v["speedup_vs_b4"] for k, v in arms.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
