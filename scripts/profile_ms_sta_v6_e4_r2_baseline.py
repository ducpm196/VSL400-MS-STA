#!/usr/bin/env python3
"""E4-R2 Phase-0 B4 teacher latency re-baseline (randomized interleaved).

Does not train, does not touch e4_w192 artifacts, does not promote registry.
Requires CUDA (RTX 2060 class) via env torch+cu build.
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

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters  # noqa: E402
from src.training.ms_sta_v5_data import (  # noqa: E402
    VSL400MSSTAv5Dataset,
    collate_v5,
    load_json,
    load_split_role_ids,
)
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _bootstrap_ci(xs: list[float], *, n_boot: int = 2000, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    n = len(xs)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boots.append(_median(sample))
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    return [lo, hi]


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_POLICY.json")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument(
        "--out",
        default="Docs/results/ms_sta_v6_e4_r2/E4_R2_B4_LATENCY_BASELINE.json",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("cuda_required")
    device = torch.device("cuda")
    gpu_name = str(torch.cuda.get_device_name(0))
    policy = load_json(_ROOT / args.policy)
    want_gpu = str((policy.get("environment_lock") or {}).get("required_gpu_name") or "")
    if want_gpu and gpu_name != want_gpu:
        raise SystemExit(f"gpu_identity_mismatch:{gpu_name!r}!={want_gpu!r}")

    teacher = policy["teacher"]
    ckpt_path = _ROOT / teacher["checkpoint_rel"]
    want = teacher["checkpoint_sha256"]
    got = sha256_file(ckpt_path)
    if got != want:
        raise SystemExit(f"teacher_sha_mismatch:{got}")

    model = build_ms_sta_v6("b4_v6_ctr", dropout=0.0, gate_entropy_reg=0.0).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    report: dict[str, Any] = {
        "schema": "ms_sta_v6_e4_r2_b4_latency_baseline_v1",
        "campaign": "E4-R2",
        "phase": "0_lock",
        "tier": "model_only_gpu",
        "teacher": teacher["variant"],
        "seed": teacher["seed"],
        "checkpoint_sha256": got,
        "param_count": count_parameters(model),
        "device": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "trials": args.trials,
        "rounds": args.rounds,
        "protocol": "randomized_interleaved_configs_per_round",
        "note": (
            "Phase-0 re-baseline. Phase-6 canonical efficiency uses warmup>=100, "
            "trials>=1000, rounds>=5 and separate cache-to-prediction tier."
        ),
        "historical_e4_profile_rel": "Docs/results/ms_sta_v6_e4/E4_PROFILE_REPORT.json",
        "historical_batch1_fp32_median_sec": 0.013211900019086897,
        "rounds_detail": [],
        "aggregate": {},
        "module_breakdown_batch1_fp32": {},
    }

    configs = [(1, False), (1, True), (8, False), (8, True)]
    per_config: dict[str, list[float]] = {f"batch{bs}_{'amp' if amp else 'fp32'}": [] for bs, amp in configs}
    per_config_vram: dict[str, list[int]] = {k: [] for k in per_config}

    for round_i in range(args.rounds):
        rng = random.Random(42 + round_i)
        order = configs[:]
        rng.shuffle(order)
        round_row: dict[str, Any] = {"round": round_i, "order": [], "configs": {}}
        for bs, amp in order:
            key = f"batch{bs}_{'amp' if amp else 'fp32'}"
            round_row["order"].append(key)
            batch = _load_batch(policy, bs)
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.cuda.reset_peak_memory_stats()

            def fwd() -> Any:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=amp):
                        return model(x)

            times = _time_fn(fwd, args.warmup, args.trials)
            med = _median(times)
            per_config[key].append(med)
            vram = int(torch.cuda.max_memory_allocated())
            per_config_vram[key].append(vram)
            round_row["configs"][key] = {
                "median_sec": med,
                "p90_sec": _p90(times),
                "mean_sec": sum(times) / len(times),
                "ci95_median_sec": _bootstrap_ci(times),
                "throughput_samples_per_sec": bs / max(med, 1e-12),
                "peak_vram_bytes": vram,
            }
        report["rounds_detail"].append(round_row)

    for key, medians in per_config.items():
        bs = 8 if key.startswith("batch8") else 1
        agg_med = _median(medians)
        report["aggregate"][key] = {
            "round_medians_sec": medians,
            "median_of_round_medians_sec": agg_med,
            "p90_of_round_medians_sec": _p90(medians),
            "ci95_of_round_medians_sec": _bootstrap_ci(medians, n_boot=1000),
            "throughput_samples_per_sec": bs / max(agg_med, 1e-12),
            "peak_vram_bytes_median": int(_median([float(x) for x in per_config_vram[key]])),
        }

    # Module breakdown once (batch1 FP32) — profiler-like block timing, not canonical claim alone.
    batch = _load_batch(policy, 1)
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def time_block(name: str, fn: Callable[[], Any]) -> None:
        times = _time_fn(fn, args.warmup, args.trials)
        report["module_breakdown_batch1_fp32"][name] = {
            "median_sec": _median(times),
            "p90_sec": _p90(times),
            "ci95_median_sec": _bootstrap_ci(times),
        }

    with torch.no_grad():

        def rgb_path() -> Any:
            return model._encode_rgb(x)

        time_block("rgb_encode_temporal", rgb_path)
        rgb_ctx, rgb_flat_pad, rgb_flat = rgb_path()

        def skel_path() -> Any:
            frames = model.skel_encoder(
                x["coordinates"], x["velocity"], x["validity"], x.get("skel_pad_mask")
            )
            return model.skel_temporal(frames, x.get("skel_pad_mask"))

        time_block("skel_ctr_temporal", skel_path)
        skel_seq, skel_pool = skel_path()

        def fuse_base() -> Any:
            return model.base_fuse(torch.cat([rgb_ctx, skel_pool], dim=-1))

        time_block("base_fuse", fuse_base)
        h_base = fuse_base()

        def cross_gate() -> Any:
            crossed = model.cross(
                skel_seq, rgb_flat, skel_pad=x.get("skel_pad_mask"), rgb_pad=rgb_flat_pad
            )
            crossed_pool = model._pool_masked(crossed, x.get("skel_pad_mask"))
            gated, _ = model.gate(rgb_ctx, crossed_pool)
            u = model.update_ln(gated)
            return h_base + model.rezero() * u

        time_block("cross_gate_rezero", cross_gate)
        h = cross_gate()

        def cls_head() -> Any:
            return model.cls(model.dropout(h))

        time_block("cls", cls_head)

    mods = report["module_breakdown_batch1_fp32"]
    ranked = sorted(mods.items(), key=lambda kv: -float(kv[1]["median_sec"]))
    report["bottleneck_rank"] = [{"module": k, **v} for k, v in ranked]
    b1 = report["aggregate"]["batch1_fp32"]["median_of_round_medians_sec"]
    hist = float(report["historical_batch1_fp32_median_sec"])
    report["vs_historical_e4_profile_batch1_fp32"] = {
        "historical_median_sec": hist,
        "phase0_median_sec": b1,
        "relative_delta": (b1 - hist) / hist,
    }
    report["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out = _ROOT / args.out
    sha = atomic_write(out, report)
    print(
        json.dumps(
            {
                "event": "e4_r2_b4_baseline_written",
                "sha256": sha,
                "batch1_fp32_median_sec": b1,
                "bottleneck": ranked[0][0] if ranked else None,
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
