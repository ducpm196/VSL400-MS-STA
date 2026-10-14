"""Benchmark B4 Skeleton stride-4 downsampling without touching E2.

This measures GPU forward latency and peak allocated memory on a frozen
validation batch. It never evaluates labels or writes defense-ready artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import git_dirty, git_rev, load_json, sha256_file
from src.models.ms_sta_features import resolve_pooled_dir, resolve_skel_dir, resolve_temporal_dir
from src.models.ms_sta_wordlevel import build_ms_sta
from src.training.train_ms_sta_vsl400 import VSL400MSSTADataset, _collate


def _percent_reduction(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError("before must be > 0")
    return 100.0 * (1.0 - after / before)


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(0.95 * len(values)) - 1)]


def _id_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def _load_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    registry = load_json(_ROOT / "data" / "processed" / "ms_sta_registry.json")
    labels = load_json(_ROOT / "data" / "processed" / "vsl400_labels.json")
    split = load_json(_ROOT / "data" / "processed" / "vsl400_e1_split_manifest.json")
    eligible = load_json(_ROOT / "data" / "processed" / "vsl400_eligible_sample_manifest.json")
    eligible_ids = set(str(x) for x in eligible["eligible_ids"])
    val_ids = [str(x) for x in split["val_ids"] if str(x) in eligible_ids]

    import pandas as pd

    metadata = pd.read_csv(_ROOT / "metadata.csv")
    val_df = metadata[(metadata["dataset"] == "VSL400") & (metadata["split"] == "val")]
    dataset = VSL400MSSTADataset(
        val_df,
        labels["gloss_to_id"],
        require_skel=True,
        require_temporal_rgb=True,
        rgb_pooled_dir=resolve_pooled_dir(_ROOT, registry),
        rgb_temporal_dir=resolve_temporal_dir(_ROOT, registry),
        skel_dir=resolve_skel_dir(_ROOT, registry),
        video_ids=val_ids,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    rgb, skel, _labels, ids, rgb_kpm = next(iter(loader))
    if skel is None or rgb_kpm is None:
        raise RuntimeError("benchmark requires Skeleton and RGB validity mask")
    return rgb, skel, rgb_kpm, ids


def _run_round(
    *,
    state: dict,
    stride: int,
    rgb: torch.Tensor,
    skel: torch.Tensor,
    rgb_kpm: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    torch.cuda.empty_cache()
    model = build_ms_sta("b4", num_classes=400, skel_temporal_stride=stride)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    rgb_d = rgb.to(device)
    skel_d = skel.to(device)
    kpm_d = rgb_kpm.to(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(rgb=rgb_d, skel=skel_d, rgb_key_padding_mask=kpm_d)
        torch.cuda.synchronize()
        steady_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(rgb=rgb_d, skel=skel_d, rgb_key_padding_mask=kpm_d)
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        peak_allocated = torch.cuda.max_memory_allocated(device)

    result = {
        "stride": stride,
        "times_ms": times,
        "peak_total_mb": peak_allocated / (1 << 20),
        "peak_incremental_mb": max(0, peak_allocated - steady_allocated) / (1 << 20),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    del model, rgb_d, skel_d, kpm_d
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled MS-STA Skeleton temporal benchmark")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/processed/ms_sta_e1_v2/b4_seed42.pth"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/ms_sta_efficiency_pilot/BENCHMARK.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA required")
        return 2
    if args.batch_size < 1 or args.warmup < 1 or args.repeats < 10 or args.rounds < 2:
        print("[FAIL] invalid benchmark counts")
        return 2
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else _ROOT / args.checkpoint
    if not checkpoint.is_file():
        print(f"[FAIL] checkpoint missing: {checkpoint}")
        return 2
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "b4" or not isinstance(payload.get("model"), dict):
        print("[FAIL] checkpoint is not a B4 model payload")
        return 2

    rgb, skel, rgb_kpm, ids = _load_batch(args.batch_size)
    if skel.shape[1] != 64:
        print(f"[FAIL] expected canonical Skeleton T=64, got {skel.shape[1]}")
        return 2
    device = torch.device("cuda")
    orders = ([1, 4], [4, 1])
    raw: dict[int, list[dict]] = {1: [], 4: []}
    order_trace: list[int] = []
    for round_idx in range(args.rounds):
        for stride in orders[round_idx % 2]:
            order_trace.append(stride)
            raw[stride].append(
                _run_round(
                    state=payload["model"],
                    stride=stride,
                    rgb=rgb,
                    skel=skel,
                    rgb_kpm=rgb_kpm,
                    device=device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

    summary = {}
    for stride in (1, 4):
        times = [x for run in raw[stride] for x in run["times_ms"]]
        summary[str(stride)] = {
            "skeleton_tokens": 64 if stride == 1 else 16,
            "median_latency_ms": statistics.median(times),
            "mean_latency_ms": statistics.mean(times),
            "p95_latency_ms": _p95(times),
            "peak_total_mb_median": statistics.median(x["peak_total_mb"] for x in raw[stride]),
            "peak_incremental_mb_median": statistics.median(
                x["peak_incremental_mb"] for x in raw[stride]
            ),
            "parameter_count": raw[stride][0]["parameter_count"],
        }
    base = summary["1"]
    down = summary["4"]
    result = {
        "schema": "ms_sta_efficiency_benchmark_v1",
        "status": "PILOT_ONLY",
        "claim_allowed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev(_ROOT),
        "git_dirty": git_dirty(_ROOT),
        "checkpoint": checkpoint.relative_to(_ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_role": "B4 E1 seed42 weights; same weights loaded for both architectures",
        "device": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "dtype": "float32",
        },
        "protocol": {
            "scope": "model forward only; excludes feature extraction, disk I/O, API and UI",
            "batch_size": args.batch_size,
            "warmup_per_round": args.warmup,
            "repeats_per_round": args.repeats,
            "rounds": args.rounds,
            "order": order_trace,
            "timing": "CUDA events with synchronization",
            "input_split": "frozen E1 validation",
            "input_ids": ids,
            "input_ids_sha256": _id_sha256(ids),
            "rgb_padded_tokens": int(rgb.shape[1]),
            "rgb_valid_tokens": sorted(set(int(x) for x in (~rgb_kpm).sum(dim=1).tolist())),
            "skeleton_tokens_before": int(skel.shape[1]),
        },
        "variants": summary,
        "reductions_pct": {
            "skeleton_self_attention_score_elements": _percent_reduction(64**2, 16**2),
            "model_forward_median_latency": _percent_reduction(
                base["median_latency_ms"], down["median_latency_ms"]
            ),
            "model_forward_peak_incremental_vram": _percent_reduction(
                base["peak_incremental_mb_median"], down["peak_incremental_mb_median"]
            ),
        },
        "limits": [
            "Pilot worktree may be dirty; result is not defense evidence.",
            "The 93.75% figure applies only to Skeleton self-attention score elements.",
            "Latency and memory are measured separately and must not be called 93.75%.",
            "Accuracy requires a clean three-seed validation study before any claim.",
            "Test E2 is not accessed by this benchmark."
        ],
    }
    out = args.out if args.out.is_absolute() else _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["reductions_pct"], indent=2))
    print(f"[PASS] benchmark -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
