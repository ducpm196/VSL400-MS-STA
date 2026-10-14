# -*- coding: utf-8 -*-
"""Benchmark E4-R4C DataLoader num_workers×pin_memory (post-HPO, pre-confirm).

Create-only: Docs/results/ms_sta_v6_e4_r4/E4_R4C_DATALOADER_BENCHMARK.json
Does not mutate trainer. Does not launch confirm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.ms_sta_e4_r4c_data import VSL400E4R4CDataset, collate_e4_r4c
from src.training.ms_sta_v5_data import load_split_role_ids

# Declared before run
NUM_WORKERS_GRID = (0, 2, 4)
PIN_MEMORY_GRID = (False, True)
SUBSET_N = 256
BATCH_SIZE = 8
WARMUP_BATCHES = 4
MEASURE_TRIALS = 3
MEASURE_BATCHES = 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
        "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
    }


def iter_batches(loader: DataLoader, n: int) -> tuple[list[float], int, int]:
    latencies: list[float] = []
    n_samples = 0
    worker_errors = 0
    it = iter(loader)
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        except Exception:
            worker_errors += 1
            continue
        dt = time.perf_counter() - t0
        latencies.append(dt)
        n_samples += int(batch["label"].shape[0])
        # Touch tensors so pin_memory transfer is measured if caller .cuda()'s later
        _ = batch["label"].sum().item()
    return latencies, n_samples, worker_errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "data/features/ms_sta_v6_e4_r4c_r1_sa_a365896280f0",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_DATALOADER_BENCHMARK.json",
    )
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise SystemExit(f"refuse_overwrite:{out}")

    train_pol = json.loads(
        (ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json").read_text(encoding="utf-8")
    )
    split = json.loads((ROOT / train_pol["split_manifest_rel"]).read_text(encoding="utf-8"))
    if sha256_file(ROOT / train_pol["split_manifest_rel"]) != train_pol["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")

    meta = load_meta(ROOT / "metadata.csv")
    labels_obj = json.loads((ROOT / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    labels = {}
    for sid, row in meta.items():
        gloss = str(row.get("text_gloss") or "").strip()
        if gloss in g2i:
            labels[sid] = int(g2i[gloss])
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ids = load_split_role_ids(split, "confirm_train")
    feature_root = args.feature_root if args.feature_root.is_absolute() else ROOT / args.feature_root
    ds = VSL400E4R4CDataset(
        ids,
        feature_root=feature_root,
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    subset = Subset(ds, list(range(min(SUBSET_N, len(ds)))))

    rows = []
    for nw in NUM_WORKERS_GRID:
        for pin in PIN_MEMORY_GRID:
            loader = DataLoader(
                subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=nw,
                pin_memory=pin,
                collate_fn=collate_e4_r4c,
                persistent_workers=(nw > 0),
            )
            # warmup
            iter_batches(loader, WARMUP_BATCHES)
            trial_sps: list[float] = []
            trial_med: list[float] = []
            trial_p90: list[float] = []
            worker_errors_total = 0
            rss_before = rss_mb()
            for _ in range(MEASURE_TRIALS):
                lats, n_samp, werr = iter_batches(loader, MEASURE_BATCHES)
                worker_errors_total += werr
                if not lats:
                    continue
                elapsed = sum(lats)
                trial_sps.append(n_samp / max(elapsed, 1e-9))
                trial_med.append(statistics.median(lats))
                sorted_l = sorted(lats)
                p90_i = min(len(sorted_l) - 1, int(0.9 * (len(sorted_l) - 1)))
                trial_p90.append(sorted_l[p90_i])
            if hasattr(loader, "_iterator") and loader._iterator is not None:
                del loader._iterator
            row = {
                "num_workers": nw,
                "pin_memory": pin,
                "samples_per_sec_mean": float(statistics.mean(trial_sps)) if trial_sps else None,
                "samples_per_sec_trials": trial_sps,
                "batch_latency_median_sec_mean": float(statistics.mean(trial_med)) if trial_med else None,
                "batch_latency_p90_sec_mean": float(statistics.mean(trial_p90)) if trial_p90 else None,
                "rss_mb_before": rss_before,
                "rss_mb_after": rss_mb(),
                "gpu": gpu_stats(),
                "worker_errors": worker_errors_total,
            }
            rows.append(row)
            print(
                f"BENCH nw={nw} pin={pin} sps={row['samples_per_sec_mean']:.2f}"
                if row["samples_per_sec_mean"] is not None
                else f"BENCH nw={nw} pin={pin} FAILED",
                flush=True,
            )

    baseline = next(r for r in rows if r["num_workers"] == 0 and r["pin_memory"] is False)
    report = {
        "schema": "ms_sta_v6_e4_r4c_dataloader_benchmark_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "protocol": {
            "role": "confirm_train",
            "subset_n": SUBSET_N,
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "warmup_batches": WARMUP_BATCHES,
            "measure_trials": MEASURE_TRIALS,
            "measure_batches": MEASURE_BATCHES,
            "num_workers_grid": list(NUM_WORKERS_GRID),
            "pin_memory_grid": list(PIN_MEMORY_GRID),
            "feature_root_rel": str(feature_root.relative_to(ROOT)).replace("\\", "/"),
            "split_manifest_sha256": train_pol["split_manifest_sha256"],
        },
        "baseline": {"num_workers": 0, "pin_memory": False, "samples_per_sec_mean": baseline["samples_per_sec_mean"]},
        "rows": rows,
        "e4_ready": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DATALOADER_BENCHMARK_OK", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
