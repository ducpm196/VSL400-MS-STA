#!/usr/bin/env python3
"""Optional B4* vs B4-E diagnostic efficiency bench (not a model gate).

Uses locked seed-42 checkpoints only. No retrain, no test-split access.
Interleaved CUDA timing on RTX-class device; reports latency/VRAM and
Skeleton temporal-attention score-element totals from val sequence lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_rev  # noqa: E402
from src.models.ms_sta_v5 import SKEL_STRIDE4, build_ms_sta_v5, count_parameters  # noqa: E402
from src.training.ms_sta_v5_data import (  # noqa: E402
    VSL400MSSTAv5Dataset,
    collate_v5,
    load_json,
    load_split_role_ids,
)
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402

CLOSEOUT_REL = "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json"
OUT_REL = "Docs/results/ms_sta_v5_e/DIAGNOSTIC_EFFICIENCY_BENCH.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def _p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[int(0.9 * (len(s) - 1))]


def _paired_bootstrap_ci(
    deltas: list[float], *, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return sum(deltas) / n, lo, hi


def _load_model(variant: str, ckpt_path: Path, dropout: float, policy: dict, device: torch.device):
    model = build_ms_sta_v5(
        variant,
        num_classes=400,
        d_model=int(policy["model"]["d_model"]),
        nhead=int(policy["model"]["nhead"]),
        temporal_layers=int(policy["model"]["temporal_layers"]),
        dropout=float(dropout),
        alpha_max=float(policy["model"]["alpha_max"]),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def _time_model(model, batch: dict[str, Any], device: torch.device, warmup: int, trials: int) -> dict[str, Any]:
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = int(torch.cuda.max_memory_allocated())
    return {
        "times_sec": times,
        "median_latency_sec": _median(times),
        "p90_latency_sec": _p90(times),
        "peak_vram_bytes": peak,
        "throughput_samples_per_sec": len(batch["label"]) / max(_median(times), 1e-12),
    }


def _score_element_totals(ds: VSL400MSSTAv5Dataset) -> dict[str, Any]:
    """Aggregate Skeleton attention score elements from actual sequence lengths."""
    ref_total = 0.0
    opt_total = 0.0
    n = 0
    exact_9375 = 0
    for i in range(len(ds)):
        item = ds[i]
        t_valid = int(item["skel_len"])
        t_matrix = int(np.ceil(t_valid / SKEL_STRIDE4) * SKEL_STRIDE4) if t_valid else 0
        t_opt = t_matrix // SKEL_STRIDE4 if t_matrix else 0
        ref = float(t_matrix * t_matrix)
        opt = float(t_opt * t_opt)
        ref_total += ref
        opt_total += opt
        n += 1
        if t_matrix and abs(1.0 - opt / max(ref, 1.0) - 0.9375) < 1e-9:
            exact_9375 += 1
    reduction = 1.0 - (opt_total / max(ref_total, 1.0))
    return {
        "n_sequences": n,
        "ref_score_elements_total": ref_total,
        "opt_score_elements_total": opt_total,
        "score_element_reduction": reduction,
        "exact_93_75_fraction": exact_9375 / max(n, 1),
        "claim_scope": (
            "Skeleton temporal-attention score-matrix elements only; "
            "not total FLOPs/latency/VRAM"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("diagnostic_requires_cuda")

    closeout = load_json(_ROOT / CLOSEOUT_REL)
    if closeout.get("status") != "CLOSED_SUPPLEMENTARY_NO_GO":
        raise SystemExit("requires_v5_e_closeout")

    policy = load_json(_ROOT / "Docs/results/ms_sta_v5_e_policy/V5_E_POLICY.json")
    m0_policy = load_json(_ROOT / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json")
    e_confirm = load_json(_ROOT / "Docs/results/ms_sta_v5_e_runs/CONFIRM_RESULTS.json")["trials"]
    b4_confirm = load_json(_ROOT / "Docs/results/ms_sta_v5_m0_runs/CONFIRM_RESULTS.json")["trials"]
    e42 = next(t for t in e_confirm if int(t["seed"]) == 42)
    b442 = next(t for t in b4_confirm if t["variant"] == "b4_star_fixed" and int(t["seed"]) == 42)

    device = torch.device("cuda")
    b4 = _load_model("b4_star_fixed", _ROOT / b442["best_checkpoint"], float(b442["dropout"]), m0_policy, device)
    e = _load_model("b4_star_e_stride4", _ROOT / e42["best_checkpoint"], float(e42["dropout"]), policy, device)

    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    val_ids = load_split_role_ids(split, "val")[: int(args.batch_size)]
    ds = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    batch = collate_v5([ds[i] for i in range(len(ds))])

    # Score-element distribution over full val (not just timing batch).
    val_all = load_split_role_ids(split, "val")
    ds_all = VSL400MSSTAv5Dataset(
        val_all,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
        require_all=True,
    )
    score_stats = _score_element_totals(ds_all)

    # Interleaved: alternate B4* and E each trial after shared warmups.
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = b4(x)
            _ = e(x)
        torch.cuda.synchronize()
        b4_times: list[float] = []
        e_times: list[float] = []
        for i in range(args.trials):
            if i % 2 == 0:
                t0 = time.perf_counter()
                _ = b4(x)
                torch.cuda.synchronize()
                b4_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                _ = e(x)
                torch.cuda.synchronize()
                e_times.append(time.perf_counter() - t0)
            else:
                t0 = time.perf_counter()
                _ = e(x)
                torch.cuda.synchronize()
                e_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                _ = b4(x)
                torch.cuda.synchronize()
                b4_times.append(time.perf_counter() - t0)
        # Peak after interleaved run (shared process).
        peak_shared = int(torch.cuda.max_memory_allocated())

    # Separate peak VRAM per model (reset between).
    b4_solo = _time_model(b4, batch, device, args.warmup, max(10, args.trials // 5))
    e_solo = _time_model(e, batch, device, args.warmup, max(10, args.trials // 5))

    deltas = [e_t - b4_t for e_t, b4_t in zip(e_times, b4_times)]
    mean_d, ci_lo, ci_hi = _paired_bootstrap_ci(deltas, n_boot=args.boot, seed=42)
    med_b4 = _median(b4_times)
    med_e = _median(e_times)
    lat_reduction = 1.0 - (med_e / max(med_b4, 1e-12))

    gate = load_json(_ROOT / "Docs/results/ms_sta_v5_e/ACCURACY_NONINFERIORITY_GATE.json")
    report = {
        "schema": "ms_sta_v5_e_diagnostic_efficiency_v1",
        "status": "SUPPLEMENTARY_ONLY_NOT_A_GATE",
        "not_model_selection_gate": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": "float32",
        "batch_size": len(ds),
        "warmup": args.warmup,
        "trials_per_model": len(b4_times),
        "interleaved": True,
        "head": git_rev(_ROOT),
        "closeout_sha256": sha256_file(_ROOT / CLOSEOUT_REL),
        "checkpoints": {
            "b4_star_fixed_seed42": str(Path(b442["best_checkpoint"]).as_posix()),
            "b4_star_e_stride4_seed42": str(Path(e42["best_checkpoint"]).as_posix()),
        },
        "param_count": {
            "b4_star_fixed": count_parameters(b4),
            "b4_star_e_stride4": count_parameters(e),
        },
        "latency_interleaved": {
            "b4_median_sec": med_b4,
            "b4_p90_sec": _p90(b4_times),
            "e_median_sec": med_e,
            "e_p90_sec": _p90(e_times),
            "b4_throughput_samples_per_sec": len(ds) / max(med_b4, 1e-12),
            "e_throughput_samples_per_sec": len(ds) / max(med_e, 1e-12),
            "median_latency_reduction_e_vs_b4": lat_reduction,
            "paired_delta_e_minus_b4_mean_sec": mean_d,
            "paired_delta_ci95_sec": [ci_lo, ci_hi],
            "shared_peak_vram_bytes_interleaved": peak_shared,
        },
        "vram_solo": {
            "b4_peak_vram_bytes": b4_solo["peak_vram_bytes"],
            "e_peak_vram_bytes": e_solo["peak_vram_bytes"],
        },
        "skeleton_score_elements_val": score_stats,
        "validation_accuracy_tradeoff_pp": {
            "mean_delta_top1_pp": gate["mean_delta_top1_pp"],
            "source": "Docs/results/ms_sta_v5_e/ACCURACY_NONINFERIORITY_GATE.json",
        },
        "allowed_conclusion_template": (
            "B4-E reduces attention-score elements and measured latency by X, with an "
            "observed validation accuracy trade-off of 1.42 pp."
        ),
        "forbidden_conclusions": [
            "B4-E preserves accuracy",
            "B4-E is superior",
            "the whole system is 93.75% faster/cheaper",
        ],
        "claim_attention_score_reduction_93_75_total_system": False,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Fill concrete allowed conclusion numbers.
    report["allowed_conclusion"] = (
        f"B4-E reduces Skeleton temporal-attention score elements by "
        f"{100.0 * score_stats['score_element_reduction']:.2f}% (score matrix only) and "
        f"median interleaved latency by {100.0 * lat_reduction:.2f}%, with an observed "
        f"validation accuracy trade-off of {abs(gate['mean_delta_top1_pp']):.2f} pp "
        f"(E below B4*; non-inferiority failed)."
    )

    out = _ROOT / OUT_REL
    sha = atomic_write(out, report)
    print(json.dumps({"event": "DIAGNOSTIC_BENCH_WRITTEN", "sha256": sha, "rel": OUT_REL, **{
        "score_reduction": score_stats["score_element_reduction"],
        "latency_reduction": lat_reduction,
        "mean_delta_top1_pp": gate["mean_delta_top1_pp"],
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
