#!/usr/bin/env python3
"""E4-R2 Phase-1 detailed CUDA / torch.profiler report for B4 bottlenecks.

Canonical latency remains the Phase-0/6 timed region with CUDA synchronize.
This script records shapes/calls/params estimates and a short profiler trace;
profiler overhead is NOT used as the efficiency gate number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

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


def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_POLICY.json")
    ap.add_argument("--out", default="Docs/results/ms_sta_v6_e4_r2/E4_R2_PROFILER_REPORT.json")
    ap.add_argument("--trace-dir", default="Docs/results/ms_sta_v6_e4_r2/profiler_trace")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("cuda_required")
    device = torch.device("cuda")
    policy = load_json(_ROOT / args.policy)
    teacher = policy["teacher"]
    ckpt = _ROOT / teacher["checkpoint_rel"]
    got = sha256_file(ckpt)
    if got != teacher["checkpoint_sha256"]:
        raise SystemExit(f"teacher_sha_mismatch:{got}")

    model = build_ms_sta_v6("b4_v6_ctr", dropout=0.0, gate_entropy_reg=0.0).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    batch = _load_batch(policy, 1)
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    shapes: dict[str, Any] = {
        "input": {k: _tensor_meta(v) for k, v in x.items() if torch.is_tensor(v)},
        "params": count_parameters(model),
        "modules": {
            "rgb_proj_regions": sorted(model.rgb_proj.keys()),
            "rgb_proj_in_features": int(model.rgb_proj["global"].in_features),
            "rgb_proj_out_features": int(model.rgb_proj["global"].out_features),
            "d_model": int(model.d_model),
            "nhead": int(getattr(model, "nhead", getattr(model.rgb_temporal, "nhead", 4))),
            "temporal_layers": int(
                getattr(model, "temporal_layers", getattr(model.rgb_temporal, "num_layers", 2))
            ),
        },
    }

    with torch.no_grad():
        rgb_ctx, rgb_flat_pad, rgb_flat = model._encode_rgb(x)
        frames = model.skel_encoder(
            x["coordinates"], x["velocity"], x["validity"], x.get("skel_pad_mask")
        )
        skel_seq, skel_pool = model.skel_temporal(frames, x.get("skel_pad_mask"))
        shapes["intermediates"] = {
            "rgb_ctx": _tensor_meta(rgb_ctx),
            "rgb_flat": _tensor_meta(rgb_flat),
            "skel_frames": _tensor_meta(frames),
            "skel_seq": _tensor_meta(skel_seq),
            "skel_pool": _tensor_meta(skel_pool),
        }

    # Short profiler capture (diagnostic only).
    trace_dir = _ROOT / args.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    key_avgs: list[dict[str, Any]] = []
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(20):
                _ = model(x)
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_dir / "b4_batch1_fp32_chrome.json"))
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    (trace_dir / "b4_batch1_fp32_table.txt").write_text(table, encoding="utf-8")
    for row in prof.key_averages():
        key_avgs.append(
            {
                "key": row.key,
                "count": int(row.count),
                "self_cuda_time_total_us": float(getattr(row, "self_cuda_time_total", 0.0)),
                "cuda_time_total_us": float(getattr(row, "cuda_time_total", 0.0)),
            }
        )
    key_avgs.sort(key=lambda r: -r["self_cuda_time_total_us"])

    report = {
        "schema": "ms_sta_v6_e4_r2_profiler_report_v1",
        "campaign": "E4-R2",
        "phase": "1_profiler",
        "teacher_checkpoint_sha256": got,
        "device": str(torch.cuda.get_device_name(0)),
        "canonical_latency_source": "Docs/results/ms_sta_v6_e4_r2/E4_R2_B4_LATENCY_BASELINE.json",
        "profiler_overhead_excluded_from_efficiency_gate": True,
        "shapes": shapes,
        "torch_profiler_top_self_cuda": key_avgs[:40],
        "trace_rel": str(Path(args.trace_dir).as_posix()),
        "hypothesis_targets": [
            "Reduce skel_ctr_temporal token/kernel work (motion-aware temporal + separable graph).",
            "Reduce rgb projection kernel launches (shared low-rank) and 4L tokens before cross-attn.",
            "Keep external caches identical; no grayscale.",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sha = atomic_write(_ROOT / args.out, report)
    print(json.dumps({"event": "e4_r2_profiler_written", "sha256": sha, "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
