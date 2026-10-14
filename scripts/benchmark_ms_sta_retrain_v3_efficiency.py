"""Retrain-v3 efficiency benchmark: resolve B4/B4-E seed-42 finals from campaign state."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import load_json, sha256_file
from src.models.ms_sta_features import resolve_pooled_dir, resolve_skel_dir, resolve_temporal_dir
from src.models.ms_sta_wordlevel import build_ms_sta
from src.training.retrain_v3 import (
    load_policy,
    require_frozen_v3_policy,
    skeleton_attention_score_elements,
)
from src.training.train_ms_sta_vsl400 import VSL400MSSTADataset, _collate
from torch.utils.data import DataLoader

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_retrain_v3_policy/RETRAIN_POLICY.json"


def _percent_reduction(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError("before must be > 0")
    return 100.0 * (1.0 - after / before)


def _resolve_confirm_final(state: dict, *, variant: str, seed: int = 42) -> dict:
    runs = ((state.get("phases") or {}).get("confirm") or {}).get("runs") or []
    matches = [
        r
        for r in runs
        if r.get("variant") == variant
        and int(r.get("seed") or -1) == int(seed)
        and int(r.get("exit_code") or 1) == 0
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one confirm run for {variant} seed={seed}, got {len(matches)}")
    row = matches[0]
    if row.get("checkpoint_status") != "final_selected_checkpoint":
        raise ValueError(f"{variant} seed={seed} is not final_selected_checkpoint")
    final_rel = row.get("final_checkpoint")
    if not final_rel:
        raise ValueError(f"{variant} seed={seed} missing final_checkpoint path")
    return row


def _load_val_batch(batch_size: int):
    import pandas as pd

    registry = load_json(_ROOT / "data/processed/ms_sta_registry.json")
    labels = load_json(_ROOT / "data/processed/vsl400_labels.json")
    split = load_json(_ROOT / "data/processed/vsl400_retrain_v3_split_manifest.json")
    val_ids = [str(x) for x in split["val_ids"]]
    metadata = pd.read_csv(_ROOT / "metadata.csv")
    df = metadata[metadata["dataset"] == "VSL400"]
    dataset = VSL400MSSTADataset(
        df,
        labels["gloss_to_id"],
        require_skel=True,
        require_temporal_rgb=True,
        rgb_pooled_dir=resolve_pooled_dir(_ROOT, registry),
        rgb_temporal_dir=resolve_temporal_dir(_ROOT, registry),
        skel_dir=resolve_skel_dir(_ROOT, registry),
        video_ids=val_ids,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    rgb, skel, _y, ids, rgb_kpm = next(iter(loader))
    if skel is None or rgb_kpm is None:
        raise RuntimeError("benchmark requires Skeleton and RGB validity mask")
    return rgb, skel, rgb_kpm, ids


def _measure(ckpt: Path, stride: int, rgb, skel, rgb_kpm, device, repeats: int, warmup: int):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_status") != "final_selected_checkpoint":
        raise ValueError(f"{ckpt} checkpoint_status != final_selected_checkpoint")
    if int(payload.get("seed") or -1) != 42:
        raise ValueError(f"{ckpt} seed != 42")
    model = build_ms_sta("b4", num_classes=int(payload["num_classes"]), skel_temporal_stride=stride)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    rgb_d = rgb.to(device)
    skel_d = skel.to(device)
    kpm_d = rgb_kpm.to(device)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(rgb=rgb_d, skel=skel_d, rgb_key_padding_mask=kpm_d)
        if device.type == "cuda":
            torch.cuda.synchronize()
    times: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            with torch.no_grad():
                _ = model(rgb=rgb_d, skel=skel_d, rgb_key_padding_mask=kpm_d)
            ender.record()
            torch.cuda.synchronize()
            times.append(float(starter.elapsed_time(ender)))
        else:
            import time

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(rgb=rgb_d, skel=skel_d, rgb_key_padding_mask=kpm_d)
            times.append((time.perf_counter() - t0) * 1000.0)
    vram = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else None
    return {
        "params": int(params),
        "raw_latency_ms": times,
        "median_forward_latency_ms": float(statistics.median(times)),
        "mean_forward_latency_ms": float(statistics.mean(times)),
        "vram_peak_mb": vram,
        "checkpoint_sha256": sha256_file(ckpt),
        "skel_temporal_stride": stride,
        "seed": 42,
        "checkpoint_status": "final_selected_checkpoint",
        "retrain_v3_policy_sha256": payload.get("retrain_v3_policy_sha256"),
        "manifest_sha256": payload.get("manifest_sha256"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    policy_sha = sha256_file(policy_path)
    status_dir = _ROOT / str((policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3")
    state_path = status_dir / "RUN_STATE.json"
    if not state_path.is_file():
        print("[FAIL] missing RUN_STATE.json")
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("policy_sha256") != policy_sha:
        print("[FAIL] RUN_STATE policy sha mismatch")
        return 2
    if ((state.get("phases") or {}).get("confirm") or {}).get("status") != "COMPLETED":
        print("[FAIL] confirm phase not COMPLETED")
        return 2

    try:
        row_b4 = _resolve_confirm_final(state, variant="b4", seed=42)
        row_b4e = _resolve_confirm_final(state, variant="b4e", seed=42)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    b4 = _ROOT / str(row_b4["final_checkpoint"])
    b4e = _ROOT / str(row_b4e["final_checkpoint"])
    if sha256_file(b4) != row_b4.get("final_checkpoint_sha256"):
        print("[FAIL] b4 final sha mismatch vs RUN_STATE")
        return 2
    if sha256_file(b4e) != row_b4e.get("final_checkpoint_sha256"):
        print("[FAIL] b4e final sha mismatch vs RUN_STATE")
        return 2

    split_sha = (policy.get("lineage") or {}).get("split_manifest_sha256")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if str(args.device).startswith("cuda") and device.type != "cuda":
        print("[FAIL] CUDA unavailable")
        return 2

    rgb, skel, kpm, ids = _load_val_batch(args.batch_size)
    m4 = _measure(b4, 1, rgb, skel, kpm, device, args.repeats, args.warmup)
    m4e = _measure(b4e, 4, rgb, skel, kpm, device, args.repeats, args.warmup)
    if m4.get("retrain_v3_policy_sha256") != policy_sha:
        print("[FAIL] b4 payload policy sha missing/mismatch")
        return 2
    if m4e.get("retrain_v3_policy_sha256") != policy_sha:
        print("[FAIL] b4e payload policy sha missing/mismatch")
        return 2
    if not split_sha or m4.get("manifest_sha256") != split_sha:
        print("[FAIL] b4 split/manifest sha missing/mismatch")
        return 2
    if not split_sha or m4e.get("manifest_sha256") != split_sha:
        print("[FAIL] b4e split/manifest sha missing/mismatch")
        return 2

    before = skeleton_attention_score_elements(64, 1)
    after = skeleton_attention_score_elements(64, 4)
    out = args.out or (status_dir / "EFFICIENCY_BENCHMARK.json")
    if not Path(out).is_absolute():
        out = _ROOT / out
    report = {
        "schema": "ms_sta_retrain_v3_efficiency_benchmark_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "resolved_from": "RUN_STATE.confirm.seed42.final",
        "batch_video_ids": ids,
        "score_elems_before": before,
        "score_elems_after": after,
        "score_reduction_pct": _percent_reduction(before, after),
        "b4": {**m4, "path": row_b4["final_checkpoint"], "variant": "b4"},
        "b4e": {**m4e, "path": row_b4e["final_checkpoint"], "variant": "b4e"},
        "latency_reduction_pct": _percent_reduction(
            m4["median_forward_latency_ms"], m4e["median_forward_latency_ms"]
        ),
        "param_count_delta": int(m4e["params"] - m4["params"]),
        "vram_b4_mb": m4.get("vram_peak_mb"),
        "vram_b4e_mb": m4e.get("vram_peak_mb"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "reuse_metrics": False,
        "cli_checkpoint_override_forbidden": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "latency_reduction_pct": report["latency_reduction_pct"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
