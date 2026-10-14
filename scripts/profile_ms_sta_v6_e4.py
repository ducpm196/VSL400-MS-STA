#!/usr/bin/env python3
"""E4 teacher profiler on RTX-class GPU — bottleneck before finite policy freeze.

Measures b4_v6_ctr seed-42 (external artifact). No train/test/demo.
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
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json, load_split_role_ids  # noqa: E402
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
    ap.add_argument("--policy", default="Docs/results/ms_sta_v6_policy/V6_POLICY.json")
    ap.add_argument("--artifact-index", default="Docs/results/ms_sta_v6_main_integration/MAIN_INTEGRATION_ARTIFACT_INDEX.json")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default="Docs/results/ms_sta_v6_e4/E4_PROFILE_REPORT.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("cuda_required")
    device = torch.device("cuda")
    policy = load_json(_ROOT / args.policy)
    art = load_json(_ROOT / args.artifact_index)
    ckpt_path = Path(art["checkpoints"]["42"]["external_path"])
    want = art["checkpoints"]["42"]["sha256"]
    got = sha256_file(ckpt_path)
    if got != want:
        raise SystemExit(f"teacher_sha_mismatch:{got}")

    model = build_ms_sta_v6("b4_v6_ctr", dropout=0.0, gate_entropy_reg=0.0).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    report: dict[str, Any] = {
        "schema": "ms_sta_v6_e4_profile_report_v1",
        "teacher": "b4_v6_ctr",
        "seed": 42,
        "checkpoint_sha256": got,
        "param_count": count_parameters(model),
        "device": str(torch.cuda.get_device_name(0)),
        "warmup": args.warmup,
        "trials": args.trials,
        "configs": {},
        "module_breakdown_batch1_fp32": {},
        "hypothesis_hints": [],
    }

    # Interleaved order of (batch, amp)
    configs = [(1, False), (1, True), (8, False), (8, True)]
    rng = random.Random(42)
    order = configs[:]
    rng.shuffle(order)

    for bs, amp in order:
        batch = _load_batch(policy, bs)
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        torch.cuda.reset_peak_memory_stats()
        # H2D already done; measure pure forward
        def fwd():
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=amp):
                    return model(x)

        times = _time_fn(fwd, args.warmup, args.trials)
        key = f"batch{bs}_{'amp' if amp else 'fp32'}"
        report["configs"][key] = {
            "batch_size": bs,
            "amp": amp,
            "median_sec": _median(times),
            "p90_sec": _p90(times),
            "mean_sec": sum(times) / len(times),
            "throughput_samples_per_sec": bs / max(_median(times), 1e-12),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }

    # Module breakdown batch1 FP32 (hooks via partial forward using model pieces)
    batch = _load_batch(policy, 1)
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def time_block(name: str, fn: Callable[[], Any]) -> None:
        times = _time_fn(fn, args.warmup, args.trials)
        report["module_breakdown_batch1_fp32"][name] = {
            "median_sec": _median(times),
            "p90_sec": _p90(times),
        }

    with torch.no_grad():
        # RGB path
        def rgb_path():
            rgb_ctx, rgb_flat_pad, rgb_flat = model._encode_rgb(x)
            return rgb_ctx, rgb_flat_pad, rgb_flat

        time_block("rgb_encode_temporal", rgb_path)
        rgb_ctx, rgb_flat_pad, rgb_flat = rgb_path()

        def skel_path():
            coords, velocity, validity = x["coordinates"], x["velocity"], x["validity"]
            skel_pad = x.get("skel_pad_mask")
            frames = model.skel_encoder(coords, velocity, validity, skel_pad)
            seq, pool = model.skel_temporal(frames, skel_pad)
            return seq, pool

        time_block("skel_ctr_temporal", skel_path)
        skel_seq, skel_pool = skel_path()

        def fuse_base():
            return model.base_fuse(torch.cat([rgb_ctx, skel_pool], dim=-1))

        time_block("base_fuse", fuse_base)
        h_base = fuse_base()

        def cross_gate():
            crossed = model.cross(skel_seq, rgb_flat, skel_pad=x.get("skel_pad_mask"), rgb_pad=rgb_flat_pad)
            crossed_pool = model._pool_masked(crossed, x.get("skel_pad_mask"))
            gated, _ = model.gate(rgb_ctx, crossed_pool)
            u = model.update_ln(gated)
            alpha = model.rezero()
            return h_base + alpha * u

        time_block("cross_gate_rezero", cross_gate)
        h = cross_gate()

        def cls_head():
            return model.cls(model.dropout(h))

        time_block("cls", cls_head)

    # Rank modules
    mods = report["module_breakdown_batch1_fp32"]
    ranked = sorted(mods.items(), key=lambda kv: -float(kv[1]["median_sec"]))
    report["bottleneck_rank"] = [{"module": k, **v} for k, v in ranked]
    top = ranked[0][0] if ranked else ""
    hints = []
    if top.startswith("skel"):
        hints.append("Prefer structured skel width/layer reduction over temporal stride (E2/E3 NO-GO).")
    if top.startswith("rgb"):
        hints.append("Prefer RGB temporal width/layer reduction; avoid score-matrix stride claims.")
    if top.startswith("cross"):
        hints.append("Prefer lighter cross/gate or distill-to-base-fuse student.")
    hints.append("Execution-preserving: AMP already profiled; durable E4 candidate must change model compute.")
    hints.append("Do not claim 93.75% whole-model speedup; only score-element reductions with shape evidence.")
    report["hypothesis_hints"] = hints
    report["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out = _ROOT / args.out
    if out.is_file():
        out.unlink()
    sha = atomic_write(out, report)
    print(json.dumps({"event": "e4_profile_written", "sha256": sha, "bottleneck": top, "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
