# -*- coding: utf-8 -*-
"""Independent incremental-VRAM verify for E4-R4B (create-only; does not overwrite R4B raw).

Formula: peak_during_stage - allocated_immediately_before_stage
Also records reserved_before / peak_reserved. Compares B4 vs R1 incremental only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import (  # noqa: E402
    R7C_AMBIGUITY_ROI_LOCK,
    b4_item_from_extract,
    e4_batch_from_frames,
    sealed_landmarks,
)
from scripts.ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
)
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.ms_sta_v6_e4_r4 import build_ms_sta_v6_e4_r4  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    FrozenSlowFastBlock4,
    extract_rgb_token_v3_for_video,
    read_video_frames_native,
)
from src.training.ms_sta_v5_data import collate_v5  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4"
PROTOCOL = OUT / "E4_R4B_STAGE_MATCHED_PROTOCOL.json"
FREEZE = OUT / "E4_R4_PILOT_SAMPLE_FREEZE.json"
VERIFY_OUT = OUT / "E4_R4B_VRAM_ACCOUNTING_VERIFY.json"
META = ROOT / "metadata.csv"
N_SAMPLES = 4
VRAM_REGRESS_MAX = 0.05


def die(msg: str) -> None:
    print(f"E4_R4B_VRAM_VERIFY_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_create_only(path: Path, payload: dict) -> str:
    if path.is_file():
        die(f"refuse_overwrite:{path}")
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return float(s[len(s) // 2]) if s else float("nan")


def snapshot() -> dict[str, float]:
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
    }


def measure_incremental(stage_fn) -> dict[str, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = snapshot()
    torch.cuda.reset_peak_memory_stats()
    stage_fn()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    after = snapshot()
    return {
        "allocated_before_mb": before["allocated_mb"],
        "reserved_before_mb": before["reserved_mb"],
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "allocated_after_mb": after["allocated_mb"],
        "reserved_after_mb": after["reserved_mb"],
        "incremental_allocated_mb": max(0.0, peak_alloc - before["allocated_mb"]),
        "incremental_reserved_mb": max(0.0, peak_reserved - before["reserved_mb"]),
    }


def load_meta() -> dict[str, dict]:
    out = {}
    with META.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def main() -> int:
    if not torch.cuda.is_available():
        die("cuda_unavailable")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gpu = torch.cuda.get_device_name(0)
    if gpu != protocol["required_gpu_name"]:
        die(f"gpu_mismatch:{gpu}")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    ids = list(freeze["pilot_val_ids"])[:N_SAMPLES]
    meta = load_meta()
    resolver = RawVideoResolver(ROOT)
    skel_root = ROOT / protocol["binds"]["b4_skel_root_rel"] / "skeleton_v3"
    labels_obj = json.loads((ROOT / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]

    device = torch.device("cuda:0")
    extractor = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / EXPECTED_SLOWFAST_CKPT_REL,
        expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
        device=device,
        repo_root=ROOT,
    ).eval()
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    e4 = build_ms_sta_v6_e4_r4(rgb_id="R1", skel_layout="S_A").to(device).eval()

    # warmup once
    sid0 = ids[0]
    row0 = meta[sid0]
    v0 = resolver.resolve(row0)
    sealed0 = skel_root / f"{sid0}.npz"
    frames0, fi0, fps0 = read_video_frames_native(str(v0))
    lm0, miss0, sfi0, sts0, sealed_d0 = sealed_landmarks(sealed0)
    _ = extract_rgb_token_v3_for_video(
        video_path=str(v0),
        extractor=extractor,
        landmarks_by_frame=lm0,
        missing_by_frame=miss0,
        skel_frame_index=sfi0,
        skel_timestamps=sts0,
        sample_id=sid0,
        enable_pose_wrist_prior=True,
        enable_ambiguity_aware_prior=True,
        ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
        repo_root=ROOT,
        frames_native=frames0,
        frame_idx=fi0,
        fps=fps0,
    )
    with torch.inference_mode():
        batch_e4 = e4_batch_from_frames(
            frames_rgb=frames0, fps=fps0, sealed=sealed_d0, rgb_id="R1", skel_layout="S_A", device=device
        )
        _ = e4(batch_e4)
    torch.cuda.synchronize()

    records = []
    for sid in ids:
        row = meta[sid]
        vpath = resolver.resolve(row)
        sealed_path = skel_root / f"{sid}.npz"
        if vpath is None or not sealed_path.is_file():
            die(f"missing_inputs:{sid}")
        label = int(g2i[str(row.get("text_gloss") or "").strip()])
        signer = str(row.get("signer_id") or "")
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        lm, miss, sfi, sts, sealed = sealed_landmarks(sealed_path)

        def b4_stage():
            extract = extract_rgb_token_v3_for_video(
                video_path=str(vpath),
                extractor=extractor,
                landmarks_by_frame=lm,
                missing_by_frame=miss,
                skel_frame_index=sfi,
                skel_timestamps=sts,
                sample_id=sid,
                enable_pose_wrist_prior=True,
                enable_ambiguity_aware_prior=True,
                ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
                repo_root=ROOT,
                frames_native=frames,
                frame_idx=frame_idx,
                fps=fps,
            )
            item = b4_item_from_extract(sid, extract, sealed, label, signer)
            batch = collate_v5([item])
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            torch.cuda.synchronize()
            with torch.inference_mode():
                out = b4(batch)
                if not torch.isfinite(out["logits"]).all() or tuple(out["logits"].shape) != (1, 400):
                    raise RuntimeError("b4_correctness_fail")

        def e4_stage():
            batch_e4 = e4_batch_from_frames(
                frames_rgb=frames, fps=fps, sealed=sealed, rgb_id="R1", skel_layout="S_A", device=device
            )
            with torch.inference_mode():
                out = e4(batch_e4)
                if not torch.isfinite(out["logits"]).all() or tuple(out["logits"].shape) != (1, 400):
                    raise RuntimeError("e4_correctness_fail")

        b4_m = measure_incremental(b4_stage)
        e4_m = measure_incremental(e4_stage)
        records.append({"sample_id": sid, "B4_V6_CTR": b4_m, "R1_S_A_d128": e4_m})
        print("verified", sid, "b4_inc", round(b4_m["incremental_allocated_mb"], 2), "r1_inc", round(e4_m["incremental_allocated_mb"], 2), flush=True)

    b4_inc = [r["B4_V6_CTR"]["incremental_allocated_mb"] for r in records]
    r1_inc = [r["R1_S_A_d128"]["incremental_allocated_mb"] for r in records]
    b4_med = median(b4_inc)
    r1_med = median(r1_inc)
    regress_frac = (r1_med - b4_med) / max(b4_med, 1e-9)
    gate_pass = regress_frac <= VRAM_REGRESS_MAX

    status = "PASS" if gate_pass else "FAIL"
    payload = {
        "schema": "ms_sta_v6_e4_r4b_vram_accounting_verify_v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "formula": "peak_during_stage_minus_allocated_immediately_before_stage",
        "records_allocated_and_reserved": True,
        "reset_peak_stats": True,
        "cuda_synchronize": True,
        "compare_mode": "incremental_vs_incremental_not_absolute",
        "n_samples": len(records),
        "sample_ids": ids,
        "protocol_sha256": sha256_file(PROTOCOL),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "b4_incremental_allocated_mb_median": b4_med,
        "r1_incremental_allocated_mb_median": r1_med,
        "vram_regress_frac": regress_frac,
        "vram_regress_max_frac": VRAM_REGRESS_MAX,
        "r1_incremental_vram_gate_pass": gate_pass,
        "records": records,
        "note": "Verify-only; does not replace canonical R4B raw timings.",
        "e4_ready": False,
    }
    write_create_only(VERIFY_OUT, payload)
    print("VRAM_VERIFY", status, "b4_med", round(b4_med, 2), "r1_med", round(r1_med, 2), "regress_frac", round(regress_frac, 4))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
