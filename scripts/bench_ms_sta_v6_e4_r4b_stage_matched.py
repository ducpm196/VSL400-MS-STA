# -*- coding: utf-8 -*-
"""E4-R4B stage-matched raw/sealed-source → logits feasibility audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    R7C_AMBIGUITY_ROI_LOCK,
)
from src.data_prep.e4_r4_rgb_pack import (  # noqa: E402
    pack_rgb_candidate,
    sample_frame_indices,
)
from src.data_prep.e4_r4_skel_compact import compact_skeleton_arrays  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v5 import RGB_REGIONS  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters as count_b4  # noqa: E402
from src.models.ms_sta_v6_e4_r4 import build_ms_sta_v6_e4_r4, count_parameters  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    FrozenSlowFastBlock4,
    extract_rgb_token_v3_for_video,
    read_video_frames_native,
)
from src.training.ms_sta_v5_data import collate_v5, load_json  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4"
PROTOCOL = OUT / "E4_R4B_STAGE_MATCHED_PROTOCOL.json"
MAX_RGB_LEN = 8
MAX_SKEL_LEN = 96


def die(msg: str) -> None:
    print(f"E4_R4B_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_create_only(path: Path, payload: dict | str) -> str:
    if path.is_file():
        die(f"refuse_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = payload.encode("utf-8") if isinstance(payload, str) else (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return float(s[len(s) // 2]) if s else float("nan")


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    return float(s[max(0, int(0.9 * (len(s) - 1)))]) if s else float("nan")


def mean(xs: list[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def bootstrap_speedup_ci(base: list[float], cand: list[float], *, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    speeds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mb, mc = float(np.median(b[idx])), float(np.median(c[idx]))
        speeds.append((mb - mc) / max(mb, 1e-9))
    lo, hi = np.quantile(speeds, [0.025, 0.975])
    return {"speedup_median": float(np.median(speeds)), "ci95_lo": float(lo), "ci95_hi": float(hi)}


def cuda_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def sealed_landmarks(sealed_path: Path) -> tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    with np.load(sealed_path, allow_pickle=False) as z:
        fi = np.asarray(z["original_frame_index"], dtype=np.int32)
        ts = np.asarray(z["timestamps_sec"], dtype=np.float64)
        lm = np.asarray(z["raw_landmarks"] if "raw_landmarks" in z.files else z["joints_raw"], dtype=np.float32)
        miss = np.asarray(z["missing_mask_prefill"], dtype=bool)
        if "effective_missing" in z.files:
            miss = miss | np.asarray(z["effective_missing"], dtype=bool)
        sealed = {k: z[k] for k in z.files}
    landmarks_by_frame = {int(fi[i]): lm[i] for i in range(len(fi))}
    missing_by_frame = {int(fi[i]): miss[i] for i in range(len(fi))}
    return landmarks_by_frame, missing_by_frame, fi, ts, sealed


def center_crop_len(n: int, max_n: int) -> slice:
    if n <= max_n:
        return slice(0, n)
    start = (n - max_n) // 2
    return slice(start, start + max_n)


def b4_item_from_extract(sid: str, extract: dict, sealed: dict, label: int, signer: str) -> dict:
    tokens = extract["tokens"]
    miss = extract["missing_region_mask"]
    L = int(tokens["global"].shape[0])
    sl = center_crop_len(L, MAX_RGB_LEN)
    T = int(sealed["coordinates"].shape[0])
    st = center_crop_len(T, MAX_SKEL_LEN)
    item: dict[str, Any] = {"sample_id": sid, "label": label, "signer": signer, "rgb_len": 0, "skel_len": 0}
    for r in RGB_REGIONS:
        key = r if r in tokens else ("face" if r == "face" else r)
        # REGIONS uses face; RGB_REGIONS same
        arr = tokens[r]
        item[f"token_{r}"] = torch.from_numpy(np.asarray(arr[sl], dtype=np.float32))
        item[f"missing_{r}"] = torch.from_numpy(np.asarray(miss[r][sl], dtype=bool))
    item["rgb_len"] = int(item["token_global"].shape[0])
    item["coordinates"] = torch.from_numpy(np.asarray(sealed["coordinates"][st], dtype=np.float32))
    item["velocity"] = torch.from_numpy(np.asarray(sealed["velocity"][st], dtype=np.float32))
    item["validity"] = torch.from_numpy(np.asarray(sealed["validity"][st], dtype=np.float32))
    item["skel_len"] = int(item["coordinates"].shape[0])
    from src.models.ms_sta_v5 import CHIRALITY_VOCAB, PART_VOCAB
    from src.data_prep.joint_layout_v3 import N_JOINTS

    if "joint_part" in sealed:
        parts = [PART_VOCAB.get(str(x), 0) for x in sealed["joint_part"].tolist()]
    else:
        parts = [0] * N_JOINTS
    if "joint_chirality" in sealed:
        chirs = [CHIRALITY_VOCAB.get(str(x), 0) for x in sealed["joint_chirality"].tolist()]
    else:
        chirs = [0] * N_JOINTS
    item["joint_part_ids"] = torch.tensor(parts, dtype=torch.long)
    item["joint_chirality_ids"] = torch.tensor(chirs, dtype=torch.long)
    return item


def e4_batch_from_frames(
    *,
    frames_rgb: np.ndarray,
    fps: float,
    sealed: dict,
    rgb_id: str,
    skel_layout: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fi_sealed = np.asarray(sealed["original_frame_index"], dtype=np.int32)
    ts_sealed = np.asarray(sealed["timestamps_sec"], dtype=np.float64)
    lm = np.asarray(sealed["raw_landmarks"] if "raw_landmarks" in sealed else sealed["joints_raw"], dtype=np.float32)[
        ..., :2
    ]
    miss = np.asarray(sealed["missing_mask_prefill"], dtype=bool)
    if "effective_missing" in sealed:
        miss = miss | np.asarray(sealed["effective_missing"], dtype=bool)
    n_native = int(frames_rgb.shape[0])
    fi = sample_frame_indices(n_native, 32, fi_sealed)
    fi_to_ts = {int(a): float(b) for a, b in zip(fi_sealed.tolist(), ts_sealed.tolist())}
    ts = np.asarray([fi_to_ts.get(int(x), float(x) / float(fps)) for x in fi.tolist()], dtype=np.float64)
    row_map = []
    for fidx in fi.tolist():
        matches = np.where(fi_sealed == int(fidx))[0]
        if matches.size:
            row_map.append(int(matches[0]))
        else:
            row_map.append(int(np.argmin(np.abs(fi_sealed.astype(np.int64) - int(fidx)))))
    row_map_a = np.asarray(row_map, dtype=np.int64)
    # Select frames by index from already-decoded array
    selected = frames_rgb[fi]
    rgb_arrays = pack_rgb_candidate(
        candidate_id=rgb_id,
        frames_rgb=selected,
        landmarks_xy=lm[row_map_a],
        missing_mask=miss[row_map_a],
        frame_indices=fi,
        timestamps_sec=ts,
        fps=fps,
    )
    compact = compact_skeleton_arrays(sealed, skel_layout)
    t_max = int(rgb_arrays["rgb_tensor"].shape[0])
    coords = np.zeros((t_max, compact["coordinates"].shape[1], 3), dtype=np.float32)
    vel = np.zeros_like(coords)
    val = np.zeros((t_max, compact["coordinates"].shape[1]), dtype=np.float32)
    spad = np.ones((t_max,), dtype=bool)
    n = min(int(compact["coordinates"].shape[0]), t_max)
    coords[:n] = compact["coordinates"][:n]
    vel[:n] = compact["velocity"][:n]
    val[:n] = compact["validity"][:n]
    spad[:n] = False
    return {
        "rgb_tensor": torch.from_numpy(np.asarray(rgb_arrays["rgb_tensor"], dtype=np.float32)).unsqueeze(0).to(device),
        "temporal_valid": torch.from_numpy(np.asarray(rgb_arrays["temporal_valid"], dtype=bool)).unsqueeze(0).to(device),
        "coordinates": torch.from_numpy(coords).unsqueeze(0).to(device),
        "velocity": torch.from_numpy(vel).unsqueeze(0).to(device),
        "validity": torch.from_numpy(val).unsqueeze(0).to(device),
        "skel_pad_mask": torch.from_numpy(spad).unsqueeze(0).to(device),
    }


def run_one(
    *,
    sid: str,
    video_path: Path,
    sealed_path: Path,
    extractor: FrozenSlowFastBlock4,
    b4_model,
    e4_models: dict,
    candidates: list,
    device: torch.device,
    label: int,
    signer: str,
    cache_mode: str,
) -> list[dict]:
    records = []
    t0 = time.perf_counter()
    frames, frame_idx, fps = read_video_frames_native(str(video_path))
    decode_ms = (time.perf_counter() - t0) * 1000.0

    landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)

    # --- B4 (incremental peak VRAM over exclusive stages: SlowFast extract → forward) ---
    torch.cuda.synchronize()
    b4_base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    rss0 = rss_mb()
    t_rgb0 = time.perf_counter()
    extract = extract_rgb_token_v3_for_video(
        video_path=str(video_path),
        extractor=extractor,
        landmarks_by_frame=landmarks_by_frame,
        missing_by_frame=missing_by_frame,
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
    b4_rgb_ms = (time.perf_counter() - t_rgb0) * 1000.0

    t_sk0 = time.perf_counter()
    item = b4_item_from_extract(sid, extract, sealed, label, signer)
    b4_skel_ms = (time.perf_counter() - t_sk0) * 1000.0

    t_h0 = time.perf_counter()
    batch = collate_v5([item])
    batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    torch.cuda.synchronize()
    b4_h2d_ms = (time.perf_counter() - t_h0) * 1000.0

    def _b4_fwd():
        with torch.inference_mode():
            out = b4_model(batch)
            if not torch.isfinite(out["logits"]).all() or tuple(out["logits"].shape) != (1, 400):
                raise RuntimeError("b4_correctness_fail")

    b4_fwd_ms = cuda_ms(_b4_fwd)
    b4_vram = max(0.0, (torch.cuda.max_memory_allocated() - b4_base_alloc) / (1024**2))
    b4_rss = rss_mb() - rss0
    b4_total = decode_ms + b4_rgb_ms + b4_skel_ms + b4_h2d_ms + b4_fwd_ms
    records.append(
        {
            "model_id": "B4_V6_CTR",
            "sample_id": sid,
            "cache_mode": cache_mode,
            "video_decode_ms": decode_ms,
            "rgb_representation_ms": b4_rgb_ms,
            "skeleton_representation_ms": b4_skel_ms,
            "h2d_ms": b4_h2d_ms,
            "model_forward_ms": b4_fwd_ms,
            "total_ms": b4_total,
            "peak_vram_mb": b4_vram,
            "rss_delta_mb": b4_rss,
            "shared_decode": True,
            "vram_scope": "incremental_b4_exclusive_extract_through_forward",
        }
    )

    # --- E4 candidates (incremental peak over exclusive compact→forward) ---
    for c in candidates:
        cid = c["candidate_id"]
        torch.cuda.synchronize()
        e4_base_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rss1 = rss_mb()

        t_sk = time.perf_counter()
        compact_skeleton_arrays(sealed, c["skel"])
        e4_skel_ms = (time.perf_counter() - t_sk) * 1000.0

        t_r = time.perf_counter()
        batch_e4 = e4_batch_from_frames(
            frames_rgb=frames,
            fps=fps,
            sealed=sealed,
            rgb_id=c["rgb"],
            skel_layout=c["skel"],
            device=device,
        )
        e4_rgb_only = (time.perf_counter() - t_r) * 1000.0

        def _e4_fwd(batch_e4=batch_e4, cid=cid):
            with torch.inference_mode():
                out = e4_models[cid](batch_e4)
                if not torch.isfinite(out["logits"]).all() or tuple(out["logits"].shape) != (1, 400):
                    raise RuntimeError(f"e4_correctness_fail:{cid}")

        e4_fwd_ms = cuda_ms(_e4_fwd)
        e4_vram = max(0.0, (torch.cuda.max_memory_allocated() - e4_base_alloc) / (1024**2))
        e4_rss = rss_mb() - rss1
        e4_total = decode_ms + e4_skel_ms + e4_rgb_only + e4_fwd_ms
        records.append(
            {
                "model_id": cid,
                "sample_id": sid,
                "cache_mode": cache_mode,
                "video_decode_ms": decode_ms,
                "rgb_representation_ms": e4_rgb_only,
                "skeleton_representation_ms": e4_skel_ms,
                "h2d_ms": 0.0,
                "h2d_note": "folded_into_rgb_representation_ms",
                "model_forward_ms": e4_fwd_ms,
                "total_ms": e4_total,
                "peak_vram_mb": e4_vram,
                "rss_delta_mb": e4_rss,
                "shared_decode": True,
                "vram_scope": "incremental_e4_exclusive_compact_through_forward",
            }
        )
    return records


def main() -> int:
    if not PROTOCOL.is_file():
        die("missing_protocol")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_MEASUREMENT":
        die("protocol_not_frozen")
    if not torch.cuda.is_available():
        die("cuda_unavailable")
    gpu = torch.cuda.get_device_name(0)
    if gpu != protocol["required_gpu_name"]:
        die(f"gpu_mismatch:{gpu}")

    freeze = json.loads((OUT / "E4_R4_PILOT_SAMPLE_FREEZE.json").read_text(encoding="utf-8"))
    if sha256_file(OUT / "E4_R4_PILOT_SAMPLE_FREEZE.json") != protocol["binds"]["pilot_freeze_sha256"]:
        die("pilot_freeze_sha_mismatch")
    if sha256_file(OUT / "E4_R4_PILOT_REPRESENTATION_AUDIT.json") != protocol["binds"]["representation_audit_sha256"]:
        die("rep_audit_sha_mismatch")
    if sha256_file(ROOT / EXPECTED_SLOWFAST_CKPT_REL) != protocol["binds"]["slowfast_ckpt_sha256"]:
        die("slowfast_sha_mismatch")

    ids = list(freeze["pilot_train_ids"]) + list(freeze["pilot_val_ids"])
    split = load_json(ROOT / "data/processed/vsl400_retrain_v4_split_manifest.json")
    test_ids = set(map(str, split["test_ids"]))
    if set(ids) & test_ids:
        die("test_leak")

    meta_rows: dict[str, dict[str, str]] = {}
    with (ROOT / "metadata.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("dataset") or "") != "VSL400":
                continue
            sid = str(row.get("video_id") or "").strip()
            if sid:
                meta_rows[sid] = dict(row)
    labels_obj = load_json(ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    resolver = RawVideoResolver(ROOT)
    sealed_root = ROOT / protocol["binds"]["b4_skel_root_rel"] / "skeleton_v3"

    device = torch.device("cuda:0")
    print("Loading SlowFast extractor...", flush=True)
    extractor = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / EXPECTED_SLOWFAST_CKPT_REL,
        expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
        device=device,
        repo_root=ROOT,
    )
    torch.manual_seed(42)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    candidates = protocol["candidates"]
    e4_models = {
        c["candidate_id"]: build_ms_sta_v6_e4_r4(rgb_id=c["rgb"], skel_layout=c["skel"]).to(device).eval()
        for c in candidates
    }

    warmup = int(protocol["benchmark"]["warmup_full_pipeline"])
    hot_trials = int(protocol["benchmark"]["hot_trials"])
    cold_trials = int(protocol["benchmark"]["cold_trials"])
    rng = random.Random(42)

    raw_path = OUT / "E4_R4B_RAW_RESULTS.jsonl"
    if raw_path.is_file():
        die("raw_exists")
    raw_f = raw_path.open("w", encoding="utf-8")

    def resolve_sid(sid: str) -> tuple[Path, Path, int, str]:
        row = meta_rows.get(sid)
        if row is None:
            die(f"missing_metadata:{sid}")
        vpath = resolver.resolve(row)
        if vpath is None or not Path(vpath).is_file():
            die(f"missing_video:{sid}")
        sealed = sealed_root / f"{sid}.npz"
        if not sealed.is_file():
            die(f"missing_sealed:{sid}")
        gloss = str(row.get("text_gloss") or "").strip()
        if gloss not in g2i:
            die(f"missing_label:{sid}")
        return Path(vpath), sealed, int(g2i[gloss]), str(row.get("signer_id") or "")

    print("Warmup...", flush=True)
    for w in range(warmup):
        sid = ids[w % len(ids)]
        vpath, sealed, label, signer = resolve_sid(sid)
        recs = run_one(
            sid=sid,
            video_path=vpath,
            sealed_path=sealed,
            extractor=extractor,
            b4_model=b4,
            e4_models=e4_models,
            candidates=candidates,
            device=device,
            label=label,
            signer=signer,
            cache_mode="warmup",
        )
        for r in recs:
            r["trial"] = w
            raw_f.write(json.dumps(r) + "\n")
        print(f"  warmup {w+1}/{warmup}", flush=True)

    print("Cold trials...", flush=True)
    cold_ids = ids[:cold_trials]
    for t, sid in enumerate(cold_ids):
        # Drop file cache influence as much as practical: re-read path
        vpath, sealed, label, signer = resolve_sid(sid)
        recs = run_one(
            sid=sid,
            video_path=vpath,
            sealed_path=sealed,
            extractor=extractor,
            b4_model=b4,
            e4_models=e4_models,
            candidates=candidates,
            device=device,
            label=label,
            signer=signer,
            cache_mode="cold",
        )
        for r in recs:
            r["trial"] = t
            raw_f.write(json.dumps(r) + "\n")
        print(f"  cold {t+1}/{cold_trials} {sid}", flush=True)

    print("Hot trials...", flush=True)
    for t in range(hot_trials):
        sid = ids[t % len(ids)]
        vpath, sealed, label, signer = resolve_sid(sid)
        # Interleave by randomizing candidate internal order already in run_one fixed; shuffle sid order via rng
        if rng.random() < 0:
            pass
        recs = run_one(
            sid=sid,
            video_path=vpath,
            sealed_path=sealed,
            extractor=extractor,
            b4_model=b4,
            e4_models=e4_models,
            candidates=candidates,
            device=device,
            label=label,
            signer=signer,
            cache_mode="hot",
        )
        # Randomize record write order only
        rng.shuffle(recs)
        for r in recs:
            r["trial"] = t
            raw_f.write(json.dumps(r) + "\n")
        if (t + 1) % 4 == 0:
            print(f"  hot {t+1}/{hot_trials}", flush=True)
    raw_f.close()

    # Aggregate hot totals
    by_model: dict[str, list[dict]] = {}
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("cache_mode") != "hot":
                continue
            by_model.setdefault(rec["model_id"], []).append(rec)

    b4_totals = [r["total_ms"] for r in by_model["B4_V6_CTR"]]
    b4_p90 = p90(b4_totals)
    b4_med = median(b4_totals)
    b4_vram = median([r["peak_vram_mb"] for r in by_model["B4_V6_CTR"]])
    b4_rss = median([r["rss_delta_mb"] for r in by_model["B4_V6_CTR"]])

    gate = protocol["gate"]
    rows = []
    promoted = []
    for c in candidates:
        cid = c["candidate_id"]
        recs = by_model[cid]
        totals = [r["total_ms"] for r in recs]
        med = median(totals)
        speedup = (b4_med - med) / max(b4_med, 1e-9)
        ci = bootstrap_speedup_ci(b4_totals, totals, n_boot=gate["n_bootstrap"], seed=gate["bootstrap_seed"])
        reasons = []
        if speedup < float(gate["primary_total_median_speedup_min"]):
            reasons.append("total_speedup_lt_8pct")
        if ci["ci95_lo"] <= 0:
            reasons.append("ci_lo_le_0")
        if p90(totals) > b4_p90:
            reasons.append("p90_regress")
        vram = median([r["peak_vram_mb"] for r in recs])
        rss = median([r["rss_delta_mb"] for r in recs])
        if (vram - b4_vram) / max(b4_vram, 1e-9) > float(gate["vram_regress_max_frac"]):
            reasons.append("vram_regress")
        if (rss - b4_rss) / max(abs(b4_rss), 1e-9) > float(gate["rss_regress_max_frac"]) and rss > b4_rss:
            reasons.append("rss_regress")
        ok = not reasons
        row = {
            "candidate_id": cid,
            "hot_n": len(totals),
            "total_median_ms": med,
            "total_p90_ms": p90(totals),
            "total_mean_ms": mean(totals),
            "speedup_vs_b4": speedup,
            "bootstrap_ci": ci,
            "stage_medians_ms": {
                "video_decode": median([r["video_decode_ms"] for r in recs]),
                "rgb_representation": median([r["rgb_representation_ms"] for r in recs]),
                "skeleton_representation": median([r["skeleton_representation_ms"] for r in recs]),
                "model_forward": median([r["model_forward_ms"] for r in recs]),
            },
            "peak_vram_mb_median": vram,
            "rss_delta_mb_median": rss,
            "gate_pass": ok,
            "reject_reasons": reasons,
            "param_count": count_parameters(e4_models[cid]),
        }
        rows.append(row)
        if ok:
            promoted.append(row)

    promoted = sorted(promoted, key=lambda r: (-r["speedup_vs_b4"], r["total_p90_ms"], r["candidate_id"]))[
        : int(gate["promote_max"])
    ]
    verdict = "E4_R4B_STAGE_MATCHED_PASS" if promoted else "E4_R4_FINAL_NO_GO"
    status = verdict

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "schema": "ms_sta_v6_e4_r4b_stage_breakdown_report_v1",
        "status": status,
        "verdict": verdict,
        "created_at": now,
        "protocol_sha256": sha256_file(PROTOCOL),
        "raw_results_sha256": sha256_file(raw_path),
        "r4_precise_verdict": "CACHE_BACKED_MODEL_FORWARD_NO_GO",
        "environment": {
            "gpu": gpu,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "baseline_b4_hot": {
            "total_median_ms": b4_med,
            "total_p90_ms": b4_p90,
            "total_mean_ms": mean(b4_totals),
            "stage_medians_ms": {
                "video_decode": median([r["video_decode_ms"] for r in by_model["B4_V6_CTR"]]),
                "rgb_representation": median([r["rgb_representation_ms"] for r in by_model["B4_V6_CTR"]]),
                "skeleton_representation": median([r["skeleton_representation_ms"] for r in by_model["B4_V6_CTR"]]),
                "h2d": median([r["h2d_ms"] for r in by_model["B4_V6_CTR"]]),
                "model_forward": median([r["model_forward_ms"] for r in by_model["B4_V6_CTR"]]),
            },
            "peak_vram_mb_median": b4_vram,
            "param_count": count_b4(b4),
            "n": len(b4_totals),
        },
        "candidate_rows": rows,
        "promoted": [r["candidate_id"] for r in promoted],
        "cold_note": "cold trials retained in raw jsonl; gate uses hot totals only",
        "e4_ready": False,
        "forbid_full_corpus": True,
    }
    report_sha = write_create_only(OUT / "E4_R4B_STAGE_BREAKDOWN_REPORT.json", report)

    closeout = {
        "schema": "ms_sta_v6_e4_r4b_closeout_v1",
        "status": status,
        "verdict": verdict,
        "promoted_candidate_ids": [r["candidate_id"] for r in promoted],
        "n_promoted": len(promoted),
        "next_phase_allowed": (
            "await_full_corpus_authorization_single_candidate"
            if promoted
            else "stop_final_no_go_negative_study_appendix_b4_remains_deployed"
        ),
        "e4_ready": False,
        "report_sha256": report_sha,
        "created_at": now,
    }
    close_sha = write_create_only(OUT / "E4_R4B_CLOSEOUT.json", closeout)

    evidence = {
        "schema": "ms_sta_v6_e4_r4b_evidence_index_v1",
        "created_at": now,
        "verdict": verdict,
        "artifacts": {
            "protocol": {"rel": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(PROTOCOL)},
            "raw": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4B_RAW_RESULTS.jsonl", "sha256": sha256_file(raw_path)},
            "report": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4B_STAGE_BREAKDOWN_REPORT.json", "sha256": report_sha},
            "closeout": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4B_CLOSEOUT.json", "sha256": close_sha},
            "r4_clarification": {
                "rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4_FEASIBILITY_VERDICT_CLARIFICATION.json",
                "sha256": sha256_file(OUT / "E4_R4_FEASIBILITY_VERDICT_CLARIFICATION.json"),
            },
        },
    }
    write_create_only(OUT / "EVIDENCE_INDEX_R4B.json", evidence)

    print("VERDICT", verdict)
    print("B4_hot_total_median_ms", round(b4_med, 2))
    for r in rows:
        print(r["candidate_id"], "total_med", round(r["total_median_ms"], 2), "speedup", round(r["speedup_vs_b4"], 4), r["reject_reasons"])
    print("PROMOTED", [r["candidate_id"] for r in promoted])
    return 0 if promoted else 3


if __name__ == "__main__":
    raise SystemExit(main())
