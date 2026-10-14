# -*- coding: utf-8 -*-
"""E4_B4SC_V1 E2 stage-matched efficiency: remasure E0/E1a/E2 (sequential decode)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import (  # noqa: E402
    b4_item_from_extract,
    sealed_landmarks,
)
from scripts.e4_b4sc_e1_factorial import cuda_event_ms  # noqa: E402
from scripts.e4_b4sc_e1_independent_verify import bootstrap_speedup, median, p90  # noqa: E402
from scripts.e4_b4sc_train_e2_screen import (  # noqa: E402
    AmpRoiFp32SlowFastTrainable,
    build_b4_batch_from_projected,
    head_logits_eval,
    load_meta,
    masked_mean_tokens_diff,
    project_tokens,
    roi_align_token_diff,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    INPUT_SIZE,
    REGIONS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    FrozenSlowFastBlock4,
    _boxes_for_native_frame,
    _local_frame_for_feat_t,
    _resize_letterbox,
    extract_rgb_token_v3_for_video,
    nearest_skel_landmark,
    read_video_frames_native,
    window_starts,
    window_tensors,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast  # noqa: E402
from src.optimization.e4_b4sc.narrow_slowfast import (  # noqa: E402
    E2_WIDTHS,
    NarrowSlowFastBlock4,
    RoiToHeadProjection,
)
from src.training.ms_sta_v5_data import collate_v5  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POL_TRAIN = OUT / "E4_B4SC_V1_E2E3_TRAINING_POLICY.json"
POL_E4 = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"
E1_LOCK = OUT / "E4_B4SC_V1_E1_SCHEDULE_LOCK.json"
CHAN_LOCK = OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json"
CKPT = OUT / "checkpoints/e2_screen_seed42/best.pt"
RAW_PATH = OUT / "E4_B4SC_V1_E2_EFFICIENCY_RAW.jsonl"
REPORT_PATH = OUT / "E4_B4SC_V1_E2_EFFICIENCY_REPORT.json"
ARMS = ("E0", "E1a", "E2")


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_E2_EFF_FAIL:{msg}")


def ensure_train_cuda() -> None:
    venv_py = ROOT / "_venvs/train_cuda/Scripts/python.exe"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        os.execv(str(venv_py), [str(venv_py), *sys.argv])


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(str(obj), encoding="utf-8")


def counterbalanced_orders(sample_ids: list[str], arms: tuple[str, ...], seed: int) -> dict[str, list[str]]:
    """Deterministic latin-like rotation of arm order per sample (no metric peeking)."""
    rng = np.random.default_rng(seed)
    base = list(arms)
    out: dict[str, list[str]] = {}
    for i, sid in enumerate(sample_ids):
        rot = i % len(base)
        order = base[rot:] + base[:rot]
        if rng.integers(0, 2) == 1:
            order = list(reversed(order))
        out[sid] = order
    return out


def sample_medians(rows: list[dict], key: str = "e2e_ms") -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("measured") and r.get("success"):
            by[r["sample_id"]].append(float(r[key]))
    return {sid: median(vs) for sid, vs in by.items()}


def main() -> int:
    ensure_train_cuda()
    ap = argparse.ArgumentParser(description="E2 stage-matched efficiency remasure")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmups", type=int, default=1)
    args = ap.parse_args()

    if not CKPT.is_file():
        die("missing_e2_ckpt")
    tpol = json.loads(POL_TRAIN.read_text(encoding="utf-8"))
    pol = json.loads(POL_E4.read_text(encoding="utf-8"))
    e1_lock = json.loads(E1_LOCK.read_text(encoding="utf-8"))
    chan = json.loads(CHAN_LOCK.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("cuda_required")
    gpu = torch.cuda.get_device_name(0)
    if gpu != pol["scientific_environment_lock"]["required_gpu_name"]:
        die(f"gpu_mismatch:{gpu}")

    sample_ids = list(e1_lock["bindings"]["sample_ids"])
    if len(sample_ids) != 24:
        die(f"sample_n:{len(sample_ids)}")
    n_rounds = int(e1_lock["measurement_schedule"]["measured_rounds_per_sample_per_arm"])
    orders = counterbalanced_orders(sample_ids, ARMS, seed=42)

    meta = load_meta()
    labels_obj = json.loads((ROOT / tpol["data"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    resolver = RawVideoResolver(ROOT)
    skel_rel = tpol["data"]["skel_rel"]

    def resolve(sid: str):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        vpath = resolver.resolve(row)
        sealed = ROOT / skel_rel / "skeleton_v3" / f"{sid}.npz"
        if not sealed.is_file():
            sealed = ROOT / skel_rel / f"{sid}.npz"
        if vpath is None or not sealed.is_file() or gloss not in g2i:
            die(f"resolve_fail:{sid}")
        return Path(vpath), sealed, int(g2i[gloss]), str(row.get("signer_id") or "")

    teacher_raw = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    teacher_fp32 = teacher_raw
    teacher_amp = AmpRoiFp32SlowFast(teacher_raw, channels_last=False)

    plan_e2 = chan["E2"]
    student_raw = NarrowSlowFastBlock4(teacher_raw, plan_e2, device=device, requires_grad_slow=False)
    student_amp = AmpRoiFp32SlowFastTrainable(student_raw)
    roi_proj = RoiToHeadProjection(E2_WIDTHS.token_dim, out_dim=2304).to(device)
    payload = torch.load(CKPT, map_location=device, weights_only=False)
    student_raw.load_state_dict(payload["student"])
    roi_proj.load_state_dict(payload["roi_to_head_projection"])
    # StageKDAdapters are TRAIN_ONLY — absent from deploy timing graph
    student_raw.eval()
    roi_proj.eval()

    b4_head = build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt = torch.load(ROOT / tpol["checkpoints"]["b4_teacher_rel"], map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()

    if RAW_PATH.exists() or REPORT_PATH.exists():
        die("efficiency_artifacts_exist")
    create_only(RAW_PATH, None)

    def student_extract_deploy(
        *,
        frames_native: np.ndarray,
        frame_idx: np.ndarray,
        fps: float,
        landmarks_by_frame: dict,
        missing_by_frame: dict,
        skel_frame_index: np.ndarray,
        skel_timestamps: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Deploy-only student extract (no teacher / no KD adapters)."""
        resized: list[np.ndarray] = []
        metas: list[dict[str, float]] = []
        for fr in frames_native:
            canvas, meta = _resize_letterbox(fr, INPUT_SIZE)
            resized.append(canvas)
            metas.append(meta)
        frames_u8 = np.stack(resized, axis=0)
        t_native = frames_u8.shape[0]
        starts = window_starts(t_native, window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE)
        region_seq: dict[str, list[torch.Tensor]] = {r: [] for r in REGIONS}
        for start in starts:
            slow_t, fast_t, _valid_mask, n_valid = window_tensors(frames_u8, start, WINDOW_SIZE)
            maps, _ = student_amp.forward_maps_with_stages(slow_t.to(device), fast_t.to(device))
            end = min(t_native, start + WINDOW_SIZE)
            center = start + (end - start) // 2
            align = nearest_skel_landmark(
                center_frame=int(frame_idx[center]),
                fps=fps,
                skel_frame_index=skel_frame_index,
                skel_timestamps=skel_timestamps,
            )
            lm_frame = int(align["landmark_frame"])
            if landmarks_by_frame.get(lm_frame) is None:
                die(f"landmark_missing:lm={lm_frame}")
            t_slow = int(maps.slow.shape[2])
            t_fast = int(maps.fast.shape[2])
            per_slow: dict[str, list[torch.Tensor]] = {r: [] for r in REGIONS}
            per_fast: dict[str, list[torch.Tensor]] = {r: [] for r in REGIONS}
            miss_slow: dict[str, list[bool]] = {r: [] for r in REGIONS}
            miss_fast: dict[str, list[bool]] = {r: [] for r in REGIONS}
            for t_idx in range(t_slow):
                local = _local_frame_for_feat_t(
                    pathway="slow", t_idx=t_idx, t_feat=t_slow, n_valid=n_valid, window_size=WINDOW_SIZE
                )
                native = int(frame_idx[min(start + local, t_native - 1)])
                boxes_t, _, _, _, _ = _boxes_for_native_frame(
                    native_frame=native,
                    canvas_meta=metas[min(start + local, t_native - 1)],
                    landmarks_by_frame=landmarks_by_frame,
                    missing_by_frame=missing_by_frame,
                    skel_frame_index=skel_frame_index,
                    skel_timestamps=skel_timestamps,
                    fps=fps,
                    enable_pose_wrist_prior=True,
                    enable_ambiguity_aware_prior=True,
                    ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
                )
                feat_t = maps.slow[:, :, t_idx, :, :]
                for region in REGIONS:
                    tok, mflag = roi_align_token_diff(feat_t, boxes_t[region])
                    per_slow[region].append(tok)
                    miss_slow[region].append(bool(mflag))
            for t_idx in range(t_fast):
                local = _local_frame_for_feat_t(
                    pathway="fast", t_idx=t_idx, t_feat=t_fast, n_valid=n_valid, window_size=WINDOW_SIZE
                )
                native = int(frame_idx[min(start + local, t_native - 1)])
                boxes_t, _, _, _, _ = _boxes_for_native_frame(
                    native_frame=native,
                    canvas_meta=metas[min(start + local, t_native - 1)],
                    landmarks_by_frame=landmarks_by_frame,
                    missing_by_frame=missing_by_frame,
                    skel_frame_index=skel_frame_index,
                    skel_timestamps=skel_timestamps,
                    fps=fps,
                    enable_pose_wrist_prior=True,
                    enable_ambiguity_aware_prior=True,
                    ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
                )
                feat_t = maps.fast[:, :, t_idx, :, :]
                for region in REGIONS:
                    tok, mflag = roi_align_token_diff(feat_t, boxes_t[region])
                    per_fast[region].append(tok)
                    miss_fast[region].append(bool(mflag))
            for region in REGIONS:
                tok_s, _ = masked_mean_tokens_diff(per_slow[region], miss_slow[region])
                tok_f, _ = masked_mean_tokens_diff(per_fast[region], miss_fast[region])
                region_seq[region].append(torch.cat([tok_s, tok_f], dim=0))
        return {r: torch.stack(v, dim=0) for r, v in region_seq.items()}

    def time_teacher(sid: str, arm: str, video_path: Path, sealed_path: Path, label: int, signer: str) -> dict:
        extractor = teacher_amp if arm == "E1a" else teacher_fp32
        t0 = time.perf_counter()
        frames, frame_idx, fps = read_video_frames_native(str(video_path))
        decode_ms = (time.perf_counter() - t0) * 1000.0
        landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        base_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
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
            pathway_window_batch_size=1,
        )
        torch.cuda.synchronize()
        slowfast_roi_ms = (time.perf_counter() - t_rgb0) * 1000.0
        item = b4_item_from_extract(sid, extract, sealed, label, signer)
        batch = collate_v5([item])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        def _fwd():
            with torch.inference_mode():
                _ = b4_head(batch)

        head_ms = cuda_event_ms(_fwd)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "decode_ms": decode_ms,
            "preprocess_ms": 0.0,
            "slowfast_roi_ms": slowfast_roi_ms,
            "head_ms": head_ms,
            "e2e_ms": e2e_ms,
            "peak_allocated_vram_mb": max(0.0, (torch.cuda.max_memory_allocated() - base_alloc) / (1024**2)),
            "peak_reserved_vram_mb": max(0.0, (torch.cuda.max_memory_reserved() - base_reserved) / (1024**2)),
            "rss_delta_mb": 0.0,
            "success": True,
        }

    def time_e2(sid: str, video_path: Path, sealed_path: Path, label: int, signer: str) -> dict:
        t0 = time.perf_counter()
        frames, frame_idx, fps = read_video_frames_native(str(video_path))
        decode_ms = (time.perf_counter() - t0) * 1000.0
        landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)
        torch.cuda.synchronize()
        base_alloc = torch.cuda.memory_allocated()
        base_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_rgb0 = time.perf_counter()
        with torch.no_grad():
            tokens = student_extract_deploy(
                frames_native=frames,
                frame_idx=frame_idx,
                fps=fps,
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                skel_frame_index=sfi,
                skel_timestamps=sts,
            )
            proj = project_tokens(roi_proj, tokens)
            batch = build_b4_batch_from_projected(
                sid=sid, label=label, signer=signer, proj_tokens=proj, sealed=sealed, device=device
            )
        torch.cuda.synchronize()
        slowfast_roi_ms = (time.perf_counter() - t_rgb0) * 1000.0

        def _fwd():
            with torch.inference_mode():
                _ = head_logits_eval(b4_head, batch)

        head_ms = cuda_event_ms(_fwd)
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "decode_ms": decode_ms,
            "preprocess_ms": 0.0,
            "slowfast_roi_ms": slowfast_roi_ms,
            "head_ms": head_ms,
            "e2e_ms": e2e_ms,
            "peak_allocated_vram_mb": max(0.0, (torch.cuda.max_memory_allocated() - base_alloc) / (1024**2)),
            "peak_reserved_vram_mb": max(0.0, (torch.cuda.max_memory_reserved() - base_reserved) / (1024**2)),
            "rss_delta_mb": 0.0,
            "success": True,
        }

    # warmups
    warm_sid = sample_ids[0]
    wv, ws, wl, wsg = resolve(warm_sid)
    for _ in range(max(1, args.warmups)):
        time_teacher(warm_sid, "E0", wv, ws, wl, wsg)
        time_teacher(warm_sid, "E1a", wv, ws, wl, wsg)
        time_e2(warm_sid, wv, ws, wl, wsg)

    rows: list[dict] = []
    with RAW_PATH.open("a", encoding="utf-8") as f:
        for rnd in range(n_rounds):
            for sid in sample_ids:
                vpath, sealed, label, signer = resolve(sid)
                for arm in orders[sid]:
                    try:
                        if arm == "E2":
                            m = time_e2(sid, vpath, sealed, label, signer)
                        else:
                            m = time_teacher(sid, arm, vpath, sealed, label, signer)
                        row = {
                            "schema": "ms_sta_v6_e4_b4sc_v1_e2_efficiency_raw_v1",
                            "sample_id": sid,
                            "arm": arm,
                            "round": rnd,
                            "measured": True,
                            "success": bool(m["success"]),
                            "arm_order": orders[sid],
                            "pathway_window_batch_size": 1,
                            "decode": "original_sequential",
                            **{k: m[k] for k in (
                                "decode_ms",
                                "preprocess_ms",
                                "slowfast_roi_ms",
                                "head_ms",
                                "e2e_ms",
                                "peak_allocated_vram_mb",
                                "peak_reserved_vram_mb",
                                "rss_delta_mb",
                            )},
                            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        }
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "sample_id": sid,
                            "arm": arm,
                            "round": rnd,
                            "measured": True,
                            "success": False,
                            "error": str(exc),
                        }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    rows.append(row)
                    print(f"E2_EFF r={rnd} {sid} {arm} e2e={row.get('e2e_ms', 'ERR')}", flush=True)

    measured = [r for r in rows if r.get("measured")]
    if any(not r.get("success") for r in measured):
        die("measured_failure")

    sample_e2e = {a: sample_medians([r for r in measured if r["arm"] == a], "e2e_ms") for a in ARMS}
    campaign_median = {a: median([sample_e2e[a][sid] for sid in sample_ids]) for a in ARMS}
    campaign_p90 = {
        a: p90([float(r["e2e_ms"]) for r in measured if r["arm"] == a and r.get("success")]) for a in ARMS
    }
    vram = {
        a: median([float(r["peak_allocated_vram_mb"]) for r in measured if r["arm"] == a and r.get("success")])
        for a in ARMS
    }

    base_list = [sample_e2e["E0"][sid] for sid in sample_ids]
    paired = {}
    for a in ("E1a", "E2"):
        cand = [sample_e2e[a][sid] for sid in sample_ids]
        paired[a] = bootstrap_speedup(
            base_list,
            cand,
            n_boot=int(tpol["efficiency_gate"]["n_boot"]),
            seed=int(tpol["efficiency_gate"]["bootstrap_seed"]),
        )

    e2_vs_e1a_ms = campaign_median["E1a"] - campaign_median["E2"]
    e2_vs_e0_ms = campaign_median["E0"] - campaign_median["E2"]
    gate = tpol["efficiency_gate"]
    e2_speed = paired["E2"]["speedup_factor"]
    e2_ci_lo = paired["E2"]["ci95_lo"]
    p90_ok = campaign_p90["E2"] <= campaign_p90["E0"] * 1.05  # no material p90 regression vs E0
    vram_ok = vram["E2"] <= vram["E0"] + 1e-6
    eff_pass = (
        e2_speed >= gate["e2e_median_speedup_vs_E0_min"]
        and e2_ci_lo >= gate["e2e_speedup_ci95_lower_min"]
        and p90_ok
        and vram_ok
    )

    report = {
        "schema": "ms_sta_v6_e4_b4sc_v1_e2_efficiency_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_ids": sample_ids,
        "n_rounds": n_rounds,
        "arm_order_by_sample": orders,
        "campaign_median_e2e_ms": campaign_median,
        "campaign_p90_e2e_ms": campaign_p90,
        "peak_allocated_vram_mb_median": vram,
        "paired_vs_E0": paired,
        "structured_compression_saving_vs_E1a_ms": e2_vs_e1a_ms,
        "total_saving_vs_E0_ms": e2_vs_e0_ms,
        "efficiency_gate_pass": eff_pass,
        "engineering_1_18x_met": e2_speed >= gate["engineering_target_speedup_vs_E0"],
        "gate": gate,
        "e4_ready": False,
        "raw_sha256": sha256_file(RAW_PATH),
        "checkpoint_sha256": sha256_file(CKPT),
        "note": "Prior E1 headline timings not reused; stage-matched remasure with sequential decode only.",
    }
    create_only(REPORT_PATH, report)
    print(
        "E4_B4SC_E2_EFF_OK" if eff_pass else "E4_B4SC_E2_EFF_NO_GO",
        f"medians={campaign_median}",
        f"speedup_E2={e2_speed:.4f}",
        f"ci_lo={e2_ci_lo:.4f}",
        f"pass={eff_pass}",
        flush=True,
    )
    return 0 if eff_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
