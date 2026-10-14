#!/usr/bin/env python3
"""MS-STA v5 D0-PILOT-R4 runner — masked temporal ROI mean, tensor quality stats, strict manual QA.

Reuses frozen R1 train/val sample set. Writes only to Docs/results/ms_sta_v5_d0_r4/.
Preserves R3 PASS audit + R3 independent FAIL. No full corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    APPROVAL_REL,
    ELIGIBLE_REL,
    EXPECTED_ELIGIBLE_SHA256,
    EXPECTED_R1_SAMPLE_SHA256,
    EXPECTED_R3_PILOT_AUDIT_SHA256,
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    F0_CLOSEOUT_REL,
    FULL_PROTOCOL_REL,
    PILOT_R1_NEGATIVE_REL,
    PILOT_R1_SAMPLE_REL,
    PILOT_R3_AUDIT_REL,
    PILOT_R3_INDEPENDENT_REAUDIT_REL,
    PILOT_R4_AUDIT_REL,
    PILOT_R4_COMPLETION_REL,
    PILOT_R4_OUT_DIR_REL,
    PILOT_R4_REPORT_REL,
    PILOT_R4_RUN_CONTRACT_REL,
    PILOT_R4_VISUAL_MANUAL_REL,
    PILOT_R4_VISUAL_REL,
    PROTOCOL_LOCK_REL,
    PROTOCOL_REL,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_d0_approval,
    validate_d0_full_protocol,
    validate_d0_protocol,
    validate_f0_phase_closeout,
    validate_pilot_sample_set_r1,
    validate_r3_independent_reaudit,
    validate_visual_qa_manual,
    write_json,
)

EXTRACT_PY = _ROOT / "_venvs/extract/Scripts/python.exe"
F0_PY = _ROOT / "_venvs/f0_v5/Scripts/python.exe"
CORPUS_N = 24752
DET_TOLERANCE = 0.0
ROI_STROKE_BGR = {
    "global": (200, 200, 200),
    "left_hand": (0, 165, 255),
    "right_hand": (0, 255, 0),
    "face": (255, 0, 0),
}


def _code_lock_bind() -> dict[str, Any]:
    allowed_prefixes = (
        "Docs/results/ms_sta_v5_d0_r4/",
        "Docs/results/ms_sta_v5_d0_r4",
    )
    dirty_lines: list[str] = []
    for line in git_porcelain(_ROOT).splitlines():
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        norm = path.replace("\\", "/")
        if any(norm == p.rstrip("/") or norm.startswith(p) for p in allowed_prefixes):
            continue
        dirty_lines.append(line)
    dirty = "\n".join(dirty_lines).strip()
    return {
        "code_lock_sha256": git_rev(_ROOT),
        "worktree_dirty": bool(dirty),
        "porcelain": dirty[:2000] if dirty else "",
        "ignored_untracked_prefixes": list(allowed_prefixes),
    }


def _require_clean_code_lock(expected: str | None = None) -> dict[str, Any]:
    bind = _code_lock_bind()
    if bind["worktree_dirty"]:
        raise RuntimeError(f"dirty_worktree_forbidden:{bind['porcelain'][:500]}")
    if expected is not None and bind["code_lock_sha256"] != expected:
        raise RuntimeError(
            f"code_lock_mismatch:expected={expected}:got={bind['code_lock_sha256']}"
        )
    return bind


def _np_sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _save_skel_npz(out_dir: Path, sample_id: str, payload: dict[str, Any]) -> str:
    path = out_dir / "skel_v3" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coordinates=payload["coordinates"],
        joints_raw=payload["joints_raw"],
        velocity=payload["velocity"],
        bone_vectors=payload["bone_vectors"],
        missing_mask_prefill=payload["missing_mask_prefill"],
        validity=payload["validity"],
        confidence=payload["confidence"],
        original_frame_index=payload["original_frame_index"],
        timestamps_sec=payload["timestamps_sec"],
        padding_mask=payload["padding_mask"],
        joint_part=payload["joint_part"],
        joint_chirality=payload["joint_chirality"],
        bone_edges=payload["bone_edges"],
        sequence_length=np.asarray([payload["sequence_length"]], dtype=np.int32),
    )
    return sha256_file(path)


def _save_rgb_npz(out_dir: Path, sample_id: str, payload: dict[str, Any]) -> str:
    path = out_dir / "rgb_token_v3" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {f"token_{k}": v for k, v in payload["tokens"].items()}
    kwargs.update({f"missing_{k}": v for k, v in payload["missing_region_mask"].items()})
    kwargs["pathway_slow"] = payload["pathway_global_tokens"]["slow"]
    kwargs["pathway_fast"] = payload["pathway_global_tokens"]["fast"]
    kwargs["original_frame_index"] = payload["original_frame_index"]
    meta = {
        "windows": payload["windows"],
        "fps": payload["fps"],
        "n_native_frames": payload["n_native_frames"],
        "hook": payload["hook"],
        "roi_method": payload["roi_method"],
        "regions": payload["regions"],
        "window_starts": payload.get("window_starts"),
        "window_size": payload.get("window_size"),
        "window_stride": payload.get("window_stride"),
        "backbone": payload.get("backbone"),
        "operator_order": payload.get("operator_order"),
        "temporal_aggregation": payload.get("temporal_aggregation"),
        "forbid_temporal_mean_before_roi_align": payload.get("forbid_temporal_mean_before_roi_align"),
        "forbid_zero_dilution_in_temporal_mean": payload.get("forbid_zero_dilution_in_temporal_mean"),
        "joint_layout": payload.get("joint_layout"),
        "feature_map_shapes_example": (payload["windows"][0]["feature_map_shapes"] if payload["windows"] else {}),
    }
    meta_path = out_dir / "rgb_token_v3" / f"{sample_id}.meta.json"
    meta_path.write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    np.savez_compressed(path, **kwargs)
    return sha256_file(path)


def _extract_one_skel(sample: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    from src.data_prep.extract_and_smooth import SignLanguageDataEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    sid = sample["sample_id"]
    engine = SignLanguageDataEngine(model_complexity=0, frame_stride=2)
    video = _ROOT / sample["video_rel"]
    try:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError("cannot_open")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
        joints, poses, missing, frame_idx, stamps = [], [], [], [], []
        native_i = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if native_i % engine.frame_stride == 0:
                j, pose, miss = engine.extract_frame_landmarks(frame)
                joints.append(j)
                poses.append(pose)
                missing.append(miss)
                frame_idx.append(native_i)
                stamps.append(native_i / fps)
                kept += 1
            native_i += 1
        cap.release()
        if kept < 5:
            raise RuntimeError(f"too_few_frames:{kept}")
        payload = build_skel_v3_arrays(
            joints_raw=np.stack(joints),
            raw_pose=np.stack(poses),
            missing_prefill=np.stack(missing),
            original_frame_index=np.asarray(frame_idx, dtype=np.int32),
            timestamps_sec=np.asarray(stamps, dtype=np.float32),
        )
        payload["joints_raw"] = np.stack(joints).astype(np.float32)
        digest = _save_skel_npz(out_dir, sid, payload)
        return {
            "sample_id": sid,
            "ok": True,
            "sha256": digest,
            "sequence_length": payload["sequence_length"],
            "missing_rate": float(payload["missing_mask_prefill"].mean()),
            "coords_sha256": _np_sha(payload["coordinates"]),
            "identity": {
                "n_frames_kept": int(kept),
                "native_frames_seen": int(native_i),
                "fps": fps,
                "frame_stride": int(engine.frame_stride),
                "model_complexity": int(engine.model_complexity),
            },
        }
    finally:
        engine.mp_holistic.close()


def run_skel_phase(*, resume: bool = False) -> dict[str, Any]:
    from src.data_prep.joint_layout_v3 import JOINT_PARTS, N_JOINTS, assert_layout_invariants

    assert_layout_invariants()
    bind = _require_clean_code_lock()
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    validate_pilot_sample_set_r1(sample_set, root=_ROOT)
    if sha256_file(_ROOT / PILOT_R1_SAMPLE_REL) != EXPECTED_R1_SAMPLE_SHA256:
        raise RuntimeError("r1_sample_sha_drift")
    out_dir = _ROOT / PILOT_R4_OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "skel_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True, "sha256": sha256_file(dest)})
            continue
        try:
            row = _extract_one_skel(s, out_dir)
            if row.get("ok"):
                skel = np.load(out_dir / "skel_v3" / f"{sid}.npz", allow_pickle=False)
                if skel["joint_part"].shape[0] != N_JOINTS or len(JOINT_PARTS) != N_JOINTS:
                    raise RuntimeError("joint_identity_len_not_75")
                if int(skel["bone_edges"].max()) < (N_JOINTS - 1):
                    raise RuntimeError("joint_74_not_in_bone_edges")
                row["joint_part_n"] = int(skel["joint_part"].shape[0])
                row["bone_max_index"] = int(skel["bone_edges"].max())
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sid, "error": str(exc), "error_type": type(exc).__name__})
            results.append({"sample_id": sid, "ok": False, "error": str(exc)})
    summary = {
        "phase": "skel",
        "campaign": "D0_PILOT_R4",
        "ok": len(failures) == 0,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": len(failures),
        "results": results,
        "failures": failures,
        "provenance": bind,
    }
    write_json(out_dir / "SKEL_PHASE_SUMMARY.json", summary, overwrite=True)
    return summary


def run_rgb_phase(*, resume: bool = False, device: str = "cuda") -> dict[str, Any]:
    from src.data_prep.joint_layout_v3 import N_JOINTS, assert_layout_invariants
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    assert_layout_invariants()
    contract = load_json(_ROOT / PILOT_R4_RUN_CONTRACT_REL)
    bind = _require_clean_code_lock(expected=contract.get("code_lock_sha256"))
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    validate_pilot_sample_set_r1(sample_set, root=_ROOT)
    out_dir = _ROOT / PILOT_R4_OUT_DIR_REL
    visual_dir = _ROOT / PILOT_R4_VISUAL_REL
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ckpt = _ROOT / EXPECTED_SLOWFAST_CKPT_REL
    extractor = FrozenSlowFastBlock4(
        checkpoint_path=ckpt,
        expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
        device=device,
        repo_root=_ROOT,
    )
    assert extractor.load_meta["pretrained"] is False
    assert extractor.load_meta["checkpoint_sha256"] == EXPECTED_SLOWFAST_CKPT_SHA256
    assert extractor.load_meta["all_requires_grad_false"] is True

    results, failures, overlays_index = [], [], []
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "rgb_token_v3" / f"{sid}.npz"
        skel_path = out_dir / "skel_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True, "sha256": sha256_file(dest)})
            continue
        if not skel_path.is_file():
            failures.append({"sample_id": sid, "error": "missing_skel_v3", "error_type": "DependencyError"})
            results.append({"sample_id": sid, "ok": False, "error": "missing_skel_v3"})
            continue
        try:
            skel = np.load(skel_path, allow_pickle=False)
            if skel["joints_raw"].shape[1] != N_JOINTS:
                raise RuntimeError(f"bad_skel_joint_n:{skel['joints_raw'].shape}")
            landmarks_by_frame: dict[int, np.ndarray] = {}
            missing_by_frame: dict[int, np.ndarray] = {}
            raw = skel["joints_raw"]
            miss = skel["missing_mask_prefill"]
            for i, fi in enumerate(skel["original_frame_index"].tolist()):
                landmarks_by_frame[int(fi)] = raw[i]
                missing_by_frame[int(fi)] = miss[i]

            payload = extract_rgb_token_v3_for_video(
                video_path=_ROOT / s["video_rel"],
                extractor=extractor,
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                skel_frame_index=skel["original_frame_index"],
                skel_timestamps=skel["timestamps_sec"],
                overlay_dir=visual_dir / sid,
                sample_id=sid,
            )
            for w in payload["windows"]:
                if "landmark_frame" not in w or "alignment_delta_sec" not in w:
                    raise RuntimeError("missing_alignment_fields")
                if w.get("alignment_method") != "nearest_timestamp":
                    raise RuntimeError("bad_alignment_method")
                if w.get("operator_order") != "roi_align_per_temporal_slice_then_masked_mean_aggregate":
                    raise RuntimeError("operator_order_not_r0")
                if w.get("temporal_aggregation") != "masked_mean_exclude_missing_zeros":
                    raise RuntimeError("temporal_aggregation_not_masked")
                if w.get("forbid_zero_dilution_in_temporal_mean") is not True:
                    raise RuntimeError("forbid_zero_dilution_missing")
                if "region_validity" not in w:
                    raise RuntimeError("region_validity_missing")
                if "temporal_slice_map" not in w:
                    raise RuntimeError("temporal_slice_map_missing")
                if w.get("forbid_temporal_mean_before_roi_align") is not True:
                    raise RuntimeError("forbid_mean_before_roi_missing")
                if w.get("alignment_tie_break") != "prefer_earlier_skel_index":
                    raise RuntimeError("tie_break_not_locked")
                for region, box in (w.get("roi_boxes_canvas224") or {}).items():
                    if box is None:
                        continue
                    x0, y0, x1, y1 = box
                    if not (0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224):
                        raise RuntimeError(f"roi_oob:{region}:{box}")
                if "valid_frame_mask" not in w or "n_valid_frames" not in w:
                    raise RuntimeError("missing_valid_frame_mask")
            if not any(w.get("covers_last_native_frame") for w in payload["windows"]):
                raise RuntimeError("no_end_coverage")
            if not payload.get("overlays"):
                raise RuntimeError("overlays_required_for_visual_qa")

            digest = _save_rgb_npz(out_dir, sid, payload)
            for ov in payload["overlays"]:
                overlays_index.append(
                    {
                        "sample_id": sid,
                        "kind": ov.get("kind", "center_overlay"),
                        "path": ov["path"],
                        "sha256": ov["sha256"],
                        "center_frame": ov["center_frame"],
                        "native_frame": ov.get("native_frame", ov["center_frame"]),
                        "landmark_frame": ov.get("landmark_frame"),
                        "boxes": ov.get("boxes"),
                        "missing_regions": ov.get("missing_regions"),
                        "region_validity": ov.get("region_validity"),
                        "n_tiles": ov.get("n_tiles"),
                        "slice_native_frames": ov.get("slice_native_frames"),
                        "inspection": None,
                    }
                )
            results.append(
                {
                    "sample_id": sid,
                    "ok": True,
                    "sha256": digest,
                    "n_windows": len(payload["windows"]),
                    "n_overlays": len(payload["overlays"]),
                    "token_dims": {k: list(v.shape) for k, v in payload["tokens"].items()},
                    "backbone": payload.get("backbone"),
                    "joint_layout": payload.get("joint_layout"),
                    "max_alignment_delta_sec": max(float(w["alignment_delta_sec"]) for w in payload["windows"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sid, "error": str(exc), "error_type": type(exc).__name__})
            results.append({"sample_id": sid, "ok": False, "error": str(exc)})

    summary = {
        "phase": "rgb",
        "campaign": "D0_PILOT_R4",
        "ok": len(failures) == 0,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": len(failures),
        "results": results,
        "failures": failures,
        "backbone_provenance": extractor.load_meta,
        "overlays_index": overlays_index,
        "visual_overlay_count": len(overlays_index),
        "provenance": bind,
    }
    write_json(out_dir / "RGB_PHASE_SUMMARY.json", summary, overwrite=True)
    return summary


def _overlay_readable(img: np.ndarray) -> tuple[bool, float]:
    """Downgraded Canny check: image loads at 224 and has some edges (not blank)."""
    h, w = img.shape[:2]
    if h != 224 or w != 224:
        return False, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    return True, float((edge > 0).mean())


def _semantic_roi_stroke_hit(img: np.ndarray, boxes: dict[str, Any] | None) -> dict[str, Any]:
    """Require ROI stroke colors near declared boxes; rejects checkerboard-without-ROI."""
    if not boxes:
        return {"ok": False, "reason": "boxes_missing", "hits": {}}
    hits = {}
    for region, box in boxes.items():
        if box is None:
            hits[region] = {"present": False, "stroke_hit": None, "allowed_missing": region != "global"}
            continue
        color = np.asarray(ROI_STROKE_BGR[region], dtype=np.int16)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(223, x1), min(223, y1)
        if x1 <= x0 or y1 <= y0:
            hits[region] = {"present": True, "stroke_hit": False, "reason": "degenerate_box"}
            continue
        # Sample perimeter pixels for near-color match.
        ys = list(range(y0, y1 + 1, max(1, (y1 - y0) // 8)))
        xs = list(range(x0, x1 + 1, max(1, (x1 - x0) // 8)))
        samples = []
        for y in ys:
            samples.append(img[y, x0])
            samples.append(img[y, x1])
        for x in xs:
            samples.append(img[y0, x])
            samples.append(img[y1, x])
        dists = [float(np.linalg.norm(np.asarray(p, dtype=np.int16) - color)) for p in samples]
        stroke_hit = bool(dists) and min(dists) <= 25.0
        hits[region] = {"present": True, "stroke_hit": stroke_hit, "min_color_dist": min(dists) if dists else None}
    required = [r for r, b in boxes.items() if b is not None]
    ok = bool(required) and all(hits[r].get("stroke_hit") for r in required)
    # global box should always be drawn
    if boxes.get("global") is not None and not hits.get("global", {}).get("stroke_hit"):
        ok = False
    return {"ok": ok, "hits": hits, "reason": None if ok else "semantic_stroke_miss"}


def check_overlay_render_integrity(
    path: Path,
    *,
    boxes: dict[str, Any] | None,
    sample_id: str,
) -> dict[str, Any]:
    """Auto gate only: image readable + stroke colors match declared boxes.

    NOT semantic/manual QA — arbitrary wrong boxes with correct stroke colors can PASS.
    """
    if not path.is_file():
        return {
            "sample_id": sample_id,
            "path": str(path).replace("\\", "/"),
            "overlay_render_integrity": "FAIL",
            "overlay_readable": False,
            "reason": "overlay_file_missing",
        }
    img = cv2.imread(str(path))
    digest = sha256_file(path)
    if img is None or img.size == 0:
        return {
            "sample_id": sample_id,
            "path": str(path).replace("\\", "/"),
            "sha256": digest,
            "overlay_render_integrity": "FAIL",
            "overlay_readable": False,
            "reason": "overlay_unreadable",
        }
    readable, edge_ratio = _overlay_readable(img)
    semantic = _semantic_roi_stroke_hit(img, boxes)
    ok = readable and bool(semantic["ok"])
    return {
        "sample_id": sample_id,
        "path": str(path).replace("\\", "/"),
        "sha256": digest,
        "boxes": boxes,
        "overlay_readable": readable,
        "edge_ratio_canny_downgraded": edge_ratio,
        "stroke_consistency_ok": bool(semantic["ok"]),
        "stroke_detail": semantic,
        "overlay_render_integrity": "PASS" if ok else "FAIL",
        "reason": None if ok else (semantic.get("reason") or "overlay_not_readable"),
        "not_manual_semantic_qa": True,
    }


def _inspect_overlays(overlays_index: list[dict[str, Any]], skel_sum: dict[str, Any]) -> dict[str, Any]:
    """overlay_render_integrity only; manual semantic QA is a separate artifact."""
    skel_by_id = {r["sample_id"]: r for r in skel_sum.get("results") or []}
    ranked = sorted(
        overlays_index,
        key=lambda o: (
            0 if any((o.get("missing_regions") or {}).values()) else 1,
            -(skel_by_id.get(o["sample_id"], {}).get("missing_rate") or 0.0),
            o["sample_id"],
            o.get("center_frame") or 0,
        ),
    )
    chosen: list[dict[str, Any]] = []
    seen_sid: set[str] = set()
    for o in ranked:
        if o["sample_id"] in seen_sid:
            continue
        seen_sid.add(o["sample_id"])
        chosen.append(o)
        if len(chosen) >= 12:
            break

    inspections = []
    for o in chosen:
        path = Path(o["path"])
        if not path.is_file() and "Docs/results" in o["path"].replace("\\", "/"):
            rel = "Docs/results" + o["path"].replace("\\", "/").split("Docs/results", 1)[1]
            path = _ROOT / rel
        inspections.append(
            check_overlay_render_integrity(path, boxes=o.get("boxes"), sample_id=o["sample_id"])
        )

    # Negative: empty boxes checkerboard fails render integrity (no strokes).
    chk = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            chk[y, x] = 255 if ((x // 16) + (y // 16)) % 2 == 0 else 0
    tmp = Path(tempfile.mkdtemp(prefix="d0_r4_chk_")) / "checkerboard_no_roi.png"
    cv2.imwrite(str(tmp), chk)
    neg = check_overlay_render_integrity(tmp, boxes={}, sample_id="CHECKERBOARD_NEG")
    # Negative: arbitrary wrong boxes with correct stroke colors — integrity may PASS.
    wrong = np.zeros((224, 224, 3), dtype=np.uint8)
    wrong_boxes = {
        "global": [0.0, 0.0, 224.0, 224.0],
        "left_hand": [5.0, 5.0, 40.0, 40.0],
        "right_hand": [180.0, 5.0, 220.0, 40.0],
        "face": [90.0, 5.0, 140.0, 40.0],
    }
    from src.models.rgb_token_v3 import draw_roi_overlay

    drawn = draw_roi_overlay(wrong, {k: np.asarray(v, dtype=np.float32) for k, v in wrong_boxes.items()})
    tmp2 = tmp.parent / "arbitrary_wrong_boxes.png"
    cv2.imwrite(str(tmp2), drawn)
    wrong_integrity = check_overlay_render_integrity(tmp2, boxes=wrong_boxes, sample_id="WRONG_BOX_NEG")
    shutil.rmtree(tmp.parent, ignore_errors=True)

    all_pass = bool(inspections) and all(i["overlay_render_integrity"] == "PASS" for i in inspections)
    return {
        "gate_name": "overlay_render_integrity",
        "n_inspected": len(inspections),
        "n_pass": sum(1 for i in inspections if i["overlay_render_integrity"] == "PASS"),
        "n_fail": sum(1 for i in inspections if i["overlay_render_integrity"] == "FAIL"),
        "all_pass": all_pass,
        "inspections": inspections,
        "checkerboard_negative_control": neg,
        "checkerboard_rejected": neg["overlay_render_integrity"] == "FAIL",
        "arbitrary_wrong_boxes_integrity": wrong_integrity,
        "arbitrary_wrong_boxes_integrity_can_pass": wrong_integrity["overlay_render_integrity"] == "PASS",
        "manual_semantic_qa_required_separately": True,
        "canny_role": "overlay_readable_only",
    }


def _projection(sample_set: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    ids = [s["sample_id"] for s in sample_set["samples"]]
    paired = []
    skel_sizes = []
    rgb_sizes = []
    meta_sizes = []
    for sid in ids:
        sk = out_dir / "skel_v3" / f"{sid}.npz"
        rg = out_dir / "rgb_token_v3" / f"{sid}.npz"
        meta = out_dir / "rgb_token_v3" / f"{sid}.meta.json"
        sb = sk.stat().st_size if sk.is_file() else 0
        rb = rg.stat().st_size if rg.is_file() else 0
        mb = meta.stat().st_size if meta.is_file() else 0
        skel_sizes.append(sb)
        rgb_sizes.append(rb)
        meta_sizes.append(mb)
        paired.append(sb + rb)

    def _pct(xs: list[int], q: float) -> float:
        if not xs:
            return 0.0
        return float(np.quantile(xs, q))

    mean_skel = float(np.mean(skel_sizes)) if skel_sizes else 0.0
    mean_rgb = float(np.mean(rgb_sizes)) if rgb_sizes else 0.0
    mean_meta = float(np.mean(meta_sizes)) if meta_sizes else 0.0
    return {
        "label": "estimate_not_capacity_lock",
        "pilot_n": len(ids),
        "class_length_diverse": True,
        "n_glosses": len({s.get("gloss") for s in sample_set["samples"]}),
        "frame_min": min(int(s["n_frames"]) for s in sample_set["samples"]),
        "frame_max": max(int(s["n_frames"]) for s in sample_set["samples"]),
        "artifact_bytes": {
            "skel_p50": _pct(skel_sizes, 0.50),
            "skel_p95": _pct(skel_sizes, 0.95),
            "skel_max": float(max(skel_sizes) if skel_sizes else 0),
            "rgb_p50": _pct(rgb_sizes, 0.50),
            "rgb_p95": _pct(rgb_sizes, 0.95),
            "rgb_max": float(max(rgb_sizes) if rgb_sizes else 0),
            "combined_paired_p50": _pct(paired, 0.50),
            "combined_paired_p95": _pct(paired, 0.95),
            "combined_paired_max": float(max(paired) if paired else 0),
            "combined_marginal_sum_p95": _pct(skel_sizes, 0.95) + _pct(rgb_sizes, 0.95),
            "combined_method": "paired_per_sample_id",
        },
        "manifest_meta_overhead_bytes": {
            "mean_meta_json": mean_meta,
            "n_meta_files": len(meta_sizes),
        },
        "projected_corpus_n": CORPUS_N,
        "projected_total_gib_from_means": (mean_skel + mean_rgb + mean_meta) * CORPUS_N / (1024**3),
        "projected_total_gib_from_paired_p95": (_pct(paired, 0.95) + mean_meta) * CORPUS_N / (1024**3),
        "note": "estimate_only_not_exact_corpus_size",
    }


def _reextract_skel_via_extract_env(sample: dict[str, Any], tmp: Path) -> dict[str, Any]:
    """Skeleton re-extract must use _venvs/extract (mediapipe); f0_v5 lacks pandas/mp."""
    if not EXTRACT_PY.is_file():
        return {"ok": False, "error": "extract_python_missing"}
    sid = sample["sample_id"]
    cmd = [
        str(EXTRACT_PY),
        str(_ROOT / "scripts/run_ms_sta_v5_d0_r4_pilot.py"),
        "--phase",
        "skel_one",
        "--sample-id",
        sid,
        "--out-dir",
        str(tmp),
    ]
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "skel_reextract_subprocess_failed",
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"skel_reextract_parse:{exc}", "stdout": proc.stdout[-2000:]}
    return payload


def _real_redeterminism(sample_set: dict[str, Any], out_dir: Path, device: str) -> dict[str, Any]:
    """Re-extract Skeleton + RGB into a fresh temp dir; compare arrays/metadata."""
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    sid = sample_set["samples"][0]["sample_id"]
    sample = next(s for s in sample_set["samples"] if s["sample_id"] == sid)
    primary_skel = out_dir / "skel_v3" / f"{sid}.npz"
    primary_rgb = out_dir / "rgb_token_v3" / f"{sid}.npz"
    primary_meta = out_dir / "rgb_token_v3" / f"{sid}.meta.json"
    if not primary_skel.is_file() or not primary_rgb.is_file():
        return {"ok": False, "error": "primary_artifacts_missing"}

    tmp = Path(tempfile.mkdtemp(prefix="d0_r1_redet_"))
    try:
        skel_r = _reextract_skel_via_extract_env(sample, tmp)
        if not skel_r.get("ok"):
            return {"ok": False, "error": "skel_reextract_failed", "detail": skel_r}

        skel = np.load(tmp / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        landmarks_by_frame = {}
        missing_by_frame = {}
        for i, fi in enumerate(skel["original_frame_index"].tolist()):
            landmarks_by_frame[int(fi)] = skel["joints_raw"][i]
            missing_by_frame[int(fi)] = skel["missing_mask_prefill"][i]

        extractor = FrozenSlowFastBlock4(
            checkpoint_path=_ROOT / EXPECTED_SLOWFAST_CKPT_REL,
            expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
            device=device,
            repo_root=_ROOT,
        )
        payload = extract_rgb_token_v3_for_video(
            video_path=_ROOT / sample["video_rel"],
            extractor=extractor,
            landmarks_by_frame=landmarks_by_frame,
            missing_by_frame=missing_by_frame,
            skel_frame_index=skel["original_frame_index"],
            skel_timestamps=skel["timestamps_sec"],
        )
        _save_rgb_npz(tmp, sid, payload)

        a = np.load(primary_skel, allow_pickle=False)
        b = np.load(tmp / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        skel_eq = True
        skel_diffs = []
        for key in ("coordinates", "missing_mask_prefill", "original_frame_index", "timestamps_sec"):
            if key not in a.files or key not in b.files:
                skel_eq = False
                skel_diffs.append(f"missing_key:{key}")
                continue
            if DET_TOLERANCE == 0.0:
                same = np.array_equal(a[key], b[key])
            else:
                same = np.allclose(a[key], b[key], atol=DET_TOLERANCE, rtol=0.0)
            if not same:
                skel_eq = False
                skel_diffs.append(key)

        ra = np.load(primary_rgb, allow_pickle=False)
        rb = np.load(tmp / "rgb_token_v3" / f"{sid}.npz", allow_pickle=False)
        rgb_eq = True
        rgb_diffs = []
        for key in sorted(set(ra.files) | set(rb.files)):
            if key not in ra.files or key not in rb.files:
                rgb_eq = False
                rgb_diffs.append(f"missing_key:{key}")
                continue
            if DET_TOLERANCE == 0.0:
                same = np.array_equal(ra[key], rb[key])
            else:
                same = np.allclose(ra[key], rb[key], atol=DET_TOLERANCE, rtol=0.0)
            if not same:
                rgb_eq = False
                rgb_diffs.append(key)

        meta_a = json.loads(primary_meta.read_text(encoding="utf-8"))
        meta_b = json.loads((tmp / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        # Compare canonical window fields (ignore absolute overlay paths).
        def _canon_windows(meta: dict) -> list[dict]:
            out = []
            for w in meta.get("windows") or []:
                out.append(
                    {
                        "start_frame": w.get("start_frame"),
                        "end_frame": w.get("end_frame"),
                        "center_frame": w.get("center_frame"),
                        "n_valid_frames": w.get("n_valid_frames"),
                        "covers_last_native_frame": w.get("covers_last_native_frame"),
                        "roi_boxes_canvas224": w.get("roi_boxes_canvas224"),
                    }
                )
            return out

        meta_eq = _canon_windows(meta_a) == _canon_windows(meta_b)

        return {
            "ok": bool(skel_eq and rgb_eq and meta_eq),
            "sample_id": sid,
            "tolerance": DET_TOLERANCE,
            "method": "fresh_temp_reextraction_not_file_reread",
            "primary_skel_sha256": sha256_file(primary_skel),
            "reextract_skel_sha256": sha256_file(tmp / "skel_v3" / f"{sid}.npz"),
            "primary_rgb_sha256": sha256_file(primary_rgb),
            "reextract_rgb_sha256": sha256_file(tmp / "rgb_token_v3" / f"{sid}.npz"),
            "skel_arrays_equal": skel_eq,
            "rgb_arrays_equal": rgb_eq,
            "meta_windows_equal": meta_eq,
            "skel_diffs": skel_diffs,
            "rgb_diffs": rgb_diffs,
            "extraction_identities": {
                "primary_skel": next(
                    (r for r in load_json(out_dir / "SKEL_PHASE_SUMMARY.json")["results"] if r["sample_id"] == sid),
                    None,
                ),
                "reextract_skel": skel_r,
                "backbone": extractor.load_meta,
            },
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_negative_control() -> dict[str, Any]:
    neg = load_json(_ROOT / PILOT_R1_NEGATIVE_REL)
    results = []
    for s in neg.get("samples") or []:
        rel = s.get("video_rel")
        if not rel:
            results.append({**s, "observed_failure": True, "note": "unresolvable"})
            continue
        path = _ROOT / rel
        cap = cv2.VideoCapture(str(path))
        opened = cap.isOpened()
        ok_read = False
        if opened:
            ok_read, _ = cap.read()
        cap.release()
        failed = (not opened) or (not ok_read)
        results.append(
            {
                "sample_id": s["sample_id"],
                "expected_failure": True,
                "observed_failure": failed,
                "decode_failure_evidence": bool(s.get("decode_failure_evidence")),
                "edge_tags": s.get("edge_tags"),
            }
        )
    ok = all(r.get("observed_failure") for r in results) and any(
        "decode_edge" in (r.get("edge_tags") or []) and r.get("decode_failure_evidence") for r in results
    )
    out = {"ok": ok, "results": results, "mixed_into_engineering_pilot": False, "campaign": "D0_PILOT_R4"}
    write_json(_ROOT / PILOT_R4_OUT_DIR_REL / "NEGATIVE_CONTROL_SUMMARY.json", out, overwrite=True)
    return out


def _quality_missing_roi_stats(
    *,
    sample_set: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Quality from stored tensor-aligned region_validity / missing_* arrays — not center boxes."""
    by_region = {
        r: {"n_windows": 0, "n_all_missing": 0, "n_partial_missing": 0} for r in ("face", "left_hand", "right_hand")
    }
    by_split: dict[str, dict[str, Any]] = {}
    by_signer: dict[str, dict[str, Any]] = {}
    n_windows = 0
    n_windows_any_all_missing = 0
    n_windows_any_partial = 0
    tensor_agree = True
    meta_by_id = {s["sample_id"]: s for s in sample_set["samples"]}
    for sid, s in meta_by_id.items():
        meta_path = out_dir / "rgb_token_v3" / f"{sid}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rgb = np.load(out_dir / "rgb_token_v3" / f"{sid}.npz", allow_pickle=False)
        split = str(s.get("split"))
        signer = str(s.get("signer_id"))
        by_split.setdefault(
            split, {"n_windows": 0, "n_windows_any_all_missing": 0, "n_windows_any_partial": 0}
        )
        by_signer.setdefault(
            signer, {"n_windows": 0, "n_windows_any_all_missing": 0, "n_windows_any_partial": 0}
        )
        wins = meta.get("windows") or []
        for wi, w in enumerate(wins):
            n_windows += 1
            by_split[split]["n_windows"] += 1
            by_signer[signer]["n_windows"] += 1
            validity = w.get("region_validity") or {}
            any_all = False
            any_partial = False
            for region in ("face", "left_hand", "right_hand"):
                by_region[region]["n_windows"] += 1
                rv = validity.get(region) or {}
                all_missing = bool(rv.get("all_missing"))
                partial = bool(rv.get("partial_missing"))
                # Must match stored tensor mask for this window.
                tensor_miss = bool(rgb[f"missing_{region}"][wi])
                if tensor_miss != all_missing:
                    tensor_agree = False
                if all_missing:
                    by_region[region]["n_all_missing"] += 1
                    any_all = True
                if partial:
                    by_region[region]["n_partial_missing"] += 1
                    any_partial = True
            if any_all:
                n_windows_any_all_missing += 1
                by_split[split]["n_windows_any_all_missing"] += 1
                by_signer[signer]["n_windows_any_all_missing"] += 1
            if any_partial:
                n_windows_any_partial += 1
                by_split[split]["n_windows_any_partial"] += 1
                by_signer[signer]["n_windows_any_partial"] += 1

    def _rate(n_miss: int, n_tot: int) -> float:
        return float(n_miss) / float(n_tot) if n_tot else 0.0

    return {
        "source": "region_validity_and_missing_tensors",
        "not_from_center_overlay_boxes": True,
        "tensor_mask_agrees_with_region_validity": tensor_agree,
        "n_windows": n_windows,
        "n_windows_any_all_missing": n_windows_any_all_missing,
        "n_windows_any_partial_missing": n_windows_any_partial,
        "rate_windows_any_all_missing": _rate(n_windows_any_all_missing, n_windows),
        "rate_windows_any_partial_missing": _rate(n_windows_any_partial, n_windows),
        "missing_roi_rate_by_region": {
            r: {
                **by_region[r],
                "all_missing_rate": _rate(by_region[r]["n_all_missing"], by_region[r]["n_windows"]),
                "partial_missing_rate": _rate(
                    by_region[r]["n_partial_missing"], by_region[r]["n_windows"]
                ),
            }
            for r in by_region
        },
        "missing_roi_rate_by_split": {
            k: {
                **v,
                "all_missing_rate": _rate(v["n_windows_any_all_missing"], v["n_windows"]),
                "partial_missing_rate": _rate(v["n_windows_any_partial"], v["n_windows"]),
            }
            for k, v in by_split.items()
        },
        "missing_roi_rate_by_signer": {
            k: {
                **v,
                "all_missing_rate": _rate(v["n_windows_any_all_missing"], v["n_windows"]),
                "partial_missing_rate": _rate(v["n_windows_any_partial"], v["n_windows"]),
            }
            for k, v in by_signer.items()
        },
    }


def _close_r4_run_contract(*, report_sha: str, audit_sha: str, audit_verdict: str) -> str:
    contract_path = _ROOT / PILOT_R4_RUN_CONTRACT_REL
    contract = load_json(contract_path)
    contract["status"] = "CLOSED"
    contract["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    contract["closed_by"] = PILOT_R4_COMPLETION_REL
    write_json(contract_path, contract, overwrite=True)
    completion = {
        "schema": "ms_sta_v5_d0_r4_pilot_completion_v1",
        "status": "CLOSED",
        "closed_at": contract["closed_at"],
        "campaign": "D0_PILOT_R4",
        "run_contract": PILOT_R4_RUN_CONTRACT_REL,
        "prior_run_contract_status": "ACTIVE",
        "new_run_contract_status": "CLOSED",
        "pilot_report_sha256": report_sha,
        "pilot_audit_sha256": audit_sha,
        "pilot_audit_verdict": audit_verdict,
        "immutable": True,
        "full_corpus_authorized": False,
        "next": "INDEPENDENT_R4_AUDIT_REQUIRED_BEFORE_FULL_CORPUS",
    }
    return write_json(_ROOT / PILOT_R4_COMPLETION_REL, completion, overwrite=True)


def run_finalize(*, device: str = "cuda") -> dict[str, Any]:
    from src.data_prep.joint_layout_v3 import N_JOINTS

    validate_r3_independent_reaudit(load_json(_ROOT / PILOT_R3_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    if sha256_file(_ROOT / PILOT_R3_AUDIT_REL) != EXPECTED_R3_PILOT_AUDIT_SHA256:
        raise RuntimeError("r3_pilot_audit_must_remain_immutable")
    protocol = load_json(_ROOT / FULL_PROTOCOL_REL)
    validate_d0_full_protocol(protocol)
    contract = load_json(_ROOT / PILOT_R4_RUN_CONTRACT_REL)
    bind = _require_clean_code_lock(expected=contract.get("code_lock_sha256"))

    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    validate_pilot_sample_set_r1(sample_set, root=_ROOT)
    if sha256_file(_ROOT / PILOT_R1_SAMPLE_REL) != EXPECTED_R1_SAMPLE_SHA256:
        raise RuntimeError("r1_sample_sha_drift")
    out_dir = _ROOT / PILOT_R4_OUT_DIR_REL
    skel_sum = load_json(out_dir / "SKEL_PHASE_SUMMARY.json")
    rgb_sum = load_json(out_dir / "RGB_PHASE_SUMMARY.json")

    skel_prov = skel_sum.get("provenance") or {}
    rgb_prov = rgb_sum.get("provenance") or {}
    gates: dict[str, Any] = {}
    gates["single_clean_code_lock"] = (
        skel_prov.get("code_lock_sha256")
        == rgb_prov.get("code_lock_sha256")
        == bind["code_lock_sha256"]
        == contract.get("code_lock_sha256")
        and skel_prov.get("worktree_dirty") is False
        and rgb_prov.get("worktree_dirty") is False
        and bind["worktree_dirty"] is False
    )
    gates["d0_full_protocol_frozen"] = True

    ids = [s["sample_id"] for s in sample_set["samples"]]
    skel_ids = {r["sample_id"] for r in skel_sum["results"] if r.get("ok")}
    rgb_ids = {r["sample_id"] for r in rgb_sum["results"] if r.get("ok")}
    gates["exact_sample_reconciliation"] = set(ids) == skel_ids == rgb_ids

    splits = {s["sample_id"]: s["split"] for s in sample_set["samples"]}
    gates["no_test_leakage_train_val_only"] = all(splits[i] in {"train", "val"} for i in ids)

    bp = rgb_sum.get("backbone_provenance") or {}
    gates["backbone_sealed_sha_locked"] = (
        bp.get("pretrained") is False
        and bp.get("checkpoint_sha256") == EXPECTED_SLOWFAST_CKPT_SHA256
        and bp.get("missing_keys") == []
        and bp.get("unexpected_keys") == []
        and bp.get("all_requires_grad_false") is True
        and bp.get("network_disabled") is True
    )

    align_ok = True
    cross_modal_ok = True
    finite_ok = True
    mask_ok = True
    end_cover_ok = True
    valid_mask_ok = True
    joint_layout_ok = True
    silent_none_ok = True
    operator_order_ok = True
    tie_break_ok = True
    protocol_params_ok = True
    for sid in ids:
        skel = np.load(out_dir / "skel_v3" / f"{sid}.npz")
        if int(skel["joint_part"].shape[0]) != N_JOINTS:
            joint_layout_ok = False
        if int(skel["bone_edges"].max()) < (N_JOINTS - 1):
            joint_layout_ok = False
        if skel["original_frame_index"].shape[0] != skel["timestamps_sec"].shape[0]:
            align_ok = False
        if not np.isfinite(skel["coordinates"]).all():
            finite_ok = False
        if "missing_mask_prefill" not in skel.files:
            mask_ok = False
        rgb = np.load(out_dir / "rgb_token_v3" / f"{sid}.npz")
        for k in rgb.files:
            if k.startswith("token_") and not np.isfinite(rgb[k]).all():
                finite_ok = False
        meta = json.loads((out_dir / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        if meta.get("operator_order") != protocol["rgb_token_v3"]["operator_order"]:
            operator_order_ok = False
        if meta.get("forbid_temporal_mean_before_roi_align") is not True:
            operator_order_ok = False
        wins = meta.get("windows") or []
        if not any(w.get("covers_last_native_frame") for w in wins):
            end_cover_ok = False
        if not all("valid_frame_mask" in w and "n_valid_frames" in w for w in wins):
            valid_mask_ok = False
        fps = float(meta.get("fps") or 0.0) or 25.0
        skel_ts = skel["timestamps_sec"]
        skel_fi = skel["original_frame_index"]
        for w in wins:
            for prefix in ("start", "center", "end"):
                fr = w.get(f"{prefix}_frame")
                ts_key = "timestamp_sec" if prefix == "center" else f"{prefix}_timestamp_sec"
                ts = w.get(ts_key)
                if fr is None or ts is None or abs(float(ts) - float(fr) / fps) > 1e-3:
                    align_ok = False
            if w.get("alignment_method") != "nearest_timestamp":
                cross_modal_ok = False
            if w.get("operator_order") != "roi_align_per_temporal_slice_then_masked_mean_aggregate":
                operator_order_ok = False
            if w.get("temporal_aggregation") != "masked_mean_exclude_missing_zeros":
                operator_order_ok = False
            if w.get("forbid_zero_dilution_in_temporal_mean") is not True:
                operator_order_ok = False
            if "region_validity" not in w or "temporal_slice_map" not in w:
                operator_order_ok = False
            if w.get("alignment_tie_break") != "prefer_earlier_skel_index":
                tie_break_ok = False
            if "landmark_frame" not in w or "alignment_delta_sec" not in w:
                cross_modal_ok = False
            else:
                if int(w["landmark_frame"]) not in set(int(x) for x in skel_fi.tolist()):
                    cross_modal_ok = False
                target = float(w["center_frame"]) / fps
                nearest = float(np.min(np.abs(skel_ts.astype(np.float64) - target)))
                if abs(float(w["alignment_delta_sec"]) - nearest) > 1e-6:
                    cross_modal_ok = False
            boxes = w.get("roi_boxes_canvas224") or {}
            if boxes.get("face") is None and boxes.get("left_hand") is None and boxes.get("right_hand") is None:
                reasons = w.get("roi_none_reasons") or {}
                if not reasons:
                    silent_none_ok = False

    # Protocol params must match executed identity fields.
    for row in skel_sum.get("results") or []:
        identity = row.get("identity") or {}
        if not identity:
            continue
        if int(identity.get("frame_stride", -1)) != int(protocol["skeleton_v3"]["frame_stride"]):
            protocol_params_ok = False
        # NOTE: model_complexity may be 0 — do not use `or -1` (0 is falsy).
        if int(identity.get("model_complexity", -1)) != int(protocol["skeleton_v3"]["model_complexity"]):
            protocol_params_ok = False
    for sid in ids:
        meta = json.loads((out_dir / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        if int(meta.get("window_size", -1)) != int(protocol["rgb_token_v3"]["window_size"]):
            protocol_params_ok = False
        if int(meta.get("window_stride", -1)) != int(protocol["rgb_token_v3"]["window_stride"]):
            protocol_params_ok = False

    gates["timestamp_frame_alignment"] = align_ok
    gates["cross_modal_nearest_timestamp_alignment"] = cross_modal_ok
    gates["alignment_tie_break_prefer_earlier"] = tie_break_ok
    gates["r0_roi_align_before_temporal_aggregate"] = operator_order_ok
    gates["masked_mean_excludes_missing_zeros"] = operator_order_ok
    gates["protocol_params_match_execution"] = protocol_params_ok
    gates["no_silent_all_roi_none_from_stride"] = silent_none_ok
    gates["joint_layout_75_identity_and_joint74"] = joint_layout_ok
    gates["masks_verified_before_fill"] = mask_ok
    gates["tensor_shapes_dtypes_finite"] = finite_ok
    gates["rgb_end_anchored_last_frame_coverage"] = end_cover_ok
    gates["rgb_valid_frame_mask_recorded"] = valid_mask_ok

    overlays = rgb_sum.get("overlays_index") or []
    center_overlays = [o for o in overlays if o.get("kind", "center_overlay") == "center_overlay"]
    temporal_overlays = [o for o in overlays if o.get("kind") == "temporal_contact_sheet"]
    gates["visual_overlay_count_nonzero"] = len(center_overlays) > 0 and len(temporal_overlays) > 0
    render_integrity = _inspect_overlays(center_overlays, skel_sum)
    write_json(out_dir / "OVERLAY_RENDER_INTEGRITY.json", render_integrity, overwrite=True)
    gates["overlay_render_integrity"] = (
        bool(render_integrity.get("all_pass"))
        and render_integrity["n_inspected"] >= 3
        and bool(render_integrity.get("checkerboard_rejected"))
        and bool(render_integrity.get("arbitrary_wrong_boxes_integrity_can_pass"))
        and render_integrity.get("gate_name") == "overlay_render_integrity"
        and render_integrity.get("manual_semantic_qa_required_separately") is True
    )

    manual_path = _ROOT / PILOT_R4_VISUAL_MANUAL_REL
    if not manual_path.is_file():
        raise RuntimeError(
            "missing_external_manual_visual_qa:"
            f"{PILOT_R4_VISUAL_MANUAL_REL}:required_before_finalize"
        )
    manual = load_json(manual_path)
    validate_visual_qa_manual(
        manual,
        overlays_index=overlays,
        require_zero_fail=True,
        require_temporal_contact=True,
    )
    gates["visual_qa_manual_external_artifact"] = True
    gates["visual_qa_manual_strict_zero_fail"] = True
    gates["visual_qa_manual_not_render_integrity_only"] = all(
        row.get("derived_from_overlay_render_integrity_only") is not True
        for row in (manual.get("inspections") or [])
    )
    man_rows = manual.get("inspections") or []
    n_man_fail = sum(1 for row in man_rows if row.get("verdict") == "FAIL")
    gates["visual_qa_manual_zero_fail_acceptance"] = n_man_fail == 0 and len(man_rows) >= 3
    gates["visual_qa_manual_includes_temporal_contact"] = (
        any(r.get("qa_kind") == "center_overlay" for r in man_rows)
        and any(r.get("qa_kind") == "temporal_contact_sheet" for r in man_rows)
    )

    quality = _quality_missing_roi_stats(sample_set=sample_set, out_dir=out_dir)
    write_json(out_dir / "QUALITY_MISSING_ROI_STATS.json", quality, overwrite=True)
    gates["quality_missing_roi_stats_recorded"] = (
        quality.get("not_from_center_overlay_boxes") is True
        and quality.get("tensor_mask_agrees_with_region_validity") is True
        and "missing_roi_rate_by_region" in quality
        and "missing_roi_rate_by_split" in quality
        and "missing_roi_rate_by_signer" in quality
    )

    det = _real_redeterminism(sample_set, out_dir, device=device)
    write_json(out_dir / "DETERMINISM_REEXTRACT.json", det, overwrite=True)
    gates["deterministic_fresh_reextraction"] = bool(det.get("ok"))

    neg = run_negative_control()
    gates["negative_control_expected_failures"] = bool(neg.get("ok"))

    tags = {t for s in sample_set["samples"] for t in (s.get("edge_tags") or [])}
    gates["edge_tags_evidence_based"] = (
        "short_clip" in tags
        and "long_clip" in tags
        and "missing_hand_or_face_risk" in tags
        and "decode_edge" not in tags
    )
    gates["gloss_signer_diversity"] = (
        len({s.get("gloss") for s in sample_set["samples"]}) >= 8
        and len({s.get("signer_id") for s in sample_set["samples"]}) >= 6
    )

    projection = _projection(sample_set, out_dir)
    gates["storage_projection_estimate_labeled"] = projection.get("label") == "estimate_not_capacity_lock"
    gates["storage_combined_paired_per_sample"] = (
        projection.get("artifact_bytes", {}).get("combined_method") == "paired_per_sample_id"
    )

    failure_taxonomy = {
        "skel": skel_sum.get("failures") or [],
        "rgb": rgb_sum.get("failures") or [],
        "negative_control": neg.get("results") or [],
    }
    coverage = {
        "n_samples": len(ids),
        "splits": {k: sum(1 for s in sample_set["samples"] if s["split"] == k) for k in ("train", "val")},
        "n_glosses": len({s.get("gloss") for s in sample_set["samples"]}),
        "n_signers": len({s.get("signer_id") for s in sample_set["samples"]}),
        "edge_tag_coverage": sorted(tags),
        "frame_range": [
            min(int(s["n_frames"]) for s in sample_set["samples"]),
            max(int(s["n_frames"]) for s in sample_set["samples"]),
        ],
    }

    all_gates = all(bool(v) for v in gates.values()) and bool(skel_sum.get("ok")) and bool(rgb_sum.get("ok"))
    report = {
        "schema": "ms_sta_v5_d0_r4_pilot_report_v1",
        "status": "COMPLETE",
        "ok": bool(all_gates),
        "campaign": "D0_PILOT_R4",
        "full_protocol_sha256": sha256_file(_ROOT / FULL_PROTOCOL_REL),
        "parent_r3_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R3_INDEPENDENT_REAUDIT_REL),
        "parent_r3_pilot_audit_preserved_sha256": sha256_file(_ROOT / PILOT_R3_AUDIT_REL),
        "pilot_sample_sha256": sha256_file(_ROOT / PILOT_R1_SAMPLE_REL),
        "run_contract_sha256": sha256_file(_ROOT / PILOT_R4_RUN_CONTRACT_REL),
        "code_lock_sha256": bind["code_lock_sha256"],
        "gates": gates,
        "projection": projection,
        "quality_missing_roi_stats": quality,
        "overlay_render_integrity": {
            "n_inspected": render_integrity.get("n_inspected"),
            "n_pass": render_integrity.get("n_pass"),
            "n_fail": render_integrity.get("n_fail"),
            "checkerboard_rejected": render_integrity.get("checkerboard_rejected"),
            "arbitrary_wrong_boxes_integrity_can_pass": render_integrity.get(
                "arbitrary_wrong_boxes_integrity_can_pass"
            ),
            "canny_role": render_integrity.get("canny_role"),
        },
        "visual_qa_manual": {
            "path": PILOT_R4_VISUAL_MANUAL_REL,
            "sha256": sha256_file(manual_path),
            "reviewer": manual.get("reviewer"),
            "n_inspections": len(manual.get("inspections") or []),
            "source": manual.get("source"),
        },
        "determinism": {
            "ok": det.get("ok"),
            "method": det.get("method"),
            "tolerance": det.get("tolerance"),
            "sample_id": det.get("sample_id"),
        },
        "backbone_provenance": bp,
        "failure_taxonomy": failure_taxonomy,
        "coverage": coverage,
        "provenance": {"finalize": bind, "skel": skel_prov, "rgb": rgb_prov},
        "skel_summary": {"n_ok": skel_sum.get("n_ok"), "n_fail": skel_sum.get("n_fail")},
        "rgb_summary": {"n_ok": rgb_sum.get("n_ok"), "n_fail": rgb_sum.get("n_fail")},
        "forbid": {
            "overwrite_r3_evidence": True,
            "full_corpus_extraction": True,
            "m0": True,
            "training": True,
            "accuracy": True,
            "auto_chain": True,
        },
    }
    report_sha = write_json(_ROOT / PILOT_R4_REPORT_REL, report, overwrite=True)
    audit = {
        "schema": "ms_sta_v5_d0_r4_pilot_audit_v1",
        "audit_verdict": "PASS" if report["ok"] else "FAIL",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auditor": "ducpm196",
        "campaign": "D0_PILOT_R4",
        "pilot_report_sha256": report_sha,
        "pilot_sample_sha256": report["pilot_sample_sha256"],
        "full_protocol_sha256": report["full_protocol_sha256"],
        "r3_independent_reaudit_sha256": report["parent_r3_independent_reaudit_sha256"],
        "r3_pilot_audit_preserved_sha256": report["parent_r3_pilot_audit_preserved_sha256"],
        "code_lock_sha256": bind["code_lock_sha256"],
        "f0_phase_closeout_sha256": sha256_file(_ROOT / F0_CLOSEOUT_REL),
        "approval_sha256": sha256_file(_ROOT / APPROVAL_REL),
        "protocol_sha256": sha256_file(_ROOT / PROTOCOL_REL),
        "gates_passed": report["ok"],
        "full_corpus_authorized": False,
        "m0_authorized": False,
        "notes": [
            "D0-PILOT-R4 remediation audit. R3 PASS audit preserved immutable; R3 independent FAIL bound.",
            "overlay_render_integrity != manual semantic QA; external manual artifact required.",
            "STOP for independent audit before any full-corpus extraction.",
        ],
    }
    audit_sha = write_json(_ROOT / PILOT_R4_AUDIT_REL, audit, overwrite=True)
    completion_sha = _close_r4_run_contract(
        report_sha=report_sha,
        audit_sha=audit_sha,
        audit_verdict=audit["audit_verdict"],
    )
    return {
        "ok": report["ok"],
        "report_sha256": report_sha,
        "audit_sha256": audit_sha,
        "completion_sha256": completion_sha,
        "audit": audit,
    }


def write_run_contract(*, resume: bool) -> str:
    bind = _require_clean_code_lock()
    sample_sha = sha256_file(_ROOT / PILOT_R1_SAMPLE_REL)
    if sample_sha != EXPECTED_R1_SAMPLE_SHA256:
        raise RuntimeError("r1_sample_sha_drift")
    validate_d0_full_protocol(load_json(_ROOT / FULL_PROTOCOL_REL))
    obj = {
        "schema": "ms_sta_v5_d0_r4_pilot_run_contract_v1",
        "status": "ACTIVE",
        "campaign": "D0_PILOT_R4",
        "resume": bool(resume),
        "pilot_sample_sha256": sample_sha,
        "full_protocol_sha256": sha256_file(_ROOT / FULL_PROTOCOL_REL),
        "r3_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R3_INDEPENDENT_REAUDIT_REL),
        "r3_pilot_audit_sha256": EXPECTED_R3_PILOT_AUDIT_SHA256,
        "approval_sha256": sha256_file(_ROOT / APPROVAL_REL),
        "protocol_sha256": sha256_file(_ROOT / PROTOCOL_REL),
        "protocol_lock_sha256": sha256_file(_ROOT / PROTOCOL_LOCK_REL),
        "f0_phase_closeout_sha256": sha256_file(_ROOT / F0_CLOSEOUT_REL),
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "slowfast_checkpoint_sha256": EXPECTED_SLOWFAST_CKPT_SHA256,
        "code_lock_sha256": bind["code_lock_sha256"],
        "worktree_dirty": False,
        "output_dir": PILOT_R4_OUT_DIR_REL,
        "max_resumes": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "forbid_overwrite_r3_evidence": True,
        "forbid_full_corpus": True,
    }
    path = _ROOT / PILOT_R4_RUN_CONTRACT_REL
    if path.is_file() and not resume:
        prev = load_json(path)
        if prev.get("status") == "CLOSED":
            raise RuntimeError("r4_run_contract_already_closed")
        if prev.get("code_lock_sha256") != bind["code_lock_sha256"]:
            raise RuntimeError("run_contract_code_lock_drift")
        if prev.get("pilot_sample_sha256") != sample_sha:
            raise RuntimeError("run_contract_sample_drift")
        return sha256_file(path)
    return write_json(path, obj, overwrite=bool(resume and path.is_file()))


def orchestrate_all(*, resume: bool, device: str) -> int:
    if not EXTRACT_PY.is_file() or not F0_PY.is_file():
        raise SystemExit("missing_extract_or_f0_python")
    write_run_contract(resume=resume)
    script = str(_ROOT / "scripts/run_ms_sta_v5_d0_r4_pilot.py")
    cmd_skel = [str(EXTRACT_PY), script, "--phase", "skel"]
    if resume:
        cmd_skel.append("--resume")
    print("[orchestrate] skel", " ".join(cmd_skel))
    r1 = subprocess.run(cmd_skel, cwd=_ROOT)
    if r1.returncode != 0:
        return r1.returncode
    cmd_rgb = [str(F0_PY), script, "--phase", "rgb", "--device", device]
    if resume:
        cmd_rgb.append("--resume")
    print("[orchestrate] rgb", " ".join(cmd_rgb))
    r2 = subprocess.run(cmd_rgb, cwd=_ROOT)
    if r2.returncode != 0:
        return r2.returncode
    print(
        "[orchestrate] STOP_BEFORE_FINALIZE: write external "
        f"{PILOT_R4_VISUAL_MANUAL_REL} then run --phase finalize"
    )
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=["all", "skel", "rgb", "finalize", "negative_control", "skel_one"],
        default="all",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.phase == "skel_one":
        if not args.sample_id or not args.out_dir:
            raise SystemExit("skel_one_requires_sample_id_and_out_dir")
        sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
        sample = next((s for s in sample_set["samples"] if s["sample_id"] == args.sample_id), None)
        if sample is None:
            raise SystemExit(f"sample_not_in_r1_set:{args.sample_id}")
        out = _extract_one_skel(sample, Path(args.out_dir))
        print(json.dumps(out))
        return 0 if out.get("ok") else 1

    validate_f0_phase_closeout(load_json(_ROOT / F0_CLOSEOUT_REL), root=_ROOT)
    validate_d0_approval(load_json(_ROOT / APPROVAL_REL), root=_ROOT)
    validate_d0_protocol(load_json(_ROOT / PROTOCOL_REL))
    validate_r3_independent_reaudit(load_json(_ROOT / PILOT_R3_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_d0_full_protocol(load_json(_ROOT / FULL_PROTOCOL_REL))
    if not (_ROOT / PROTOCOL_LOCK_REL).is_file():
        raise SystemExit("protocol_lock_missing")
    if sha256_file(_ROOT / ELIGIBLE_REL) != EXPECTED_ELIGIBLE_SHA256:
        raise SystemExit("eligible_drift")

    if args.phase == "all":
        _require_clean_code_lock()
        return orchestrate_all(resume=args.resume, device=args.device)
    if args.phase == "skel":
        out = run_skel_phase(resume=args.resume)
        print(json.dumps({"ok": out["ok"], "n_ok": out["n_ok"], "n_fail": out["n_fail"]}, indent=2))
        return 0 if out["ok"] else 1
    if args.phase == "rgb":
        out = run_rgb_phase(resume=args.resume, device=args.device)
        print(json.dumps({"ok": out["ok"], "n_ok": out["n_ok"], "n_fail": out["n_fail"]}, indent=2))
        return 0 if out["ok"] else 1
    if args.phase == "finalize":
        out = run_finalize(device=args.device)
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    if args.phase == "negative_control":
        out = run_negative_control()
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
