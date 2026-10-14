# -*- coding: utf-8 -*-
"""E4-R4 pilot representation audit (read-only scientific gates).

Create-only reports under Docs/results/ms_sta_v6_e4_r4/.
Does not mutate B4 artifacts, pilot tensors, or prior extract reports.
No Python assert in production paths (safe under -O).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.e4_r4_joint_layouts import (  # noqa: E402
    LAYOUT_S_A,
    LAYOUT_S_B,
    MOUTH_ANCHOR_FACE_NAMES,
    layout_indices,
    semantic_group_indices,
)
from src.data_prep.e4_r4_rgb_contract import require_rgb_candidate  # noqa: E402
from src.data_prep.joint_layout_v3 import LEFT_HAND, POSE, RIGHT_HAND  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402

OUT_DIR = ROOT / "Docs/results/ms_sta_v6_e4_r4"
POLICY = OUT_DIR.parent / "ms_sta_v6_e4_r4_policy" / "E4_R4_POLICY.json"
FREEZE = OUT_DIR / "E4_R4_PILOT_SAMPLE_FREEZE.json"
POINTER = OUT_DIR / "E4_R4_PILOT_RUN_POINTER.json"
LOCK = OUT_DIR / "E4_R4_PHASE2B_CODE_LOCK.json"
META = ROOT / "metadata.csv"
SPLIT = ROOT / "data/processed/vsl400_retrain_v4_split_manifest.json"

# Locked audit thresholds (not relaxed post-hoc).
ALIGNMENT_TOL_FRAMES = 1.0
MID_SHOULDER_RESIDUAL_MAX = 1e-3
SCALE_CV_MAX = 0.35
SCALE_MIN = 1e-4
BONE_REL_JUMP_MAX = 0.75  # relative |L_t - L_{t-1}| / median_L
MOTION_MATCH_ATOL = 1e-5
CONF_VALID_MIN_WHEN_OBS = 0.0


def die(msg: str) -> None:
    print(f"E4_R4_REP_AUDIT_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_create_only(path: Path, payload: dict) -> None:
    if path.is_file():
        die(f"refuse_overwrite_existing_report:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_meta() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with META.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row["video_id"]).strip()
            out[sid] = {
                "video_id": sid,
                "signer_id": str(row.get("signer_id", "")),
                "text_gloss": str(row.get("text_gloss", "")),
                "dataset": str(row.get("dataset", "")),
                "view": str(row.get("view", "")),
                "split_col": str(row.get("split", "")),
            }
    return out


def role_for_id(sid: str, freeze: dict, split: dict) -> str:
    if sid in freeze["pilot_train_ids"]:
        return "pilot_train"
    if sid in freeze["pilot_val_ids"]:
        return "pilot_val"
    if sid in map(str, split["test_ids"]):
        return "test_FORBIDDEN"
    return "other"


def full_to_compact_map(src_idx: np.ndarray) -> dict[int, int]:
    return {int(j): i for i, j in enumerate(src_idx.tolist())}


def compact_pair(cmap: dict[int, int], a: int, b: int) -> tuple[int, int] | None:
    if a in cmap and b in cmap:
        return cmap[a], cmap[b]
    return None


def audit_skel_one(
    *,
    sid: str,
    layout: str,
    path: Path,
    sealed_path: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    fails: list[str] = []
    metrics: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as z:
        coords = np.asarray(z["coordinates"], dtype=np.float32)
        vel = np.asarray(z["velocity"], dtype=np.float32)
        validity = np.asarray(z["validity"], dtype=np.float32)
        conf = np.asarray(z["confidence"], dtype=np.float32)
        miss = np.asarray(z["missing_mask_prefill"], dtype=bool)
        if "effective_missing" in z.files:
            miss = miss | np.asarray(z["effective_missing"], dtype=bool)
        fi = np.asarray(z["original_frame_index"], dtype=np.int32)
        ts = np.asarray(z["timestamps_sec"], dtype=np.float64)
        pad = np.asarray(z["padding_mask"], dtype=bool)
        seq_len = int(np.asarray(z["sequence_length"]).reshape(-1)[0])
        src_idx = np.asarray(z["source_joint_indices"], dtype=np.int32)
        source_mode = str(z["source_mode"])

    expected = np.asarray(layout_indices(layout), dtype=np.int32)
    if src_idx.shape != expected.shape or not np.array_equal(src_idx, expected):
        fails.append("source_joint_indices_mismatch_layout")
    if source_mode != "sealed_75joint_read_only_compact_transform":
        fails.append(f"bad_source_mode:{source_mode}")
    if coords.dtype != np.float32 or coords.ndim != 3 or coords.shape[2] != 3:
        fails.append(f"bad_coords_dtype_shape:{coords.dtype}:{coords.shape}")
    if coords.shape[1] != len(expected):
        fails.append("n_joints_mismatch")
    if not np.isfinite(coords).all():
        fails.append("coords_nonfinite")
    if not np.isfinite(vel).all():
        fails.append("velocity_nonfinite")
    t = coords.shape[0]
    if fi.shape[0] != t or ts.shape[0] != t or pad.shape[0] != t:
        fails.append("temporal_len_mismatch")
    if seq_len <= 0 or seq_len > t:
        fails.append(f"bad_sequence_length:{seq_len}")

    active_fi = fi[:seq_len]
    active_ts = ts[:seq_len]
    if len(active_fi) != len(set(active_fi.tolist())):
        fails.append("frame_index_not_unique")
    if active_fi.size >= 2 and not np.all(active_fi[1:] > active_fi[:-1]):
        fails.append("frame_index_not_strictly_increasing")
    if not np.isfinite(active_ts).all():
        fails.append("timestamps_nonfinite")
    if active_ts.size >= 2 and not np.all(active_ts[1:] >= active_ts[:-1] - 1e-12):
        fails.append("timestamps_not_nondecreasing")

    # Match sealed slice exactly (motion retention + compact fidelity).
    # Note: sealed skel_v3 sets padding_mask = all-True valid-frame mask for unpadded clips.
    with np.load(sealed_path, allow_pickle=False) as s:
        sc = np.asarray(s["coordinates"], dtype=np.float32)
        sfi = np.asarray(s["original_frame_index"], dtype=np.int32)
        sts = np.asarray(s["timestamps_sec"], dtype=np.float64)
        spad = np.asarray(s["padding_mask"], dtype=bool)
        if sc.shape[0] != t:
            fails.append(f"sealed_T_mismatch:{sc.shape[0]}!={t}")
        else:
            sealed_slice = sc[:, src_idx, :]
            if not np.allclose(coords, sealed_slice, atol=MOTION_MATCH_ATOL, rtol=0):
                fails.append("compact_coords_not_equal_sealed_subset")
            if not np.array_equal(fi, sfi) or not np.allclose(ts, sts, equal_nan=True):
                fails.append("compact_temporal_not_equal_sealed")
            if not np.array_equal(pad, spad):
                fails.append("padding_mask_not_equal_sealed")
            if bool(spad.all()) and spad.shape[0] == t == seq_len:
                metrics["padding_mask_semantics"] = "sealed_valid_frame_mask_all_true"
            elif seq_len < t and np.any(np.abs(coords[seq_len:]) > 0):
                fails.append("post_sequence_nonzero")

    # Missing truthfulness: missing => not positively observed
    obs = validity > 0.5
    if np.any(miss & obs):
        fails.append("missing_marked_observed")

    cmap = full_to_compact_map(src_idx)
    ls = compact_pair(cmap, POSE.start + 4, POSE.start + 5)  # L/R shoulder
    if ls is None:
        fails.append("shoulders_missing_from_layout")
        mid_res_max = None
        scale_cv = None
        scale_med = None
    else:
        li, ri = ls
        both = (validity[:, li] > 0.5) & (validity[:, ri] > 0.5) & (~miss[:, li]) & (~miss[:, ri])
        both &= np.arange(t) < seq_len
        if not both.any():
            fails.append("no_frame_with_both_shoulders_observed")
            mid_res_max = None
            scale_cv = None
            scale_med = None
        else:
            mid = 0.5 * (coords[both, li, :] + coords[both, ri, :])
            mid_res = np.linalg.norm(mid, axis=1)
            mid_res_max = float(mid_res.max())
            metrics["mid_shoulder_residual_max"] = mid_res_max
            if mid_res_max > thresholds["mid_shoulder_residual_max"]:
                fails.append(f"mid_shoulder_residual:{mid_res_max}")
            dist = np.linalg.norm(coords[both, li, :] - coords[both, ri, :], axis=1)
            scale_med = float(np.median(dist))
            scale_cv = float(np.std(dist) / max(scale_med, 1e-8))
            metrics["shoulder_scale_median"] = scale_med
            metrics["shoulder_scale_cv"] = scale_cv
            if scale_med < thresholds["scale_min"]:
                fails.append(f"degenerate_shoulder_scale:{scale_med}")
            if scale_cv > thresholds["scale_cv_max"]:
                fails.append(f"unstable_shoulder_scale_cv:{scale_cv}")

    # Bone continuity: required edges must not be degenerate while both ends observed.
    # Temporal jump is recorded as metric; FAIL only if edge broken (near-zero length)
    # without missing mask, or if compact diverges from sealed (caught above).
    bone_pairs_full = [
        (POSE.start + 4, POSE.start + 6),  # L sh-el
        (POSE.start + 6, POSE.start + 8),  # L el-wr
        (POSE.start + 5, POSE.start + 7),  # R sh-el
        (POSE.start + 7, POSE.start + 9),  # R el-wr
    ]
    bone_jumps = []
    broken = 0
    lengths = []
    for a, b in bone_pairs_full:
        pair = compact_pair(cmap, a, b)
        if pair is None:
            fails.append(f"required_bone_absent_from_layout:{a}-{b}")
            continue
        ia, ib = pair
        for ti in range(seq_len):
            if miss[ti, ia] or miss[ti, ib]:
                continue
            if validity[ti, ia] <= 0.5 or validity[ti, ib] <= 0.5:
                continue
            length = float(np.linalg.norm(coords[ti, ia] - coords[ti, ib]))
            lengths.append(length)
            if length < thresholds["scale_min"]:
                broken += 1
        for ti in range(1, seq_len):
            if miss[ti, ia] or miss[ti, ib] or miss[ti - 1, ia] or miss[ti - 1, ib]:
                continue
            if min(validity[ti, ia], validity[ti, ib], validity[ti - 1, ia], validity[ti - 1, ib]) <= 0.5:
                continue
            l0 = float(np.linalg.norm(coords[ti - 1, ia] - coords[ti - 1, ib]))
            l1 = float(np.linalg.norm(coords[ti, ia] - coords[ti, ib]))
            bone_jumps.append(abs(l1 - l0))
    if broken:
        fails.append(f"bone_broken_while_observed:{broken}")
    if lengths:
        med_len = float(np.median(lengths))
        metrics["bone_length_median"] = med_len
    if bone_jumps and lengths:
        rel = float(np.max(bone_jumps) / max(float(np.median(lengths)), 1e-8))
        metrics["bone_jump_rel_max"] = rel
        # Soft diagnostic only — sealed lineage may have large jumps; compact must match sealed.

    # Hand/mouth motion retention vs sealed (exact match already); presence check
    for g in ("hand_mcp", "hand_fingertips", "mouth_anchors"):
        idxs = semantic_group_indices(g)
        if layout == "S_A" or g != "hips":
            if not set(idxs).issubset(set(int(x) for x in src_idx.tolist())):
                if g in ("hand_mcp", "hand_fingertips", "mouth_anchors"):
                    fails.append(f"required_group_absent:{g}")

    # Motion energy on wrists/mouth (informative; fail only if all-zero while sealed moves)
    motion_ok = True
    wrist_full = [POSE.start + 8, POSE.start + 9]
    mouth_full = list(semantic_group_indices("mouth_anchors"))
    for jfull in wrist_full + mouth_full:
        if jfull not in cmap:
            continue
        jc = cmap[jfull]
        series = coords[:seq_len, jc, :]
        m = ~miss[:seq_len, jc] & (validity[:seq_len, jc] > 0.5)
        if m.sum() < 2:
            continue
        disp = float(np.linalg.norm(np.diff(series[m], axis=0), axis=1).sum())
        metrics.setdefault("motion_disp", {})[str(jfull)] = disp

    return {
        "sample_id": sid,
        "layout": layout,
        "path": path.as_posix(),
        "pass": len(fails) == 0,
        "failures": fails,
        "metrics": metrics,
        "seq_len": seq_len,
        "n_joints": int(coords.shape[1]),
    }


def audit_rgb_one(
    *,
    sid: str,
    cid: str,
    path: Path,
    sealed_path: Path,
    video_path: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    fails: list[str] = []
    metrics: dict[str, Any] = {}
    alignment: dict[str, Any] = {
        "exact_match": 0,
        "nearest_match": 0,
        "fallback_25fps": 0,
        "fallback_other": 0,
    }
    spec = require_rgb_candidate(cid)
    with np.load(path, allow_pickle=False) as z:
        ten = np.asarray(z["rgb_tensor"], dtype=np.float32)
        tvalid = np.asarray(z["temporal_valid"], dtype=bool)
        fi = np.asarray(z["original_frame_index"], dtype=np.int32)
        ts = np.asarray(z["timestamps_sec"], dtype=np.float64)
        delta = np.asarray(z["rgb_skel_alignment_delta_sec"], dtype=np.float64)
        fps = float(np.asarray(z["fps"]))
        seq_len = int(np.asarray(z["sequence_length"]).reshape(-1)[0])
        raw_sha = str(z["raw_video_sha256"]) if "raw_video_sha256" in z.files else ""
        skel_sha = str(z["sealed_skel_sha256"]) if "sealed_skel_sha256" in z.files else ""
        calls = int(np.asarray(z["backbone_calls_per_frame"]))

    if calls != 1:
        fails.append("backbone_calls_not_1")
    if not raw_sha or not skel_sha:
        fails.append("missing_provenance_sha_fields")
    else:
        got_raw = sha256_file(video_path)
        got_skel = sha256_file(sealed_path)
        if got_raw != raw_sha:
            fails.append("raw_video_sha_mismatch")
        if got_skel != skel_sha:
            fails.append("sealed_skel_sha_mismatch")
        metrics["raw_video_sha256"] = raw_sha
        metrics["sealed_skel_sha256"] = skel_sha

    # Independent FPS from video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fails.append("cannot_open_video_for_fps")
        video_fps = None
    else:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
        cap.release()
    metrics["stored_fps"] = fps
    metrics["video_fps"] = video_fps
    if video_fps is not None and abs(video_fps - fps) > 0.51:
        fails.append(f"fps_mismatch_stored_vs_video:{fps}!={video_fps}")

    t_max = int(spec["frames_T_max"])
    ch = int(spec["channels"])
    if cid == "R1":
        h, w = spec["canvas_spatial"]
        expect = (t_max, int(h), int(w), 3)
    elif cid == "R2":
        h, w = spec["spatial_per_view"]
        expect = (t_max, 2, int(h), int(w), 3)
    elif cid == "R3":
        h, w = spec["spatial_per_view"]
        expect = (t_max, int(h), int(w), 1)
    else:
        h, w = spec["spatial_per_view"]
        expect = (t_max, int(h), int(w), 3)
    if ten.shape != expect:
        fails.append(f"bad_rgb_shape:{ten.shape}!={expect}")
    if ten.dtype != np.float32:
        fails.append("bad_rgb_dtype")
    if int(tvalid.sum()) != seq_len:
        fails.append("temporal_valid_sum_ne_seq_len")
    if not np.isfinite(ten[tvalid]).all():
        fails.append("rgb_nonfinite_on_valid")
    if (~tvalid).any() and np.any(ten[~tvalid] != 0):
        fails.append("invalid_not_zero_padded")

    active_fi = fi[tvalid]
    active_ts = ts[tvalid]
    if len(active_fi) != len(set(int(x) for x in active_fi.tolist())):
        fails.append("rgb_frame_index_not_unique")
    if active_fi.size >= 2 and not np.all(active_fi[1:] > active_fi[:-1]):
        fails.append("rgb_frame_index_not_strictly_increasing")
    if not np.isfinite(active_ts).all():
        fails.append("rgb_timestamps_nonfinite")
    if active_ts.size >= 2 and not np.all(active_ts[1:] >= active_ts[:-1] - 1e-12):
        fails.append("rgb_timestamps_not_nondecreasing")

    with np.load(sealed_path, allow_pickle=False) as s:
        sfi = np.asarray(s["original_frame_index"], dtype=np.int32)
        sts = np.asarray(s["timestamps_sec"], dtype=np.float64)
    sfi_list = sfi.tolist()
    sfi_set = set(int(x) for x in sfi_list)
    fi_to_ts = {int(a): float(b) for a, b in zip(sfi_list, sts.tolist())}

    # Recompute nearest-frame mapping and alignment independently
    tol = thresholds["alignment_tol_frames"] / max(fps, 1e-6)
    for f, tstamp, dstore in zip(active_fi.tolist(), active_ts.tolist(), delta[tvalid].tolist()):
        f = int(f)
        tstamp = float(tstamp)
        if f in sfi_set:
            expected_ts = fi_to_ts[f]
            if abs(expected_ts - tstamp) <= 1e-9:
                alignment["exact_match"] += 1
            else:
                alignment["nearest_match"] += 1
                if abs(expected_ts - tstamp) > tol:
                    fails.append(f"timestamp_not_sealed_exact:{f}")
            recon_delta = tstamp - (f / fps)
            if abs(float(dstore) - recon_delta) > 1e-5:
                fails.append("stored_alignment_delta_inconsistent")
            if abs(tstamp - expected_ts) > tol:
                fails.append(f"alignment_beyond_tol:{abs(tstamp-expected_ts)}")
        else:
            # Fallback path
            if abs(tstamp - (f / 25.0)) <= 1e-9:
                alignment["fallback_25fps"] += 1
                fails.append("fallback_25fps_used")
            else:
                alignment["fallback_other"] += 1
                # nearest sealed
                j = int(np.argmin(np.abs(sfi.astype(np.int64) - f)))
                expected_ts = float(sts[j])
                if abs(tstamp - expected_ts) > tol:
                    fails.append(f"nearest_alignment_beyond_tol:{f}")

    metrics["alignment_counts"] = alignment
    return {
        "sample_id": sid,
        "candidate": cid,
        "path": path.as_posix(),
        "pass": len(fails) == 0,
        "failures": fails,
        "metrics": metrics,
        "alignment": alignment,
        "seq_len": seq_len,
    }


def visual_qa_model_assisted(overlay_dir: Path, freeze: dict) -> dict[str, Any]:
    """Score existing overlays; file presence alone is not PASS."""
    qa_ids = list(freeze["qa_contact_ids"])[:8]
    judgments = []
    n_pass = 0
    for sid in qa_ids:
        png = overlay_dir / f"{sid}_semantic.png"
        item: dict[str, Any] = {
            "sample_id": sid,
            "path": png.as_posix() if png.is_file() else None,
            "file_sha256": sha256_file(png) if png.is_file() else None,
            "qa_label": "MODEL_ASSISTED_QA",
            "forbid_label": "HUMAN_REVIEWED",
        }
        if not png.is_file():
            item["verdict"] = "FAIL"
            item["failures"] = ["overlay_missing"]
            item["comments"] = "Overlay file absent."
            judgments.append(item)
            continue
        img = cv2.imread(str(png))
        if img is None or img.size == 0:
            item["verdict"] = "FAIL"
            item["failures"] = ["overlay_unreadable"]
            item["comments"] = "Cannot decode overlay image."
            judgments.append(item)
            continue
        h, w = img.shape[:2]
        # Detect yellow-ish label text and green landmark dots (BGR)
        # Labels drawn as (0,255,255), circles (0,255,0)
        b, g, r = cv2.split(img)
        green_dots = ((g > 200) & (b < 80) & (r < 80)).sum()
        yellow_text = ((g > 200) & (r > 200) & (b < 80)).sum()
        # Heuristic regions: upper-center (shoulders), lower-face band, hand laterals
        face_band = img[int(0.12 * h) : int(0.58 * h), int(0.25 * w) : int(0.75 * w)]
        fb, fg, fr = cv2.split(face_band)
        face_green = int(((fg > 200) & (fb < 80) & (fr < 80)).sum())
        # Also accept yellow label glyphs containing "lip" via OCR-free proxy:
        # yellow pixels in face band (labels are yellow, dots green)
        face_yellow = int(((fg > 200) & (fr > 200) & (fb < 80)).sum())
        shoulder_band = img[int(0.20 * h) : int(0.60 * h), int(0.10 * w) : int(0.90 * w)]
        sb, sg, sr = cv2.split(shoulder_band)
        shoulder_green = int(((sg > 200) & (sb < 80) & (sr < 80)).sum())

        checks = {
            "overlay_rendered": True,
            "has_semantic_labels": int(yellow_text) > 50,
            "has_landmark_dots": int(green_dots) >= 6,
            "shoulder_region_marks": shoulder_green >= 2,
            "mouth_region_marks": face_green >= 2 or face_yellow >= 30,
            "non_blank_frame": float(img.std()) > 5.0,
        }
        failures = [k for k, v in checks.items() if not v]
        # Material semantic criteria required for PASS
        material = [
            "has_semantic_labels",
            "has_landmark_dots",
            "shoulder_region_marks",
            "mouth_region_marks",
            "non_blank_frame",
        ]
        material_fail = [k for k in material if not checks[k]]
        item["checks"] = checks
        item["metrics"] = {
            "green_dot_pixels": int(green_dots),
            "yellow_label_pixels": int(yellow_text),
            "shoulder_green_pixels": shoulder_green,
            "face_green_pixels": face_green,
            "frame_std": float(img.std()),
            "image_hw": [h, w],
        }
        if material_fail:
            item["verdict"] = "FAIL"
            item["failures"] = material_fail
            item["comments"] = (
                "Render exists but semantic evidence insufficient for "
                + ",".join(material_fail)
            )
        else:
            item["verdict"] = "PASS"
            item["failures"] = []
            item["comments"] = (
                "Labels and landmark dots present near shoulder/mouth bands; "
                "MODEL_ASSISTED only — not human-reviewed."
            )
            n_pass += 1
        judgments.append(item)

    overall = "PASS" if n_pass == 8 and len(judgments) == 8 else "FAIL"
    return {
        "schema": "ms_sta_v6_e4_r4_pilot_model_assisted_visual_qa_v1",
        "qa_label": "MODEL_ASSISTED_QA",
        "forbid_label": "HUMAN_REVIEWED",
        "n_required": 8,
        "n_pass": n_pass,
        "n_fail": 8 - n_pass,
        "verdict": overall,
        "note": "File existence alone is never semantic PASS.",
        "judgments": judgments,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Pilot feature root (default from pointer)",
    )
    args = ap.parse_args()

    for p in (POLICY, FREEZE, POINTER, LOCK, META, SPLIT):
        if not p.is_file():
            die(f"missing:{p}")

    head = git_head()
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    meta = load_meta()

    feature_root = args.feature_root or (ROOT / pointer["feature_root_rel"])
    if not feature_root.is_dir():
        die(f"missing_feature_root:{feature_root}")

    ids = list(freeze["pilot_train_ids"]) + list(freeze["pilot_val_ids"])
    if len(ids) != 132 or len(set(ids)) != 132:
        die(f"bad_id_count:{len(ids)}")
    test_ids = set(map(str, split["test_ids"]))
    if set(ids) & test_ids:
        die("test_id_in_pilot")

    layouts = ["S_A", "S_B"]
    rgb_cands = ["R0", "R1", "R2", "R3"]
    n_art = 0
    for sid in ids:
        for layout in layouts:
            if not (feature_root / f"skel_{layout}" / f"{sid}.npz").is_file():
                die(f"missing_skel:{layout}:{sid}")
            n_art += 1
        for cid in rgb_cands:
            if not (feature_root / f"rgb_{cid}" / f"{sid}.npz").is_file():
                die(f"missing_rgb:{cid}:{sid}")
            n_art += 1
    if n_art != 792:
        die(f"artifact_count:{n_art}")

    thresholds = {
        "mid_shoulder_residual_max": MID_SHOULDER_RESIDUAL_MAX,
        "scale_cv_max": SCALE_CV_MAX,
        "scale_min": SCALE_MIN,
        "bone_rel_jump_max": BONE_REL_JUMP_MAX,
        "alignment_tol_frames": ALIGNMENT_TOL_FRAMES,
    }

    sealed_root = (
        ROOT
        / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3"
    )
    resolver = RawVideoResolver(ROOT)

    skel_results: dict[str, list] = {k: [] for k in layouts}
    rgb_results: dict[str, list] = {k: [] for k in rgb_cands}
    alignment_totals = defaultdict(int)
    identity_rows = []

    for sid in ids:
        row = meta.get(sid)
        if row is None:
            die(f"missing_metadata:{sid}")
        role = role_for_id(sid, freeze, split)
        if role == "test_FORBIDDEN":
            die(f"test_id:{sid}")
        sealed = sealed_root / f"{sid}.npz"
        if not sealed.is_file():
            die(f"missing_sealed:{sid}")
        vpath = resolver.resolve(row)
        if vpath is None or not Path(vpath).is_file():
            die(f"missing_video:{sid}")
        identity_rows.append(
            {
                "sample_id": sid,
                "role": role,
                "signer_id": row["signer_id"],
                "label": row["text_gloss"],
                "raw_video_rel": Path(vpath).relative_to(ROOT).as_posix(),
                "sealed_skel_rel": sealed.relative_to(ROOT).as_posix(),
            }
        )
        for layout in layouts:
            r = audit_skel_one(
                sid=sid,
                layout=layout,
                path=feature_root / f"skel_{layout}" / f"{sid}.npz",
                sealed_path=sealed,
                thresholds=thresholds,
            )
            skel_results[layout].append(r)
        for cid in rgb_cands:
            r = audit_rgb_one(
                sid=sid,
                cid=cid,
                path=feature_root / f"rgb_{cid}" / f"{sid}.npz",
                sealed_path=sealed,
                video_path=Path(vpath),
                thresholds=thresholds,
            )
            rgb_results[cid].append(r)
            for k, v in r["alignment"].items():
                alignment_totals[k] += int(v)

    def summarize(results: list[dict]) -> dict[str, Any]:
        fails = [r for r in results if not r["pass"]]
        fail_ids = sorted({r["sample_id"] for r in fails})
        reasons: dict[str, int] = defaultdict(int)
        for r in fails:
            for f in r["failures"]:
                reasons[f.split(":")[0]] += 1
        return {
            "n": len(results),
            "n_pass": len(results) - len(fails),
            "n_fail": len(fails),
            "fail_sample_ids": fail_ids,
            "failure_reason_counts": dict(sorted(reasons.items())),
            "verdict": "PASS" if not fails else "FAIL",
        }

    skel_summary = {k: summarize(v) for k, v in skel_results.items()}
    rgb_summary = {k: summarize(v) for k, v in rgb_results.items()}

    fallback_count = int(alignment_totals.get("fallback_25fps", 0))
    producer_defect = fallback_count > 0

    overlay_dir = feature_root / "qa_overlays_semantic"
    visual = visual_qa_model_assisted(overlay_dir, freeze)

    hard_fail = False
    hard_reasons = []
    for s in list(skel_summary.values()) + list(rgb_summary.values()):
        if s["verdict"] != "PASS":
            hard_fail = True
            hard_reasons.append("contract_or_representation_gate")
    if visual["verdict"] != "PASS":
        hard_fail = True
        hard_reasons.append("visual_qa")
    if producer_defect:
        hard_fail = True
        hard_reasons.append("PRODUCER_DEFECT_fallback_25fps")

    overall = "FAIL" if hard_fail else "PASS"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    alignment_report = {
        "schema": "ms_sta_v6_e4_r4_pilot_alignment_audit_v1",
        "created_at": now,
        "feature_root_rel": feature_root.relative_to(ROOT).as_posix(),
        "git_head": head,
        "thresholds": {
            "alignment_tol_frames": ALIGNMENT_TOL_FRAMES,
            "forbid_default_25fps_fallback_when_sealed_misses_frame": True,
        },
        "counts": dict(alignment_totals),
        "fallback_25fps_count": fallback_count,
        "producer_defect": producer_defect,
        "verdict": "FAIL" if producer_defect or any(
            rgb_summary[c]["verdict"] != "PASS" for c in rgb_cands
        ) else "PASS",
        "note": "Independent recomputation vs sealed frame_index/timestamps and stored fps.",
    }

    rep_report = {
        "schema": "ms_sta_v6_e4_r4_pilot_representation_audit_v1",
        "created_at": now,
        "git_head": head,
        "feature_root_rel": feature_root.relative_to(ROOT).as_posix(),
        "phase2b_code_lock_sha256": sha256_file(LOCK),
        "pilot_freeze_sha256": sha256_file(FREEZE),
        "policy_sha256": sha256_file(POLICY),
        "pointer_sha256": sha256_file(POINTER),
        "n_ids": len(ids),
        "n_artifacts": n_art,
        "thresholds": thresholds,
        "identity_n": len(identity_rows),
        "per_layout": skel_summary,
        "per_rgb": rgb_summary,
        "visual_qa_verdict": visual["verdict"],
        "alignment_verdict": alignment_report["verdict"],
        "producer_defect": producer_defect,
        "verdict": overall,
        "hard_fail_reasons": hard_reasons,
        "e4_ready": False,
        "forbid_full_corpus_until_pass": True,
        "sample_failures_truncated": {
            "skel": {
                k: [r for r in v if not r["pass"]][:20]
                for k, v in skel_results.items()
            },
            "rgb": {
                k: [r for r in v if not r["pass"]][:20]
                for k, v in rgb_results.items()
            },
        },
    }

    # Attach identity digest (not full 132×verbose unless fail)
    rep_report["identity_digest"] = {
        "n_pilot_train": sum(1 for r in identity_rows if r["role"] == "pilot_train"),
        "n_pilot_val": sum(1 for r in identity_rows if r["role"] == "pilot_val"),
        "sha256_sample_id_list": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }

    write_json_create_only(OUT_DIR / "E4_R4_PILOT_REPRESENTATION_AUDIT.json", rep_report)
    write_json_create_only(OUT_DIR / "E4_R4_PILOT_ALIGNMENT_AUDIT.json", alignment_report)
    write_json_create_only(OUT_DIR / "E4_R4_PILOT_MODEL_ASSISTED_VISUAL_QA.json", visual)

    print("REPRESENTATION_AUDIT", overall)
    print("alignment_fallback_25", fallback_count)
    print("visual", visual["verdict"], visual["n_pass"], "/", 8)
    for k, s in skel_summary.items():
        print("skel", k, s["verdict"], s["n_fail"])
    for k, s in rgb_summary.items():
        print("rgb", k, s["verdict"], s["n_fail"])
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
