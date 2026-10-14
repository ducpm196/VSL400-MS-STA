"""E4-R4 RGB packing — R0/R1 mosaic/R2 budget/R3 luma (one backbone call).

Uses sealed skeleton landmarks for ROI geometry + raw front video frames.
No Python assert (safe under python -O).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.data_prep.e4_r4_rgb_contract import (
    luma_from_rgb_uint8,
    require_rgb_candidate,
    theoretical_work_units,
)
from src.data_prep.joint_layout_v3 import FACE, LEFT_HAND, POSE, RIGHT_HAND

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _die(msg: str) -> None:
    raise RuntimeError(msg)


def letterbox_rgb(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        _die(f"bad_frame_shape:{frame_rgb.shape}")
    h, w = frame_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def _norm_xy_to_px(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = xy[..., 0] * float(w)
    out[..., 1] = xy[..., 1] * float(h)
    return out


def _bbox_from_xy(
    xy_px: np.ndarray,
    valid: np.ndarray,
    *,
    w: int,
    h: int,
    margin: float = 0.15,
) -> tuple[int, int, int, int] | None:
    pts = xy_px[valid]
    if pts.shape[0] < 2:
        return None
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0 -= bw * margin
    x1 += bw * margin
    y0 -= bh * margin
    y1 += bh * margin
    x0 = int(max(0, np.floor(x0)))
    y0 = int(max(0, np.floor(y0)))
    x1 = int(min(w, np.ceil(x1)))
    y1 = int(min(h, np.ceil(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _crop_resize(frame: np.ndarray, box: tuple[int, int, int, int] | None, size: int) -> tuple[np.ndarray, bool]:
    if box is None:
        return np.zeros((size, size, 3), dtype=np.uint8), False
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8), False
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR), True


def _upper_body_box(xy_px: np.ndarray, valid: np.ndarray, w: int, h: int) -> tuple[int, int, int, int] | None:
    # shoulders, elbows, wrists, hips (pose block locals in sealed 75 layout)
    pose_names = (4, 5, 6, 7, 8, 9, 10, 11)  # local pose indices inside POSE slice
    idxs = [POSE.start + i for i in pose_names]
    mask = np.zeros(xy_px.shape[0], dtype=bool)
    for i in idxs:
        if 0 <= i < valid.shape[0]:
            mask[i] = bool(valid[i])
    return _bbox_from_xy(xy_px, mask, w=w, h=h, margin=0.2)


def _signing_composite_box(xy_px: np.ndarray, valid: np.ndarray, w: int, h: int) -> tuple[int, int, int, int] | None:
    mask = np.zeros(xy_px.shape[0], dtype=bool)
    for sl in (POSE, LEFT_HAND, RIGHT_HAND, FACE):
        mask[sl] = valid[sl]
    return _bbox_from_xy(xy_px, mask, w=w, h=h, margin=0.12)


def sample_frame_indices(n_native: int, t_max: int, sealed_frame_idx: np.ndarray) -> np.ndarray:
    """Prefer sealed skeleton frame indices; subsample to t_max if needed."""
    idx = np.asarray(sealed_frame_idx, dtype=np.int32)
    idx = idx[(idx >= 0) & (idx < n_native)]
    if idx.size == 0:
        if n_native <= 0:
            _die("empty_video")
        if n_native <= t_max:
            return np.arange(n_native, dtype=np.int32)
        return np.linspace(0, n_native - 1, t_max).astype(np.int32)
    if idx.size <= t_max:
        return idx
    sel = np.linspace(0, idx.size - 1, t_max).astype(np.int32)
    return idx[sel]


def read_selected_frames(video_path: str, frame_indices: np.ndarray) -> tuple[np.ndarray, float]:
    """Read RGB uint8 frames at exact indices; returns (T,H,W,3), fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot_open_video:{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    want = set(int(i) for i in frame_indices.tolist())
    got: dict[int, np.ndarray] = {}
    i = 0
    while want and i <= max(want):
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            want.discard(i)
        i += 1
    cap.release()
    missing = [int(x) for x in frame_indices.tolist() if int(x) not in got]
    if missing:
        raise RuntimeError(f"missing_video_frames:{missing[:8]}")
    frames = np.stack([got[int(x)] for x in frame_indices.tolist()], axis=0)
    return frames, fps


def normalize_rgb_u8(frames: np.ndarray) -> np.ndarray:
    x = frames.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def pack_rgb_candidate(
    *,
    candidate_id: str,
    frames_rgb: np.ndarray,
    landmarks_xy: np.ndarray,
    missing_mask: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    """Pack one RGB candidate tensor payload.

    frames_rgb: (T,H,W,3) uint8
    landmarks_xy: (T,75,2) normalized image coords from sealed skeleton
    missing_mask: (T,75) bool True=missing
    """
    spec = require_rgb_candidate(candidate_id)
    if theoretical_work_units(candidate_id) > theoretical_work_units("R0") and candidate_id != "R3":
        _die(f"non_compact_candidate:{candidate_id}")

    t = int(frames_rgb.shape[0])
    if landmarks_xy.shape[0] != t or missing_mask.shape[0] != t:
        _die("rgb_skel_T_mismatch")
    if len(frame_indices) != t or len(timestamps_sec) != t:
        _die("index_ts_len_mismatch")

    h0, w0 = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    valid = ~np.asarray(missing_mask, dtype=bool)

    if candidate_id == "R0":
        size = int(spec["spatial_per_view"][0])
        canvases = np.stack([letterbox_rgb(frames_rgb[i], size) for i in range(t)], axis=0)
        tensor = normalize_rgb_u8(canvases)  # T,H,W,3
        tile_valid = np.ones((t, 1), dtype=bool)
        views = ("global",)
        canvas_shape = (size, size)

    elif candidate_id == "R1":
        tile = int(spec["spatial_per_view"][0])
        ch, cw = spec["canvas_spatial"]
        mosaics = np.zeros((t, int(ch), int(cw), 3), dtype=np.uint8)
        tile_valid = np.zeros((t, 4), dtype=bool)
        for i in range(t):
            frame = frames_rgb[i]
            xy = _norm_xy_to_px(landmarks_xy[i], w0, h0)
            v = valid[i]
            boxes = {
                "global": (0, 0, w0, h0),
                "upper_body": _upper_body_box(xy, v, w0, h0),
                "left_hand": _bbox_from_xy(xy, v.copy() & _mask_slice(v, LEFT_HAND), w=w0, h=h0),
                "right_hand": _bbox_from_xy(xy, v.copy() & _mask_slice(v, RIGHT_HAND), w=w0, h=h0),
            }
            # Force global always present
            tiles = []
            for j, name in enumerate(("global", "upper_body", "left_hand", "right_hand")):
                crop, ok = _crop_resize(frame, boxes[name], tile)
                tiles.append(crop)
                tile_valid[i, j] = ok if name != "global" else True
            mosaics[i, 0:tile, 0:tile] = tiles[0]
            mosaics[i, 0:tile, tile : 2 * tile] = tiles[1]
            mosaics[i, tile : 2 * tile, 0:tile] = tiles[2]
            mosaics[i, tile : 2 * tile, tile : 2 * tile] = tiles[3]
        tensor = normalize_rgb_u8(mosaics)
        views = ("global", "upper_body", "left_hand", "right_hand")
        canvas_shape = (int(ch), int(cw))

    elif candidate_id == "R2":
        size = int(spec["spatial_per_view"][0])
        out = np.zeros((t, 2, size, size, 3), dtype=np.uint8)
        tile_valid = np.zeros((t, 2), dtype=bool)
        for i in range(t):
            frame = frames_rgb[i]
            xy = _norm_xy_to_px(landmarks_xy[i], w0, h0)
            v = valid[i]
            g = letterbox_rgb(frame, size)
            sbox = _signing_composite_box(xy, v, w0, h0)
            scrop, ok = _crop_resize(frame, sbox, size)
            out[i, 0] = g
            out[i, 1] = scrop
            tile_valid[i, 0] = True
            tile_valid[i, 1] = ok
        # Stack views as batch dimension for one shared backbone forward (measured later)
        tensor = normalize_rgb_u8(out.reshape(t * 2, size, size, 3)).reshape(t, 2, size, size, 3)
        views = ("global", "signing_composite")
        canvas_shape = (size, size)

    elif candidate_id == "R3":
        size = int(spec["spatial_per_view"][0])
        canvases = np.stack([letterbox_rgb(frames_rgb[i], size) for i in range(t)], axis=0)
        luma = np.stack([luma_from_rgb_uint8(canvases[i]) for i in range(t)], axis=0)
        # luma float 0..255 → normalize single channel
        tensor = (luma / 255.0 - 0.5) / 0.5  # T,H,W,1
        if tensor.shape[-1] != 1:
            _die("r3_must_remain_1ch")
        tile_valid = np.ones((t, 1), dtype=bool)
        views = ("global",)
        canvas_shape = (size, size)
    else:
        _die(f"unknown_candidate:{candidate_id}")

    # Pad to T_max with zeros + validity
    t_max = int(spec["frames_T_max"])
    pad_n = max(0, t_max - t)
    temporal_valid = np.zeros((t_max,), dtype=bool)
    temporal_valid[:t] = True
    if pad_n:
        if candidate_id == "R2":
            pad = np.zeros((pad_n, 2, canvas_shape[0], canvas_shape[1], tensor.shape[-1]), dtype=np.float32)
            tensor = np.concatenate([tensor.astype(np.float32), pad], axis=0)
            tile_valid = np.concatenate([tile_valid, np.zeros((pad_n, tile_valid.shape[1]), dtype=bool)], axis=0)
        else:
            pad = np.zeros((pad_n, canvas_shape[0], canvas_shape[1], tensor.shape[-1]), dtype=np.float32)
            tensor = np.concatenate([tensor.astype(np.float32), pad], axis=0)
            tile_valid = np.concatenate([tile_valid, np.zeros((pad_n, tile_valid.shape[1]), dtype=bool)], axis=0)
        fi = np.concatenate([frame_indices.astype(np.int32), np.full(pad_n, -1, dtype=np.int32)])
        ts = np.concatenate([timestamps_sec.astype(np.float64), np.full(pad_n, np.nan, dtype=np.float64)])
    else:
        fi = frame_indices.astype(np.int32)
        ts = timestamps_sec.astype(np.float64)
        tensor = tensor.astype(np.float32)

    if np.isnan(tensor[temporal_valid]).any() or np.isinf(tensor[temporal_valid]).any():
        _die("rgb_tensor_nan_inf")

    skel_delta = np.zeros((t_max,), dtype=np.float64)
    # alignment delta: video timestamp vs sealed timestamp for valid frames
    for i in range(t):
        skel_delta[i] = float(ts[i]) - (float(fi[i]) / float(fps) if fi[i] >= 0 else np.nan)

    return {
        "candidate_id": np.asarray(candidate_id),
        "rgb_tensor": tensor,
        "temporal_valid": temporal_valid,
        "tile_or_view_valid": tile_valid,
        "original_frame_index": fi,
        "timestamps_sec": ts,
        "rgb_skel_alignment_delta_sec": skel_delta.astype(np.float64),
        "views": np.asarray(views),
        "backbone_calls_per_frame": np.asarray(1, dtype=np.int32),
        "canvas_spatial": np.asarray(canvas_shape, dtype=np.int32),
        "channels": np.asarray(int(tensor.shape[-1]), dtype=np.int32),
        "sequence_length": np.asarray(t, dtype=np.int32),
        "fps": np.asarray(fps, dtype=np.float64),
        "crop_construction": np.asarray(str(spec["crop_construction"])),
        "resize_interpolation": np.asarray("bilinear"),
        "bgr_to_rgb": np.asarray(True),
        "normalization": np.asarray(str(spec["normalization"])),
        "missing_roi_policy": np.asarray(str(spec["missing_roi_policy"])),
        "source_mode": np.asarray("raw_video_plus_sealed_skel_roi"),
    }


def _mask_slice(valid: np.ndarray, sl: slice) -> np.ndarray:
    m = np.zeros_like(valid, dtype=bool)
    m[sl] = True
    return m


def write_rgb_npz_atomic(path, arrays: dict[str, Any]) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.writing.npz"
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    with np.load(tmp, allow_pickle=False) as z:
        if "rgb_tensor" not in z.files:
            raise RuntimeError("tmp_missing_rgb_tensor")
        ten = z["rgb_tensor"]
        mask = np.asarray(z["temporal_valid"], dtype=bool)
        valid_ten = ten[mask]
        if np.isnan(valid_ten).any() or np.isinf(valid_ten).any():
            raise RuntimeError("tmp_rgb_nan_inf")
    tmp.replace(path)
