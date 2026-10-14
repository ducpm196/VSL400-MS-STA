"""Frozen SlowFast block-4 feature maps → compact ROI tokens (RGB token v3).

Requires sealed local Kinetics checkpoint (no Torch-Hub pretrained download).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.data_prep.joint_layout_v3 import FACE, LEFT_HAND, N_JOINTS, RIGHT_HAND

KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)
INPUT_SIZE = 224
WINDOW_SIZE = 32
WINDOW_STRIDE = 16
ROI_OUT = 3
REGIONS = ("global", "left_hand", "right_hand", "face")

SEALED_SLOWFAST_REL = (
    "Docs/results/ms_sta_v5_f0/weights/slowfast_r50_partial_unfreeze/SLOWFAST_8x8_R50.pyth"
)
SEALED_SLOWFAST_SHA256 = (
    "454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_state_dict(state: Any) -> dict[str, Any]:
    if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
        return state["model_state"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise RuntimeError("checkpoint_state_dict_invalid")


def window_starts(t_native: int, window_size: int = WINDOW_SIZE, window_stride: int = WINDOW_STRIDE) -> list[int]:
    """Stride starts plus an end-anchored window covering the last native frame."""
    if t_native < 1:
        raise ValueError("empty_clip")
    if t_native <= window_size:
        return [0]
    starts = list(range(0, t_native - window_size + 1, window_stride))
    end_anchored = t_native - window_size
    if end_anchored not in starts:
        starts.append(end_anchored)
    return sorted(starts)


@dataclass(frozen=True)
class PathwayMaps:
    slow: torch.Tensor
    fast: torch.Tensor


class FrozenSlowFastBlock4(nn.Module):
    """Frozen SlowFast-R50 truncated after parameterized block 4 (before pooling)."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        expected_sha256: str = SEALED_SLOWFAST_SHA256,
        device: torch.device | str = "cuda",
        repo_root: Path | None = None,
    ) -> None:
        super().__init__()
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        ckpt = Path(checkpoint_path) if checkpoint_path is not None else root / SEALED_SLOWFAST_REL
        # Fail closed on network weight fetch.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TORCH_HOME"] = str(root / "_torch_cache_unused_d0")

        if not ckpt.is_file():
            raise FileNotFoundError(f"slowfast_checkpoint_missing:{ckpt}")
        digest = sha256_file(ckpt)
        if digest != expected_sha256:
            raise RuntimeError(f"slowfast_checkpoint_sha_mismatch:{digest}")

        from pytorchvideo.models.hub import slowfast_r50

        self.device = torch.device(device)
        full = slowfast_r50(pretrained=False)
        blob = torch.load(str(ckpt), map_location="cpu")
        state = _extract_state_dict(blob)
        missing, unexpected = full.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"slowfast_strict_load_failed:missing={list(missing)}:unexpected={list(unexpected)}"
            )
        for p in full.parameters():
            p.requires_grad = False
        self.blocks = nn.ModuleList(list(full.blocks.children())[:5])  # 0..4
        for p in self.parameters():
            p.requires_grad = False
        self.to(self.device)
        self.eval()
        self.checkpoint_path = str(ckpt).replace("\\", "/")
        self.checkpoint_sha256 = digest
        self.load_meta = {
            "pretrained": False,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": digest,
            "missing_keys": [],
            "unexpected_keys": [],
            "all_requires_grad_false": True,
            "hook": "blocks.4_before_blocks.5_pooling",
            "network_disabled": True,
        }

    @torch.no_grad()
    def forward_maps(self, slow: torch.Tensor, fast: torch.Tensor) -> PathwayMaps:
        x: Any = [slow.to(self.device), fast.to(self.device)]
        for block in self.blocks:
            x = block(x)
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise RuntimeError(f"unexpected_block4_output:{type(x)}")
        return PathwayMaps(slow=x[0], fast=x[1])


def read_video_frames_native(video_path: str | Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (frames RGB uint8 HxWx3, original_frame_index, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot_open_video:{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 25.0
    frames: list[np.ndarray] = []
    indices: list[int] = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        indices.append(idx)
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"empty_video:{video_path}")
    return np.stack(frames, axis=0), np.asarray(indices, dtype=np.int32), fps


def _resize_letterbox(frame: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, dict[str, float]]:
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    meta = {
        "scale": float(scale),
        "pad_top": float(top),
        "pad_left": float(left),
        "src_h": float(h),
        "src_w": float(w),
        "dst": float(size),
    }
    return canvas, meta


def landmark_xy_to_resized(xy: np.ndarray, meta: dict[str, float]) -> np.ndarray:
    x = xy[..., 0] * meta["src_w"] * meta["scale"] + meta["pad_left"]
    y = xy[..., 1] * meta["src_h"] * meta["scale"] + meta["pad_top"]
    return np.stack([x, y], axis=-1)


def _normalize_clip(frames_u8: np.ndarray) -> np.ndarray:
    x = frames_u8.astype(np.float32) / 255.0
    mean = np.array(KINETICS_MEAN, dtype=np.float32)
    std = np.array(KINETICS_STD, dtype=np.float32)
    return (x - mean) / std


def window_tensors(
    frames_u8: np.ndarray, start: int, win: int = WINDOW_SIZE
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    """Build slow/fast tensors; return valid_frame_mask (win,) and n_valid."""
    t_native = frames_u8.shape[0]
    clip = frames_u8[start : start + win]
    n_valid = int(clip.shape[0])
    valid = np.zeros((win,), dtype=bool)
    valid[:n_valid] = True
    if n_valid < win:
        pad = np.repeat(clip[-1:], win - n_valid, axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    clip_n = _normalize_clip(clip)
    fast = torch.from_numpy(clip_n.transpose(3, 0, 1, 2)).unsqueeze(0).float()
    slow_idx = np.linspace(0, win - 1, 8).astype(int)
    slow = torch.from_numpy(clip_n[slow_idx].transpose(3, 0, 1, 2)).unsqueeze(0).float()
    return slow, fast, valid, n_valid


def _bbox_from_points(points: np.ndarray, valid: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray | None:
    pts = points[valid]
    if pts.shape[0] < 2:
        return None
    x0 = float(np.clip(pts[:, 0].min() - 8.0, 0, size - 1))
    y0 = float(np.clip(pts[:, 1].min() - 8.0, 0, size - 1))
    x1 = float(np.clip(pts[:, 0].max() + 8.0, 1, size))
    y1 = float(np.clip(pts[:, 1].max() + 8.0, 1, size))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def nearest_skel_landmark(
    *,
    center_frame: int,
    fps: float,
    skel_frame_index: np.ndarray,
    skel_timestamps: np.ndarray,
    tie_break: str = "prefer_earlier_skel_index",
) -> dict[str, Any]:
    """Map RGB center frame to nearest Skeleton observation by timestamp."""
    if skel_frame_index.size < 1 or skel_timestamps.size < 1:
        raise RuntimeError("empty_skeleton_timeline")
    if skel_frame_index.shape[0] != skel_timestamps.shape[0]:
        raise RuntimeError("skel_frame_timestamp_len_mismatch")
    if tie_break != "prefer_earlier_skel_index":
        raise RuntimeError(f"unsupported_tie_break:{tie_break}")
    target_ts = float(center_frame) / float(fps)
    # Canonical float64 timeline from native frame index (avoids float32 stamp noise).
    skel_ts64 = skel_frame_index.astype(np.float64) / float(fps)
    deltas = np.abs(skel_ts64 - target_ts)
    min_d = float(deltas.min())
    # Treat near-equal nearest distances as ties (covers residual float noise).
    tie_atol = max(1e-12, (0.5 / float(fps)) * 1e-6)
    candidates = np.flatnonzero(np.abs(deltas - min_d) <= tie_atol)
    idx = int(candidates[0])
    return {
        "landmark_index": idx,
        "landmark_frame": int(skel_frame_index[idx]),
        "landmark_timestamp_sec": float(skel_ts64[idx]),
        "center_frame": int(center_frame),
        "center_timestamp_sec": float(target_ts),
        "alignment_delta_sec": float(deltas[idx]),
        "alignment_method": "nearest_timestamp",
        "alignment_tie_break": tie_break,
        "alignment_tie_n": int(candidates.size),
        "alignment_timestamp_dtype": "float64",
    }


def region_boxes_from_landmarks(
    landmarks_xy_resized: np.ndarray,
    missing_mask: np.ndarray,
) -> dict[str, np.ndarray | None]:
    if landmarks_xy_resized.shape[0] != N_JOINTS or missing_mask.shape[0] != N_JOINTS:
        raise ValueError(f"bad_landmark_len:{landmarks_xy_resized.shape}:{missing_mask.shape}")
    global_box = np.asarray([0.0, 0.0, float(INPUT_SIZE), float(INPUT_SIZE)], dtype=np.float32)
    face = _bbox_from_points(landmarks_xy_resized[FACE], ~missing_mask[FACE])
    lh = _bbox_from_points(landmarks_xy_resized[LEFT_HAND], ~missing_mask[LEFT_HAND])
    rh = _bbox_from_points(landmarks_xy_resized[RIGHT_HAND], ~missing_mask[RIGHT_HAND])
    return {"global": global_box, "left_hand": lh, "right_hand": rh, "face": face}


# Selected-pose indices inside the 75-joint layout (SignLanguageDataEngine pose_indices order).
_POSE_L_SHOULDER = 4
_POSE_R_SHOULDER = 5
_POSE_L_ELBOW = 6
_POSE_R_ELBOW = 7
_POSE_L_WRIST = 8
_POSE_R_WRIST = 9


def pose_wrist_prior_box(
    landmarks_xy_resized: np.ndarray,
    missing_mask: np.ndarray,
    *,
    which: str,
    crop_scale: float = 1.0,
    min_side: float = 24.0,
    max_area_fraction: float = 0.08,
) -> tuple[np.ndarray | None, str | None]:
    """Legacy R7B single-wrist square prior (kept for unit tests / fixture)."""
    if which not in {"left_hand", "right_hand"}:
        raise ValueError(f"bad_which:{which}")
    wi = _POSE_L_WRIST if which == "left_hand" else _POSE_R_WRIST
    ei = _POSE_L_ELBOW if which == "left_hand" else _POSE_R_ELBOW
    if bool(missing_mask[wi]):
        return None, "pose_wrist_missing"
    if bool(missing_mask[_POSE_L_SHOULDER]) or bool(missing_mask[_POSE_R_SHOULDER]):
        shoulder_w = float(INPUT_SIZE) * 0.22
    else:
        shoulder_w = float(
            np.linalg.norm(
                landmarks_xy_resized[_POSE_L_SHOULDER, :2] - landmarks_xy_resized[_POSE_R_SHOULDER, :2]
            )
        )
        shoulder_w = max(shoulder_w, float(INPUT_SIZE) * 0.12)
    half = max(min_side / 2.0, crop_scale * 0.5 * shoulder_w)
    max_half = 0.5 * float(INPUT_SIZE) * float(np.sqrt(max_area_fraction))
    if half > max_half:
        half = max_half
    cx, cy = float(landmarks_xy_resized[wi, 0]), float(landmarks_xy_resized[wi, 1])
    if not bool(missing_mask[ei]):
        ex, ey = float(landmarks_xy_resized[ei, 0]), float(landmarks_xy_resized[ei, 1])
        vx, vy = cx - ex, cy - ey
        n = max(float(np.hypot(vx, vy)), 1.0)
        cx = cx + 0.15 * (vx / n) * shoulder_w
        cy = cy + 0.15 * (vy / n) * shoulder_w
    x0 = float(np.clip(cx - half, 0.0, float(INPUT_SIZE - 1)))
    y0 = float(np.clip(cy - half, 0.0, float(INPUT_SIZE - 1)))
    x1 = float(np.clip(cx + half, 1.0, float(INPUT_SIZE)))
    y1 = float(np.clip(cy + half, 1.0, float(INPUT_SIZE)))
    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return None, "pose_prior_crop_too_small"
    area_frac = ((x1 - x0) * (y1 - y0)) / float(INPUT_SIZE * INPUT_SIZE)
    if area_frac > max_area_fraction + 1e-6:
        return None, "pose_prior_area_exceeds_lock"
    return np.asarray([x0, y0, x1, y1], dtype=np.float32), None


def _shoulder_width(landmarks_xy_resized: np.ndarray, missing_mask: np.ndarray) -> float:
    if bool(missing_mask[_POSE_L_SHOULDER]) or bool(missing_mask[_POSE_R_SHOULDER]):
        return float(INPUT_SIZE) * 0.22
    shoulder_w = float(
        np.linalg.norm(
            landmarks_xy_resized[_POSE_L_SHOULDER, :2] - landmarks_xy_resized[_POSE_R_SHOULDER, :2]
        )
    )
    return max(shoulder_w, float(INPUT_SIZE) * 0.12)


def classify_pose_prior_mode(
    landmarks_xy_resized: np.ndarray,
    missing_mask: np.ndarray,
    *,
    bimanual_wrist_distance_frac_of_shoulder: float = 0.55,
) -> str:
    """Return 'bimanual' when both wrists present and close; else 'single'."""
    lw_m = bool(missing_mask[_POSE_L_WRIST])
    rw_m = bool(missing_mask[_POSE_R_WRIST])
    if lw_m or rw_m:
        return "single"
    shoulder_w = _shoulder_width(landmarks_xy_resized, missing_mask)
    dist = float(
        np.linalg.norm(
            landmarks_xy_resized[_POSE_L_WRIST, :2] - landmarks_xy_resized[_POSE_R_WRIST, :2]
        )
    )
    if dist <= bimanual_wrist_distance_frac_of_shoulder * shoulder_w:
        return "bimanual"
    return "single"


def single_hand_pose_prior_box(
    landmarks_xy_resized: np.ndarray,
    missing_mask: np.ndarray,
    *,
    which: str,
    crop_scale: float = 0.85,
    min_side: float = 24.0,
    max_area_fraction: float = 0.06,
    max_center_offset_frac_of_half: float = 0.35,
) -> tuple[np.ndarray | None, str | None]:
    """Elbow→wrist directed prior; reject if contralateral wrist falls inside box."""
    if which not in {"left_hand", "right_hand"}:
        raise ValueError(f"bad_which:{which}")
    wi = _POSE_L_WRIST if which == "left_hand" else _POSE_R_WRIST
    ei = _POSE_L_ELBOW if which == "left_hand" else _POSE_R_ELBOW
    contra = _POSE_R_WRIST if which == "left_hand" else _POSE_L_WRIST
    if bool(missing_mask[wi]):
        return None, "pose_wrist_missing"
    shoulder_w = _shoulder_width(landmarks_xy_resized, missing_mask)
    half = max(min_side / 2.0, crop_scale * 0.5 * shoulder_w)
    max_half = 0.5 * float(INPUT_SIZE) * float(np.sqrt(max_area_fraction))
    if half > max_half:
        half = max_half
    wx, wy = float(landmarks_xy_resized[wi, 0]), float(landmarks_xy_resized[wi, 1])
    cx, cy = wx, wy
    if not bool(missing_mask[ei]):
        ex, ey = float(landmarks_xy_resized[ei, 0]), float(landmarks_xy_resized[ei, 1])
        vx, vy = wx - ex, wy - ey
        n = max(float(np.hypot(vx, vy)), 1.0)
        # Center slightly beyond wrist along elbow→wrist.
        cx = wx + 0.25 * (vx / n) * half
        cy = wy + 0.25 * (vy / n) * half
    x0 = float(np.clip(cx - half, 0.0, float(INPUT_SIZE - 1)))
    y0 = float(np.clip(cy - half, 0.0, float(INPUT_SIZE - 1)))
    x1 = float(np.clip(cx + half, 1.0, float(INPUT_SIZE)))
    y1 = float(np.clip(cy + half, 1.0, float(INPUT_SIZE)))
    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return None, "pose_prior_crop_too_small"
    area_frac = ((x1 - x0) * (y1 - y0)) / float(INPUT_SIZE * INPUT_SIZE)
    if area_frac > max_area_fraction + 1e-6:
        return None, "pose_prior_area_exceeds_lock"
    # Target wrist must be near box center.
    bx = 0.5 * (x0 + x1)
    by = 0.5 * (y0 + y1)
    offset = float(np.hypot(wx - bx, wy - by))
    if offset > max_center_offset_frac_of_half * half:
        return None, "pose_prior_target_wrist_off_center"
    # Contralateral wrist must be outside.
    if not bool(missing_mask[contra]):
        cxw, cyw = float(landmarks_xy_resized[contra, 0]), float(landmarks_xy_resized[contra, 1])
        if x0 <= cxw <= x1 and y0 <= cyw <= y1:
            return None, "pose_prior_contralateral_wrist_inside"
    return np.asarray([x0, y0, x1, y1], dtype=np.float32), None


def bimanual_shared_prior_box(
    landmarks_xy_resized: np.ndarray,
    missing_mask: np.ndarray,
    *,
    margin_scale: float = 0.35,
    min_side: float = 24.0,
    max_area_fraction: float = 0.08,
) -> tuple[np.ndarray | None, str | None]:
    """One shared ROI covering both wrists; not labeled as left-only or right-only."""
    if bool(missing_mask[_POSE_L_WRIST]) or bool(missing_mask[_POSE_R_WRIST]):
        return None, "bimanual_requires_both_wrists"
    shoulder_w = _shoulder_width(landmarks_xy_resized, missing_mask)
    pts = landmarks_xy_resized[[_POSE_L_WRIST, _POSE_R_WRIST], :2]
    x0 = float(pts[:, 0].min()) - margin_scale * shoulder_w
    y0 = float(pts[:, 1].min()) - margin_scale * shoulder_w
    x1 = float(pts[:, 0].max()) + margin_scale * shoulder_w
    y1 = float(pts[:, 1].max()) + margin_scale * shoulder_w
    x0 = float(np.clip(x0, 0.0, float(INPUT_SIZE - 1)))
    y0 = float(np.clip(y0, 0.0, float(INPUT_SIZE - 1)))
    x1 = float(np.clip(x1, 1.0, float(INPUT_SIZE)))
    y1 = float(np.clip(y1, 1.0, float(INPUT_SIZE)))
    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return None, "bimanual_crop_too_small"
    area_frac = ((x1 - x0) * (y1 - y0)) / float(INPUT_SIZE * INPUT_SIZE)
    if area_frac > max_area_fraction + 1e-6:
        return None, "bimanual_area_exceeds_lock"
    return np.asarray([x0, y0, x1, y1], dtype=np.float32), None


def region_boxes_with_rgb_sources(
    landmarks_xy_resized: np.ndarray,
    missing_mask: np.ndarray,
    *,
    enable_pose_wrist_prior: bool = False,
    enable_ambiguity_aware_prior: bool = False,
    ambiguity_lock: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray | None], dict[str, str], dict[str, str | None]]:
    """Return boxes + rgb_roi_source + rejection reasons.

    Skeleton missing is owned by missing_mask; this never flips skeleton bits.
    R7C: ambiguity-aware single-hand or bimanual_shared_prior (same box on L+R,
    no extra region token — B3/B4 four-region parity preserved).
    """
    boxes = region_boxes_from_landmarks(landmarks_xy_resized, missing_mask)
    sources = {
        "global": "frame",
        "face": "face_landmarks" if boxes["face"] is not None else "missing",
        "left_hand": "hand_landmarks" if boxes["left_hand"] is not None else "missing",
        "right_hand": "hand_landmarks" if boxes["right_hand"] is not None else "missing",
    }
    reasons: dict[str, str | None] = {
        "face": None if boxes["face"] is not None else "face_landmarks_missing",
        "left_hand": None if boxes["left_hand"] is not None else "hand_landmarks_missing",
        "right_hand": None if boxes["right_hand"] is not None else "hand_landmarks_missing",
    }
    need_prior = boxes["left_hand"] is None or boxes["right_hand"] is None
    if enable_ambiguity_aware_prior and need_prior:
        lock = ambiguity_lock or {}
        mode = classify_pose_prior_mode(
            landmarks_xy_resized,
            missing_mask,
            bimanual_wrist_distance_frac_of_shoulder=float(
                lock.get("bimanual_wrist_distance_frac_of_shoulder", 0.55)
            ),
        )
        if mode == "bimanual" and need_prior:
            shared, reason = bimanual_shared_prior_box(
                landmarks_xy_resized,
                missing_mask,
                margin_scale=float(lock.get("bimanual_margin_scale", 0.35)),
                max_area_fraction=float(lock.get("bimanual_max_area_fraction", 0.08)),
            )
            if shared is not None:
                for which in ("left_hand", "right_hand"):
                    if boxes[which] is None:
                        boxes[which] = shared.copy()
                        sources[which] = "bimanual_shared_prior"
                        reasons[which] = None
            else:
                for which in ("left_hand", "right_hand"):
                    if boxes[which] is None:
                        sources[which] = "missing"
                        reasons[which] = reason
        else:
            for which in ("left_hand", "right_hand"):
                if boxes[which] is not None:
                    continue
                prior, reason = single_hand_pose_prior_box(
                    landmarks_xy_resized,
                    missing_mask,
                    which=which,
                    crop_scale=float(lock.get("single_hand_crop_scale", 0.85)),
                    max_area_fraction=float(lock.get("single_hand_max_area_fraction", 0.06)),
                    max_center_offset_frac_of_half=float(
                        lock.get("single_hand_max_center_offset_frac_of_half", 0.35)
                    ),
                )
                if prior is not None:
                    boxes[which] = prior
                    sources[which] = "pose_wrist_prior"
                    reasons[which] = None
                else:
                    sources[which] = "missing"
                    reasons[which] = reason
    elif enable_pose_wrist_prior:
        for which in ("left_hand", "right_hand"):
            if boxes[which] is not None:
                continue
            prior, reason = pose_wrist_prior_box(landmarks_xy_resized, missing_mask, which=which)
            if prior is not None:
                boxes[which] = prior
                sources[which] = "pose_wrist_prior"
                reasons[which] = None
            else:
                sources[which] = "missing"
                reasons[which] = reason
    return boxes, sources, reasons


def _masked_mean_tokens(
    tokens: list[np.ndarray],
    missing: list[bool],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mean only over valid ROI slices; missing zeros never enter the denominator."""
    if len(tokens) != len(missing):
        raise RuntimeError("masked_mean_len_mismatch")
    n_slices = len(tokens)
    valid = [tok for tok, miss in zip(tokens, missing) if not miss]
    n_valid = len(valid)
    dim = int(tokens[0].shape[0]) if tokens else 0
    if n_valid == 0:
        out = np.zeros((dim,), dtype=np.float32)
    else:
        out = np.mean(np.stack(valid, axis=0), axis=0).astype(np.float32)
    return out, {
        "n_slices": int(n_slices),
        "n_valid": int(n_valid),
        "valid_fraction": (float(n_valid) / float(n_slices)) if n_slices else 0.0,
        "all_missing": bool(n_valid == 0),
        "partial_missing": bool(0 < n_valid < n_slices),
    }


def _roi_align_token(feat: torch.Tensor, box: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """ROIAlign on a single temporal slice feat shaped [B,C,H,W]."""
    if box is None:
        return np.zeros((feat.shape[1],), dtype=np.float32), True
    if feat.ndim != 4:
        raise RuntimeError(f"roi_align_expects_bchw:{tuple(feat.shape)}")
    from torchvision.ops import roi_align

    _, _, h, w = feat.shape
    scale_x = w / float(INPUT_SIZE)
    scale_y = h / float(INPUT_SIZE)
    b = box.copy()
    b[0] *= scale_x
    b[2] *= scale_x
    b[1] *= scale_y
    b[3] *= scale_y
    rois = torch.tensor([[0.0, b[0], b[1], b[2], b[3]]], device=feat.device, dtype=feat.dtype)
    out = roi_align(feat, rois, output_size=(ROI_OUT, ROI_OUT), spatial_scale=1.0, aligned=True)
    token = out.mean(dim=(2, 3)).squeeze(0).detach().float().cpu().numpy()
    return token.astype(np.float32), False


def _roi_align_tokens_batched(
    feat: torch.Tensor,
    boxes_by_region: dict[str, np.ndarray | None],
    *,
    regions: tuple[str, ...] = REGIONS,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """Batch ROIAlign for one temporal slice across regions in *code* order.

    Region order is locked to ``REGIONS`` = global, left_hand, right_hand, face.
    Missing boxes yield zero tokens + missing=True without participating in the kernel.
    """
    if feat.ndim != 4:
        raise RuntimeError(f"roi_align_expects_bchw:{tuple(feat.shape)}")
    if tuple(regions) != tuple(REGIONS):
        raise RuntimeError(f"roi_region_order_mismatch:{regions}")
    from torchvision.ops import roi_align

    tokens: dict[str, np.ndarray] = {}
    missing: dict[str, bool] = {}
    _, _, h, w = feat.shape
    scale_x = w / float(INPUT_SIZE)
    scale_y = h / float(INPUT_SIZE)
    rois_rows: list[list[float]] = []
    present: list[str] = []
    for region in regions:
        box = boxes_by_region.get(region)
        if box is None:
            tokens[region] = np.zeros((feat.shape[1],), dtype=np.float32)
            missing[region] = True
            continue
        b = np.asarray(box, dtype=np.float32).copy()
        b[0] *= scale_x
        b[2] *= scale_x
        b[1] *= scale_y
        b[3] *= scale_y
        rois_rows.append([0.0, float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        present.append(region)
    if rois_rows:
        rois = torch.tensor(rois_rows, device=feat.device, dtype=feat.dtype)
        out = roi_align(feat, rois, output_size=(ROI_OUT, ROI_OUT), spatial_scale=1.0, aligned=True)
        pooled = out.mean(dim=(2, 3)).detach().float().cpu().numpy()
        for i, region in enumerate(present):
            tokens[region] = pooled[i].astype(np.float32)
            missing[region] = False
    # Ensure all regions present in outputs
    for region in regions:
        if region not in tokens:
            tokens[region] = np.zeros((feat.shape[1],), dtype=np.float32)
            missing[region] = True
    return tokens, missing


def build_temporal_contact_sheet(
    *,
    frames_u8: np.ndarray,
    slice_overlays: list[dict[str, Any]],
    cols: int = 4,
) -> np.ndarray:
    """Build a contact sheet of temporal ROI overlays for slow/fast QA."""
    if not slice_overlays:
        raise RuntimeError("empty_temporal_contact_sheet")
    tiles = []
    for row in slice_overlays:
        idx = int(row["local_frame"])
        idx = int(np.clip(idx, 0, frames_u8.shape[0] - 1))
        bgr = cv2.cvtColor(frames_u8[idx], cv2.COLOR_RGB2BGR)
        boxes = {
            k: (None if v is None else np.asarray(v, dtype=np.float32))
            for k, v in (row.get("boxes") or {}).items()
        }
        vis = draw_roi_overlay(bgr, boxes, rgb_roi_sources=row.get("rgb_roi_sources"))
        label = f"{row.get('pathway','?')} t={row.get('t_idx')}"
        cv2.putText(vis, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(vis)
    cols = max(1, int(cols))
    rows = int(np.ceil(len(tiles) / cols))
    h, w = tiles[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    return canvas


def _local_frame_for_feat_t(
    *,
    pathway: str,
    t_idx: int,
    t_feat: int,
    n_valid: int,
    window_size: int,
) -> int:
    """Map feature-map temporal index → local window frame index (0..window_size-1)."""
    n_valid = max(1, int(n_valid))
    if pathway == "slow":
        src = np.linspace(0, window_size - 1, num=t_feat).astype(int)
        return int(min(int(src[t_idx]), n_valid - 1))
    if pathway == "fast":
        return int(min(int(t_idx), n_valid - 1))
    raise RuntimeError(f"unknown_pathway:{pathway}")


def _boxes_for_native_frame(
    *,
    native_frame: int,
    canvas_meta: dict[str, float],
    landmarks_by_frame: dict[int, np.ndarray],
    missing_by_frame: dict[int, np.ndarray],
    skel_frame_index: np.ndarray,
    skel_timestamps: np.ndarray,
    fps: float,
    enable_pose_wrist_prior: bool = False,
    enable_ambiguity_aware_prior: bool = False,
    ambiguity_lock: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray | None], dict[str, Any], dict[str, str], dict[str, str], dict[str, str | None]]:
    align = nearest_skel_landmark(
        center_frame=int(native_frame),
        fps=fps,
        skel_frame_index=skel_frame_index,
        skel_timestamps=skel_timestamps,
    )
    lm_frame = int(align["landmark_frame"])
    lm = landmarks_by_frame.get(lm_frame)
    miss = missing_by_frame.get(lm_frame)
    if lm is None or miss is None:
        raise RuntimeError(f"landmark_missing_for_native_frame:{native_frame}:{lm_frame}")
    xy = landmark_xy_to_resized(lm[:, :2], canvas_meta)
    boxes, sources, prior_reasons = region_boxes_with_rgb_sources(
        xy,
        miss,
        enable_pose_wrist_prior=enable_pose_wrist_prior,
        enable_ambiguity_aware_prior=enable_ambiguity_aware_prior,
        ambiguity_lock=ambiguity_lock,
    )
    none_reasons: dict[str, str] = {}
    for region, sl in (("face", FACE), ("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
        if boxes[region] is None:
            if prior_reasons.get(region):
                none_reasons[region] = str(prior_reasons[region])
            else:
                part_miss = bool(np.all(miss[sl]))
                none_reasons[region] = "all_landmarks_missing" if part_miss else "degenerate_bbox"
    return boxes, align, none_reasons, sources, prior_reasons


def draw_roi_overlay(
    canvas_bgr: np.ndarray,
    boxes: dict[str, np.ndarray | None],
    *,
    rgb_roi_sources: dict[str, str] | None = None,
) -> np.ndarray:
    """Draw ROI boxes on a letterboxed BGR frame for visual QA."""
    colors = {
        "global": (200, 200, 200),
        "left_hand": (0, 165, 255),
        "right_hand": (0, 255, 0),
        "face": (255, 0, 0),
    }
    prior_colors = {
        "left_hand": (255, 0, 255),  # magenta
        "right_hand": (255, 255, 0),  # yellow
    }
    bimanual_color = (0, 140, 255)  # orange-ish BGR
    out = canvas_bgr.copy()
    for region, box in boxes.items():
        if box is None:
            continue
        src = (rgb_roi_sources or {}).get(region, "hand_landmarks")
        if src == "bimanual_shared_prior":
            color = bimanual_color
            label = f"{region}|bimanual"
            thickness = 2
        elif src == "pose_wrist_prior":
            color = prior_colors.get(region, (255, 0, 255))
            label = f"{region}|pose_prior"
            thickness = 2
        else:
            color = colors.get(region, (255, 255, 255))
            label = region
            thickness = 2
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        cv2.putText(out, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


@torch.no_grad()
def extract_rgb_token_v3_for_video(
    *,
    video_path: str | Any,
    extractor: FrozenSlowFastBlock4,
    landmarks_by_frame: dict[int, np.ndarray],
    missing_by_frame: dict[int, np.ndarray],
    skel_frame_index: np.ndarray | None = None,
    skel_timestamps: np.ndarray | None = None,
    window_size: int = WINDOW_SIZE,
    window_stride: int = WINDOW_STRIDE,
    overlay_dir: Path | None = None,
    sample_id: str | None = None,
    forbid_exact_frame_miss_silent_none: bool = True,
    enable_pose_wrist_prior: bool = False,
    enable_ambiguity_aware_prior: bool = False,
    ambiguity_lock: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    frames_native: np.ndarray | None = None,
    frame_idx: np.ndarray | None = None,
    fps: float | None = None,
    pathway_window_batch_size: int = 1,
    batch_roi_regions: bool = False,
) -> dict[str, Any]:
    """Extract compact Slow/Fast ROI tokens with timestamp mapping.

    If frames_native/frame_idx/fps are provided together, skip video decode
    (stage-matched benchmarks). Otherwise decode from video_path.

    pathway_window_batch_size > 1 batches SlowFast forward_maps across windows
    of the *same* clip only (E1b/E1c). BatchNorm is eval/frozen so outputs must
    match sequential windows for FP32 extractors.

    batch_roi_regions: when True, ROIAlign batches the four code-order regions
    (global, left_hand, right_hand, face) into one call per temporal slice (S3).
    """
    if frames_native is not None:
        if frame_idx is None or fps is None:
            raise ValueError("frames_native_requires_frame_idx_and_fps")
        frames_native = np.asarray(frames_native)
        frame_idx = np.asarray(frame_idx, dtype=np.int32)
        fps = float(fps)
    else:
        frames_native, frame_idx, fps = read_video_frames_native(video_path)
    resized = []
    metas = []
    for fr in frames_native:
        canvas, meta = _resize_letterbox(fr, INPUT_SIZE)
        resized.append(canvas)
        metas.append(meta)
    frames_u8 = np.stack(resized, axis=0)
    t_native = frames_u8.shape[0]

    if skel_frame_index is None:
        skel_frame_index = np.asarray(sorted(landmarks_by_frame.keys()), dtype=np.int32)
    else:
        skel_frame_index = np.asarray(skel_frame_index, dtype=np.int32)
    if skel_timestamps is None:
        skel_timestamps = skel_frame_index.astype(np.float64) / float(fps)
    else:
        skel_timestamps = np.asarray(skel_timestamps, dtype=np.float64)

    starts = window_starts(t_native, window_size=window_size, window_stride=window_stride)
    bs = max(1, int(pathway_window_batch_size))

    window_inputs: list[tuple[int, torch.Tensor, torch.Tensor, np.ndarray, int]] = []
    for start in starts:
        slow_t, fast_t, valid_mask, n_valid = window_tensors(frames_u8, start, window_size)
        window_inputs.append((start, slow_t, fast_t, valid_mask, n_valid))

    maps_list: list[PathwayMaps] = []
    for i0 in range(0, len(window_inputs), bs):
        chunk = window_inputs[i0 : i0 + bs]
        if len(chunk) == 1:
            maps_list.append(extractor.forward_maps(chunk[0][1], chunk[0][2]))
        else:
            slow_b = torch.cat([c[1] for c in chunk], dim=0)
            fast_b = torch.cat([c[2] for c in chunk], dim=0)
            maps_b = extractor.forward_maps(slow_b, fast_b)
            for j in range(len(chunk)):
                maps_list.append(PathwayMaps(slow=maps_b.slow[j : j + 1], fast=maps_b.fast[j : j + 1]))

    tokens: dict[str, list[np.ndarray]] = {r: [] for r in REGIONS}
    missing_region: dict[str, list[bool]] = {r: [] for r in REGIONS}
    pathway_tokens: dict[str, list[np.ndarray]] = {"slow": [], "fast": []}
    window_meta = []
    overlays: list[dict[str, Any]] = []

    for (start, _slow_t, _fast_t, valid_mask, n_valid), maps in zip(window_inputs, maps_list):
        end = min(t_native, start + window_size)
        center = start + (end - start) // 2

        align = nearest_skel_landmark(
            center_frame=int(frame_idx[center]),
            fps=fps,
            skel_frame_index=skel_frame_index,
            skel_timestamps=skel_timestamps,
        )
        lm_frame = int(align["landmark_frame"])
        # Forbid silent None caused by exact-frame miss under stride subsample.
        if forbid_exact_frame_miss_silent_none and int(frame_idx[center]) not in landmarks_by_frame:
            if lm_frame not in landmarks_by_frame:
                raise RuntimeError(f"alignment_landmark_missing:{sample_id}:{lm_frame}")
        lm = landmarks_by_frame.get(lm_frame)
        miss = missing_by_frame.get(lm_frame)
        if lm is None or miss is None:
            raise RuntimeError(
                f"stride_miss_silent_none_forbidden:{sample_id}:center={int(frame_idx[center])}:landmark={lm_frame}"
            )
        xy = landmark_xy_to_resized(lm[:, :2], metas[center])
        boxes_center, sources_center, prior_reasons_center = region_boxes_with_rgb_sources(
            xy,
            miss,
            enable_pose_wrist_prior=enable_pose_wrist_prior,
            enable_ambiguity_aware_prior=enable_ambiguity_aware_prior,
            ambiguity_lock=ambiguity_lock,
        )
        transform = metas[center]
        none_reasons = {}
        for region, sl in (("face", FACE), ("left_hand", LEFT_HAND), ("right_hand", RIGHT_HAND)):
            if boxes_center[region] is None:
                if prior_reasons_center.get(region):
                    none_reasons[region] = str(prior_reasons_center[region])
                else:
                    part_miss = bool(np.all(miss[sl]))
                    none_reasons[region] = "all_landmarks_missing" if part_miss else "degenerate_bbox"

        # R0 operator order: ROIAlign on F[t,h,w,c] per temporal slice, then aggregate.
        if maps.slow.ndim != 5 or maps.fast.ndim != 5:
            raise RuntimeError(f"unexpected_map_rank:slow={maps.slow.shape}:fast={maps.fast.shape}")
        t_slow = int(maps.slow.shape[2])
        t_fast = int(maps.fast.shape[2])
        per_region_slow: dict[str, list[np.ndarray]] = {r: [] for r in REGIONS}
        per_region_fast: dict[str, list[np.ndarray]] = {r: [] for r in REGIONS}
        miss_slow: dict[str, list[bool]] = {r: [] for r in REGIONS}
        miss_fast: dict[str, list[bool]] = {r: [] for r in REGIONS}
        temporal_trace = []
        contact_slice_rows: list[dict[str, Any]] = []

        for t_idx in range(t_slow):
            local = _local_frame_for_feat_t(
                pathway="slow", t_idx=t_idx, t_feat=t_slow, n_valid=n_valid, window_size=window_size
            )
            native = int(frame_idx[min(start + local, t_native - 1)])
            boxes_t, align_t, _nr, sources_t, _pr = _boxes_for_native_frame(
                native_frame=native,
                canvas_meta=metas[min(start + local, t_native - 1)],
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                skel_frame_index=skel_frame_index,
                skel_timestamps=skel_timestamps,
                fps=fps,
                enable_pose_wrist_prior=enable_pose_wrist_prior,
                enable_ambiguity_aware_prior=enable_ambiguity_aware_prior,
                ambiguity_lock=ambiguity_lock,
            )
            feat_t = maps.slow[:, :, t_idx, :, :]
            region_missing = {}
            if batch_roi_regions:
                toks_b, miss_b = _roi_align_tokens_batched(feat_t, boxes_t, regions=REGIONS)
                for region in REGIONS:
                    per_region_slow[region].append(toks_b[region])
                    miss_slow[region].append(bool(miss_b[region]))
                    region_missing[region] = bool(miss_b[region])
            else:
                for region in REGIONS:
                    tok, mflag = _roi_align_token(feat_t, boxes_t[region])
                    per_region_slow[region].append(tok)
                    miss_slow[region].append(bool(mflag))
                    region_missing[region] = bool(mflag)
            row = {
                "pathway": "slow",
                "t_idx": t_idx,
                "local_frame": local,
                "native_frame": native,
                "landmark_frame": align_t["landmark_frame"],
                "alignment_delta_sec": align_t["alignment_delta_sec"],
                "boxes": {
                    k: (None if v is None else [float(x) for x in v.tolist()]) for k, v in boxes_t.items()
                },
                "region_missing": region_missing,
                "rgb_roi_sources": sources_t,
            }
            temporal_trace.append(row)
            contact_slice_rows.append(row)

        for t_idx in range(t_fast):
            local = _local_frame_for_feat_t(
                pathway="fast", t_idx=t_idx, t_feat=t_fast, n_valid=n_valid, window_size=window_size
            )
            native = int(frame_idx[min(start + local, t_native - 1)])
            boxes_t, align_t, _nr, sources_t, _pr = _boxes_for_native_frame(
                native_frame=native,
                canvas_meta=metas[min(start + local, t_native - 1)],
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                skel_frame_index=skel_frame_index,
                skel_timestamps=skel_timestamps,
                fps=fps,
                enable_pose_wrist_prior=enable_pose_wrist_prior,
                enable_ambiguity_aware_prior=enable_ambiguity_aware_prior,
                ambiguity_lock=ambiguity_lock,
            )
            feat_t = maps.fast[:, :, t_idx, :, :]
            region_missing = {}
            if batch_roi_regions:
                toks_b, miss_b = _roi_align_tokens_batched(feat_t, boxes_t, regions=REGIONS)
                for region in REGIONS:
                    per_region_fast[region].append(toks_b[region])
                    miss_fast[region].append(bool(miss_b[region]))
                    region_missing[region] = bool(miss_b[region])
            else:
                for region in REGIONS:
                    tok, mflag = _roi_align_token(feat_t, boxes_t[region])
                    per_region_fast[region].append(tok)
                    miss_fast[region].append(bool(mflag))
                    region_missing[region] = bool(mflag)
            row = {
                "pathway": "fast",
                "t_idx": t_idx,
                "local_frame": local,
                "native_frame": native,
                "landmark_frame": align_t["landmark_frame"],
                "alignment_delta_sec": align_t["alignment_delta_sec"],
                "boxes": {
                    k: (None if v is None else [float(x) for x in v.tolist()]) for k, v in boxes_t.items()
                },
                "region_missing": region_missing,
                "rgb_roi_sources": sources_t,
            }
            temporal_trace.append(row)
            contact_slice_rows.append(row)

        # Pathway global + region tokens: masked mean over valid ROI slices only.
        region_validity: dict[str, Any] = {}
        tok_s_global, val_s_global = _masked_mean_tokens(per_region_slow["global"], miss_slow["global"])
        tok_f_global, val_f_global = _masked_mean_tokens(per_region_fast["global"], miss_fast["global"])
        pathway_tokens["slow"].append(tok_s_global)
        pathway_tokens["fast"].append(tok_f_global)

        for region in REGIONS:
            tok_s, val_s = _masked_mean_tokens(per_region_slow[region], miss_slow[region])
            tok_f, val_f = _masked_mean_tokens(per_region_fast[region], miss_fast[region])
            tokens[region].append(np.concatenate([tok_s, tok_f], axis=0))
            # Tensor missing mask = all temporal slices missing on both pathways.
            all_missing = bool(val_s["all_missing"] and val_f["all_missing"])
            missing_region[region].append(all_missing)
            region_validity[region] = {
                "slow": val_s,
                "fast": val_f,
                "n_slices": int(val_s["n_slices"] + val_f["n_slices"]),
                "n_valid": int(val_s["n_valid"] + val_f["n_valid"]),
                "valid_fraction": (
                    float(val_s["n_valid"] + val_f["n_valid"])
                    / float(val_s["n_slices"] + val_f["n_slices"])
                    if (val_s["n_slices"] + val_f["n_slices"])
                    else 0.0
                ),
                "all_missing": all_missing,
                "partial_missing": bool(
                    (val_s["partial_missing"] or val_f["partial_missing"])
                    or (val_s["all_missing"] != val_f["all_missing"])
                    or (
                        (val_s["n_valid"] + val_f["n_valid"]) > 0
                        and (val_s["n_valid"] + val_f["n_valid"])
                        < (val_s["n_slices"] + val_f["n_slices"])
                    )
                ),
            }

        covers_last = int(frame_idx[end - 1]) == int(frame_idx[-1])
        window_meta.append(
            {
                "start_frame": int(frame_idx[start]),
                "end_frame": int(frame_idx[end - 1]),
                "center_frame": int(frame_idx[center]),
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "center_idx": int(center),
                "timestamp_sec": float(frame_idx[center] / fps),
                "start_timestamp_sec": float(frame_idx[start] / fps),
                "end_timestamp_sec": float(frame_idx[end - 1] / fps),
                "n_valid_frames": int(n_valid),
                "valid_frame_mask": valid_mask.astype(bool).tolist(),
                "covers_last_native_frame": bool(covers_last),
                "end_anchored": bool(start == starts[-1] and len(starts) > 1 and start == t_native - window_size),
                "transform": transform,
                "feature_map_shapes": {
                    "slow": list(maps.slow.shape),
                    "fast": list(maps.fast.shape),
                },
                "roi_boxes_canvas224": {
                    k: (None if v is None else [float(x) for x in v.tolist()]) for k, v in boxes_center.items()
                },
                "rgb_roi_sources_center": sources_center,
                "pose_prior_rejection_reasons_center": {
                    k: v for k, v in prior_reasons_center.items() if v is not None
                },
                "enable_pose_wrist_prior": bool(enable_pose_wrist_prior),
                "enable_ambiguity_aware_prior": bool(enable_ambiguity_aware_prior),
                "roi_none_reasons": none_reasons,
                "landmark_frame": align["landmark_frame"],
                "landmark_timestamp_sec": align["landmark_timestamp_sec"],
                "alignment_delta_sec": align["alignment_delta_sec"],
                "alignment_method": align["alignment_method"],
                "alignment_tie_break": align.get("alignment_tie_break"),
                "operator_order": "roi_align_per_temporal_slice_then_masked_mean_aggregate",
                "temporal_aggregation": "masked_mean_exclude_missing_zeros",
                "forbid_temporal_mean_before_roi_align": True,
                "forbid_zero_dilution_in_temporal_mean": True,
                "temporal_roi_trace_n": len(temporal_trace),
                "region_validity": region_validity,
                "pathway_global_validity": {"slow": val_s_global, "fast": val_f_global},
                "temporal_slice_map": [
                    {
                        "pathway": t["pathway"],
                        "t_idx": t["t_idx"],
                        "local_frame": t["local_frame"],
                        "native_frame": t["native_frame"],
                        "landmark_frame": t["landmark_frame"],
                        "region_missing": t["region_missing"],
                        "rgb_roi_sources": t.get("rgb_roi_sources"),
                    }
                    for t in temporal_trace
                ],
            }
        )

        if overlay_dir is not None and sample_id is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(frames_u8[center], cv2.COLOR_RGB2BGR)
            vis = draw_roi_overlay(bgr, boxes_center, rgb_roi_sources=sources_center)
            rel = f"{sample_id}_w{start:04d}_c{center:04d}.png"
            out_path = overlay_dir / rel
            cv2.imwrite(str(out_path), vis)
            # Temporal contact sheet: evenly sample up to 4 slow + 4 fast slices.
            slow_rows = [r for r in contact_slice_rows if r["pathway"] == "slow"]
            fast_rows = [r for r in contact_slice_rows if r["pathway"] == "fast"]
            def _sample(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
                if not rows:
                    return []
                if len(rows) <= k:
                    return rows
                idxs = np.linspace(0, len(rows) - 1, num=k).astype(int).tolist()
                return [rows[i] for i in idxs]
            sheet_rows = _sample(slow_rows, 4) + _sample(fast_rows, 4)
            tiles = [
                {
                    "pathway": r["pathway"],
                    "t_idx": int(r["t_idx"]),
                    "local_frame": int(r["local_frame"]),
                    "native_frame": int(r["native_frame"]),
                    "landmark_frame": int(r["landmark_frame"]),
                    "boxes": r["boxes"],
                    "region_missing": r["region_missing"],
                    "rgb_roi_sources": r.get("rgb_roi_sources"),
                }
                for r in sheet_rows
            ]
            sheet = build_temporal_contact_sheet(
                frames_u8=frames_u8[start:end],
                slice_overlays=[
                    {
                        **r,
                        "local_frame": int(r["local_frame"]),
                    }
                    for r in sheet_rows
                ],
                cols=4,
            )
            sheet_rel = f"{sample_id}_w{start:04d}_temporal_contact.png"
            sheet_path = overlay_dir / sheet_rel
            cv2.imwrite(str(sheet_path), sheet)
            center_box_missing = {
                r: boxes_center[r] is None for r in ("face", "left_hand", "right_hand")
            }
            overlays.append(
                {
                    "kind": "center_overlay",
                    "path": (
                        out_path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
                        if repo_root is not None
                        else str(out_path).replace("\\", "/")
                    ),
                    "sha256": sha256_file(out_path),
                    "center_frame": int(frame_idx[center]),
                    "native_frame": int(frame_idx[center]),
                    "landmark_frame": align["landmark_frame"],
                    "boxes": window_meta[-1]["roi_boxes_canvas224"],
                    "missing_regions": center_box_missing,
                    "rgb_roi_sources": sources_center,
                    "region_validity": {
                        r: region_validity[r] for r in REGIONS if r != "global"
                    },
                }
            )
            overlays.append(
                {
                    "kind": "temporal_contact_sheet",
                    "path": (
                        sheet_path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
                        if repo_root is not None
                        else str(sheet_path).replace("\\", "/")
                    ),
                    "sha256": sha256_file(sheet_path),
                    "center_frame": int(frame_idx[center]),
                    "native_frame": int(frame_idx[center]),
                    "landmark_frame": align["landmark_frame"],
                    "n_tiles": len(tiles),
                    "slice_native_frames": [int(t["native_frame"]) for t in tiles],
                    "tiles": tiles,
                    "boxes": window_meta[-1]["roi_boxes_canvas224"],
                    "missing_regions": center_box_missing,
                    "rgb_roi_sources": sources_center,
                    "region_validity": {
                        r: region_validity[r] for r in REGIONS if r != "global"
                    },
                }
            )

    # Provenance: last window must cover final native frame.
    if not any(w["covers_last_native_frame"] for w in window_meta):
        raise RuntimeError("end_anchored_window_missing_last_frame_coverage")

    out_tokens = {k: np.stack(v, axis=0) for k, v in tokens.items()}
    out_missing = {k: np.asarray(v, dtype=bool) for k, v in missing_region.items()}
    return {
        "tokens": out_tokens,
        "missing_region_mask": out_missing,
        "pathway_global_tokens": {k: np.stack(v, axis=0) for k, v in pathway_tokens.items()},
        "windows": window_meta,
        "overlays": overlays,
        "fps": fps,
        "n_native_frames": int(t_native),
        "original_frame_index": frame_idx,
        "hook": "blocks.4_before_blocks.5_pooling",
        "roi_method": "torchvision.ops.roi_align",
        "roi_output_size": ROI_OUT,
        "regions": list(REGIONS),
        "input_size": INPUT_SIZE,
        "window_size": window_size,
        "window_stride": window_stride,
        "window_starts": starts,
        "pathway_window_batch_size": int(bs),
        "batch_roi_regions": bool(batch_roi_regions),
        "forbid_partial_unfreeze": True,
        "compact_tokens_only": True,
        "backbone": extractor.load_meta,
        "operator_order": "roi_align_per_temporal_slice_then_masked_mean_aggregate",
        "temporal_aggregation": "masked_mean_exclude_missing_zeros",
        "forbid_temporal_mean_before_roi_align": True,
        "forbid_zero_dilution_in_temporal_mean": True,
        "enable_pose_wrist_prior": bool(enable_pose_wrist_prior),
        "enable_ambiguity_aware_prior": bool(enable_ambiguity_aware_prior),
        "joint_layout": {
            "pose": [0, 12],
            "face": [12, 33],
            "left_hand": [33, 54],
            "right_hand": [54, 75],
            "n_joints": N_JOINTS,
        },
    }
