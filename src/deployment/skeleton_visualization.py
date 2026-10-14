"""Render the sealed Skeleton v3 cache for the defense UI.

Display stabilization is VIZ-ONLY and never mutates the cache or the tensors
consumed by Unified Attention.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from src.data_prep.joint_layout_v3 import FACE, LEFT_HAND, POSE, RIGHT_HAND, bone_edge_list


_PARTS = (
    (POSE, "#38bdf8"),
    (FACE, "#fbbf24"),
    (LEFT_HAND, "#34d399"),
    (RIGHT_HAND, "#fb7185"),
)
# Local pose order is nose, eyes, mouth, L/R shoulder, L/R elbow,
# L/R wrist, L/R hip. The persisted v3 bone graph has legacy pose edges;
# these display-only edges render the anatomical topology without changing D0.
_POSE_RENDER_EDGES = (
    (4, 5),
    (4, 6),
    (6, 8),
    (5, 7),
    (7, 9),
    (4, 10),
    (5, 11),
    (10, 11),
)
_STABLE_RENDER_IDS = np.asarray(
    list(range(0, 8)) + [10, 11] + list(range(FACE.start, FACE.stop)),
    dtype=np.int64,
)


def _part_color(joint: int) -> str:
    for part, color in _PARTS:
        if part.start <= joint < part.stop:
            return color
    return "#e5e7eb"


def render_edge_list() -> list[tuple[int, int]]:
    """Return anatomical display edges; do not alter persisted v3 bone tensors."""
    non_pose = [(a, b) for a, b in bone_edge_list() if a >= POSE.stop or b >= POSE.stop]
    return list(_POSE_RENDER_EDGES) + non_pose


def shoulder_center_for_display(
    coords: np.ndarray, validity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recenter each display frame at the two-shoulder midpoint.

    Skeleton v3 is already shoulder-normalized. Recomputing this translation makes
    the UI invariant explicit and protects future renderers from tiny numeric drift.
    """
    out = np.asarray(coords, dtype=np.float32).copy()
    valid = np.asarray(validity, dtype=bool)
    if out.ndim != 3 or out.shape[1:] != (75, 3) or valid.shape != out.shape[:2]:
        raise ValueError("invalid_skeleton_v3_for_display")
    centers = np.zeros((out.shape[0], 2), dtype=np.float32)
    previous = np.zeros((2,), dtype=np.float32)
    for t in range(out.shape[0]):
        if valid[t, 4] and valid[t, 5]:
            previous = (out[t, 4, :2] + out[t, 5, :2]) / 2.0
        centers[t] = previous
        out[t, :, :2] -= previous
    return out, centers


def stabilize_non_hand_display(
    coords: np.ndarray, validity: np.ndarray, *, alpha: float = 0.45
) -> np.ndarray:
    """Validity-aware EMA for torso/face rendering only; hands stay unsmoothed."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("display_smoothing_alpha_out_of_range")
    out = np.asarray(coords, dtype=np.float32).copy()
    valid = np.asarray(validity, dtype=bool)
    for t in range(1, out.shape[0]):
        ids = _STABLE_RENDER_IDS[valid[t, _STABLE_RENDER_IDS] & valid[t - 1, _STABLE_RENDER_IDS]]
        if ids.size:
            out[t, ids, :2] = alpha * out[t, ids, :2] + (1.0 - alpha) * out[t - 1, ids, :2]
    return out


def skeleton_summary(npz_path: Path) -> dict[str, float | int]:
    with np.load(npz_path, allow_pickle=False) as data:
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        validity = np.asarray(data["validity"], dtype=np.float32)
        timestamps = np.asarray(data["timestamps_sec"], dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (75, 3):
        raise ValueError(f"invalid_skeleton_shape:{coords.shape}")
    if validity.shape != coords.shape[:2] or timestamps.shape != (coords.shape[0],):
        raise ValueError("skeleton_alignment_mismatch")
    shoulder_mid = (coords[:, 4, :2] + coords[:, 5, :2]) / 2.0
    return {
        "frames": int(coords.shape[0]),
        "joints": int(coords.shape[1]),
        "duration_sec": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "valid_joint_fraction": float((validity > 0).mean()),
        "shoulder_center_max_abs": float(np.abs(shoulder_mid).max()),
    }


def _frame_durations_ms(timestamps: np.ndarray, frame_ids: np.ndarray, fps: int | None) -> list[int]:
    if fps is not None:
        return [max(40, int(round(1000 / max(fps, 1))))] * len(frame_ids)
    selected = timestamps[frame_ids]
    if len(selected) <= 1:
        return [100]
    deltas = np.diff(selected)
    fallback = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 0.1
    deltas = np.concatenate([deltas, [fallback]])
    return [int(np.clip(round(1000.0 * max(float(dt), 1e-3)), 40, 250)) for dt in deltas]


def render_skeleton_gif(
    npz_path: Path,
    *,
    max_frames: int = 48,
    fps: int | None = None,
    stabilize_display: bool = True,
) -> bytes:
    """Return a shoulder-centered animated GIF from sealed Skeleton v3 coordinates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with np.load(npz_path, allow_pickle=False) as data:
        coords = np.asarray(data["coordinates"], dtype=np.float32)
        validity = np.asarray(data["validity"], dtype=np.float32) > 0
        timestamps = np.asarray(data["timestamps_sec"], dtype=np.float64)
    if (
        coords.ndim != 3
        or coords.shape[1:] != (75, 3)
        or validity.shape != coords.shape[:2]
        or timestamps.shape != (coords.shape[0],)
    ):
        raise ValueError("invalid_skeleton_v3_for_render")

    display, _centers = shoulder_center_for_display(coords, validity)
    if stabilize_display:
        display = stabilize_non_hand_display(display, validity)
    valid_xy = display[..., :2][validity]
    if valid_xy.size == 0:
        raise ValueError("skeleton_has_no_valid_joints")
    extent = float(np.nanpercentile(np.abs(valid_xy), 99.5))
    half_span = max(extent, 1e-3) * 1.12
    xlim = (-half_span, half_span)
    ylim = (half_span, -half_span)

    if coords.shape[0] <= max_frames:
        frame_ids = np.arange(coords.shape[0], dtype=np.int64)
    else:
        frame_ids = np.unique(
            np.linspace(0, coords.shape[0] - 1, max_frames).round().astype(np.int64)
        )
    durations = _frame_durations_ms(timestamps, frame_ids, fps)
    edges = render_edge_list()
    frames: list[Image.Image] = []
    for frame_id in frame_ids:
        fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=100)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")
        xy = display[frame_id, :, :2]
        ok = validity[frame_id]
        for a, b in edges:
            if ok[a] and ok[b]:
                ax.plot(
                    [xy[a, 0], xy[b, 0]],
                    [xy[a, 1], xy[b, 1]],
                    color=_part_color(a),
                    linewidth=1.7,
                    alpha=0.88,
                )
        for part, color in _PARTS:
            ids = np.arange(part.start, part.stop)
            ids = ids[ok[ids]]
            if ids.size:
                ax.scatter(xy[ids, 0], xy[ids, 1], s=10, color=color, edgecolors="none")
        ax.scatter([0.0], [0.0], s=18, marker="+", color="#f8fafc", linewidths=1.0)
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.text(
            0.02,
            0.96,
            f"Skeleton v3  |  frame {frame_id + 1}/{coords.shape[0]}",
            transform=ax.transAxes,
            color="#f8fafc",
            fontsize=9,
            va="top",
        )
        fig.tight_layout(pad=0.15)
        png = io.BytesIO()
        fig.savefig(png, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        png.seek(0)
        frames.append(Image.open(png).convert("RGB").copy())
    out = io.BytesIO()
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return out.getvalue()
