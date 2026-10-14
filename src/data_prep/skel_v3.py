"""Skeleton v3 representation helpers for MS-STA v5 D0 pilot.

Persists pre-fill missing masks, native frame indices/timestamps, velocity,
bone vectors, and joint identity — never reconstruct masks from filled v2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data_prep.joint_layout_v3 import (
    JOINT_CHIRALITY,
    JOINT_PARTS,
    N_JOINTS,
    assert_layout_invariants,
    bone_edge_list,
)

assert_layout_invariants()


def shoulder_normalize(
    joints: np.ndarray,
    raw_pose: np.ndarray,
    missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize by mid-shoulder / shoulder width; return (normed, validity)."""
    joints = np.asarray(joints, dtype=np.float32)
    raw_pose = np.asarray(raw_pose, dtype=np.float32)
    missing = np.asarray(missing, dtype=bool)
    t_len = joints.shape[0]
    out = joints.copy()
    validity = (~missing).astype(np.float32)

    left = raw_pose[:, 11, :].copy()
    right = raw_pose[:, 12, :].copy()
    pose_missing = missing[:, 0].copy()
    dist = np.linalg.norm(left - right, axis=1)
    valid_pose = (~pose_missing) & (dist > 1e-3)
    if not np.any(valid_pose):
        raise ValueError("no_valid_shoulder_frame")
    first = int(np.argmax(valid_pose))
    if first > 0:
        left[:first] = left[first]
        right[:first] = right[first]
    for t in range(1, t_len):
        if (not valid_pose[t]) or np.linalg.norm(left[t] - right[t]) <= 1e-3:
            left[t] = left[t - 1]
            right[t] = right[t - 1]
    mid = (left + right) / 2.0
    width = np.linalg.norm(left - right, axis=1, keepdims=True)
    if np.any(width <= 1e-3):
        raise ValueError("degenerate_shoulder_width")
    for t in range(t_len):
        out[t] = (out[t] - mid[t]) / width[t]
    never = ~np.any(~missing, axis=0)
    if np.any(never):
        out[:, never, :] = 0.0
        validity[:, never] = 0.0
    if not np.isfinite(out).all():
        raise ValueError("non_finite_after_normalize")
    return out, validity


def _fill_for_derivatives(joints: np.ndarray, missing: np.ndarray) -> np.ndarray:
    filled = joints.copy()
    miss = missing.copy()
    t_len, n_j, _ = filled.shape
    for j in range(n_j):
        valid = ~miss[:, j]
        if not np.any(valid):
            continue
        first = int(np.argmax(valid))
        if first > 0:
            filled[:first, j, :] = filled[first, j, :]
        for t in range(1, t_len):
            if miss[t, j]:
                filled[t, j, :] = filled[t - 1, j, :]
    return filled


def build_skel_v3_arrays(
    *,
    joints_raw: np.ndarray,
    raw_pose: np.ndarray,
    missing_prefill: np.ndarray,
    original_frame_index: np.ndarray,
    timestamps_sec: np.ndarray,
    confidence: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build Skeleton v3 tensors from raw MediaPipe observations."""
    joints_raw = np.asarray(joints_raw, dtype=np.float32)
    missing_prefill = np.asarray(missing_prefill, dtype=bool)
    original_frame_index = np.asarray(original_frame_index, dtype=np.int32)
    timestamps_sec = np.asarray(timestamps_sec, dtype=np.float64)
    if joints_raw.ndim != 3 or joints_raw.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"bad_joints_shape:{joints_raw.shape}")
    if missing_prefill.shape != joints_raw.shape[:2]:
        raise ValueError("missing_mask_shape")
    if original_frame_index.shape[0] != joints_raw.shape[0]:
        raise ValueError("frame_index_len")
    if timestamps_sec.shape[0] != joints_raw.shape[0]:
        raise ValueError("timestamps_len")
    if len(JOINT_PARTS) != N_JOINTS or len(JOINT_CHIRALITY) != N_JOINTS:
        raise ValueError("joint_identity_len_not_75")

    coords, validity = shoulder_normalize(joints_raw, raw_pose, missing_prefill)
    filled = _fill_for_derivatives(coords, missing_prefill)
    velocity = np.zeros_like(filled)
    velocity[1:] = filled[1:] - filled[:-1]

    edges = bone_edge_list()
    covered = {i for e in edges for i in e}
    if (N_JOINTS - 1) not in covered:
        raise ValueError("joint_74_not_covered_by_bones")
    bones = np.zeros((filled.shape[0], len(edges), 3), dtype=np.float32)
    for bi, (a, b) in enumerate(edges):
        bones[:, bi, :] = filled[:, b, :] - filled[:, a, :]

    if confidence is None:
        confidence = validity.copy()
    else:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape != validity.shape:
            raise ValueError("confidence_shape")

    t_len = int(coords.shape[0])
    pad_mask = np.ones((t_len,), dtype=bool)

    return {
        "coordinates": coords,
        "velocity": velocity,
        "bone_vectors": bones,
        "missing_mask_prefill": missing_prefill.astype(bool),
        "validity": validity,
        "confidence": confidence,
        "original_frame_index": original_frame_index,
        "timestamps_sec": timestamps_sec,
        "padding_mask": pad_mask,
        "sequence_length": t_len,
        "joint_part": np.asarray(JOINT_PARTS),
        "joint_chirality": np.asarray(JOINT_CHIRALITY),
        "bone_edges": np.asarray(edges, dtype=np.int32),
        "joint_layout": {
            "pose": [0, 12],
            "face": [12, 33],
            "left_hand": [33, 54],
            "right_hand": [54, 75],
            "n_joints": N_JOINTS,
        },
    }
