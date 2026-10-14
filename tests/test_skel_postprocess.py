"""Regression: missing landmark groups must backfill; never blow up shoulder norm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_prep.skel_postprocess import postprocess_sequence

# Joint group slices (must match extract_frame_landmarks)
POSE = slice(0, 12)
FACE = slice(12, 33)
LH = slice(33, 54)
RH = slice(54, 75)


def _base_clip(t: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = np.zeros((t, 75, 3), dtype=np.float32)
    pose_raw = np.zeros((t, 33, 3), dtype=np.float32)
    missing = np.zeros((t, 75), dtype=bool)
    for t_i in range(t):
        left = np.array([0.4, 0.5, 0.0], dtype=np.float32)
        right = np.array([0.6, 0.5, 0.0], dtype=np.float32)
        pose_raw[t_i, 11] = left
        pose_raw[t_i, 12] = right
        joints[t_i, 0] = left
        joints[t_i, 1] = right
        joints[t_i, POSE] = 0.45 + 0.01 * np.arange(12).reshape(-1, 1).astype(np.float32) * np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        joints[t_i, 0] = left
        joints[t_i, 1] = right
        joints[t_i, FACE] = 0.5
        joints[t_i, LH] = left + np.array([0.05, -0.1, 0.0], dtype=np.float32)
        joints[t_i, RH] = right + np.array([-0.05, -0.1, 0.0], dtype=np.float32)
    return joints, pose_raw, missing


def _mark_group_missing(joints, pose_raw, missing, group: slice, frames: range) -> None:
    for t_i in frames:
        missing[t_i, group] = True
        joints[t_i, group] = 0.0
        if group == POSE:
            pose_raw[t_i] = 0.0


def test_missing_pose_mid_frame_no_blowup() -> None:
    joints, pose_raw, missing = _base_clip()
    _mark_group_missing(joints, pose_raw, missing, POSE, range(3, 4))
    # hands still present on that frame
    joints[3, LH] = np.array([0.45, 0.4, 0.0], dtype=np.float32)
    joints[3, RH] = np.array([0.55, 0.4, 0.0], dtype=np.float32)
    missing[3, LH] = False
    missing[3, RH] = False
    out = postprocess_sequence(joints, pose_raw, missing)
    assert out is not None
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 50.0


def test_leading_missing_pose_backfills() -> None:
    joints, pose_raw, missing = _base_clip()
    _mark_group_missing(joints, pose_raw, missing, POSE, range(0, 4))
    out = postprocess_sequence(joints, pose_raw, missing)
    assert out is not None
    # Leading frames should not remain absolute-zero ghosts after norm
    assert np.isfinite(out[:4]).all()
    assert float(np.max(np.abs(out))) < 50.0


@pytest.mark.parametrize(
    "group",
    [FACE, LH, RH],
    ids=["face", "left_hand", "right_hand"],
)
def test_leading_missing_landmark_group_backfills(group: slice) -> None:
    joints, pose_raw, missing = _base_clip()
    _mark_group_missing(joints, pose_raw, missing, group, range(0, 3))
    out = postprocess_sequence(joints, pose_raw, missing)
    assert out is not None
    assert np.isfinite(out).all()
    # Backfilled leading frames should match first valid after fill+norm (finite, non-exploded)
    assert float(np.max(np.abs(out[:3, group]))) < 50.0


def test_never_detected_joint_becomes_origin() -> None:
    joints, pose_raw, missing = _base_clip()
    # Joint 40 (in LH group) never detected across all frames
    missing[:, 40] = True
    joints[:, 40] = 0.0
    out = postprocess_sequence(joints, pose_raw, missing)
    assert out is not None
    assert np.allclose(out[:, 40, :], 0.0, atol=1e-5)


def test_all_pose_missing_returns_none() -> None:
    joints, pose_raw, missing = _base_clip()
    _mark_group_missing(joints, pose_raw, missing, POSE, range(10))
    out = postprocess_sequence(joints, pose_raw, missing)
    assert out is None


def test_no_missing_pose_finite() -> None:
    joints, pose_raw, missing = _base_clip()
    out = postprocess_sequence(joints, pose_raw, missing)
    assert out is not None
    assert np.isfinite(out).all()
    assert out.shape == (10, 75, 3)
