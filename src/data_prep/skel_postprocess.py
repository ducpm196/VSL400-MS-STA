"""Pure skeleton sequence postprocess (no MediaPipe / video I/O).

Used by D1 extract pipeline and unit tests. Safe to import in CI without mediapipe.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

OUTLIER_ABS_MAX = 50.0
SAVGOL_WINDOW = 5
SAVGOL_POLY = 2


def postprocess_sequence(
    video_data_arr: np.ndarray,
    raw_pose_history_arr: np.ndarray,
    missing_arr: np.ndarray,
    *,
    outlier_abs_max: float = OUTLIER_ABS_MAX,
    savgol_window: int = SAVGOL_WINDOW,
    savgol_poly: int = SAVGOL_POLY,
) -> np.ndarray | None:
    """Fill missing joints + shoulder refs, normalize, smooth. Testable without video I/O."""
    if video_data_arr.ndim != 3 or video_data_arr.shape[1:] != (75, 3):
        raise ValueError(f"bad joints shape {video_data_arr.shape}")
    t_len = int(video_data_arr.shape[0])
    if t_len < 5:
        return None
    video_data_arr = np.array(video_data_arr, dtype=np.float32, copy=True)
    missing_arr = np.array(missing_arr, dtype=bool, copy=True)
    raw_pose_history_arr = np.array(raw_pose_history_arr, dtype=np.float32, copy=True)

    # Track joints never detected (cannot backfill from any real observation).
    never_detected = ~np.any(~missing_arr, axis=0)  # (75,)

    # Per-joint: backfill leading gaps from first valid, then forward-fill.
    for j in range(75):
        if never_detected[j]:
            continue
        valid = ~missing_arr[:, j]
        first = int(np.argmax(valid))
        if first > 0:
            video_data_arr[:first, j, :] = video_data_arr[first, j, :]
            missing_arr[:first, j] = False
        for t in range(1, t_len):
            if missing_arr[t, j]:
                video_data_arr[t, j, :] = video_data_arr[t - 1, j, :]
                missing_arr[t, j] = False

    pose_missing = missing_arr[:, 0].copy()
    left_shoulders = raw_pose_history_arr[:, 11, :].copy()
    right_shoulders = raw_pose_history_arr[:, 12, :].copy()
    dist0 = np.linalg.norm(left_shoulders - right_shoulders, axis=1)
    valid_pose = (~pose_missing) & (dist0 > 1e-3)
    if not np.any(valid_pose):
        return None
    first = int(np.argmax(valid_pose))
    if first > 0:
        left_shoulders[:first] = left_shoulders[first]
        right_shoulders[:first] = right_shoulders[first]
    for t in range(1, t_len):
        if (not valid_pose[t]) or np.linalg.norm(left_shoulders[t] - right_shoulders[t]) <= 1e-3:
            left_shoulders[t] = left_shoulders[t - 1]
            right_shoulders[t] = right_shoulders[t - 1]

    mid_shoulders = (left_shoulders + right_shoulders) / 2.0
    shoulder_distances = np.linalg.norm(left_shoulders - right_shoulders, axis=1, keepdims=True)
    if np.any(shoulder_distances <= 1e-3):
        return None

    for t in range(t_len):
        video_data_arr[t] = (video_data_arr[t] - mid_shoulders[t]) / shoulder_distances[t]

    # Never-detected joints → body-centered origin (not fake absolute zeros after norm).
    if np.any(never_detected):
        video_data_arr[:, never_detected, :] = 0.0

    if not np.isfinite(video_data_arr).all():
        return None
    if float(np.max(np.abs(video_data_arr))) > float(outlier_abs_max):
        return None

    n_joints, n_dims = video_data_arr.shape[1], video_data_arr.shape[2]
    flat_features = video_data_arr.reshape(t_len, n_joints * n_dims)
    window = int(savgol_window)
    if window % 2 == 0:
        window += 1
    if t_len < window:
        return video_data_arr
    smoothed_flat = savgol_filter(
        flat_features, window_length=window, polyorder=int(savgol_poly), axis=0
    )
    return smoothed_flat.reshape(t_len, n_joints, n_dims)


# Provenance keys written into D1 end manifests / promote checks.
POSTPROCESS_PROVENANCE = {
    "shoulder_fill": True,
    "leading_joint_backfill": True,
    "never_detected_to_origin": True,
    "outlier_abs_max": OUTLIER_ABS_MAX,
    "savgol_window": SAVGOL_WINDOW,
    "savgol_poly": SAVGOL_POLY,
    "shape_contract": "(T, 75, 3)",
}
