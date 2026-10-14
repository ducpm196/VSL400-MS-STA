"""E4-R4 Skeleton compact transform from sealed 75-joint observations (read-only).

Primary campaign path: sealed B4 skeleton_v3 → layout subset. No raw re-extract here.
No Python assert (safe under python -O).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data_prep.e4_r4_joint_layouts import layout_indices, validate_layout_invariants

REQUIRED_KEYS = (
    "coordinates",
    "velocity",
    "validity",
    "confidence",
    "missing_mask_prefill",
    "original_frame_index",
    "timestamps_sec",
    "padding_mask",
    "sequence_length",
    "joint_part",
    "joint_chirality",
)


def compact_skeleton_arrays(
    sealed: dict[str, np.ndarray],
    layout_id: str,
) -> dict[str, Any]:
    """Slice sealed 75-joint tensors to a compact layout."""
    validate_layout_invariants()
    idxs = np.asarray(layout_indices(layout_id), dtype=np.int64)
    if idxs.ndim != 1 or idxs.size == 0:
        raise RuntimeError(f"empty_layout:{layout_id}")

    for key in REQUIRED_KEYS:
        if key not in sealed:
            raise RuntimeError(f"missing_sealed_key:{key}")

    coords = np.asarray(sealed["coordinates"])
    if coords.ndim != 3 or coords.shape[1] != 75 or coords.shape[2] != 3:
        raise RuntimeError(f"bad_coordinates_shape:{coords.shape}")

    t = int(coords.shape[0])
    out: dict[str, Any] = {
        "layout_id": np.asarray(layout_id),
        "source_joint_indices": idxs.astype(np.int32),
        "coordinates": coords[:, idxs, :].astype(np.float32, copy=False),
        "velocity": np.asarray(sealed["velocity"], dtype=np.float32)[:, idxs, :],
        "validity": np.asarray(sealed["validity"], dtype=np.float32)[:, idxs],
        "confidence": np.asarray(sealed["confidence"], dtype=np.float32)[:, idxs],
        "missing_mask_prefill": np.asarray(sealed["missing_mask_prefill"], dtype=bool)[:, idxs],
        "original_frame_index": np.asarray(sealed["original_frame_index"], dtype=np.int32),
        "timestamps_sec": np.asarray(sealed["timestamps_sec"], dtype=np.float64),
        "padding_mask": np.asarray(sealed["padding_mask"], dtype=bool),
        "sequence_length": np.asarray(sealed["sequence_length"], dtype=np.int32),
        "joint_part": np.asarray(sealed["joint_part"])[idxs],
        "joint_chirality": np.asarray(sealed["joint_chirality"])[idxs],
        "n_joints": np.asarray(int(idxs.size), dtype=np.int32),
        "source_n_joints": np.asarray(75, dtype=np.int32),
        "source_mode": np.asarray("sealed_75joint_read_only_compact_transform"),
    }

    # Optional observation provenance if present on sealed source.
    for opt in (
        "observation_source_left_hand",
        "observation_source_right_hand",
        "wrist_assignment_distance_left",
        "wrist_assignment_distance_right",
        "raw_holistic_missing",
        "effective_missing",
    ):
        if opt in sealed:
            arr = np.asarray(sealed[opt])
            if arr.ndim >= 2 and arr.shape[1] == 75:
                out[opt] = arr[:, idxs]
            else:
                out[opt] = arr

    if out["coordinates"].shape != (t, idxs.size, 3):
        raise RuntimeError("compact_shape_mismatch")
    if np.isnan(out["coordinates"]).any() or np.isinf(out["coordinates"]).any():
        raise RuntimeError("compact_coords_nan_inf")
    if len(out["original_frame_index"]) != t or len(out["timestamps_sec"]) != t:
        raise RuntimeError("alignment_length_mismatch")
    return out


def load_sealed_skeleton(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_sealed_skel:{path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_compact_npz_atomic(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.writing.npz"
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    # Validate before replace
    with np.load(tmp, allow_pickle=False) as z:
        if "coordinates" not in z.files:
            raise RuntimeError("tmp_missing_coordinates")
        coords = z["coordinates"]
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise RuntimeError(f"tmp_bad_coords:{coords.shape}")
        if np.isnan(coords).any() or np.isinf(coords).any():
            raise RuntimeError("tmp_coords_nan_inf")
    tmp.replace(path)


def compact_one_sample(
    sealed_path: Path,
    out_path: Path,
    layout_id: str,
) -> dict[str, Any]:
    sealed = load_sealed_skeleton(sealed_path)
    arrays = compact_skeleton_arrays(sealed, layout_id)
    write_compact_npz_atomic(out_path, arrays)
    return {
        "layout_id": layout_id,
        "n_joints": int(arrays["n_joints"]),
        "T": int(arrays["coordinates"].shape[0]),
        "bytes": out_path.stat().st_size,
    }
