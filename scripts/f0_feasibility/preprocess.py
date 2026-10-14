"""Preprocessing shape assertions for F0 feasibility (CPU, no model load)."""

from __future__ import annotations

from typing import Any

from .contracts import LAYOUT_CONTRACT


def assert_layout_tensor_spec(candidate_id: str, batch: dict[str, Any]) -> None:
    """Validate synthetic batch keys/shapes against the sealed plan layout contract.

    Does not import torch to stay hermetic in CI without GPU deps.
    """
    if candidate_id not in LAYOUT_CONTRACT:
        raise ValueError(f"unknown_candidate:{candidate_id}")
    spec = LAYOUT_CONTRACT[candidate_id]
    for key in spec["paths"]:
        if key not in batch:
            raise ValueError(f"missing_tensor:{candidate_id}:{key}")
        meta = batch[key]
        if not isinstance(meta, dict) or "shape" not in meta:
            raise ValueError(f"tensor_meta_invalid:{candidate_id}:{key}")
        shape = list(meta["shape"])
        layout = spec["layout"]
        if layout == "NTCHW":
            # N,T,C,H,W
            if len(shape) != 5 or shape[2] != 3:
                raise ValueError(f"ntchw_shape_invalid:{candidate_id}:{shape}")
        elif layout == "CTHW":
            # N,C,T,H,W
            if len(shape) != 5 or shape[1] != 3:
                raise ValueError(f"cthw_shape_invalid:{candidate_id}:{shape}")
        elif layout == "CTHW_dual_pathway":
            if len(shape) != 5 or shape[1] != 3:
                raise ValueError(f"dual_pathway_shape_invalid:{candidate_id}:{key}:{shape}")
        else:
            raise ValueError(f"unknown_layout:{layout}")


def synthetic_batch_for(candidate_id: str) -> dict[str, Any]:
    """Return metadata-only synthetic batch (no torch tensors)."""
    if candidate_id.startswith("videomae"):
        return {"pixel_values_ntchw": {"shape": [1, 16, 3, 224, 224], "dtype": "float32"}}
    if candidate_id == "video_swin_tiny":
        return {"video_cthw": {"shape": [1, 3, 32, 224, 224], "dtype": "float32"}}
    if candidate_id == "slowfast_r50_partial_unfreeze":
        return {
            "slow_path_ncthw": {"shape": [1, 3, 8, 224, 224], "dtype": "float32"},
            "fast_path_ncthw": {"shape": [1, 3, 32, 224, 224], "dtype": "float32"},
        }
    raise ValueError(f"unknown_candidate:{candidate_id}")
