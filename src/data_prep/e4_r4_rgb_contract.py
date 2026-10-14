"""E4-R4 RGB input contract — pixel-budget compact candidates (one backbone pass).

R0: single global 192x192.
R1: 2x2 ROI mosaic on one 160x160 canvas (one backbone call).
R2: global + one composite signing crop, total pixels <= R0, shared batching measured.
R3: true 1-ch luma ablation (not default promote).

Candidates with theoretical work > R0 are marked NON_COMPACT and must not promote.
"""

from __future__ import annotations

from typing import Any, Final

RGB_CANDIDATES: Final[dict[str, dict[str, Any]]] = {
    "R0": {
        "kind": "global_rgb",
        "channels": 3,
        "frames_T_max": 32,
        "spatial_per_view": (192, 192),
        "views": ("global",),
        "backbone_calls_per_frame": 1,
        "canvas": "single_global",
        "crop_construction": "letterbox_center_global",
        "resize_interpolation": "bilinear",
        "color_space": "RGB",
        "bgr_to_rgb": True,
        "normalization": "imagenet_or_registered_mean_std",
        "missing_roi_policy": "n/a",
        "validity_mask": "temporal_pad_zero_mask",
        "role": "compact_baseline",
    },
    "R1": {
        "kind": "roi_mosaic_2x2",
        "channels": 3,
        "frames_T_max": 32,
        "spatial_per_view": (80, 80),  # 2x2 tiles → 160x160 canvas
        "canvas_spatial": (160, 160),
        "views": ("global", "upper_body", "left_hand", "right_hand"),
        "backbone_calls_per_frame": 1,
        "canvas": "mosaic_2x2_one_pass",
        "crop_construction": "four_rois_resized_into_2x2_tiles",
        "resize_interpolation": "bilinear",
        "color_space": "RGB",
        "bgr_to_rgb": True,
        "normalization": "imagenet_or_registered_mean_std",
        "missing_roi_policy": "zero_tile_plus_tile_validity_bit",
        "validity_mask": "temporal_and_tile_validity",
        "role": "compact_candidate",
    },
    "R2": {
        "kind": "global_plus_signing_composite",
        "channels": 3,
        "frames_T_max": 32,
        "spatial_per_view": (128, 128),
        "views": ("global", "signing_composite"),
        "backbone_calls_per_frame": 1,  # stacked as batch=2 then mean/concat — measured in feasibility
        "batch_views_in_one_forward": True,
        "canvas": "two_views_shared_backbone_batch",
        "crop_construction": "global_letterbox_plus_union_hand_torso_crop",
        "resize_interpolation": "bilinear",
        "color_space": "RGB",
        "bgr_to_rgb": True,
        "normalization": "imagenet_or_registered_mean_std",
        "missing_roi_policy": "zero_signing_view_plus_validity",
        "validity_mask": "temporal_and_view_validity",
        "role": "compact_candidate",
        "pixel_budget_must_le_r0": True,
    },
    "R3": {
        "kind": "true_luma_1ch",
        "channels": 1,
        "frames_T_max": 32,
        "spatial_per_view": (160, 160),
        "views": ("global",),
        "backbone_calls_per_frame": 1,
        "first_conv_in_channels": 1,
        "forbid_replicate_to_3ch": True,
        "canvas": "single_global_luma",
        "crop_construction": "letterbox_center_global",
        "resize_interpolation": "bilinear",
        "color_space": "LUMA_BT601",
        "bgr_to_rgb": True,
        "normalization": "luma_mean_std_registered",
        "missing_roi_policy": "n/a",
        "validity_mask": "temporal_pad_zero_mask",
        "role": "ablation_only_not_default_promote",
    },
}


def require_rgb_candidate(cid: str) -> dict[str, Any]:
    """Fail-closed RGB candidate contract (no Python assert; safe under -O)."""
    if cid not in RGB_CANDIDATES:
        raise KeyError(cid)
    spec = RGB_CANDIDATES[cid]
    if cid == "R3":
        if int(spec["channels"]) != 1 or int(spec.get("first_conv_in_channels", 0)) != 1:
            raise RuntimeError("r3_must_be_true_1ch")
        if not spec.get("forbid_replicate_to_3ch", False):
            raise RuntimeError("r3_must_forbid_3ch_replicate")
    else:
        if int(spec["channels"]) != 3:
            raise RuntimeError("color_candidates_must_be_3ch")
    if int(spec["backbone_calls_per_frame"]) != 1:
        raise RuntimeError(f"{cid}_must_use_one_backbone_call_per_frame")
    if cid == "R1" and spec.get("canvas") != "mosaic_2x2_one_pass":
        raise RuntimeError("r1_must_be_mosaic_one_pass")
    if cid == "R2":
        h, w = spec["spatial_per_view"]
        n_views = len(spec["views"])
        ch = int(spec["channels"])
        r2_px = int(h) * int(w) * ch * n_views
        r0 = RGB_CANDIDATES["R0"]
        rh, rw = r0["spatial_per_view"]
        r0_px = int(rh) * int(rw) * int(r0["channels"])
        if r2_px > r0_px:
            raise RuntimeError("r2_pixel_budget_exceeds_r0")
    return spec


def pixels_per_frame(cid: str) -> int:
    spec = require_rgb_candidate(cid)
    ch = int(spec["channels"])
    if cid == "R1":
        h, w = spec["canvas_spatial"]
        return int(h) * int(w) * ch
    if cid == "R2":
        h, w = spec["spatial_per_view"]
        n_views = len(spec["views"])
        return int(h) * int(w) * ch * n_views
    h, w = spec["spatial_per_view"]
    return int(h) * int(w) * ch


def backbone_calls_per_frame(cid: str) -> int:
    return int(require_rgb_candidate(cid)["backbone_calls_per_frame"])


def theoretical_work_units(cid: str) -> int:
    """Proxy work = pixels_per_frame * backbone_calls (not a latency claim)."""
    return pixels_per_frame(cid) * backbone_calls_per_frame(cid)


def is_non_compact(cid: str) -> bool:
    if cid == "R3":
        return False  # ablation; promote forbidden separately
    return theoretical_work_units(cid) > theoretical_work_units("R0")


def validate_rgb_budget_table() -> dict[str, Any]:
    rows = {}
    r0 = theoretical_work_units("R0")
    for cid in ("R0", "R1", "R2", "R3"):
        require_rgb_candidate(cid)
        work = theoretical_work_units(cid)
        rows[cid] = {
            "pixels_per_frame": pixels_per_frame(cid),
            "backbone_calls_per_frame": backbone_calls_per_frame(cid),
            "theoretical_work_units": work,
            "vs_r0": round(work / r0, 4),
            "non_compact": is_non_compact(cid),
            "promote_forbidden": cid == "R3" or is_non_compact(cid),
        }
    if rows["R1"]["non_compact"]:
        raise RuntimeError("r1_marked_non_compact_after_redesign")
    if rows["R2"]["non_compact"]:
        raise RuntimeError("r2_marked_non_compact_after_redesign")
    if rows["R1"]["pixels_per_frame"] > rows["R0"]["pixels_per_frame"]:
        raise RuntimeError("r1_pixels_exceed_r0")
    return rows


def luma_from_rgb_uint8(rgb: Any) -> Any:
    """True luma (BT.601) → single channel; never tile to 3ch."""
    import numpy as np

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("expected_hxwx3")
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    y = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    return y[..., None]


# Fail-closed at import.
validate_rgb_budget_table()
