# -*- coding: utf-8 -*-
"""Validate E4-R4 pilot extract artifacts (shapes/dtypes/NaN/Inf/padding/alignment)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.e4_r4_joint_layouts import layout_indices
from src.data_prep.e4_r4_rgb_contract import require_rgb_candidate


def die(msg: str) -> None:
    print(f"E4_R4_VALIDATE_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def validate_skel(path: Path, layout: str) -> None:
    idxs = layout_indices(layout)
    with np.load(path, allow_pickle=False) as z:
        coords = z["coordinates"]
        if coords.dtype != np.float32:
            die(f"{path.name}:coords_dtype")
        if coords.ndim != 3 or coords.shape[1] != len(idxs) or coords.shape[2] != 3:
            die(f"{path.name}:coords_shape:{coords.shape}")
        if np.isnan(coords).any() or np.isinf(coords).any():
            die(f"{path.name}:coords_nan_inf")
        t = coords.shape[0]
        for key in ("velocity", "validity", "confidence", "original_frame_index", "timestamps_sec", "padding_mask"):
            if key not in z.files:
                die(f"{path.name}:missing:{key}")
        if z["original_frame_index"].shape[0] != t:
            die(f"{path.name}:frame_index_len")
        if z["timestamps_sec"].shape[0] != t:
            die(f"{path.name}:ts_len")
        if str(z["source_mode"]) != "sealed_75joint_read_only_compact_transform":
            die(f"{path.name}:bad_source_mode")
        if "read_only_or_reextract" in str(z["source_mode"]):
            die(f"{path.name}:ambiguous_source_mode")


def validate_rgb(path: Path, cid: str) -> None:
    spec = require_rgb_candidate(cid)
    with np.load(path, allow_pickle=False) as z:
        ten = z["rgb_tensor"]
        mask = np.asarray(z["temporal_valid"], dtype=bool)
        if ten.dtype != np.float32:
            die(f"{path.name}:rgb_dtype")
        if np.isnan(ten[mask]).any() or np.isinf(ten[mask]).any():
            die(f"{path.name}:rgb_nan_inf")
        t_max = int(spec["frames_T_max"])
        if cid == "R1":
            ch, cw = spec["canvas_spatial"]
            if ten.shape != (t_max, int(ch), int(cw), 3):
                die(f"{path.name}:r1_shape:{ten.shape}")
        elif cid == "R2":
            h, w = spec["spatial_per_view"]
            if ten.shape != (t_max, 2, int(h), int(w), 3):
                die(f"{path.name}:r2_shape:{ten.shape}")
        elif cid == "R3":
            h, w = spec["spatial_per_view"]
            if ten.shape != (t_max, int(h), int(w), 1):
                die(f"{path.name}:r3_shape:{ten.shape}")
        else:
            h, w = spec["spatial_per_view"]
            if ten.shape != (t_max, int(h), int(w), 3):
                die(f"{path.name}:r0_shape:{ten.shape}")
        if int(z["backbone_calls_per_frame"]) != 1:
            die(f"{path.name}:backbone_calls_not_1")
        if "original_frame_index" not in z.files or "timestamps_sec" not in z.files:
            die(f"{path.name}:missing_alignment")
        if "rgb_skel_alignment_delta_sec" not in z.files:
            die(f"{path.name}:missing_alignment_delta")
        # Padding must be zero on invalid timesteps (no duplicate-frame fill)
        invalid = ~mask
        if invalid.any():
            if np.any(ten[invalid] != 0):
                die(f"{path.name}:invalid_timesteps_not_zero_padded")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, default=ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4_PILOT_SAMPLE_FREEZE.json")
    args = ap.parse_args()
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    ids = list(freeze["pilot_train_ids"]) + list(freeze["pilot_val_ids"])

    contract = out_root / "_run_contract.json"
    if not contract.is_file():
        die("missing_run_contract")
    c = json.loads(contract.read_text(encoding="utf-8"))
    layouts = c.get("layouts") or ["S_A", "S_B"]
    rgb_cands = c.get("rgb_candidates") or ["R0", "R1", "R2", "R3"]

    n_ok = 0
    for sid in ids:
        for layout in layouts:
            p = out_root / f"skel_{layout}" / f"{sid}.npz"
            if not p.is_file():
                die(f"missing_skel:{layout}:{sid}")
            validate_skel(p, layout)
            n_ok += 1
        for cid in rgb_cands:
            p = out_root / f"rgb_{cid}" / f"{sid}.npz"
            if not p.is_file():
                die(f"missing_rgb:{cid}:{sid}")
            validate_rgb(p, cid)
            n_ok += 1

    report = {
        "schema": "ms_sta_v6_e4_r4_pilot_validate_v1",
        "status": "PASS",
        "n_artifacts_checked": n_ok,
        "n_ids": len(ids),
        "qa_label": "MODEL_ASSISTED_QA",
        "pilot_val_role": "representation_qa_only_forbid_architecture_selection",
    }
    (out_root / "E4_R4_PILOT_VALIDATE.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("VALIDATE_PASS", n_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
