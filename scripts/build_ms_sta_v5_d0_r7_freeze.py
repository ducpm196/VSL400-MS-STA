#!/usr/bin/env python3
"""Build R7 freeze list (exact 11) + truly-absent negatives before algorithm."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    PILOT_R6_HOLDOUT_FREEZE_REL,
    PILOT_R6_VISUAL_FREEZE_REL,
    PILOT_R7_VISUAL_FREEZE_REL,
    load_json,
    validate_r7_visual_freeze_list,
    write_json,
)


def main() -> int:
    prim = load_json(_ROOT / PILOT_R6_VISUAL_FREEZE_REL)
    hold = load_json(_ROOT / PILOT_R6_HOLDOUT_FREEZE_REL)
    windows = []
    for w in prim["windows"]:
        windows.append({**w, "r7_role": "primary_r5_carryforward"})
    for w in hold["windows"]:
        windows.append({**w, "r7_role": "holdout_r6_carryforward"})
    if len(windows) != 11:
        raise RuntimeError(f"expected_11_got_{len(windows)}")

    # Pilot corpus has essentially no pose-wrist-missing frames. Bind unit-backed
    # negatives that prove pose-prior refuses without a valid wrist and never
    # falsifies skeleton missing bits.
    negs = [
        {
            "kind": "unit_backed",
            "test": "test_pose_wrist_prior_does_not_flip_skeleton_missing",
            "expected": "pose_prior_box_allowed_only_when_wrist_present_skeleton_mask_unchanged",
        },
        {
            "kind": "unit_backed",
            "test": "test_pose_prior_absent_when_wrist_missing",
            "expected": "no_pose_prior_box_when_pose_wrist_missing",
        },
    ]

    obj = {
        "schema": "ms_sta_v5_d0_r7_visual_qa_freeze_list_v1",
        "frozen_before_algorithm": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": "exact_r6_primary7_plus_holdout4",
        "n_windows": 11,
        "windows": windows,
        "truly_absent_negative_controls": negs,
        "notes": [
            "Frozen before R7 algorithm edits.",
            "Primary + holdout zero-fail required.",
            "Truly-absent negatives unit-backed: no wrist => no pose-prior box; skeleton missing untouched.",
        ],
    }
    validate_r7_visual_freeze_list(obj)
    write_json(_ROOT / PILOT_R7_VISUAL_FREEZE_REL, obj, overwrite=True)
    print(json.dumps({"ok": True, "n": 11, "n_neg": len(negs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
