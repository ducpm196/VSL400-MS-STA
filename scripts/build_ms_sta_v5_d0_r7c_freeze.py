#!/usr/bin/env python3
"""Build R7C freeze: exact same 11 windows + empirical absent-wrist fixture."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    PILOT_R7B_FIXTURE_ABSENT_REL,
    PILOT_R7B_VISUAL_FREEZE_REL,
    PILOT_R7C_VISUAL_FREEZE_REL,
    R7C_AMBIGUITY_ROI_LOCK,
    load_json,
    validate_r7_visual_freeze_list,
    write_json,
)


def main() -> int:
    parent = load_json(_ROOT / PILOT_R7B_VISUAL_FREEZE_REL)
    windows = []
    for w in parent["windows"]:
        row = dict(w)
        row["r7c_role"] = row.get("r7_role") or row.get("r7b_role")
        windows.append(row)
    if len(windows) != 11:
        raise RuntimeError(f"expected_11_got_{len(windows)}")
    negs = [
        {
            "kind": "unit_backed",
            "test": "test_pose_prior_absent_when_wrist_missing",
            "expected": "no_pose_prior_box_when_pose_wrist_missing",
        },
        {
            "kind": "unit_backed",
            "test": "test_bimanual_shared_prior_when_wrists_close",
            "expected": "close_wrists_use_bimanual_shared_prior_source",
        },
        {
            "kind": "unit_backed",
            "test": "test_single_hand_prior_rejects_contralateral_inside",
            "expected": "single_hand_prior_fails_closed_if_other_wrist_inside",
        },
        {
            "kind": "empirical_fixture",
            "sample_id": "NEG_ABSENT_WRIST",
            "window_start_idx": 0,
            "fixture_rel": PILOT_R7B_FIXTURE_ABSENT_REL,
            "proof_rel": f"{PILOT_R7B_FIXTURE_ABSENT_REL}/PROOF.json",
            "expected": "no_pose_prior_box_on_blank_video_all_missing_skel",
        },
    ]
    obj = {
        "schema": "ms_sta_v5_d0_r7c_visual_qa_freeze_list_v1",
        "frozen_before_algorithm": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": "exact_r7_11_carryforward_no_swap",
        "parent_r7b_freeze_rel": PILOT_R7B_VISUAL_FREEZE_REL,
        "n_windows": 11,
        "windows": windows,
        "truly_absent_negative_controls": negs,
        "ambiguity_roi_lock": R7C_AMBIGUITY_ROI_LOCK,
        "notes": [
            "Exact same 11 windows as R7/R7B; no sample swap.",
            "Ambiguity-aware single/bimanual lock frozen before algo replay.",
        ],
    }
    validate_r7_visual_freeze_list(obj)
    write_json(_ROOT / PILOT_R7C_VISUAL_FREEZE_REL, obj, overwrite=True)
    print(json.dumps({"ok": True, "n": 11, "n_neg": len(negs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
