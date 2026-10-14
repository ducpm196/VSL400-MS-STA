#!/usr/bin/env python3
"""Build R7B freeze: exact same 11 R7 windows + empirical absent-wrist fixture."""

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
    PILOT_R7_VISUAL_FREEZE_REL,
    load_json,
    validate_r7_visual_freeze_list,
    write_json,
)


def main() -> int:
    parent = load_json(_ROOT / PILOT_R7_VISUAL_FREEZE_REL)
    windows = []
    for w in parent["windows"]:
        row = dict(w)
        row["r7b_role"] = row.get("r7_role")
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
            "test": "test_pose_prior_area_respects_geometry_lock",
            "expected": "pose_prior_area_frac_le_0_08",
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
        "schema": "ms_sta_v5_d0_r7b_visual_qa_freeze_list_v1",
        "frozen_before_algorithm": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": "exact_r7_11_carryforward_no_swap",
        "parent_r7_freeze_rel": PILOT_R7_VISUAL_FREEZE_REL,
        "n_windows": 11,
        "windows": windows,
        "truly_absent_negative_controls": negs,
        "pose_prior_geometry_lock_ref": "Docs/results/ms_sta_v5_d0/D0_FULL_PROTOCOL_R7B_GATE_HARDENING_AMENDMENT.json",
        "notes": [
            "Exact same 11 windows as R7; no sample swap.",
            "Empirical blank-video fixture required (not unit-only).",
            "Frozen before R7B geometry/validator replay extract.",
        ],
    }
    validate_r7_visual_freeze_list(obj)
    write_json(_ROOT / PILOT_R7B_VISUAL_FREEZE_REL, obj, overwrite=True)
    print(json.dumps({"ok": True, "n": 11, "n_neg": len(negs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
