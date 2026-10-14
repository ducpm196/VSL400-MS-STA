#!/usr/bin/env python3
"""Emit R7B visual QA TEMPLATE only — never assigns PASS/FAIL or check booleans."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    PILOT_R7B_OVERLAYS_INDEX_REL,
    PILOT_R7B_VISUAL_FREEZE_REL,
    PILOT_R7B_VISUAL_MANUAL_TEMPLATE_REL,
    load_json,
    write_json,
)


def main() -> int:
    freeze = load_json(_ROOT / PILOT_R7B_VISUAL_FREEZE_REL)
    overlays = load_json(_ROOT / PILOT_R7B_OVERLAYS_INDEX_REL)["overlays_index"]
    role_by = {
        (w["sample_id"], int(w["window_start_idx"])): w.get("r7_role") or w.get("r7b_role")
        for w in freeze["windows"]
    }
    inspections = []
    for o in overlays:
        key = (o["sample_id"], int(o["window_start_idx"]))
        if key not in role_by:
            continue
        row = {
            "sample_id": o["sample_id"],
            "qa_kind": o["kind"],
            "path": o["path"],
            "sha256": o["sha256"],
            "center_frame": o.get("center_frame"),
            "native_frame": o.get("native_frame"),
            "landmark_frame": o.get("landmark_frame"),
            "window_start_idx": o["window_start_idx"],
            "r7_role": role_by[key],
            "boxes": o.get("boxes"),
            "missing_regions": o.get("missing_regions"),
            "rgb_roi_sources": o.get("rgb_roi_sources"),
            # Reviewer MUST fill — tool leaves nulls (no default PASS).
            "box_on_correct_body_region": None,
            "handedness_ok": None,
            "visible_but_missing_consistent": None,
            "verdict": None,
            "notes": "",
            "derived_from_overlay_render_integrity_only": False,
        }
        if o["kind"] == "temporal_contact_sheet":
            row["n_tiles"] = o.get("n_tiles")
            row["slice_native_frames"] = o.get("slice_native_frames")
            row["tiles"] = o.get("tiles")
        inspections.append(row)

    template = {
        "schema": "ms_sta_v5_d0_r7b_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "synthesized_by_finalizer": False,
        "synthesized_by_template_tool": True,
        "status": "TEMPLATE_AWAITING_HUMAN_REVIEW",
        "reviewer": "",
        "reviewed_at": "",
        "campaign": "D0_PILOT_R7B",
        "freeze_list_rel": PILOT_R7B_VISUAL_FREEZE_REL,
        "auto_stroke_color_sufficient": False,
        "acceptance_rule": "primary_and_holdout_all_pass_zero_fail_allowed",
        "arbitrary_wrong_box_negative_accepted": False,
        "method": "",
        "n_pass": None,
        "n_fail": None,
        "instructions": [
            "Copy this template to D0_R7B_VISUAL_QA_MANUAL.json",
            "Set synthesized_by_template_tool=false after human fill",
            "Fill reviewer, reviewed_at, method",
            "For EACH of 22 rows set the three checks and verdict independently — no batch PASS",
            "Finalize refuses templates and null verdicts",
        ],
        "inspections": inspections,
    }
    write_json(_ROOT / PILOT_R7B_VISUAL_MANUAL_TEMPLATE_REL, template, overwrite=True)
    print(
        json.dumps(
            {
                "ok": True,
                "template": PILOT_R7B_VISUAL_MANUAL_TEMPLATE_REL,
                "n_rows": len(inspections),
                "verdicts_assigned": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
