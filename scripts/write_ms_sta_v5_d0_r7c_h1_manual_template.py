#!/usr/bin/env python3
"""Emit R7C-H1 visual QA TEMPLATE only — all judgment fields remain null."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL,
    PILOT_R7C_OVERLAYS_INDEX_REL,
    PILOT_R7C_VISUAL_FREEZE_REL,
    RGB_PRIOR_SOURCES,
    load_json,
    validate_manual_template_null_prior_slots,
    write_json,
)


def _prior_slots(kind: str, row: dict) -> list[dict]:
    slots: list[dict] = []
    if kind == "center_overlay":
        src = row.get("rgb_roi_sources") or {}
        for hand in ("left_hand", "right_hand"):
            if src.get(hand) in RGB_PRIOR_SOURCES:
                slots.append(
                    {
                        "pathway": "center",
                        "t_idx": None,
                        "native_frame": row.get("native_frame"),
                        "hand": hand,
                        "source": src.get(hand),
                        "box_on_correct_body_region": None,
                        "handedness_ok": None,
                        "verdict": None,
                        "notes": "",
                    }
                )
    elif kind == "temporal_contact_sheet":
        for tile in row.get("tiles") or []:
            tsrc = tile.get("rgb_roi_sources") or {}
            for hand in ("left_hand", "right_hand"):
                if tsrc.get(hand) in RGB_PRIOR_SOURCES:
                    slots.append(
                        {
                            "pathway": tile.get("pathway"),
                            "t_idx": tile.get("t_idx"),
                            "native_frame": tile.get("native_frame"),
                            "hand": hand,
                            "source": tsrc.get(hand),
                            "box_on_correct_body_region": None,
                            "handedness_ok": None,
                            "verdict": None,
                            "notes": "",
                        }
                    )
    return slots


def main() -> int:
    freeze = load_json(_ROOT / PILOT_R7C_VISUAL_FREEZE_REL)
    overlays = load_json(_ROOT / PILOT_R7C_OVERLAYS_INDEX_REL)["overlays_index"]
    role_by = {
        (w["sample_id"], int(w["window_start_idx"])): (
            w.get("r7_role") or w.get("r7b_role") or w.get("r7c_role")
        )
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
            "box_on_correct_body_region": None,
            "handedness_ok": None,
            "visible_but_missing_consistent": None,
            "verdict": None,
            "notes": "",
            "derived_from_overlay_render_integrity_only": False,
            "prior_tile_judgments": _prior_slots(o["kind"], o),
        }
        if o["kind"] == "temporal_contact_sheet":
            row["n_tiles"] = o.get("n_tiles")
            row["slice_native_frames"] = o.get("slice_native_frames")
            row["tiles"] = o.get("tiles")
        inspections.append(row)
    template = {
        "schema": "ms_sta_v5_d0_r7c_h1_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "synthesized_by_finalizer": False,
        "synthesized_by_template_tool": True,
        "status": "TEMPLATE_AWAITING_HUMAN_REVIEW",
        "reviewer": "",
        "reviewed_at": "",
        "campaign": "D0_PILOT_R7C_H1",
        "freeze_list_rel": PILOT_R7C_VISUAL_FREEZE_REL,
        "auto_stroke_color_sufficient": False,
        "acceptance_rule": "primary_and_holdout_all_pass_zero_fail_allowed",
        "method": "",
        "n_pass": None,
        "n_fail": None,
        "instructions": [
            "Copy to D0_R7C_H1_VISUAL_QA_MANUAL.json OR merge via human judgments file",
            "Set synthesized_by_template_tool=false",
            "Fill reviewer/reviewed_at/method and each of 22 rows after viewing overlays",
            "Fill every prior_tile_judgments entry independently (exact 23); never use fill_ms_sta_v5_d0_r7c_h_manual.py",
        ],
        "inspections": inspections,
    }
    validate_manual_template_null_prior_slots(template)
    write_json(_ROOT / PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL, template, overwrite=True)
    n_prior = sum(len(r.get("prior_tile_judgments") or []) for r in inspections)
    print(json.dumps({"ok": True, "n_rows": len(inspections), "n_prior_slots": n_prior}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
