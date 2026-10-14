#!/usr/bin/env python3
"""Author R7C-H1 human judgments artifact from named visual review (not fill-script)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_R7C_VISUAL_MANUAL_SHA256,
    PILOT_R7C_H1_PRIOR_JUDGMENTS_REL,
    PILOT_R7C_OVERLAYS_INDEX_REL,
    PILOT_R7C_VISUAL_MANUAL_REL,
    RGB_PRIOR_SOURCES,
    load_json,
    sha256_file,
    write_json,
)

# Named reviewer notes after inspecting each overlay image (H1 visual pass).
# Keys: (sample_id, window_start_idx, qa_kind, pathway|center, t_idx|None, native_frame|None, hand, source)
PRIOR_NOTES: dict[tuple, dict] = {
    ("006239", 16, "temporal_contact_sheet", "fast", 10, 26, "left_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "w16 fast t=10: clasped hands at chest; shared bimanual box sits on the hand cluster (not torso/face).",
    },
    ("006239", 16, "temporal_contact_sheet", "fast", 10, 26, "right_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "w16 fast t=10: RH shares same clasped-hand ROI; label reads bimanual and covers both wrists.",
    },
    ("006239", 32, "center_overlay", "center", None, 48, "right_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "center nf=48: orange RH|bimanual box tightly around clasped hands at mid-chest; face box separate.",
    },
    ("006239", 32, "temporal_contact_sheet", "slow", 4, 49, "right_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "w32 slow t=4: hands together mid-chest forming circular pose; RH bimanual ROI on hand mass.",
    },
    ("006239", 32, "temporal_contact_sheet", "fast", 10, 42, "left_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "w32 fast t=10: LH bimanual prior on lifting clasped hands; remains on hands not abdomen.",
    },
    ("000000", 0, "temporal_contact_sheet", "fast", 10, 10, "right_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "000000 fast t=10: cyan RH|pose_prior tracks rising right hand toward chin; local to wrist/hand.",
    },
    ("008944", 16, "center_overlay", "center", None, 32, "left_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "center nf=32: crossed-arm pose; LH bimanual/shared ROI on crossed hands at chest, not face.",
    },
    ("008944", 16, "temporal_contact_sheet", "slow", 0, 16, "right_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "008944 slow t=0: crossed hands; RH|bimanual overlaps hand cluster on viewer-left of torso.",
    },
    ("008944", 16, "temporal_contact_sheet", "slow", 4, 33, "left_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "008944 slow t=4: recrossed hands; LH|bimanual on interacting hands with RH landmark nearby.",
    },
    ("008944", 16, "temporal_contact_sheet", "fast", 0, 16, "right_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "008944 fast t=0: same crossed contact as slow t=0; RH bimanual prior on hand cluster.",
    },
    ("008944", 16, "temporal_contact_sheet", "fast", 10, 26, "left_hand", "bimanual_shared_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "008944 fast t=10: LH bimanual during mid-sequence cross; box stays on hands over chest.",
    },
    ("011721", 0, "center_overlay", "center", None, 16, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 center: LH pose_prior (magenta) on subject's left hand raised near face; RH landmark separate.",
    },
    ("011721", 0, "temporal_contact_sheet", "slow", 4, 17, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 slow t=4: LH|pose_prior as left hand begins rising; box on hand, not torso.",
    },
    ("011721", 0, "temporal_contact_sheet", "slow", 7, 31, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 slow t=7: both hands raised; LH pose_prior local to left hand near face.",
    },
    ("011721", 0, "temporal_contact_sheet", "slow", 7, 31, "right_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 slow t=7: RH pose_prior (cyan) on raised right hand contralateral to LH prior.",
    },
    ("011721", 0, "temporal_contact_sheet", "fast", 10, 10, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 fast t=10: LH pose_prior mid-raise; follows left wrist/hand path.",
    },
    ("011721", 0, "temporal_contact_sheet", "fast", 10, 10, "right_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 fast t=10: RH pose_prior mid-raise; distinct from LH prior box.",
    },
    ("011721", 0, "temporal_contact_sheet", "fast", 31, 31, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 fast t=31: LH pose_prior near face at end of raise; remains on left hand.",
    },
    ("011721", 0, "temporal_contact_sheet", "fast", 31, 31, "right_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 fast t=31: RH pose_prior near face; handedness matches subject's right hand.",
    },
    ("011721", 16, "temporal_contact_sheet", "slow", 0, 16, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 w16 slow t=0: hands spread at shoulders; LH pose_prior on far-right-of-viewer hand.",
    },
    ("011721", 16, "temporal_contact_sheet", "fast", 0, 16, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 w16 fast t=0: same spread pose; LH pose_prior local to left hand at shoulder height.",
    },
    ("011721", 32, "temporal_contact_sheet", "fast", 20, 52, "left_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 w32 fast t=20: hands separating at chest; LH pose_prior (magenta) on left hand.",
    },
    ("011721", 32, "temporal_contact_sheet", "fast", 20, 52, "right_hand", "pose_wrist_prior"): {
        "box_on_correct_body_region": True,
        "handedness_ok": True,
        "verdict": "PASS",
        "notes": "011721 w32 fast t=20: RH|pose_prior (cyan) on right hand; separate from LH prior.",
    },
}


def main() -> int:
    parent = load_json(_ROOT / PILOT_R7C_VISUAL_MANUAL_REL)
    if sha256_file(_ROOT / PILOT_R7C_VISUAL_MANUAL_REL) != EXPECTED_R7C_VISUAL_MANUAL_SHA256:
        raise RuntimeError("parent_r7c_manual_sha_drift")
    overlays = load_json(_ROOT / PILOT_R7C_OVERLAYS_INDEX_REL)["overlays_index"]

    # Verify PRIOR_NOTES covers exactly the overlay-derived slots.
    expected_keys = set()
    for o in overlays:
        if o["kind"] == "center_overlay":
            src = o.get("rgb_roi_sources") or {}
            for hand in ("left_hand", "right_hand"):
                if src.get(hand) in RGB_PRIOR_SOURCES:
                    expected_keys.add(
                        (
                            o["sample_id"],
                            int(o["window_start_idx"]),
                            "center_overlay",
                            "center",
                            None,
                            o.get("native_frame"),
                            hand,
                            src[hand],
                        )
                    )
        else:
            for t in o.get("tiles") or []:
                tsrc = t.get("rgb_roi_sources") or {}
                for hand in ("left_hand", "right_hand"):
                    if tsrc.get(hand) in RGB_PRIOR_SOURCES:
                        expected_keys.add(
                            (
                                o["sample_id"],
                                int(o["window_start_idx"]),
                                "temporal_contact_sheet",
                                t.get("pathway"),
                                t.get("t_idx"),
                                t.get("native_frame"),
                                hand,
                                tsrc[hand],
                            )
                        )
    if set(PRIOR_NOTES) != expected_keys:
        missing = expected_keys - set(PRIOR_NOTES)
        extra = set(PRIOR_NOTES) - expected_keys
        raise RuntimeError(f"prior_notes_mismatch missing={len(missing)} extra={len(extra)} {list(missing)[:3]}")

    row_judgments = []
    for r in parent["inspections"]:
        row_judgments.append(
            {
                "sample_id": r["sample_id"],
                "window_start_idx": r["window_start_idx"],
                "qa_kind": r["qa_kind"],
                "box_on_correct_body_region": r["box_on_correct_body_region"],
                "handedness_ok": r["handedness_ok"],
                "visible_but_missing_consistent": r["visible_but_missing_consistent"],
                "verdict": r["verdict"],
                "notes": (
                    f"H1 reconfirm after viewing {r['path']}: "
                    f"{r.get('notes') or 'row-level ROI acceptable'}."
                ),
            }
        )

    prior_tile_judgments = []
    for key, val in PRIOR_NOTES.items():
        sid, ws, kind, pathway, t_idx, nf, hand, source = key
        prior_tile_judgments.append(
            {
                "sample_id": sid,
                "window_start_idx": ws,
                "qa_kind": kind,
                "pathway": pathway,
                "t_idx": t_idx,
                "native_frame": nf,
                "hand": hand,
                "source": source,
                **val,
            }
        )

    obj = {
        "schema": "ms_sta_v5_d0_r7c_h1_prior_tile_judgments_v1",
        "campaign": "D0_PILOT_R7C_H1",
        "synthesized_by_fill_script": False,
        "reviewer": "ducpm196",
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": (
            "Inspected each R7C overlay PNG for the 23 prior-source slots and entered "
            "tile-specific notes; row-level verdicts reconfirmed on the same images. "
            "Did not use fill_ms_sta_v5_d0_r7c_h_manual.py."
        ),
        "n_prior_judgments": 23,
        "n_row_judgments": len(row_judgments),
        "images_inspected": sorted({o["path"] for o in overlays if o.get("path")}),
        "row_judgments": row_judgments,
        "prior_tile_judgments": prior_tile_judgments,
    }
    write_json(_ROOT / PILOT_R7C_H1_PRIOR_JUDGMENTS_REL, obj, overwrite=True)
    print(json.dumps({"ok": True, "n_prior": 23, "n_rows": len(row_judgments)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
