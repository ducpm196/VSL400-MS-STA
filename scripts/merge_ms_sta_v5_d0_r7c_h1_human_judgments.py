#!/usr/bin/env python3
"""Merge a complete human judgments file onto the H1 null template.

Never invents PASS/True. Every judgment field must already be filled in the
human file. Rejects retired H fill-script note fingerprints.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_R7C_H_VISUAL_MANUAL_SHA256,
    PILOT_R7C_H1_PRIOR_JUDGMENTS_REL,
    PILOT_R7C_H1_VISUAL_MANUAL_REL,
    PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL,
    PILOT_R7C_OVERLAYS_INDEX_REL,
    PILOT_R7C_VISUAL_FREEZE_REL,
    PILOT_R7C_VISUAL_REL,
    SCRIPTED_PRIOR_TILE_NOTE_MARKERS,
    load_json,
    sha256_file,
    validate_manual_template_null_prior_slots,
    validate_r7_visual_qa_manual,
    write_json,
)


def _slot_key(j: dict, *, center: bool) -> tuple:
    if center:
        return ("center", None, None, j.get("hand"), j.get("source"))
    return (j.get("pathway"), j.get("t_idx"), j.get("native_frame"), j.get("hand"), j.get("source"))


def main() -> int:
    tmpl = load_json(_ROOT / PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL)
    validate_manual_template_null_prior_slots(tmpl)
    human = load_json(_ROOT / PILOT_R7C_H1_PRIOR_JUDGMENTS_REL)
    if human.get("schema") != "ms_sta_v5_d0_r7c_h1_prior_tile_judgments_v1":
        raise RuntimeError("bad_human_judgments_schema")
    if human.get("synthesized_by_fill_script") is not False:
        raise RuntimeError("human_judgments_must_declare_not_fill_script")
    if not str(human.get("reviewer") or "").strip():
        raise RuntimeError("human_judgments_reviewer_required")
    if not str(human.get("reviewed_at") or "").strip():
        raise RuntimeError("human_judgments_reviewed_at_required")
    if int(human.get("n_prior_judgments") or -1) != 23:
        raise RuntimeError("human_judgments_must_declare_exact_23")

    row_map = {
        (r["sample_id"], int(r["window_start_idx"]), r["qa_kind"]): r
        for r in (human.get("row_judgments") or [])
    }
    prior_by_row: dict[tuple, dict[tuple, dict]] = {}
    for item in human.get("prior_tile_judgments") or []:
        rkey = (item["sample_id"], int(item["window_start_idx"]), item["qa_kind"])
        center = item["qa_kind"] == "center_overlay"
        skey = _slot_key(item, center=center)
        prior_by_row.setdefault(rkey, {})[skey] = item

    out_rows = []
    for row in tmpl["inspections"]:
        key = (row["sample_id"], int(row["window_start_idx"]), row["qa_kind"])
        if key not in row_map:
            raise RuntimeError(f"missing_row_judgment:{key}")
        src = row_map[key]
        filled = copy.deepcopy(row)
        for k in (
            "box_on_correct_body_region",
            "handedness_ok",
            "visible_but_missing_consistent",
            "verdict",
            "notes",
        ):
            if src.get(k) is None and k != "notes":
                raise RuntimeError(f"row_judgment_null:{key}:{k}")
            filled[k] = src[k]
        expected_slots = filled.get("prior_tile_judgments") or []
        human_slots = prior_by_row.get(key, {})
        merged = []
        for slot in expected_slots:
            center = filled["qa_kind"] == "center_overlay"
            skey = _slot_key(slot, center=center)
            if skey not in human_slots:
                raise RuntimeError(f"missing_prior_judgment:{key}:{skey}")
            hj = human_slots[skey]
            for ck in ("box_on_correct_body_region", "handedness_ok", "verdict"):
                if hj.get(ck) is None:
                    raise RuntimeError(f"prior_judgment_null:{key}:{skey}:{ck}")
            note = str(hj.get("notes") or "").strip()
            if not note:
                raise RuntimeError(f"prior_judgment_notes_required:{key}:{skey}")
            if any(m in note for m in SCRIPTED_PRIOR_TILE_NOTE_MARKERS):
                raise RuntimeError(f"scripted_prior_note_forbidden:{key}:{skey}")
            j = dict(slot)
            j["box_on_correct_body_region"] = hj["box_on_correct_body_region"]
            j["handedness_ok"] = hj["handedness_ok"]
            j["verdict"] = hj["verdict"]
            j["notes"] = note
            merged.append(j)
        if len(human_slots) != len(expected_slots):
            raise RuntimeError(f"prior_judgment_count_mismatch:{key}")
        filled["prior_tile_judgments"] = merged
        out_rows.append(filled)

    n_prior = sum(len(r.get("prior_tile_judgments") or []) for r in out_rows)
    if n_prior != 23:
        raise RuntimeError(f"expected_23_prior_got_{n_prior}")

    manual = {
        "schema": "ms_sta_v5_d0_r7c_h1_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "synthesized_by_finalizer": False,
        "synthesized_by_template_tool": False,
        "synthesized_by_fill_script": False,
        "status": "HUMAN_REVIEW_COMPLETE",
        "reviewer": human["reviewer"],
        "reviewed_at": human.get("reviewed_at")
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": "D0_PILOT_R7C_H1",
        "freeze_list_rel": PILOT_R7C_VISUAL_FREEZE_REL,
        "auto_stroke_color_sufficient": False,
        "acceptance_rule": "primary_and_holdout_all_pass_zero_fail_allowed",
        "method": human.get("method")
        or (
            "R7C-H1 genuine manual review: inspected each overlay image and entered "
            "independent prior_tile_judgments; fill script disabled."
        ),
        "n_pass": sum(1 for r in out_rows if r["verdict"] == "PASS"),
        "n_fail": sum(1 for r in out_rows if r["verdict"] == "FAIL"),
        "parent_r7c_h_visual_manual_sha256": EXPECTED_R7C_H_VISUAL_MANUAL_SHA256,
        "human_judgments_sha256": sha256_file(_ROOT / PILOT_R7C_H1_PRIOR_JUDGMENTS_REL),
        "inspections": out_rows,
    }
    write_json(_ROOT / PILOT_R7C_H1_VISUAL_MANUAL_REL, manual, overwrite=True)
    overlays = load_json(_ROOT / PILOT_R7C_OVERLAYS_INDEX_REL)["overlays_index"]
    freeze = load_json(_ROOT / PILOT_R7C_VISUAL_FREEZE_REL)
    result = validate_r7_visual_qa_manual(
        manual,
        overlays_index=overlays,
        freeze_list=freeze,
        root=_ROOT,
        visual_dir_rel=PILOT_R7C_VISUAL_REL,
        require_exact_22=True,
        require_prior_tile_judgments=True,
    )
    print(json.dumps({**result, "n_prior_judgments": n_prior}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
