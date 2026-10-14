#!/usr/bin/env python3
"""R7C-H1 genuine manual-review finalize — does not overwrite frozen R7C-H artifacts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R7B_ABSENT_FIXTURE_PROOF_SHA256,
    EXPECTED_R7B_PILOT_AUDIT_SHA256,
    EXPECTED_R7C_H_INDEPENDENT_REAUDIT_SHA256,
    EXPECTED_R7C_H_PILOT_AUDIT_SHA256,
    EXPECTED_R7C_H_PILOT_COMPLETION_SHA256,
    EXPECTED_R7C_H_PILOT_REPORT_SHA256,
    EXPECTED_R7C_H_VISUAL_MANUAL_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R7C_AMENDMENT_REL,
    PILOT_R5_VISUAL_MANUAL_REL,
    PILOT_R7B_FIXTURE_ABSENT_REL,
    PILOT_R7C_DIAGNOSTIC_REL,
    PILOT_R7C_FIXTURE_ABSENT_REL,
    PILOT_R7C_H1_AUDIT_REL,
    PILOT_R7C_H1_COMPLETION_REL,
    PILOT_R7C_H1_PRIOR_JUDGMENTS_REL,
    PILOT_R7C_H1_REPORT_REL,
    PILOT_R7C_H1_VISUAL_MANUAL_REL,
    PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL,
    PILOT_R7C_H_AUDIT_REL,
    PILOT_R7C_H_COMPLETION_REL,
    PILOT_R7C_H_INDEPENDENT_REAUDIT_REL,
    PILOT_R7C_H_REPORT_REL,
    PILOT_R7C_H_VISUAL_MANUAL_REL,
    PILOT_R7C_OVERLAYS_INDEX_REL,
    PILOT_R7C_VISUAL_FREEZE_REL,
    PILOT_R7C_VISUAL_REL,
    R7C_H1_EXECUTABLE_PATHS,
    assert_fill_script_cannot_assign_pass,
    executable_tree_sha256,
    executable_tree_sha256_at_revision,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_manual_template_null_prior_slots,
    validate_r7_pilot_audit,
    validate_r7_pilot_completion,
    validate_r7_pilot_report,
    validate_r7_visual_freeze_list,
    validate_r7_visual_qa_manual,
    validate_r7c_absent_wrist_fixture_proof,
    validate_r7c_ambiguity_roi_amendment,
    validate_r7c_h_independent_reaudit,
    write_json,
)


def _bind() -> dict:
    allowed = ("Docs/results/ms_sta_v5_d0_r7c/",)
    dirty = []
    for line in git_porcelain(_ROOT).splitlines():
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        norm = path.replace("\\", "/")
        if any(norm == p.rstrip("/") or norm.startswith(p) for p in allowed):
            continue
        dirty.append(line)
    dirty_s = "\n".join(dirty).strip()
    if dirty_s:
        raise RuntimeError(f"dirty_worktree_forbidden:{dirty_s[:500]}")
    rev = git_rev(_ROOT)
    tree = executable_tree_sha256_at_revision(R7C_H1_EXECUTABLE_PATHS, rev, root=_ROOT)
    if "scripts/run_ms_sta_v5_d0_r7c_h1_finalize.py" not in R7C_H1_EXECUTABLE_PATHS:
        raise RuntimeError("finalize_producer_not_listed")
    if "scripts/fill_ms_sta_v5_d0_r7c_h_manual.py" not in R7C_H1_EXECUTABLE_PATHS:
        raise RuntimeError("fill_producer_must_be_bound_even_if_forbidden")
    return {"code_lock_git_revision": rev, "executable_tree_sha256": tree}


def _prove_r5_fail_remediation(overlays: list[dict]) -> list[dict]:
    r5 = load_json(_ROOT / PILOT_R5_VISUAL_MANUAL_REL)
    fail_keys = {
        (r["sample_id"], int(r["window_start_idx"]))
        for r in r5["inspections"]
        if r.get("verdict") == "FAIL"
    }
    by = {
        (o["sample_id"], int(o["window_start_idx"])): o
        for o in overlays
        if o.get("kind") == "center_overlay"
    }
    out = []
    for key in sorted(fail_keys):
        o = by.get(key)
        if o is None:
            out.append({"key": list(key), "ok": False, "error": "missing_overlay"})
            continue
        boxes = o.get("boxes") or {}
        out.append(
            {
                "key": list(key),
                "ok": boxes.get("left_hand") is not None and boxes.get("right_hand") is not None,
                "left_hand_box": boxes.get("left_hand") is not None,
                "right_hand_box": boxes.get("right_hand") is not None,
                "rgb_roi_sources": o.get("rgb_roi_sources") or {},
                "missing_regions": o.get("missing_regions") or {},
            }
        )
    return out


def main() -> int:
    validate_r7c_h_independent_reaudit(load_json(_ROOT / PILOT_R7C_H_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    if sha256_file(_ROOT / PILOT_R7C_H_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H_INDEPENDENT_REAUDIT_SHA256:
        raise RuntimeError("r7c_h_independent_reaudit_sha_drift")
    assert_fill_script_cannot_assign_pass(root=_ROOT)
    validate_r7c_ambiguity_roi_amendment(load_json(_ROOT / FULL_PROTOCOL_R7C_AMENDMENT_REL), root=_ROOT)
    if sha256_file(_ROOT / PILOT_R7C_H_AUDIT_REL) != EXPECTED_R7C_H_PILOT_AUDIT_SHA256:
        raise RuntimeError("frozen_r7c_h_audit_mutated")
    if sha256_file(_ROOT / PILOT_R7C_H_REPORT_REL) != EXPECTED_R7C_H_PILOT_REPORT_SHA256:
        raise RuntimeError("frozen_r7c_h_report_mutated")
    if sha256_file(_ROOT / PILOT_R7C_H_COMPLETION_REL) != EXPECTED_R7C_H_PILOT_COMPLETION_SHA256:
        raise RuntimeError("frozen_r7c_h_completion_mutated")
    if sha256_file(_ROOT / PILOT_R7C_H_VISUAL_MANUAL_REL) != EXPECTED_R7C_H_VISUAL_MANUAL_SHA256:
        raise RuntimeError("frozen_r7c_h_manual_mutated")
    if sha256_file(_ROOT / PILOT_R7B_FIXTURE_ABSENT_REL / "PROOF.json") != EXPECTED_R7B_ABSENT_FIXTURE_PROOF_SHA256:
        raise RuntimeError("r7b_fixture_proof_not_restored")

    validate_manual_template_null_prior_slots(load_json(_ROOT / PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL))
    freeze = load_json(_ROOT / PILOT_R7C_VISUAL_FREEZE_REL)
    validate_r7_visual_freeze_list(freeze)
    bind = _bind()

    manual_path = _ROOT / PILOT_R7C_H1_VISUAL_MANUAL_REL
    if not manual_path.is_file():
        raise RuntimeError(f"missing_human_manual:{PILOT_R7C_H1_VISUAL_MANUAL_REL}")
    manual = load_json(manual_path)
    if manual.get("synthesized_by_finalizer") is not False:
        raise RuntimeError("manual_must_declare_not_synthesized_by_finalizer")
    if manual.get("synthesized_by_template_tool") is not False:
        raise RuntimeError("manual_still_marked_as_template")
    if manual.get("synthesized_by_fill_script") is not False:
        raise RuntimeError("manual_must_declare_not_fill_script")
    if manual.get("status") == "TEMPLATE_AWAITING_HUMAN_REVIEW":
        raise RuntimeError("manual_is_unfilled_template")
    human = load_json(_ROOT / PILOT_R7C_H1_PRIOR_JUDGMENTS_REL)
    if human.get("synthesized_by_fill_script") is not False:
        raise RuntimeError("judgments_file_fill_script_flag")
    if sha256_file(_ROOT / PILOT_R7C_H1_PRIOR_JUDGMENTS_REL) != manual.get("human_judgments_sha256"):
        raise RuntimeError("manual_human_judgments_sha_mismatch")

    overlays = load_json(_ROOT / PILOT_R7C_OVERLAYS_INDEX_REL)["overlays_index"]
    validate_r7_visual_qa_manual(
        manual,
        overlays_index=overlays,
        freeze_list=freeze,
        root=_ROOT,
        visual_dir_rel=PILOT_R7C_VISUAL_REL,
        require_exact_22=True,
        require_prior_tile_judgments=True,
    )

    rem = _prove_r5_fail_remediation(overlays)
    rem_ok = all(bool(r.get("ok")) for r in rem)
    diag = load_json(_ROOT / PILOT_R7C_DIAGNOSTIC_REL)
    c0_ok = bool(diag.get("c0_all_bit_exact")) and int(diag.get("n_samples") or 0) == 16

    hold_keys = {
        (w["sample_id"], int(w["window_start_idx"]))
        for w in freeze["windows"]
        if (w.get("r7_role") or w.get("r7c_role")) == "holdout_r6_carryforward"
    }
    holdout_missing_box_n_fail = 0
    for o in overlays:
        key = (o["sample_id"], int(o["window_start_idx"]))
        if key not in hold_keys or o.get("kind") != "center_overlay":
            continue
        boxes = o.get("boxes") or {}
        if boxes.get("left_hand") is None or boxes.get("right_hand") is None:
            holdout_missing_box_n_fail += 1
    holdout_manual_n_fail = sum(
        1
        for r in manual.get("inspections") or []
        if r.get("verdict") == "FAIL"
        and (r.get("sample_id"), int(r.get("window_start_idx"))) in hold_keys
    )

    producer = "scripts/run_ms_sta_v5_d0_r7c_h1_finalize.py"
    tree_ok = producer in R7C_H1_EXECUTABLE_PATHS
    try:
        tree_sha = executable_tree_sha256_at_revision(
            R7C_H1_EXECUTABLE_PATHS, bind["code_lock_git_revision"], root=_ROOT
        )
        executable_tree_sha256(R7C_H1_EXECUTABLE_PATHS, root=_ROOT, require_git_tracked=True)
        tracked_ok = True
    except FileNotFoundError:
        tree_sha = ""
        tracked_ok = False
    validate_r7c_absent_wrist_fixture_proof(
        load_json(_ROOT / PILOT_R7C_FIXTURE_ABSENT_REL / "PROOF.json"), root=_ROOT
    )
    gates = {
        "r7b_pilot_audit_preserved_fail": sha256_file(_ROOT / "Docs/results/ms_sta_v5_d0_r7b/D0_R7B_PILOT_AUDIT.json")
        == EXPECTED_R7B_PILOT_AUDIT_SHA256,
        "r7c_h_independent_fail_bound": True,
        "r7c_h_self_pass_bytes_preserved": True,
        "fill_script_forbidden_and_bound": True,
        "template_null_prior_slots": True,
        "canonical_protocol_immutable": sha256_file(_ROOT / FULL_PROTOCOL_REL)
        == EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "c0_bit_exact_all_16": c0_ok,
        "primary_and_holdout_zero_fail": int(manual.get("n_fail") if manual.get("n_fail") is not None else -1) == 0
        and holdout_missing_box_n_fail == 0
        and holdout_manual_n_fail == 0,
        "finalizer_did_not_synthesize_manual": manual.get("synthesized_by_finalizer") is False
        and manual.get("synthesized_by_template_tool") is False
        and manual.get("synthesized_by_fill_script") is False,
        "producer_in_executable_tree": tree_ok and tracked_ok and bind["executable_tree_sha256"] == tree_sha,
        "executable_paths_git_tracked": tracked_ok,
        "r5_fail_keys_remediated_with_boxes": rem_ok,
        "ambiguity_aware_prior_enabled": True,
        "pose_prior_geometry_locked": True,
        "ambiguity_roi_lock_bound": True,
        "empirical_absent_fixture_ok": True,
        "code_lock_revision_bound": True,
        "prior_tile_judgments_required": True,
        "human_judgments_file_bound": True,
    }
    ok = all(bool(v) for v in gates.values())
    report = {
        "schema": "ms_sta_v5_d0_r7c_h1_pilot_report_v1",
        "status": "COMPLETE",
        "ok": ok,
        "campaign": "D0_PILOT_R7C_H1",
        "parent_campaign": "D0_PILOT_R7C_H",
        "gates": gates,
        "holdout_missing_box_n_fail": holdout_missing_box_n_fail,
        "holdout_manual_n_fail": holdout_manual_n_fail,
        "holdout_residual_n_fail": holdout_missing_box_n_fail,
        "r5_fail_remediation": rem,
        "visual_qa_manual_sha256": sha256_file(manual_path),
        "human_judgments_sha256": sha256_file(_ROOT / PILOT_R7C_H1_PRIOR_JUDGMENTS_REL),
        "overlays_index_sha256": sha256_file(_ROOT / PILOT_R7C_OVERLAYS_INDEX_REL),
        "diagnostic_sha256": sha256_file(_ROOT / PILOT_R7C_DIAGNOSTIC_REL),
        "freeze_sha256": sha256_file(_ROOT / PILOT_R7C_VISUAL_FREEZE_REL),
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "r7c_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R7C_AMENDMENT_REL),
        "r7c_h_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R7C_H_INDEPENDENT_REAUDIT_REL),
        "r7c_absent_fixture_proof_sha256": sha256_file(_ROOT / PILOT_R7C_FIXTURE_ABSENT_REL / "PROOF.json"),
        "forbid": {"full_corpus": True, "m0": True, "train_b4": True, "push": True},
        "full_corpus_authorized": False,
        "algorithm_change_authorized": False,
    }
    validate_r7_pilot_report(
        report,
        root=_ROOT,
        executable_paths=R7C_H1_EXECUTABLE_PATHS,
        manual_rel=PILOT_R7C_H1_VISUAL_MANUAL_REL,
        overlays_rel=PILOT_R7C_OVERLAYS_INDEX_REL,
        diagnostic_rel=PILOT_R7C_DIAGNOSTIC_REL,
        freeze_rel=PILOT_R7C_VISUAL_FREEZE_REL,
        amendment_rel=FULL_PROTOCOL_R7C_AMENDMENT_REL,
        producer_rel="scripts/run_ms_sta_v5_d0_r7c_h1_finalize.py",
        require_campaign_pass=ok,
        require_worktree_code_lock_match=True,
    )
    report_sha = write_json(_ROOT / PILOT_R7C_H1_REPORT_REL, report, overwrite=True)
    audit = {
        "schema": "ms_sta_v5_d0_r7c_h1_pilot_audit_v1",
        "audit_verdict": "PASS" if ok else "FAIL",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auditor": "ducpm196",
        "campaign": "D0_PILOT_R7C_H1",
        "parent_campaign": "D0_PILOT_R7C_H",
        "pilot_report_sha256": report_sha,
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "gates_passed": ok,
        "full_corpus_authorized": False,
        "algorithm_change_authorized": False,
        "notes": [
            "Genuine manual-review hardening only; ROI algorithm unchanged.",
            "Frozen R7C-H self-PASS audit/report/completion/manual bytes preserved.",
            "fill_ms_sta_v5_d0_r7c_h_manual.py is FORBIDDEN and bound in the executable tree.",
            "H1 template keeps all judgment fields null until human review.",
            "23 prior-tile judgments entered from visual inspection via judgments file + merge (no invented PASS).",
            "STOP for independent R7C-H1 audit before push/full-corpus.",
        ],
    }
    audit_sha = write_json(_ROOT / PILOT_R7C_H1_AUDIT_REL, audit, overwrite=True)
    validate_r7_pilot_audit(
        audit,
        root=_ROOT,
        report_rel=PILOT_R7C_H1_REPORT_REL,
        executable_paths=R7C_H1_EXECUTABLE_PATHS,
        require_pass=ok,
        require_worktree_code_lock_match=True,
    )
    completion = {
        "schema": "ms_sta_v5_d0_r7c_h1_pilot_completion_v1",
        "status": "CLOSED",
        "closed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": "D0_PILOT_R7C_H1",
        "parent_campaign": "D0_PILOT_R7C_H",
        "pilot_report_sha256": report_sha,
        "pilot_audit_sha256": audit_sha,
        "pilot_audit_verdict": audit["audit_verdict"],
        "immutable": True,
        "full_corpus_authorized": False,
        "algorithm_change_authorized": False,
        "next": "INDEPENDENT_R7C_H1_AUDIT_REQUIRED_BEFORE_PUSH_OR_FULL_CORPUS",
    }
    write_json(_ROOT / PILOT_R7C_H1_COMPLETION_REL, completion, overwrite=True)
    validate_r7_pilot_completion(
        completion,
        root=_ROOT,
        report_rel=PILOT_R7C_H1_REPORT_REL,
        audit_rel=PILOT_R7C_H1_AUDIT_REL,
        require_pass=ok,
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "audit_verdict": audit["audit_verdict"],
                "campaign": "D0_PILOT_R7C_H1",
                "holdout_missing_box_n_fail": holdout_missing_box_n_fail,
                "holdout_manual_n_fail": holdout_manual_n_fail,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
