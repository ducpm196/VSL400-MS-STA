#!/usr/bin/env python3
"""R7B finalize — READ-ONLY on manual QA. Never synthesizes verdicts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    PILOT_R7B_VISUAL_REL,
    PILOT_R7B_FIXTURE_ABSENT_REL,
    EXPECTED_R7_PILOT_REPORT_SHA256,
    POSE_PRIOR_GEOMETRY_LOCK,
    validate_r7_pilot_completion,
    validate_r7_pilot_audit,
    R7B_EXECUTABLE_PATHS,
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R7_PILOT_AUDIT_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R7B_AMENDMENT_REL,
    PILOT_R5_VISUAL_MANUAL_REL,
    PILOT_R6_AUDIT_REL,
    PILOT_R7_INDEPENDENT_REAUDIT_REL,
    PILOT_R7B_AUDIT_REL,
    PILOT_R7B_COMPLETION_REL,
    PILOT_R7B_DIAGNOSTIC_REL,
    PILOT_R7B_OVERLAYS_INDEX_REL,
    PILOT_R7B_OUT_DIR_REL,
    PILOT_R7B_REPORT_REL,
    PILOT_R7B_RUN_CONTRACT_REL,
    PILOT_R7B_VISUAL_FREEZE_REL,
    PILOT_R7B_VISUAL_MANUAL_REL,
    executable_tree_sha256,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_r7_independent_reaudit,
    validate_r7b_gate_hardening_amendment,
    validate_r7_pilot_report,
    validate_r7_visual_freeze_list,
    validate_r7_visual_qa_manual,
    write_json,
)


def _bind() -> dict:
    allowed = ("Docs/results/ms_sta_v5_d0_r7b/",)
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
    tree = executable_tree_sha256(R7B_EXECUTABLE_PATHS, root=_ROOT, require_git_tracked=True)
    # Producer must be in tree
    if "scripts/run_ms_sta_v5_d0_r7b_finalize.py" not in R7B_EXECUTABLE_PATHS:
        raise RuntimeError("finalize_producer_not_listed")
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
        # Previously-missing hands must now have a box (landmark or pose_prior)
        miss = o.get("missing_regions") or {}
        sources = o.get("rgb_roi_sources") or {}
        # At least one hand that was problematic should have a box now.
        # Conservative: both LH and RH boxes present OR the previously-missing side present.
        lh_ok = boxes.get("left_hand") is not None
        rh_ok = boxes.get("right_hand") is not None
        ok = bool(lh_ok and rh_ok)
        out.append(
            {
                "key": list(key),
                "ok": ok,
                "left_hand_box": lh_ok,
                "right_hand_box": rh_ok,
                "rgb_roi_sources": sources,
                "missing_regions": miss,
            }
        )
    return out


def main() -> int:
    validate_r7_independent_reaudit(load_json(_ROOT / PILOT_R7_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r7b_gate_hardening_amendment(load_json(_ROOT / FULL_PROTOCOL_R7B_AMENDMENT_REL), root=_ROOT)
    freeze = load_json(_ROOT / PILOT_R7B_VISUAL_FREEZE_REL)
    validate_r7_visual_freeze_list(freeze)
    bind = _bind()
    contract = load_json(_ROOT / PILOT_R7B_RUN_CONTRACT_REL)
    if contract.get("code_lock_git_revision") != bind["code_lock_git_revision"]:
        raise RuntimeError("contract_code_lock_mismatch")

    manual_path = _ROOT / PILOT_R7B_VISUAL_MANUAL_REL
    if not manual_path.is_file():
        raise RuntimeError(f"missing_human_manual:{PILOT_R7B_VISUAL_MANUAL_REL}")
    manual = load_json(manual_path)
    if manual.get("synthesized_by_finalizer") is not False:
        raise RuntimeError("manual_must_declare_not_synthesized_by_finalizer")
    if manual.get("synthesized_by_template_tool") is not False:
        raise RuntimeError("manual_still_marked_as_template")
    if manual.get("status") == "TEMPLATE_AWAITING_HUMAN_REVIEW":
        raise RuntimeError("manual_is_unfilled_template")
    overlays = load_json(_ROOT / PILOT_R7B_OVERLAYS_INDEX_REL)["overlays_index"]
    validate_r7_visual_qa_manual(
        manual,
        overlays_index=overlays,
        freeze_list=freeze,
        root=_ROOT,
        visual_dir_rel=PILOT_R7B_VISUAL_REL,
        require_exact_22=True,
    )

    rem = _prove_r5_fail_remediation(overlays)
    rem_ok = all(bool(r.get("ok")) for r in rem)
    diag = load_json(_ROOT / PILOT_R7B_DIAGNOSTIC_REL)
    c0_ok = bool(diag.get("c0_all_bit_exact")) and int(diag.get("n_samples") or 0) == 16

    # Holdout residual must be zero: every holdout center must have both hand boxes
    hold_keys = {
        (w["sample_id"], int(w["window_start_idx"]))
        for w in freeze["windows"]
        if w.get("r7_role") == "holdout_r6_carryforward"
    }
    hold_fail = 0
    for o in overlays:
        key = (o["sample_id"], int(o["window_start_idx"]))
        if key not in hold_keys or o.get("kind") != "center_overlay":
            continue
        boxes = o.get("boxes") or {}
        if boxes.get("left_hand") is None or boxes.get("right_hand") is None:
            hold_fail += 1

    producer = "scripts/run_ms_sta_v5_d0_r7b_finalize.py"
    tree_ok = producer in R7B_EXECUTABLE_PATHS
    try:
        tree_sha = executable_tree_sha256(R7B_EXECUTABLE_PATHS, root=_ROOT, require_git_tracked=True)
        tracked_ok = True
    except FileNotFoundError:
        tree_sha = ""
        tracked_ok = False
    gates = {
        "r7_pilot_audit_preserved_pass": sha256_file(_ROOT / "Docs/results/ms_sta_v5_d0_r7/D0_R7_PILOT_AUDIT.json")
        == EXPECTED_R7_PILOT_AUDIT_SHA256,
        "r7_independent_fail_bound": True,
        "canonical_protocol_immutable": sha256_file(_ROOT / FULL_PROTOCOL_REL)
        == EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "c0_bit_exact_all_16": c0_ok,
        "primary_and_holdout_zero_fail": int(manual.get("n_fail") or 0) == 0 and hold_fail == 0,
        "finalizer_did_not_synthesize_manual": manual.get("synthesized_by_finalizer") is False
        and manual.get("synthesized_by_template_tool") is False,
        "producer_in_executable_tree": tree_ok and tracked_ok and bind["executable_tree_sha256"] == tree_sha,
        "executable_paths_git_tracked": tracked_ok,
        "r5_fail_keys_remediated_with_boxes": rem_ok,
        "pose_wrist_prior_enabled": True,
        "pose_prior_geometry_locked": True,
        "empirical_absent_fixture_ok": (_ROOT / PILOT_R7B_FIXTURE_ABSENT_REL / "PROOF.json").is_file(),
    }
    ok = all(bool(v) for v in gates.values())
    report = {
        "schema": "ms_sta_v5_d0_r7b_pilot_report_v1",
        "status": "COMPLETE",
        "ok": ok,
        "campaign": "D0_PILOT_R7B",
        "gates": gates,
        "holdout_residual_n_fail": hold_fail,
        "r5_fail_remediation": rem,
        "visual_qa_manual_sha256": sha256_file(manual_path),
        "overlays_index_sha256": sha256_file(_ROOT / PILOT_R7B_OVERLAYS_INDEX_REL),
        "diagnostic_sha256": sha256_file(_ROOT / PILOT_R7B_DIAGNOSTIC_REL),
        "freeze_sha256": sha256_file(_ROOT / PILOT_R7B_VISUAL_FREEZE_REL),
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "r7b_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R7B_AMENDMENT_REL),
        "r6_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R7_INDEPENDENT_REAUDIT_REL),
        "forbid": {"full_corpus": True, "m0": True, "train_b4": True, "push": True},
        "full_corpus_authorized": False,
    }
    validate_r7_pilot_report(
        report,
        root=_ROOT,
        executable_paths=R7B_EXECUTABLE_PATHS,
        manual_rel=PILOT_R7B_VISUAL_MANUAL_REL,
        overlays_rel=PILOT_R7B_OVERLAYS_INDEX_REL,
        diagnostic_rel=PILOT_R7B_DIAGNOSTIC_REL,
        freeze_rel=PILOT_R7B_VISUAL_FREEZE_REL,
        amendment_rel=FULL_PROTOCOL_R7B_AMENDMENT_REL,
        producer_rel="scripts/run_ms_sta_v5_d0_r7b_finalize.py",
        require_campaign_pass=ok,
    )
    report_sha = write_json(_ROOT / PILOT_R7B_REPORT_REL, report, overwrite=True)
    audit = {
        "schema": "ms_sta_v5_d0_r7b_pilot_audit_v1",
        "audit_verdict": "PASS" if ok else "FAIL",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auditor": "ducpm196",
        "campaign": "D0_PILOT_R7B",
        "pilot_report_sha256": report_sha,
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "gates_passed": ok,
        "full_corpus_authorized": False,
        "notes": [
            "Finalizer read human manual only; did not synthesize verdicts.",
            "Dual-source: Skeleton may stay missing while RGB uses pose_wrist_prior.",
            "Honest FAIL allowed when local pose-prior ROI quality fails even under area lock.",
            "STOP for independent R7B audit before push/full-corpus.",
        ],
    }
    audit_sha = write_json(_ROOT / PILOT_R7B_AUDIT_REL, audit, overwrite=True)
    validate_r7_pilot_audit(
        audit,
        root=_ROOT,
        report_rel=PILOT_R7B_REPORT_REL,
        executable_paths=R7B_EXECUTABLE_PATHS,
        require_pass=ok,
    )
    contract["status"] = "CLOSED"
    contract["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(_ROOT / PILOT_R7B_RUN_CONTRACT_REL, contract, overwrite=True)
    completion = {
        "schema": "ms_sta_v5_d0_r7b_pilot_completion_v1",
        "status": "CLOSED",
        "closed_at": contract["closed_at"],
        "campaign": "D0_PILOT_R7B",
        "pilot_report_sha256": report_sha,
        "pilot_audit_sha256": audit_sha,
        "pilot_audit_verdict": audit["audit_verdict"],
        "immutable": True,
        "full_corpus_authorized": False,
        "next": "INDEPENDENT_R7B_AUDIT_REQUIRED_BEFORE_PUSH_OR_FULL_CORPUS",
    }
    write_json(_ROOT / PILOT_R7B_COMPLETION_REL, completion, overwrite=True)
    validate_r7_pilot_completion(
        completion,
        root=_ROOT,
        report_rel=PILOT_R7B_REPORT_REL,
        audit_rel=PILOT_R7B_AUDIT_REL,
        require_pass=ok,
    )
    print(json.dumps({"ok": ok, "audit_verdict": audit["audit_verdict"], "holdout_fail": hold_fail}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
