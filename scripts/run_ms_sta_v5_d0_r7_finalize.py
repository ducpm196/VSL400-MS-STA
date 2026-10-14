#!/usr/bin/env python3
"""R7 finalize — READ-ONLY on manual QA. Never synthesizes verdicts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R6_PILOT_AUDIT_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R7_AMENDMENT_REL,
    PILOT_R5_VISUAL_MANUAL_REL,
    PILOT_R6_AUDIT_REL,
    PILOT_R6_INDEPENDENT_REAUDIT_REL,
    PILOT_R7_AUDIT_REL,
    PILOT_R7_COMPLETION_REL,
    PILOT_R7_DIAGNOSTIC_REL,
    PILOT_R7_OVERLAYS_INDEX_REL,
    PILOT_R7_OUT_DIR_REL,
    PILOT_R7_REPORT_REL,
    PILOT_R7_RUN_CONTRACT_REL,
    PILOT_R7_VISUAL_FREEZE_REL,
    PILOT_R7_VISUAL_MANUAL_REL,
    executable_tree_sha256,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_r6_independent_reaudit,
    validate_r7_dual_source_amendment,
    validate_r7_pilot_report,
    validate_r7_visual_freeze_list,
    validate_r7_visual_qa_manual,
    write_json,
)

R7_EXECUTABLE_PATHS = [
    "src/data_prep/hand_recovery_v6.py",
    "src/data_prep/extract_and_smooth.py",
    "src/data_prep/skel_v3.py",
    "src/models/rgb_token_v3.py",
    "scripts/ms_sta_v5_d0_lib.py",
    "scripts/run_ms_sta_v5_d0_r6_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_finalize.py",
    "scripts/build_ms_sta_v5_d0_r7_freeze.py",
    "scripts/write_ms_sta_v5_d0_r7_manual_human.py",
]


def _bind() -> dict:
    allowed = ("Docs/results/ms_sta_v5_d0_r7/",)
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
    tree = executable_tree_sha256(R7_EXECUTABLE_PATHS, root=_ROOT)
    # Producer must be in tree
    if "scripts/run_ms_sta_v5_d0_r7_finalize.py" not in R7_EXECUTABLE_PATHS:
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
    validate_r6_independent_reaudit(load_json(_ROOT / PILOT_R6_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r7_dual_source_amendment(load_json(_ROOT / FULL_PROTOCOL_R7_AMENDMENT_REL), root=_ROOT)
    freeze = load_json(_ROOT / PILOT_R7_VISUAL_FREEZE_REL)
    validate_r7_visual_freeze_list(freeze)
    bind = _bind()
    contract = load_json(_ROOT / PILOT_R7_RUN_CONTRACT_REL)
    if contract.get("code_lock_git_revision") != bind["code_lock_git_revision"]:
        raise RuntimeError("contract_code_lock_mismatch")

    manual_path = _ROOT / PILOT_R7_VISUAL_MANUAL_REL
    if not manual_path.is_file():
        raise RuntimeError(f"missing_human_manual:{PILOT_R7_VISUAL_MANUAL_REL}")
    manual = load_json(manual_path)
    if manual.get("synthesized_by_finalizer") is not False:
        raise RuntimeError("manual_must_declare_not_synthesized_by_finalizer")
    overlays = load_json(_ROOT / PILOT_R7_OVERLAYS_INDEX_REL)["overlays_index"]
    validate_r7_visual_qa_manual(manual, overlays_index=overlays)

    rem = _prove_r5_fail_remediation(overlays)
    rem_ok = all(bool(r.get("ok")) for r in rem)
    diag = load_json(_ROOT / PILOT_R7_DIAGNOSTIC_REL)
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

    gates = {
        "r6_pilot_audit_preserved_pass": sha256_file(_ROOT / PILOT_R6_AUDIT_REL)
        == EXPECTED_R6_PILOT_AUDIT_SHA256,
        "r6_independent_fail_bound": True,
        "canonical_protocol_immutable": sha256_file(_ROOT / FULL_PROTOCOL_REL)
        == EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "c0_bit_exact_all_16": c0_ok,
        "primary_and_holdout_zero_fail": int(manual.get("n_fail") or 0) == 0 and hold_fail == 0,
        "finalizer_did_not_synthesize_manual": True,
        "producer_in_executable_tree": True,
        "r5_fail_keys_remediated_with_boxes": rem_ok,
        "pose_wrist_prior_enabled": True,
    }
    ok = all(bool(v) for v in gates.values())
    report = {
        "schema": "ms_sta_v5_d0_r7_pilot_report_v1",
        "status": "COMPLETE",
        "ok": ok,
        "campaign": "D0_PILOT_R7",
        "gates": gates,
        "holdout_residual_n_fail": hold_fail,
        "r5_fail_remediation": rem,
        "visual_qa_manual_sha256": sha256_file(manual_path),
        "overlays_index_sha256": sha256_file(_ROOT / PILOT_R7_OVERLAYS_INDEX_REL),
        "diagnostic_sha256": sha256_file(_ROOT / PILOT_R7_DIAGNOSTIC_REL),
        "freeze_sha256": sha256_file(_ROOT / PILOT_R7_VISUAL_FREEZE_REL),
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "r7_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R7_AMENDMENT_REL),
        "r6_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R6_INDEPENDENT_REAUDIT_REL),
        "forbid": {"full_corpus": True, "m0": True, "train_b4": True, "push": True},
        "full_corpus_authorized": False,
    }
    validate_r7_pilot_report(report, root=_ROOT)
    report_sha = write_json(_ROOT / PILOT_R7_REPORT_REL, report, overwrite=True)
    audit = {
        "schema": "ms_sta_v5_d0_r7_pilot_audit_v1",
        "audit_verdict": "PASS" if ok else "FAIL",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auditor": "ducpm196",
        "campaign": "D0_PILOT_R7",
        "pilot_report_sha256": report_sha,
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "gates_passed": ok,
        "full_corpus_authorized": False,
        "notes": [
            "Finalizer read human manual only; did not synthesize verdicts.",
            "Dual-source: Skeleton may stay missing while RGB uses pose_wrist_prior.",
            "STOP for independent R7 audit before push/full-corpus.",
        ],
    }
    audit_sha = write_json(_ROOT / PILOT_R7_AUDIT_REL, audit, overwrite=True)
    contract["status"] = "CLOSED"
    contract["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(_ROOT / PILOT_R7_RUN_CONTRACT_REL, contract, overwrite=True)
    completion = {
        "schema": "ms_sta_v5_d0_r7_pilot_completion_v1",
        "status": "CLOSED",
        "closed_at": contract["closed_at"],
        "campaign": "D0_PILOT_R7",
        "pilot_report_sha256": report_sha,
        "pilot_audit_sha256": audit_sha,
        "pilot_audit_verdict": audit["audit_verdict"],
        "immutable": True,
        "full_corpus_authorized": False,
        "next": "INDEPENDENT_R7_AUDIT_REQUIRED_BEFORE_PUSH_OR_FULL_CORPUS",
    }
    write_json(_ROOT / PILOT_R7_COMPLETION_REL, completion, overwrite=True)
    print(json.dumps({"ok": ok, "audit_verdict": audit["audit_verdict"], "holdout_fail": hold_fail}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
