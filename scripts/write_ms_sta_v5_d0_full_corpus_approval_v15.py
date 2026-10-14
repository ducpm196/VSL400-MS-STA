#!/usr/bin/env python3
"""Write DRAFT or SEALED D0 full-corpus v15 approval/protocol/run-contract.

Default / --draft / --rewrite-draft: draft scaffolding (dirty worktree OK).
--seal: clean worktree required; SEALED APPROVED artifacts; does NOT write
external reaudit PASS or post-audit auth; does not launch RGB/M0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    CORPUS_N,
    D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
    D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL,
    D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER_REL,
    D0_FULL_CORPUS_V15_INDEPENDENT_AUDIT_FAIL_ORCHESTRATION_REL,
    D0_FULL_CORPUS_V15_WATCHDOG_COLD_START_FAIL_REL,
    D0_FULL_CORPUS_V15_INTERRUPTED_RGB_REL,
    D0_FULL_CORPUS_V15_INTERRUPTED_BOUND_WORKER_RECHECKED_COLD_START_GATE_REL,
    D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R2_AUDIT_FAIL_REMEDIATION_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R3_AUDIT_FAIL_REMEDIATION_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R4_EXECUTION_PROVENANCE_REMEDIATION_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R5_META_CONTRACT_REMEDIATION_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R6_FAIL_CLOSED_REMEDIATION_REL,
    D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL,
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL_REL,
    D0_FULL_CORPUS_V15_SUPERSESSION_REL,
    DEFAULT_QUALITY_THRESHOLDS,
    EVIDENCE_ROOT_V15_REL,
    EXTRACT_PY_REL,
    F0_PY_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    MAX_RESUMES,
    OUTPUT_VERSION_V15,
    PHASES_EXACT,
    PSUTIL_PIN,
    SPLIT_COUNTS,
    V14_SEALED_SKELETON_BINARY_ROOT_REL,
    assert_binaries_gitignored,
    assert_binary_root_absent_or_empty,
    assert_no_forbidden_symbols,
    assert_psutil_available,
    atomic_create_json,
    binary_root_rel,
    default_quality_thresholds,
    env_provenance,
    load_json,
    validate_approval_v15,
    validate_execution_protocol_v15,
    validate_rgb_f0_smoke_report_v2,
    validate_run_contract_v15,
)
from ms_sta_v5_d0_lib import (  # noqa: E402
    ELIGIBLE_REL,
    EXPECTED_ELIGIBLE_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    EXPECTED_SPLIT_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R7C_AMENDMENT_REL,
    METADATA_REL,
    PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL,
    SPLIT_REL,
    executable_tree_sha256_at_revision,
    git_porcelain,
    git_rev,
    sha256_file,
)


def _unlink_draft_only(path: Path) -> None:
    """Allow rewrite of DRAFT remediation artifacts only (never sealed)."""
    if not path.is_file():
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    status = str(obj.get("status") or obj.get("audit_verdict") or "")
    if "DRAFT" not in status.upper():
        raise SystemExit(f"refuse_rewrite_non_draft:{path}")
    path.unlink()


def _prepare_reseal(
    *,
    reason: str,
    companion_rel: str,
    companion_field: str,
) -> dict[str, str]:
    """Supersede prior sealed companions; keep bytes via rename. Create-only new SUPERSESSION."""
    companion_path = _ROOT / companion_rel
    if not companion_path.is_file():
        raise SystemExit(f"reseal_companion_missing:{companion_rel}")
    companion_sha = sha256_file(companion_path)

    supersession_path = _ROOT / D0_FULL_CORPUS_V15_SUPERSESSION_REL
    approval_path = _ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL
    if approval_path.is_file():
        prior_approval = load_json(approval_path)
        shorthead = str(prior_approval.get("code_lock_git_revision") or git_rev(_ROOT))[:12]
    else:
        shorthead = git_rev(_ROOT)[:12]

    relocated_prior_supersession = None
    if supersession_path.is_file():
        dest = supersession_path.parent / f"D0_FULL_CORPUS_V15_SEAL_SUPERSESSION_SUPERSEDED_{shorthead}.json"
        if dest.exists():
            raise SystemExit(f"superseded_supersession_exists:{dest}")
        supersession_path.rename(dest)
        relocated_prior_supersession = str(dest.relative_to(_ROOT)).replace("\\", "/")

    prior_paths = {
        "prior_approval_v15_sha256": _ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL,
        "prior_execution_protocol_v15_sha256": _ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        "prior_run_contract_v15_sha256": _ROOT / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
        "prior_approval_audit_v15_sha256": _ROOT / D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
        "prior_external_reaudit_v15_sha256": _ROOT / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
        "prior_post_audit_execution_auth_v15_sha256": _ROOT
        / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
        "prior_rgb_f0_smoke_report_sha256": _ROOT / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL,
    }
    prior_shas: dict[str, str | None] = {}
    for key, path in prior_paths.items():
        prior_shas[key] = sha256_file(path) if path.is_file() else None

    def _relocate_active(rel: str, stem: str) -> str | None:
        src = _ROOT / rel
        if not src.is_file():
            return None
        dest = src.parent / f"{stem}_SUPERSEDED_{shorthead}.json"
        if dest.exists():
            raise SystemExit(f"superseded_target_exists:{dest}")
        src.rename(dest)
        return str(dest.relative_to(_ROOT)).replace("\\", "/")

    relocated_external = _relocate_active(
        D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
        "D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT",
    )
    relocated_post_auth = _relocate_active(
        D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
        "D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION",
    )

    supersession = {
        "schema": "ms_sta_v5_d0_full_corpus_v15_seal_supersession_v1",
        "status": "PRIOR_SEAL_SUPERSEDED",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "reason": reason,
        "superseded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "superseded_shorthead": shorthead,
        companion_field: companion_sha,
        "prior_approval_v15_sha256": prior_shas["prior_approval_v15_sha256"],
        "prior_execution_protocol_v15_sha256": prior_shas["prior_execution_protocol_v15_sha256"],
        "prior_run_contract_v15_sha256": prior_shas["prior_run_contract_v15_sha256"],
        "prior_approval_audit_v15_sha256": prior_shas["prior_approval_audit_v15_sha256"],
        "prior_external_reaudit_v15_sha256": prior_shas["prior_external_reaudit_v15_sha256"],
        "prior_post_audit_execution_auth_v15_sha256": prior_shas[
            "prior_post_audit_execution_auth_v15_sha256"
        ],
        "prior_rgb_f0_smoke_report_sha256": prior_shas["prior_rgb_f0_smoke_report_sha256"],
        "relocated_external_reaudit_rel": relocated_external,
        "relocated_post_auth_rel": relocated_post_auth,
        "relocated_prior_supersession_rel": relocated_prior_supersession,
        "notes": [
            "Prior seal companions relocated or superseded (not deleted) before reseal.",
            "Does not authorize RGB launch; new external PASS + post-auth still required.",
        ],
    }
    supersession_sha = atomic_create_json(supersession_path, supersession)

    for rel in (
        D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        D0_FULL_CORPUS_APPROVAL_V15_REL,
        D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
        D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
    ):
        path = _ROOT / rel
        if path.is_file():
            path.unlink()

    return {
        companion_field: companion_sha,
        "supersession_sha256": supersession_sha,
    }


def _prepare_reseal_after_prelaunch_blocker() -> dict[str, str]:
    return _prepare_reseal(
        reason="monitor_v14_identity+utf8_bom",
        companion_rel=D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER_REL,
        companion_field="prelaunch_blocker_sha256",
    )


def _prepare_reseal_after_orchestration_fail() -> dict[str, str]:
    return _prepare_reseal(
        reason="carry_forward_orchestration_rgb_unreachable",
        companion_rel=D0_FULL_CORPUS_V15_INDEPENDENT_AUDIT_FAIL_ORCHESTRATION_REL,
        companion_field="orchestration_audit_fail_sha256",
    )


def _prepare_reseal_after_watchdog_cold_start_fail() -> dict[str, str]:
    return _prepare_reseal(
        reason="watchdog_cold_start_binary_root_bootstrap",
        companion_rel=D0_FULL_CORPUS_V15_WATCHDOG_COLD_START_FAIL_REL,
        companion_field="watchdog_cold_start_fail_sha256",
    )


def _prepare_reseal_after_interrupted_rgb() -> dict[str, str]:
    return _prepare_reseal(
        reason="interrupted_rgb_worker_exit_without_progress",
        companion_rel=D0_FULL_CORPUS_V15_INTERRUPTED_RGB_REL,
        companion_field="interrupted_rgb_sha256",
    )


def _prepare_reseal_after_interrupted_bound_worker_rechecked_cold_start_gate() -> dict[str, str]:
    return _prepare_reseal(
        reason="interrupted_bound_worker_rechecked_cold_start_gate",
        companion_rel=D0_FULL_CORPUS_V15_INTERRUPTED_BOUND_WORKER_RECHECKED_COLD_START_GATE_REL,
        companion_field="interrupted_bound_worker_rechecked_cold_start_gate_sha256",
    )


def _prepare_reseal_after_quality_scan_local_skel_manifest() -> dict[str, str]:
    return _prepare_reseal(
        reason="quality_scan_used_local_skel_manifest_under_carry_forward",
        companion_rel=D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL,
        companion_field="quality_scan_used_local_skel_manifest_under_carry_forward_sha256",
    )


def _prepare_reseal_after_scan_recovery_r1_perf() -> dict[str, str]:
    return _prepare_reseal(
        reason="scan_recovery_r1_perf_remediation",
        companion_rel=D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL,
        companion_field="scan_recovery_r1_perf_remediation_sha256",
    )


def _prepare_reseal_after_scan_recovery_r2_audit_fail() -> dict[str, str]:
    return _prepare_reseal(
        reason="scan_recovery_r2_audit_fail_remediation",
        companion_rel=D0_FULL_CORPUS_V15_SCAN_RECOVERY_R2_AUDIT_FAIL_REMEDIATION_REL,
        companion_field="scan_recovery_r2_audit_fail_remediation_sha256",
    )


def _prepare_reseal_after_scan_recovery_r3_audit_fail() -> dict[str, str]:
    return _prepare_reseal(
        reason="scan_recovery_r3_audit_fail_remediation",
        companion_rel=D0_FULL_CORPUS_V15_SCAN_RECOVERY_R3_AUDIT_FAIL_REMEDIATION_REL,
        companion_field="scan_recovery_r3_audit_fail_remediation_sha256",
    )


def _prepare_reseal_after_scan_recovery_r4_execution_provenance() -> dict[str, str]:
    return _prepare_reseal(
        reason="scan_recovery_r4_execution_provenance_remediation",
        companion_rel=D0_FULL_CORPUS_V15_SCAN_RECOVERY_R4_EXECUTION_PROVENANCE_REMEDIATION_REL,
        companion_field="scan_recovery_r4_execution_provenance_remediation_sha256",
    )


def _prepare_reseal_after_scan_recovery_r5_meta_contract() -> dict[str, str]:
    return _prepare_reseal(
        reason="scan_recovery_r5_meta_contract_remediation",
        companion_rel=D0_FULL_CORPUS_V15_SCAN_RECOVERY_R5_META_CONTRACT_REMEDIATION_REL,
        companion_field="scan_recovery_r5_meta_contract_remediation_sha256",
    )


def _prepare_reseal_after_scan_recovery_r6_fail_closed() -> dict[str, str]:
    return _prepare_reseal(
        reason="scan_recovery_r6_fail_closed_remediation",
        companion_rel=D0_FULL_CORPUS_V15_SCAN_RECOVERY_R6_FAIL_CLOSED_REMEDIATION_REL,
        companion_field="scan_recovery_r6_fail_closed_remediation_sha256",
    )


def _write_draft(*, rewrite: bool) -> int:
    if rewrite:
        for rel in (
            D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
            D0_FULL_CORPUS_APPROVAL_V15_REL,
            D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
            D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
        ):
            _unlink_draft_only(_ROOT / rel)

    rev = git_rev(_ROOT)
    bin_rel = binary_root_rel(rev, root=_ROOT, version=OUTPUT_VERSION_V15)
    env = env_provenance(root=_ROOT)
    thr = default_quality_thresholds()
    thr.update(
        {
            "require_quality_stats_artifact": True,
            "fail_closed_on_threshold_breach": True,
            "tensor_authoritative": True,
        }
    )

    evidence = _ROOT / EVIDENCE_ROOT_V15_REL
    evidence.mkdir(parents=True, exist_ok=True)

    parent_halt_sha = None
    halt_path = _ROOT / D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL
    if halt_path.is_file():
        parent_halt_sha = sha256_file(halt_path)

    protocol = {
        "schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v15",
        "status": "DRAFT_REMEDIATION_NOT_SEALED",
        "output_version": OUTPUT_VERSION_V15,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "skeleton_engine": "HandRecoveryEngine",
        "eligible_n": CORPUS_N,
        "split_counts": dict(SPLIT_COUNTS),
        "code_lock_git_revision": rev,
        "executable_tree_sha256": None,
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "extract_python_rel": EXTRACT_PY_REL,
        "f0_python_rel": F0_PY_REL,
        "env_provenance": env,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "psutil_pin": PSUTIL_PIN,
        "allow_m0": False,
        "allow_auto_chain": False,
        "max_resumes": MAX_RESUMES,
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "carry_forward_protocol_rel": D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL_REL,
        "notes": [
            "DRAFT only — not sealed; do not launch RGB full corpus.",
            "Skeleton carry-forward from sealed v14; RGB writes under new v15 binary root.",
            "Await independent carry-forward audit PASS + post-audit execution authorization.",
        ],
    }
    protocol_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL, protocol)

    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "status": "DRAFT_REMEDIATION_NOT_SEALED",
        "decision": "DRAFT_NOT_APPROVED",
        "scope": "d0_full_corpus_rgb_v15_draft_scaffolding",
        "operator": "ducpm196",
        "drafted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "one_shot": True,
        "allow_full_corpus_extraction": False,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "output_version": OUTPUT_VERSION_V15,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "psutil_pin": PSUTIL_PIN,
        "max_resumes": MAX_RESUMES,
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "eligible_n": CORPUS_N,
        "eligible_by_split": dict(SPLIT_COUNTS),
        "eligible_manifest": ELIGIBLE_REL,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata": METADATA_REL,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers": SPLIT_REL,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
        "full_protocol": FULL_PROTOCOL_REL,
        "full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "r7c_amendment": FULL_PROTOCOL_R7C_AMENDMENT_REL,
        "r7c_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R7C_AMENDMENT_REL),
        "r7c_h1_independent_reaudit": PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "execution_protocol_v15": D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        "execution_protocol_v15_sha256": protocol_sha,
        "backbone_rel": EXPECTED_SLOWFAST_CKPT_REL,
        "backbone_sha256": EXPECTED_SLOWFAST_CKPT_SHA256,
        "code_lock_git_revision": rev,
        "executable_tree_sha256": None,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "carry_forward_protocol_rel": D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL_REL,
        "quality_thresholds": {**DEFAULT_QUALITY_THRESHOLDS, **thr},
        "hard_stop": {
            "after_full_scan": True,
            "forbid_auto_chain_to_m0": True,
            "forbid_train_hpo_test_demo": True,
            "require_separate_authorization_for_m0": True,
            "pass_verdict": "D0_FULL_CORPUS_SCAN_COMPLETE_PASS",
            "fail_verdict": "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL",
        },
        "notes": [
            "DRAFT_REMEDIATION_NOT_SEALED — identity scaffolding only.",
            "allow_full_corpus_extraction=false; do not launch RGB/M0/train.",
            "Worktree may be dirty; binary_root_rel uses tip SHA for draft binding.",
            "Seal + independent audit + post-audit auth required before extraction.",
        ],
    }
    approval_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL, approval)

    audit = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_audit_v15",
        "audit_verdict": "DRAFT_NOT_AUDITED",
        "auditor": "ducpm196",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval_v15_sha256": approval_sha,
        "execution_protocol_v15_sha256": protocol_sha,
        "code_lock_git_revision": rev,
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "gpu_full_corpus_extraction_authorized": False,
        "extraction_started": False,
        "awaiting_carry_forward_audit_pass": True,
        "awaiting_external_reaudit_pass": True,
        "awaiting_post_audit_execution_authorization": True,
        "auto_chain": False,
        "m0_authorized": False,
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "notes": [
            "Draft scaffolding audit placeholder; independent PASS required before seal.",
        ],
    }
    audit_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL, audit)

    contract = {
        "schema": "ms_sta_v5_d0_full_corpus_run_contract_v15",
        "status": "DRAFT_REMEDIATION_NOT_SEALED",
        "campaign": CAMPAIGN_TOKEN_V15,
        "full_corpus_authorized": False,
        "auto_chain_to_m0": False,
        "output_version": OUTPUT_VERSION_V15,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "approval_v15_rel": D0_FULL_CORPUS_APPROVAL_V15_REL,
        "approval_v15_sha256": approval_sha,
        "execution_protocol_v15_rel": D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        "execution_protocol_v15_sha256": protocol_sha,
        "code_lock_git_revision": rev,
        "eligible_n": CORPUS_N,
        "split_counts": dict(SPLIT_COUNTS),
        "max_resumes": MAX_RESUMES,
        "resumes_used": 0,
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "started_at": None,
        "closed_at": None,
        "notes": [
            "DRAFT only. Not authorized to start extraction.",
            "Hard STOP after scan; do not open M0.",
        ],
    }
    contract_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL, contract)

    print(
        json.dumps(
            {
                "ok": True,
                "status": "DRAFT_REMEDIATION_NOT_SEALED",
                "approval_v15_sha256": approval_sha,
                "execution_protocol_v15_sha256": protocol_sha,
                "approval_audit_v15_sha256": audit_sha,
                "run_contract_v15_sha256": contract_sha,
                "code_lock_git_revision": rev,
                "binary_root_rel": bin_rel,
                "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
                "carry_forward_required": True,
                "allow_full_corpus_extraction": False,
                "parent_v14_halt_closeout_sha256": parent_halt_sha,
                "extraction_started": False,
                "sealed": False,
            },
            indent=2,
        )
    )
    return 0


def _seal(
    *,
    reseal_after_prelaunch_blocker: bool = False,
    reseal_after_orchestration_fail: bool = False,
    reseal_after_watchdog_cold_start_fail: bool = False,
    reseal_after_interrupted_rgb: bool = False,
    reseal_after_interrupted_bound_worker_rechecked_cold_start_gate: bool = False,
    reseal_after_quality_scan_local_skel_manifest: bool = False,
    reseal_after_scan_recovery_r1_perf: bool = False,
    reseal_after_scan_recovery_r2_audit_fail: bool = False,
    reseal_after_scan_recovery_r3_audit_fail: bool = False,
    reseal_after_scan_recovery_r4_execution_provenance: bool = False,
    reseal_after_scan_recovery_r5_meta_contract: bool = False,
    reseal_after_scan_recovery_r6_fail_closed: bool = False,
) -> int:
    dirty = git_porcelain(_ROOT).strip()
    if dirty:
        raise SystemExit(f"dirty_worktree_forbidden:{dirty[:400]}")

    reseal_meta: dict[str, str] | None = None
    reseal_flags = (
        int(bool(reseal_after_prelaunch_blocker))
        + int(bool(reseal_after_orchestration_fail))
        + int(bool(reseal_after_watchdog_cold_start_fail))
        + int(bool(reseal_after_interrupted_rgb))
        + int(bool(reseal_after_interrupted_bound_worker_rechecked_cold_start_gate))
        + int(bool(reseal_after_quality_scan_local_skel_manifest))
        + int(bool(reseal_after_scan_recovery_r1_perf))
        + int(bool(reseal_after_scan_recovery_r2_audit_fail))
        + int(bool(reseal_after_scan_recovery_r3_audit_fail))
        + int(bool(reseal_after_scan_recovery_r4_execution_provenance))
        + int(bool(reseal_after_scan_recovery_r5_meta_contract))
        + int(bool(reseal_after_scan_recovery_r6_fail_closed))
    )
    if reseal_flags > 1:
        raise SystemExit("reseal_flags_mutually_exclusive")
    if reseal_after_prelaunch_blocker:
        reseal_meta = _prepare_reseal_after_prelaunch_blocker()
    elif reseal_after_orchestration_fail:
        reseal_meta = _prepare_reseal_after_orchestration_fail()
    elif reseal_after_watchdog_cold_start_fail:
        reseal_meta = _prepare_reseal_after_watchdog_cold_start_fail()
    elif reseal_after_interrupted_rgb:
        reseal_meta = _prepare_reseal_after_interrupted_rgb()
    elif reseal_after_interrupted_bound_worker_rechecked_cold_start_gate:
        reseal_meta = _prepare_reseal_after_interrupted_bound_worker_rechecked_cold_start_gate()
    elif reseal_after_quality_scan_local_skel_manifest:
        reseal_meta = _prepare_reseal_after_quality_scan_local_skel_manifest()
    elif reseal_after_scan_recovery_r1_perf:
        reseal_meta = _prepare_reseal_after_scan_recovery_r1_perf()
    elif reseal_after_scan_recovery_r2_audit_fail:
        reseal_meta = _prepare_reseal_after_scan_recovery_r2_audit_fail()
    elif reseal_after_scan_recovery_r3_audit_fail:
        reseal_meta = _prepare_reseal_after_scan_recovery_r3_audit_fail()
    elif reseal_after_scan_recovery_r4_execution_provenance:
        reseal_meta = _prepare_reseal_after_scan_recovery_r4_execution_provenance()
    elif reseal_after_scan_recovery_r5_meta_contract:
        reseal_meta = _prepare_reseal_after_scan_recovery_r5_meta_contract()
    elif reseal_after_scan_recovery_r6_fail_closed:
        reseal_meta = _prepare_reseal_after_scan_recovery_r6_fail_closed()

    rev = git_rev(_ROOT)
    assert_binaries_gitignored(root=_ROOT)
    assert_no_forbidden_symbols(FULL_CORPUS_EXECUTABLE_PATHS_V15, root=_ROOT)
    assert_psutil_available()

    tree_rev = executable_tree_sha256_at_revision(FULL_CORPUS_EXECUTABLE_PATHS_V15, rev, root=_ROOT)
    for rel in FULL_CORPUS_EXECUTABLE_PATHS_V15:
        try:
            subprocess.check_call(
                ["git", "diff", "--quiet", rev, "--", rel],
                cwd=_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"executable_path_differs_from_head:{rel}") from exc

    halt_path = _ROOT / D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL
    if not halt_path.is_file():
        raise SystemExit("v14_halt_closeout_missing")
    parent_halt_sha = sha256_file(halt_path)

    carry_path = _ROOT / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
    if not carry_path.is_file():
        raise SystemExit("carry_forward_audit_v15_missing")
    carry = load_json(carry_path)
    if carry.get("ok") is not True or carry.get("verify_all_shas") is not True:
        raise SystemExit("carry_forward_audit_v15_not_pass")
    carry_sha = sha256_file(carry_path)

    smoke_path = _ROOT / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL
    if not smoke_path.is_file():
        raise SystemExit("rgb_f0_smoke_report_v15_missing")
    smoke = load_json(smoke_path)
    try:
        validate_rgb_f0_smoke_report_v2(smoke, root=_ROOT)
    except ValueError as exc:
        raise SystemExit(f"rgb_f0_smoke_report_v15_invalid:{exc}") from exc
    smoke_sha = sha256_file(smoke_path)

    bin_rel = binary_root_rel(rev, root=_ROOT, version=OUTPUT_VERSION_V15)
    assert_binary_root_absent_or_empty(_ROOT / bin_rel)

    env = env_provenance(root=_ROOT)
    thr = default_quality_thresholds()
    thr.update(
        {
            "require_quality_stats_artifact": True,
            "fail_closed_on_threshold_breach": True,
            "tensor_authoritative": True,
        }
    )

    evidence = _ROOT / EVIDENCE_ROOT_V15_REL
    evidence.mkdir(parents=True, exist_ok=True)
    for rel in (
        D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        D0_FULL_CORPUS_APPROVAL_V15_REL,
        D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
        D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
    ):
        _unlink_draft_only(_ROOT / rel)

    protocol = {
        "schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v15",
        "status": "APPROVED",
        "output_version": OUTPUT_VERSION_V15,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "skeleton_engine": "HandRecoveryEngine",
        "eligible_n": CORPUS_N,
        "split_counts": dict(SPLIT_COUNTS),
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "executable_paths": list(FULL_CORPUS_EXECUTABLE_PATHS_V15),
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "extract_python_rel": EXTRACT_PY_REL,
        "f0_python_rel": F0_PY_REL,
        "env_provenance": env,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "psutil_pin": PSUTIL_PIN,
        "worker_registry_required": True,
        "sample_transaction_required": True,
        "atomic_manifest_required": True,
        "allow_m0": False,
        "allow_auto_chain": False,
        "max_resumes": MAX_RESUMES,
        "watchdog_interval_sec": 600,
        "stop_after_pass": "D0_FULL_CORPUS_SCAN_COMPLETE_PASS",
        "stop_after_fail": "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL",
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "skeleton_carry_forward_audit_sha256": carry_sha,
        "rgb_f0_smoke_report_sha256": smoke_sha,
        "carry_forward_protocol_rel": D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL_REL,
        "notes": [
            "SEALED execution protocol v15; does not authorize launch alone.",
            "Requires independent external reaudit PASS + post-audit execution authorization.",
            "Empty binary root; skeleton carry-forward from sealed v14 only.",
        ],
    }
    if reseal_meta:
        protocol.update({k: v for k, v in reseal_meta.items() if k.endswith("_sha256")})
        protocol["notes"].append(
            "Resealed after remediation companion; prior seal/external/post-auth superseded."
        )
    validate_execution_protocol_v15(protocol, root=_ROOT)
    protocol_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL, protocol)

    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "status": "APPROVED",
        "decision": "APPROVED",
        "scope": "d0_full_corpus_extraction_only_v15",
        "operator": "ducpm196",
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "one_shot": True,
        "allow_full_corpus_extraction": True,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "output_version": OUTPUT_VERSION_V15,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "psutil_pin": PSUTIL_PIN,
        "max_resumes": MAX_RESUMES,
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "eligible_n": CORPUS_N,
        "eligible_by_split": dict(SPLIT_COUNTS),
        "eligible_manifest": ELIGIBLE_REL,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata": METADATA_REL,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers": SPLIT_REL,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
        "full_protocol": FULL_PROTOCOL_REL,
        "full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "r7c_amendment": FULL_PROTOCOL_R7C_AMENDMENT_REL,
        "r7c_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R7C_AMENDMENT_REL),
        "r7c_h1_independent_reaudit": PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "execution_protocol_v15": D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        "execution_protocol_v15_sha256": protocol_sha,
        "backbone_rel": EXPECTED_SLOWFAST_CKPT_REL,
        "backbone_sha256": EXPECTED_SLOWFAST_CKPT_SHA256,
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "executable_paths": list(FULL_CORPUS_EXECUTABLE_PATHS_V15),
        "env_provenance": env,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "skeleton_carry_forward_audit_sha256": carry_sha,
        "rgb_f0_smoke_report_sha256": smoke_sha,
        "carry_forward_protocol_rel": D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL_REL,
        "quality_thresholds": {**DEFAULT_QUALITY_THRESHOLDS, **thr},
        "hard_stop": {
            "after_full_scan": True,
            "forbid_auto_chain_to_m0": True,
            "forbid_train_hpo_test_demo": True,
            "require_separate_authorization_for_m0": True,
            "pass_verdict": "D0_FULL_CORPUS_SCAN_COMPLETE_PASS",
            "fail_verdict": "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL",
        },
        "notes": [
            "SEALED approval does NOT authorize launch without external reaudit PASS + post-audit auth.",
            "allow_full_corpus_extraction=true only after those companions exist.",
            "Does not authorize M0/train/HPO/test/demo/auto-chain.",
            "This writer does not start extraction and does not write EXTERNAL_REAUDIT or POST_AUTH.",
        ],
    }
    if reseal_meta:
        approval.update({k: v for k, v in reseal_meta.items() if k.endswith("_sha256")})
        approval["notes"].extend(
            [f"{k}={v}" for k, v in reseal_meta.items() if k.endswith("_sha256")]
            + ["Resealed after remediation companion; prior seal/external/post-auth superseded."]
        )
    validate_approval_v15(approval, root=_ROOT)
    approval_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL, approval)

    audit = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_audit_v15",
        "audit_verdict": "PASS",
        "auditor": "ducpm196",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval_v15_sha256": approval_sha,
        "execution_protocol_v15_sha256": protocol_sha,
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "skeleton_carry_forward_audit_sha256": carry_sha,
        "rgb_f0_smoke_report_sha256": smoke_sha,
        "gpu_full_corpus_extraction_authorized": False,
        "extraction_started": False,
        "awaiting_external_reaudit_pass": True,
        "awaiting_post_audit_execution_authorization": True,
        "auto_chain": False,
        "m0_authorized": False,
        "notes": [
            "Operator seal PASS; independent external reaudit PASS still required.",
            "Sealed approval alone does not authorize launch.",
        ],
    }
    if reseal_meta:
        audit.update({k: v for k, v in reseal_meta.items() if k.endswith("_sha256")})
    audit_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL, audit)

    contract = {
        "schema": "ms_sta_v5_d0_full_corpus_run_contract_v15",
        "status": "SEALED",
        "campaign": CAMPAIGN_TOKEN_V15,
        "full_corpus_authorized": True,
        "auto_chain_to_m0": False,
        "output_version": OUTPUT_VERSION_V15,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "binary_root_rel": bin_rel,
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
        "approval_v15_rel": D0_FULL_CORPUS_APPROVAL_V15_REL,
        "approval_v15_sha256": approval_sha,
        "execution_protocol_v15_rel": D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        "execution_protocol_v15_sha256": protocol_sha,
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "parent_v14_halt_closeout_sha256": parent_halt_sha,
        "skeleton_carry_forward_audit_sha256": carry_sha,
        "rgb_f0_smoke_report_sha256": smoke_sha,
        "eligible_n": CORPUS_N,
        "split_counts": dict(SPLIT_COUNTS),
        "max_resumes": MAX_RESUMES,
        "resumes_used": 0,
        "quality_thresholds": approval["quality_thresholds"],
        "started_at": None,
        "closed_at": None,
        "notes": [
            "SEALED but not launched. Await external reaudit PASS and post-audit execution authorization.",
            "Hard STOP after SCAN_COMPLETE; do not open M0.",
        ],
    }
    if reseal_meta:
        contract.update({k: v for k, v in reseal_meta.items() if k.endswith("_sha256")})
    validate_run_contract_v15(contract, root=_ROOT)
    contract_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL, contract)

    print(
        json.dumps(
            {
                "ok": True,
                "status": "SEALED",
                "approval_v15_sha256": approval_sha,
                "execution_protocol_v15_sha256": protocol_sha,
                "approval_audit_v15_sha256": audit_sha,
                "run_contract_v15_sha256": contract_sha,
                "code_lock_git_revision": rev,
                "executable_tree_sha256": tree_rev,
                "binary_root_rel": bin_rel,
                "allow_full_corpus_extraction": True,
                "extraction_started": False,
                "awaiting_external_reaudit_pass": True,
                "awaiting_post_audit_execution_authorization": True,
                "sealed": True,
                "resealed": bool(reseal_meta),
                "reseal_meta": reseal_meta,
                "wrote_external_reaudit": False,
                "wrote_post_audit_auth": False,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", action="store_true", help="Write draft artifacts")
    ap.add_argument(
        "--rewrite-draft",
        action="store_true",
        help="Replace existing DRAFT_* artifacts only (refuses non-draft)",
    )
    ap.add_argument("--seal", action="store_true", help="Seal APPROVED artifacts (clean worktree)")
    ap.add_argument(
        "--reseal-after-prelaunch-blocker",
        action="store_true",
        help="With --seal: supersede prior seal after PRELAUNCH_BLOCKER",
    )
    ap.add_argument(
        "--reseal-after-orchestration-fail",
        action="store_true",
        help="With --seal: supersede prior seal after orchestration independent-audit FAIL",
    )
    ap.add_argument(
        "--reseal-after-watchdog-cold-start-fail",
        action="store_true",
        help="With --seal: supersede prior seal after watchdog cold-start FAIL companion",
    )
    ap.add_argument(
        "--reseal-after-interrupted-rgb",
        action="store_true",
        help="With --seal: supersede prior seal after INTERRUPTED RGB worker-exit companion",
    )
    ap.add_argument(
        "--reseal-after-interrupted-bound-worker-rechecked-cold-start-gate",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after INTERRUPTED bound-worker "
            "rechecked cold-start gate companion"
        ),
    )
    ap.add_argument(
        "--reseal-after-quality-scan-local-skel-manifest",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after QUALITY_SCAN used local "
            "skeleton manifest under carry-forward companion"
        ),
    )
    ap.add_argument(
        "--reseal-after-scan-recovery-r1-perf",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after SCAN_RECOVERY R1 performance "
            "remediation companion"
        ),
    )
    ap.add_argument(
        "--reseal-after-scan-recovery-r2-audit-fail",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after SCAN_RECOVERY R2 independent "
            "audit FAIL remediation companion"
        ),
    )
    ap.add_argument(
        "--reseal-after-scan-recovery-r3-audit-fail",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after SCAN_RECOVERY R3 independent "
            "audit FAIL remediation companion"
        ),
    )
    ap.add_argument(
        "--reseal-after-scan-recovery-r4-execution-provenance",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after SCAN_RECOVERY R4 execution "
            "provenance remediation companion"
        ),
    )
    ap.add_argument(
        "--reseal-after-scan-recovery-r5-meta-contract",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after SCAN_RECOVERY R5 meta-contract "
            "remediation companion"
        ),
    )
    ap.add_argument(
        "--reseal-after-scan-recovery-r6-fail-closed",
        action="store_true",
        help=(
            "With --seal: supersede prior seal after SCAN_RECOVERY R6 fail-closed "
            "remediation companion"
        ),
    )
    args = ap.parse_args()

    if args.reseal_after_prelaunch_blocker and not args.seal:
        raise SystemExit("reseal_after_prelaunch_blocker_requires_seal")
    if args.reseal_after_orchestration_fail and not args.seal:
        raise SystemExit("reseal_after_orchestration_fail_requires_seal")
    if args.reseal_after_watchdog_cold_start_fail and not args.seal:
        raise SystemExit("reseal_after_watchdog_cold_start_fail_requires_seal")
    if args.reseal_after_interrupted_rgb and not args.seal:
        raise SystemExit("reseal_after_interrupted_rgb_requires_seal")
    if args.reseal_after_interrupted_bound_worker_rechecked_cold_start_gate and not args.seal:
        raise SystemExit(
            "reseal_after_interrupted_bound_worker_rechecked_cold_start_gate_requires_seal"
        )
    if args.reseal_after_quality_scan_local_skel_manifest and not args.seal:
        raise SystemExit("reseal_after_quality_scan_local_skel_manifest_requires_seal")
    if args.reseal_after_scan_recovery_r1_perf and not args.seal:
        raise SystemExit("reseal_after_scan_recovery_r1_perf_requires_seal")
    if args.reseal_after_scan_recovery_r2_audit_fail and not args.seal:
        raise SystemExit("reseal_after_scan_recovery_r2_audit_fail_requires_seal")
    if args.reseal_after_scan_recovery_r3_audit_fail and not args.seal:
        raise SystemExit("reseal_after_scan_recovery_r3_audit_fail_requires_seal")
    if args.reseal_after_scan_recovery_r4_execution_provenance and not args.seal:
        raise SystemExit("reseal_after_scan_recovery_r4_execution_provenance_requires_seal")
    if args.reseal_after_scan_recovery_r5_meta_contract and not args.seal:
        raise SystemExit("reseal_after_scan_recovery_r5_meta_contract_requires_seal")
    if args.reseal_after_scan_recovery_r6_fail_closed and not args.seal:
        raise SystemExit("reseal_after_scan_recovery_r6_fail_closed_requires_seal")
    if args.seal:
        return _seal(
            reseal_after_prelaunch_blocker=bool(args.reseal_after_prelaunch_blocker),
            reseal_after_orchestration_fail=bool(args.reseal_after_orchestration_fail),
            reseal_after_watchdog_cold_start_fail=bool(args.reseal_after_watchdog_cold_start_fail),
            reseal_after_interrupted_rgb=bool(args.reseal_after_interrupted_rgb),
            reseal_after_interrupted_bound_worker_rechecked_cold_start_gate=bool(
                args.reseal_after_interrupted_bound_worker_rechecked_cold_start_gate
            ),
            reseal_after_quality_scan_local_skel_manifest=bool(
                args.reseal_after_quality_scan_local_skel_manifest
            ),
            reseal_after_scan_recovery_r1_perf=bool(args.reseal_after_scan_recovery_r1_perf),
            reseal_after_scan_recovery_r2_audit_fail=bool(
                args.reseal_after_scan_recovery_r2_audit_fail
            ),
            reseal_after_scan_recovery_r3_audit_fail=bool(
                args.reseal_after_scan_recovery_r3_audit_fail
            ),
            reseal_after_scan_recovery_r4_execution_provenance=bool(
                args.reseal_after_scan_recovery_r4_execution_provenance
            ),
            reseal_after_scan_recovery_r5_meta_contract=bool(
                args.reseal_after_scan_recovery_r5_meta_contract
            ),
            reseal_after_scan_recovery_r6_fail_closed=bool(
                args.reseal_after_scan_recovery_r6_fail_closed
            ),
        )
    return _write_draft(rewrite=bool(args.rewrite_draft))


if __name__ == "__main__":
    raise SystemExit(main())
