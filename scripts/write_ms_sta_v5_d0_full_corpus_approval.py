#!/usr/bin/env python3
"""Write separate D0 full-corpus approval + run-contract (no extraction)."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    D0_FULL_CORPUS_APPROVAL_AUDIT_REL,
    D0_FULL_CORPUS_APPROVAL_REL,
    D0_FULL_CORPUS_OUT_DIR_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_REL,
    ELIGIBLE_REL,
    EXPECTED_ELIGIBLE_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_R0_AUDIT_SHA256,
    EXPECTED_R0_LOCK_SHA256,
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
    EXPECTED_SPLIT_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R7C_AMENDMENT_REL,
    METADATA_REL,
    PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL,
    R0_AUDIT_REL,
    R0_LOCK_REL,
    SPLIT_REL,
    git_rev,
    load_json,
    sha256_file,
    validate_d0_full_corpus_approval,
    validate_d0_full_corpus_run_contract,
    validate_r7c_h1_independent_reaudit,
    write_json,
)


def main() -> int:
    validate_r7c_h1_independent_reaudit(load_json(_ROOT / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL))
    eligible = load_json(_ROOT / ELIGIBLE_REL)
    usage = shutil.disk_usage(str(_ROOT))
    free_gb = usage.free / (1024**3)
    # Empirical pilot footprint ~0.4 MB/sample; use 2.0 MB/sample conservative bound.
    estimated_need_gb = float(eligible["eligible_n"]) * 2.0 / 1024.0
    min_free_gb = max(80.0, estimated_need_gb * 1.25)

    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v1",
        "status": "APPROVED",
        "decision": "APPROVED",
        "scope": "d0_full_corpus_extraction_only",
        "operator": "ducpm196",
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "one_shot_full_corpus_authorization": True,
        "allow_full_corpus_extraction": True,
        "allow_m0": False,
        "allow_training": False,
        "allow_accuracy": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "stop_after": "D0_FULL_CORPUS_SCAN_COMPLETE",
        "output_version": "v1",
        "output_dir_rel": D0_FULL_CORPUS_OUT_DIR_REL,
        "eligible_n": int(eligible["eligible_n"]),
        "eligible_by_split": eligible.get("eligible_by_split"),
        "eligible_manifest": ELIGIBLE_REL,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata": METADATA_REL,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers": SPLIT_REL,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
        "r0_lock": R0_LOCK_REL,
        "r0_lock_sha256": EXPECTED_R0_LOCK_SHA256,
        "r0_audit": R0_AUDIT_REL,
        "r0_audit_sha256": EXPECTED_R0_AUDIT_SHA256,
        "full_protocol": FULL_PROTOCOL_REL,
        "full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "r7c_amendment": FULL_PROTOCOL_R7C_AMENDMENT_REL,
        "r7c_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R7C_AMENDMENT_REL),
        "r7c_h1_independent_reaudit": PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "code_lock_git_revision_at_approval": git_rev(_ROOT),
        "resume_contract": {
            "enabled": True,
            "max_resumes": 20,
            "idempotent_skip_existing_sha_matched_outputs": True,
            "resume_requires_same_output_version": True,
            "resume_requires_same_approval_sha256": True,
        },
        "disk_preflight": {
            "required": True,
            "min_free_gb": round(min_free_gb, 2),
            "estimated_need_gb": round(estimated_need_gb, 2),
            "observed_free_gb_at_approval": round(free_gb, 2),
            "preflight_pass_at_approval": free_gb >= min_free_gb,
            "fail_closed_if_insufficient": True,
        },
        "quality_thresholds": {
            "max_missing_roi_rate_face": 0.05,
            "max_missing_roi_rate_left_hand": 0.15,
            "max_missing_roi_rate_right_hand": 0.15,
            "max_partial_missing_rate_any_region": 0.25,
            "require_quality_stats_artifact": True,
            "fail_closed_on_threshold_breach": True,
        },
        "quality_stats_required": [
            "missing_roi_rate_by_region_from_tensor_validity",
            "missing_roi_rate_by_split_from_tensor_validity",
            "missing_roi_rate_by_signer_from_tensor_validity",
            "partial_missing_rate_by_region",
        ],
        "hard_stop": {
            "after_full_scan": True,
            "forbid_auto_chain_to_m0": True,
            "forbid_train_hpo_test_demo": True,
            "require_separate_authorization_for_m0": True,
        },
        "notes": [
            "Separate authorization after independent R7C-H1 PASS.",
            "Authorizes D0 full-corpus extraction into Docs/results/ms_sta_v5_d0_full/v1 only.",
            "Does not authorize M0, training, HPO, test, demo, or auto-chain.",
            "Hard STOP after full-scan completion and quality-stats emission.",
            "Disk preflight is fail-closed at extract start; approval may record insufficient free space.",
        ],
    }
    validate_d0_full_corpus_approval(approval, root=_ROOT)
    approval_sha = write_json(_ROOT / D0_FULL_CORPUS_APPROVAL_REL, approval, overwrite=True)

    audit = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_audit_v1",
        "audit_verdict": "PASS",
        "auditor": "ducpm196",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval_sha256": approval_sha,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "gpu_full_corpus_extraction_authorized": True,
        "auto_chain": False,
        "m0_authorized": False,
        "notes": [
            "Full-corpus approval chain PASS for extraction scope only.",
            "Run contract remains AUTHORIZED_NOT_STARTED until an explicit extract command.",
            "Hard STOP after D0_FULL_CORPUS_SCAN_COMPLETE; no M0 auto-chain.",
        ],
    }
    write_json(_ROOT / D0_FULL_CORPUS_APPROVAL_AUDIT_REL, audit, overwrite=True)

    contract = {
        "schema": "ms_sta_v5_d0_full_corpus_run_contract_v1",
        "status": "AUTHORIZED_NOT_STARTED",
        "campaign": "D0_FULL_CORPUS_V1",
        "full_corpus_authorized": True,
        "auto_chain_to_m0": False,
        "stop_after": "D0_FULL_CORPUS_SCAN_COMPLETE",
        "output_version": "v1",
        "output_dir_rel": D0_FULL_CORPUS_OUT_DIR_REL,
        "approval_rel": D0_FULL_CORPUS_APPROVAL_REL,
        "approval_sha256": approval_sha,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "eligible_n": int(eligible["eligible_n"]),
        "resume": {
            "enabled": True,
            "max_resumes": 20,
            "resumes_used": 0,
            "idempotent_skip_existing_sha_matched_outputs": True,
        },
        "disk_preflight": approval["disk_preflight"],
        "quality_thresholds": approval["quality_thresholds"],
        "quality_stats_required": approval["quality_stats_required"],
        "phases": ["skeleton_v3", "rgb_token_v3", "quality_stats", "completion_seal"],
        "started_at": None,
        "closed_at": None,
        "notes": [
            "Authorized but not started. Explicit extract command required.",
            "Hard STOP after full-scan; do not open M0.",
        ],
    }
    validate_d0_full_corpus_run_contract(contract, root=_ROOT)
    write_json(_ROOT / D0_FULL_CORPUS_RUN_CONTRACT_REL, contract, overwrite=True)
    print(
        json.dumps(
            {
                "ok": True,
                "approval_sha256": approval_sha,
                "status": "AUTHORIZED_NOT_STARTED",
                "output_dir_rel": D0_FULL_CORPUS_OUT_DIR_REL,
                "disk_preflight_pass": approval["disk_preflight"]["preflight_pass_at_approval"],
                "extraction_started": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
