#!/usr/bin/env python3
"""Seal D0 full-corpus execution protocol/approval/run-contract v13 (no extraction).

Frozen historical writer: uses only *_V13 constants / validators.
Binds immutable v12 approval and sealed v12 external re-audit FAIL companion.
Does not overwrite sealed v2–v12 artifacts. Does not write v14.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V13,
    CORPUS_N,
    D0_FULL_CORPUS_APPROVAL_AUDIT_V13_REL,
    D0_FULL_CORPUS_APPROVAL_V12_REL,
    D0_FULL_CORPUS_APPROVAL_V13_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V12_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_V13_REL,
    DEFAULT_QUALITY_THRESHOLDS,
    EVIDENCE_ROOT_V13_REL,
    EXTRACT_PY_REL,
    EXPECTED_V12_APPROVAL_SHA256,
    EXPECTED_V12_EXTERNAL_REAUDIT_SHA256,
    F0_PY_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V13,
    MAX_RESUMES,
    OUTPUT_VERSION_V13,
    PHASES_EXACT,
    PSUTIL_PIN,
    SPLIT_COUNTS,
    assert_binaries_gitignored,
    assert_no_forbidden_symbols,
    assert_psutil_available,
    atomic_create_json,
    binary_root_rel,
    default_quality_thresholds,
    env_provenance,
    validate_approval_v13,
    validate_execution_protocol_v13,
    validate_run_contract_v13,
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
    load_json,
    sha256_file,
    validate_r7c_h1_independent_reaudit,
)


def main() -> int:
    dirty = git_porcelain(_ROOT).strip()
    if dirty:
        raise SystemExit(f"dirty_worktree_forbidden:{dirty[:400]}")
    rev = git_rev(_ROOT)
    validate_r7c_h1_independent_reaudit(load_json(_ROOT / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL))
    assert_binaries_gitignored(root=_ROOT)
    assert_no_forbidden_symbols(FULL_CORPUS_EXECUTABLE_PATHS_V13, root=_ROOT)
    assert_psutil_available()

    tree_rev = executable_tree_sha256_at_revision(FULL_CORPUS_EXECUTABLE_PATHS_V13, rev, root=_ROOT)
    for rel in FULL_CORPUS_EXECUTABLE_PATHS_V13:
        try:
            subprocess.check_call(
                ["git", "diff", "--quiet", rev, "--", rel],
                cwd=_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"executable_path_differs_from_head:{rel}") from exc

    if sha256_file(_ROOT / EXPECTED_SLOWFAST_CKPT_REL) != EXPECTED_SLOWFAST_CKPT_SHA256:
        raise SystemExit("backbone_sha_drift")
    if sha256_file(_ROOT / D0_FULL_CORPUS_APPROVAL_V12_REL) != EXPECTED_V12_APPROVAL_SHA256:
        raise SystemExit("v12_approval_drift_forbidden")
    reaudit_v12_path = _ROOT / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V12_REL
    if not reaudit_v12_path.is_file():
        raise SystemExit("v12_external_reaudit_missing")
    if sha256_file(reaudit_v12_path) != EXPECTED_V12_EXTERNAL_REAUDIT_SHA256:
        raise SystemExit("v12_external_reaudit_drift_forbidden")

    eligible = load_json(_ROOT / ELIGIBLE_REL)
    if int(eligible["eligible_n"]) != CORPUS_N:
        raise SystemExit("eligible_n_mismatch")
    obs = {k: int((eligible.get("eligible_by_split") or {}).get(k) or -1) for k in SPLIT_COUNTS}
    if obs != SPLIT_COUNTS:
        raise SystemExit(f"split_counts_mismatch:{obs}")

    usage = shutil.disk_usage(str(_ROOT))
    free_gb = usage.free / (1024**3)
    estimated_need_gb = float(CORPUS_N) * 2.0 / 1024.0
    min_free_gb = max(80.0, estimated_need_gb * 1.25)
    env = env_provenance(root=_ROOT)
    if not env["extract_python_exists"] or not env["f0_python_exists"]:
        raise SystemExit("isolated_envs_missing")
    if not env.get("psutil_all_envs_ok"):
        raise SystemExit("psutil_pin_missing_in_envs")

    bin_rel = binary_root_rel(rev, root=_ROOT, version=OUTPUT_VERSION_V13)
    thr = default_quality_thresholds()
    thr.update(
        {
            "require_quality_stats_artifact": True,
            "fail_closed_on_threshold_breach": True,
            "tensor_authoritative": True,
        }
    )

    evidence = _ROOT / EVIDENCE_ROOT_V13_REL
    evidence.mkdir(parents=True, exist_ok=True)
    for rel in (
        D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL,
        D0_FULL_CORPUS_APPROVAL_V13_REL,
        D0_FULL_CORPUS_APPROVAL_AUDIT_V13_REL,
        D0_FULL_CORPUS_RUN_CONTRACT_V13_REL,
    ):
        if (_ROOT / rel).is_file():
            raise SystemExit(f"v13_artifact_already_sealed:{rel}")

    protocol = {
        "schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v13",
        "output_version": OUTPUT_VERSION_V13,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "skeleton_engine": "HandRecoveryEngine",
        "eligible_n": CORPUS_N,
        "split_counts": dict(SPLIT_COUNTS),
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "executable_paths": list(FULL_CORPUS_EXECUTABLE_PATHS_V13),
        "binary_root_rel": bin_rel,
        "evidence_root_rel": EVIDENCE_ROOT_V13_REL,
        "extract_python_rel": EXTRACT_PY_REL,
        "f0_python_rel": F0_PY_REL,
        "env_provenance": env,
        "campaign_token": CAMPAIGN_TOKEN_V13,
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
        "immutable_v12_approval_sha256": EXPECTED_V12_APPROVAL_SHA256,
        "parent_external_reaudit_v12_sha256": EXPECTED_V12_EXTERNAL_REAUDIT_SHA256,
        "notes": [
            "Execution protocol v13 after R11 remediation over immutable v12 FAIL companion.",
            "Empty binary root required; do not reuse prior diagnostic NPZ as canonical.",
            "Does not overwrite sealed v2–v12. Await external PASS then separate post-auth then explicit --launch.",
        ],
    }
    validate_execution_protocol_v13(protocol, root=_ROOT)
    protocol_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL, protocol)

    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v13",
        "status": "APPROVED",
        "decision": "APPROVED",
        "scope": "d0_full_corpus_extraction_only_v13",
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
        "output_version": OUTPUT_VERSION_V13,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "psutil_pin": PSUTIL_PIN,
        "max_resumes": MAX_RESUMES,
        "binary_root_rel": bin_rel,
        "evidence_root_rel": EVIDENCE_ROOT_V13_REL,
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
        "execution_protocol_v13": D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL,
        "execution_protocol_v13_sha256": protocol_sha,
        "immutable_v12_approval_sha256": EXPECTED_V12_APPROVAL_SHA256,
        "parent_external_reaudit_v12_sha256": EXPECTED_V12_EXTERNAL_REAUDIT_SHA256,
        "backbone_rel": EXPECTED_SLOWFAST_CKPT_REL,
        "backbone_sha256": EXPECTED_SLOWFAST_CKPT_SHA256,
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "executable_paths": list(FULL_CORPUS_EXECUTABLE_PATHS_V13),
        "env_provenance": env,
        "campaign_token": CAMPAIGN_TOKEN_V13,
        "resume_contract": {
            "enabled": True,
            "max_resumes": MAX_RESUMES,
            "idempotent_skip_existing_sha_matched_outputs": True,
            "require_manifest_sha_for_resume": True,
            "require_source_sha256": True,
            "worker_registry_blocks_resume": True,
            "sample_transaction_heal": True,
            "atomic_manifest_records_required": True,
            "resume_requires_same_output_version": True,
            "resume_requires_same_approval_sha256": True,
            "resume_requires_same_code_lock": True,
            "resume_requires_same_executable_tree": True,
            "live_process_tree_blocks_resume": True,
            "immutable_run_contract_separate_from_mutable_runtime": True,
            "require_post_audit_execution_authorization": True,
            "require_external_reaudit_pass": True,
        },
        "disk_preflight": {
            "required": True,
            "min_free_gb": round(min_free_gb, 2),
            "estimated_need_gb": round(estimated_need_gb, 2),
            "observed_free_gb_at_approval": round(free_gb, 2),
            "preflight_pass_at_approval": free_gb >= min_free_gb,
            "fail_closed_if_insufficient": True,
        },
        "quality_thresholds": {**DEFAULT_QUALITY_THRESHOLDS, **thr},
        "hard_stop": {
            "after_full_scan": True,
            "forbid_auto_chain_to_m0": True,
            "forbid_train_hpo_test_demo": True,
            "require_separate_authorization_for_m0": True,
            "pass_verdict": "D0_FULL_CORPUS_SCAN_COMPLETE_PASS",
            "fail_verdict": "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL",
        },
        "supersedes_v12_reason": [
            "R11 remediation after immutable v12 FAIL companion.",
            "v12 artifacts remain immutable including FAIL reaudit; prior binary roots are non-canonical.",
        ],
        "notes": [
            "Authorizes protocol/approval only; --launch requires external re-audit PASS at v13 evidence path.",
            "Requires separate D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json before extraction.",
            "Empty binary root; do not reuse prior diagnostic NPZ.",
            "Await external PASS then separate post-auth then explicit --launch.",
            "Does not authorize M0/train/HPO/test/demo/auto-chain.",
            "This writer does not start extraction.",
        ],
    }
    validate_approval_v13(approval, root=_ROOT)
    approval_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V13_REL, approval)

    audit = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_audit_v13",
        "audit_verdict": "PASS",
        "auditor": "ducpm196",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval_v13_sha256": approval_sha,
        "execution_protocol_v13_sha256": protocol_sha,
        "immutable_v12_approval_sha256": EXPECTED_V12_APPROVAL_SHA256,
        "parent_external_reaudit_v12_sha256": EXPECTED_V12_EXTERNAL_REAUDIT_SHA256,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "binary_root_rel": bin_rel,
        "evidence_root_rel": EVIDENCE_ROOT_V13_REL,
        "gpu_full_corpus_extraction_authorized": False,
        "extraction_started": False,
        "awaiting_external_reaudit_pass": True,
        "awaiting_post_audit_execution_authorization": True,
        "auto_chain": False,
        "m0_authorized": False,
        "notes": [
            "R11 remediation sealed as approval v13; external PASS re-audit and post-audit auth still required.",
            "Run contract remains AUTHORIZED_NOT_STARTED.",
            "v12 FAIL companion remains immutable; do not promote prior roots.",
        ],
    }
    atomic_create_json(_ROOT / D0_FULL_CORPUS_APPROVAL_AUDIT_V13_REL, audit)

    contract = {
        "schema": "ms_sta_v5_d0_full_corpus_run_contract_v13",
        "status": "AUTHORIZED_NOT_STARTED",
        "campaign": CAMPAIGN_TOKEN_V13,
        "full_corpus_authorized": True,
        "auto_chain_to_m0": False,
        "output_version": OUTPUT_VERSION_V13,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "binary_root_rel": bin_rel,
        "evidence_root_rel": EVIDENCE_ROOT_V13_REL,
        "approval_v13_rel": D0_FULL_CORPUS_APPROVAL_V13_REL,
        "approval_v13_sha256": approval_sha,
        "execution_protocol_v13_rel": D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL,
        "execution_protocol_v13_sha256": protocol_sha,
        "immutable_v12_approval_sha256": EXPECTED_V12_APPROVAL_SHA256,
        "parent_external_reaudit_v12_sha256": EXPECTED_V12_EXTERNAL_REAUDIT_SHA256,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_rev,
        "eligible_n": CORPUS_N,
        "split_counts": dict(SPLIT_COUNTS),
        "max_resumes": MAX_RESUMES,
        "resumes_used": 0,
        "disk_preflight": approval["disk_preflight"],
        "quality_thresholds": approval["quality_thresholds"],
        "started_at": None,
        "closed_at": None,
        "notes": [
            "Authorized but not started. Await external re-audit PASS and post-audit execution authorization.",
            "Hard STOP after SCAN_COMPLETE_PASS/FAIL; do not open M0.",
            "Empty binary root; do not reuse prior diagnostic NPZ.",
        ],
    }
    validate_run_contract_v13(contract, root=_ROOT)
    contract_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_RUN_CONTRACT_V13_REL, contract)

    print(
        json.dumps(
            {
                "ok": True,
                "approval_v13_sha256": approval_sha,
                "execution_protocol_v13_sha256": protocol_sha,
                "run_contract_v13_sha256": contract_sha,
                "status": "AUTHORIZED_NOT_STARTED",
                "code_lock_git_revision": rev,
                "executable_tree_sha256": tree_rev,
                "binary_root_rel": bin_rel,
                "evidence_root_rel": EVIDENCE_ROOT_V13_REL,
                "extraction_started": False,
                "awaiting_external_reaudit_pass": True,
                "awaiting_post_audit_execution_authorization": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
