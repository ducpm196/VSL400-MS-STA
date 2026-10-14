#!/usr/bin/env python3
"""Create versioned F0-R1 execution DAG (approval→lock→audit→seal). No GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from f0_feasibility.campaign_paths import F0_R1_PATHS  # noqa: E402
from ms_sta_v5_f0_lib import (  # noqa: E402
    ENV_REALIZATION_REL,
    EXPECTED_INDEX_POLICY_SHA256,
    EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
    EXPECTED_PROTOCOL_PLAN_SHA256,
    EXPECTED_REQUIREMENTS_LOCK_SHA256,
    EXPECTED_SNAPSHOT_CONTRACT_SHA256,
    executable_tree_sha,
    load_json,
    sha256_file,
    validate_execution_approval,
    validate_execution_audit,
    validate_execution_lock,
    validate_execution_seal,
    validate_realization_manifest_audit,
    validate_videomae_small_conversion_evidence,
)


def _write(rel: str, obj: dict) -> str:
    path = _ROOT / rel
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError(f"crlf_forbidden:{rel}")
    if path.is_file():
        raise RuntimeError(f"already_exists:{rel}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256_file(path)


def main() -> int:
    weights_sha = sha256_file(_ROOT / F0_R1_PATHS.weights_manifest_rel)
    env_sha = sha256_file(_ROOT / ENV_REALIZATION_REL)
    man_sha = sha256_file(_ROOT / F0_R1_PATHS.realization_manifest_rel)
    raudit_sha = sha256_file(_ROOT / F0_R1_PATHS.realization_manifest_audit_rel)
    conv_sha = sha256_file(_ROOT / F0_R1_PATHS.conversion_evidence_rel)
    tree_sha = executable_tree_sha(root=_ROOT)

    validate_realization_manifest_audit(
        load_json(_ROOT / F0_R1_PATHS.realization_manifest_audit_rel),
        root=_ROOT,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        weights_manifest_sha256=weights_sha,
        env_realization_sha256=env_sha,
        realization_manifest_sha256=man_sha,
        realization_manifest_rel=F0_R1_PATHS.realization_manifest_rel,
    )
    validate_videomae_small_conversion_evidence(
        load_json(_ROOT / F0_R1_PATHS.conversion_evidence_rel),
        root=_ROOT,
        weights_manifest_sha256=weights_sha,
        weights_manifest_rel=F0_R1_PATHS.weights_manifest_rel,
    )

    ts = "2026-08-13T01:00:00Z"
    approval = {
        "schema": "ms_sta_v5_f0_execution_approval_v1",
        "status": "APPROVED",
        "decision": "APPROVED",
        "one_shot": True,
        "campaign": "f0_r1",
        "operator": "ducpm196",
        "approved_at": ts,
        "scope": "backbone_feasibility_gpu_only",
        "allow_gpu_feasibility": True,
        "allow_accuracy": False,
        "allow_d0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "env_realization_sha256": env_sha,
        "realization_manifest_sha256": man_sha,
        "weights_manifest_sha256": weights_sha,
        "executable_tree_sha256": tree_sha,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "requirements_lock_sha256": EXPECTED_REQUIREMENTS_LOCK_SHA256,
        "realization_manifest_audit_sha256": raudit_sha,
        "videomae_small_conversion_evidence_sha256": conv_sha,
        "stop_after": "F0_R1_FINAL_AUDIT",
        "note": (
            "Fresh F0-R1 one-shot after SlowFast matcher + VideoMAE metadata remediation. "
            "Original F0_FINAL_AUDIT=FAIL remains immutable."
        ),
        "parent_f0_final_audit": "Docs/results/ms_sta_v5_f0/F0_FINAL_AUDIT.json",
    }
    approval_sha = _write(F0_R1_PATHS.execution_approval_rel, approval)
    validate_execution_approval(
        approval,
        root=_ROOT,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        env_realization_sha256=env_sha,
        realization_manifest_sha256=man_sha,
        weights_manifest_sha256=weights_sha,
        executable_tree_sha256=tree_sha,
        realization_manifest_audit_sha256=raudit_sha,
        videomae_small_conversion_evidence_sha256=conv_sha,
        realization_manifest_audit_rel=F0_R1_PATHS.realization_manifest_audit_rel,
        conversion_evidence_rel=F0_R1_PATHS.conversion_evidence_rel,
        realization_manifest_rel=F0_R1_PATHS.realization_manifest_rel,
        weights_manifest_rel=F0_R1_PATHS.weights_manifest_rel,
    )

    lock = {
        "schema": "ms_sta_v5_f0_execution_lock_v1",
        "status": "EXECUTION_LOCKED",
        "campaign": "f0_r1",
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "weights_manifest_sha256": weights_sha,
        "env_realization_sha256": env_sha,
        "realization_manifest_sha256": man_sha,
        "executable_tree_sha256": tree_sha,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "execution_approval_sha256": approval_sha,
    }
    lock_sha = _write(F0_R1_PATHS.execution_lock_rel, lock)
    validate_execution_lock(
        lock,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        weights_manifest_sha256=weights_sha,
        env_realization_sha256=env_sha,
        realization_manifest_sha256=man_sha,
        executable_tree_sha256=tree_sha,
        execution_approval_sha256=approval_sha,
    )

    audit = {
        "schema": "ms_sta_v5_f0_execution_audit_v1",
        "audit_verdict": "PASS",
        "campaign": "f0_r1",
        "gpu_authorized": False,
        "auditor": "ducpm196",
        "audited_at": ts,
        "execution_approval_sha256": approval_sha,
        "execution_lock_sha256": lock_sha,
        "executable_tree_sha256": tree_sha,
        "notes": [
            "F0-R1 execution DAG audit PASS. GPU authorization deferred to R1 seal tip only.",
        ],
    }
    audit_sha = _write(F0_R1_PATHS.execution_audit_rel, audit)
    validate_execution_audit(
        audit,
        execution_approval_sha256=approval_sha,
        execution_lock_sha256=lock_sha,
        executable_tree_sha256=tree_sha,
    )

    seal = {
        "schema": "ms_sta_v5_f0_execution_seal_v1",
        "status": "SEALED",
        "campaign": "f0_r1",
        "gpu_authorized": True,
        "one_shot": True,
        "operator": "ducpm196",
        "sealed_at": ts,
        "execution_approval_sha256": approval_sha,
        "execution_lock_sha256": lock_sha,
        "execution_audit_sha256": audit_sha,
        "executable_tree_sha256": tree_sha,
        "dag": "approval -> lock -> audit -> seal",
        "stop_after": "F0_R1_FINAL_AUDIT",
        "forbid": ["d0", "training", "hpo", "test", "accuracy", "auto_chain"],
        "parent_f0_final_audit_immutable": True,
    }
    seal_sha = _write(F0_R1_PATHS.execution_seal_rel, seal)
    validate_execution_seal(
        seal,
        execution_approval_sha256=approval_sha,
        execution_lock_sha256=lock_sha,
        execution_audit_sha256=audit_sha,
        executable_tree_sha256=tree_sha,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "campaign": "f0_r1",
                "execution_approval_sha256": approval_sha,
                "execution_lock_sha256": lock_sha,
                "execution_audit_sha256": audit_sha,
                "execution_seal_sha256": seal_sha,
                "executable_tree_sha256": tree_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
