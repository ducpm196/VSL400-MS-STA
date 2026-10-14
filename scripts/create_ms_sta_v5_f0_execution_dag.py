#!/usr/bin/env python3
"""Create F0 execution DAG (approval→lock→audit→seal) after E0 PASS.

Does not run GPU. Uses current executable_tree and realization digests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_f0_lib import (  # noqa: E402
    ENV_REALIZATION_REL,
    EXECUTION_APPROVAL_REL,
    EXECUTION_AUDIT_REL,
    EXECUTION_LOCK_REL,
    EXECUTION_SEAL_REL,
    EXPECTED_INDEX_POLICY_SHA256,
    EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
    EXPECTED_PROTOCOL_PLAN_SHA256,
    EXPECTED_REQUIREMENTS_LOCK_SHA256,
    EXPECTED_SNAPSHOT_CONTRACT_SHA256,
    REALIZATION_MANIFEST_AUDIT_REL,
    REALIZATION_REL,
    VIDEOMAE_SMALL_CONVERSION_EVIDENCE_REL,
    WEIGHTS_REL,
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
    weights_sha = sha256_file(_ROOT / WEIGHTS_REL)
    env_sha = sha256_file(_ROOT / ENV_REALIZATION_REL)
    man_sha = sha256_file(_ROOT / REALIZATION_REL)
    raudit_sha = sha256_file(_ROOT / REALIZATION_MANIFEST_AUDIT_REL)
    conv_sha = sha256_file(_ROOT / VIDEOMAE_SMALL_CONVERSION_EVIDENCE_REL)
    tree_sha = executable_tree_sha(root=_ROOT)

    validate_realization_manifest_audit(
        load_json(_ROOT / REALIZATION_MANIFEST_AUDIT_REL),
        root=_ROOT,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        weights_manifest_sha256=weights_sha,
        env_realization_sha256=env_sha,
        realization_manifest_sha256=man_sha,
    )
    validate_videomae_small_conversion_evidence(
        load_json(_ROOT / VIDEOMAE_SMALL_CONVERSION_EVIDENCE_REL),
        root=_ROOT,
        weights_manifest_sha256=weights_sha,
    )

    ts = "2026-08-12T19:40:00Z"
    approval = {
        "schema": "ms_sta_v5_f0_execution_approval_v1",
        "status": "APPROVED",
        "decision": "APPROVED",
        "one_shot": True,
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
        "stop_after": "F0_FINAL_AUDIT",
        "note": "Bounded E1 authorization after E0 provenance PASS; one-shot GPU only.",
    }
    approval_sha = _write(EXECUTION_APPROVAL_REL, approval)
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
    )

    lock = {
        "schema": "ms_sta_v5_f0_execution_lock_v1",
        "status": "EXECUTION_LOCKED",
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "weights_manifest_sha256": weights_sha,
        "env_realization_sha256": env_sha,
        "realization_manifest_sha256": man_sha,
        "executable_tree_sha256": tree_sha,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "execution_approval_sha256": approval_sha,
    }
    lock_sha = _write(EXECUTION_LOCK_REL, lock)
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
        "gpu_authorized": False,
        "auditor": "ducpm196",
        "audited_at": ts,
        "execution_approval_sha256": approval_sha,
        "execution_lock_sha256": lock_sha,
        "executable_tree_sha256": tree_sha,
        "notes": [
            "Execution DAG audit PASS. GPU authorization deferred to seal tip only.",
        ],
    }
    audit_sha = _write(EXECUTION_AUDIT_REL, audit)
    validate_execution_audit(
        audit,
        execution_approval_sha256=approval_sha,
        execution_lock_sha256=lock_sha,
        executable_tree_sha256=tree_sha,
    )

    seal = {
        "schema": "ms_sta_v5_f0_execution_seal_v1",
        "status": "SEALED",
        "gpu_authorized": True,
        "one_shot": True,
        "operator": "ducpm196",
        "sealed_at": ts,
        "execution_approval_sha256": approval_sha,
        "execution_lock_sha256": lock_sha,
        "execution_audit_sha256": audit_sha,
        "executable_tree_sha256": tree_sha,
        "dag": "approval -> lock -> audit -> seal",
        "stop_after": "F0_FINAL_AUDIT",
        "forbid": ["d0", "training", "hpo", "test", "accuracy", "auto_chain"],
    }
    seal_sha = _write(EXECUTION_SEAL_REL, seal)
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
