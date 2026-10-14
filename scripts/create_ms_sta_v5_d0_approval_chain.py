#!/usr/bin/env python3
"""Create immutable D0 approval → protocol → lock → approval-audit chain (no extraction)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    APPROVAL_AUDIT_REL,
    APPROVAL_REL,
    ELIGIBLE_REL,
    EXPECTED_ELIGIBLE_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_R0_AUDIT_SHA256,
    EXPECTED_R0_LOCK_SHA256,
    EXPECTED_SPLIT_SHA256,
    F0_CLOSEOUT_REL,
    METADATA_REL,
    PROTOCOL_LOCK_REL,
    PROTOCOL_REL,
    R0_AUDIT_REL,
    R0_LOCK_REL,
    SPLIT_REL,
    git_rev,
    sha256_file,
    validate_d0_approval,
    validate_d0_protocol,
    validate_f0_phase_closeout,
    write_json,
)


def main() -> int:
    validate_f0_phase_closeout(
        json.loads((_ROOT / F0_CLOSEOUT_REL).read_text(encoding="utf-8")), root=_ROOT
    )
    closeout_sha = sha256_file(_ROOT / F0_CLOSEOUT_REL)
    code_lock = git_rev(_ROOT)
    ts = "2026-08-13T06:10:00Z"

    for rel, expected in (
        (R0_LOCK_REL, EXPECTED_R0_LOCK_SHA256),
        (R0_AUDIT_REL, EXPECTED_R0_AUDIT_SHA256),
        (ELIGIBLE_REL, EXPECTED_ELIGIBLE_SHA256),
        (METADATA_REL, EXPECTED_METADATA_SHA256),
        (SPLIT_REL, EXPECTED_SPLIT_SHA256),
    ):
        if sha256_file(_ROOT / rel) != expected:
            raise SystemExit(f"bind_drift:{rel}")

    approval = {
        "schema": "ms_sta_v5_d0_approval_v1",
        "status": "APPROVED",
        "decision": "APPROVED",
        "scope": "d0_pilot_only",
        "one_shot_pilot": True,
        "operator": "ducpm196",
        "approved_at": ts,
        "max_pilot_resumes": 2,
        "allow_full_corpus_extraction": False,
        "allow_m0": False,
        "allow_training": False,
        "allow_accuracy": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "allow_f0_r2": False,
        "r0_lock": R0_LOCK_REL,
        "r0_lock_sha256": EXPECTED_R0_LOCK_SHA256,
        "r0_audit": R0_AUDIT_REL,
        "r0_audit_sha256": EXPECTED_R0_AUDIT_SHA256,
        "f0_phase_closeout": F0_CLOSEOUT_REL,
        "f0_phase_closeout_sha256": closeout_sha,
        "eligible_manifest": ELIGIBLE_REL,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata": METADATA_REL,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers": SPLIT_REL,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
        "code_lock_commit": code_lock,
        "code_lock_sha256": code_lock,
        "stop_after": "D0_PILOT_AUDIT",
        "notes": [
            "Bounded D0 pilot authorization after F0 phase close-out.",
            "Stage-B SlowFast partial-unfreeze remains excluded (CONTRACT_INVALID_NOT_TESTED).",
            "Full-corpus extraction requires a separate later approval.",
        ],
    }
    approval_sha = write_json(_ROOT / APPROVAL_REL, approval)
    validate_d0_approval(approval, root=_ROOT)

    protocol = {
        "schema": "ms_sta_v5_d0_protocol_v1",
        "status": "PILOT",
        "approval_sha256": approval_sha,
        "full_corpus_authorized": False,
        "skeleton_v3": {
            "native_timestamps": True,
            "original_frame_index": True,
            "prefill_missing_mask": True,
            "joint_validity_confidence": True,
            "normalized_coordinates": True,
            "velocity": True,
            "bone_vectors": True,
            "joint_part_chirality_identity": True,
            "real_sequence_length": True,
            "padding_mask": True,
            "forbid_reconstruct_masks_from_v2": True,
        },
        "rgb_token_v3": {
            "backbone": "frozen_slowfast_r50",
            "hook": "blocks.4_before_blocks.5_pooling",
            "retain_slow_fast_pathways": True,
            "roi_method": "torchvision.ops.roi_align",
            "regions": ["global", "left_hand", "right_hand", "face"],
            "compact_tokens_only": True,
            "forbid_partial_unfreeze": True,
            "forbid_accuracy": True,
            "project_landmarks_into_frame_coords": True,
        },
        "pilot_gates": [
            "exact_sample_reconciliation",
            "no_split_test_leakage",
            "timestamp_frame_alignment",
            "roi_coordinates_bounded_visual_audit",
            "masks_verified_before_fill",
            "tensor_shapes_dtypes_finite",
            "deterministic_reextraction_hash",
            "storage_runtime_projection_24752",
            "failure_taxonomy_coverage_report",
            "clean_versioned_output",
        ],
        "envs": {
            "skeleton": "_venvs/extract/Scripts/python.exe",
            "rgb_tokens": "_venvs/f0_v5/Scripts/python.exe",
        },
    }
    protocol_sha = write_json(_ROOT / PROTOCOL_REL, protocol)
    validate_d0_protocol(protocol)

    lock = {
        "schema": "ms_sta_v5_d0_protocol_lock_v1",
        "status": "LOCKED",
        "protocol_sha256": protocol_sha,
        "approval_sha256": approval_sha,
        "f0_phase_closeout_sha256": closeout_sha,
        "code_lock_sha256": code_lock,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "locked_at": ts,
    }
    lock_sha = write_json(_ROOT / PROTOCOL_LOCK_REL, lock)

    audit = {
        "schema": "ms_sta_v5_d0_approval_audit_v1",
        "audit_verdict": "PASS",
        "auditor": "ducpm196",
        "audited_at": ts,
        "approval_sha256": approval_sha,
        "protocol_sha256": protocol_sha,
        "protocol_lock_sha256": lock_sha,
        "f0_phase_closeout_sha256": closeout_sha,
        "notes": [
            "D0 approval chain PASS. Extraction still requires frozen pilot sample + run contract.",
        ],
        "gpu_full_corpus_authorized": False,
        "auto_chain": False,
    }
    audit_sha = write_json(_ROOT / APPROVAL_AUDIT_REL, audit)

    print(
        json.dumps(
            {
                "ok": True,
                "approval_sha256": approval_sha,
                "protocol_sha256": protocol_sha,
                "protocol_lock_sha256": lock_sha,
                "approval_audit_sha256": audit_sha,
                "f0_phase_closeout_sha256": closeout_sha,
                "code_lock_sha256": code_lock,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
