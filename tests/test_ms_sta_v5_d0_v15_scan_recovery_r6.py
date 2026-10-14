"""Adversarial R6 fail-closed remediation tests for meta/legacy/diagnostic binds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
    D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION_V15,
    SCAN_RECOVERY_EXTERNAL_REAUDIT_SCHEMA_V15,
    assert_meta_tensor_agreement,
    sha256_file,
    validate_scan_recovery_external_reaudit_pass_v15,
)


def test_executable_paths_include_r6() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_scan_recovery_r6.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def _rv(*, all_missing: bool, partial_missing: bool, n_valid: int, n_slices: int) -> dict:
    return {
        "n_slices": n_slices,
        "n_valid": n_valid,
        "valid_fraction": (float(n_valid) / float(n_slices)) if n_slices else 0.0,
        "all_missing": all_missing,
        "partial_missing": partial_missing,
    }


def _plant(
    tmp_path: Path, *, windows: list[dict], miss: dict[str, list[bool]], part: dict[str, list[bool]]
) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rgb = tmp_path / "sample.npz"
    payload = {}
    for r in ("face", "left_hand", "right_hand"):
        payload[f"missing_{r}"] = np.asarray(miss[r], dtype=bool)
        payload[f"partial_{r}"] = np.asarray(part[r], dtype=bool)
    np.savez(rgb, **payload)
    meta = {"sample_id": "000001", "windows": windows}
    return rgb, meta


def _good_region_validity(*, face_missing: bool = False) -> dict:
    return {
        "face": _rv(
            all_missing=face_missing,
            partial_missing=False,
            n_valid=0 if face_missing else 40,
            n_slices=40,
        ),
        "left_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
        "right_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
    }


def test_zero_slice_accepted_is_false(tmp_path: Path) -> None:
    """ZERO_SLICE_ACCEPTED must be false."""
    windows = [
        {
            "region_validity": {
                "face": _rv(all_missing=True, partial_missing=False, n_valid=0, n_slices=0),
                "left_hand": _rv(all_missing=True, partial_missing=False, n_valid=0, n_slices=0),
                "right_hand": _rv(all_missing=True, partial_missing=False, n_valid=0, n_slices=0),
            }
        }
    ]
    miss = {r: [True] for r in ("face", "left_hand", "right_hand")}
    part = {r: [False] for r in ("face", "left_hand", "right_hand")}
    rgb, meta = _plant(tmp_path, windows=windows, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="counts_inconsistent"):
        assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")
    print("ZERO_SLICE_ACCEPTED=false")


def test_empty_legacy_mask_accepted_is_false(tmp_path: Path) -> None:
    """EMPTY_LEGACY_MASK_ACCEPTED must be false."""
    windows = [
        {
            "missing_region_mask": {},
            "region_validity": _good_region_validity(),
        }
    ]
    miss = {r: [False] for r in ("face", "left_hand", "right_hand")}
    part = {r: [False] for r in ("face", "left_hand", "right_hand")}
    rgb, meta = _plant(tmp_path, windows=windows, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="legacy_missing_region_mask_keys"):
        assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")
    print("EMPTY_LEGACY_MASK_ACCEPTED=false")


def test_incomplete_legacy_mask_accepted_is_false(tmp_path: Path) -> None:
    """INCOMPLETE_LEGACY_MASK_ACCEPTED must be false."""
    windows = [
        {
            "missing_region_mask": {"face": False, "left_hand": False},
            "region_validity": _good_region_validity(),
        }
    ]
    miss = {r: [False] for r in ("face", "left_hand", "right_hand")}
    part = {r: [False] for r in ("face", "left_hand", "right_hand")}
    rgb, meta = _plant(tmp_path, windows=windows, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="legacy_missing_region_mask_keys"):
        assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")
    print("INCOMPLETE_LEGACY_MASK_ACCEPTED=false")


def test_non_boolean_legacy_value_accepted_is_false(tmp_path: Path) -> None:
    """NON_BOOLEAN_LEGACY_VALUE_ACCEPTED must be false."""
    windows = [
        {
            "missing_region_mask": {
                "face": 0,  # not bool
                "left_hand": False,
                "right_hand": False,
            },
            "region_validity": _good_region_validity(),
        }
    ]
    miss = {r: [False] for r in ("face", "left_hand", "right_hand")}
    part = {r: [False] for r in ("face", "left_hand", "right_hand")}
    rgb, meta = _plant(tmp_path, windows=windows, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="legacy_missing_region_mask_bad_type"):
        assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")
    print("NON_BOOLEAN_LEGACY_VALUE_ACCEPTED=false")


def test_omitted_diagnostic_sha_accepted_is_false(tmp_path: Path) -> None:
    """OMITTED_DIAGNOSTIC_SHA_ACCEPTED must be false."""
    approval = {
        "status": "APPROVED",
        "decision": "APPROVED",
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": "b" * 64,
    }
    for rel, payload in (
        (D0_FULL_CORPUS_APPROVAL_V15_REL, approval),
        (D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL, {"x": 1}),
        (D0_FULL_CORPUS_RUN_CONTRACT_V15_REL, {"x": 1}),
        (D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL, {"x": 1}),
        (D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE_REL, {"ok": True}),
        (D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC_REL, {"ok": True}),
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    obj = {
        "schema": SCAN_RECOVERY_EXTERNAL_REAUDIT_SCHEMA_V15,
        "audit_verdict": "PASS",
        "auditor": "independent",
        "scope": "scan_recovery_only",
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "allow_extraction": False,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "blocking_findings": [],
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": "b" * 64,
        "approval_v15_sha256": sha256_file(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL),
        "execution_protocol_v15_sha256": sha256_file(
            tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
        ),
        "run_contract_v15_sha256": sha256_file(tmp_path / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL),
        "scan_recovery_r1_perf_remediation_sha256": sha256_file(
            tmp_path / D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL
        ),
        "scan_recovery_perf_smoke_sha256": sha256_file(
            tmp_path / D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE_REL
        ),
        # intentionally omit rgb_region_validity_diagnostic_sha256
    }
    with pytest.raises(ValueError, match="rgb_region_validity_diagnostic_sha256_required"):
        validate_scan_recovery_external_reaudit_pass_v15(obj, root=tmp_path)
    print("OMITTED_DIAGNOSTIC_SHA_ACCEPTED=false")
