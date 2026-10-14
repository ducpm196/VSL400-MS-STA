"""Adversarial R2 scan-recovery audit-fail remediation tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tests"))

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
    atomic_create_json,
    default_quality_thresholds,
    sha256_file,
    validate_scan_recovery_external_reaudit_pass_v15,
)
from scan_ms_sta_v5_d0_full_corpus import SourceShaVerifier, scan_full_corpus  # noqa: E402
from test_ms_sta_v5_d0_v15_scan_recovery import _patch_identity, _plant_dual_root  # noqa: E402


def test_executable_paths_include_r2() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_scan_recovery_r2.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def _minimal_pass_obj(root: Path) -> dict:
    return {
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
        "approval_v15_sha256": "c" * 64,
        "execution_protocol_v15_sha256": "d" * 64,
        "run_contract_v15_sha256": "e" * 64,
        "scan_recovery_r1_perf_remediation_sha256": "f" * 64,
        "scan_recovery_perf_smoke_sha256": "1" * 64,
        "rgb_region_validity_diagnostic_sha256": "2" * 64,
    }


def test_omitted_approval_sha_accepted_is_false(tmp_path: Path) -> None:
    """OMITTED_APPROVAL_SHA_ACCEPTED must be false."""
    # Plant approval-like files so other binds can be checked independently.
    evidence = tmp_path / "Docs" / "results" / "ms_sta_v5_d0_full" / "v15"
    evidence.mkdir(parents=True)
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

    obj = _minimal_pass_obj(tmp_path)
    # Correct all SHAs except omit approval field.
    obj["execution_protocol_v15_sha256"] = sha256_file(tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL)
    obj["run_contract_v15_sha256"] = sha256_file(tmp_path / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL)
    obj["scan_recovery_r1_perf_remediation_sha256"] = sha256_file(
        tmp_path / D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL
    )
    obj["scan_recovery_perf_smoke_sha256"] = sha256_file(
        tmp_path / D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE_REL
    )
    obj["rgb_region_validity_diagnostic_sha256"] = sha256_file(
        tmp_path / D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC_REL
    )
    del obj["approval_v15_sha256"]
    with pytest.raises(ValueError, match="approval_v15_sha256_required"):
        validate_scan_recovery_external_reaudit_pass_v15(obj, root=tmp_path)
    print("OMITTED_APPROVAL_SHA_ACCEPTED=false")


def test_missing_metadata_row_accepted_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MISSING_METADATA_ROW_ACCEPTED must be false."""
    ids = ["000001"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)
    # Force empty meta_by (adversarial bypass attempt).
    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus._load_maps",
        lambda root: ({i: "train" for i in ids}, {i: "s1" for i in ids}, {}),
    )
    thr = default_quality_thresholds()
    thr.update(
        {
            "max_missing_roi_rate_face": 1.0,
            "max_missing_roi_rate_left_hand": 1.0,
            "max_missing_roi_rate_right_hand": 1.0,
        }
    )
    report = scan_full_corpus(
        binary_root=rgb_bin,
        runtime_dir=rgb_rt,
        thresholds=thr,
        root=tmp_path,
        skeleton_tensor_root=skel_bin,
        skeleton_manifest_runtime=skel_rt,
        rgb_manifest_runtime=rgb_rt,
        output_version=OUTPUT_VERSION_V15,
        write_artifacts=False,
    )
    assert report["ok"] is False
    assert any(
        str(f.get("error")) == "metadata_row_missing:000001" for f in report["failures"]
    )
    # Direct verifier proof
    v = SourceShaVerifier(root=tmp_path, meta_by={})
    with pytest.raises(RuntimeError, match="metadata_row_missing:000001"):
        v.verify_sample(sample_id="000001", skel_declared="a" * 64, rgb_declared="a" * 64)
    print("MISSING_METADATA_ROW_ACCEPTED=false")


def test_crash_before_scan_report_has_immutable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CRASH_BEFORE_SCAN_REPORT_HAS_IMMUTABLE_FAILURE must be true."""
    import run_ms_sta_v5_d0_full_corpus_scan_recovery_v15 as recovery

    monkeypatch.setattr(recovery, "_ROOT", tmp_path)
    # Minimal companion + approval so execute path can start.
    evidence = tmp_path / "Docs" / "results" / "ms_sta_v5_d0_full" / "v15"
    evidence.mkdir(parents=True)
    companion = {
        "status": "QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD",
        "binary_root_rel": "data/features/rgb",
        "skeleton_source_root_rel": "data/features/skel",
    }
    atomic_create_json(
        tmp_path / "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD.json",
        companion,
    )
    # Create tensor roots so plan resolves.
    (tmp_path / "data/features/rgb").mkdir(parents=True)
    (tmp_path / "data/features/skel").mkdir(parents=True)

    # Stub approval validation and readiness to authorized.
    monkeypatch.setattr(
        recovery,
        "load_json",
        lambda path: (
            companion
            if "QUALITY_SCAN" in str(path)
            else {
                "output_version": OUTPUT_VERSION_V15,
                "quality_thresholds": default_quality_thresholds(),
                "code_lock_git_revision": "a" * 40,
            }
        ),
    )
    monkeypatch.setattr(recovery, "validate_approval_v15", lambda *a, **k: None)
    monkeypatch.setattr(
        recovery,
        "evaluate_scan_recovery_execute_readiness_v15",
        lambda root=None: {"execution_ready": True, "errors": []},
    )
    monkeypatch.setattr(
        recovery,
        "preflight_scan_recovery_execute",
        lambda root=None: {
            "ok": True,
            "execution_head": "f" * 40,
            "code_lock_git_revision": "g" * 40,
            "executable_tree_sha256": "h" * 64,
        },
    )
    monkeypatch.setattr(
        recovery,
        "build_scan_recovery_attempt_auth_binds",
        lambda root=None: {
            "approval_v15_sha256": "a" * 64,
            "execution_protocol_v15_sha256": "b" * 64,
            "run_contract_v15_sha256": "c" * 64,
            "scan_recovery_external_reaudit_sha256": "d" * 64,
            "scan_recovery_post_audit_auth_sha256": "e" * 64,
            "execution_head": "f" * 40,
            "code_lock_git_revision": "g" * 40,
            "executable_tree_sha256": "h" * 64,
        },
    )
    monkeypatch.setattr(
        recovery,
        "validate_scan_recovery_attempt_auth_binds",
        lambda obj, root=None: {"ok": True},
    )

    def boom(**kwargs):  # noqa: ANN003
        raise RuntimeError("injected_crash_before_report")

    monkeypatch.setattr(recovery, "scan_full_corpus", boom)

    monkeypatch.setattr(
        sys,
        "argv",
        ["recovery", "--execute", "--attempt-id", "crash_before_report"],
    )
    rc = recovery.main()
    assert rc == 2
    attempt = (
        tmp_path
        / "Docs/results/ms_sta_v5_d0_full/v15/scan_recovery/attempts/crash_before_report"
    )
    started = attempt / "ATTEMPT_STARTED.json"
    failure = attempt / "ATTEMPT_FAILURE.json"
    assert started.is_file()
    assert failure.is_file()
    started_obj = json.loads(started.read_text(encoding="utf-8"))
    failure_obj = json.loads(failure.read_text(encoding="utf-8"))
    for field in (
        "approval_v15_sha256",
        "execution_protocol_v15_sha256",
        "run_contract_v15_sha256",
        "scan_recovery_external_reaudit_sha256",
        "scan_recovery_post_audit_auth_sha256",
        "execution_head",
        "code_lock_git_revision",
    ):
        assert started_obj.get(field)
        assert failure_obj.get(field) == started_obj.get(field)
    assert not (attempt / "SCAN_REPORT.json").is_file()
    # Immutable: second write must refuse
    with pytest.raises(FileExistsError):
        atomic_create_json(failure, {"schema": "overwrite"})
    print("CRASH_BEFORE_SCAN_REPORT_HAS_IMMUTABLE_FAILURE=true")
