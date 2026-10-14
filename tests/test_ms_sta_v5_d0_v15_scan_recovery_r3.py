"""Adversarial R3 scan-recovery audit-fail remediation tests."""

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
    resolve_scan_recovery_attempt_dir,
    sha256_file,
    validate_scan_recovery_attempt_auth_binds,
    validate_scan_recovery_attempt_id,
    validate_scan_recovery_external_reaudit_pass_v15,
)


def test_executable_paths_include_r3() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_scan_recovery_r3.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def _minimal_pass_obj() -> dict:
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


def _plant_bind_files(root: Path) -> None:
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
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_path_traversal_attempt_accepted_is_false(tmp_path: Path) -> None:
    """PATH_TRAVERSAL_ATTEMPT_ACCEPTED must be false."""
    with pytest.raises(ValueError, match="path_traversal|absolute"):
        validate_scan_recovery_attempt_id("../../../../outside_evidence")
    with pytest.raises(ValueError, match="path_traversal|absolute|invalid"):
        resolve_scan_recovery_attempt_dir("../../../../outside_evidence", root=tmp_path)

    # Absolute-style and separator forms also rejected.
    for bad in (r"C:\outside", "/tmp/outside", r"..\..\escape", "a/b", "a\\b"):
        with pytest.raises(ValueError):
            validate_scan_recovery_attempt_id(bad)

    # Valid id stays under attempts root.
    aid, attempt_rt, _ = resolve_scan_recovery_attempt_dir("safe_attempt_1", root=tmp_path)
    attempts_root = (
        tmp_path / "Docs/results/ms_sta_v5_d0_full/v15/scan_recovery/attempts"
    ).resolve()
    assert aid == "safe_attempt_1"
    assert attempt_rt.resolve().relative_to(attempts_root)
    print("PATH_TRAVERSAL_ATTEMPT_ACCEPTED=false")


def test_contradictory_external_pass_accepted_is_false(tmp_path: Path) -> None:
    """CONTRADICTORY_EXTERNAL_PASS_ACCEPTED must be false."""
    _plant_bind_files(tmp_path)
    obj = _minimal_pass_obj()
    obj["approval_v15_sha256"] = sha256_file(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL)
    obj["execution_protocol_v15_sha256"] = sha256_file(
        tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
    )
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
    # Adversarial: blocking finding + allow_hpo/test/demo true — must reject.
    obj["blocking_findings"] = ["P0_CONTRADICTION"]
    obj["allow_hpo"] = True
    obj["allow_test"] = True
    obj["allow_demo"] = True
    with pytest.raises(ValueError) as excinfo:
        validate_scan_recovery_external_reaudit_pass_v15(obj, root=tmp_path)
    msg = str(excinfo.value)
    assert "blocking_findings_must_be_empty" in msg
    assert "allow_hpo_must_be_false" in msg
    assert "allow_test_must_be_false" in msg
    assert "allow_demo_must_be_false" in msg
    print("CONTRADICTORY_EXTERNAL_PASS_ACCEPTED=false")


def test_unbound_auth_attempt_accepted_is_false() -> None:
    """UNBOUND_AUTH_ATTEMPT_ACCEPTED must be false."""
    # Companion/code-lock only — missing approval/protocol/run-contract/PASS/post-auth/HEAD.
    unbound = {
        "companion_sha256": "a" * 64,
        "code_lock_git_revision": "b" * 40,
    }
    with pytest.raises(ValueError, match="attempt_auth_bind_"):
        validate_scan_recovery_attempt_auth_binds(unbound)
    # Syntactically complete but forged hex must also be rejected.
    forged = {
        "approval_v15_sha256": "a" * 64,
        "execution_protocol_v15_sha256": "b" * 64,
        "run_contract_v15_sha256": "c" * 64,
        "scan_recovery_external_reaudit_sha256": "d" * 64,
        "scan_recovery_post_audit_auth_sha256": "e" * 64,
        "execution_head": "f" * 40,
        "code_lock_git_revision": "1" * 40,
        "executable_tree_sha256": "2" * 64,
    }
    with pytest.raises(ValueError, match="attempt_auth_bind_"):
        validate_scan_recovery_attempt_auth_binds(forged)
    print("UNBOUND_AUTH_ATTEMPT_ACCEPTED=false")


def test_runner_rejects_path_traversal_attempt_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import run_ms_sta_v5_d0_full_corpus_scan_recovery_v15 as recovery

    monkeypatch.setattr(recovery, "_ROOT", tmp_path)
    evidence = tmp_path / "Docs/results/ms_sta_v5_d0_full/v15"
    evidence.mkdir(parents=True)
    companion = {
        "status": "QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD",
        "binary_root_rel": "data/features/rgb",
        "skeleton_source_root_rel": "data/features/skel",
    }
    (tmp_path / "Docs/results/ms_sta_v5_d0_full/v15/"
     "D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD.json"
     ).write_text(json.dumps(companion) + "\n", encoding="utf-8")
    (tmp_path / "data/features/rgb").mkdir(parents=True)
    (tmp_path / "data/features/skel").mkdir(parents=True)
    monkeypatch.setattr(
        recovery,
        "load_json",
        lambda path: (
            companion
            if "QUALITY_SCAN" in str(path)
            else {
                "output_version": OUTPUT_VERSION_V15,
                "quality_thresholds": {},
                "code_lock_git_revision": "a" * 40,
            }
        ),
    )
    monkeypatch.setattr(recovery, "validate_approval_v15", lambda *a, **k: None)
    monkeypatch.setattr(
        recovery,
        "evaluate_scan_recovery_execute_readiness_v15",
        lambda root=None: {"execution_ready": False, "errors": ["missing"]},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["recovery", "--attempt-id", "../../../../outside_evidence"],
    )
    rc = recovery.main()
    assert rc == 2
    # Must not create anything outside attempts root.
    outside = (tmp_path / "outside_evidence").resolve()
    assert not outside.exists()
    print("PATH_TRAVERSAL_ATTEMPT_ACCEPTED=false")


def test_started_failure_require_auth_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute path must refuse unbound auth when bind files absent."""
    import run_ms_sta_v5_d0_full_corpus_scan_recovery_v15 as recovery

    monkeypatch.setattr(recovery, "_ROOT", tmp_path)
    companion = {
        "status": "QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD",
        "binary_root_rel": "data/features/rgb",
        "skeleton_source_root_rel": "data/features/skel",
    }
    p = (
        tmp_path
        / "Docs/results/ms_sta_v5_d0_full/v15/"
        "D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD.json"
    )
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(companion) + "\n", encoding="utf-8")
    (tmp_path / "data/features/rgb").mkdir(parents=True)
    (tmp_path / "data/features/skel").mkdir(parents=True)

    monkeypatch.setattr(
        recovery,
        "load_json",
        lambda path: (
            companion
            if "QUALITY_SCAN" in str(path)
            else {
                "output_version": OUTPUT_VERSION_V15,
                "quality_thresholds": {},
                "code_lock_git_revision": "a" * 40,
                "executable_tree_sha256": "b" * 64,
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
            "execution_head": "a" * 40,
            "code_lock_git_revision": "b" * 40,
            "executable_tree_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        recovery,
        "build_scan_recovery_attempt_auth_binds",
        lambda root=None: (_ for _ in ()).throw(
            ValueError("scan_recovery_auth_bind_missing:approval_v15_sha256")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["recovery", "--execute", "--attempt-id", "unbound_auth_probe"],
    )
    with pytest.raises(ValueError, match="scan_recovery_auth_bind_missing"):
        recovery.main()
    print("UNBOUND_AUTH_ATTEMPT_ACCEPTED=false")
