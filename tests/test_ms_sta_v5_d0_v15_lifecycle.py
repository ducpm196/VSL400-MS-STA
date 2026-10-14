"""Hermetic v15 lifecycle hardening tests (no live seal / no reaudit write)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
    D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL,
    D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER_REL,
    D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL,
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
    EVIDENCE_ROOT_V15_REL,
    EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION,
    EXTERNAL_REAUDIT_PASS_SCHEMA_V15,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION,
    OUTPUT_VERSION_V15,
    V14_SEALED_SKELETON_BINARY_ROOT_REL,
    assert_binary_root_absent_or_empty,
    atomic_create_json,
    evaluate_launch_readiness_v15,
    load_json,
    validate_approval_v15,
    validate_external_reaudit_pass_v15,
    validate_post_audit_execution_authorization_v15,
    validate_rgb_f0_smoke_report_v2,
)


def test_hermetic_default_output_version_remains_v14():
    assert OUTPUT_VERSION == "v14"


def test_executable_paths_v15_include_lifecycle_scripts():
    required = {
        "scripts/write_ms_sta_v5_d0_full_corpus_approval_v15.py",
        "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v15.py",
        "scripts/validate_ms_sta_v5_d0_full_corpus_external_reaudit_v15.py",
        "scripts/validate_ms_sta_v5_d0_full_corpus_v15_skeleton_carry_forward.py",
        "scripts/smoke_ms_sta_v5_d0_full_corpus_v15_rgb_f0.py",
        "scripts/watch_ms_sta_v5_d0_full_corpus_v15.ps1",
        "scripts/verify_ms_sta_v5_d0_full_corpus_v14_halt_closeout.py",
        "requirements-d0-full-corpus.txt",
        "tests/test_ms_sta_v5_d0_full_corpus.py",
        "tests/test_ms_sta_v5_d0_v15_rgb_remediation.py",
        "tests/test_ms_sta_v5_d0_v15_lifecycle.py",
        "tests/test_ms_sta_v5_d0_v15_prelaunch_remediation.py",
        "tests/test_ms_sta_v5_d0_v15_carry_forward_orchestration.py",
        "tests/test_ms_sta_v5_d0_v15_watchdog_cold_start.py",
        "tests/test_ms_sta_v5_d0_v15_worker_liveness.py",
        "tests/test_ms_sta_v5_d0_v15_bound_worker_attach.py",
        "tests/test_ms_sta_v5_d0_v15_scan_recovery.py",
        "tests/test_ms_sta_v5_d0_v15_scan_recovery_r1.py",
        "tests/test_ms_sta_v5_d0_v15_scan_recovery_r2.py",
        "tests/test_ms_sta_v5_d0_v15_scan_recovery_r3.py",
        "tests/test_ms_sta_v5_d0_v15_scan_recovery_r4.py",
        "tests/test_ms_sta_v5_d0_v15_scan_recovery_r5.py",
        "tests/test_ms_sta_v5_d0_v15_scan_recovery_r6.py",
        "scripts/run_ms_sta_v5_d0_full_corpus_scan_recovery_v15.py",
        "scripts/smoke_ms_sta_v5_d0_full_corpus_scan_recovery_perf_v15.py",
        "scripts/diagnose_ms_sta_v5_d0_v15_rgb_region_validity.py",
    }
    assert required.issubset(set(FULL_CORPUS_EXECUTABLE_PATHS_V15))
    assert len(FULL_CORPUS_EXECUTABLE_PATHS_V15) == len(set(FULL_CORPUS_EXECUTABLE_PATHS_V15))


def test_assert_binary_root_absent_or_empty_ok_missing(tmp_path: Path) -> None:
    assert_binary_root_absent_or_empty(tmp_path / "missing")


def test_assert_binary_root_absent_or_empty_rejects_files(tmp_path: Path) -> None:
    root = tmp_path / "bin"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "x.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="binary_root_not_empty"):
        assert_binary_root_absent_or_empty(root)


def test_seal_rejects_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.import_module("write_ms_sta_v5_d0_full_corpus_approval_v15")
    monkeypatch.setattr(mod, "git_porcelain", lambda root=None: " M dirty.txt\n")
    with pytest.raises(SystemExit, match="dirty_worktree_forbidden"):
        mod._seal()


def test_validate_approval_v15_rejects_null_executable_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    halt = tmp_path / D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL
    halt.parent.mkdir(parents=True, exist_ok=True)
    halt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.verify_code_lock_revision_binding",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.binary_root_rel",
        lambda rev=None, root=None, version=None: "data/features/ms_sta_v5_d0_full_v15_deadbeefdead",
    )
    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_file", lambda p: "a" * 64)
    obj = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "status": "APPROVED",
        "decision": "APPROVED",
        "allow_full_corpus_extraction": True,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": None,
        "binary_root_rel": "data/features/ms_sta_v5_d0_full_v15_deadbeefdead",
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "parent_v14_halt_closeout_sha256": "a" * 64,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
    }
    with pytest.raises(ValueError, match="approval_v15_executable_tree_null"):
        validate_approval_v15(obj, root=tmp_path)


def test_validate_approval_v15_rejects_wrong_executable_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    halt = tmp_path / D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL
    halt.parent.mkdir(parents=True, exist_ok=True)
    halt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.verify_code_lock_revision_binding",
        lambda *a, **k: ["code_lock_tree_not_bound_to_revision"],
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.binary_root_rel",
        lambda rev=None, root=None, version=None: "data/features/ms_sta_v5_d0_full_v15_deadbeefdead",
    )
    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_file", lambda p: "a" * 64)
    obj = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "status": "APPROVED",
        "decision": "APPROVED",
        "allow_full_corpus_extraction": True,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": "b" * 64,
        "binary_root_rel": "data/features/ms_sta_v5_d0_full_v15_deadbeefdead",
        "skeleton_source_root_rel": V14_SEALED_SKELETON_BINARY_ROOT_REL,
        "carry_forward_required": True,
        "parent_v14_halt_closeout_sha256": "a" * 64,
        "evidence_root_rel": EVIDENCE_ROOT_V15_REL,
    }
    with pytest.raises(ValueError, match="code_lock_tree_not_bound_to_revision"):
        validate_approval_v15(obj, root=tmp_path)


def _write_v15_sealed_artifacts(root: Path, *, code_lock: str = "a" * 40) -> dict[str, str]:
    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "status": "APPROVED",
        "decision": "APPROVED",
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": "e" * 64,
    }
    protocol = {"schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v15"}
    run_contract = {"schema": "ms_sta_v5_d0_full_corpus_run_contract_v15", "status": "SEALED"}
    audit = {"schema": "ms_sta_v5_d0_full_corpus_approval_audit_v15", "audit_verdict": "PASS"}
    carry = {"ok": True, "verify_all_shas": True, "mutated": False}
    smoke = {
        "schema": "ms_sta_v5_d0_full_corpus_v15_rgb_f0_smoke_report_v2",
        "status": "PASS",
        "ok": True,
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": "e" * 64,
        "python_exe": "py",
        "pip_freeze_sha256": "f" * 64,
        "backbone_sha256": "b" * 64,
        "source_video_sha256": "c" * 64,
        "skeleton_sha256": "d" * 64,
        "sample_id": "019484",
        "required_keys_ok": True,
        "shapes": {"token_global": [3, 2304]},
        "dtypes": {"token_global": "float32"},
        "finite_ok": True,
        "token_content_sha256": "t" * 64,
    }
    paths = {
        "approval": root / D0_FULL_CORPUS_APPROVAL_V15_REL,
        "protocol": root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
        "run_contract": root / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
        "audit": root / D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
        "carry": root / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
        "smoke": root / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL,
    }
    payloads = {
        "approval": approval,
        "protocol": protocol,
        "run_contract": run_contract,
        "audit": audit,
        "carry": carry,
        "smoke": smoke,
    }
    shas: dict[str, str] = {}
    for key, payload in payloads.items():
        path = paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        shas[key] = atomic_create_json(path, payload)
    return shas


def _strict_pass_payload_v15(shas: dict[str, str], *, code_lock: str = "a" * 40) -> dict:
    return {
        "schema": EXTERNAL_REAUDIT_PASS_SCHEMA_V15,
        "audit_verdict": "PASS",
        "auditor": "independent",
        "worktree_clean_before_audit_artifact": True,
        "head_at_audit": code_lock,
        "origin_main_at_audit": code_lock,
        "code_lock_git_revision": code_lock,
        "approval_v15_sha256": shas["approval"],
        "execution_protocol_v15_sha256": shas["protocol"],
        "run_contract_v15_sha256": shas["run_contract"],
        "approval_audit_v15_sha256": shas["audit"],
        "skeleton_carry_forward_audit_sha256": shas["carry"],
        "rgb_f0_smoke_report_sha256": shas["smoke"],
        "blocking_findings": [],
        "decision": dict(EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION),
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
    }


def _patch_external_reaudit_pass_v15_deps(
    monkeypatch: pytest.MonkeyPatch,
    shas: dict[str, str],
    *,
    code_lock: str,
    git_blob_sha: dict[str, str] | None = None,
    commit_exists: bool = True,
) -> None:
    def _fake_sha(path: Path) -> str:
        text = str(path).replace("\\", "/")
        mapping = {
            D0_FULL_CORPUS_APPROVAL_V15_REL: shas["approval"],
            D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL: shas["protocol"],
            D0_FULL_CORPUS_RUN_CONTRACT_V15_REL: shas["run_contract"],
            D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL: shas["audit"],
            D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL: shas["carry"],
            D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL: shas["smoke"],
        }
        for rel, digest in mapping.items():
            if text.endswith(rel):
                return digest
        raise AssertionError(f"unexpected_sha_path:{path}")

    def _fake_git_blob(rel: str, rev: str, *, root: Path) -> str:
        if git_blob_sha is not None and rel in git_blob_sha:
            return git_blob_sha[rel]
        mapping = {
            D0_FULL_CORPUS_APPROVAL_V15_REL: shas["approval"],
            D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL: shas["protocol"],
            D0_FULL_CORPUS_RUN_CONTRACT_V15_REL: shas["run_contract"],
            D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL: shas["audit"],
            D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL: shas["carry"],
            D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL: shas["smoke"],
        }
        if rel in mapping:
            return mapping[rel]
        raise FileNotFoundError(rel)

    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_file", _fake_sha)
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.git_commit_exists", lambda rev, root=None: commit_exists
    )
    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_git_blob_at_revision", _fake_git_blob)


def test_forged_external_pass_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v15_sealed_artifacts(tmp_path)
    _patch_external_reaudit_pass_v15_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload_v15(shas)
    obj["approval_v15_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="approval_v15_sha256_mismatch"):
        validate_external_reaudit_pass_v15(obj, root=tmp_path)


def test_stale_code_lock_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v15_sealed_artifacts(tmp_path, code_lock="a" * 40)
    _patch_external_reaudit_pass_v15_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload_v15(shas, code_lock="c" * 40)
    with pytest.raises(ValueError, match="external_reaudit_v15_code_lock_mismatch"):
        validate_external_reaudit_pass_v15(obj, root=tmp_path)


def test_post_auth_wrong_audit_sha_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v15_sealed_artifacts(tmp_path, code_lock="a" * 40)
    reaudit = _strict_pass_payload_v15(shas, code_lock="a" * 40)
    reaudit_path = tmp_path / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    reaudit_sha = atomic_create_json(reaudit_path, reaudit)

    def _fake_sha(path: Path) -> str:
        text = str(path).replace("\\", "/")
        if text.endswith(D0_FULL_CORPUS_APPROVAL_V15_REL):
            return shas["approval"]
        if text.endswith(D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL):
            return shas["protocol"]
        if text.endswith(D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL):
            return reaudit_sha
        if text.endswith(D0_FULL_CORPUS_RUN_CONTRACT_V15_REL):
            return shas["run_contract"]
        if text.endswith(D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL):
            return shas["audit"]
        if text.endswith(D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL):
            return shas["carry"]
        if text.endswith(D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL):
            return shas["smoke"]
        raise AssertionError(f"unexpected_sha_path:{path}")

    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_file", _fake_sha)
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_external_reaudit_pass_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.verify_code_lock_revision_binding",
        lambda *a, **k: [],
    )
    auth = {
        "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v15",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "launch_authorized": True,
        "approval_v15_sha256": shas["approval"],
        "execution_protocol_v15_sha256": shas["protocol"],
        "external_reaudit_sha256": "0" * 64,
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": "e" * 64,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
    }
    with pytest.raises(ValueError, match="external_reaudit_sha_mismatch"):
        validate_post_audit_execution_authorization_v15(auth, root=tmp_path)


def test_draft_approval_rejected_by_readiness(tmp_path: Path) -> None:
    appr = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    appr.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_json(
        appr,
        {
            "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
            "status": "DRAFT_REMEDIATION_NOT_SEALED",
            "decision": "DRAFT_NOT_APPROVED",
            "output_version": OUTPUT_VERSION_V15,
            "campaign_token": CAMPAIGN_TOKEN_V15,
            "allow_full_corpus_extraction": False,
        },
    )
    status = evaluate_launch_readiness_v15(root=tmp_path)
    assert status["ready"] is False
    assert status["draft_friendly"] is False
    assert any("approval_v15_draft_rejected" in e for e in status["errors"])


def test_readiness_rejects_carry_forward_audit_ok_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v15_sealed_artifacts(tmp_path, code_lock="a" * 40)
    carry_path = tmp_path / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
    carry_path.write_text(
        json.dumps({"ok": False, "verify_all_shas": True, "mutated": False, "errors": ["x"]}),
        encoding="utf-8",
    )
    reaudit_path = tmp_path / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    atomic_create_json(reaudit_path, _strict_pass_payload_v15(shas, code_lock="a" * 40))
    auth_path = tmp_path / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
    atomic_create_json(
        auth_path,
        {
            "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v15",
            "launch_authorized": True,
        },
    )

    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.assert_no_out_of_provenance_sitecustomize",
        lambda root=None: None,
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_approval_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_execution_protocol_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_external_reaudit_pass_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_post_audit_execution_authorization_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_rgb_f0_smoke_report_v2",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.assert_binary_root_absent_or_empty",
        lambda path: None,
    )

    # Overwrite approval to non-draft so draft gate doesn't short-circuit first.
    appr = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    appr.unlink()
    atomic_create_json(
        appr,
        {
            "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
            "status": "APPROVED",
            "decision": "APPROVED",
            "allow_full_corpus_extraction": True,
            "binary_root_rel": "data/features/ms_sta_v5_d0_full_v15_deadbeefdead",
        },
    )
    status = evaluate_launch_readiness_v15(root=tmp_path)
    assert status["ready"] is False
    assert any("carry_forward_audit_v15_not_ok" in e for e in status["errors"])


def test_readiness_rejects_missing_carry_forward_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    appr = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    appr.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_json(
        appr,
        {
            "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
            "status": "APPROVED",
            "decision": "APPROVED",
            "allow_full_corpus_extraction": True,
            "binary_root_rel": "data/features/ms_sta_v5_d0_full_v15_deadbeefdead",
        },
    )
    proto = tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
    atomic_create_json(proto, {"schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v15"})
    reaudit = tmp_path / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    atomic_create_json(reaudit, {"audit_verdict": "PASS"})
    auth = tmp_path / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
    atomic_create_json(auth, {"launch_authorized": True})
    smoke = tmp_path / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL
    atomic_create_json(smoke, {"schema": "ms_sta_v5_d0_full_corpus_v15_rgb_f0_smoke_report_v2", "ok": True})

    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.assert_no_out_of_provenance_sitecustomize",
        lambda root=None: None,
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_approval_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_execution_protocol_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_external_reaudit_pass_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_post_audit_execution_authorization_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_rgb_f0_smoke_report_v2",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.assert_binary_root_absent_or_empty",
        lambda path: None,
    )
    status = evaluate_launch_readiness_v15(root=tmp_path)
    assert status["ready"] is False
    assert any("carry_forward_audit_v15_missing" in e for e in status["errors"])


def test_external_reaudit_rejects_missing_carry_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v15_sealed_artifacts(tmp_path, code_lock="a" * 40)
    (tmp_path / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL).unlink()
    _patch_external_reaudit_pass_v15_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload_v15(shas, code_lock="a" * 40)
    with pytest.raises(ValueError, match="skeleton_carry_forward_audit_missing"):
        validate_external_reaudit_pass_v15(obj, root=tmp_path)


def test_smoke_report_v2_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="smoke_report_v2_bad_schema"):
        validate_rgb_f0_smoke_report_v2({"schema": "v1", "ok": True, "status": "PASS"}, root=_ROOT)


def test_positive_synthetic_readiness_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v15_sealed_artifacts(tmp_path, code_lock="a" * 40)
    # Non-draft sealed approval with extraction flag
    appr = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    appr.unlink()
    atomic_create_json(
        appr,
        {
            "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
            "status": "APPROVED",
            "decision": "APPROVED",
            "allow_full_corpus_extraction": True,
            "binary_root_rel": "data/features/ms_sta_v5_d0_full_v15_deadbeefdead",
            "code_lock_git_revision": "a" * 40,
            "executable_tree_sha256": "e" * 64,
        },
    )
    reaudit_path = tmp_path / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    atomic_create_json(reaudit_path, _strict_pass_payload_v15(shas, code_lock="a" * 40))
    auth_path = tmp_path / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
    atomic_create_json(
        auth_path,
        {
            "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v15",
            "launch_authorized": True,
            "approval_v15_sha256": shas["approval"],
            "external_reaudit_sha256": "r" * 64,
        },
    )

    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.assert_no_out_of_provenance_sitecustomize",
        lambda root=None: None,
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_approval_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_execution_protocol_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_external_reaudit_pass_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_post_audit_execution_authorization_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_rgb_f0_smoke_report_v2",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.assert_binary_root_absent_or_empty",
        lambda path: None,
    )

    status = evaluate_launch_readiness_v15(root=tmp_path)
    assert status["ready"] is True
    assert status["draft_friendly"] is False
    assert status["extraction_authorized"] is True
    assert status["errors"] == []


def test_live_v15_prelaunch_blocker_or_readiness_semantics() -> None:
    """Live tree may be mid-remediation (blocker) or still sealed+post-auth.

    Prefer PRELAUNCH_BLOCKER presence over requiring readiness==True until reseal.
    """
    blocker_path = _ROOT / D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER_REL
    if blocker_path.is_file():
        blocker = load_json(blocker_path)
        assert blocker.get("status") == "BLOCKER_CONFIRMED"
        assert blocker.get("campaign_token") == CAMPAIGN_TOKEN_V15
        assert blocker.get("output_version") == OUTPUT_VERSION_V15
        assert blocker.get("rgb_started") is False
        status = evaluate_launch_readiness_v15(root=_ROOT)
        # Tree may still look ready until reseal relocates companions; RGB must not start.
        assert status.get("draft_friendly") is False
        return

    live = _ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL
    assert live.is_file()
    obj = load_json(live)
    assert obj.get("status") == "APPROVED"
    assert obj.get("decision") == "APPROVED"
    assert obj.get("allow_full_corpus_extraction") is True
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        assert obj.get(key) is False
    reaudit = _ROOT / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    assert reaudit.is_file()
    validate_external_reaudit_pass_v15(load_json(reaudit), root=_ROOT)
    auth_path = _ROOT / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
    assert auth_path.is_file()
    validate_post_audit_execution_authorization_v15(load_json(auth_path), root=_ROOT)
    status = evaluate_launch_readiness_v15(root=_ROOT)
    assert status["ready"] is True
    assert status["errors"] == []
    assert status["extraction_authorized"] is True
    assert status["draft_friendly"] is False
