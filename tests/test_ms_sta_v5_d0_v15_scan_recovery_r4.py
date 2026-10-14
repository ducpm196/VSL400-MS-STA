"""Adversarial R4 scan-recovery execution-provenance remediation tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
    D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION_V15,
    SCAN_RECOVERY_EXTERNAL_REAUDIT_SCHEMA_V15,
    SCAN_RECOVERY_POST_AUDIT_AUTH_SCHEMA_V15,
    executable_tree_sha256_at_revision,
    preflight_scan_recovery_execute,
    validate_scan_recovery_attempt_auth_binds,
)
import ms_sta_v5_d0_full_corpus_lib as lib  # noqa: E402


def test_executable_paths_include_r4() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_scan_recovery_r4.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _blob_sha(cwd: Path, rev: str, rel: str) -> str:
    blob = subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=cwd)
    return hashlib.sha256(blob).hexdigest()


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _plant_provenance_repo(tmp_path: Path) -> tuple[str, dict, list[str]]:
    """Minimal git repo with sealed auth artifacts + one executable path."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "r4@test")
    _git(tmp_path, "config", "user.name", "r4")
    exe_rel = "scripts/r4_stub.py"
    (tmp_path / exe_rel).parent.mkdir(parents=True)
    (tmp_path / exe_rel).write_text("# r4 stub\n", encoding="utf-8")
    paths = [exe_rel]

    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "decision": "APPROVED",
        "status": "APPROVED",
        "code_lock_git_revision": "0" * 40,
        "executable_tree_sha256": "0" * 64,
    }
    protocol = {"schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v15", "x": 1}
    run_contract = {"schema": "ms_sta_v5_d0_full_corpus_run_contract_v15", "x": 1}
    reaudit = {
        "schema": SCAN_RECOVERY_EXTERNAL_REAUDIT_SCHEMA_V15,
        "audit_verdict": "PASS",
        "blocking_findings": [],
    }
    post_auth = {
        "schema": SCAN_RECOVERY_POST_AUDIT_AUTH_SCHEMA_V15,
        "scan_recovery_execute_authorized": True,
    }
    _write(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL, approval)
    _write(tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL, protocol)
    _write(tmp_path / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL, run_contract)
    _write(tmp_path / D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL, reaudit)
    _write(tmp_path / D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL, post_auth)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "seed")
    code_lock = _git(tmp_path, "rev-parse", "HEAD")
    tree = executable_tree_sha256_at_revision(paths, code_lock, root=tmp_path)
    approval["code_lock_git_revision"] = code_lock
    approval["executable_tree_sha256"] = tree
    _write(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL, approval)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "bind-tree")
    head = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "branch", "-M", "main")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", head)

    binds = {
        "execution_head": head,
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": tree,
        "approval_v15_sha256": _blob_sha(tmp_path, head, D0_FULL_CORPUS_APPROVAL_V15_REL),
        "execution_protocol_v15_sha256": _blob_sha(
            tmp_path, head, D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
        ),
        "run_contract_v15_sha256": _blob_sha(tmp_path, head, D0_FULL_CORPUS_RUN_CONTRACT_V15_REL),
        "scan_recovery_external_reaudit_sha256": _blob_sha(
            tmp_path, head, D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL
        ),
        "scan_recovery_post_audit_auth_sha256": _blob_sha(
            tmp_path, head, D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL
        ),
    }
    return head, binds, paths


def test_dirty_worktree_execute_accepted_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """DIRTY_WORKTREE_EXECUTE_ACCEPTED must be false."""
    monkeypatch.setattr(lib, "git_porcelain", lambda root=None: " M scripts/evil.py\n")
    with pytest.raises(ValueError, match="dirty_worktree"):
        preflight_scan_recovery_execute(root=_ROOT)
    print("DIRTY_WORKTREE_EXECUTE_ACCEPTED=false")


def test_executable_drift_accepted_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EXECUTABLE_DRIFT_ACCEPTED must be false."""
    _head, _binds, paths = _plant_provenance_repo(tmp_path)
    monkeypatch.setattr(lib, "FULL_CORPUS_EXECUTABLE_PATHS_V15", paths)
    monkeypatch.setattr(lib, "git_porcelain", lambda root=None: "")
    monkeypatch.setattr(lib, "validate_approval_v15", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        lib,
        "evaluate_scan_recovery_execute_readiness_v15",
        lambda root=None: {"execution_ready": True, "errors": []},
    )
    # Disk diverges from sealed code-lock while porcelain check is bypassed.
    (tmp_path / paths[0]).write_text("# drifted executable\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executable_drift"):
        preflight_scan_recovery_execute(root=tmp_path)
    print("EXECUTABLE_DRIFT_ACCEPTED=false")


def test_forged_auth_bind_accepted_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FORGED_AUTH_BIND_ACCEPTED must be false."""
    _head, binds, paths = _plant_provenance_repo(tmp_path)
    monkeypatch.setattr(lib, "FULL_CORPUS_EXECUTABLE_PATHS_V15", paths)
    # Positive control: honest binds pass.
    assert validate_scan_recovery_attempt_auth_binds(binds, root=tmp_path)["ok"] is True
    forged = dict(binds)
    forged["approval_v15_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="forged_or_stale:approval_v15_sha256"):
        validate_scan_recovery_attempt_auth_binds(forged, root=tmp_path)
    print("FORGED_AUTH_BIND_ACCEPTED=false")


def test_nonexistent_execution_head_accepted_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NONEXISTENT_EXECUTION_HEAD_ACCEPTED must be false."""
    _head, binds, paths = _plant_provenance_repo(tmp_path)
    monkeypatch.setattr(lib, "FULL_CORPUS_EXECUTABLE_PATHS_V15", paths)
    bad = dict(binds)
    bad["execution_head"] = "a" * 40  # valid hex, not in repo
    with pytest.raises(ValueError, match="execution_head_missing"):
        validate_scan_recovery_attempt_auth_binds(bad, root=tmp_path)
    print("NONEXISTENT_EXECUTION_HEAD_ACCEPTED=false")


def test_stale_auth_blob_accepted_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STALE_AUTH_BLOB_ACCEPTED must be false."""
    head, binds, paths = _plant_provenance_repo(tmp_path)
    monkeypatch.setattr(lib, "FULL_CORPUS_EXECUTABLE_PATHS_V15", paths)
    # Capture honest approval SHA, then supersede the blob with a new commit.
    stale_approval_sha = binds["approval_v15_sha256"]
    approval_path = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    obj = json.loads(approval_path.read_text(encoding="utf-8"))
    obj["notes"] = ["superseded_for_stale_proof"]
    _write(approval_path, obj)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "supersede-approval")
    new_head = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", new_head)

    stale = dict(binds)
    stale["execution_head"] = new_head
    stale["approval_v15_sha256"] = stale_approval_sha  # old blob SHA
    # Update other fields to new head where needed except forged approval.
    for field, rel in (
        ("execution_protocol_v15_sha256", D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        ("run_contract_v15_sha256", D0_FULL_CORPUS_RUN_CONTRACT_V15_REL),
        ("scan_recovery_external_reaudit_sha256", D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL),
        ("scan_recovery_post_audit_auth_sha256", D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL),
    ):
        stale[field] = _blob_sha(tmp_path, new_head, rel)
    # code_lock/tree from new approval blob
    new_approval = json.loads(
        subprocess.check_output(
            ["git", "show", f"{new_head}:{D0_FULL_CORPUS_APPROVAL_V15_REL}"],
            cwd=tmp_path,
        ).decode("utf-8")
    )
    stale["code_lock_git_revision"] = new_approval["code_lock_git_revision"]
    stale["executable_tree_sha256"] = new_approval["executable_tree_sha256"]
    with pytest.raises(ValueError, match="forged_or_stale:approval_v15_sha256"):
        validate_scan_recovery_attempt_auth_binds(stale, root=tmp_path)
    print("STALE_AUTH_BLOB_ACCEPTED=false")


def test_preflight_blocks_before_attempt_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            }
        ),
    )
    monkeypatch.setattr(recovery, "validate_approval_v15", lambda *a, **k: None)
    monkeypatch.setattr(
        recovery,
        "evaluate_scan_recovery_execute_readiness_v15",
        lambda root=None: {"execution_ready": True, "errors": []},
    )

    def boom_preflight(*, root=None):  # noqa: ANN001
        raise ValueError("scan_recovery_preflight_dirty_worktree")

    monkeypatch.setattr(recovery, "preflight_scan_recovery_execute", boom_preflight)
    monkeypatch.setattr(
        sys,
        "argv",
        ["recovery", "--execute", "--attempt-id", "preflight_block"],
    )
    rc = recovery.main()
    assert rc == 2
    attempt = (
        tmp_path
        / "Docs/results/ms_sta_v5_d0_full/v15/scan_recovery/attempts/preflight_block"
    )
    assert not attempt.exists()
    print("DIRTY_WORKTREE_EXECUTE_ACCEPTED=false")
