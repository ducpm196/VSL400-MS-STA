"""Hermetic v15 pre-launch remediation tests (no seal / smoke / launch / PASS write)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL,
    D0_FULL_CORPUS_APPROVAL_V14_REL,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_V15_REL,
    D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL,
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
    EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION,
    EXTERNAL_REAUDIT_PASS_SCHEMA_V15,
    OUTPUT_VERSION,
    OUTPUT_VERSION_V15,
    assert_binary_root_absent_or_empty,
    atomic_create_json,
    load_json,
    validate_external_reaudit_pass_v15,
    validate_post_audit_execution_authorization_v15,
)
from ms_sta_v5_d0_lib import load_json as d0_load_json  # noqa: E402


def test_hermetic_default_output_version_remains_v14() -> None:
    assert OUTPUT_VERSION == "v14"
    assert CAMPAIGN_TOKEN == "D0_FULL_CORPUS_V14"


def test_load_json_bom_and_non_bom_same_object(tmp_path: Path) -> None:
    obj = {"ok": True, "n": 1, "nested": {"a": "b"}}
    plain = tmp_path / "plain.json"
    bom = tmp_path / "bom.json"
    body = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    plain.write_bytes(body)
    bom.write_bytes(b"\xef\xbb\xbf" + body)
    assert d0_load_json(plain) == obj
    assert d0_load_json(bom) == obj
    assert load_json(plain) == load_json(bom) == obj


def test_load_json_rejects_malformed_and_bom_only(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xef\xbb\xbf{not-json")
    with pytest.raises(json.JSONDecodeError):
        d0_load_json(bad)
    bom_only = tmp_path / "bom_only.json"
    bom_only.write_bytes(b"\xef\xbb\xbf")
    with pytest.raises(json.JSONDecodeError):
        d0_load_json(bom_only)


def test_assert_binary_root_absent_or_empty(tmp_path: Path) -> None:
    assert_binary_root_absent_or_empty(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "nested").mkdir()
    assert_binary_root_absent_or_empty(empty)
    filled = tmp_path / "filled"
    (filled / "nested").mkdir(parents=True)
    (filled / "nested" / "x.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="binary_root_not_empty"):
        assert_binary_root_absent_or_empty(filled)


def _write_approval(path: Path, *, version: str, token: str, binary_root_rel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_create_json(
        path,
        {
            "schema": f"ms_sta_v5_d0_full_corpus_approval_{version}",
            "status": "APPROVED",
            "decision": "APPROVED",
            "output_version": version,
            "campaign_token": token,
            "binary_root_rel": binary_root_rel,
        },
    )


def test_monitor_default_and_v14_use_v14_approval_root(tmp_path: Path) -> None:
    mon = importlib.import_module("monitor_ms_sta_v5_d0_full_corpus")
    v14_rel = "data/features/ms_sta_v5_d0_full_v14_aaaaaaaaaaaa"
    _write_approval(
        tmp_path / D0_FULL_CORPUS_APPROVAL_V14_REL,
        version=OUTPUT_VERSION,
        token=CAMPAIGN_TOKEN,
        binary_root_rel=v14_rel,
    )
    for kwargs in ({}, {"output_version": OUTPUT_VERSION, "campaign_token": CAMPAIGN_TOKEN}):
        payload = mon.build_monitor_payload(root=tmp_path, **kwargs)
        assert payload["ok"] is True
        assert payload["output_version"] == OUTPUT_VERSION
        assert payload["campaign_token"] == CAMPAIGN_TOKEN
        assert payload["binary_root_rel"] == v14_rel
        assert "ms_sta_v5_d0_full_v14_" in payload["binary_root_rel"]


def test_monitor_v15_uses_approval_owned_root_never_v14(tmp_path: Path) -> None:
    mon = importlib.import_module("monitor_ms_sta_v5_d0_full_corpus")
    v15_rel = "data/features/ms_sta_v5_d0_full_v15_bbbbbbbbbbbb"
    _write_approval(
        tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL,
        version=OUTPUT_VERSION_V15,
        token=CAMPAIGN_TOKEN_V15,
        binary_root_rel=v15_rel,
    )
    # Also plant a v14 approval so default identity would differ if wrongly chosen.
    _write_approval(
        tmp_path / D0_FULL_CORPUS_APPROVAL_V14_REL,
        version=OUTPUT_VERSION,
        token=CAMPAIGN_TOKEN,
        binary_root_rel="data/features/ms_sta_v5_d0_full_v14_cccccccccccc",
    )
    payload = mon.build_monitor_payload(
        root=tmp_path,
        output_version=OUTPUT_VERSION_V15,
        campaign_token=CAMPAIGN_TOKEN_V15,
    )
    assert payload["ok"] is True
    assert payload["output_version"] == OUTPUT_VERSION_V15
    assert payload["campaign_token"] == CAMPAIGN_TOKEN_V15
    assert payload["binary_root_rel"] == v15_rel
    assert "ms_sta_v5_d0_full_v14_" not in payload["binary_root_rel"]
    assert "ms_sta_v5_d0_full_v14_" not in payload["binary_root"]

    # Force refuse path: rewrite approval to point at v14 binary.
    appr = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    appr.unlink()
    _write_approval(
        appr,
        version=OUTPUT_VERSION_V15,
        token=CAMPAIGN_TOKEN_V15,
        binary_root_rel="data/features/ms_sta_v5_d0_full_v14_dddddddddddd",
    )
    refused = mon.build_monitor_payload(
        root=tmp_path,
        output_version=OUTPUT_VERSION_V15,
        campaign_token=CAMPAIGN_TOKEN_V15,
    )
    assert refused["ok"] is False
    assert refused["error"] == "monitor_v15_refuses_v14_binary_root"


def test_monitor_wrong_or_missing_campaign_rejected(tmp_path: Path) -> None:
    mon = importlib.import_module("monitor_ms_sta_v5_d0_full_corpus")
    _write_approval(
        tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL,
        version=OUTPUT_VERSION_V15,
        token=CAMPAIGN_TOKEN_V15,
        binary_root_rel="data/features/ms_sta_v5_d0_full_v15_eeeeeeeeeeee",
    )
    incomplete = mon.build_monitor_payload(
        root=tmp_path, output_version=OUTPUT_VERSION_V15, campaign_token=None
    )
    assert incomplete["ok"] is False
    assert "monitor_identity_incomplete" in str(incomplete["error"])

    wrong = mon.build_monitor_payload(
        root=tmp_path,
        output_version=OUTPUT_VERSION_V15,
        campaign_token="D0_FULL_CORPUS_V14",
    )
    assert wrong["ok"] is False
    assert "monitor_v15_campaign_token_required" in str(wrong["error"])

    unknown = mon.build_monitor_payload(
        root=tmp_path,
        output_version="v99",
        campaign_token="D0_FULL_CORPUS_V99",
    )
    assert unknown["ok"] is False
    assert "monitor_unknown_identity" in str(unknown["error"])


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


def test_stale_external_and_post_auth_fail_after_approval_sha_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old external/post-auth bound to prior approval SHA become invalid after tree rewrite."""
    shas = _write_v15_sealed_artifacts(tmp_path, code_lock="a" * 40)
    reaudit = _strict_pass_payload_v15(shas, code_lock="a" * 40)
    reaudit_path = tmp_path / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    reaudit_sha = atomic_create_json(reaudit_path, reaudit)

    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.git_commit_exists", lambda rev, root=None: True
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.sha256_git_blob_at_revision",
        lambda rel, rev, root=None: {
            D0_FULL_CORPUS_APPROVAL_V15_REL: shas["approval"],
            D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL: shas["protocol"],
            D0_FULL_CORPUS_RUN_CONTRACT_V15_REL: shas["run_contract"],
            D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL: shas["audit"],
            D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL: shas["carry"],
            D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL: shas["smoke"],
        }.get(rel, "0" * 64),
    )

    # Rewrite approval → new disk SHA; prior external PASS is stale.
    appr = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    appr.unlink()
    new_approval_sha = atomic_create_json(
        appr,
        {
            "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
            "status": "APPROVED",
            "decision": "APPROVED",
            "code_lock_git_revision": "a" * 40,
            "executable_tree_sha256": "e" * 64,
            "notes": ["tree_changed_for_stale_test"],
        },
    )
    assert new_approval_sha != shas["approval"]
    with pytest.raises(ValueError, match="approval_v15_sha256_mismatch"):
        validate_external_reaudit_pass_v15(load_json(reaudit_path), root=tmp_path)

    auth = {
        "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v15",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "launch_authorized": True,
        "approval_v15_sha256": shas["approval"],  # stale vs new disk approval
        "execution_protocol_v15_sha256": shas["protocol"],
        "external_reaudit_sha256": reaudit_sha,
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": "e" * 64,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
    }
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_external_reaudit_pass_v15",
        lambda obj, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.verify_code_lock_revision_binding",
        lambda *a, **k: [],
    )
    with pytest.raises(ValueError, match="approval_v15_sha_mismatch"):
        validate_post_audit_execution_authorization_v15(auth, root=tmp_path)


def test_monitor_cli_v15_args_exit_0_optional() -> None:
    """Optional live CLI check: v15 identity args accepted when approval exists."""
    appr = _ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL
    if not appr.is_file():
        pytest.skip("approval_v15_missing")
    proc = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts" / "monitor_ms_sta_v5_d0_full_corpus.py"),
            "--output-version",
            OUTPUT_VERSION_V15,
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
            "--json",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload.get("ok") is True
    assert payload.get("output_version") == OUTPUT_VERSION_V15
    assert payload.get("campaign_token") == CAMPAIGN_TOKEN_V15
    assert "ms_sta_v5_d0_full_v14_" not in str(payload.get("binary_root_rel") or "")
