"""Hermetic v15 carry-forward orchestration tests (no seal / smoke / launch / PASS)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    CAMPAIGN_TOKEN_V15,
    CORPUS_N,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
    EXTRACT_PY_REL,
    F0_PY_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION_V15,
    atomic_create_json,
)


def test_executable_paths_v15_include_carry_forward_orchestration_tests() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_carry_forward_orchestration.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def _seed_venvs(root: Path) -> None:
    for rel in (EXTRACT_PY_REL, F0_PY_REL):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# hermetic stub\n", encoding="utf-8")


def _approval_v15(*, binary_rel: str, carry_forward: bool = True) -> dict[str, Any]:
    return {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "status": "APPROVED",
        "decision": "APPROVED",
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "binary_root_rel": binary_rel,
        "skeleton_source_root_rel": "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
        "carry_forward_required": carry_forward,
        "eligible_n": CORPUS_N,
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": "b" * 64,
        "allow_full_corpus_extraction": True,
    }


def _patch_gates(monkeypatch: pytest.MonkeyPatch, mod: Any, *, approval: dict, runtime: Path, binary: Path) -> None:
    protocol = {"schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v15"}
    monkeypatch.setattr(
        mod,
        "_require_launch_gates",
        lambda **kwargs: (approval, protocol, binary, runtime),
    )
    monkeypatch.setattr(mod, "_load_mutable", lambda rt: {"status": "RUNNING", "resumes_used": 0})
    monkeypatch.setattr(mod, "_update_mutable", lambda rt, obj: None)


def test_v15_orchestrate_skips_skeleton_spawns_rgb_then_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    _seed_venvs(tmp_path)
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhash0001"
    binary = tmp_path / binary_rel
    runtime = binary / "runtime"
    runtime.mkdir(parents=True)
    approval = _approval_v15(binary_rel=binary_rel, carry_forward=True)
    audit_path = tmp_path / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
    audit_sha = atomic_create_json(
        audit_path,
        {
            "ok": True,
            "verify_all_shas": True,
            "mutated": False,
            "summary_n_ok": CORPUS_N,
            "source_root_rel": approval["skeleton_source_root_rel"],
        },
    )
    approval["skeleton_carry_forward_audit_sha256"] = audit_sha
    _patch_gates(monkeypatch, mod, approval=approval, runtime=runtime, binary=binary)

    calls: list[tuple[str, list[str]]] = []

    def _fake_worker(cmd: list[str], *, runtime, phase, root, campaign_token=CAMPAIGN_TOKEN) -> int:
        calls.append((phase, list(cmd)))
        return 0

    monkeypatch.setattr(mod, "_run_phase_worker", _fake_worker)
    monkeypatch.setattr(
        mod,
        "verify_skeleton_carry_forward_from_v14",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should_use_sealed_audit")),
    )

    rc = mod.orchestrate(
        resume=False,
        device="cpu",
        root=tmp_path,
        campaign_token=CAMPAIGN_TOKEN_V15,
    )
    assert rc == 0
    assert [p for p, _ in calls] == ["rgb_token_v3", "quality_scan"]
    assert all(CAMPAIGN_TOKEN_V15 in cmd for _, cmd in calls)
    assert all(str(tmp_path / EXTRACT_PY_REL) not in cmd for _, cmd in calls)
    assert all("--phase" in cmd and "skeleton_v3" not in cmd for _, cmd in calls)
    summary = runtime / "CARRY_FORWARD_SUMMARY.json"
    assert summary.is_file()
    body = summary.read_text(encoding="utf-8")
    assert "read_only_carry_forward" in body
    assert str(CORPUS_N) in body


def test_v15_carry_forward_mismatch_blocks_before_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    _seed_venvs(tmp_path)
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhash0002"
    binary = tmp_path / binary_rel
    runtime = binary / "runtime"
    runtime.mkdir(parents=True)
    approval = _approval_v15(binary_rel=binary_rel, carry_forward=True)
    _patch_gates(monkeypatch, mod, approval=approval, runtime=runtime, binary=binary)

    calls: list[str] = []

    def _fake_worker(cmd: list[str], *, runtime, phase, root, campaign_token=CAMPAIGN_TOKEN) -> int:
        calls.append(phase)
        return 0

    monkeypatch.setattr(mod, "_run_phase_worker", _fake_worker)
    monkeypatch.setattr(
        mod,
        "verify_skeleton_carry_forward_from_v14",
        lambda **kwargs: {"ok": False, "errors": ["forced_mismatch"]},
    )

    rc = mod.orchestrate(
        resume=False,
        device="cpu",
        root=tmp_path,
        campaign_token=CAMPAIGN_TOKEN_V15,
    )
    assert rc == 2
    assert calls == []
    assert not (runtime / "CARRY_FORWARD_SUMMARY.json").is_file()


def test_v14_orchestrate_still_spawns_skeleton_then_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    _seed_venvs(tmp_path)
    binary_rel = "data/features/ms_sta_v5_d0_full_v14_testhash0003"
    binary = tmp_path / binary_rel
    runtime = binary / "runtime"
    runtime.mkdir(parents=True)
    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v14",
        "carry_forward_required": False,
        "binary_root_rel": binary_rel,
        "code_lock_git_revision": "c" * 40,
        "executable_tree_sha256": "d" * 64,
    }
    _patch_gates(monkeypatch, mod, approval=approval, runtime=runtime, binary=binary)

    calls: list[str] = []

    def _fake_worker(cmd: list[str], *, runtime, phase, root, campaign_token=CAMPAIGN_TOKEN) -> int:
        calls.append(phase)
        assert campaign_token == CAMPAIGN_TOKEN
        return 0

    monkeypatch.setattr(mod, "_run_phase_worker", _fake_worker)
    rc = mod.orchestrate(resume=False, device="cpu", root=tmp_path, campaign_token=CAMPAIGN_TOKEN)
    assert rc == 0
    assert calls == ["skeleton_v3", "rgb_token_v3", "quality_scan"]


def test_monitor_v15_carry_forward_counts(tmp_path: Path) -> None:
    mon = importlib.import_module("monitor_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_monitor0001"
    approval_path = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    carry_path = tmp_path / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
    carry_sha = atomic_create_json(
        carry_path,
        {
            "ok": True,
            "verify_all_shas": True,
            "mutated": False,
            "summary_n_ok": CORPUS_N,
            "index_n_records": CORPUS_N,
        },
    )
    atomic_create_json(
        approval_path,
        {
            "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
            "status": "APPROVED",
            "decision": "APPROVED",
            "output_version": OUTPUT_VERSION_V15,
            "campaign_token": CAMPAIGN_TOKEN_V15,
            "binary_root_rel": binary_rel,
            "carry_forward_required": True,
            "eligible_n": CORPUS_N,
            "skeleton_carry_forward_audit_sha256": carry_sha,
        },
    )
    runtime = tmp_path / binary_rel / "runtime"
    runtime.mkdir(parents=True)

    payload = mon.build_monitor_payload(
        root=tmp_path,
        output_version=OUTPUT_VERSION_V15,
        campaign_token=CAMPAIGN_TOKEN_V15,
    )
    assert payload["ok"] is True
    assert payload["committed_counts"]["skeleton_v3"] == CORPUS_N
    assert payload["committed_counts"]["rgb_token_v3"] == 0
    assert payload["remaining_aggregate"] == CORPUS_N
    assert payload["execution_aggregate_target"] == CORPUS_N
    assert payload["skeleton_source"] == "read_only_carry_forward"
    assert payload["remaining_by_phase"]["skeleton_v3"] == 0
    assert payload["remaining_by_phase"]["rgb_token_v3"] == CORPUS_N
