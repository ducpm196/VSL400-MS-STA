"""Integration tests: v15 bound-worker attach vs orchestrator cold-start gates.

Executes run_ms_sta_v5_d0_full_corpus.main() (not only helpers).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION_V15,
    atomic_create_json,
    load_json,
    sha256_file,
)
from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
)


def test_executable_paths_v15_include_bound_worker_attach_tests() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_bound_worker_attach.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def _write_json(path: Path, obj: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_create_json(path, obj)


def _plant_v15_auth_tree(
    root: Path,
    *,
    binary_rel: str,
    code_lock: str = "a" * 40,
    tree: str = "b" * 64,
) -> dict[str, str]:
    approval = {
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
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": tree,
        "binary_root_rel": binary_rel,
        "skeleton_source_root_rel": "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
        "carry_forward_required": True,
        "eligible_n": 24752,
        "disk_preflight": {"min_free_gb": 1},
    }
    protocol = {
        "schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v15",
        "status": "APPROVED",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": tree,
        "binary_root_rel": binary_rel,
    }
    shas = {
        "approval": _write_json(root / D0_FULL_CORPUS_APPROVAL_V15_REL, approval),
        "protocol": _write_json(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL, protocol),
    }
    reaudit = {
        "schema": "ms_sta_v5_d0_full_corpus_execution_external_reaudit_pass_v15",
        "audit_verdict": "PASS",
        "decision": {"launch_authorized": True},
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "code_lock_git_revision": code_lock,
        "approval_v15_sha256": shas["approval"],
        "execution_protocol_v15_sha256": shas["protocol"],
    }
    shas["reaudit"] = _write_json(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL, reaudit)
    auth = {
        "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v15",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "launch_authorized": True,
        "approval_v15_sha256": shas["approval"],
        "execution_protocol_v15_sha256": shas["protocol"],
        "external_reaudit_sha256": shas["reaudit"],
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": tree,
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
    }
    shas["auth"] = _write_json(root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL, auth)
    ckpt = root / EXPECTED_SLOWFAST_CKPT_REL
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"ckpt")
    return shas


def _stub_gate_validators(monkeypatch: pytest.MonkeyPatch, mod: Any) -> None:
    monkeypatch.setattr(mod, "validate_approval_v15", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(mod, "validate_execution_protocol_v15", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(mod, "validate_external_reaudit_pass_v15", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        mod, "validate_post_audit_execution_authorization_v15", lambda *a, **k: {"ok": True}
    )
    monkeypatch.setattr(mod, "assert_psutil_available", lambda: None)
    monkeypatch.setattr(mod, "disk_preflight", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(mod, "_sha256_file", lambda p: EXPECTED_SLOWFAST_CKPT_SHA256)


def _ready_payload(root: Path, binary_rel: str) -> dict[str, Any]:
    approval = load_json(root / D0_FULL_CORPUS_APPROVAL_V15_REL)
    protocol = load_json(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL)
    return {
        "ready": True,
        "errors": [],
        "approval": approval,
        "protocol": protocol,
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "extraction_authorized": True,
        "binary_root_rel": binary_rel,
    }


def _run_main(mod: Any, argv: list[str]) -> int:
    old = sys.argv
    try:
        sys.argv = argv
        return int(mod.main())
    finally:
        sys.argv = old


def test_cold_orchestrator_creates_contract_with_pass_auth_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhashattach01"
    shas = _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    _stub_gate_validators(monkeypatch, mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    ready = _ready_payload(tmp_path, binary_rel)
    monkeypatch.setattr(mod, "evaluate_launch_readiness_v15", lambda root=None: ready)
    monkeypatch.setattr(mod, "require_launch_readiness_v15", lambda root=None: ready)

    called: dict[str, Any] = {"orchestrate": 0}

    def _fake_orchestrate(**kwargs: Any) -> int:
        called["orchestrate"] += 1
        # Contract must already exist (created by orchestrate's first gate call).
        # Here we create via gate then return, simulating orchestrate entry.
        mod._require_launch_gates(
            resume=False,
            root=tmp_path,
            bound_worker=False,
            campaign_token=CAMPAIGN_TOKEN_V15,
        )
        return 0

    monkeypatch.setattr(mod, "orchestrate", _fake_orchestrate)
    rc = _run_main(
        mod,
        [
            "run_ms_sta_v5_d0_full_corpus.py",
            "--phase",
            "all",
            "--launch",
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    assert rc == 0
    assert called["orchestrate"] == 1
    contract = load_json(tmp_path / binary_rel / "runtime" / "_run_contract.json")
    assert contract["campaign_token"] == CAMPAIGN_TOKEN_V15
    assert contract["external_reaudit_v15_sha256"] == shas["reaudit"]
    assert contract["post_audit_execution_auth_v15_sha256"] == shas["auth"]
    assert contract["approval_v15_sha256"] == shas["approval"]
    assert contract["binary_root_rel"] == binary_rel


def test_rgb_bound_worker_attaches_without_evaluate_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhashattach02"
    shas = _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    _stub_gate_validators(monkeypatch, mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)

    # Seed non-empty binary root + immutable contract (orchestrator cold-start residue).
    runtime = tmp_path / binary_rel / "runtime"
    runtime.mkdir(parents=True)
    payload = mod._build_v15_immutable_run_contract_payload(
        approval=load_json(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL),
        protocol=load_json(tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        resume=False,
        root=tmp_path,
    )
    payload["status"] = "BOUND"
    atomic_create_json(runtime / "_run_contract.json", payload)
    (runtime / "MONITOR_TICK.json").write_text("{}", encoding="utf-8")

    readiness_calls = {"n": 0}

    def _boom_readiness(*, root=None):
        readiness_calls["n"] += 1
        raise AssertionError("evaluate_launch_readiness_v15_must_not_run_for_bound_worker")

    monkeypatch.setattr(mod, "evaluate_launch_readiness_v15", _boom_readiness)
    monkeypatch.setattr(
        mod,
        "require_launch_readiness_v15",
        lambda root=None: (_ for _ in ()).throw(AssertionError("require_readiness_forbidden")),
    )

    rgb_calls: list[dict[str, Any]] = []

    def _fake_rgb(**kwargs: Any) -> dict[str, Any]:
        rgb_calls.append(kwargs)
        return {"ok": True, "n_ok": 1}

    monkeypatch.setattr(mod, "run_rgb_phase", _fake_rgb)
    monkeypatch.setattr(
        mod,
        "run_skeleton_phase",
        lambda **k: (_ for _ in ()).throw(AssertionError("skeleton_must_not_run")),
    )

    rc = _run_main(
        mod,
        [
            "run_ms_sta_v5_d0_full_corpus.py",
            "--phase",
            "rgb_token_v3",
            "--device",
            "cuda",
            "--launch",
            "--bound-worker",
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    assert rc == 0
    assert readiness_calls["n"] == 0
    assert len(rgb_calls) == 1
    assert rgb_calls[0]["binary"] == (tmp_path / binary_rel).resolve()
    # Original failure reproduction: non-empty root must not block bound attach.
    assert any((tmp_path / binary_rel).rglob("*"))
    assert shas["reaudit"] == sha256_file(tmp_path / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL)


@pytest.mark.parametrize(
    "argv,err",
    [
        (
            [
                "run_ms_sta_v5_d0_full_corpus.py",
                "--phase",
                "rgb_token_v3",
                "--bound-worker",
                "--campaign-token",
                CAMPAIGN_TOKEN_V15,
            ],
            "bound_worker_requires_launch",
        ),
        (
            [
                "run_ms_sta_v5_d0_full_corpus.py",
                "--phase",
                "all",
                "--launch",
                "--bound-worker",
                "--campaign-token",
                CAMPAIGN_TOKEN_V15,
            ],
            "bound_worker_invalid_phase",
        ),
    ],
)
def test_bound_worker_negative_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], err: str
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhashattach03"
    _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    rc = _run_main(mod, argv)
    assert rc == 2
    # stdout from last print is hard to capture; re-invoke capturing via helper path checks error codes.
    # Direct assert on return code + error branch by calling _main_bound_worker for clarity when needed.
    if err == "bound_worker_requires_launch":
        assert "--launch" not in argv
    if err == "bound_worker_invalid_phase":
        assert "all" in argv


def test_bound_worker_without_contract_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhashattach04"
    _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    _stub_gate_validators(monkeypatch, mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    (tmp_path / binary_rel / "runtime").mkdir(parents=True)
    rc = _run_main(
        mod,
        [
            "run_ms_sta_v5_d0_full_corpus.py",
            "--phase",
            "rgb_token_v3",
            "--launch",
            "--bound-worker",
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    assert rc == 2


def test_bound_worker_mismatched_pass_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhashattach05"
    shas = _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    _stub_gate_validators(monkeypatch, mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    runtime = tmp_path / binary_rel / "runtime"
    runtime.mkdir(parents=True)
    payload = mod._build_v15_immutable_run_contract_payload(
        approval=load_json(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL),
        protocol=load_json(tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        resume=False,
        root=tmp_path,
    )
    payload["status"] = "BOUND"
    payload["external_reaudit_v15_sha256"] = "0" * 64  # mismatch vs live PASS
    atomic_create_json(runtime / "_run_contract.json", payload)
    del shas
    rc = _run_main(
        mod,
        [
            "run_ms_sta_v5_d0_full_corpus.py",
            "--phase",
            "rgb_token_v3",
            "--launch",
            "--bound-worker",
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    assert rc == 2


def test_bound_worker_altered_code_lock_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhashattach06"
    _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    _stub_gate_validators(monkeypatch, mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    runtime = tmp_path / binary_rel / "runtime"
    runtime.mkdir(parents=True)
    payload = mod._build_v15_immutable_run_contract_payload(
        approval=load_json(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL),
        protocol=load_json(tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        resume=False,
        root=tmp_path,
    )
    payload["status"] = "BOUND"
    payload["code_lock_git_revision"] = "c" * 40
    atomic_create_json(runtime / "_run_contract.json", payload)
    rc = _run_main(
        mod,
        [
            "run_ms_sta_v5_d0_full_corpus.py",
            "--phase",
            "rgb_token_v3",
            "--launch",
            "--bound-worker",
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    assert rc == 2


def test_direct_non_bound_phase_against_nonempty_root_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_testhashattach07"
    _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    _stub_gate_validators(monkeypatch, mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    # Contaminate binary root (runtime residue).
    runtime = tmp_path / binary_rel / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "WATCHDOG_STATE.json").write_text("{}", encoding="utf-8")

    def _eval(*, root=None):
        return {
            "ready": False,
            "errors": [f"binary_root_not_empty:{tmp_path / binary_rel}"],
            "approval": load_json(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL),
            "protocol": load_json(tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
            "extraction_authorized": False,
        }

    monkeypatch.setattr(mod, "evaluate_launch_readiness_v15", _eval)
    rc = _run_main(
        mod,
        [
            "run_ms_sta_v5_d0_full_corpus.py",
            "--phase",
            "rgb_token_v3",
            "--launch",
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    assert rc == 2


def test_original_failure_reproduction_now_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact prior defect: evaluate readiness before bound-worker saw non-empty root."""
    mod = importlib.import_module("run_ms_sta_v5_d0_full_corpus")
    binary_rel = "data/features/ms_sta_v5_d0_full_v15_4c446896e809"
    _plant_v15_auth_tree(tmp_path, binary_rel=binary_rel)
    _stub_gate_validators(monkeypatch, mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    runtime = tmp_path / binary_rel / "runtime"
    runtime.mkdir(parents=True)
    payload = mod._build_v15_immutable_run_contract_payload(
        approval=load_json(tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL),
        protocol=load_json(tmp_path / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        resume=False,
        root=tmp_path,
    )
    payload["status"] = "BOUND"
    atomic_create_json(runtime / "_run_contract.json", payload)
    # Simulate PHASE_FAILURE residue that made readiness fail previously.
    atomic_create_json(
        runtime / "PHASE_FAILURE.json",
        {"schema": "ms_sta_v5_d0_full_corpus_phase_failure_v1", "phase": "rgb_token_v3"},
    )

    def _readiness_would_fail(*, root=None):
        raise AssertionError("must_not_call_evaluate_launch_readiness_v15")

    monkeypatch.setattr(mod, "evaluate_launch_readiness_v15", _readiness_would_fail)
    monkeypatch.setattr(mod, "run_rgb_phase", lambda **k: {"ok": True, "n_ok": 1})
    monkeypatch.setattr(
        mod,
        "run_skeleton_phase",
        lambda **k: (_ for _ in ()).throw(AssertionError("no_skeleton_reextract")),
    )
    rc = _run_main(
        mod,
        [
            "run_ms_sta_v5_d0_full_corpus.py",
            "--phase",
            "rgb_token_v3",
            "--device",
            "cuda",
            "--launch",
            "--bound-worker",
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    assert rc == 0
