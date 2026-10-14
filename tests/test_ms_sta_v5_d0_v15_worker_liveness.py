"""Integration tests for v15 RGB worker liveness / registry / startup-stall remediation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    CAMPAIGN_TOKEN_V15,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    STARTUP_STALL_SEC,
    campaign_related_process_alive,
    is_campaign_runner_cmdline,
    load_json,
    load_worker_registry_document,
    register_worker,
    save_worker_registry,
    unregister_worker,
)


def test_executable_paths_include_worker_liveness_tests() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_worker_liveness.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def test_observer_cmdline_with_campaign_token_is_not_runner() -> None:
    blob = (
        "pwsh -Command monitor_ms_sta_v5_d0_full_corpus.py --campaign-token "
        f"{CAMPAIGN_TOKEN_V15} MATERIAL_ORCHESTRATOR"
    )
    assert is_campaign_runner_cmdline(blob, CAMPAIGN_TOKEN_V15) is False


def test_scan_ignores_observer_containing_campaign_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.scan_campaign_processes",
        lambda *, campaign_token=None: [],
    )
    # Even if a monitor process exists in the table, helper must reject it.
    assert (
        is_campaign_runner_cmdline(
            f"python monitor_ms_sta_v5_d0_full_corpus.py --campaign-token {CAMPAIGN_TOKEN_V15}",
            CAMPAIGN_TOKEN_V15,
        )
        is False
    )
    assert campaign_related_process_alive(0, campaign_token=CAMPAIGN_TOKEN_V15) is False


def test_dead_or_reused_orchestrator_pid_reports_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib._process_identity", lambda pid: None)
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.scan_campaign_processes",
        lambda *, campaign_token=None: [],
    )
    assert (
        campaign_related_process_alive(
            999_999_991,
            campaign_token=CAMPAIGN_TOKEN_V15,
            runtime=runtime,
            orchestrator_create_time=123.0,
        )
        is False
    )

    # PID reuse: same pid, different create_time.
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib._process_identity",
        lambda pid: {
            "pid": pid,
            "create_time": 999.0,
            "cmdline": f"python run_ms_sta_v5_d0_full_corpus.py --campaign-token {CAMPAIGN_TOKEN_V15}",
            "name": "python",
        },
    )
    assert (
        campaign_related_process_alive(
            42,
            campaign_token=CAMPAIGN_TOKEN_V15,
            runtime=runtime,
            orchestrator_create_time=123.0,
        )
        is False
    )


def test_v15_registry_remains_v15_after_unregister(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runner = tmp_path / "run_ms_sta_v5_d0_full_corpus.py"
    runner.write_text("import time\ntime.sleep(1.5)\n", encoding="utf-8")
    child = subprocess.Popen(
        [
            sys.executable,
            str(runner),
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
            "--phase",
            "rgb_token_v3",
            "--bound-worker",
        ]
    )
    try:
        register_worker(
            runtime, pid=child.pid, phase="rgb_token_v3", campaign_token=CAMPAIGN_TOKEN_V15
        )
        doc = load_worker_registry_document(runtime)
        assert doc is not None
        assert doc["campaign_token"] == CAMPAIGN_TOKEN_V15
        unregister_worker(runtime, child.pid, campaign_token=CAMPAIGN_TOKEN_V15)
        doc2 = load_worker_registry_document(runtime)
        assert doc2 is not None
        assert doc2["campaign_token"] == CAMPAIGN_TOKEN_V15
        assert doc2["workers"] == []
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_registry_rejects_campaign_mismatch(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    save_worker_registry(runtime, [], campaign_token=CAMPAIGN_TOKEN_V15)
    with pytest.raises(ValueError, match="worker_registry_campaign_mismatch"):
        save_worker_registry(
            runtime,
            [{"pid": 1, "create_time": 1.0, "campaign_token": CAMPAIGN_TOKEN, "phase": "x"}],
            campaign_token=CAMPAIGN_TOKEN_V15,
        )


def test_rgb_worker_nonzero_exit_writes_phase_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_ms_sta_v5_d0_full_corpus as runner

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    bad = tmp_path / "run_ms_sta_v5_d0_full_corpus.py"
    bad.write_text("import sys\nsys.exit(7)\n", encoding="utf-8")

    # Bypass register identity check by monkeypatching register/unregister to no-op wrappers
    # that still use real registry persistence for campaign token.
    def _reg(runtime, *, pid, phase, campaign_token=CAMPAIGN_TOKEN):
        save_worker_registry(
            runtime,
            [
                {
                    "pid": pid,
                    "create_time": time.time(),
                    "phase": phase,
                    "campaign_token": campaign_token,
                    "cmdline": f"{bad} --campaign-token {campaign_token}",
                }
            ],
            campaign_token=campaign_token,
        )
        return {}

    monkeypatch.setattr(runner, "register_worker", _reg)
    monkeypatch.setattr(
        runner,
        "unregister_worker",
        lambda runtime, pid, *, campaign_token=None: save_worker_registry(
            runtime, [], campaign_token=campaign_token or CAMPAIGN_TOKEN_V15
        ),
    )
    code = runner._run_phase_worker(
        [sys.executable, str(bad), "--campaign-token", CAMPAIGN_TOKEN_V15],
        runtime=runtime,
        phase="rgb_token_v3",
        root=tmp_path,
        campaign_token=CAMPAIGN_TOKEN_V15,
    )
    assert code == 7
    fail_path = runtime / "PHASE_FAILURE.json"
    assert fail_path.is_file()
    fail = load_json(fail_path)
    assert fail["phase"] == "rgb_token_v3"
    assert fail["exit_code"] == 7
    assert fail["campaign_token"] == CAMPAIGN_TOKEN_V15
    assert fail["stdout_sha256"]
    assert fail["stderr_sha256"]
    assert (runtime / fail["stdout_rel"]).is_file()
    assert (runtime / fail["stderr_rel"]).is_file()


def test_startup_stall_becomes_material_when_no_progress(tmp_path: Path) -> None:
    import monitor_ms_sta_v5_d0_full_corpus as mon

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    now = time.time()
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - STARTUP_STALL_SEC - 10))
    metrics = mon._progress_metrics(
        runtime,
        current_counts={"skeleton_v3": 24752, "rgb_token_v3": 0},
        status="RUNNING",
        now_ts=now,
        aggregate_target_override=24752,
        phase="rgb_token_v3",
        workers_alive=0,
        run_updated_at=old,
        contract_created_at=old,
    )
    assert metrics["startup_stall_alert"] is True
    assert metrics["stale_progress_alert"] is True
    assert metrics["last_progress_at"] is None


def test_watchdog_script_fail_closed_on_phase_failure_and_stall() -> None:
    text = (_ROOT / "scripts/watch_ms_sta_v5_d0_full_corpus_v15.ps1").read_text(encoding="utf-8")
    assert "PHASE_FAILURE.json" in text
    assert "startup_stall" in text
    assert "phase_failure_no_restart" in text


def test_no_second_worker_helpers_present() -> None:
    text = (_ROOT / "scripts/watch_ms_sta_v5_d0_full_corpus_v15.ps1").read_text(encoding="utf-8")
    assert "Assert-NoSecondWorker" in text
    assert "live_process_tree_blocks_second_worker" in text
