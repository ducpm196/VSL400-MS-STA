"""Real PowerShell cold-start integration tests for v15 watchdog bootstrap order."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION_V15,
    assert_binary_root_absent_or_empty,
)

WATCHDOG = _ROOT / "scripts" / "watch_ms_sta_v5_d0_full_corpus_v15.ps1"
PWSH = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(PWSH is None, reason="pwsh/powershell required")


def test_executable_paths_include_watchdog_cold_start_tests() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_watchdog_cold_start.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15


def test_watchdog_script_default_restart_state_includes_last_at() -> None:
    text = WATCHDOG.read_text(encoding="utf-8")
    assert "function New-RestartState" in text
    assert "last_at" in text
    assert "restart_count = 0" in text
    assert "cold_start_launch" in text
    assert "watchdog_state_requires_run_contract" in text
    assert "cold_start_runner_failed_before_contract" in text
    assert "param([switch]$WriteTick)" in text
    assert 'if ($WriteTick) { $monitorArgs += "--write-tick" }' in text


def test_assert_binary_root_absent_or_empty_unchanged(tmp_path: Path) -> None:
    assert_binary_root_absent_or_empty(tmp_path / "missing")
    filled = tmp_path / "filled"
    (filled / "runtime").mkdir(parents=True)
    (filled / "runtime" / "WATCHDOG_STATE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="binary_root_not_empty"):
        assert_binary_root_absent_or_empty(filled)


def _approval_and_bin(tmp_path: Path) -> tuple[Path, Path, str]:
    bin_rel = "data/features/ms_sta_v5_d0_full_v15_testhash12"
    binary = tmp_path / bin_rel
    approval_path = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
        "status": "APPROVED",
        "decision": "APPROVED",
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "code_lock_git_revision": "a" * 40,
        "executable_tree_sha256": "b" * 64,
        "binary_root_rel": bin_rel,
        "carry_forward_required": True,
        "eligible_n": 24752,
    }
    approval_path.write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
    return binary, approval_path, bin_rel


def _write_monitor_script(path: Path, *, bin_rel: str, runtime: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            import argparse, json, sys
            from pathlib import Path
            ap = argparse.ArgumentParser()
            ap.add_argument("--json", action="store_true")
            ap.add_argument("--write-tick", action="store_true")
            ap.add_argument("--output-version")
            ap.add_argument("--campaign-token")
            args = ap.parse_args()
            runtime = Path(r"{runtime}")
            contract = runtime / "_run_contract.json"
            payload = {{
                "ok": True,
                "output_version": "v15",
                "campaign_token": "D0_FULL_CORPUS_V15",
                "binary_root": str(Path(r"{runtime.parent}")),
                "binary_root_rel": r"{bin_rel}".replace("\\\\", "/"),
                "runtime_dir": str(runtime),
                "run_state_exists": False,
                "run_contract_exists": contract.is_file(),
                "run_state": None,
                "phase": None,
                "disk_free_gb": 100.0,
                "worker_registry": {{"registered": 0, "alive": 0, "phases": [], "pids": []}},
                "live_orchestrator": False,
                "committed_counts": {{"skeleton_v3": 24752, "rgb_token_v3": 0}},
                "remaining_aggregate": 24752,
            }}
            if args.write_tick:
                if not contract.is_file():
                    print(json.dumps({{"ok": False, "error": "write_tick_requires_run_contract"}}))
                    sys.exit(2)
                runtime.mkdir(parents=True, exist_ok=True)
                (runtime / "MONITOR_TICK.json").write_text(json.dumps(payload), encoding="utf-8")
            print(json.dumps(payload))
            """
        ),
        encoding="utf-8",
    )


def _write_runner_success(path: Path, *, runtime: Path, approval_path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            import json, sys, time
            from pathlib import Path
            runtime = Path(r"{runtime}")
            approval = json.loads(Path(r"{approval_path}").read_text(encoding="utf-8"))
            runtime.mkdir(parents=True, exist_ok=True)
            for name in ("WATCHDOG_STATE.json", "MONITOR_TICK.json"):
                if (runtime / name).is_file():
                    raise SystemExit(f"pre_contract_artifact_present:{{name}}")
            contract = {{
                "schema": "ms_sta_v5_d0_full_corpus_immutable_run_contract_v15",
                "campaign_token": "D0_FULL_CORPUS_V15",
                "output_version": "v15",
                "code_lock_git_revision": approval["code_lock_git_revision"],
                "executable_tree_sha256": approval["executable_tree_sha256"],
                "binary_root_rel": approval["binary_root_rel"],
                "status": "BOUND",
            }}
            (runtime / "_run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            time.sleep(8)
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )


def _write_runner_fail_before_contract(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            import sys, time
            time.sleep(1)
            sys.exit(2)
            """
        ),
        encoding="utf-8",
    )


def _write_runner_resume_only(path: Path, *, runtime: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            import os, sys, time
            from pathlib import Path
            runtime = Path(r"{runtime}")
            args = sys.argv[1:]
            if "--resume" not in args:
                raise SystemExit("resume_required")
            if not (runtime / "_run_contract.json").is_file():
                raise SystemExit("missing_contract")
            lock = runtime / "WORKER_LOCK.flag"
            if lock.is_file():
                raise SystemExit("second_worker_refused")
            lock.write_text(str(os.getpid()), encoding="utf-8")
            time.sleep(6)
            lock.unlink(missing_ok=True)
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )


def _run_watchdog(
    tmp_path: Path,
    *,
    monitor: Path,
    runner: Path,
    approval_rel: str,
    once: bool = True,
) -> subprocess.CompletedProcess[str]:
    args = [
        PWSH,
        "-NoProfile",
        "-File",
        str(WATCHDOG),
        "-Launch",
        "-Device",
        "cpu",
        "-IntervalSec",
        "1",
        "-MaxRestarts",
        "3",
        "-ContractWaitSec",
        "20",
        "-ContractPollSec",
        "1",
        "-RootOverride",
        str(tmp_path),
        "-F0Override",
        sys.executable,
        "-MonitorOverride",
        str(monitor),
        "-RunnerOverride",
        str(runner),
        "-ApprovalRelOverride",
        approval_rel,
    ]
    if once:
        args.append("-Once")
    return subprocess.run(
        args,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_cold_start_no_binary_root_before_runner_then_contract_and_ticks(tmp_path: Path) -> None:
    binary, approval_path, bin_rel = _approval_and_bin(tmp_path)
    runtime = binary / "runtime"
    monitor = tmp_path / "fake_monitor.py"
    runner = tmp_path / "fake_runner.py"
    _write_monitor_script(monitor, bin_rel=bin_rel, runtime=runtime)
    _write_runner_success(runner, runtime=runtime, approval_path=approval_path)

    assert not binary.exists()
    proc = _run_watchdog(
        tmp_path,
        monitor=monitor,
        runner=runner,
        approval_rel=D0_FULL_CORPUS_APPROVAL_V15_REL,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "watchdog_v15_cold_start_restart=1" in out, out
    assert "watchdog_v15_contract_bound" in out, out
    assert (runtime / "_run_contract.json").is_file()
    assert (runtime / "WATCHDOG_STATE.json").is_file()
    assert (runtime / "MONITOR_TICK.json").is_file()
    state = json.loads((runtime / "WATCHDOG_STATE.json").read_text(encoding="utf-8"))
    assert state.get("last_at") is not None
    assert state["restart_count"] == 1
    assert proc.returncode == 0, out


def test_dead_runner_before_contract_fails_closed_no_fabricated_state(tmp_path: Path) -> None:
    binary, _approval_path, bin_rel = _approval_and_bin(tmp_path)
    runtime = binary / "runtime"
    monitor = tmp_path / "fake_monitor.py"
    runner = tmp_path / "fake_runner.py"
    _write_monitor_script(monitor, bin_rel=bin_rel, runtime=runtime)
    _write_runner_fail_before_contract(runner)

    proc = _run_watchdog(
        tmp_path,
        monitor=monitor,
        runner=runner,
        approval_rel=D0_FULL_CORPUS_APPROVAL_V15_REL,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "cold_start_runner_failed_before_contract" in out, out
    assert proc.returncode == 3, out
    assert not (runtime / "WATCHDOG_STATE.json").is_file()
    assert not (runtime / "MONITOR_TICK.json").is_file()
    assert not (runtime / "_run_contract.json").is_file()


def test_resume_attaches_same_contract_never_second_worker(tmp_path: Path) -> None:
    binary, approval_path, bin_rel = _approval_and_bin(tmp_path)
    runtime = binary / "runtime"
    runtime.mkdir(parents=True)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    contract = {
        "schema": "ms_sta_v5_d0_full_corpus_immutable_run_contract_v15",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "code_lock_git_revision": approval["code_lock_git_revision"],
        "executable_tree_sha256": approval["executable_tree_sha256"],
        "binary_root_rel": approval["binary_root_rel"],
        "status": "BOUND",
    }
    (runtime / "_run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (runtime / "WATCHDOG_STATE.json").write_text(
        json.dumps({"restart_count": 1, "last_action": "cold_start_launch", "last_at": None}),
        encoding="utf-8",
    )

    monitor = tmp_path / "fake_monitor.py"
    runner = tmp_path / "fake_runner.py"
    _write_monitor_script(monitor, bin_rel=bin_rel, runtime=runtime)
    _write_runner_resume_only(runner, runtime=runtime)

    proc = _run_watchdog(
        tmp_path,
        monitor=monitor,
        runner=runner,
        approval_rel=D0_FULL_CORPUS_APPROVAL_V15_REL,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "watchdog_v15_resume_restart=2" in out, out
    assert proc.returncode == 0, out
    assert json.loads((runtime / "_run_contract.json").read_text(encoding="utf-8")) == contract
    state = json.loads((runtime / "WATCHDOG_STATE.json").read_text(encoding="utf-8"))
    assert state["restart_count"] == 2
    assert state["last_at"] is not None


def test_monitor_write_tick_refuses_without_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    mon = importlib.import_module("monitor_ms_sta_v5_d0_full_corpus")
    bin_rel = "data/features/ms_sta_v5_d0_full_v15_cccccccccccc"
    approval_path = tmp_path / D0_FULL_CORPUS_APPROVAL_V15_REL
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    approval_path.write_text(
        json.dumps(
            {
                "schema": "ms_sta_v5_d0_full_corpus_approval_v15",
                "status": "APPROVED",
                "decision": "APPROVED",
                "output_version": OUTPUT_VERSION_V15,
                "campaign_token": CAMPAIGN_TOKEN_V15,
                "binary_root_rel": bin_rel,
                "code_lock_git_revision": "c" * 40,
                "executable_tree_sha256": "d" * 64,
                "carry_forward_required": True,
                "eligible_n": 24752,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mon, "_ROOT", tmp_path)
    payload = mon.build_monitor_payload(
        root=tmp_path, output_version=OUTPUT_VERSION_V15, campaign_token=CAMPAIGN_TOKEN_V15
    )
    assert payload["ok"] is True
    binary = tmp_path / bin_rel
    assert_binary_root_absent_or_empty(binary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor",
            "--json",
            "--write-tick",
            "--output-version",
            OUTPUT_VERSION_V15,
            "--campaign-token",
            CAMPAIGN_TOKEN_V15,
        ],
    )
    rc = mon.main()
    assert rc == 2
    assert_binary_root_absent_or_empty(binary)
