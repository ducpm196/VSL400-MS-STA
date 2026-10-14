"""One-shot monitor snapshot for MS-STA retrain-v3 (Path B)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import load_policy, require_frozen_v3_policy, sha256_file

DEFAULT_POLICY = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3_policy" / "RETRAIN_POLICY.json"


def _git() -> tuple[str, bool]:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True).strip()
        )
        return rev, dirty
    except Exception:
        return "", True


def _nvidia() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        util = float(parts[0]) if parts else None
        return {
            "util_pct": util,
            "mem_used_mb": float(parts[1]) if len(parts) > 1 else None,
            "mem_total_mb": float(parts[2]) if len(parts) > 2 else None,
            "temp_c": float(parts[3]) if len(parts) > 3 else None,
            "raw": out,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = os.statvfs if hasattr(os, "statvfs") else None
        if usage:
            st = os.statvfs(path)
            return (st.f_bavail * st.f_frsize) / (1024**3)
        import shutil

        total, used, free = shutil.disk_usage(path)
        return free / (1024**3)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        pass
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}"], text=True, stderr=subprocess.DEVNULL
        )
        return str(pid) in out and "No tasks are running" not in out
    except Exception:
        return False


def _live_trainer_pids() -> list[int]:
    """Detect in-flight train_ms_sta_vsl400 processes (incl. orphans)."""
    out: list[int] = []
    try:
        import psutil  # type: ignore

        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = (p.info.get("name") or "").lower()
                if "python" not in name:
                    continue
                cmd = " ".join(p.info.get("cmdline") or [])
                if "train_ms_sta_vsl400" in cmd:
                    out.append(int(p.info["pid"]))
            except Exception:
                continue
        return out
    except Exception:
        return out


def _latest_metrics(status_dir: Path) -> dict:
    files = sorted(status_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    skip_names = {
        "RUN_STATE.json",
        "_run_contract.json",
        "orchestrator.pid.json",
        "monitor_latest.json",
        "EVAL_LOCK.json",
        "watchdog_restarts.json",
    }
    for p in files:
        if p.name in skip_names or p.name.startswith("RUN_STATE.") or p.name.startswith("watchdog"):
            continue
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (st.get("history") or st.get("best_val_top1") is not None or st.get("phase")):
            continue
        hist = st.get("history") or []
        last = hist[-1] if hist else {}
        return {
            "status_file": p.name,
            "phase": st.get("phase"),
            "variant": st.get("experiment_variant") or st.get("variant"),
            "config_id": st.get("config_id"),
            "seed": st.get("seed"),
            "epoch": last.get("epoch"),
            "train_loss": last.get("train_loss"),
            "train_top1": last.get("train_top1"),
            "val_top1": (last.get("val") or {}).get("top1") if isinstance(last.get("val"), dict) else st.get("best_val_top1"),
            "val_top5": (last.get("val") or {}).get("top5") if isinstance(last.get("val"), dict) else st.get("best_val_top5"),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
    return {}


def snapshot(policy_path: Path, *, history: list[dict] | None = None) -> dict:
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    mon = policy.get("monitoring") or {}
    status_dir = _ROOT / str((policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3")
    status_dir.mkdir(parents=True, exist_ok=True)
    rev, dirty = _git()
    gpu = _nvidia()
    pid_info = {}
    pid_path = status_dir / "orchestrator.pid.json"
    if pid_path.is_file():
        pid_info = json.loads(pid_path.read_text(encoding="utf-8"))
    state = {}
    state_path = status_dir / "RUN_STATE.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    metrics = _latest_metrics(status_dir)
    orch_pid = int(pid_info.get("pid") or 0)
    alive = _pid_alive(orch_pid) if orch_pid else False
    trainer_pids = _live_trainer_pids()
    util = gpu.get("util_pct")
    disk = _disk_free_gb(_ROOT)
    stall_minutes = float(mon.get("stall_minutes") or 20)
    now = datetime.now(timezone.utc)

    verdict = "IDLE_EXPECTED"
    if state.get("phases"):
        latest_phase = None
        for ph, blob in state["phases"].items():
            if blob.get("status") == "RUNNING":
                latest_phase = ph
                break
        if latest_phase:
            if alive or trainer_pids:
                # Keep ACTIVE while orchestrator or an orphan trainer still runs.
                verdict = "ACTIVE"
            else:
                verdict = "FAILED"
        elif all(blob.get("status") == "COMPLETED" for blob in state["phases"].values()):
            verdict = "COMPLETED"

    # Stall detection using history of low util + no metric progress
    hist = history or []
    if metrics.get("mtime"):
        try:
            mt = datetime.fromisoformat(metrics["mtime"])
            age_min = (now - mt).total_seconds() / 60.0
        except Exception:
            age_min = 0.0
    else:
        age_min = 0.0
    low_util = util is not None and util < float(mon.get("stall_gpu_util_pct_max") or 5)
    if alive and age_min >= stall_minutes and low_util:
        consecutive = 1
        for prev in reversed(hist[-10:]):
            if prev.get("verdict") in {"STALLED", "ACTIVE"} and prev.get("gpu", {}).get("util_pct", 99) < 5:
                consecutive += 1
            else:
                break
        if consecutive >= int(mon.get("stall_consecutive_snapshots") or 3):
            verdict = "STALLED"

    if disk is not None and disk < float(mon.get("disk_free_gb_min") or 20):
        verdict = "FAILED"

    snap = {
        "schema": "ms_sta_retrain_v3_monitor_snapshot_v1",
        "ts": now.isoformat(),
        "verdict": verdict,
        "policy_sha256": sha256_file(policy_path),
        "split_sha256": (policy.get("lineage") or {}).get("split_manifest_sha256"),
        "git_revision": rev,
        "git_dirty": dirty,
        "orchestrator": {"pid": orch_pid, "alive": alive, **pid_info},
        "trainer_pids": trainer_pids,
        "state_summary": {
            "evaluation_locked": state.get("evaluation_locked"),
            "active_variant": state.get("active_variant"),
            "phases": {
                k: {
                    "status": v.get("status"),
                    "completed": v.get("completed"),
                    "total": v.get("total"),
                }
                for k, v in (state.get("phases") or {}).items()
            },
        },
        "metrics": metrics,
        "gpu": gpu,
        "disk_free_gb": disk,
        "path": "B",
        "test_claim_label": policy.get("test_claim_label"),
    }
    latest = status_dir / "monitor_latest.json"
    latest.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hist_path = status_dir / "monitor_history.jsonl"
    with hist_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")
    return snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = ap.parse_args()
    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    snap = snapshot(policy_path)
    print(json.dumps({"verdict": snap["verdict"], "metrics": snap.get("metrics"), "gpu": snap.get("gpu")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
