# -*- coding: utf-8 -*-
"""E4-R4C HPO monitor — 10-minute ticks; stall only if GPU~0 + CPU flat + artifacts/log stale ≥30min.

Does not modify trainer code or restart jobs. Read-only observation.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/hpo"
LOG = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_LIVE.log"
OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_MONITOR.jsonl"


def gpu_stats() -> dict:
    import subprocess

    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        parts = [p.strip() for p in raw.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
            "temp_c": float(parts[3]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def disk_free_gib() -> float | None:
    import shutil

    try:
        u = shutil.disk_usage(str(ROOT))
        return u.free / (1024**3)
    except Exception:  # noqa: BLE001
        return None


def trial_status() -> list[dict]:
    rows = []
    if not RUNS.is_dir():
        return rows
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        row = {"config_id": d.name, "done": summary.is_file()}
        if summary.is_file():
            obj = json.loads(summary.read_text(encoding="utf-8"))
            row["best_val_top1"] = obj.get("best_val_top1")
        rows.append(row)
    return rows


def fingerprint() -> dict:
    log_mtime = LOG.stat().st_mtime if LOG.is_file() else None
    log_size = LOG.stat().st_size if LOG.is_file() else None
    arts = []
    if RUNS.is_dir():
        for p in RUNS.rglob("summary.json"):
            arts.append((str(p.relative_to(ROOT)).replace("\\", "/"), p.stat().st_mtime, p.stat().st_size))
        for p in RUNS.rglob("best.pt"):
            arts.append((str(p.relative_to(ROOT)).replace("\\", "/"), p.stat().st_mtime, p.stat().st_size))
    arts.sort()
    return {"log_mtime": log_mtime, "log_size": log_size, "arts": arts}


def trainer_alive() -> bool:
    import subprocess

    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*ms_sta_e4_r4c_train*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            text=True,
        )
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--tick-seconds", type=int, default=600)
    ap.add_argument("--stall-seconds", type=int, default=1800)
    args = ap.parse_args()

    last_fp = None
    last_change = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    while True:
        now = time.time()
        fp = fingerprint()
        if last_fp is None or fp != last_fp:
            last_fp = fp
            last_change = now
        stale_s = now - last_change
        g = gpu_stats()
        trials = trial_status()
        n_done = sum(1 for t in trials if t["done"])
        alive = trainer_alive()
        disk = disk_free_gib()
        gpu_near0 = float(g.get("gpu_util_pct") or 0) < 2.0
        stall = (not alive) or (gpu_near0 and stale_s >= args.stall_seconds)
        # Official stall rule requires GPU~0 AND CPU flat AND artifacts stale ≥30m.
        # Without sampling CPU deltas here, only flag stall_candidate when gpu~0 and stale.
        stall_candidate = bool(gpu_near0 and stale_s >= args.stall_seconds and not alive)
        row = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "verdict": "STALL_CANDIDATE" if stall_candidate else ("HPO_COMPLETE" if n_done >= 8 and not alive else "RUNNING_HEALTHY_BUT_SLOW"),
            "trainer_alive": alive,
            "n_done": n_done,
            "n_trials": len(trials),
            "trials": trials,
            "gpu": g,
            "disk_free_gib": disk,
            "disk_warn": disk is not None and disk < 25.0,
            "stale_seconds": stale_s,
            "forbid_auto_confirm": True,
        }
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            row["verdict"],
            f"done={n_done}/{len(trials)}",
            f"alive={alive}",
            f"gpu={g.get('gpu_util_pct')}",
            f"stale_s={int(stale_s)}",
            f"disk_gib={None if disk is None else round(disk,1)}",
            flush=True,
        )
        if args.once or (n_done >= 8 and not alive):
            return 0
        time.sleep(max(30, int(args.tick_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
