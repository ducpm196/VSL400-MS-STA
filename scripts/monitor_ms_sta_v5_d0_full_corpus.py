#!/usr/bin/env python3
"""Monitor D0 full-corpus runtime state (read-only) and emit MONITOR_TICK.json."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    CAMPAIGN_TOKEN_V15,
    CORPUS_N,
    D0_FULL_CORPUS_APPROVAL_V14_REL,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
    MANIFEST_PHASES,
    OUTPUT_VERSION,
    OUTPUT_VERSION_V15,
    STARTUP_STALL_SEC,
    WATCHDOG_INTERVAL_SEC,
    atomic_replace_json,
    binary_root_path,
    campaign_related_process_alive,
    load_json,
    load_worker_registry,
    count_manifest_records,
    manifest_index_path,
    runtime_dir,
    sha256_file,
    worker_entry_alive,
)


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return round(usage.free / (1024**3), 2)


def _worker_summary(runtime: Path) -> dict:
    workers = load_worker_registry(runtime)
    alive = [w for w in workers if worker_entry_alive(w)]
    return {
        "registered": len(workers),
        "alive": len(alive),
        "phases": sorted({str(w.get("phase") or "") for w in alive if w.get("phase")}),
        "pids": [int(w.get("pid") or 0) for w in alive],
    }


def _committed_counts(runtime: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phase in MANIFEST_PHASES:
        n_samples = count_manifest_records(runtime, phase)
        if n_samples > 0:
            counts[phase] = n_samples
            continue
        idx_path = manifest_index_path(runtime, phase)
        if idx_path.is_file():
            counts[phase] = int(load_json(idx_path).get("n_records") or 0)
        else:
            counts[phase] = 0
    return counts


def _parse_ts(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    except ValueError:
        return None


def _progress_metrics(
    runtime: Path,
    *,
    current_counts: dict[str, int],
    status: str | None,
    now_ts: float,
    aggregate_target_override: int | None = None,
    phase: str | None = None,
    workers_alive: int = 0,
    run_updated_at: str | None = None,
    contract_created_at: str | None = None,
) -> dict:
    tick_path = runtime / "MONITOR_TICK.json"
    prev = load_json(tick_path) if tick_path.is_file() else None
    prev_counts = (prev or {}).get("committed_counts") or {}
    prev_ts = _parse_ts((prev or {}).get("timestamp"))
    last_progress_at = (prev or {}).get("last_progress_at") if prev else None

    if aggregate_target_override is not None:
        # Carry-forward v15: execution aggregate is RGB-only.
        aggregate_target = int(aggregate_target_override)
        total_now = int(current_counts.get("rgb_token_v3") or 0)
        total_prev = int(prev_counts.get("rgb_token_v3") or 0)
        remaining_aggregate = max(0, aggregate_target - total_now)
    else:
        aggregate_target = 2 * CORPUS_N
        total_now = sum(int(current_counts.get(p) or 0) for p in MANIFEST_PHASES)
        total_prev = sum(int(prev_counts.get(p) or 0) for p in MANIFEST_PHASES)
        remaining_aggregate = max(0, aggregate_target - total_now)

    if total_now > total_prev:
        last_progress_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))

    remaining_by_phase = {
        p: max(0, CORPUS_N - int(current_counts.get(p) or 0)) for p in MANIFEST_PHASES
    }

    progress_rate_per_hour: float | None = None
    eta_hours_aggregate: float | None = None
    eta_hours_by_phase: dict[str, float | None] = {p: None for p in MANIFEST_PHASES}
    if prev_ts is not None and total_now > total_prev and prev_ts < now_ts:
        delta = total_now - total_prev
        hours = (now_ts - prev_ts) / 3600.0
        if hours > 0:
            progress_rate_per_hour = round(delta / hours, 4)
            if progress_rate_per_hour > 0:
                eta_hours_aggregate = round(remaining_aggregate / progress_rate_per_hour, 2)
                for p in MANIFEST_PHASES:
                    phase_delta = int(current_counts.get(p) or 0) - int(prev_counts.get(p) or 0)
                    if phase_delta > 0:
                        phase_rate = phase_delta / hours
                        phase_remaining = remaining_by_phase[p]
                        if phase_rate > 0 and phase_remaining > 0:
                            eta_hours_by_phase[p] = round(phase_remaining / phase_rate, 2)

    stale_progress_alert = False
    startup_stall_alert = False
    baseline_ts = _parse_ts(last_progress_at)
    if baseline_ts is None:
        baseline_ts = _parse_ts(run_updated_at) or _parse_ts(contract_created_at)

    if status in {"RUNNING", "RESUMED"} and baseline_ts is not None:
        age = now_ts - baseline_ts
        if last_progress_at and age > (2 * WATCHDOG_INTERVAL_SEC):
            stale_progress_alert = True
        rgb_now = int(current_counts.get("rgb_token_v3") or 0)
        phase_name = str(phase or "")
        if (
            age > STARTUP_STALL_SEC
            and int(workers_alive) <= 0
            and rgb_now == 0
            and phase_name in {"", "rgb_token_v3", "carry_forward_verified"}
        ):
            startup_stall_alert = True
            stale_progress_alert = True

    return {
        "committed_counts": dict(current_counts),
        "last_progress_at": last_progress_at,
        "progress_rate_per_hour": progress_rate_per_hour,
        "eta_hours": eta_hours_aggregate,
        "eta_hours_aggregate": eta_hours_aggregate,
        "eta_hours_by_phase": eta_hours_by_phase,
        "remaining_aggregate": remaining_aggregate,
        "remaining_by_phase": remaining_by_phase,
        "stale_progress_alert": stale_progress_alert,
        "startup_stall_alert": startup_stall_alert,
        "execution_aggregate_target": aggregate_target,
    }


def _resolve_identity(
    *,
    root: Path,
    output_version: str | None,
    campaign_token: str | None,
) -> tuple[str, str, Path, dict]:
    """Resolve approval identity. Default remains hermetic v14 when unspecified."""
    ver = (output_version or "").strip() or None
    token = (campaign_token or "").strip() or None

    if ver is None and token is None:
        ver = OUTPUT_VERSION
        token = CAMPAIGN_TOKEN
    elif ver is None or token is None:
        raise ValueError("monitor_identity_incomplete:require_both_output_version_and_campaign_token_or_neither")

    if ver == OUTPUT_VERSION_V15 or token == CAMPAIGN_TOKEN_V15:
        if ver != OUTPUT_VERSION_V15:
            raise ValueError("monitor_v15_output_version_required")
        if token != CAMPAIGN_TOKEN_V15:
            raise ValueError("monitor_v15_campaign_token_required")
        approval_path = root / D0_FULL_CORPUS_APPROVAL_V15_REL
        if not approval_path.is_file():
            raise FileNotFoundError("approval_v15_missing")
        approval = load_json(approval_path)
        if approval.get("campaign_token") != CAMPAIGN_TOKEN_V15:
            raise ValueError("monitor_v15_approval_campaign_mismatch")
        if approval.get("output_version") != OUTPUT_VERSION_V15:
            raise ValueError("monitor_v15_approval_output_version_mismatch")
        return OUTPUT_VERSION_V15, CAMPAIGN_TOKEN_V15, approval_path, approval

    if ver != OUTPUT_VERSION or token != CAMPAIGN_TOKEN:
        raise ValueError(f"monitor_unknown_identity:version={ver}:token={token}")

    approval_path = root / D0_FULL_CORPUS_APPROVAL_V14_REL
    if not approval_path.is_file():
        raise FileNotFoundError("approval_v14_missing")
    approval = load_json(approval_path)
    return OUTPUT_VERSION, CAMPAIGN_TOKEN, approval_path, approval


def build_monitor_payload(
    *,
    root: Path = _ROOT,
    output_version: str | None = None,
    campaign_token: str | None = None,
) -> dict:
    try:
        out_ver, camp, _approval_path, approval = _resolve_identity(
            root=root, output_version=output_version, campaign_token=campaign_token
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    bin_rel = approval.get("binary_root_rel")
    if not (isinstance(bin_rel, str) and bin_rel.strip()):
        return {"ok": False, "error": "approval_binary_root_rel_missing"}
    # Always use sealed approval-owned binary root; never derive from HEAD.
    binary = (root / bin_rel.replace("\\", "/")).resolve()
    if out_ver == OUTPUT_VERSION_V15 and "ms_sta_v5_d0_full_v14_" in str(binary).replace("\\", "/"):
        return {"ok": False, "error": "monitor_v15_refuses_v14_binary_root"}

    runtime = runtime_dir(binary)
    state_path = runtime / "RUNTIME_MUTABLE.json"
    if not state_path.is_file():
        state_path = runtime / "RUN_STATE.json"
    contract_path = runtime / "_run_contract.json"
    run_state = load_json(state_path) if state_path.is_file() else None
    now_ts = time.time()
    status = (run_state or {}).get("status")
    committed_counts = _committed_counts(runtime)
    aggregate_target_override: int | None = None
    skeleton_source: str | None = None
    if out_ver == OUTPUT_VERSION_V15 and approval.get("carry_forward_required") is True:
        n_skel = int(approval.get("eligible_n") or CORPUS_N)
        carry_path = root / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
        if carry_path.is_file():
            try:
                carry = load_json(carry_path)
                if carry.get("ok") is True:
                    for key in ("summary_n_ok", "index_n_records"):
                        if carry.get(key) is not None:
                            n_skel = int(carry[key])
                            break
                    bind = approval.get("skeleton_carry_forward_audit_sha256")
                    if bind is None or bind == sha256_file(carry_path):
                        pass
                    else:
                        n_skel = int(approval.get("eligible_n") or CORPUS_N)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                n_skel = int(approval.get("eligible_n") or CORPUS_N)
        committed_counts["skeleton_v3"] = n_skel
        aggregate_target_override = CORPUS_N
        skeleton_source = "read_only_carry_forward"
    worker_registry = _worker_summary(runtime)
    phase_name = (run_state or {}).get("phase")
    contract_obj = load_json(contract_path) if contract_path.is_file() else None
    progress = _progress_metrics(
        runtime,
        current_counts=committed_counts,
        status=status,
        now_ts=now_ts,
        aggregate_target_override=aggregate_target_override,
        phase=str(phase_name) if phase_name is not None else None,
        workers_alive=int(worker_registry.get("alive") or 0),
        run_updated_at=str((run_state or {}).get("updated_at") or "") or None,
        contract_created_at=str((contract_obj or {}).get("created_at") or "") or None,
    )
    payload: dict = {
        "ok": True,
        "output_version": out_ver,
        "campaign_token": camp,
        "binary_root": str(binary),
        "binary_root_rel": bin_rel.replace("\\", "/"),
        "runtime_dir": str(runtime),
        "run_state_exists": state_path.is_file(),
        "run_contract_exists": contract_path.is_file(),
        "run_state": run_state,
        "phase": phase_name,
        "disk_free_gb": _disk_free_gb(root),
        "worker_registry": worker_registry,
        "live_orchestrator": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
        **progress,
    }
    if skeleton_source is not None:
        payload["skeleton_source"] = skeleton_source
    if run_state:
        pid = int(run_state.get("orchestrator_pid") or 0)
        token = str(run_state.get("campaign_token") or camp)
        create_time = run_state.get("orchestrator_create_time")
        try:
            create_time_f = float(create_time) if create_time is not None else None
        except (TypeError, ValueError):
            create_time_f = None
        payload["orchestrator_pid"] = pid
        payload["live_orchestrator"] = bool(
            campaign_related_process_alive(
                pid,
                campaign_token=token,
                runtime=runtime,
                orchestrator_create_time=create_time_f,
            )
        )
        payload["resumes_used"] = run_state.get("resumes_used")
        payload["restart_count"] = run_state.get("restart_count")
        payload["status"] = run_state.get("status")
    elif contract_path.is_file():
        payload["status"] = (contract_obj or load_json(contract_path)).get("status")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-tick", action="store_true", help="Write MONITOR_TICK.json to runtime dir.")
    ap.add_argument(
        "--output-version",
        default=None,
        help="Exact output version (v14 default when omitted with campaign-token).",
    )
    ap.add_argument(
        "--campaign-token",
        default=None,
        help="Exact campaign token. For v15 must be D0_FULL_CORPUS_V15.",
    )
    args = ap.parse_args()
    payload = build_monitor_payload(
        root=_ROOT,
        output_version=args.output_version,
        campaign_token=args.campaign_token,
    )
    if payload.get("ok") is not True:
        print(json.dumps(payload, indent=2))
        return 2
    if args.write_tick:
        # Fail-closed: never mkdir/write MONITOR_TICK before runner-owned immutable contract.
        runtime = Path(payload["runtime_dir"])
        contract_path = runtime / "_run_contract.json"
        if not contract_path.is_file():
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "write_tick_requires_run_contract",
                        "runtime_dir": str(runtime),
                    },
                    indent=2,
                )
            )
            return 2
        try:
            contract = load_json(contract_path)
            _out_ver, _camp, _apath, approval = _resolve_identity(
                root=_ROOT,
                output_version=args.output_version,
                campaign_token=args.campaign_token,
            )
        except (OSError, json.JSONDecodeError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"ok": False, "error": f"write_tick_contract_unreadable:{exc}"}, indent=2))
            return 2
        tick_errors: list[str] = []
        if contract.get("campaign_token") != payload.get("campaign_token"):
            tick_errors.append("write_tick_contract_campaign_mismatch")
        if contract.get("output_version") != payload.get("output_version"):
            tick_errors.append("write_tick_contract_output_version_mismatch")
        if contract.get("code_lock_git_revision") != approval.get("code_lock_git_revision"):
            tick_errors.append("write_tick_contract_code_lock_mismatch")
        if contract.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
            tick_errors.append("write_tick_contract_executable_tree_mismatch")
        c_bin = str(contract.get("binary_root_rel") or "").replace("\\", "/")
        a_bin = str(approval.get("binary_root_rel") or "").replace("\\", "/")
        if c_bin != a_bin:
            tick_errors.append("write_tick_contract_binary_root_mismatch")
        if tick_errors:
            print(json.dumps({"ok": False, "error": ";".join(tick_errors)}, indent=2))
            return 2
        runtime.mkdir(parents=True, exist_ok=True)
        atomic_replace_json(runtime / "MONITOR_TICK.json", payload)
    if args.json or True:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
