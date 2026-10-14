"""Canonical MS-STA retrain-v3 orchestrator (Path B).

Phases: hpo_screen, hpo_full, confirm, lock, test, d4.
Never auto-chains confirm -> test. One CUDA trainer at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import (
    active_variant_from_efficiency_gate,
    confirm_plan,
    expand_search_space,
    final_stem_from_candidate_stem,
    hpo_full_plan,
    hpo_screen_plan,
    load_policy,
    model_variant_name,
    promote_top_k_from_screen,
    require_complete_phase_rows,
    require_confirm_final_lifecycle,
    require_frozen_v3_policy,
    require_homogeneous_git_revision,
    require_hpo_candidate_lifecycle,
    run_key,
    select_config_from_full_rows,
    sha256_file,
    verify_d4_acceptance_artifacts,
    verify_test_matrix_artifacts,
)

DEFAULT_POLICY = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3_policy" / "RETRAIN_POLICY.json"
TRAIN_SCRIPT = _ROOT / "src" / "training" / "train_ms_sta_vsl400.py"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_ROOT.resolve()).as_posix()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _artifact_dirs(policy: dict) -> tuple[Path, Path, Path]:
    art = policy.get("artifact_dirs") or {}
    ckpt = _ROOT / str(art.get("checkpoints") or "data/processed/ms_sta_retrain_v3")
    status = _ROOT / str(art.get("status") or "logs/ms_sta_retrain_v3")
    fail = _ROOT / str(art.get("failures") or "logs/ms_sta_retrain_v3/failures")
    return ckpt, status, fail


def _stem(trial: dict) -> str:
    return (
        f"{trial['phase']}__{trial['variant']}__{trial['config_id']}__seed{trial['seed']}"
        f"__stride{trial.get('skel_temporal_stride', 1)}"
    )


def _paths_for(trial: dict, ckpt_dir: Path, status_dir: Path) -> tuple[Path, Path, Path]:
    stem = _stem(trial)
    return (
        ckpt_dir / f"{stem}.pth",
        status_dir / f"{stem}.json",
        ckpt_dir / f"{final_stem_from_candidate_stem(stem)}.pth",
    )


def _complete(
    trial: dict,
    ckpt_dir: Path,
    status_dir: Path,
    *,
    policy_sha: str,
    require_final: bool,
) -> bool:
    ckpt, status, final = _paths_for(trial, ckpt_dir, status_dir)
    if not (ckpt.is_file() and status.is_file()):
        return False
    st = json.loads(status.read_text(encoding="utf-8"))
    if st.get("collapse_guard_failed"):
        return False
    if st.get("best_val_top1") is None:
        return False
    if st.get("test_eval") != "deferred":
        return False
    if st.get("retrain_v3_policy_sha256") and st["retrain_v3_policy_sha256"] != policy_sha:
        return False
    if require_final:
        if not final.is_file():
            return False
        if st.get("checkpoint_status") != "final_selected_checkpoint":
            return False
        if not st.get("final_checkpoint_sha256"):
            return False
        if sha256_file(final) != st.get("final_checkpoint_sha256"):
            return False
    return True


def _row(trial: dict, *, exit_code: int, elapsed_s: float, ckpt_dir: Path, status_dir: Path) -> dict:
    ckpt, status, final = _paths_for(trial, ckpt_dir, status_dir)
    row = {
        **trial,
        "run_key": run_key(trial),
        "exit_code": exit_code,
        "elapsed_s": elapsed_s,
        "checkpoint": _rel(ckpt) if ckpt.is_file() else None,
        "checkpoint_sha256": sha256_file(ckpt) if ckpt.is_file() else None,
        "final_checkpoint": _rel(final) if final.is_file() else None,
        "final_checkpoint_sha256": sha256_file(final) if final.is_file() else None,
        "status_json": _rel(status) if status.is_file() else None,
        "checkpoint_status": "candidate",
    }
    if status.is_file():
        st = json.loads(status.read_text(encoding="utf-8"))
        row["best_val_top1"] = st.get("best_val_top1")
        row["best_val_top5"] = st.get("best_val_top5")
        row["init_mode"] = st.get("init_mode")
        row["test_eval"] = st.get("test_eval")
        row["collapse_guard_failed"] = st.get("collapse_guard_failed")
        row["checkpoint_status"] = st.get("checkpoint_status") or row["checkpoint_status"]
        row["source_candidate_sha256"] = st.get("source_candidate_sha256")
        row["git_revision"] = st.get("git_revision")
        row["manifest"] = st.get("manifest")
        row["manifest_sha256"] = st.get("manifest_sha256")
        if st.get("final_checkpoint_sha256"):
            row["final_checkpoint_sha256"] = st.get("final_checkpoint_sha256")
        if st.get("median_forward_latency_ms") is not None:
            row["median_forward_latency_ms"] = st.get("median_forward_latency_ms")
    return row


def _build_plan(phase: str, policy: dict, state: dict) -> list[dict]:
    if phase == "hpo_screen":
        return hpo_screen_plan(policy)
    if phase == "hpo_full":
        promoted = (state.get("hpo") or {}).get("promoted_configs") or {}
        return hpo_full_plan(policy, top_configs_by_variant=promoted)
    if phase == "confirm":
        selected = (state.get("hpo") or {}).get("selected_configs") or {}
        return confirm_plan(policy, selected_config_by_variant=selected)
    if phase in ("lock", "test", "d4"):
        return []
    raise SystemExit(f"[FAIL] unknown phase {phase}")


def _promote_from_screen(runs: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for variant in ("b0", "b3", "b4"):
        rows = [r for r in runs if r.get("variant") == variant and r.get("phase") == "hpo_screen"]
        out[variant] = promote_top_k_from_screen(rows, k=2)
    return out


def _select_from_full(runs: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for variant in ("b0", "b3", "b4"):
        rows = [r for r in runs if r.get("variant") == variant and r.get("phase") == "hpo_full"]
        out[variant] = select_config_from_full_rows(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain-v3 orchestrator")
    ap.add_argument("--phase", required=False, default="hpo_screen", choices=["hpo_screen", "hpo_full", "confirm", "lock", "test", "d4"])
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument(
        "--python",
        type=Path,
        default=_ROOT / "_venvs" / "train_cuda" / "Scripts" / "python.exe",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--dry-plan",
        action="store_true",
        help="Print plan counts only; do not train",
    )
    ap.add_argument(
        "--campaign-dry-plan",
        action="store_true",
        help="Print full campaign plan counts (24+12+21+lock+test+d4) without training",
    )
    args = ap.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    policy_sha = sha256_file(policy_path)
    ckpt_dir, status_dir, fail_dir = _artifact_dirs(policy)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_plan and not args.campaign_dry_plan:
        # Ignore console-close signals so detaching from the launcher shell does not kill us.
        if os.name == "nt":
            try:
                import ctypes

                HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

                def _ignore_ctrl(_ctrl_type: int) -> bool:
                    return True

                # Keep a process-global ref so the callback is not GC'd.
                global _CONSOLE_CTRL_HANDLER  # type: ignore[name-defined]
                _CONSOLE_CTRL_HANDLER = HandlerRoutine(_ignore_ctrl)
                ctypes.windll.kernel32.SetConsoleCtrlHandler(_CONSOLE_CTRL_HANDLER, True)
            except Exception:
                pass

        # Own console log so launchers need not wrap through cmd.exe redirection.
        orch_log = status_dir / f"orchestrator_{args.phase}.out.log"
        log_f = orch_log.open("a", encoding="utf-8")

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    try:
                        s.write(data)
                        s.flush()
                    except Exception:
                        pass

            def flush(self):
                for s in self.streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        sys.stdout = _Tee(sys.__stdout__, log_f)  # type: ignore[assignment]
        sys.stderr = _Tee(sys.__stderr__, log_f)  # type: ignore[assignment]

    if args.campaign_dry_plan:
        demo_state = {
            "hpo": {
                "promoted_configs": {
                    v: [c["config_id"] for c in expand_search_space(policy)[:2]]
                    for v in ("b0", "b3", "b4")
                },
                "selected_configs": {
                    v: expand_search_space(policy)[0]["config_id"] for v in ("b0", "b3", "b4")
                },
            }
        }
        counts = {
            "hpo_screen": len(hpo_screen_plan(policy)),
            "hpo_full": len(hpo_full_plan(policy, top_configs_by_variant=demo_state["hpo"]["promoted_configs"])),
            "confirm": len(confirm_plan(policy, selected_config_by_variant=demo_state["hpo"]["selected_configs"])),
            "lock": 1,
            "test": 4,
            "d4": 1,
        }
        total_runs = counts["hpo_screen"] + counts["hpo_full"] + counts["confirm"]
        print(
            json.dumps(
                {
                    "campaign": "ms_sta_retrain_v3_path_b",
                    "counts": counts,
                    "training_runs_total": total_runs,
                    "expected": {
                        "hpo_screen": 24,
                        "hpo_full": 12,
                        "confirm": 21,
                        "lock": 1,
                        "test": 4,
                        "d4": 1,
                        "training_runs_total": 57,
                    },
                    "ok": counts
                    == {
                        "hpo_screen": 24,
                        "hpo_full": 12,
                        "confirm": 21,
                        "lock": 1,
                        "test": 4,
                        "d4": 1,
                    }
                    and total_runs == 57,
                    "scripts": {
                        "test": "scripts/evaluate_ms_sta_retrain_v3.py",
                        "d4": "scripts/run_d4_retrain_v3_acceptance.py",
                        "efficiency": "scripts/benchmark_ms_sta_retrain_v3_efficiency.py",
                        "canonicalize": "scripts/canonicalize_ms_sta_retrain_v3.py",
                    },
                },
                indent=2,
            )
        )
        return 0 if total_runs == 57 else 2


    code_lock = policy.get("code_lock") or {}
    head = _git_rev()
    dirty = _git_dirty()
    if code_lock.get("require_clean_worktree", True) and dirty and not args.dry_plan:
        print("[FAIL] clean worktree required for retrain-v3 launch")
        return 2

    state_path = status_dir / "RUN_STATE.json"
    contract_path = status_dir / "_run_contract.json"
    pid_path = status_dir / "orchestrator.pid.json"

    state = {
        "schema": "ms_sta_retrain_v3_run_state_v1",
        "policy": _rel(policy_path),
        "policy_sha256": policy_sha,
        "git_revision": head,
        "git_dirty": dirty,
        "path": "B",
        "test_claim_label": policy.get("test_claim_label"),
        "hpo": {"promoted_configs": {}, "selected_configs": {}},
        "phases": {},
        "active_variant": None,
        "evaluation_locked": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if state_path.is_file():
        prev = json.loads(state_path.read_text(encoding="utf-8"))
        if args.resume or args.phase in ("lock", "test", "d4", "hpo_full", "confirm"):
            if prev.get("policy_sha256") != policy_sha:
                print("[FAIL] resume/continue policy sha mismatch")
                return 2
            if prev.get("git_revision") != head and args.phase not in ("lock", "test", "d4"):
                # Training phases require exact code lock; lock/test/d4 may continue on same policy.
                print("[FAIL] resume git revision mismatch")
                return 2
            if prev.get("git_revision") != head and args.phase in ("lock", "test", "d4"):
                print("[FAIL] eval phases require identical code-lock git revision")
                return 2
            state = prev
        else:
            print("[FAIL] RUN_STATE exists; pass --resume or clear versioned outputs")
            return 2

    py = args.python if args.python.is_absolute() else _ROOT / args.python
    plan = _build_plan(args.phase, policy, state)
    contract = {
        "schema": "ms_sta_retrain_v3_run_contract_v1",
        "phase": args.phase,
        "policy_sha256": policy_sha,
        "split_manifest_sha256": (policy.get("lineage") or {}).get("split_manifest_sha256"),
        "git_revision": head,
        "git_dirty": dirty,
        "device": args.device,
        "plan_n": len(plan),
        "plan_keys": [run_key(t) for t in plan],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "forbid_auto_chain_to_test": True,
    }
    if args.dry_plan:
        expected_n = {
            "hpo_screen": 24,
            "hpo_full": 12,
            "confirm": 21,
            "lock": 1,
            "test": 4,
            "d4": 1,
        }.get(args.phase)
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "plan_n": len(plan) if args.phase not in ("lock", "test", "d4") else expected_n,
                    "expected": expected_n,
                    "sample": plan[:3],
                    "implemented": True,
                },
                indent=2,
            )
        )
        return 0

    if args.phase == "lock":
        return _phase_lock(policy, state, state_path, status_dir, ckpt_dir)
    if args.phase == "test":
        return _phase_test(policy_path, status_dir, state, state_path, py=py)
    if args.phase == "d4":
        return _phase_d4(policy_path, status_dir, state, state_path, py=py)
    _atomic_write(contract_path, contract)
    _atomic_write(
        pid_path,
        {
            "pid": os.getpid(),
            "phase": args.phase,
            "policy_sha256": policy_sha,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    if not py.is_file():
        print(f"[FAIL] python not found: {py}")
        return 2

    space = policy.get("hpo_search_space") or {}
    patience = int(space.get("patience_early_stopping", 5))
    grad_clip = float(space.get("gradient_clipping", 1.0))
    phase_runs: list[dict] = list((state.get("phases") or {}).get(args.phase, {}).get("runs") or [])
    require_final = args.phase == "confirm"

    # Persist RUNNING immediately so monitor/watchdog do not see a silent gap before trial 1 ends.
    state.setdefault("phases", {})[args.phase] = {
        "runs": phase_runs,
        "completed": sum(1 for r in phase_runs if r.get("exit_code") == 0),
        "total": len(plan),
        "status": "RUNNING",
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)

    for i, trial in enumerate(plan, start=1):
        key = run_key(trial)
        print(f"\n=== [{i}/{len(plan)}] {key} ===", flush=True)
        # Fail-closed: refuse HEAD / dirty drift while a training phase is in flight.
        live_rev = _git_rev()
        if live_rev != state.get("git_revision"):
            print(
                "[FAIL] HEAD changed during phase; refuse drifted code-lock "
                f"({live_rev} != {state.get('git_revision')})",
                flush=True,
            )
            return 2
        if _git_dirty():
            print("[FAIL] worktree became dirty during phase; refuse continue", flush=True)
            return 2
        # Drop prior failed attempts for this key so resume can retry cleanly.
        phase_runs = [
            r
            for r in phase_runs
            if not (r.get("run_key") == key and int(r.get("exit_code", 1)) != 0)
        ]
        if any(r.get("run_key") == key and r.get("exit_code") == 0 for r in phase_runs):
            if _complete(
                trial, ckpt_dir, status_dir, policy_sha=policy_sha, require_final=require_final
            ):
                print("[skip-existing] complete artifacts present", flush=True)
                continue
            print("[FAIL] partial/mismatched existing artifacts; refuse overwrite")
            return 2
        if _complete(
            trial, ckpt_dir, status_dir, policy_sha=policy_sha, require_final=require_final
        ):
            row = _row(trial, exit_code=0, elapsed_s=0.0, ckpt_dir=ckpt_dir, status_dir=status_dir)
            row["skipped_existing"] = True
            phase_runs.append(row)
            state.setdefault("phases", {})[args.phase] = {
                "runs": phase_runs,
                "completed": len(phase_runs),
                "total": len(plan),
                "status": "RUNNING",
            }
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write(state_path, state)
            print(f"[skip-existing] val@1={row.get('best_val_top1')}", flush=True)
            continue

        model_var = model_variant_name(trial["variant"])
        cmd = [
            str(py),
            "-u",
            str(TRAIN_SCRIPT),
            "--variant",
            model_var,
            "--experiment-variant",
            trial["variant"],
            "--seed",
            str(trial["seed"]),
            "--epochs",
            str(trial["epochs"]),
            "--batch-size",
            str(int(space.get("batch_size", 32))),
            "--lr",
            str(trial["lr"]),
            "--weight-decay",
            str(trial["weight_decay"]),
            "--grad-clip",
            str(grad_clip),
            "--patience",
            str(patience),
            "--device",
            args.device,
            "--retrain-v3-policy",
            str(policy_path),
            "--config-id",
            trial["config_id"],
            "--phase",
            trial["phase"],
            "--train-role",
            trial["train_role"],
            "--val-role",
            trial["val_role"],
            "--artifact-stem",
            _stem(trial),
        ]
        log_path = status_dir / f"{_stem(trial)}.log"
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=_ROOT, stdout=logf, stderr=subprocess.STDOUT)
        elapsed = round(time.time() - t0, 1)
        row = _row(
            trial,
            exit_code=int(proc.returncode),
            elapsed_s=elapsed,
            ckpt_dir=ckpt_dir,
            status_dir=status_dir,
        )
        row["log"] = _rel(log_path)
        phase_runs.append(row)
        state.setdefault("phases", {})[args.phase] = {
            "runs": phase_runs,
            "completed": sum(1 for r in phase_runs if r.get("exit_code") == 0),
            "total": len(plan),
            "status": "RUNNING",
        }
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(state_path, state)
        if proc.returncode != 0:
            fail_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write(fail_dir / f"{_stem(trial)}.json", row)
            print(f"[FAIL] trainer exit={proc.returncode}", flush=True)
            return int(proc.returncode) or 1

    # Phase finalization — COMPLETED only after reading/verifying output artifacts.
    try:
        if args.phase == "hpo_screen":
            require_complete_phase_rows(phase_runs, expected_n=24, phase="hpo_screen")
            require_homogeneous_git_revision(
                phase_runs, expected_revision=str(state.get("git_revision") or ""), phase="hpo_screen"
            )
            require_hpo_candidate_lifecycle(phase_runs, phase="hpo_screen")
            state.setdefault("hpo", {})["promoted_configs"] = _promote_from_screen(phase_runs)
        elif args.phase == "hpo_full":
            require_complete_phase_rows(phase_runs, expected_n=12, phase="hpo_full")
            require_homogeneous_git_revision(
                phase_runs, expected_revision=str(state.get("git_revision") or ""), phase="hpo_full"
            )
            require_hpo_candidate_lifecycle(phase_runs, phase="hpo_full")
            state.setdefault("hpo", {})["selected_configs"] = _select_from_full(phase_runs)
            state["hpo"]["hpo_checkpoints_disposable"] = True
        elif args.phase == "confirm":
            require_complete_phase_rows(phase_runs, expected_n=21, phase="confirm")
            require_homogeneous_git_revision(
                phase_runs, expected_revision=str(state.get("git_revision") or ""), phase="confirm"
            )
            require_confirm_final_lifecycle(phase_runs)
            for row in phase_runs:
                final_rel = row.get("final_checkpoint")
                if not final_rel:
                    raise ValueError(f"confirm missing final_checkpoint for {row.get('run_key')}")
                final_path = _ROOT / final_rel
                if not final_path.is_file():
                    raise ValueError(f"confirm final missing on disk: {final_rel}")
                if sha256_file(final_path) != row.get("final_checkpoint_sha256"):
                    raise ValueError(f"confirm final sha mismatch for {row.get('run_key')}")
            state["evaluation_locked"] = False
            state["note"] = (
                "Stop for P4 approval; run efficiency benchmark + canonicalize before lock; "
                "do not auto-open test."
            )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    state["phases"][args.phase]["status"] = "COMPLETED"
    state["phases"][args.phase]["runs"] = phase_runs
    state["phases"][args.phase]["completed"] = len(phase_runs)
    state["phases"][args.phase]["total"] = len(plan)
    state["phases"][args.phase]["artifacts_verified"] = True
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)
    print(json.dumps({"ok": True, "phase": args.phase, "completed": len(phase_runs)}, indent=2))
    return 0


def _phase_lock(policy: dict, state: dict, state_path: Path, status_dir: Path, ckpt_dir: Path) -> int:
    """Freeze active variant + primary-seed FINAL checkpoint SHAs before test.

    Fail-closed: architecture gate must pass and CONFIRM_VALIDATION_REPORT SHA must bind.
    """
    if not (state.get("phases") or {}).get("confirm", {}).get("status") == "COMPLETED":
        print("[FAIL] lock requires completed confirm phase")
        return 2
    if state.get("campaign_status") == "VALIDATION_NO_GO":
        print("[FAIL] campaign closed as VALIDATION_NO_GO; refuse lock")
        return 2
    if state.get("campaign_closed") and state.get("campaign_status") != "VALIDATION_GO":
        print("[FAIL] campaign_closed without VALIDATION_GO; refuse lock")
        return 2
    gates = state.get("gates") or {}
    if gates.get("architecture_passed") is not True:
        print(
            "[FAIL] architecture_passed != true; refuse lock "
            "(run scripts/build_confirm_validation_report_v3.py)"
        )
        return 2
    report_rel = gates.get("confirm_validation_report")
    report_sha = gates.get("confirm_validation_report_sha256")
    if not report_rel or not report_sha:
        print("[FAIL] missing confirm_validation_report SHA binding")
        return 2
    report_path = _ROOT / str(report_rel)
    if not report_path.is_file():
        print(f"[FAIL] missing validation report: {report_path}")
        return 2
    actual_report_sha = sha256_file(report_path)
    if actual_report_sha != report_sha:
        print("[FAIL] confirm_validation_report sha mismatch vs RUN_STATE binding")
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("architecture_passed"):
        print("[FAIL] validation report architecture_passed is false; refuse lock")
        return 2
    if report.get("campaign_verdict") == "VALIDATION_NO_GO":
        print("[FAIL] validation report verdict is VALIDATION_NO_GO; refuse lock")
        return 2

    eff = gates.get("efficiency_passed")
    if eff is None:
        print("[FAIL] missing gates.efficiency_passed; run canonicalize with EFFICIENCY_BENCHMARK.json")
        return 2
    active = active_variant_from_efficiency_gate(bool(eff))
    selected = (state.get("hpo") or {}).get("selected_configs") or {}
    primary = int(((policy.get("confirmation") or {}).get("primary_seed") or 42))
    lock = {
        "schema": "ms_sta_retrain_v3_eval_lock_v1",
        "evaluation_locked": True,
        "test_eval": "deferred",
        "active_variant": active,
        "primary_seed": primary,
        "selected_configs": selected,
        "test_claim_label": policy.get("test_claim_label"),
        "checkpoints": {},
        "efficiency_passed": bool(eff),
        "architecture_passed": True,
        "confirm_validation_report": str(report_rel),
        "confirm_validation_report_sha256": report_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reuse_metrics": False,
        "reuse_checkpoints": False,
        "reuse_vsl_weights": False,
    }
    for variant in ("b0", "b3", "b4", "b4e"):
        cid = selected.get(variant) if variant in ("b0", "b3", "b4") else selected.get("b4")
        trial = {
            "phase": "confirm",
            "variant": variant,
            "config_id": cid,
            "seed": primary,
            "skel_temporal_stride": 4 if variant == "b4e" else 1,
        }
        _cand, status, final = _paths_for(trial, ckpt_dir, status_dir)
        if not final.is_file():
            print(f"[FAIL] missing FINAL checkpoint for {variant}: {final}")
            return 2
        if not status.is_file():
            print(f"[FAIL] missing status for {variant}: {status}")
            return 2
        st = json.loads(status.read_text(encoding="utf-8"))
        if st.get("checkpoint_status") != "final_selected_checkpoint":
            print(f"[FAIL] {variant} status is not final_selected_checkpoint")
            return 2
        final_sha = sha256_file(final)
        if st.get("final_checkpoint_sha256") and st["final_checkpoint_sha256"] != final_sha:
            print(f"[FAIL] {variant} final sha mismatch vs status")
            return 2
        lock["checkpoints"][variant] = {
            "path": _rel(final),
            "sha256": final_sha,
            "source_candidate_sha256": st.get("source_candidate_sha256") or sha256_file(_cand),
            "status_json": _rel(status),
            "checkpoint_status": "final_selected_checkpoint",
        }
    if set(lock["checkpoints"]) != {"b0", "b3", "b4", "b4e"}:
        print(f"[FAIL] EVAL_LOCK variants incomplete: {sorted(lock['checkpoints'])}")
        return 2
    lock_path = status_dir / "EVAL_LOCK.json"
    _atomic_write(lock_path, lock)
    if not lock_path.is_file():
        print("[FAIL] EVAL_LOCK.json not written")
        return 2
    state["evaluation_locked"] = True
    state["active_variant"] = active
    state["eval_lock"] = _rel(lock_path)
    state["phases"]["lock"] = {
        "status": "COMPLETED",
        "total": 1,
        "completed": 1,
        "artifacts_verified": True,
        "eval_lock": _rel(lock_path),
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)
    print(json.dumps({"ok": True, "active_variant": active, "lock": _rel(lock_path)}, indent=2))
    return 0


def _phase_test(policy_path: Path, status_dir: Path, state: dict, state_path: Path, *, py: Path) -> int:
    if not state.get("evaluation_locked"):
        print("[FAIL] test requires evaluation_locked=true (run lock first)")
        return 2
    if not py.is_file():
        print(f"[FAIL] python not found: {py}")
        return 2
    cmd = [
        str(py),
        "-u",
        str(_ROOT / "scripts/evaluate_ms_sta_retrain_v3.py"),
        "--policy",
        str(policy_path),
        "--confirm-retrospective-test",
    ]
    proc = subprocess.run(cmd, cwd=_ROOT)
    if proc.returncode != 0:
        return int(proc.returncode) or 1
    try:
        verified = verify_test_matrix_artifacts(status_dir, expected_n=6165)
    except ValueError as exc:
        print(f"[FAIL] test artifacts not verified: {exc}")
        return 2
    state["phases"]["test"] = {
        "status": "COMPLETED",
        "total": 4,
        "completed": 4,
        "artifacts_verified": True,
        "verification": verified,
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)
    print(json.dumps({"ok": True, "phase": "test", "completed": 4, "verified": verified}, indent=2))
    return 0


def _phase_d4(policy_path: Path, status_dir: Path, state: dict, state_path: Path, *, py: Path) -> int:
    if not state.get("evaluation_locked"):
        print("[FAIL] d4 requires evaluation_locked=true")
        return 2
    if not (status_dir / "TEST_MATRIX.json").is_file():
        print("[FAIL] d4 requires TEST_MATRIX.json from test phase")
        return 2
    if not py.is_file():
        print(f"[FAIL] python not found: {py}")
        return 2
    cmd = [
        str(py),
        "-u",
        str(_ROOT / "scripts/run_d4_retrain_v3_acceptance.py"),
        "--policy",
        str(policy_path),
        "--confirm-d4",
    ]
    proc = subprocess.run(cmd, cwd=_ROOT)
    if proc.returncode != 0:
        return int(proc.returncode) or 1
    try:
        verified = verify_d4_acceptance_artifacts(status_dir)
    except ValueError as exc:
        print(f"[FAIL] d4 artifacts not verified: {exc}")
        return 2
    state["phases"]["d4"] = {
        "status": "COMPLETED",
        "total": 1,
        "completed": 1,
        "artifacts_verified": True,
        "verification": verified,
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)
    print(json.dumps({"ok": True, "phase": "d4", "completed": 1, "verified": verified}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
