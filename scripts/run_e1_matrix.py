"""Sequential fail-fast E1 matrix orchestrator (canonical, no pilot flags).

Order: seeds [42,123,456]; within each seed variants B0->B1->B3->B4->B5.
Does not compute test KPI and does not unlock E2.
Artifact dirs come from frozen E1 policy (E1-v2: gate_e1_v2 / ms_sta_e1_v2).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = _ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY.json"
SEEDS = [42, 123, 456]
VARIANTS = ["b0", "b1", "b3", "b4", "b5"]  # B2 optional diagnostic excluded


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True)
            .strip()
        )
    except Exception:
        return ""


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_ROOT.resolve()).as_posix()


def _artifact_dirs(policy: dict) -> tuple[Path, Path]:
    art = policy.get("artifact_dirs") or {}
    ckpt = _ROOT / str(art.get("checkpoints") or "data/processed/ms_sta_e1_v2")
    status = _ROOT / str(art.get("status") or "logs/gate_e1_v2")
    return ckpt, status


def _row_from_artifacts(
    i: int,
    variant: str,
    seed: int,
    *,
    log_path: Path,
    exit_code: int,
    elapsed_s: float,
    status_dir: Path,
    ckpt_dir: Path,
) -> dict:
    status_path = status_dir / f"{variant}_seed{seed}.json"
    ckpt_path = ckpt_dir / f"{variant}_seed{seed}.pth"
    row = {
        "index": i,
        "variant": variant,
        "seed": seed,
        "exit_code": exit_code,
        "elapsed_s": elapsed_s,
        "log": _rel(log_path) if log_path.exists() else None,
        "status_json": _rel(status_path) if status_path.is_file() else None,
        "checkpoint": _rel(ckpt_path) if ckpt_path.is_file() else None,
        "checkpoint_sha256": _sha256(ckpt_path) if ckpt_path.is_file() else None,
    }
    if status_path.is_file():
        st = json.loads(status_path.read_text(encoding="utf-8"))
        row["best_val_top1"] = st.get("best_val_top1")
        row["best_val_top5"] = st.get("best_val_top5")
        row["init_mode"] = st.get("init_mode")
        row["e1_policy_sha256"] = st.get("e1_policy_sha256")
        row["test_eval"] = st.get("test_eval")
        row["collapse_guard_failed"] = st.get("collapse_guard_failed")
    return row


def _already_complete(variant: str, seed: int, *, status_dir: Path, ckpt_dir: Path) -> bool:
    status_path = status_dir / f"{variant}_seed{seed}.json"
    ckpt_path = ckpt_dir / f"{variant}_seed{seed}.pth"
    if not (status_path.is_file() and ckpt_path.is_file()):
        return False
    st = json.loads(status_path.read_text(encoding="utf-8"))
    if st.get("collapse_guard_failed"):
        return False
    return st.get("best_val_top1") is not None and st.get("test_eval") == "deferred"


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonical E1 sequential matrix")
    ap.add_argument(
        "--python",
        type=Path,
        default=_ROOT / "_venvs" / "train_cuda" / "Scripts" / "python.exe",
    )
    ap.add_argument("--e1-policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="MATRIX_RUN.json path (default: {policy.status}/MATRIX_RUN.json)",
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; only aggregate existing status JSONs",
    )
    args = ap.parse_args()

    py = args.python if args.python.is_absolute() else _ROOT / args.python
    if not py.is_file():
        print(f"[FAIL] python not found: {py}")
        return 2
    policy_path = args.e1_policy if args.e1_policy.is_absolute() else _ROOT / args.e1_policy
    if not policy_path.is_file():
        print(f"[FAIL] missing policy: {policy_path}")
        return 2
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_sha = _sha256(policy_path)
    if policy.get("status") != "FROZEN":
        print("[FAIL] policy not FROZEN")
        return 2
    schema = str(policy.get("schema") or "")
    if "v2" not in schema and schema != "option_b_e1_policy_v2":
        print(f"[FAIL] expected E1 policy v2 schema, got {schema!r}")
        return 2

    ckpt_dir, status_dir = _artifact_dirs(policy)
    # Refuse writing into preserved E1-v1 dirs.
    preserve = policy.get("e1_v1_preserve") or {}
    for key, path in (("checkpoints", ckpt_dir), ("status", status_dir)):
        forbidden = preserve.get(key)
        if forbidden and _rel(path) == str(forbidden).replace("\\", "/"):
            print(f"[FAIL] refuse writing E1-v2 into preserved v1 {key}={forbidden}")
            return 2

    out_path = args.out
    if out_path is None:
        out_path = status_dir / "MATRIX_RUN.json"
    elif not out_path.is_absolute():
        out_path = _ROOT / out_path

    head = _git_rev()
    plan = [{"seed": s, "variant": v} for s in SEEDS for v in VARIANTS]
    run_log: dict = {
        "schema": "option_b_e1_matrix_run_v2",
        "e1_version": "v2",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": head,
        "e1_policy": _rel(policy_path),
        "e1_policy_sha256": policy_sha,
        "python": str(py),
        "device": args.device,
        "artifact_dirs": {
            "checkpoints": _rel(ckpt_dir),
            "status": _rel(status_dir),
        },
        "order": "seeds_outer_[42,123,456]__variants_B0_B1_B3_B4_B5",
        "parallel": False,
        "fail_fast": True,
        "test_eval": "deferred",
        "e2_auto": False,
        "plan": plan,
        "runs": [],
        "status": "RUNNING",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    def _flush() -> None:
        out_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _flush()
    if args.aggregate_only:
        return _aggregate(run_log, out_path, policy_path=policy_path, status_dir=status_dir)

    train_script = _ROOT / "src" / "training" / "train_ms_sta_vsl400.py"
    for i, item in enumerate(plan, start=1):
        seed = item["seed"]
        variant = item["variant"]
        log_path = (out_path.parent / f"matrix_{variant}_seed{seed}.log").resolve()
        print(f"\n=== [{i}/{len(plan)}] {variant} seed={seed} ===", flush=True)

        if _already_complete(variant, seed, status_dir=status_dir, ckpt_dir=ckpt_dir):
            row = _row_from_artifacts(
                i,
                variant,
                seed,
                log_path=log_path,
                exit_code=0,
                elapsed_s=0.0,
                status_dir=status_dir,
                ckpt_dir=ckpt_dir,
            )
            row["skipped_existing"] = True
            run_log["runs"].append(row)
            _flush()
            print(
                f"[skip-existing] val@1={row.get('best_val_top1')} ckpt={row.get('checkpoint')}",
                flush=True,
            )
            continue

        cmd = [
            str(py),
            "-u",
            str(train_script),
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--e1-policy",
            str(policy_path),
        ]
        print("[cmd]", " ".join(cmd), flush=True)
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = round(time.time() - t0, 1)
        row = _row_from_artifacts(
            i,
            variant,
            seed,
            log_path=log_path,
            exit_code=proc.returncode,
            elapsed_s=elapsed,
            status_dir=status_dir,
            ckpt_dir=ckpt_dir,
        )
        run_log["runs"].append(row)
        _flush()
        print(
            f"[done] exit={proc.returncode} elapsed={elapsed}s "
            f"val@1={row.get('best_val_top1')} ckpt={row.get('checkpoint')}",
            flush=True,
        )
        if proc.returncode != 0:
            run_log["status"] = "FAILED"
            run_log["failed_at"] = datetime.now(timezone.utc).isoformat()
            run_log["failed_run"] = row
            _flush()
            print(f"[FAIL-FAST] stopping matrix at {variant} seed={seed}", flush=True)
            return proc.returncode

    run_log["status"] = "COMPLETED"
    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    _flush()
    return _aggregate(run_log, out_path, policy_path=policy_path, status_dir=status_dir)


def _aggregate(
    run_log: dict,
    out_path: Path,
    *,
    policy_path: Path,
    status_dir: Path,
) -> int:
    """Validation-only aggregation; no test KPI; no E2 unlock."""
    by_variant: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for seed in SEEDS:
        for variant in VARIANTS:
            p = status_dir / f"{variant}_seed{seed}.json"
            if not p.is_file():
                print(f"[AGG] missing status: {p}")
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
            by_variant[variant].append(
                {
                    "seed": seed,
                    "top1": float(st.get("best_val_top1") or 0.0),
                    "top5": float(st.get("best_val_top5") or 0.0),
                    "n": (st.get("history") or [{}])[-1].get("val", {}).get("n")
                    if st.get("history")
                    else None,
                    "path": _rel(p),
                }
            )

    def _mean_std(vals: list[float]) -> dict:
        import statistics

        if not vals:
            return {"mean": None, "std": None, "n": 0}
        if len(vals) == 1:
            return {"mean": vals[0], "std": 0.0, "n": 1}
        return {
            "mean": float(statistics.mean(vals)),
            "std": float(statistics.stdev(vals)),
            "n": len(vals),
        }

    summary = {}
    for variant, rows in by_variant.items():
        t1 = [r["top1"] for r in rows]
        t5 = [r["top5"] for r in rows]
        summary[variant] = {
            "per_seed": rows,
            "top1": _mean_std(t1),
            "top5": _mean_std(t5),
        }

    deltas = []
    for seed in SEEDS:
        b0 = next((r for r in by_variant["b0"] if r["seed"] == seed), None)
        b3 = next((r for r in by_variant["b3"] if r["seed"] == seed), None)
        b4 = next((r for r in by_variant["b4"] if r["seed"] == seed), None)
        if not (b0 and b3 and b4):
            continue
        deltas.append(
            {
                "seed": seed,
                "b4_minus_b0_top1": b4["top1"] - b0["top1"],
                "b4_minus_b3_top1": b4["top1"] - b3["top1"],
                "b4_minus_b0_top5": b4["top5"] - b0["top5"],
                "b4_minus_b3_top5": b4["top5"] - b3["top5"],
            }
        )

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    mat_pp = float(
        (policy.get("consistent_improvement_claim") or {}).get("material_top5_regression_pp")
        or 1.0
    )
    all_pos_b0 = bool(deltas) and all(d["b4_minus_b0_top1"] > 0 for d in deltas)
    all_pos_b3 = bool(deltas) and all(d["b4_minus_b3_top1"] > 0 for d in deltas)
    mean_d_b0 = (
        sum(d["b4_minus_b0_top1"] for d in deltas) / len(deltas) if deltas else None
    )
    mean_d_b3 = (
        sum(d["b4_minus_b3_top1"] for d in deltas) / len(deltas) if deltas else None
    )
    mean_d_b0_t5 = (
        sum(d["b4_minus_b0_top5"] for d in deltas) / len(deltas) if deltas else None
    )
    top5_ok = mean_d_b0_t5 is not None and mean_d_b0_t5 >= -mat_pp
    consistent = bool(
        all_pos_b0
        and all_pos_b3
        and mean_d_b0 is not None
        and mean_d_b0 > 0
        and mean_d_b3 is not None
        and mean_d_b3 > 0
        and top5_ok
    )
    limited_claim = (not all_pos_b0) and all_pos_b3

    agg = {
        "schema": "option_b_e1_matrix_aggregate_v2",
        "e1_version": "v2",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation_only",
        "test_eval": "deferred",
        "e2_auto": False,
        "registry_status_unchanged": "NO-GO",
        "primary_seed": 42,
        "manual_active_lock_required": True,
        "summary": summary,
        "b4_deltas_per_seed": deltas,
        "consistent_improvement_gate": {
            "b4_gt_b0_all_seeds": all_pos_b0,
            "b4_gt_b3_all_seeds": all_pos_b3,
            "mean_delta_top1_vs_b0": mean_d_b0,
            "mean_delta_top1_vs_b3": mean_d_b3,
            "mean_delta_top5_vs_b0": mean_d_b0_t5,
            "top5_no_material_regression": top5_ok,
            "material_top5_regression_pp": mat_pp,
            "pass": consistent,
            "limited_claim_if_only_b4_gt_b3": limited_claim,
            "note": "E2 confirmation still required before defense claim; do not auto-unlock.",
        },
    }
    agg_path = out_path.with_name("MATRIX_AGGREGATE.json")
    agg_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_log["aggregate"] = _rel(agg_path)
    run_log["aggregate_pass_consistent_improvement"] = consistent
    out_path.write_text(json.dumps(run_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate": _rel(agg_path),
                "consistent_improvement": consistent,
                "deltas": deltas,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
