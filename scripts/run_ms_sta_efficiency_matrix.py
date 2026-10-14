"""Run the frozen three-seed Skeleton stride-4 validation study.

The study is auxiliary and never accesses test/E2. A clean worktree is
required before launch; outputs are isolated from defense checkpoints.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import git_dirty, git_rev, load_json, sha256_file


def _metric_summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def _aggregate(policy: dict, policy_path: Path, status_dir: Path, ckpt_dir: Path) -> dict:
    base_pack = load_json(_ROOT / policy["lineage"]["base_b4_aggregate"])
    base_rows = {int(x["seed"]): x for x in base_pack["summary"]["b4"]["per_seed"]}
    rows = []
    for seed in policy["seeds"]["list"]:
        status_path = status_dir / f"b4_seed{seed}.json"
        ckpt_path = ckpt_dir / f"b4_seed{seed}.pth"
        if not status_path.is_file() or not ckpt_path.is_file():
            raise ValueError(f"missing seed artifact: {seed}")
        status = load_json(status_path)
        if status.get("test_eval") != "deferred":
            raise ValueError(f"seed {seed} test_eval is not deferred")
        if status.get("skel_temporal_stride") != 4:
            raise ValueError(f"seed {seed} stride is not 4")
        if status.get("e1_policy_sha256") != sha256_file(policy_path):
            raise ValueError(f"seed {seed} policy sha mismatch")
        base = base_rows[int(seed)]
        rows.append(
            {
                "seed": int(seed),
                "n": int(status["history"][-1]["val"]["n"]),
                "base_top1": float(base["top1"]),
                "stride4_top1": float(status["best_val_top1"]),
                "delta_top1_pp": float(status["best_val_top1"]) - float(base["top1"]),
                "base_top5": float(base["top5"]),
                "stride4_top5": float(status["best_val_top5"]),
                "delta_top5_pp": float(status["best_val_top5"]) - float(base["top5"]),
                "status": status_path.relative_to(_ROOT).as_posix(),
                "checkpoint": ckpt_path.relative_to(_ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(ckpt_path),
            }
        )

    top1 = [r["stride4_top1"] for r in rows]
    top5 = [r["stride4_top5"] for r in rows]
    d1 = [r["delta_top1_pp"] for r in rows]
    d5 = [r["delta_top5_pp"] for r in rows]
    rule = policy["decision_rule_predeclared"]["accuracy"]
    checks = {
        "mean_top1_delta_pass": statistics.mean(d1)
        >= float(rule["mean_top1_delta_vs_b4_pp_min"]),
        "each_seed_top1_delta_pass": min(d1)
        >= float(rule["each_seed_top1_delta_vs_b4_pp_min"]),
        "mean_top5_delta_pass": statistics.mean(d5)
        >= float(rule["mean_top5_delta_vs_b4_pp_min"]),
    }
    checks["pass"] = all(checks.values())
    return {
        "schema": "ms_sta_efficiency_matrix_v1",
        "status": "VALIDATION_PASS" if checks["pass"] else "VALIDATION_FAIL",
        "claim_allowed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev(_ROOT),
        "git_dirty": git_dirty(_ROOT),
        "policy": policy_path.relative_to(_ROOT).as_posix(),
        "policy_sha256": sha256_file(policy_path),
        "split": "validation_only",
        "test_access": "forbidden_and_not_used",
        "rows": rows,
        "stride4_top1": _metric_summary(top1),
        "stride4_top5": _metric_summary(top5),
        "paired_seed_delta_top1_pp": _metric_summary(d1),
        "paired_seed_delta_top5_pp": _metric_summary(d5),
        "decision_rule": rule,
        "checks": checks,
        "next_gate": "canonical benchmark on the same clean git revision",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen MS-STA efficiency matrix")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("Docs/results/ms_sta_efficiency_policy/EFFICIENCY_POLICY.json"),
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    if not policy_path.is_file():
        print(f"[FAIL] policy missing: {policy_path}")
        return 2
    policy = load_json(policy_path)
    if policy.get("status") != "FROZEN":
        print("[FAIL] policy is not FROZEN")
        return 2
    if git_dirty(_ROOT) and not args.aggregate_only:
        print("[FAIL] canonical efficiency matrix requires clean worktree")
        return 2

    art = policy["artifact_dirs"]
    ckpt_dir = _ROOT / art["checkpoints"]
    status_dir = _ROOT / art["status"]
    if not args.aggregate_only:
        for directory in (ckpt_dir, status_dir):
            if directory.exists() and any(directory.iterdir()):
                print(f"[FAIL] output dir is not empty: {directory}")
                return 2
            directory.mkdir(parents=True, exist_ok=True)

        for seed in policy["seeds"]["list"]:
            log_path = status_dir / f"train_seed{seed}.log"
            cmd = [
                sys.executable,
                "-m",
                "src.training.train_ms_sta_vsl400",
                "--variant",
                "b4",
                "--seed",
                str(seed),
                "--epochs",
                str(policy["train_hyperparameters"]["epochs"]),
                "--batch-size",
                str(policy["train_hyperparameters"]["batch_size"]),
                "--lr",
                str(policy["train_hyperparameters"]["lr"]),
                "--patience",
                str(policy["train_hyperparameters"]["patience_early_stopping"]),
                "--device",
                "cuda",
                "--e1-policy",
                str(policy_path),
            ]
            started = time.time()
            proc = subprocess.run(
                cmd,
                cwd=_ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log_path.write_text(proc.stdout, encoding="utf-8")
            print(
                f"[seed {seed}] exit={proc.returncode} elapsed_s={time.time() - started:.1f} "
                f"log={log_path.relative_to(_ROOT)}",
                flush=True,
            )
            if proc.returncode != 0:
                print(proc.stdout[-4000:])
                return proc.returncode or 2

    try:
        aggregate = _aggregate(policy, policy_path, status_dir, ckpt_dir)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    out = status_dir / "MATRIX_AGGREGATE.json"
    out.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate["checks"], indent=2))
    print(f"[PASS] aggregate -> {out}")
    return 0 if aggregate["checks"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
