"""Canonicalize the frozen validation-only B4 accuracy study."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import git_dirty, git_rev, sha256_file

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_accuracy_policy/ACCURACY_POLICY.json"
DEFAULT_BASE = _ROOT / "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json"
DEFAULT_STATUS = _ROOT / "logs/ms_sta_accuracy_v1"
DEFAULT_OUT = _ROOT / "Docs/results/ms_sta_accuracy"


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "n": len(values),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(_ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--status-dir", type=Path, default=DEFAULT_STATUS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    base_path = args.base if args.base.is_absolute() else _ROOT / args.base
    status_dir = args.status_dir if args.status_dir.is_absolute() else _ROOT / args.status_dir
    out_dir = args.output_dir if args.output_dir.is_absolute() else _ROOT / args.output_dir
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN" or policy.get("claim_allowed") is not False:
        print("[FAIL] accuracy policy must be FROZEN with claim_allowed=false")
        return 2
    if policy["evaluation"].get("test_access") != "forbidden":
        print("[FAIL] policy does not forbid test access")
        return 2

    policy_sha = sha256_file(policy_path)
    seeds = [int(seed) for seed in policy["seeds"]["list"]]
    base_rows = {int(row["seed"]): row for row in base["summary"]["b4"]["per_seed"]}
    per_seed = []
    revisions = set()
    for seed in seeds:
        status_path = status_dir / f"b4_seed{seed}.json"
        if not status_path.is_file():
            print(f"[FAIL] missing status: {status_path}")
            return 2
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("variant") != "b4" or int(status.get("seed", -1)) != seed:
            print(f"[FAIL] variant/seed mismatch: {status_path}")
            return 2
        if status.get("e1_policy_sha256") != policy_sha:
            print(f"[FAIL] policy SHA mismatch: {status_path}")
            return 2
        ckpt_path = Path(str(status["best_ckpt"]))
        if not ckpt_path.is_absolute():
            ckpt_path = _ROOT / ckpt_path
        if not ckpt_path.is_file():
            print(f"[FAIL] missing checkpoint: {ckpt_path}")
            return 2
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if payload.get("e1_policy_sha256") != policy_sha:
            print(f"[FAIL] checkpoint policy SHA mismatch: {ckpt_path}")
            return 2
        if payload.get("git_dirty") is not False:
            print(f"[FAIL] checkpoint git_dirty is not false: {ckpt_path}")
            return 2
        if payload.get("test_eval") != "deferred":
            print(f"[FAIL] checkpoint test_eval is not deferred: {ckpt_path}")
            return 2
        revisions.add(str(payload.get("git_revision")))
        base_row = base_rows[seed]
        top1 = float(status["best_val_top1"])
        top5 = float(status["best_val_top5"])
        per_seed.append(
            {
                "seed": seed,
                "n": int(payload["val"]["n"]),
                "base_top1": float(base_row["top1"]),
                "new_top1": top1,
                "delta_top1_pp": top1 - float(base_row["top1"]),
                "base_top5": float(base_row["top5"]),
                "new_top5": top5,
                "delta_top5_pp": top5 - float(base_row["top5"]),
                "status": _relative(status_path),
                "status_sha256": sha256_file(status_path),
                "checkpoint": _relative(ckpt_path),
                "checkpoint_sha256": sha256_file(ckpt_path),
            }
        )
    if len(revisions) != 1 or "None" in revisions:
        print(f"[FAIL] checkpoints do not share one code-lock revision: {sorted(revisions)}")
        return 2

    top1_values = [row["new_top1"] for row in per_seed]
    top5_values = [row["new_top5"] for row in per_seed]
    d1_values = [row["delta_top1_pp"] for row in per_seed]
    d5_values = [row["delta_top5_pp"] for row in per_seed]
    rule = policy["decision_rule_predeclared"]
    checks = {
        "mean_top1_delta_pass": statistics.mean(d1_values) >= float(rule["mean_top1_delta_pp_min"]),
        "each_seed_top1_delta_pass": all(
            delta >= float(rule["each_seed_top1_delta_pp_min"]) for delta in d1_values
        ),
        "mean_top5_delta_pass": statistics.mean(d5_values) >= float(rule["mean_top5_delta_pp_min"]),
    }
    checks["pass"] = all(checks.values())
    report = {
        "schema": "ms_sta_accuracy_final_report_v1",
        "status": "PASS" if checks["pass"] else "FAIL",
        "claim_allowed": bool(checks["pass"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonicalizer_git_revision": git_rev(_ROOT),
        "canonicalizer_git_dirty": git_dirty(_ROOT),
        "evidence_git_revision": next(iter(revisions)),
        "scope": "auxiliary validation-only optimization study; locked B4 E2/D4 unchanged",
        "policy": {"path": _relative(policy_path), "sha256": policy_sha},
        "base": {"path": _relative(base_path), "sha256": sha256_file(base_path)},
        "change": {
            "learning_rate_before": 0.001,
            "learning_rate_after": float(policy["train_hyperparameters"]["lr"]),
            "epochs_before": 8,
            "epochs_after": int(policy["train_hyperparameters"]["epochs"]),
            "architecture_change": "none",
        },
        "validation_result": {
            "n_per_seed": per_seed[0]["n"],
            "seeds": seeds,
            "top1": _mean_std(top1_values),
            "top5": _mean_std(top5_values),
            "paired_delta_top1_pp": _mean_std(d1_values),
            "paired_delta_top5_pp": _mean_std(d5_values),
            "per_seed": per_seed,
            "decision_checks": checks,
        },
        "allowed_wording": policy["allowed_claim_if_pass"] if checks["pass"] else None,
        "limits": [
            "No test/E2 access; this is validation-only auxiliary evidence.",
            "The locked B4 defense checkpoint and D4 demo engine are unchanged.",
            "This study changes the training learning rate and epoch budget, not the B4 architecture.",
            "Do not compare this validation result directly with VSL400 paper Table 3.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "FINAL_REPORT.json"
    report_path = out_dir / "REPORT.md"
    _write_json(final_path, report)
    report_path.write_text(
        "# MS-STA accuracy study\n\n"
        f"**Status:** {report['status']} · validation-only · locked E2/D4 unchanged.\n\n"
        f"- B4 optimized Top-1: **{report['validation_result']['top1']['mean']:.2f} ± "
        f"{report['validation_result']['top1']['std']:.2f}%**.\n"
        f"- Paired Δ Top-1 vs locked B4: **{report['validation_result']['paired_delta_top1_pp']['mean']:+.2f} ± "
        f"{report['validation_result']['paired_delta_top1_pp']['std']:.2f} pp**.\n"
        f"- Top-5: **{report['validation_result']['top5']['mean']:.2f} ± "
        f"{report['validation_result']['top5']['std']:.2f}%**.\n\n"
        "This result does not replace the B4 checkpoint used for E2/D4 and makes no test claim.\n",
        encoding="utf-8",
    )
    checksums = {
        final_path.name: sha256_file(final_path),
        report_path.name: sha256_file(report_path),
    }
    _write_json(out_dir / "CHECKSUMS.json", checksums)
    print(
        f"[{report['status']}] Top-1={report['validation_result']['top1']['mean']:.4f} "
        f"delta={report['validation_result']['paired_delta_top1_pp']['mean']:+.4f} pp"
    )
    return 0 if checks["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
