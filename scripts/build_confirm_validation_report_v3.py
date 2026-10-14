"""Build CONFIRM_VALIDATION_REPORT.json independent of efficiency benchmark.

Always writes the artifact. Architecture gate failure yields verdict VALIDATION_NO_GO
without opening test or creating EVAL_LOCK.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import (
    CONFIRM_SEEDS,
    evaluate_architecture_gates,
    load_policy,
    require_complete_phase_rows,
    require_confirm_final_lifecycle,
    require_frozen_v3_policy,
    require_homogeneous_git_revision,
    sha256_file,
)

DEFAULT_POLICY = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3_policy" / "RETRAIN_POLICY.json"


def _git_rev() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return None


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument(
        "--validation-metrics",
        type=Path,
        default=None,
        help="Optional VALIDATION_METRICS.json from validation-only evaluator (macro-F1/per-signer).",
    )
    ap.add_argument(
        "--close-campaign",
        action="store_true",
        help="Mark RUN_STATE campaign_status from architecture gate (NO_GO or ready_for_lock_review).",
    )
    args = ap.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    status_dir = _ROOT / str(
        (policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3"
    )
    state_path = status_dir / "RUN_STATE.json"
    if not state_path.is_file():
        print("[FAIL] missing RUN_STATE.json")
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    confirm = (state.get("phases") or {}).get("confirm") or {}
    runs = list(confirm.get("runs") or [])
    try:
        require_complete_phase_rows(runs, expected_n=21, phase="confirm")
        require_confirm_final_lifecycle(runs)
        require_homogeneous_git_revision(
            runs,
            expected_revision=str(state.get("git_revision") or ""),
            phase="confirm",
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    arch = evaluate_architecture_gates(runs, policy=policy, seeds=CONFIRM_SEEDS)

    by_variant: dict[str, list[dict]] = {}
    for r in runs:
        by_variant.setdefault(str(r["variant"]), []).append(r)

    variants_summary: dict[str, dict] = {}
    for variant, rows in sorted(by_variant.items()):
        rows_sorted = sorted(rows, key=lambda r: int(r.get("seed") or 0))
        t1 = [float(r["best_val_top1"]) for r in rows_sorted]
        t5 = [float(r["best_val_top5"]) for r in rows_sorted if r.get("best_val_top5") is not None]
        m1, s1 = _mean_std(t1)
        m5, s5 = _mean_std(t5)
        variants_summary[variant] = {
            "n": len(rows_sorted),
            "config_id": rows_sorted[0].get("config_id"),
            "top1_by_seed": {str(r.get("seed")): r.get("best_val_top1") for r in rows_sorted},
            "top5_by_seed": {str(r.get("seed")): r.get("best_val_top5") for r in rows_sorted},
            "top1_mean": m1,
            "top1_sample_std": s1,
            "top5_mean": m5,
            "top5_sample_std": s5,
            "final_checkpoints": [
                {
                    "seed": r.get("seed"),
                    "final_checkpoint": r.get("final_checkpoint"),
                    "final_checkpoint_sha256": r.get("final_checkpoint_sha256"),
                    "source_candidate_sha256": r.get("source_candidate_sha256"),
                }
                for r in rows_sorted
            ],
        }

    deep = None
    deep_sha = None
    if args.validation_metrics is not None:
        vm = args.validation_metrics
        vm = vm if vm.is_absolute() else _ROOT / vm
        if not vm.is_file():
            print(f"[FAIL] --validation-metrics not found: {vm}")
            return 2
        deep = json.loads(vm.read_text(encoding="utf-8"))
        if deep.get("split_role") != "val":
            print("[FAIL] validation metrics must use split_role=val (no test)")
            return 2
        if deep.get("test_ids_touched"):
            print("[FAIL] validation metrics touched test IDs")
            return 2
        deep_sha = sha256_file(vm)

    # Efficiency accuracy component from confirm means (no latency claim here).
    b4_mean = (variants_summary.get("b4") or {}).get("top1_mean")
    b4e_mean = (variants_summary.get("b4e") or {}).get("top1_mean")
    mean_b4e_delta = None
    efficiency_accuracy_ok = None
    if b4_mean is not None and b4e_mean is not None:
        mean_b4e_delta = float(b4e_mean) - float(b4_mean)
        efficiency_accuracy_ok = mean_b4e_delta >= -1.0

    verdict = "VALIDATION_GO" if arch.get("passed") else "VALIDATION_NO_GO"
    report = {
        "schema": "ms_sta_retrain_v3_confirm_validation_report_v1",
        "path": "B",
        "campaign_verdict": verdict,
        "architecture_passed": bool(arch.get("passed")),
        "architecture_gates": arch,
        "efficiency_accuracy_component": {
            "mean_b4e_minus_b4_pp": mean_b4e_delta,
            "limit_pp": -1.0,
            "passed": efficiency_accuracy_ok,
            "note": (
                "Accuracy-only component of efficiency gate from confirmation means. "
                "93.75% score-element reduction is a design identity of stride-4, not a "
                "performance claim. Latency/param checks require EFFICIENCY_BENCHMARK.json."
            ),
        },
        "policy_sha256": sha256_file(policy_path),
        "split_manifest_sha256": (policy.get("lineage") or {}).get("split_manifest_sha256"),
        "git_revision": state.get("git_revision"),
        "report_builder_git_revision": _git_rev(),
        "report_builder_git_dirty": _git_dirty(),
        "validation_metrics_sha256": deep_sha,
        "test_claim_label": policy.get("test_claim_label"),
        "test_eval": "deferred",
        "forbid_lock_without_architecture_pass": True,
        "forbid_test_access": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variants": variants_summary,
        "validation_metrics": deep,
        "notes": [
            "Do not claim B4 superior to B3 unless architecture_passed is true.",
            "Do not use 93.75% attention-score reduction as primary evidence when "
            "B4-E accuracy regression exceeds -1.0 pp.",
            "Historical E1-v2/E2/D4 must not replace this v3 validation conclusion.",
        ],
    }

    out = status_dir / "CONFIRM_VALIDATION_REPORT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_sha = sha256_file(out)

    state.setdefault("gates", {})
    state["gates"]["architecture_passed"] = bool(arch.get("passed"))
    state["gates"]["architecture"] = arch
    state["gates"]["confirm_validation_report"] = out.relative_to(_ROOT).as_posix()
    state["gates"]["confirm_validation_report_sha256"] = report_sha
    if efficiency_accuracy_ok is not None:
        state["gates"]["efficiency_accuracy_component_passed"] = bool(efficiency_accuracy_ok)
        state["gates"]["mean_b4e_minus_b4_pp"] = mean_b4e_delta

    if args.close_campaign:
        state["campaign_status"] = verdict
        # Only NO_GO permanently closes the campaign. GO remains open for lock review.
        if verdict == "VALIDATION_NO_GO":
            state["campaign_closed"] = True
            state["evaluation_locked"] = False
            state["note"] = (
                "Campaign closed as VALIDATION_NO_GO. Architecture gate FAIL. "
                "Do not open lock/test/D4. Do not rewrite claims."
            )
            forbid_lock = True
        else:
            state["campaign_closed"] = False
            state["ready_for_lock_review"] = True
            state["evaluation_locked"] = False
            state["note"] = (
                "VALIDATION_GO recorded. Campaign remains open for lock review "
                "(architecture_passed=true required)."
            )
            forbid_lock = False
        marker = {
            "schema": "ms_sta_retrain_v3_campaign_close_v1",
            "campaign_status": verdict,
            "architecture_passed": bool(arch.get("passed")),
            "confirm_validation_report_sha256": report_sha,
            "validation_metrics_sha256": deep_sha,
            "git_revision": state.get("git_revision"),
            "report_builder_git_revision": report.get("report_builder_git_revision"),
            "policy_sha256": report["policy_sha256"],
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "forbid_lock": forbid_lock,
            "forbid_test": True,
            "campaign_closed": bool(state["campaign_closed"]),
        }
        marker_path = status_dir / f"{verdict}.json"
        marker_path.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        state["campaign_close_marker"] = marker_path.relative_to(_ROOT).as_posix()
    if deep_sha:
        state["gates"]["validation_metrics_sha256"] = deep_sha

    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "verdict": verdict,
                "architecture_passed": bool(arch.get("passed")),
                "report": out.as_posix(),
                "report_sha256": report_sha,
                "mean_b4_minus_b3_pp": arch.get("mean_b4_minus_b3_pp"),
                "mean_b4e_minus_b4_pp": mean_b4e_delta,
                "efficiency_accuracy_component_passed": efficiency_accuracy_ok,
            },
            indent=2,
        )
    )
    # Artifact always written; non-zero exit signals NO_GO for operators/CI.
    return 0 if arch.get("passed") else 3


if __name__ == "__main__":
    raise SystemExit(main())
