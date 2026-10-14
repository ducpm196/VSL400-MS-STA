"""Canonicalize retrain-v3 aggregates from measurement artifacts (no historical merge)."""

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
    evaluate_efficiency_gate,
    load_policy,
    require_complete_phase_rows,
    require_frozen_v3_policy,
    sha256_file,
)

DEFAULT_POLICY = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3_policy" / "RETRAIN_POLICY.json"


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
        "--efficiency-benchmark",
        type=Path,
        default=None,
        help="Path to EFFICIENCY_BENCHMARK.json (required to set efficiency gate)",
    )
    args = ap.parse_args()
    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    status_dir = _ROOT / str((policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3")
    state_path = status_dir / "RUN_STATE.json"
    if not state_path.is_file():
        print("[FAIL] missing RUN_STATE.json")
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    confirm = (state.get("phases") or {}).get("confirm") or {}
    runs = confirm.get("runs") or []
    try:
        require_complete_phase_rows(runs, expected_n=21, phase="confirm")
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    by_variant: dict[str, list[dict]] = {}
    for r in runs:
        if r.get("exit_code") != 0:
            continue
        by_variant.setdefault(str(r["variant"]), []).append(r)

    summary = {
        "schema": "ms_sta_retrain_v3_canonical_aggregate_v1",
        "path": "B",
        "test_claim_label": policy.get("test_claim_label"),
        "policy_sha256": sha256_file(policy_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variants": {},
        "historical_metrics_merged": False,
        "reuse_metrics": False,
        "reuse_checkpoints": False,
        "reuse_vsl_weights": False,
        "note": (
            "Validation reported as mean ± sample std over seeds 42/123/456. "
            "Do not cherry-pick seeds. Final checkpoints carry lifecycle source/final SHA."
        ),
    }
    for variant, rows in sorted(by_variant.items()):
        tops = [float(r["best_val_top1"]) for r in rows if r.get("best_val_top1") is not None]
        top5s = [float(r["best_val_top5"]) for r in rows if r.get("best_val_top5") is not None]
        m1, s1 = _mean_std(tops)
        m5, s5 = _mean_std(top5s)
        summary["variants"][variant] = {
            "n": len(rows),
            "seeds": [r.get("seed") for r in rows],
            "top1_mean": m1,
            "top1_sample_std": s1,
            "top5_mean": m5,
            "top5_sample_std": s5,
            "config_id": rows[0].get("config_id") if rows else None,
            "rows": [
                {
                    "seed": r.get("seed"),
                    "best_val_top1": r.get("best_val_top1"),
                    "best_val_top5": r.get("best_val_top5"),
                    "checkpoint_sha256": r.get("checkpoint_sha256"),
                    "final_checkpoint": r.get("final_checkpoint"),
                    "final_checkpoint_sha256": r.get("final_checkpoint_sha256"),
                    "source_candidate_sha256": r.get("source_candidate_sha256"),
                    "checkpoint_status": r.get("checkpoint_status")
                    or (
                        "final_selected_checkpoint"
                        if r.get("final_checkpoint_sha256")
                        else "candidate"
                    ),
                }
                for r in rows
            ],
        }

    # Efficiency is optional for validation-only canonicalize; required only when
    # operator wants gates.efficiency_passed for a future lock after architecture PASS.
    bench_path = args.efficiency_benchmark
    if bench_path is None:
        default_bench = status_dir / "EFFICIENCY_BENCHMARK.json"
        bench_path = default_bench if default_bench.is_file() else None
    else:
        bench_path = bench_path if bench_path.is_absolute() else _ROOT / bench_path

    if bench_path is not None and Path(bench_path).is_file():
        measurement = json.loads(Path(bench_path).read_text(encoding="utf-8"))
        if "mean_delta_top1_pp" not in measurement:
            b4 = summary["variants"].get("b4") or {}
            b4e = summary["variants"].get("b4e") or {}
            if b4.get("top1_mean") is None or b4e.get("top1_mean") is None:
                print("[FAIL] cannot derive mean_delta_top1_pp without b4/b4e confirm means")
                return 2
            measurement = dict(measurement)
            measurement["mean_delta_top1_pp"] = float(b4e["top1_mean"]) - float(b4["top1_mean"])
        gate = evaluate_efficiency_gate(measurement)
        summary["efficiency_gate"] = gate
        summary["efficiency_benchmark_sha256"] = sha256_file(Path(bench_path))
        state.setdefault("gates", {})["efficiency_passed"] = bool(gate["passed"])
        state["gates"]["efficiency_measurement"] = gate
        state["gates"]["efficiency_benchmark"] = Path(bench_path).relative_to(_ROOT).as_posix()
        eff_passed = bool(gate["passed"])
        active = gate["active_variant"]
    else:
        summary["efficiency_gate"] = None
        summary["note_efficiency"] = (
            "EFFICIENCY_BENCHMARK.json absent; architecture/validation report is independent. "
            "Do not treat 93.75% score-element reduction as a performance claim."
        )
        eff_passed = None
        active = None

    out = status_dir / "CANONICAL_AGGREGATE.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state["canonical_aggregate"] = out.relative_to(_ROOT).as_posix()
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "out": out.as_posix(),
                "variants": list(summary["variants"]),
                "efficiency_passed": eff_passed,
                "active_variant": active,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
