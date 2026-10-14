"""Canonicalize the locked validation matrix and GPU benchmark.

This script never loads test predictions. It opens only the narrowly scoped
Skeleton temporal efficiency claim when every predeclared gate passes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import git_dirty, git_rev, load_json, sha256_file


def _fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize MS-STA efficiency evidence")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("Docs/results/ms_sta_efficiency_policy/EFFICIENCY_POLICY.json"),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("logs/ms_sta_efficiency_v1/MATRIX_AGGREGATE.json"),
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("logs/ms_sta_efficiency_v1/BENCHMARK.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Docs/results/ms_sta_efficiency"),
    )
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else _ROOT / path

    policy_path, matrix_path, benchmark_path = map(
        resolve, (args.policy, args.matrix, args.benchmark)
    )
    for path in (policy_path, matrix_path, benchmark_path):
        if not path.is_file():
            return _fail(f"missing input: {path}")
    policy = load_json(policy_path)
    matrix = load_json(matrix_path)
    benchmark = load_json(benchmark_path)

    if policy.get("status") != "FROZEN":
        return _fail("policy status is not FROZEN")
    if matrix.get("status") != "VALIDATION_PASS" or matrix.get("checks", {}).get("pass") is not True:
        return _fail("validation matrix did not pass")
    if matrix.get("git_dirty") is not False or benchmark.get("git_dirty") is not False:
        return _fail("matrix/benchmark provenance is dirty")
    evidence_revision = str(matrix.get("git_revision") or "")
    if not evidence_revision or benchmark.get("git_revision") != evidence_revision:
        return _fail("matrix and benchmark git revisions differ")
    if matrix.get("policy_sha256") != sha256_file(policy_path):
        return _fail("matrix policy SHA mismatch")
    if matrix.get("split") != "validation_only" or matrix.get("test_access") != "forbidden_and_not_used":
        return _fail("matrix test isolation contract failed")
    if benchmark.get("protocol", {}).get("input_split") != "frozen E1 validation":
        return _fail("benchmark did not use frozen validation input")

    reductions = benchmark.get("reductions_pct") or {}
    expected_score = float(
        policy["decision_rule_predeclared"]["efficiency"][
            "skeleton_self_attention_score_reduction_pct_exact"
        ]
    )
    if float(reductions.get("skeleton_self_attention_score_elements", -1)) != expected_score:
        return _fail("score-element reduction mismatch")
    latency_reduction = float(reductions.get("model_forward_median_latency", -1))
    min_latency = float(
        policy["decision_rule_predeclared"]["efficiency"][
            "whole_model_median_forward_latency_reduction_pct_min"
        ]
    )
    if latency_reduction <= min_latency:
        return _fail("model-forward latency did not improve")
    variants = benchmark.get("variants") or {}
    if variants.get("1", {}).get("parameter_count") != variants.get("4", {}).get(
        "parameter_count"
    ):
        return _fail("parameter count changed")
    rows = matrix.get("rows", [])
    if sorted(int(row["seed"]) for row in rows) != [42, 123, 456]:
        return _fail("matrix seeds are incomplete")
    if any(int(row.get("n", -1)) != 2595 for row in rows):
        return _fail("matrix validation sample counts differ")
    accuracy_rule = policy["decision_rule_predeclared"]["accuracy"]
    if matrix.get("decision_rule") != accuracy_rule:
        return _fail("matrix decision rule differs from frozen policy")
    delta_top1 = [float(row["delta_top1_pp"]) for row in rows]
    delta_top5 = [float(row["delta_top5_pp"]) for row in rows]
    if statistics.mean(delta_top1) < float(accuracy_rule["mean_top1_delta_vs_b4_pp_min"]):
        return _fail("recomputed mean Top-1 gate failed")
    if min(delta_top1) < float(accuracy_rule["each_seed_top1_delta_vs_b4_pp_min"]):
        return _fail("recomputed per-seed Top-1 gate failed")
    if statistics.mean(delta_top5) < float(accuracy_rule["mean_top5_delta_vs_b4_pp_min"]):
        return _fail("recomputed mean Top-5 gate failed")

    expected_checkpoint_sha = policy["lineage"]["base_b4_checkpoint_sha256_by_seed"]["42"]
    if benchmark.get("checkpoint_sha256") != expected_checkpoint_sha:
        return _fail("benchmark checkpoint SHA differs from frozen B4 seed 42")
    protocol = benchmark.get("protocol", {})
    if (
        int(protocol.get("batch_size", -1)) != 32
        or int(protocol.get("warmup_per_round", -1)) < 30
        or int(protocol.get("repeats_per_round", -1)) < 100
        or int(protocol.get("rounds", -1)) < 4
        or benchmark.get("device", {}).get("dtype") != "float32"
    ):
        return _fail("benchmark protocol differs from frozen minimum")
    if variants.get("1", {}).get("skeleton_tokens") != 64 or variants.get("4", {}).get(
        "skeleton_tokens"
    ) != 16:
        return _fail("benchmark token counts differ")
    recomputed_latency_reduction = 100.0 * (
        1.0
        - float(variants["4"]["median_latency_ms"])
        / float(variants["1"]["median_latency_ms"])
    )
    if not math.isclose(latency_reduction, recomputed_latency_reduction, abs_tol=1e-9):
        return _fail("benchmark latency reduction was not reproducible from raw values")
    if git_dirty(_ROOT):
        return _fail("canonicalizer requires a clean worktree")

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_copy = out_dir / "MATRIX_AGGREGATE.json"
    benchmark_copy = out_dir / "BENCHMARK.json"
    matrix_copy.write_bytes(matrix_path.read_bytes())
    benchmark_copy.write_bytes(benchmark_path.read_bytes())

    final = {
        "schema": "ms_sta_efficiency_final_report_v1",
        "status": "PASS",
        "claim_allowed": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonicalizer_git_revision": git_rev(_ROOT),
        "canonicalizer_git_dirty": False,
        "evidence_git_revision": evidence_revision,
        "scope": "auxiliary validation and model-forward efficiency study; locked B4 E2/D4 unchanged",
        "policy": {
            "path": policy_path.relative_to(_ROOT).as_posix(),
            "sha256": sha256_file(policy_path),
        },
        "inputs": {
            "matrix": matrix_copy.relative_to(_ROOT).as_posix(),
            "matrix_sha256": sha256_file(matrix_copy),
            "benchmark": benchmark_copy.relative_to(_ROOT).as_posix(),
            "benchmark_sha256": sha256_file(benchmark_copy),
            "base_b4_aggregate": policy["lineage"]["base_b4_aggregate"],
            "base_b4_aggregate_sha256": policy["lineage"]["base_b4_aggregate_sha256"],
        },
        "architecture": {
            "stream": "Skeleton",
            "operation": "parameter-free avg_pool1d",
            "stride": 4,
            "tokens_before": 64,
            "tokens_after": 16,
            "placement": policy["model_overrides"]["placement"],
            "parameter_count_delta": 0,
        },
        "theoretical_result": {
            "metric": "Skeleton temporal self-attention score elements",
            "before_per_head_per_sample": 4096,
            "after_per_head_per_sample": 256,
            "reduction_pct": expected_score,
        },
        "validation_result": {
            "n_per_seed": 2595,
            "seeds": [42, 123, 456],
            "stride4_top1_mean_pct": matrix["stride4_top1"]["mean"],
            "stride4_top1_std_pct": matrix["stride4_top1"]["std"],
            "paired_delta_top1_mean_pp": matrix["paired_seed_delta_top1_pp"]["mean"],
            "paired_delta_top1_std_pp": matrix["paired_seed_delta_top1_pp"]["std"],
            "stride4_top5_mean_pct": matrix["stride4_top5"]["mean"],
            "stride4_top5_std_pct": matrix["stride4_top5"]["std"],
            "paired_delta_top5_mean_pp": matrix["paired_seed_delta_top5_pp"]["mean"],
            "paired_delta_top5_std_pp": matrix["paired_seed_delta_top5_pp"]["std"],
            "decision_checks": matrix["checks"],
            "interpretation": (
                "Accuracy preserved under the predeclared validation rule; "
                "not a consistent accuracy-improvement claim."
            ),
            "per_seed": matrix["rows"],
        },
        "benchmark_result": {
            "device": benchmark["device"],
            "protocol": benchmark["protocol"],
            "baseline_stride1": variants["1"],
            "stride4": variants["4"],
            "median_model_forward_latency_reduction_pct": latency_reduction,
            "peak_incremental_vram_reduction_pct": reductions[
                "model_forward_peak_incremental_vram"
            ],
        },
        "allowed_wording": (
            "With Skeleton T=64, parameter-free stride-4 downsampling reduces Skeleton "
            "temporal self-attention score elements by 93.75%. On the locked RTX 2060 "
            "model-forward benchmark it reduces median latency by the measured value "
            "reported here, while the three-seed validation gate passes."
        ),
        "forbidden_wording": policy["forbidden_claims"],
        "limits": [
            "No test/E2 access; this is validation-only auxiliary evidence.",
            "Seed 123 Top-1 decreased by 0.27 pp; do not claim consistent accuracy improvement.",
            "The 93.75% figure is not whole-model FLOPs, latency, or VRAM.",
            "Peak allocated VRAM did not decrease in the locked benchmark.",
            "The stride-4 model does not replace the B4 defense checkpoint or demo engine.",
            "Benchmark excludes feature extraction, disk I/O, API, and UI.",
        ],
    }
    report_path = out_dir / "FINAL_REPORT.json"
    report_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = f"""# MS-STA Skeleton Temporal Efficiency - Canonical Result

**Status:** PASS (auxiliary validation study; B4 E2/D4 unchanged).

| Evidence | Result |
|---|---:|
| Skeleton tokens before -> after | 64 -> 16 |
| Skeleton self-attention score elements | 4,096 -> 256 (**-{expected_score:.2f}%**) |
| B4 stride-4 validation Top-1 | **{matrix['stride4_top1']['mean']:.2f} +/- {matrix['stride4_top1']['std']:.2f}%** |
| Paired seed delta Top-1 vs original B4 | **{matrix['paired_seed_delta_top1_pp']['mean']:+.2f} +/- {matrix['paired_seed_delta_top1_pp']['std']:.2f} pp** |
| Median model-forward latency | {variants['1']['median_latency_ms']:.2f} -> {variants['4']['median_latency_ms']:.2f} ms (**-{latency_reduction:.2f}%**) |
| Peak allocated VRAM | {variants['1']['peak_total_mb_median']:.2f} -> {variants['4']['peak_total_mb_median']:.2f} MB (**no observed reduction**) |
| Parameter count | {variants['1']['parameter_count']:,} -> {variants['4']['parameter_count']:,} |

## Claim boundary

The 93.75% result applies only to the number of score elements in the Skeleton temporal self-attention matrix at fixed T=64->16. It is not a claim about whole-model FLOPs, latency, VRAM, real-time behavior, or test accuracy.

The measured latency result covers model forward only on the locked RTX 2060 protocol. The study does not access E2 and does not replace the defense B4 checkpoint. Seed 123 decreased by 0.27 pp, so the accuracy conclusion is preservation under the predeclared gate, not consistent improvement.
"""
    md_path = out_dir / "REPORT.md"
    md_path.write_text(markdown, encoding="utf-8")
    checksums = {
        "schema": "ms_sta_efficiency_checksums_v1",
        "FINAL_REPORT.json": sha256_file(report_path),
        "REPORT.md": sha256_file(md_path),
        "MATRIX_AGGREGATE.json": sha256_file(matrix_copy),
        "BENCHMARK.json": sha256_file(benchmark_copy),
    }
    checksum_path = out_dir / "CHECKSUMS.json"
    checksum_path.write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "claim_allowed": True,
                "score_reduction_pct": expected_score,
                "latency_reduction_pct": latency_reduction,
                "validation_delta_top1_pp": matrix["paired_seed_delta_top1_pp"]["mean"],
            },
            indent=2,
        )
    )
    print(f"[PASS] canonical report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
