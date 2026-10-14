from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.benchmark_ms_sta_efficiency import _percent_reduction
from scripts.run_ms_sta_efficiency_matrix import _metric_summary
from src.deployment.backend_api import _load_efficiency_ablation

_ROOT = Path(__file__).resolve().parents[1]


def test_stride4_score_reduction_is_exactly_9375_pct() -> None:
    assert _percent_reduction(64**2, 16**2) == 93.75


def test_efficiency_pilot_policy_is_nonclaim_and_predeclared() -> None:
    policy = json.loads(
        (
            _ROOT / "Docs" / "experiments" / "ms_sta_efficiency" / "PILOT_POLICY.json"
        ).read_text(encoding="utf-8")
    )
    assert policy["status"] == "PILOT_ONLY"
    assert policy["claim_allowed"] is False
    change = policy["change_under_test"]
    assert change["learned_parameters_added"] == 0
    assert change["canonical_input_tokens"] == 64
    assert change["output_tokens"] == 16
    assert change["skeleton_self_attention_score_reduction_pct"] == 93.75


def test_efficiency_canonical_policy_is_frozen_but_claim_locked() -> None:
    policy = json.loads(
        (
            _ROOT
            / "Docs"
            / "results"
            / "ms_sta_efficiency_policy"
            / "EFFICIENCY_POLICY.json"
        ).read_text(encoding="utf-8")
    )
    assert policy["status"] == "FROZEN"
    assert policy["claim_allowed"] is False
    assert policy["code_lock"]["require_clean_worktree"] is True
    assert policy["model_overrides"]["skel_temporal_stride"] == 4
    assert policy["evaluation"]["test_access"] == "forbidden"
    assert policy["decision_rule_predeclared"]["pass_requires_all"] is True


def test_efficiency_metric_summary_uses_sample_std() -> None:
    summary = _metric_summary([1.0, 2.0, 3.0])
    assert summary == {"mean": 2.0, "std": 1.0, "n": 3}


def test_efficiency_canonical_report_is_self_contained_and_claim_limited() -> None:
    result_dir = _ROOT / "Docs" / "results" / "ms_sta_efficiency"
    report = json.loads((result_dir / "FINAL_REPORT.json").read_text(encoding="utf-8"))
    checksums = json.loads((result_dir / "CHECKSUMS.json").read_text(encoding="utf-8"))

    assert report["status"] == "PASS"
    assert report["claim_allowed"] is True
    assert report["canonicalizer_git_dirty"] is False
    assert report["canonicalizer_git_revision"] == "939895c20eb96f61b3e2792889cd4e19ae518fe7"
    assert report["theoretical_result"]["reduction_pct"] == 93.75
    assert report["architecture"]["tokens_before"] == 64
    assert report["architecture"]["tokens_after"] == 16
    assert report["architecture"]["parameter_count_delta"] == 0
    assert report["validation_result"]["decision_checks"]["pass"] is True
    assert report["benchmark_result"]["median_model_forward_latency_reduction_pct"] > 0
    assert report["benchmark_result"]["peak_incremental_vram_reduction_pct"] == 0
    assert "No test/E2 access" in report["limits"][0]

    for name in (
        "FINAL_REPORT.json",
        "REPORT.md",
        "MATRIX_AGGREGATE.json",
        "BENCHMARK.json",
    ):
        artifact = result_dir / name
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == checksums[name]

    assert report["inputs"]["matrix"] == (
        "Docs/results/ms_sta_efficiency/MATRIX_AGGREGATE.json"
    )
    assert report["inputs"]["benchmark"] == (
        "Docs/results/ms_sta_efficiency/BENCHMARK.json"
    )


def test_demo_efficiency_payload_is_derived_from_locked_report() -> None:
    payload = _load_efficiency_ablation(
        _ROOT / "Docs" / "results" / "ms_sta_efficiency" / "FINAL_REPORT.json"
    )
    assert payload["ready"] is True
    assert payload["scope"] == "validation_only_auxiliary"
    assert abs(payload["base_top1_mean_pct"] - 51.95889531149647) < 1e-9
    assert abs(payload["stride4_top1_mean_pct"] - 52.42132305716121) < 1e-9
    assert payload["score_elements_before"] == 4096
    assert payload["score_elements_after"] == 256
    assert payload["score_reduction_pct"] == 93.75
    assert payload["latency_reduction_pct"] > 0
    assert payload["demo_engine_uses_stride4"] is False


def test_demo_surface_suppresses_historical_efficiency_outside_active_campaign() -> None:
    payload = _load_efficiency_ablation(None)
    assert payload.get("ready") is False
    assert payload.get("suppressed") is True
    assert "not part of the active campaign" in str(payload.get("reason") or "")


def test_demo_efficiency_payload_fails_closed_on_incomplete_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"status":"PASS","claim_allowed":true}', encoding="utf-8")
    assert _load_efficiency_ablation(report) == {
        "ready": False,
        "scope": "validation_only_auxiliary",
    }
