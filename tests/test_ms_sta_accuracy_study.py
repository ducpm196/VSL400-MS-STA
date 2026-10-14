from __future__ import annotations

import json
from pathlib import Path

from scripts.canonicalize_ms_sta_accuracy import _mean_std
from src.deployment.backend_api import _load_accuracy_study

_ROOT = Path(__file__).resolve().parents[1]


def test_accuracy_screen_is_bounded_and_test_blind() -> None:
    policy = json.loads(
        (_ROOT / "Docs/experiments/ms_sta_accuracy/SCREEN_POLICY.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["status"] == "PREDECLARED_PILOT_ONLY"
    assert policy["claim_allowed"] is False
    assert policy["scope"]["test_access"] == "forbidden"
    assert policy["search_cap"]["maximum_candidates"] == 2
    assert len(policy["candidates"]) == 2


def test_accuracy_pilot_selection_matches_predeclared_candidates() -> None:
    selection = json.loads(
        (_ROOT / "Docs/results/ms_sta_accuracy_policy/PILOT_SELECTION.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["status"] == "PASS"
    assert selection["claim_allowed"] is False
    assert selection["selected_candidate"] == "lr3e4"
    assert selection["test_access"] == "forbidden"
    assert selection["locked_e2_d4_unchanged"] is True


def test_accuracy_canonical_policy_is_frozen_before_three_seed_run() -> None:
    policy = json.loads(
        (_ROOT / "Docs/results/ms_sta_accuracy_policy/ACCURACY_POLICY.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["status"] == "FROZEN"
    assert policy["claim_allowed"] is False
    assert policy["train_hyperparameters"]["lr"] == 0.0003
    assert policy["model_overrides"]["skel_temporal_stride"] == 1
    assert policy["seeds"]["list"] == [42, 123, 456]
    assert policy["decision_rule_predeclared"]["mean_top1_delta_pp_min"] == 2.0
    assert policy["evaluation"]["test_access"] == "forbidden"
    assert policy["code_lock"]["require_clean_worktree"] is True


def test_accuracy_summary_uses_sample_std() -> None:
    assert _mean_std([1.0, 2.0, 3.0]) == {"mean": 2.0, "std": 1.0, "n": 3}


def test_accuracy_final_report_passes_predeclared_gate() -> None:
    report = json.loads(
        (_ROOT / "Docs/results/ms_sta_accuracy/FINAL_REPORT.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["claim_allowed"] is True
    assert report["canonicalizer_git_dirty"] is False
    assert report["evidence_git_revision"] == "18d515dcf24a8e5201e1318e71b8bae745b5d176"
    result = report["validation_result"]
    assert result["decision_checks"]["pass"] is True
    assert result["top1"]["mean"] > 60.0
    assert result["paired_delta_top1_pp"]["mean"] > 8.0
    assert all(row["delta_top1_pp"] > 0 for row in result["per_seed"])
    assert report["change"]["architecture_change"] == "none"
    assert "No test/E2 access" in report["limits"][0]


def test_demo_accuracy_payload_is_validation_only() -> None:
    payload = _load_accuracy_study(
        _ROOT / "Docs/results/ms_sta_accuracy/FINAL_REPORT.json"
    )
    assert payload["ready"] is True
    assert payload["scope"] == "validation_only_auxiliary"
    assert abs(payload["optimized_top1_mean_pct"] - 60.012845215157355) < 1e-9
    assert abs(payload["paired_delta_top1_mean_pp"] - 8.053949903660884) < 1e-9
    assert payload["seeds"] == [42, 123, 456]
    assert payload["demo_engine_uses_optimized_checkpoint"] is False


def test_demo_surface_suppresses_historical_accuracy_outside_active_campaign() -> None:
    payload = _load_accuracy_study(None)
    assert payload.get("ready") is False
    assert payload.get("suppressed") is True
    assert "not part of the active campaign" in str(payload.get("reason") or "")
