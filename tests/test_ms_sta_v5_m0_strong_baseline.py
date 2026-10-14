"""Adversarial / integrity tests for M0 strong-baseline canonicalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.ms_sta_v5_m0_strong_baseline import (
    StrongBaselineAuditError,
    assert_aligned_panels,
    assert_v1_lock_not_canonical,
    evaluate_canonical_gate,
    hierarchical_bootstrap_ci,
    identify_strongest_non_attention,
    sample_bootstrap_ci_lower,
    seed_averaged_sample_deltas,
    PredRow,
)

_ROOT = Path(__file__).resolve().parents[1]


def _row(sid: str, signer: str, correct: bool, top5: bool = True, label: int = 0) -> PredRow:
    return PredRow(sample_id=sid, signer=signer, label=label, correct=correct, top5_hit=top5)


def test_omitted_b3_mlp_baseline_rejected() -> None:
    with pytest.raises(StrongBaselineAuditError, match="non_attention_baseline_omitted"):
        identify_strongest_non_attention(
            {
                "rgb_fixed": {"mean_top1": 0.4},
                "b3_star_fixed": {"mean_top1": 0.5},
                # b3_mlp_roi_fixed omitted
            }
        )


def test_strongest_is_mlp_when_highest() -> None:
    best = identify_strongest_non_attention(
        {
            "rgb_fixed": {"mean_top1": 0.42},
            "b3_star_fixed": {"mean_top1": 0.57},
            "b3_mlp_roi_fixed": {"mean_top1": 0.59},
        }
    )
    assert best == "b3_mlp_roi_fixed"


def test_swapped_sample_ids_rejected() -> None:
    a = {"000001": _row("000001", "signer_1", True), "000002": _row("000002", "signer_2", False)}
    b = {"000001": _row("000001", "signer_1", True), "000003": _row("000003", "signer_2", False)}
    with pytest.raises(StrongBaselineAuditError, match="sample_id_mismatch"):
        assert_aligned_panels({"a": a, "b": b})


def test_swapped_signers_rejected() -> None:
    a = {"000001": _row("000001", "signer_1", True)}
    b = {"000001": _row("000001", "signer_2", True)}
    with pytest.raises(StrongBaselineAuditError, match="signer_mismatch"):
        assert_aligned_panels({"a": a, "b": b})


def test_provisional_v1_lock_not_canonical() -> None:
    assert_v1_lock_not_canonical(
        {
            "schema": "ms_sta_v5_m0_validation_lock_v1",
            "status": "LOCKED",
            "immutable": True,
        }
    )
    with pytest.raises(StrongBaselineAuditError, match="provisional_v1_lock"):
        assert_v1_lock_not_canonical(
            {
                "schema": "ms_sta_v5_m0_validation_lock_v1",
                "status": "CANONICAL",
                "strong_baseline_variant": "b3_mlp_roi_fixed",
            }
        )
    with pytest.raises(StrongBaselineAuditError, match="provisional_v1_lock"):
        assert_v1_lock_not_canonical(
            {
                "schema": "ms_sta_v5_m0_validation_lock_v1",
                "canonical_strong_baseline": True,
            }
        )


def test_hierarchical_differs_from_flat_sample_bootstrap() -> None:
    # Unbalanced signers: flat bootstrap ignores cluster structure.
    deltas = [0.2] * 90 + [-0.1] * 10
    signers = ["signer_a"] * 90 + ["signer_b"] * 10
    _mean, h_lo, _h_hi = hierarchical_bootstrap_ci(deltas, signers, n_boot=2000, seed=42)
    flat_lo = sample_bootstrap_ci_lower(deltas, n_boot=2000, seed=42)
    # Adversarial: substituting flat for hierarchical must be detectable as different.
    assert h_lo != pytest.approx(flat_lo, abs=1e-12)


def test_canonical_gate_requires_all_checks() -> None:
    ok = evaluate_canonical_gate(
        primary_seed_top1=[0.65, 0.64, 0.65],
        baseline_seed_top1=[0.58, 0.59, 0.60],
        primary_seed_top5=[0.91, 0.90, 0.90],
        baseline_seed_top5=[0.87, 0.88, 0.88],
        primary_seed_f1=[0.63, 0.63, 0.64],
        baseline_seed_f1=[0.57, 0.58, 0.59],
        ci_low_top1_pp=3.5,
        signer_deltas_pp={"signer_1": 5.0, "signer_2": 4.0, "signer_14": 3.0},
    )
    assert ok["verdict"] == "VALIDATION_PASS"

    bad = evaluate_canonical_gate(
        primary_seed_top1=[0.58, 0.64, 0.65],  # seed0 not greater
        baseline_seed_top1=[0.58, 0.59, 0.60],
        primary_seed_top5=[0.91, 0.90, 0.90],
        baseline_seed_top5=[0.87, 0.88, 0.88],
        primary_seed_f1=[0.63, 0.63, 0.64],
        baseline_seed_f1=[0.57, 0.58, 0.59],
        ci_low_top1_pp=3.5,
        signer_deltas_pp={"signer_1": 5.0},
    )
    assert bad["verdict"] == "VALIDATION_NO_GO"
    assert bad["checks"]["b4_gt_strong_baseline_every_seed"] is False


def test_seed_averaged_deltas_shape() -> None:
    ids = ["a", "b"]
    primary = {
        42: {"a": _row("a", "s1", True), "b": _row("b", "s2", False)},
        123: {"a": _row("a", "s1", True), "b": _row("b", "s2", True)},
        456: {"a": _row("a", "s1", False), "b": _row("b", "s2", True)},
    }
    baseline = {
        42: {"a": _row("a", "s1", False), "b": _row("b", "s2", False)},
        123: {"a": _row("a", "s1", True), "b": _row("b", "s2", False)},
        456: {"a": _row("a", "s1", False), "b": _row("b", "s2", True)},
    }
    d1, d5, signers = seed_averaged_sample_deltas(ids, primary, baseline)
    assert len(d1) == 2 and len(d5) == 2
    assert signers == ["s1", "s2"]
    # a: (1-0)+(1-1)+(0-0)/3 = 1/3
    assert d1[0] == pytest.approx(1.0 / 3.0)


def test_missing_checkpoint_sha_rejected(tmp_path: Path) -> None:
    from src.eval.ms_sta_v5_m0_strong_baseline import require_checkpoint

    with pytest.raises(StrongBaselineAuditError, match="checkpoint_missing"):
        require_checkpoint(tmp_path, "no/such/best.pt")
    with pytest.raises(StrongBaselineAuditError, match="checkpoint_rel_missing"):
        require_checkpoint(tmp_path, None)


def test_missing_seed_panel_fails_confirm_bundle(tmp_path: Path) -> None:
    from src.eval.ms_sta_v5_m0_strong_baseline import load_confirm_bundle

    runs = tmp_path / "Docs/results/ms_sta_v5_m0_runs"
    runs.mkdir(parents=True)
    (runs / "CONFIRM_RESULTS.json").write_text(
        json.dumps({"trials": [{"variant": "rgb_fixed", "seed": 42, "best_val_top1": 0.1}]}),
        encoding="utf-8",
    )
    with pytest.raises(StrongBaselineAuditError, match="confirm_trials_count_invalid"):
        load_confirm_bundle(tmp_path)


@pytest.mark.requires_artifacts
def test_live_audit_binds_and_preserves_v1() -> None:
    from src.eval.ms_sta_v5_m0_strong_baseline import run_strong_baseline_audit, sha256_file

    v1 = _ROOT / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK.json"
    if not v1.is_file():
        pytest.skip("v1 lock missing")
    before = sha256_file(v1)
    audit = run_strong_baseline_audit(_ROOT)
    assert sha256_file(v1) == before
    assert audit["strongest_non_attention_baseline"] == "b3_mlp_roi_fixed"
    assert audit["n_val_samples"] == 2629
    assert audit["canonical_gate_against_strong_baseline"]["verdict"] == "VALIDATION_PASS"
    assert audit["bootstrap"]["method"] == "paired_hierarchical_signer_then_sample"
    # v1 gate must remain provisional reference, not replaced.
    assert Path(_ROOT / "Docs/results/ms_sta_v5_m0/ACCURACY_GATE.json").is_file()
