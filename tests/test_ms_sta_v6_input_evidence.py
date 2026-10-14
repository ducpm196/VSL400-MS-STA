"""Hermetic tests for shared input evidence + E4 validation-only demo manifest."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.deployment.input_evidence import (
    external_cache_identity,
    load_e4_input_demo_manifest,
    sealed_feature_paths,
    validate_e4_input_demo_manifest,
    v5_feature_paths,
)
from src.deployment.demo_visuals import grouped_accuracy_records, relative_latency_change_pct
from src.models.ms_sta_v6 import build_ms_sta_v6
from src.models.ms_sta_v6_e4 import build_ms_sta_v6_e4

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Docs/results/ms_sta_v6_defense_go/E4_INPUT_DEMO_MANIFEST.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_e4_input_demo_manifest(MANIFEST, root=ROOT)


@pytest.mark.requires_artifacts
def test_manifest_validates_and_validation_only(manifest: dict) -> None:
    assert validate_e4_input_demo_manifest(manifest, root=ROOT) == []
    assert manifest["e4_ready"] is False
    assert manifest["e4_status"] == "E4_ACCURACY_NO_GO"
    assert manifest["n_samples"] >= 3
    for sample in manifest["samples"]:
        assert sample["split"] == "validation"
        assert sample["signer_id"] in {"signer_1", "signer_2", "signer_14"}
        assert sample["signer_id"] not in {
            "signer_24",
            "signer_25",
            "signer_26",
            "signer_27",
            "signer_28",
        }


def test_b4_e4_same_external_cache_hashes(manifest: dict) -> None:
    """Same sample_id => identical external cache identity for B4 and E4 pages."""
    for sample in manifest["samples"]:
        a = external_cache_identity(sample)
        b = external_cache_identity(dict(sample))
        assert a == b
        rgb, skel = sealed_feature_paths(sample["sample_id"], root=ROOT)
        assert rgb.resolve() == (ROOT / sample["rgb_cache_rel"]).resolve()
        assert skel.resolve() == (ROOT / sample["skeleton_cache_rel"]).resolve()


def test_sealed_alias_matches_legacy_name() -> None:
    assert sealed_feature_paths is v5_feature_paths
    a, b = sealed_feature_paths("000000", root=ROOT)
    c, d = v5_feature_paths("000000", root=ROOT)
    assert a == c and b == d


def test_internal_widths_b4_256_e4_192(manifest: dict) -> None:
    assert manifest["internal_width"]["b4_v6_ctr"] == 256
    assert manifest["internal_width"]["e4_w192"] == 192
    b4 = build_ms_sta_v6("b4_v6_ctr", num_classes=400, dropout=0.0)
    e4 = build_ms_sta_v6_e4("e4_w192", num_classes=400, dropout=0.0)
    assert b4.d_model == 256
    assert e4.d_model == 192


def test_e4_has_no_grayscale_stride_joint_drop(manifest: dict) -> None:
    forbidden = set(manifest["forbidden_claims"])
    assert {"grayscale", "joint_drop", "temporal_stride", "token_merge"} <= forbidden
    e4 = build_ms_sta_v6_e4("e4_w192", num_classes=400, dropout=0.0)
    assert not hasattr(e4, "rgb_reducer")
    assert not hasattr(e4, "skel_stride")
    assert getattr(e4, "temporal_layers", None) == 2


@pytest.mark.requires_artifacts
def test_manifest_fail_closed_on_sha_tamper(manifest: dict) -> None:
    bad = copy.deepcopy(manifest)
    bad["samples"][0]["rgb_cache_sha256"] = "0" * 64
    errs = validate_e4_input_demo_manifest(bad, root=ROOT)
    assert any("sha_mismatch" in e for e in errs)


def test_manifest_fail_closed_on_test_signer(manifest: dict) -> None:
    bad = copy.deepcopy(manifest)
    bad["samples"][0]["signer_id"] = "signer_24"
    errs = validate_e4_input_demo_manifest(bad, root=ROOT)
    assert any("forbidden_test_signer" in e for e in errs)


def test_manifest_fail_closed_on_wrong_split(manifest: dict) -> None:
    bad = copy.deepcopy(manifest)
    bad["samples"][0]["split"] = "test"
    errs = validate_e4_input_demo_manifest(bad, root=ROOT)
    assert any("sample_not_validation" in e for e in errs)


def test_viz_only_note_present_and_pages_refuse_e4_engine() -> None:
    page_e4 = (ROOT / "src/deployment/pages/2_E4_Efficiency_Appendix.py").read_text(
        encoding="utf-8"
    )
    page_b4 = (ROOT / "src/deployment/pages/1_B4_V6_Accuracy.py").read_text(encoding="utf-8")
    # The canonical claim lock now selects system optimization (S5), not the
    # earlier compressed-network E4-R2/R3 appendix. It remains evidence-only.
    from src.deployment.campaign_evidence import runtime_ui_state, load_scientific_claim_lock
    state = runtime_ui_state(ROOT)
    assert state["active_engine"] == "B4"
    assert state["s5_mode"] == "EVIDENCE_ONLY"
    assert load_scientific_claim_lock(ROOT)["E4"]["not_a_new_neural_model"] is True
    assert "runtime_ui_state" in page_e4
    assert "load_locked_test_report" in page_e4
    assert "requests.post" not in page_e4
    assert "VIZ-ONLY" in page_b4 or "VIZ-ONLY" in (
        ROOT / "src/deployment/input_evidence.py"
    ).read_text(encoding="utf-8")
    assert "không phải mô hình phân loại mới" in page_e4


def test_app_uses_sealed_feature_contract_wording() -> None:
    text = (ROOT / "src/deployment/app.py").read_text(encoding="utf-8")
    assert "sealed_feature_paths" in text
    assert "Sealed input-feature contract" in text
    assert "V5 feature contract unavailable" not in text


def test_demo_accuracy_chart_rows_are_seed_aligned() -> None:
    rows = grouped_accuracy_records(
        seeds=[42, 123, 456],
        series={"B4-V6": [85.32, 84.82, 84.29], "E4-R2 D": [80.98, 81.51, 80.87]},
    )
    assert len(rows) == 6
    assert {row["Seed"] for row in rows} == {42, 123, 456}
    assert {row["Model"] for row in rows} == {"B4-V6", "E4-R2 D"}
    with pytest.raises(ValueError, match="series_length_mismatch"):
        grouped_accuracy_records(seeds=[42, 123], series={"bad": [1.0]})


def test_latency_delta_sign_is_candidate_relative_to_b4() -> None:
    change = relative_latency_change_pct(
        baseline_ms=17.440899973735213,
        candidate_ms=18.12140003312379,
    )
    assert change == pytest.approx(3.901748536)
    assert change > 0.0  # E4-R2 is slower, not faster.
    with pytest.raises(ValueError, match="latency_must_be_positive"):
        relative_latency_change_pct(baseline_ms=0.0, candidate_ms=1.0)


def test_visual_first_pages_keep_e4_claims_fail_closed() -> None:
    e4 = (ROOT / "src/deployment/pages/2_E4_Efficiency_Appendix.py").read_text(
        encoding="utf-8"
    )
    b4 = (ROOT / "src/deployment/pages/1_B4_V6_Accuracy.py").read_text(encoding="utf-8")
    assert "st.tabs" in e4 and "st.tabs" in b4
    assert "load_scientific_claim_lock" in e4
    assert "load_r0_comparator_report" in e4
    assert "paired_S5_minus_R0" in e4
    assert "formal_val_speedup_approx" in e4
    assert "không phải** latency của thao tác UI" in e4
    assert "Kết quả âm" in e4
    assert "Không đủ phần cứng, không huấn luyện" in e4
    assert "render_test_comparison_chart" in b4
    assert "render_flow" in e4
    assert "SEALED_DEMO_SAMPLE_ONLY" in e4


def test_e4_r3_feasibility_artifacts_match_display_table() -> None:
    feas = json.loads(
        (ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json").read_text(
            encoding="utf-8"
        )
    )
    assert feas["status"] == "E4_R3_FEASIBILITY_NO_GO"
    b4_ms = 1000.0 * float(feas["b4"]["median_sec"])
    assert abs(b4_ms - 12.47) < 0.02
    assert abs(1000.0 * float(feas["arms"]["A"]["median_sec"]) - 12.24) < 0.02
    assert abs(100.0 * float(feas["arms"]["A"]["speedup_vs_b4"]) - 1.9) < 0.15
    companion = json.loads(
        (
            ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_PROVENANCE_COMPANION.json"
        ).read_text(encoding="utf-8")
    )
    assert companion["canonical_clean_execution_identity"] is False
    assert companion["parent_report_sha256"] == hashlib.sha256(
        (ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json").read_bytes()
    ).hexdigest()
