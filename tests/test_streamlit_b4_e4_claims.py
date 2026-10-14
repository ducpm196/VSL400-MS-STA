# -*- coding: utf-8 -*-
"""Streamlit claim/evidence contracts for two-contribution thesis demo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.deployment.campaign_evidence import (
    forbidden_term_scan,
    load_json,
    load_promotion_authorization,
    load_r0_comparator_report,
    load_s5_efficiency_report,
    load_scientific_claim_lock,
    runtime_ui_state,
)

ROOT = Path(__file__).resolve().parents[1]


def test_claim_lock_exactly_two_contributions() -> None:
    claim = load_scientific_claim_lock(ROOT)
    assert claim["contributions_exactly"] == 2
    assert "B4" in claim and "E4" in claim
    assert claim["E4"]["not_a_new_neural_model"] is True
    assert abs(claim["B4"]["top1"] - 0.5154906731549067) < 1e-12
    assert abs(claim["E4"]["S5_online"]["top1"] - 0.4994322789943228) < 1e-12
    assert abs(claim["E4"]["R0_online"]["top1"] - 0.4989456609894566) < 1e-12
    assert abs(claim["B4"]["delta_top1_pp_vs_b3_mlp_ctr"] - 9.326845) < 1e-6
    assert claim["E4"]["accuracy_gate_pass"] is True


def test_runtime_ui_state_b4_active_s5_evidence_only() -> None:
    st = runtime_ui_state(ROOT)
    assert st["active_engine"] == "B4"
    assert st["capability"] == "SEALED_DEMO_SAMPLE_ONLY"
    assert st["s5_mode"] == "EVIDENCE_ONLY"
    assert "đơn lẻ" in st["limitation_banner_vi"]
    assert "dịch câu" in st["limitation_banner_vi"]
    assert st["selected"] == "S5"
    assert st["e4_ready"] is True
    assert st["demo_cutover_authorized"] is False


def test_promotion_authorization_verbatim_and_no_cutover() -> None:
    auth = load_promotion_authorization(ROOT)
    assert auth["user_authorization_received"] is True
    assert auth["demo_cutover"] == "NOT_AUTHORIZED"
    assert "promote S5" in auth["authorization_verbatim"]
    assert "demo cutover" in auth["authorization_verbatim"].lower() or "không cho phép demo cutover" in auth[
        "authorization_verbatim"
    ]
    assert auth["transition_applied"]["selected"] == "S5"
    assert auth["transition_applied"]["e4_ready"] is True
    assert auth["candidate_code_sha"].startswith("b52c23b")
    assert auth["policy_sha256"].startswith("83cd88f9")


def test_efficiency_loaded_from_canonical_json_not_hardcoded_ui() -> None:
    eff = load_s5_efficiency_report(ROOT)
    assert abs(float(eff["speedup_S5_vs_R0"]["speedup_factor"]) - 1.2202268930346734) < 1e-12
    assert abs(float(eff["arms"]["R0"]["median_e2e_ms"]) - 1198.0813001282513) < 1e-9
    assert abs(float(eff["arms"]["S5"]["median_e2e_ms"]) - 981.8512499332428) < 1e-9


def test_cross_path_separation_in_claim_lock() -> None:
    claim = load_scientific_claim_lock(ROOT)
    assert claim["forbid_cross_path_metric_mixing"] is True
    assert claim["B4"]["canonical_path_class"] == "cache_backed_feature_npz"
    assert claim["E4"]["online_path_class"] == "online_decode_roi"
    # Must not treat cache Top-1 as online S5.
    assert abs(claim["B4"]["top1"] - claim["E4"]["S5_online"]["top1"]) > 0.01


def test_r0_s5_paired_gate() -> None:
    r0 = load_r0_comparator_report(ROOT)
    assert r0.get("accuracy_gate_pass") is True
    paired = r0.get("paired_S5_minus_R0") or {}
    assert abs(float(paired.get("paired_top1_delta_pp", 0)) - 0.04866180048661801) < 1e-6


def test_no_forbidden_signer_term_in_pages() -> None:
    paths = list((ROOT / "src/deployment").rglob("*.py"))
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert forbidden_term_scan(text) == [], path
    assert forbidden_term_scan("mã người ký của video") == ["người ký"]


def test_no_file_uploader_in_streamlit_surface() -> None:
    for path in (ROOT / "src/deployment").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "file_uploader" not in text, path


def test_required_statements_present_on_pages() -> None:
    required = [
        "Đây là hệ thống nhận dạng ký hiệu đơn lẻ",
        "chưa dịch câu" if False else "dịch câu",  # pages use limitation wording
        "signer_id",
        "S5 giữ mô hình/checkpoint B4" if False else "checkpoint B4",
    ]
    overview = (ROOT / "src/deployment/app.py").read_text(encoding="utf-8")
    assert "Đây là hệ thống nhận dạng ký hiệu đơn lẻ" in overview
    assert "signer_id" in overview
    assert (
        "feature cache của Unified Attention" in overview
        or "feature cache B4" in overview
        or "cache B4" in overview
    )
    lim = (ROOT / "src/deployment/pages/4_Limitations.py").read_text(encoding="utf-8")
    assert "chưa dịch câu" in lim
    ev = (ROOT / "src/deployment/pages/3_Evidence.py").read_text(encoding="utf-8")
    assert "không dùng 51,55%" in ev.lower() or "Không dùng 51,55%" in ev


def test_path_mismatch_explanation_present() -> None:
    claim = load_scientific_claim_lock(ROOT)
    expl = claim["E4"]["path_mismatch_explanation"]
    assert "cache-backed" in expl
    assert "online" in expl
    assert "formal E4 comparison" in expl


def test_fail_closed_corrupt_evidence_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(bad)


def test_fail_closed_sha_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "SCIENTIFIC_CLAIM_LOCK.json"
    p.write_text('{"contributions_exactly": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_sha_mismatch"):
        load_json(p, expected_sha="0" * 64)


def test_demo_cutover_remains_false_in_final_terminal() -> None:
    term = load_json(ROOT / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_FINAL_TERMINAL.json")
    assert term["streamlit_demo_cutover"] is False
    assert term["streamlit_s5_mode"] == "EVIDENCE_ONLY"
    assert term["streamlit_active_engine"] == "B4_CACHE_BACKED"
    assert term["terminal"] == "E4_SYSKD_V1_S5_PROMOTED_DEMO_NOT_CUTOVER"


def test_tex_two_contribution_macros() -> None:
    tex = (ROOT / "Docs/thesis/VSL_MSSTA_REPORT.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\SfiveSpeedup}{1{,}220}" in tex
    assert r"\newcommand{\EfourSyskd}{Chiến dịch Tối ưu Hệ thống}" in tex
    assert "E4_SYSKD" not in tex
    assert "SYSKD" not in tex
    assert "StudentName" in tex
    assert forbidden_term_scan(tex) == []
