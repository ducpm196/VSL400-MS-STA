"""Contract gate for the pdfdocx MCP front door.

The layout engine is not reimplemented here. These checks bind the MCP wrapper
to scripts/convert_datn_pdf_to_docx.py and to the submitted thesis files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("docx")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pdfdocx"))

from engine.contract import (  # noqa: E402
    DATN_CHECKS,
    compare_backend,
    compare_docx_pdf,
    convert_pdf_to_docx,
    converter_module,
    engine_contract,
    inspect_pdf,
    verify_docx,
)
from engine.runtime import runtime_status  # noqa: E402

PDF = ROOT / "Docs/thesis/25410035_PhamMinhDuc_25410012_NguyenThiKieuThanh_DATN.pdf"
DOCX = ROOT / "Docs/thesis/25410035_phamminhduc_25410012_nguyenthikieuthanh_datn.docx"


def test_engine_contract_names_the_verified_converter() -> None:
    contract = engine_contract()
    assert contract["ok"] is True
    assert contract["engine"] == "scripts/convert_datn_pdf_to_docx.py"
    assert contract["fonts"] == ["Times New Roman", "Cambria Math", "Courier New"]
    assert contract["calibrated_text_box_pt"]["baseline_factor"] == 0.8
    assert contract["datn_checks"] == list(DATN_CHECKS)
    assert len(contract["datn_checks"]) == 17
    assert contract["verify_default_profile"] == "layout"
    assert "allow_uncalibrated" in contract["convert_policy"]
    assert "51,55" in " ".join(contract["limits"])
    assert (ROOT / contract["engine"]).is_file()
    assert (ROOT / contract["default_pdf"]).is_file()
    assert (ROOT / contract["default_docx"]).is_file()


def test_inspect_thesis_pdf_matches_calibrated_margins() -> None:
    if not PDF.is_file():
        pytest.skip("thesis PDF is not in this checkout")
    info = inspect_pdf(str(PDF))
    assert info["ok"] is True
    assert info["pages"] == 83
    assert abs(info["page_width_pt"] - 595.28) < 1
    assert abs(info["page_height_pt"] - 841.89) < 1
    assert info["geometry_matches_calibration"] is True


def test_verify_thesis_docx_datn_profile() -> None:
    if not DOCX.is_file():
        pytest.skip("thesis DOCX is not in this checkout")
    result = verify_docx(str(DOCX), profile="datn")
    assert result["ok"] is True
    assert result["failed"] == []
    assert result["missing_checks"] == []
    for name in DATN_CHECKS:
        assert result[name] is True


def test_verify_default_profile_is_layout() -> None:
    if not DOCX.is_file():
        pytest.skip("thesis DOCX is not in this checkout")
    result = verify_docx(str(DOCX))
    assert result["profile"] == "layout"
    assert result["ok"] is True


def test_verify_thesis_docx_layout_profile() -> None:
    if not DOCX.is_file():
        pytest.skip("thesis DOCX is not in this checkout")
    result = verify_docx(str(DOCX), profile="layout")
    assert result["ok"] is True
    assert result["failed"] == []
    assert result["page_breaks"] == 82
    assert set(result["fonts"]) <= {"Times New Roman", "Cambria Math", "Courier New"}


def test_missing_pdf_and_unknown_profile_fail_closed() -> None:
    missing = convert_pdf_to_docx("Docs/thesis/does-not-exist.pdf")
    assert missing["ok"] is False
    assert "not found" in missing["error"]
    unknown = verify_docx(str(DOCX) if DOCX.is_file() else "missing.docx", profile="other")
    assert unknown["ok"] is False
    assert "unknown profile" in unknown["error"]
    bad_pages = compare_docx_pdf(str(PDF), str(DOCX), pages="a,b")
    assert bad_pages["ok"] is False


def test_uncalibrated_pdf_is_refused_and_opt_in_still_builds(tmp_path: Path) -> None:
    pdf = tmp_path / "narrow.pdf"
    out = tmp_path / "narrow.docx"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 100), "This line sits far from the calibrated thesis margin.", fontsize=12)
    doc.save(pdf)
    doc.close()
    refused = convert_pdf_to_docx(str(pdf), str(out))
    assert refused["ok"] is False
    assert refused["inspect"]["geometry_matches_calibration"] is False
    assert not out.exists()
    built = convert_pdf_to_docx(str(pdf), str(out), allow_uncalibrated=True)
    assert built["ok"] is True
    assert built["allow_uncalibrated"] is True
    assert out.is_file()
    layout = verify_docx(str(out))
    assert layout["profile"] == "layout"
    assert layout["ok"] is True


def test_uncalibrated_a4_applies_measured_text_box(tmp_path: Path) -> None:
    pdf = tmp_path / "shifted.pdf"
    out = tmp_path / "shifted.docx"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((80, 140), "M" * 40, fontsize=12)
    doc.save(pdf)
    doc.close()
    built = convert_pdf_to_docx(str(pdf), str(out), allow_uncalibrated=True)
    assert built["ok"] is True
    assert built["margin_fit"]["applied"] is True
    assert built["margin_fit"]["x0"] != 99.21
    from docx import Document

    section = Document(out).sections[0]
    assert abs(section.left_margin.pt - built["margin_fit"]["x0"]) < 0.2
    assert converter_module().X0 == 99.21
    assert converter_module().X1 == 538.58


def test_compare_backend_follows_the_registered_server() -> None:
    backend = compare_backend()
    if sys.platform != "win32":
        assert backend["ready"] is False
        return
    assert backend["ready"] is True
    assert backend["product"] in {"wps", "microsoft_word"}
    assert backend["progid"] == backend["launch_order"][0]
    server = backend["server"].lower()
    if "wps.exe" in server:
        assert backend["product"] == "wps"
    if "winword.exe" in server:
        assert backend["product"] == "microsoft_word"


def test_single_line_equations_stay_text_and_stacked_equations_stay_images() -> None:
    if not PDF.is_file():
        pytest.skip("thesis PDF is not in this checkout")
    mod = converter_module()
    pdf = pymupdf.open(PDF)
    try:
        linear = pdf[37]
        stacked = pdf[23]
        linear_regions = mod._math_regions(mod._bands(mod._page_glyphs(linear, [])), [], linear.get_drawings())
        stacked_regions = mod._math_regions(mod._bands(mod._page_glyphs(stacked, [])), [], stacked.get_drawings())
    finally:
        pdf.close()
    assert linear_regions == []
    assert stacked_regions


def test_runtime_status_matches_verified_pins() -> None:
    status = runtime_status()
    assert status["ok"] is True
    assert status["packages"]["pymupdf"]["installed"] == "1.28.2"
    assert status["packages"]["python-docx"]["installed"] == "1.2.0"
    assert status["packages"]["mcp"]["installed"] == "1.27.2"
    assert status["package_failures"] == []
    assert status["fonts_ready"] is True
    assert status["fonts"]["faces"]["Times New Roman"]["found"] is True
    assert status["fonts"]["faces"]["Cambria Math"]["found"] is True
    assert status["fonts"]["faces"]["Courier New"]["found"] is True
    assert status["compare_ready"] is True
    assert status["engine_ready"] is True


def test_mcp_registers_the_tools() -> None:
    pytest.importorskip("mcp")
    from engine.mcp_server import build_server

    server = build_server()
    names = sorted(tool.name for tool in server._tool_manager.list_tools())
    assert names == [
        "compare_docx_pdf",
        "convert_pdf_to_docx",
        "engine_contract",
        "inspect_pdf",
        "runtime_status",
        "verify_docx",
    ]
