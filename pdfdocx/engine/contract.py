# -*- coding: utf-8 -*-
"""Reusable front door for the verified PDF→DOCX layout engine.

The layout rules live in ``scripts/convert_datn_pdf_to_docx.py``. This module
does not copy them. It only resolves paths, selects a verify profile, and
returns structured results an MCP client can reuse.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import statistics
import sys
from functools import lru_cache
from pathlib import Path

DATN_CHECKS = (
    "cover_city",
    "student",
    "vietnamese_u_horn",
    "no_broken_u_horn",
    "top1",
    "speedup",
    "n_test",
    "ci_contains_zero",
    "optimal_name",
    "no_glued_city",
    "no_glued_name",
    "fonts_allowed",
    "toc_e1_line",
    "toc_c_line",
    "figure_b1_line",
    "date_line",
    "abbr_precision_same_cell",
)

GUARANTEES = (
    "One PDF page becomes one DOCX page, with the same line breaks.",
    "Word/WPS baseline = line top + 0.8 × exact line spacing; space-before is measured from that.",
    "Full-width paragraph lines are joined with line breaks and justified.",
    "Booktabs tables become Word tables that keep only the PDF horizontal rules.",
    "Display equations that stack or contain a fraction bar stay cropped from the PDF. Single-line formulas stay editable text from the PDF glyph layer. OCR is not used.",
    "Body text is Times New Roman; math letters are Cambria Math; monospace is Courier New.",
    "pdfLaTeX Vietnamese word spaces are rebuilt from glyph gaps.",
    "Page numbers are no-wrap text boxes, so they may overlap body text the way a LaTeX footer does.",
    "Run properties are schema-ordered, language is vi-VN, and spellcheck is off (noProof).",
)

LIMITS = (
    "Equations and figures are images cut from the PDF, not editable Office math.",
    "Margins are calibrated to the submitted DATN PDF (text block about 99.21–538.58 pt on A4).",
    "convert_pdf_to_docx refuses a PDF whose text block does not match that calibration, unless allow_uncalibrated is true. On A4, that opt-in applies the measured text box and restores the calibrated margins afterwards.",
    "compare_docx_pdf uses the registered Word.Application server when it is Word or WPS, and otherwise kwps.Application.",
    "The datn verify profile checks this thesis (Top-1 51,55%, n=6165, speedup 1,220, CI contains 0). The default profile is layout.",
    "Install with python pdfdocx/install.py. Do not reuse a system-site-packages environment.",
)


def repo_root() -> Path:
    override = os.environ.get("PDFDOCX_REPO_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    return candidate.resolve()


def _load(module_name: str, relative: str):
    path = repo_root() / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing engine file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the class module through sys.modules during class body execution.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@lru_cache(maxsize=1)
def converter_module():
    return _load("datn_pdf_to_docx", "scripts/convert_datn_pdf_to_docx.py")


@lru_cache(maxsize=1)
def compare_module():
    return _load("datn_docx_pdf_compare", "scripts/compare_datn_docx_pdf.py")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


def _fail(exc: Exception) -> dict:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def engine_contract() -> dict:
    """Guarantees, limits, fonts, and the thesis verify checks this engine owns."""
    try:
        conv = converter_module()
    except Exception as exc:
        return _fail(exc)
    return {
        "ok": True,
        "name": "pdfdocx",
        "engine": "scripts/convert_datn_pdf_to_docx.py",
        "compare": "scripts/compare_datn_docx_pdf.py",
        "default_pdf": _rel(Path(conv.SRC)),
        "default_docx": _rel(Path(conv.OUT)),
        "fonts": [conv.FONT, conv.MATH_FONT, conv.MONO_FONT],
        "calibrated_text_box_pt": {
            "x0": conv.X0,
            "x1": conv.X1,
            "top_margin": conv.TOP_MARGIN,
            "bottom_margin": conv.BOTTOM_MARGIN,
            "baseline_factor": conv.BASE_K,
        },
        "profiles": {
            "datn": "Seventeen content checks for the submitted VSL thesis DOCX. Pass profile=datn explicitly.",
            "layout": "Font allow-list and a non-empty body. This is the default.",
        },
        "verify_default_profile": "layout",
        "convert_policy": "Refuse when geometry_matches_calibration is false unless allow_uncalibrated is true.",
        "install": "python pdfdocx/install.py",
        "requirements": "pdfdocx/requirements.txt",
        "datn_checks": list(DATN_CHECKS),
        "guarantees": list(GUARANTEES),
        "limits": list(LIMITS),
    }


def inspect_pdf(pdf_path: str) -> dict:
    """Page size and whether the text block matches the calibrated DATN margins."""
    path = _resolve(pdf_path)
    if not path.is_file():
        return {"ok": False, "error": f"PDF not found: {path}"}
    try:
        import pymupdf

        conv = converter_module()
        doc = pymupdf.open(path)
        try:
            page0 = doc[0]
            lefts: list[float] = []
            rights: list[float] = []
            for page in doc:
                for block in page.get_text("dict")["blocks"]:
                    if block.get("type") != 0:
                        continue
                    for line in block["lines"]:
                        text = "".join(span["text"] for span in line["spans"]).strip()
                        if len(text) < 12:
                            continue
                        x0, _, x1, _ = line["bbox"]
                        if x1 - x0 < 80:
                            continue
                        lefts.append(x0)
                        rights.append(x1)
            # Mode of the left edge is the body margin. The median is pulled inward by indents.
            mode_left = statistics.mode([round(x) for x in lefts]) if lefts else None
            wide_rights = [x1 for x0, x1 in zip(lefts, rights) if x1 - x0 >= 300]
            mode_right = statistics.mode([round(x) for x in wide_rights]) if wide_rights else None
            matches = (
                mode_left is not None
                and mode_right is not None
                and abs(mode_left - conv.X0) <= 3.0
                and abs(mode_right - conv.X1) <= 8.0
            )
            return {
                "ok": True,
                "path": str(path),
                "pages": doc.page_count,
                "page_width_pt": round(page0.rect.width, 2),
                "page_height_pt": round(page0.rect.height, 2),
                "text_left_mode_pt": None if mode_left is None else float(mode_left),
                "text_right_mode_pt": None if mode_right is None else float(mode_right),
                "calibrated_text_box_pt": {"x0": conv.X0, "x1": conv.X1},
                "geometry_matches_calibration": matches,
                "note": (
                    "Text block matches the calibrated DATN margins."
                    if matches
                    else "Text block differs from the calibrated DATN margins. convert_pdf_to_docx will refuse unless allow_uncalibrated is true."
                ),
            }
        finally:
            doc.close()
    except Exception as exc:
        return _fail(exc)


def convert_pdf_to_docx(pdf_path: str, docx_path: str = "", allow_uncalibrated: bool = False) -> dict:
    """Convert one PDF with the verified layout engine.

    ``docx_path`` empty writes ``<pdf stem>.docx`` beside the PDF. A PDF whose
    text block does not match the calibrated margins is refused, and no DOCX is
    written, unless ``allow_uncalibrated`` is true. Progress lines from the
    engine are captured so they do not break an MCP stdio session.
    """
    src = _resolve(pdf_path)
    if not src.is_file():
        return {"ok": False, "error": f"PDF not found: {src}"}
    inspected = inspect_pdf(str(src))
    if not inspected.get("ok"):
        return inspected
    if not inspected["geometry_matches_calibration"] and not allow_uncalibrated:
        return {
            "ok": False,
            "error": "text block does not match the calibrated margins; conversion refused",
            "hint": "Pass allow_uncalibrated=true to convert. On A4 the measured text box is applied for that run only.",
            "inspect": inspected,
        }
    out = _resolve(docx_path) if docx_path else src.with_suffix(".docx")
    conv = converter_module()
    margin_fit, saved = _margin_override(conv, inspected, allow_uncalibrated)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            if saved is not None:
                conv.X0, conv.X1 = margin_fit["x0"], margin_fit["x1"]
            info = conv.build(src, out)
        info["ok"] = True
        info["pdf"] = str(src)
        info["log"] = log.getvalue().strip()
        info["geometry_matches_calibration"] = inspected["geometry_matches_calibration"]
        info["allow_uncalibrated"] = bool(allow_uncalibrated)
        info["margin_fit"] = margin_fit
        return info
    except Exception as exc:
        return _fail(exc)
    finally:
        if saved is not None:
            conv.X0, conv.X1 = saved


def _verify_layout(path: Path) -> dict:
    conv = converter_module()
    doc = conv.Document(path)
    fonts: set[str] = set()
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run.font.name:
                fonts.add(run.font.name)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if run.font.name:
                            fonts.add(run.font.name)
    allowed = {conv.FONT, conv.MATH_FONT, conv.MONO_FONT}
    page_breaks = sum(1 for paragraph in doc.paragraphs if paragraph.paragraph_format.page_break_before)
    checks = {
        "fonts_allowed": fonts <= allowed,
        "has_body": len(doc.paragraphs) > 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not failed,
        "profile": "layout",
        "path": str(path),
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "page_breaks": page_breaks,
        "fonts": sorted(fonts),
        "checks": checks,
        "failed": failed,
    }


def verify_docx(docx_path: str, profile: str = "layout") -> dict:
    """Check a DOCX. Default ``layout`` is reusable. ``datn`` is the 17 thesis checks."""
    if profile not in {"datn", "layout"}:
        return {"ok": False, "error": f"unknown profile: {profile}", "profiles": ["datn", "layout"]}
    path = _resolve(docx_path)
    if not path.is_file():
        return {"ok": False, "error": f"DOCX not found: {path}"}
    try:
        if profile == "layout":
            return _verify_layout(path)
        checks = converter_module().verify(path)
        failed = list(checks.get("failed") or [])
        missing = [name for name in DATN_CHECKS if name not in checks]
        failed.extend(name for name in missing if name not in failed)
        return {"ok": not failed, "profile": "datn", "path": str(path), "missing_checks": missing, **checks, "failed": failed}
    except Exception as exc:
        return _fail(exc)


def _margin_override(conv, inspected: dict, allow_uncalibrated: bool) -> tuple[dict, tuple[float, float] | None]:
    """Use the measured A4 text box for one uncalibrated run. The thesis path does not enter here."""
    if inspected["geometry_matches_calibration"] or not allow_uncalibrated:
        return {"applied": False, "reason": "calibrated margins kept"}, None
    if abs(inspected["page_width_pt"] - 595.28) > 2 or abs(inspected["page_height_pt"] - 841.89) > 2:
        return {"applied": False, "reason": "page is not A4; calibrated page size stays"}, None
    left = inspected.get("text_left_mode_pt")
    right = inspected.get("text_right_mode_pt")
    page_w = inspected["page_width_pt"]
    if left is None or right is None or right - left < 200 or not (12 <= left < right <= page_w - 12):
        return {"applied": False, "reason": "text block edges were not measured inside the page"}, None
    return {"applied": True, "x0": float(left), "x1": float(right)}, (conv.X0, conv.X1)


def compare_backend() -> dict:
    """Which Word or WPS server is registered. Does not launch the application."""
    from .office import discover

    return discover()


def compare_docx_pdf(pdf_path: str, docx_path: str, pages: str = "1", work_dir: str = "") -> dict:
    """Render the DOCX through Word/WPS and compare it with the PDF.

    ``pages`` is a comma-separated 1-based page list. Side-by-side PNGs and
    ``report.json`` are written under ``work_dir`` (default ``tmp/docx_compare``).
    """
    pdf = _resolve(pdf_path)
    docx = _resolve(docx_path)
    if not pdf.is_file():
        return {"ok": False, "error": f"PDF not found: {pdf}"}
    if not docx.is_file():
        return {"ok": False, "error": f"DOCX not found: {docx}"}
    try:
        selected = [int(part) for part in pages.split(",") if part.strip()]
    except ValueError:
        return {"ok": False, "error": f"pages must be comma-separated integers, got: {pages}"}
    if not selected:
        return {"ok": False, "error": "pages is empty"}
    backend = compare_backend()
    if not backend["ready"]:
        return {"ok": False, "error": backend["reason"], "compare_backend": backend}
    work = _resolve(work_dir) if work_dir else repo_root() / "tmp" / "docx_compare"
    try:
        report = compare_module().compare(pdf, docx, selected, work)
        report["ok"] = True
        report["pdf"] = str(pdf)
        report["docx"] = str(docx)
        report["work_dir"] = str(work)
        return report
    except Exception as exc:
        return _fail(exc)
