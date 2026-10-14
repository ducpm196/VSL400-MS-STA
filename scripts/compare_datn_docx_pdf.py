# -*- coding: utf-8 -*-
"""Render the thesis DOCX through WPS/Word and compare it page by page with the PDF."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "Docs/thesis/25410035_PhamMinhDuc_25410012_NguyenThiKieuThanh_DATN.pdf"
DOCX = ROOT / "Docs/thesis/25410035_phamminhduc_25410012_nguyenthikieuthanh_datn.docx"
WORK = ROOT / "tmp/docx_compare"


def _office():
    kit = ROOT / "pdfdocx"
    kit_str = str(kit)
    if kit_str not in sys.path:
        sys.path.insert(0, kit_str)
    from engine.office import discover

    return discover


def export_pdf(docx: Path, out_pdf: Path) -> dict:
    """Export through the registered Word or WPS server. Word.Application wins when it is registered."""
    import win32com.client

    choice = _office()()
    if not choice["ready"]:
        raise RuntimeError(choice["reason"])
    failures: list[str] = []
    for progid in choice["launch_order"]:
        try:
            app = win32com.client.DispatchEx(progid)
        except Exception as exc:
            failures.append(f"{progid}: {type(exc).__name__}: {exc}")
            continue
        try:
            try:
                app.DisplayAlerts = 0
            except Exception:
                pass
            app.Visible = False
            document = app.Documents.Open(str(docx.resolve()), ReadOnly=True)
            try:
                document.ExportAsFixedFormat(str(out_pdf.resolve()), 17)
            finally:
                document.Close(False)
        except Exception as exc:
            failures.append(f"{progid}: {type(exc).__name__}: {exc}")
            try:
                app.Quit()
            except Exception:
                pass
            continue
        try:
            app.Quit()
        except Exception:
            pass
        used = next(app_info for app_info in choice["apps"] if app_info["progid"] == progid)
        return {"progid": progid, "product": used["product"], "server": used["server"]}
    raise RuntimeError("no Word or WPS server exported the DOCX; " + " | ".join(failures))


def page_lines(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    rows = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                rows.append((round(line["bbox"][0], 1), round(line["bbox"][1], 1), text))
    return rows


def side_by_side(src: pymupdf.Document, out: pymupdf.Document, pages: list[int], dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for n in pages:
        if n > src.page_count or n > out.page_count:
            continue
        a = src[n - 1].get_pixmap(matrix=pymupdf.Matrix(1.0, 1.0), alpha=False)
        b = out[n - 1].get_pixmap(matrix=pymupdf.Matrix(1.0, 1.0), alpha=False)
        canvas = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, a.width + b.width + 12, max(a.height, b.height)), False)
        canvas.clear_with(200)
        canvas.copy(a, a.irect)
        b.set_origin(a.width + 12, 0)
        canvas.copy(b, b.irect)
        path = dest / f"page_{n:03d}.png"
        canvas.save(path)
        written.append(str(path))
    return written


def compare(pdf: Path, docx: Path, pages: list[int], work: Path) -> dict:
    """Render DOCX through Word/WPS and compare page text with the source PDF."""
    work.mkdir(parents=True, exist_ok=True)
    out_pdf = work / "docx_render.pdf"
    if out_pdf.exists():
        out_pdf.unlink()
    office = export_pdf(docx, out_pdf)
    if not out_pdf.exists() or out_pdf.stat().st_mtime < docx.stat().st_mtime:
        raise RuntimeError("render failed: docx_render.pdf is missing or older than the DOCX")
    src = pymupdf.open(pdf)
    out = pymupdf.open(out_pdf)
    try:
        mismatched = []
        for i in range(min(src.page_count, out.page_count)):
            a = [t for _, _, t in page_lines(src[i])]
            b = [t for _, _, t in page_lines(out[i])]
            if a and b and a[0].replace(" ", "")[:20] != b[0].replace(" ", "")[:20]:
                mismatched.append(i + 1)
        images = side_by_side(src, out, pages, work)
        report = {
            "pdf_pages": src.page_count,
            "docx_pages": out.page_count,
            "first_line_mismatch_pages": mismatched,
            "images": images,
            "office": office,
        }
    finally:
        src.close()
        out.close()
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="1,2,4,7,9,10,13,18,19,21,25,40,50,73,83")
    parser.add_argument("--pdf", default=str(PDF))
    parser.add_argument("--docx", default=str(DOCX))
    parser.add_argument("--work", default=str(WORK))
    args = parser.parse_args()
    pages = [int(x) for x in args.pages.split(",") if x.strip()]
    report = compare(Path(args.pdf), Path(args.docx), pages, Path(args.work))
    print(json.dumps({k: v for k, v in report.items() if k != "images"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
