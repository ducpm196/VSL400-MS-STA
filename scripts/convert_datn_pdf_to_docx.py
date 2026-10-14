# -*- coding: utf-8 -*-
"""Convert the submitted thesis PDF into a DOCX that keeps the PDF page layout.

Every PDF line is placed at its own baseline. Word and WPS put the baseline at
0.8 x the exact line spacing below the top of the line box; the converter uses
that relation to turn PDF coordinates into space-before values. Consequences:

* one PDF page maps to one DOCX page, with the same line breaks;
* full-width lines of a paragraph are joined with line breaks and justified;
* booktabs tables become Word tables with only the PDF horizontal rules;
* display equations, TikZ diagrams and raster figures are cropped from the PDF
  and anchored at their PDF position, so LaTeX notation stays exact;
* text uses Times New Roman; math letters use Cambria Math with real
  subscripts and superscripts.

pdfLaTeX draws some Vietnamese word spaces as horizontal shifts instead of
space glyphs, so spaces are rebuilt from glyph gaps.
"""
from __future__ import annotations

import copy
import io
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Docs/thesis/25410035_PhamMinhDuc_25410012_NguyenThiKieuThanh_DATN.pdf"
OUT = ROOT / "Docs/thesis/25410035_phamminhduc_25410012_nguyenthikieuthanh_datn.docx"

FONT = "Times New Roman"
MATH_FONT = "Cambria Math"
MONO_FONT = "Courier New"
X0 = 99.21
X1 = 538.58
TEXT_CX = (X0 + X1) / 2
TOP_MARGIN = 28.0
BOTTOM_MARGIN = 12.0
BASE_K = 0.8
SEG_GAP = 11.0
TABLE_GAP = 6.0
MONO_SCALE = 88
MSBM = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ", "E": "𝔼", "P": "ℙ"}
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


# ---------------------------------------------------------------- extraction


@dataclass
class Glyph:
    ch: str
    x0: float
    x1: float
    top: float
    bottom: float
    base: float
    size: float
    family: str
    bold: bool
    italic: bool
    math: bool


@dataclass
class Band:
    glyphs: list[Glyph]
    base: float
    size: float
    top: float
    bottom: float
    segs: list[list[Glyph]] = field(default_factory=list)
    fine: list[list[Glyph]] = field(default_factory=list)

    @property
    def x0(self) -> float:
        return self.segs[0][0].x0

    @property
    def x1(self) -> float:
        return self.segs[-1][-1].x1

    def text(self) -> str:
        return "".join(t for t, _ in _runs(self, self.glyphs))

    def seg_text(self, i: int) -> str:
        return "".join(t for t, _ in _runs(self, self.segs[i])).strip()


def _classify(font: str) -> tuple[str, bool, bool, bool]:
    bold = "Bold" in font or "BX" in font
    italic = "Italic" in font or "Oblique" in font or "LMMathItalic" in font
    if "MSBM" in font or "Math" in font or "Symbol" in font:
        return MATH_FONT, False, "MathItalic" in font, True
    if "Mono" in font:
        return MONO_FONT, bold, italic, False
    return FONT, bold, italic, False


def _page_glyphs(page: pymupdf.Page, exclude: list[pymupdf.Rect]) -> list[list[Glyph]]:
    flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_LIGATURES
    lines: list[list[Glyph]] = []
    for block in page.get_text("rawdict", flags=flags)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            glyphs: list[Glyph] = []
            for span in line["spans"]:
                font = span.get("font") or ""
                family, bold, italic, math = _classify(font)
                for c in span["chars"]:
                    ch = c["c"]
                    if "MSBM" in font:
                        ch = MSBM.get(ch, ch)
                    ch = LIGATURES.get(ch, ch)
                    if ch in ("\u00ad",) or (ord(ch[0]) < 32):
                        continue
                    x0, top, x1, bottom = c["bbox"]
                    cx, cy = (x0 + x1) / 2, (top + bottom) / 2
                    if any(r.x0 - 1 <= cx <= r.x1 + 1 and r.y0 - 1 <= cy <= r.y1 + 1 for r in exclude):
                        continue
                    glyphs.append(Glyph(ch, x0, x1, top, bottom, c["origin"][1], span["size"], family, bold, italic, math))
            if glyphs:
                lines.append(glyphs)
    return lines


def _bands(lines: list[list[Glyph]]) -> list[Band]:
    def line_base(glyphs: list[Glyph]) -> tuple[float, float]:
        big = max(g.size for g in glyphs)
        bases = [g.base for g in glyphs if g.size >= big - 0.3 and g.ch.strip()]
        return (statistics.median(bases) if bases else glyphs[0].base), big

    rows = sorted((line_base(g) + (g,) for g in lines), key=lambda r: r[0])
    bands: list[Band] = []
    for base, size, glyphs in rows:
        if bands:
            band = bands[-1]
            tol = 0.42 * max(size, band.size)
            if abs(base - band.base) <= tol:
                band.glyphs.extend(glyphs)
                if size > band.size + 0.3:
                    band.base, band.size = base, size
                continue
        bands.append(Band(list(glyphs), base, size, 0.0, 0.0))
    for band in bands:
        band.glyphs = _attach_accents(band.glyphs)
        band.glyphs.sort(key=lambda g: g.x0)
        band.top = min(g.top for g in band.glyphs)
        band.bottom = max(g.bottom for g in band.glyphs)
        band.segs = _split(band.glyphs, SEG_GAP)
        band.fine = _split(band.glyphs, TABLE_GAP)
    return [b for b in bands if b.segs]


def _split(glyphs: list[Glyph], limit: float) -> list[list[Glyph]]:
    segs: list[list[Glyph]] = [[glyphs[0]]]
    for g in glyphs[1:]:
        prev = segs[-1][-1]
        gap = g.x0 - prev.x1
        dots = prev.ch == "." and g.ch == "."
        if gap > limit and not dots:
            segs.append([g])
        else:
            segs[-1].append(g)
    return [s for s in segs if any(g.ch.strip() for g in s)]


# ---------------------------------------------------------------- run styling


def _style_of(band: Band, g: Glyph) -> tuple:
    vert = None
    if g.size < 0.86 * band.size and g.ch.strip():
        shift = g.base - band.base
        if shift > 0.8:
            vert = "sub"
        elif shift < -0.8:
            vert = "sup"
    size = band.size if vert else g.size
    return (g.family, g.bold, g.italic, round(size * 2) / 2, vert)


ACCENTS = {"¯": "\u0304", "ˆ": "\u0302", "˜": "\u0303", "˙": "\u0307", "¨": "\u0308", "→": "\u20d7", "ˇ": "\u030c"}


def _attach_accents(glyphs: list[Glyph]) -> list[Glyph]:
    """Turn a spacing accent drawn over a letter into a combining mark after it."""
    letters = [g for g in glyphs if g.ch not in ACCENTS]
    out: list[Glyph] = []
    host_mark: dict[int, str] = {}
    absorbed: set[int] = set()
    for g in glyphs:
        if g.ch not in ACCENTS:
            continue
        cx = (g.x0 + g.x1) / 2
        width = max(0.1, g.x1 - g.x0)

        def overlap(h: Glyph) -> float:
            return max(0.0, min(g.x1, h.x1) - max(g.x0, h.x0)) / min(width, max(0.1, h.x1 - h.x0))

        hosts = [h for h in letters if h.ch.strip() and overlap(h) >= 0.6]
        if hosts:
            host = min(hosts, key=lambda h: abs((h.x0 + h.x1) / 2 - cx))
            host_mark[id(host)] = host_mark.get(id(host), "") + ACCENTS[g.ch]
            absorbed.add(id(g))
    for g in glyphs:
        if id(g) in absorbed:
            continue
        mark = host_mark.get(id(g))
        if mark:
            g = copy.copy(g)
            g.ch = g.ch + mark
        out.append(g)
    return out


def _runs(band: Band, glyphs: list[Glyph]) -> list[tuple[str, tuple]]:
    runs: list[list] = []
    prev: Glyph | None = None
    glyphs = _attach_accents(glyphs)
    for g in glyphs:
        style = _style_of(band, g)
        if prev is not None and g.ch != " ":
            gap = g.x0 - prev.x1
            if gap >= max(1.6, 0.18 * max(g.size, prev.size)) and runs and not runs[-1][0].endswith(" "):
                runs[-1][0] += " "
        if g.ch == " ":
            if runs and not runs[-1][0].endswith(" "):
                runs[-1][0] += " "
            prev = g
            continue
        if runs and runs[-1][1] == style:
            runs[-1][0] += g.ch
        else:
            runs.append([g.ch, style])
        prev = g
    return [(t, s) for t, s in runs]


def _apply_style(run, style: tuple) -> None:
    family, bold, italic, size, vert = style
    run.font.name = family
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if vert == "sub":
        run.font.subscript = True
    elif vert == "sup":
        run.font.superscript = True
    rpr = run._element.get_or_add_rPr()
    if family == MONO_FONT:
        scale = OxmlElement("w:w")
        scale.set(qn("w:val"), str(MONO_SCALE))
        rpr.append(scale)
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), family)
    if rpr.find(qn("w:noProof")) is None:
        rpr.append(OxmlElement("w:noProof"))
    lang = rpr.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        rpr.append(lang)
    lang.set(qn("w:val"), "vi-VN")
    _order_rpr(rpr)


_RPR_ORDER = [
    "rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
    "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
    "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
    "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath",
]
_RPR_RANK = {qn(f"w:{name}"): i for i, name in enumerate(_RPR_ORDER)}


def _order_rpr(rpr) -> None:
    children = list(rpr)
    children.sort(key=lambda el: _RPR_RANK.get(el.tag, len(_RPR_ORDER)))
    for el in children:
        rpr.remove(el)
    for el in children:
        rpr.append(el)


_FONT_FILES = {
    (False, False): "times.ttf",
    (True, False): "timesbd.ttf",
    (False, True): "timesi.ttf",
    (True, True): "timesbi.ttf",
}
_METRICS: dict = {}


def _text_width(text: str, style: tuple) -> float:
    family, bold, italic, size, _ = style
    key = (bold, italic) if family == FONT else None
    if key is None:
        return 0.5 * size * len(text)
    if key not in _METRICS:
        from fontTools.ttLib import TTFont

        font = TTFont(str(Path("C:/Windows/Fonts") / _FONT_FILES[key]), lazy=True)
        _METRICS[key] = (font.getBestCmap(), font["hmtx"].metrics, font["head"].unitsPerEm)
    cmap, hmtx, upem = _METRICS[key]
    total = 0
    for ch in text:
        glyph = cmap.get(ord(ch))
        total += hmtx[glyph][0] if glyph in hmtx else upem // 2
    return total * size / upem


def _emit_runs(paragraph, pieces: list[tuple[str, tuple]]) -> None:
    for text, style in pieces:
        if text:
            _apply_style(paragraph.add_run(text), style)


def _strip_pieces(pieces: list[tuple[str, tuple]]) -> list[tuple[str, tuple]]:
    out = [list(p) for p in pieces if p[0]]
    if out:
        out[0][0] = out[0][0].lstrip()
        out[-1][0] = out[-1][0].rstrip()
    return [(t, s) for t, s in out if t]


# ---------------------------------------------------------------- page regions


def _rules(page: pymupdf.Page, figures: list[pymupdf.Rect] | None = None) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """Table rules (booktabs, possibly narrower than the text block) and short signature blanks."""
    figures = figures or []
    lines = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None or r.height > 2.5 or r.width <= 15:
            continue
        r = pymupdf.Rect(r)
        if any(f.x0 - 2 <= r.x0 and r.x1 <= f.x1 + 2 and f.y0 - 2 <= r.y0 <= f.y1 + 2 for f in figures):
            continue
        lines.append(r)
    words = page.get_text("words")

    def is_blank(r: pymupdf.Rect) -> bool:
        return any(abs(w[3] - r.y0) < 4.5 and r.x0 - 20 <= w[2] <= r.x0 + 3 for w in words)

    wide, short = [], []
    for r in lines:
        if is_blank(r):
            short.append(r)
            continue
        twins = [o for o in lines if o is not r and abs(o.x0 - r.x0) < 2 and abs(o.x1 - r.x1) < 2]
        if r.width > 300 or (r.width > 100 and twins):
            wide.append(r)
        elif r.width <= 200:
            short.append(r)
    return sorted(wide, key=lambda r: r.y0), short


def _drawing_figures(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        r = pymupdf.Rect(r)
        if r.height <= 2.5 and r.width > 300:
            continue
        if r.height < 1:
            r = pymupdf.Rect(r.x0, r.y0 - 0.6, r.x1, r.y1 + 0.6)
        if r.width < 1:
            r = pymupdf.Rect(r.x0 - 0.6, r.y0, r.x1 + 0.6, r.y1)
        rects.append(r)
    parent = list(range(len(rects)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(rects):
        grown = pymupdf.Rect(a.x0 - 12, a.y0 - 12, a.x1 + 12, a.y1 + 12)
        for j in range(i + 1, len(rects)):
            if grown.intersects(rects[j]):
                parent[find(j)] = find(i)
    groups: dict[int, pymupdf.Rect] = {}
    for i, r in enumerate(rects):
        k = find(i)
        groups[k] = pymupdf.Rect(r) if k not in groups else groups[k] | r
    return [r for r in groups.values() if r.height >= 20 and r.width >= 60 and r.height < 700]


def _image_figures(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") == 1:
            r = pymupdf.Rect(block["bbox"])
            if r.width > 20 and r.height > 20:
                rects.append(r)
    return rects


def _expand_with_text(rect: pymupdf.Rect, lines: list[list[Glyph]]) -> pymupdf.Rect:
    grown = pymupdf.Rect(rect)
    for glyphs in lines:
        for g in glyphs:
            cx, cy = (g.x0 + g.x1) / 2, (g.top + g.bottom) / 2
            if rect.x0 - 1 <= cx <= rect.x1 + 1 and rect.y0 - 1 <= cy <= rect.y1 + 1:
                grown |= pymupdf.Rect(g.x0, g.top, g.x1, g.bottom)
    return grown


_EQ_NO = re.compile(r"^\([A-Z]?\d+(?:\.\d+)?\)$")


def _stacked_math(group: list[Band]) -> bool:
    """Two math pieces sit on different baselines and overlap horizontally, like a fraction."""
    for i, left in enumerate(group):
        for right in group[i + 1:]:
            delta = abs(left.base - right.base)
            # A fraction sits a few points above its denominator. The next
            # display equation is about one line further down and is not a stack.
            if not (4 < delta < 16):
                continue
            if left.x0 < right.x1 and right.x0 < left.x1:
                return True
    return False


def _bar_in_region(region: pymupdf.Rect, drawings: list) -> bool:
    """A short vector bar is a fraction line or a brace, not a page-wide table rule."""
    for drawing in drawings:
        rect = drawing["rect"]
        bar = pymupdf.Rect(rect.x0, min(rect.y0, rect.y1) - 1, rect.x1, max(rect.y0, rect.y1) + 1)
        if bar.height <= 2.5 and 8 <= bar.width <= 200 and bar.intersects(region):
            return True
    return False


def _math_regions(bands: list[Band], skip: list[tuple[float, float]], drawings: list | None = None) -> list[pymupdf.Rect]:
    def is_math(b: Band) -> bool:
        if any(y0 <= b.base <= y1 for y0, y1 in skip):
            return False
        chars = [g for g in b.glyphs if g.ch.strip()]
        if len(chars) < 2:
            return False
        ratio = sum(g.math for g in chars) / len(chars)
        numbered = b.x0 > X0 + 20 and b.x1 > X1 - 5 and re.search(r"\([A-Z]?\d+\.\d+\)\s*$", b.text() or "")
        return ratio >= 0.45 or bool(numbered)

    regions: list[pymupdf.Rect] = []
    group: list[Band] = []

    def flush() -> None:
        if not group:
            return
        y0 = min(b.top for b in group) - 2
        y1 = max(b.bottom for b in group) + 2
        region = pymupdf.Rect(min(b.x0 for b in group) - 2, y0, max(b.x1 for b in group) + 2, y1)
        # A single text-layer formula has no fraction bar and no stacked pieces.
        # OCR is not used: the glyphs are already in the PDF. Only the 2D cases
        # become a crop, because Word has no place to put the bar.
        if _stacked_math(group) or _bar_in_region(region, drawings or []):
            regions.append(pymupdf.Rect(X0 - 18, y0, X1 + 18, y1))
        group.clear()

    for b in bands:
        if is_math(b):
            if group and b.top - group[-1].bottom > 16:
                flush()
            group.append(b)
        else:
            flush()
    flush()
    return regions


# ---------------------------------------------------------------- layout model


@dataclass
class Para:
    kind: str
    top: float
    height: float
    bands: list[Band] = field(default_factory=list)
    pitch: float = 0.0
    align: str = "left"
    toc: dict | None = None


@dataclass
class TableBlock:
    top: float
    height: float
    rows: list[dict]
    bounds: list[float]
    right: float
    pitch: float
    rules: list[tuple[int, str, float]]


def _is_centered(b: Band) -> bool:
    if len(b.segs) != 1:
        return False
    mid = (b.x0 + b.x1) / 2
    return abs(mid - TEXT_CX) < 12 and b.x0 > X0 + 10 and b.x1 < X1 - 8


_TOC_TAIL = re.compile(r"^(.*?)[\s.]*(?:\.\s*){2,}\s*(\d{1,3}|[ivxlcdm]+)\s*$", re.I)


_TOC_HEAD = re.compile(r"^(Chương \d+\.|Phụ lục [A-Z]\.|[A-Z]?\d+(?:\.\d+)*\.|[A-Z]\.\d+(?:\.\d+)*\.|Hình [A-Z0-9.]+|Bảng [A-Z0-9.]+)")


def _toc_parts(b: Band) -> dict | None:
    text = b.text().strip()
    m = _TOC_TAIL.match(text)
    if not m and _TOC_HEAD.match(text):
        m = re.match(r"^(.*\S)\s+(\d{1,3})$", text)
    if not m:
        return None
    page_x1 = b.segs[-1][-1].x1
    if page_x1 < X1 - 6:
        return None
    number, title_x = "", None
    if len(b.segs) >= 2 and _TOC_HEAD.fullmatch(b.seg_text(0)):
        number = b.seg_text(0)
        title_x = b.segs[1][0].x0
    title = m.group(1).strip()
    if number and title.startswith(number):
        title = title[len(number):].strip()
    style = _style_of(b, next(g for g in b.glyphs if g.ch.strip()))
    return {"number": number, "title": title, "page": m.group(2), "title_x": title_x, "page_x1": page_x1, "style": style}


def _can_join(prev: Band, nxt: Band, pitch: float | None) -> bool:
    if len(prev.segs) != 1 or len(nxt.segs) != 1:
        return False
    if _is_centered(prev) or _is_centered(nxt):
        return False
    if prev.x1 < X1 - 4:
        return False
    if abs(prev.size - nxt.size) > 0.4:
        return False
    step = nxt.base - prev.base
    if step > 1.75 * prev.size or step < 0.9 * prev.size:
        return False
    if pitch is not None and abs(step - pitch) > 0.8:
        return False
    return abs(nxt.x0 - X0) < 3 or abs(nxt.x0 - prev.x0) < 3


_LABEL = re.compile(r"^(\[\d+\]|\d+\.|[a-z]\)|[•–·-])$")


def _split_label(b: Band, nxt: Band | None) -> tuple[list[Glyph], list[Glyph], float] | None:
    if len(b.segs) == 2 and _LABEL.match(b.seg_text(0)):
        return b.segs[0], b.segs[1], b.segs[1][0].x0
    if len(b.segs) != 1 or nxt is None or nxt.x0 < b.x0 + 5:
        return None
    m = re.match(r"^(\[\d+\]|\d+\.)\s", b.text().strip())
    if not m:
        return None
    want = len(m.group(1))
    count = 0
    for k, g in enumerate(b.segs[0]):
        if g.ch.strip():
            count += 1
        if count == want:
            label = b.segs[0][: k + 1]
            return label, b.segs[0][k + 1:], nxt.x0
    return None


def _hanging_group(free: list[Band], i: int) -> Para | None:
    b = free[i]
    nxt = free[i + 1] if i + 1 < len(free) else None
    split = _split_label(b, nxt)
    if not split:
        return None
    label_glyphs, body_glyphs, body_x = split
    group = [b]
    pitch = None
    while i + len(group) < len(free):
        prev, nxt = group[-1], free[i + len(group)]
        if len(nxt.segs) != 1 or abs(nxt.x0 - body_x) > 3 or prev.x1 < X1 - 4:
            break
        if abs(prev.size - nxt.size) > 0.4:
            break
        step = nxt.base - prev.base
        if step > 1.75 * prev.size or (pitch is not None and abs(step - pitch) > 0.8):
            break
        pitch = pitch or step
        group.append(nxt)
    if len(group) == 1:
        return None
    para = Para("hang", b.base - BASE_K * pitch, pitch * len(group), group, pitch, "justify")
    para.body_x = body_x
    para.label_glyphs = label_glyphs
    para.body_glyphs = body_glyphs
    return para


def _line_height(b: Band) -> float:
    return round(1.5 * b.size, 2)


def _intervals(bands: list[Band]) -> list[list[float]]:
    spans = sorted((seg[0].x0, seg[-1].x1) for b in bands for seg in b.fine)
    merged: list[list[float]] = []
    for a, z in spans:
        if merged and a <= merged[-1][1] + 4:
            merged[-1][1] = max(merged[-1][1], z)
        else:
            merged.append([a, z])
    return merged


def _table_block(bands: list[Band], rules: list[pymupdf.Rect]) -> TableBlock | None:
    if not bands:
        return None
    ivs = _intervals(bands)
    left = min([ivs[0][0]] + [r.x0 for r in rules])
    right = max([ivs[-1][1]] + [r.x1 for r in rules])
    bounds = [left] + [iv[0] - 1.5 for iv in ivs[1:]]

    def col_of(x: float) -> int:
        k = 0
        for i, bx in enumerate(bounds):
            if x + 2 >= bx:
                k = i
        return k

    steps = [b2.base - b1.base for b1, b2 in zip(bands, bands[1:]) if b2.base - b1.base < 2.0 * b1.size]
    pitch = statistics.median(steps) if steps else 1.3 * bands[0].size
    rows: list[dict] = []
    for b in bands:
        cells: dict[int, list[Glyph]] = {}
        for seg in b.fine:
            cells.setdefault(col_of(seg[0].x0), []).extend(seg)
        first = cells.get(0)
        first_text = "".join(g.ch for g in first).strip() if first else ""
        step = b.base - rows[-1]["bands"][-1].base if rows else 0.0
        lead = first_text[:1]
        greek = "\u0370" <= lead <= "\u03ff"
        if first is None:
            continuation = bool(rows) and step < 1.7 * b.size
        else:
            wraps = lead in ")]" or (lead.islower() and not greek)
            continuation = bool(rows) and wraps and step < 1.6 * b.size
        if continuation:
            rows[-1]["bands"].append(b)
            rows[-1]["lines"].append(cells)
        else:
            rows.append({"bands": [b], "lines": [cells]})
    inner = [b2.base - b1.base for r in rows for b1, b2 in zip(r["bands"], r["bands"][1:])]
    pitch = statistics.median(inner) if inner else min(pitch, 1.3 * bands[0].size)
    tops = [r["bands"][0].base - BASE_K * pitch for r in rows]
    heights = []
    for i, r in enumerate(rows):
        if i + 1 < len(rows):
            heights.append(tops[i + 1] - tops[i])
        else:
            below = [rule.y0 for rule in rules if rule.y0 > r["bands"][-1].base]
            bottom = min(below) if below else r["bands"][-1].base + 0.35 * pitch
            heights.append(max(len(r["bands"]) * pitch, bottom - tops[i]))
    for r, h in zip(rows, heights):
        r["height"] = max(h, pitch * 0.9)
    marks: list[tuple[int, str, float]] = []
    for rule in rules:
        y = rule.y0
        if y < rows[0]["bands"][0].top:
            marks.append((0, "top", 0.8))
            continue
        idx = max(i for i, r in enumerate(rows) if r["bands"][0].base < y)
        width = 0.8 if rule is rules[-1] and idx == len(rows) - 1 else 0.5
        marks.append((idx, "bottom", width))
    return TableBlock(tops[0], sum(r["height"] for r in rows), rows, bounds, right, pitch, marks)


def _is_body_line(b: Band) -> bool:
    return len(b.segs) == 1 and abs(b.x0 - X0) < 4 and b.x1 > X1 - 5


_CAPTION = re.compile(r"^(Bảng|Hình) [A-Z]?\d+(?:\.\d+)*:")


def _is_caption(b: Band) -> bool:
    return _CAPTION.match(b.text().strip()) is not None and b.x0 < X0 + 90


def _fits_columns(b: Band, ivs: list[list[float]]) -> bool:
    for seg in b.fine:
        a, z = seg[0].x0, seg[-1].x1
        k = None
        for i, iv in enumerate(ivs):
            if a >= iv[0] - 6:
                k = i
        if k is None:
            return False
        limit = ivs[k + 1][0] - 2 if k + 1 < len(ivs) else X1 + 3
        if z > limit:
            return False
    return True


def _collect_tables(bands: list[Band], wide: list[pymupdf.Rect]) -> tuple[list[TableBlock], set[int]]:
    tables: list[TableBlock] = []
    used: set[int] = set()
    groups: list[list[pymupdf.Rect]] = []
    for r in wide:
        if groups:
            prev = groups[-1][-1]
            between = [b for b in bands if prev.y0 < b.base < r.y0]
            if not any(_is_caption(b) or _is_body_line(b) for b in between):
                groups[-1].append(r)
                continue
        groups.append([r])
    for rules in groups:
        if len(rules) < 2:
            continue
        y_top, y_bot = rules[0].y0, rules[-1].y0
        inside = [i for i, b in enumerate(bands) if y_top < b.base < y_bot + 2 and i not in used]
        if not inside:
            continue
        last = inside[-1]
        ref = [bands[i] for i in inside]
        pitch_guess = statistics.median(
            [b2.base - b1.base for b1, b2 in zip(ref, ref[1:])] or [1.35 * ref[0].size]
        )
        ivs = _intervals(ref)
        j = last + 1
        while j < len(bands):
            b = bands[j]
            if b.base - bands[j - 1].base > 2.3 * pitch_guess or b.base > 755:
                break
            if _is_caption(b) or _is_body_line(b) or not _fits_columns(b, ivs):
                break
            inside.append(j)
            j += 1
        block = _table_block([bands[i] for i in inside], rules)
        if block:
            tables.append(block)
            used.update(inside)
    return tables, used


def _layout_page(page: pymupdf.Page) -> tuple[list, list[pymupdf.Rect]]:
    figures = _drawing_figures(page) + _image_figures(page)
    wide, short = _rules(page, figures)
    raw_lines = _page_glyphs(page, [])
    figures = [_expand_with_text(f, raw_lines) if f.height > 30 else f for f in figures]
    lines = _page_glyphs(page, figures)
    bands = _bands(lines)
    tables, used = _collect_tables(bands, wide)
    spans = [(t.top, t.top + t.height) for t in tables]
    math = _math_regions([b for i, b in enumerate(bands) if i not in used], spans, page.get_drawings())
    if math:
        figures.extend(math)
        lines = _page_glyphs(page, figures)
        bands = _bands(lines)
        tables, used = _collect_tables(bands, wide)
    flow: list = list(tables)
    free = [b for i, b in enumerate(bands) if i not in used]
    i = 0
    while i < len(free):
        b = free[i]
        toc = _toc_parts(b)
        if toc:
            h = _line_height(b)
            flow.append(Para("toc", b.base - BASE_K * h, h, [b], h, toc=toc))
            i += 1
            continue
        hang = _hanging_group(free, i)
        if hang:
            flow.append(hang)
            i += len(hang.bands)
            continue
        group = [b]
        pitch = None
        while i + len(group) < len(free) and _can_join(group[-1], free[i + len(group)], pitch):
            nxt = free[i + len(group)]
            pitch = pitch or (nxt.base - group[-1].base)
            group.append(nxt)
        if len(group) > 1:
            flow.append(Para("text", b.base - BASE_K * pitch, pitch * len(group), group, pitch, "justify"))
        else:
            h = _line_height(b)
            align = "center" if _is_centered(b) else "left"
            if len(b.segs) == 1 and align == "left" and b.x1 > X1 - 3 and b.x0 > X0 + 150:
                align = "right"
            para = Para("text", b.base - BASE_K * h, h, [b], h, align)
            para.blanks = [r for r in short if abs(r.y0 - b.base) < 4]
            flow.append(para)
        i += len(group)
    flow.sort(key=lambda e: e.top)
    return flow, figures


# ---------------------------------------------------------------- docx writing


def _exact(pf, height: float) -> None:
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(max(height, 1.0))
    pf.space_after = Pt(0)


def _band_pieces(b: Band, seg_index: int) -> list[tuple[str, tuple]]:
    return _strip_pieces(_runs(b, b.segs[seg_index]))


def _write_para(doc: Document, para: Para, before: float, page_break: bool):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if page_break:
        pf.page_break_before = True
    pf.space_before = Pt(max(0.0, before))
    pf.widow_control = False
    _exact(pf, para.pitch)
    first = para.bands[0]
    if para.kind == "toc":
        toc = para.toc
        pf.left_indent = Pt(max(0.0, first.x0 - X0))
        text = (toc["number"] + "\t" if toc["number"] else "") + toc["title"] + "\t" + toc["page"]
        if toc["title_x"] is not None:
            pf.tab_stops.add_tab_stop(Pt(toc["title_x"] - X0), WD_TAB_ALIGNMENT.LEFT)
        pf.tab_stops.add_tab_stop(Pt(toc["page_x1"] - X0), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        _apply_style(p.add_run(text), toc["style"])
        return p
    if para.kind == "hang":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        body_x = para.body_x
        pf.left_indent = Pt(body_x - X0)
        pf.first_line_indent = Pt(first.x0 - body_x)
        label = _strip_pieces(_runs(first, para.label_glyphs))
        _emit_runs(p, label)
        style = label[-1][1] if label else _style_of(first, first.glyphs[0])
        used = sum(_text_width(t, s) for t, s in label) + _text_width(" ", style)
        gap = p.add_run(" ")
        _apply_style(gap, style)
        extra = (body_x - first.x0) - used
        if abs(extra) > 0.2:
            spacing = OxmlElement("w:spacing")
            spacing.set(qn("w:val"), str(int(round(extra * 20))))
            gap_rpr = gap._element.get_or_add_rPr()
            gap_rpr.append(spacing)
            _order_rpr(gap_rpr)
        _emit_runs(p, _strip_pieces(_runs(first, para.body_glyphs)))
        for b in para.bands[1:]:
            p.add_run().add_break(WD_BREAK.LINE)
            _emit_runs(p, _band_pieces(b, 0))
        return p
    if para.align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        rest = [b.x0 for b in para.bands[1:]]
        left = min(rest) if rest else first.x0
        pf.left_indent = Pt(max(0.0, left - X0))
        if abs(first.x0 - left) > 1.5:
            pf.first_line_indent = Pt(first.x0 - left)
        for k, b in enumerate(para.bands):
            _emit_runs(p, _band_pieces(b, 0))
            if k < len(para.bands) - 1:
                p.add_run().add_break(WD_BREAK.LINE)
        return p
    if para.align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _emit_runs(p, _band_pieces(first, 0))
        return p
    if para.align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pf.right_indent = Pt(max(0.0, X1 - first.x1))
        _emit_runs(p, _band_pieces(first, 0))
        return p
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Pt(max(0.0, first.x0 - X0))
    blanks = sorted(getattr(para, "blanks", []), key=lambda r: r.x0)
    for k in range(len(first.segs)):
        seg = first.segs[k]
        if k > 0:
            prev_end = first.segs[k - 1][-1].x1
            for blank in blanks:
                if prev_end - 2 <= blank.x0 and blank.x1 <= seg[0].x0 + 2:
                    pf.tab_stops.add_tab_stop(Pt(blank.x1 - X0), WD_TAB_ALIGNMENT.LEFT, WD_TAB_LEADER.LINES)
                    p.add_run("\t")
            last = k == len(first.segs) - 1
            if last and seg[-1].x1 > X1 - 3 and seg[0].x0 > X1 - 70:
                pf.tab_stops.add_tab_stop(Pt(seg[-1].x1 - X0), WD_TAB_ALIGNMENT.RIGHT)
            else:
                pf.tab_stops.add_tab_stop(Pt(seg[0].x0 - X0), WD_TAB_ALIGNMENT.LEFT)
            p.add_run("\t")
        _emit_runs(p, _band_pieces(first, k))
    tail = first.segs[-1][-1].x1
    for blank in blanks:
        if blank.x0 >= tail - 2:
            pf.tab_stops.add_tab_stop(Pt(blank.x1 - X0), WD_TAB_ALIGNMENT.LEFT, WD_TAB_LEADER.LINES)
            p.add_run("\t")
            tail = blank.x1
    return p


def _set_cell_border(cell, edge: str, width_pt: float) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    borders = tcpr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcpr.append(borders)
    el = borders.find(qn(f"w:{edge}"))
    if el is None:
        el = OxmlElement(f"w:{edge}")
        borders.append(el)
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(max(2, int(round(width_pt * 8)))))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "000000")


def _table_xml_setup(table, widths: list[float], indent: float) -> None:
    tbl = table._tbl
    tblpr = tbl.tblPr
    for tag in ("w:tblBorders", "w:tblCellMar", "w:tblLayout", "w:tblInd", "w:tblW"):
        old = tblpr.find(qn(tag))
        if old is not None:
            tblpr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "nil")
        borders.append(e)
    tblpr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblpr.append(layout)
    ind = OxmlElement("w:tblInd")
    ind.set(qn("w:w"), str(int(round(indent * 20))))
    ind.set(qn("w:type"), "dxa")
    tblpr.append(ind)
    tblw = OxmlElement("w:tblW")
    tblw.set(qn("w:w"), str(int(round(sum(widths) * 20))))
    tblw.set(qn("w:type"), "dxa")
    tblpr.append(tblw)
    mar = OxmlElement("w:tblCellMar")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:w"), "0")
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblpr.append(mar)
    grid = tbl.tblGrid
    for col, w in zip(grid.findall(qn("w:gridCol")), widths):
        col.set(qn("w:w"), str(int(round(w * 20))))


def _write_table(doc: Document, block: TableBlock) -> None:
    widths = [b2 - b1 for b1, b2 in zip(block.bounds, block.bounds[1:])]
    widths.append(max(20.0, block.right - block.bounds[-1] + 2))
    ncols = len(widths)
    table = doc.add_table(rows=len(block.rows), cols=ncols)
    table.style = None
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    _table_xml_setup(table, widths, block.bounds[0] - X0)
    edges = list(block.bounds) + [block.bounds[-1] + widths[-1]]
    col_align: list[str] = []
    for ci in range(ncols):
        segs = [ln[ci] for row in block.rows for ln in row["lines"] if ln.get(ci)]
        if len(segs) < 2:
            col_align.append("left")
            continue
        spread = {
            "left": max(s[0].x0 for s in segs) - min(s[0].x0 for s in segs),
            "right": max(s[-1].x1 for s in segs) - min(s[-1].x1 for s in segs),
            "center": max((s[0].x0 + s[-1].x1) / 2 for s in segs) - min((s[0].x0 + s[-1].x1) / 2 for s in segs),
        }
        best = "left" if spread["left"] < 2.5 else min(spread, key=spread.get)
        col_align.append(best)
    for ri, row in enumerate(block.rows):
        tr = table.rows[ri]
        trpr = tr._tr.get_or_add_trPr()
        height = OxmlElement("w:trHeight")
        height.set(qn("w:val"), str(int(round(row["height"] * 20))))
        height.set(qn("w:hRule"), "exact")
        trpr.append(height)
        cant = OxmlElement("w:cantSplit")
        trpr.append(cant)
        for ci in range(ncols):
            cell = tr.cells[ci]
            cell.width = Pt(widths[ci])
            p = cell.paragraphs[0]
            pf = p.paragraph_format
            pf.space_before = Pt(0)
            pf.widow_control = False
            _exact(pf, block.pitch)
            lines = [ln.get(ci) for ln in row["lines"]]
            while lines and lines[-1] is None:
                lines.pop()
            if not lines:
                continue
            present = [seg for seg in lines if seg]
            lo, hi = edges[ci], edges[ci + 1]
            if col_align[ci] == "right":
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                pf.right_indent = Pt(max(0.0, hi - max(s[-1].x1 for s in present)))
            elif col_align[ci] == "center":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                mid = statistics.mean((s[0].x0 + s[-1].x1) / 2 for s in present)
                shift = 2 * mid - lo - hi
                if shift >= 0:
                    pf.left_indent = Pt(shift)
                else:
                    pf.right_indent = Pt(-shift)
            else:
                offset = present[0][0].x0 - lo
                if offset > 1.0:
                    pf.left_indent = Pt(offset)
            for li, glyphs in enumerate(lines):
                if li > 0:
                    p.add_run().add_break(WD_BREAK.LINE)
                if glyphs:
                    band = row["bands"][li]
                    _emit_runs(p, _strip_pieces(_runs(band, glyphs)))
    for idx, edge, width in block.rules:
        for cell in table.rows[idx].cells:
            _set_cell_border(cell, edge, width)


def _anchor_picture(paragraph, page: pymupdf.Page, rect: pymupdf.Rect, key: int) -> None:
    clip = pymupdf.Rect(rect.x0 - 1.5, rect.y0 - 1.5, rect.x1 + 1.5, rect.y1 + 1.5) & page.rect
    pymupdf.TOOLS.set_graphics_min_line_width(2.2)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=clip, alpha=False)
    pymupdf.TOOLS.set_graphics_min_line_width(0)
    run = paragraph.add_run()
    run.add_picture(io.BytesIO(pix.tobytes("png")), width=Pt(clip.width), height=Pt(clip.height))
    inline = run._element.find(".//" + qn("wp:inline"))
    anchor = OxmlElement("wp:anchor")
    for k, v in {
        "distT": "0", "distB": "0", "distL": "0", "distR": "0", "simplePos": "0",
        "relativeHeight": str(251658240 + key), "behindDoc": "0", "locked": "0",
        "layoutInCell": "1", "allowOverlap": "1",
    }.items():
        anchor.set(k, v)
    simple = OxmlElement("wp:simplePos")
    simple.set("x", "0")
    simple.set("y", "0")
    anchor.append(simple)
    for axis, value in (("H", clip.x0), ("V", clip.y0)):
        pos = OxmlElement(f"wp:position{axis}")
        pos.set("relativeFrom", "page")
        off = OxmlElement("wp:posOffset")
        off.text = str(int(Emu(Pt(value))))
        pos.append(off)
        anchor.append(pos)
    for child in list(inline):
        if child.tag == qn("wp:extent"):
            anchor.append(copy.deepcopy(child))
            effect = OxmlElement("wp:effectExtent")
            for side in ("l", "t", "r", "b"):
                effect.set(side, "0")
            anchor.append(effect)
            anchor.append(OxmlElement("wp:wrapNone"))
        elif child.tag in (qn("wp:docPr"), qn("wp:cNvGraphicFramePr"), qn("a:graphic")):
            anchor.append(copy.deepcopy(child))
    inline.getparent().replace(inline, anchor)


def _configure(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Pt(595.276)
    sec.page_height = Pt(841.89)
    sec.left_margin = Pt(X0)
    sec.right_margin = Pt(595.276 - X1)
    sec.top_margin = Pt(TOP_MARGIN)
    sec.bottom_margin = Pt(BOTTOM_MARGIN)
    sec.header_distance = Pt(0)
    sec.footer_distance = Pt(0)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)
    if rpr.find(qn("w:noProof")) is None:
        rpr.append(OxmlElement("w:noProof"))
    lang = rpr.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        rpr.append(lang)
    lang.set(qn("w:val"), "vi-VN")
    _order_rpr(rpr)
    pf = normal.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    settings = doc.settings.element
    compat = settings.find(qn("w:compat"))
    if compat is None:
        compat = OxmlElement("w:compat")
        settings.append(compat)


_PAGE_NO_TEXT = re.compile(r"^(?:[ivxlcdm]+|\d{1,3})$")


def _is_page_number(element) -> bool:
    if not isinstance(element, Para) or element.kind != "text" or len(element.bands) != 1:
        return False
    b = element.bands[0]
    return b.base > 755 and len(b.segs) == 1 and _PAGE_NO_TEXT.match(b.text().strip()) is not None


def _write_page_number(doc: Document, anchor_para, para: Para, key: int) -> None:
    """Page numbers sit at a fixed spot and may overlap text, like a LaTeX footer.

    A no-wrap text box does that; a frame would push the body text below it.
    """
    from docx.oxml import parse_xml

    b = para.bands[0]
    height = _line_height(b)
    text = b.text().strip()
    style = _style_of(b, next(g for g in b.glyphs if g.ch.strip()))
    half_points = int(round(style[3] * 2))
    emu = lambda pt: int(round(pt * 12700))  # noqa: E731
    width = X1 - X0
    top = b.base - BASE_K * height
    xml = f"""
<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
     xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
     xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
     xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
  <w:drawing>
    <wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" relativeHeight="{251659000 + key}"
               behindDoc="0" locked="0" layoutInCell="1" allowOverlap="1">
      <wp:simplePos x="0" y="0"/>
      <wp:positionH relativeFrom="page"><wp:posOffset>{emu(X0)}</wp:posOffset></wp:positionH>
      <wp:positionV relativeFrom="page"><wp:posOffset>{emu(top)}</wp:posOffset></wp:positionV>
      <wp:extent cx="{emu(width)}" cy="{emu(height)}"/>
      <wp:effectExtent l="0" t="0" r="0" b="0"/>
      <wp:wrapNone/>
      <wp:docPr id="{900000 + key}" name="PageNumber {key}"/>
      <wp:cNvGraphicFramePr/>
      <a:graphic>
        <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
          <wps:wsp>
            <wps:cNvSpPr txBox="1"/>
            <wps:spPr>
              <a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu(width)}" cy="{emu(height)}"/></a:xfrm>
              <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              <a:noFill/>
              <a:ln><a:noFill/></a:ln>
            </wps:spPr>
            <wps:txbx>
              <w:txbxContent>
                <w:p>
                  <w:pPr>
                    <w:spacing w:before="0" w:after="0" w:line="{int(round(height * 20))}" w:lineRule="exact"/>
                    <w:jc w:val="center"/>
                  </w:pPr>
                  <w:r>
                    <w:rPr>
                      <w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>
                      <w:noProof/>
                      <w:sz w:val="{half_points}"/>
                      <w:szCs w:val="{half_points}"/>
                    </w:rPr>
                    <w:t>{text}</w:t>
                  </w:r>
                </w:p>
              </w:txbxContent>
            </wps:txbx>
            <wps:bodyPr rot="0" vert="horz" wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t">
              <a:noAutofit/>
            </wps:bodyPr>
          </wps:wsp>
        </a:graphicData>
      </a:graphic>
    </wp:anchor>
  </w:drawing>
</w:r>"""
    anchor_para._p.append(parse_xml(xml))


def _spacer(doc: Document, height: float, page_break: bool):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if page_break:
        pf.page_break_before = True
    pf.space_before = Pt(0)
    _exact(pf, max(1.0, height))
    run = p.add_run("")
    run.font.size = Pt(1)
    return p


def build(src: Path = SRC, out: Path = OUT) -> dict:
    pdf = pymupdf.open(src)
    doc = Document()
    _configure(doc)
    body = doc.element.body
    for child in list(body):
        if child.tag == qn("w:p"):
            body.remove(child)
    figures_total = 0
    tables_total = 0
    for index, page in enumerate(pdf):
        flow, figures = _layout_page(page)
        footers = [e for e in flow if _is_page_number(e)]
        flow = [e for e in flow if not _is_page_number(e)]
        cursor = TOP_MARGIN
        anchor_para = None
        first = True
        for element in flow:
            if isinstance(element, TableBlock):
                gap = element.top - cursor
                spacer = _spacer(doc, gap, first and index > 0)
                anchor_para = anchor_para or spacer
                cursor += max(1.0, gap)
                _write_table(doc, element)
                cursor += element.height
                tables_total += 1
            else:
                before = element.top - cursor
                p = _write_para(doc, element, before, first and index > 0)
                anchor_para = anchor_para or p
                cursor += max(0.0, before) + element.height
            first = False
        if anchor_para is None:
            anchor_para = _spacer(doc, 1.0, index > 0)
        for footer in footers:
            _write_page_number(doc, anchor_para, footer, index)
        for k, rect in enumerate(sorted(figures, key=lambda r: r.y0)):
            _anchor_picture(anchor_para, page, rect, index * 100 + k)
            figures_total += 1
        if index % 20 == 0:
            print(f"page {index + 1}/{pdf.page_count}", flush=True)
    try:
        doc.save(out)
    except PermissionError:
        alt = out.with_name(out.stem + "_fixed.docx")
        doc.save(alt)
        out = alt
        print("locked; wrote", alt, flush=True)
    pages = pdf.page_count
    pdf.close()
    return {"pages": pages, "figures": figures_total, "tables": tables_total, "bytes": out.stat().st_size, "path": str(out)}


# ---------------------------------------------------------------- verification


def _expected_toc_pages(pdf_path: Path = SRC) -> dict:
    """Printed page numbers of three TOC/LOF entries, read from the source PDF's outline and page labels."""
    pdf = pymupdf.open(pdf_path)
    expected = {}
    for _level, title, page_no in pdf.get_toc():
        if "Lựa chọn nguồn tín hiệu" in title:
            expected.setdefault("e1", pdf[page_no - 1].get_label())
        if "KẾT QUẢ ÂM VÀ GIỚI HẠN" in title:
            expected.setdefault("c", pdf[page_no - 1].get_label())
    for page in pdf:
        if "Hình B.1:" in page.get_text():
            expected["b1"] = page.get_label()
            break
    pdf.close()
    return expected


def verify(path: Path = OUT) -> dict:
    expected = _expected_toc_pages()
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs]
    fonts = set()
    for p in doc.paragraphs:
        for run in p.runs:
            if run.font.name:
                fonts.add(run.font.name)
    cells = []
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                cells.append(cell.text)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if run.font.name:
                            fonts.add(run.font.name)
    text = "\n".join(parts + cells)
    checks = {
        "cover_city": "THÀNH PHỐ HỒ CHÍ MINH" in text,
        "student": "Nguyễn Thị Kiều Thanh" in text,
        "vietnamese_u_horn": all(w in text for w in ("Đức", "cứu", "ngữ ký hiệu", "từ video", "tối ưu")),
        "no_broken_u_horn": not any(w in text for w in ("Đfíc", "cfíu", "ngfi", "tfi video", "ffu")),
        "top1": "51,55" in text,
        "speedup": "1,220" in text,
        "n_test": "6.165" in text,
        "ci_contains_zero": "chứa 0" in text,
        "optimal_name": "Suy luận tối ưu" in text,
        "no_glued_city": "PHỐHỒCHÍ" not in text,
        "no_glued_name": "ThịKiều" not in text,
        "fonts_allowed": fonts <= {FONT, MATH_FONT, MONO_FONT},
        "toc_e1_line": bool(expected.get("e1")) and any("Lựa chọn nguồn tín hiệu" in t and "\t" in t and t.rstrip().endswith("\t" + expected["e1"]) for t in parts),
        "toc_c_line": bool(expected.get("c")) and any(t.startswith("Phụ lục C.") and "\t" in t and t.rstrip().endswith("\t" + expected["c"]) for t in parts),
        "figure_b1_line": bool(expected.get("b1")) and any("Hình B.1" in t and "\t" in t and t.rstrip().endswith("\t" + expected["b1"]) for t in parts),
        "date_line": any("ngày" in t and "tháng" in t and "năm 2026" in t for t in parts),
        "abbr_precision_same_cell": any("Automatic Mixed" in c and "Precision" in c for c in cells),
    }
    checks["page_breaks"] = sum(1 for p in doc.paragraphs if p.paragraph_format.page_break_before)
    checks["fonts"] = sorted(fonts)
    checks["failed"] = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    return checks


def main() -> int:
    info = build()
    checks = verify(Path(info["path"]))
    print(info)
    print(checks)
    return 0 if not checks["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
