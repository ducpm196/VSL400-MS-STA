# -*- coding: utf-8 -*-
"""Generate static defense flow figures (SVG + PNG) from SoT v6 labels."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Docs" / "assets" / "figures"
THESIS_FIG = ROOT / "Docs" / "thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
THESIS_FIG.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def rounded(draw, xy, fill, outline, width=2, radius=10):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, x1, y1, x2, y2, fill=(28, 25, 23)):
    draw.line((x1, y1, x2, y2), fill=fill, width=2)
    # simple arrowhead
    if abs(y2 - y1) >= abs(x2 - x1):
        # vertical
        s = 1 if y2 > y1 else -1
        draw.polygon([(x2, y2), (x2 - 5, y2 - 8 * s), (x2 + 5, y2 - 8 * s)], fill=fill)
    else:
        s = 1 if x2 > x1 else -1
        draw.polygon([(x2, y2), (x2 - 8 * s, y2 - 5), (x2 - 8 * s, y2 + 5)], fill=fill)


def wrap_text(draw, text, fnt, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_box(draw, x, y, w, h, title, lines, fill, outline, title_f, body_f):
    rounded(draw, (x, y, x + w, y + h), fill=fill, outline=outline, width=2)
    draw.text((x + 10, y + 8), title, fill=outline, font=title_f)
    yy = y + 28
    for ln in lines:
        for wrapped in wrap_text(draw, ln, body_f, w - 20):
            draw.text((x + 10, yy), wrapped, fill=(28, 25, 23), font=body_f)
            yy += 16


def save_both(img: Image.Image, stem: str):
    png = FIG / f"{stem}.png"
    img.save(png, "PNG", optimize=True)
    (THESIS_FIG / f"{stem}.png").write_bytes(png.read_bytes())
    # lightweight SVG wrapper referencing dimensions (raster embedded as note — also write pure SVG later)
    return png


def fig_m1_full_flow():
    W, H = 1400, 1750
    img = Image.new("RGB", (W, H), (250, 250, 249))
    d = ImageDraw.Draw(img)
    t = font(22, True)
    h = font(16, True)
    b = font(14)
    d.text((40, 24), "Hinh M1 — Full-flow MS-STA v6 / B4-V6 (Input → Output)", fill=(15, 76, 92), font=t)

    # A
    draw_box(d, 80, 70, 1240, 90, "A · INPUT", [
        "Video Front · 1 signer · 1 gloss/clip · VSL400 · 400 lop closed-set · eligible N=24.752",
    ], (234, 242, 248), (23, 54, 93), h, b)
    arrow(d, 700, 160, 700, 190)

    # B dual
    draw_box(d, 80, 195, 600, 170, "B1 · RGB-token v3", [
        "Cua so thoi gian → SlowFast dong bang → ROIAlign (global/face/L-hand/R-hand)",
        "Output: RGB cache L x 4 x 2304 (niem phong)",
    ], (236, 254, 255), (14, 116, 144), h, b)
    draw_box(d, 720, 195, 600, 170, "B2 · Skeleton v3", [
        "Holistic + C3 hand recovery → 75 joints + validity",
        "Toa do + van toc + xuong → Skeleton cache NPZ",
    ], (255, 247, 237), (194, 65, 12), h, b)
    arrow(d, 380, 365, 380, 395)
    arrow(d, 1020, 365, 1020, 395)
    d.line((380, 395, 1020, 395), fill=(28, 25, 23), width=2)
    arrow(d, 700, 395, 700, 425)

    draw_box(d, 80, 430, 1240, 55, "Quality scan D0", [
        "Chi dung cache PASS — cung sample-id cho moi variant",
    ], (245, 245, 244), (28, 25, 23), h, b)
    arrow(d, 700, 485, 700, 515)

    draw_box(d, 80, 520, 1240, 130, "C · MODEL STAGE (equal-budget / CTR-matched)", [
        "RGB-strong · B3 CTR concat · B3+MLP CTR (doi chung manh) · B4-V6 (DE XUAT) · E4 e4_w192 (phu luc NO-GO)",
        "Moi variant doc CUNG cache D0",
    ], (254, 243, 199), (180, 83, 9), h, b)
    arrow(d, 700, 650, 700, 680)

    draw_box(d, 80, 685, 1240, 120, "D · VALIDATION GATE (3 seed 42/123/456)", [
        "B4-V6 vs B3+MLP CTR · thang 3/3 seed · mean Delta = +4,27 pp · CI95 [+2,94; +5,69]",
        "PASS → khoa ckpt seed 42   |   FAIL → NO_GO (khong mo test/demo)",
    ], (241, 245, 249), (15, 23, 42), h, b)
    arrow(d, 700, 805, 700, 835)

    draw_box(d, 80, 840, 1240, 130, "E · TEST retrospective signer-disjoint (KHONG pristine blind)", [
        "Signer 24–28 · n=6.165 · oneshot seed 42",
        "B4-V6: Top-1 51,55% · Top-5 73,88% · Macro-F1 51,37% · Delta +9,33 pp · CI95 [+6,37; +11,34]",
        "McNemar 1047 vs 472 · p ≈ 2,8e-50",
    ], (220, 252, 231), (22, 101, 52), h, b)
    arrow(d, 700, 970, 700, 1000)

    draw_box(d, 80, 1005, 1240, 110, "F · DEMO fail-closed", [
        "D4 15/15 · SHA checkpoint + lineage cache v3",
        "UI: RGB Front || Skeleton 75 joints · GT tach khoi Top-1/Top-5",
        "Sai SHA / thieu cache / clip ngoai demo → DUNG, khong fallback",
    ], (236, 254, 255), (14, 116, 144), h, b)
    arrow(d, 700, 1115, 700, 1145)

    draw_box(d, 80, 1150, 1240, 130, "G · OUTPUT", [
        "Softmax 400 lop → Top-1 / Top-5 → nguoi xac nhan",
        "Claim hop le: B4-V6 outperform B3+MLP CTR tren protocol cuc bo = DEFENSE_GO",
        "KHONG claim: SOTA paper 76–93% · pristine blind · real-time · production · dich cau · E4 nhanh hon",
    ], (209, 250, 229), (4, 120, 87), h, b)

    d.text((40, 1310), "SoT: Docs/B4_CURRENT_REPORT.md · Campaign DEFENSE_GO · Model b4_v6_ctr", fill=(87, 83, 78), font=b)
    d.text((40, 1335), "ASCII labels for portable fonts; Vietnamese prose lives in TeX/HTML captions.", fill=(87, 83, 78), font=b)

    # Also write a taller cropped version without footer waste
    cropped = img.crop((0, 0, W, 1380))
    save_both(cropped, "m1_full_flow_v6")
    return cropped


def fig_m2_b4v6():
    W, H = 1400, 520
    img = Image.new("RGB", (W, H), (250, 250, 249))
    d = ImageDraw.Draw(img)
    t = font(22, True)
    h = font(15, True)
    b = font(13)
    d.text((40, 20), "Hinh M2 — Inside B4-V6 (MS-STA + bounded residual)", fill=(15, 76, 92), font=t)

    boxes = [
        (40, 70, 200, 90, "RGB tokens v3", (236, 254, 255), (14, 116, 144)),
        (280, 70, 220, 90, "Region proj +\ntemporal attn", (236, 254, 255), (14, 116, 144)),
        (40, 200, 200, 90, "Skeleton v3", (255, 247, 237), (194, 65, 12)),
        (280, 200, 220, 90, "Spatial →\ntemporal attn", (255, 247, 237), (194, 65, 12)),
        (560, 130, 260, 100, "Cross-attn\nSkeleton → RGB", (254, 243, 199), (180, 83, 9)),
        (860, 130, 200, 100, "Stream gate", (254, 243, 199), (180, 83, 9)),
        (560, 280, 260, 90, "B3 CTR residual base", (245, 245, 244), (28, 25, 23)),
        (860, 280, 220, 90, "Bounded ReZero", (254, 243, 199), (180, 83, 9)),
        (1120, 200, 240, 100, "400 logits →\nSoftmax Top-k", (220, 252, 231), (22, 101, 52)),
    ]
    for x, y, w, hh, title, fill, outline in boxes:
        rounded(d, (x, y, x + w, y + hh), fill=fill, outline=outline, width=2)
        lines = title.split("\n")
        yy = y + (hh - 18 * len(lines)) // 2
        for ln in lines:
            tw = d.textlength(ln, font=h)
            d.text((x + (w - tw) / 2, yy), ln, fill=outline, font=h)
            yy += 18

    arrow(d, 240, 115, 275, 115)
    arrow(d, 240, 245, 275, 245)
    arrow(d, 500, 115, 560, 160)
    arrow(d, 500, 245, 560, 200)
    arrow(d, 820, 180, 855, 180)
    arrow(d, 1060, 180, 1115, 230)
    arrow(d, 820, 325, 855, 325)
    arrow(d, 1080, 325, 1120, 280)
    d.text((40, 400), "CTR-matched: B3+MLP shares Skeleton encoder budget; no RGB-temporal/cross/gate/residual.", fill=(87, 83, 78), font=b)
    d.text((40, 425), "Proposed model only: B4-V6 (b4_v6_ctr). E4 is appendix NO-GO.", fill=(87, 83, 78), font=b)
    save_both(img, "m2_b4v6_internals")
    return img


def fig_m3_fsm():
    W, H = 1200, 900
    img = Image.new("RGB", (W, H), (250, 250, 249))
    d = ImageDraw.Draw(img)
    t = font(22, True)
    h = font(15, True)
    b = font(13)
    d.text((40, 20), "Hinh M3 — Operational state machine (happy / fail-closed / forbidden)", fill=(15, 76, 92), font=t)

    steps = [
        (400, 70, "0 Lock protocol + metrics + gates"),
        (400, 150, "1 Rebuild D0 features v3"),
        (400, 230, "2 HPO equal-budget (dev only)"),
        (400, 310, "3 Confirm 3 seeds"),
        (400, 390, "4 VALIDATION GATE"),
    ]
    for x, y, label in steps:
        draw_box(d, x, y, 400, 55, label, [], (236, 254, 255), (15, 76, 92), h, b)
        if y < 390:
            arrow(d, 600, y + 55, 600, y + 75)

    # branches
    draw_box(d, 80, 520, 320, 90, "FAIL → NO_GO", [
        "No test / no outperform claim",
    ], (254, 226, 226), (153, 27, 27), h, b)
    draw_box(d, 800, 520, 320, 90, "PASS → lock seed 42", [
        "Oneshot test → D4 15/15 → demo",
    ], (220, 252, 231), (22, 101, 52), h, b)
    draw_box(d, 800, 650, 320, 70, "DEFENSE_GO (SoT)", [
        "B4-V6 live registry",
    ], (209, 250, 229), (4, 120, 87), h, b)

    d.line((500, 445, 240, 520), fill=(153, 27, 27), width=2)
    d.line((700, 445, 960, 520), fill=(22, 101, 52), width=2)
    arrow(d, 960, 610, 960, 645)

    draw_box(d, 80, 750, 1040, 100, "FORBIDDEN", [
        "Open test before gate · cherry-pick seed · fallback historical checkpoint",
        "Compare 84,8%/51,6% to paper 76–93% as same protocol · present E4 as proposed model",
    ], (255, 247, 237), (194, 65, 12), h, b)
    save_both(img, "m3_ops_fsm_v6")
    return img


def write_svg_m1():
    """Pure SVG (offline HTML) — Vietnamese labels."""
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 1180" font-family="Segoe UI, Arial, sans-serif">
  <rect width="900" height="1180" fill="#fafaf9"/>
  <text x="24" y="28" fill="#0f4c5c" font-size="16" font-weight="700">Hình M1 · Full-flow MS-STA v6 / B4-V6 (Input → Output)</text>
  <rect x="40" y="48" width="820" height="56" fill="#eaf2f8" stroke="#17365d" stroke-width="2" rx="6"/>
  <text x="56" y="72" fill="#17365d" font-size="13" font-weight="700">A · INPUT</text>
  <text x="56" y="92" fill="#1c1917" font-size="12">Video Front · 1 gloss/clip · 400 lớp · eligible N=24.752 · signer-independent</text>
  <line x1="450" y1="104" x2="450" y2="122" stroke="#1c1917" stroke-width="2" marker-end="url(#a)"/>
  <defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#1c1917"/></marker></defs>

  <rect x="40" y="124" width="390" height="100" fill="#ecfeff" stroke="#0e7490" stroke-width="2" rx="6"/>
  <text x="56" y="148" fill="#0e7490" font-size="13" font-weight="700">B1 · RGB-token v3</text>
  <text x="56" y="168" fill="#1c1917" font-size="12">SlowFast đóng băng → ROIAlign 4 vùng</text>
  <text x="56" y="188" fill="#44403c" font-size="11">global / face / L-hand / R-hand → cache</text>
  <text x="56" y="208" fill="#44403c" font-size="11">L × 4 × 2304</text>

  <rect x="470" y="124" width="390" height="100" fill="#fff7ed" stroke="#c2410c" stroke-width="2" rx="6"/>
  <text x="486" y="148" fill="#9a3412" font-size="13" font-weight="700">B2 · Skeleton v3</text>
  <text x="486" y="168" fill="#1c1917" font-size="12">75 khớp + validity + vận tốc</text>
  <text x="486" y="188" fill="#44403c" font-size="11">Holistic + phục hồi tay C3</text>
  <text x="486" y="208" fill="#44403c" font-size="11">→ Skeleton NPZ niêm phong</text>

  <line x1="235" y1="224" x2="235" y2="242" stroke="#1c1917" stroke-width="2"/>
  <line x1="665" y1="224" x2="665" y2="242" stroke="#1c1917" stroke-width="2"/>
  <line x1="235" y1="242" x2="665" y2="242" stroke="#1c1917" stroke-width="2"/>
  <line x1="450" y1="242" x2="450" y2="258" stroke="#1c1917" stroke-width="2" marker-end="url(#a)"/>

  <rect x="40" y="260" width="820" height="44" fill="#f5f5f4" stroke="#1c1917" stroke-width="1.5" rx="6"/>
  <text x="56" y="288" fill="#1c1917" font-size="12" font-weight="700">Quality scan D0 — chỉ cache PASS, cùng sample-id cho mọi variant</text>
  <line x1="450" y1="304" x2="450" y2="322" stroke="#1c1917" stroke-width="2" marker-end="url(#a)"/>

  <rect x="40" y="324" width="820" height="72" fill="#fef3c7" stroke="#b45309" stroke-width="2" rx="6"/>
  <text x="56" y="348" fill="#92400e" font-size="13" font-weight="700">C · MODEL (CTR-matched / equal-budget)</text>
  <text x="56" y="370" fill="#1c1917" font-size="12">RGB-strong · B3 CTR concat · B3+MLP CTR (đối chứng) · B4-V6 (đề xuất) · E4 (phụ lục NO-GO)</text>
  <text x="56" y="388" fill="#44403c" font-size="11">Mọi biến thể đọc cùng cache D0</text>
  <line x1="450" y1="396" x2="450" y2="414" stroke="#1c1917" stroke-width="2" marker-end="url(#a)"/>

  <rect x="40" y="416" width="820" height="88" fill="#f1f5f9" stroke="#0f172a" stroke-width="2" rx="6"/>
  <text x="56" y="440" fill="#0f172a" font-size="13" font-weight="700">D · VALIDATION GATE · 3 seed {42,123,456}</text>
  <text x="56" y="462" fill="#1c1917" font-size="12">B4-V6 84,81±0,51% vs B3+MLP 80,54±1,57% · Δ+4,27 pp · CI95 [+2,94; +5,69]</text>
  <text x="56" y="484" fill="#166534" font-size="12" font-weight="700">PASS → khóa ckpt seed 42     FAIL → NO_GO (không mở test/demo)</text>
  <line x1="450" y1="504" x2="450" y2="522" stroke="#1c1917" stroke-width="2" marker-end="url(#a)"/>

  <rect x="40" y="524" width="820" height="100" fill="#dcfce7" stroke="#166534" stroke-width="2" rx="6"/>
  <text x="56" y="548" fill="#14532d" font-size="13" font-weight="700">E · TEST hồi cứu signer-disjoint (không pristine blind)</text>
  <text x="56" y="570" fill="#1c1917" font-size="12">Signer 24–28 · n=6.165 · oneshot seed 42</text>
  <text x="56" y="590" fill="#1c1917" font-size="12">B4-V6 Top-1 51,55% · Top-5 73,88% · Macro-F1 51,37% · Δ+9,33 pp · CI [+6,37; +11,34]</text>
  <text x="56" y="610" fill="#1c1917" font-size="12">McNemar 1047 vs 472 · p ≈ 2,8×10⁻⁵⁰</text>
  <line x1="450" y1="624" x2="450" y2="642" stroke="#1c1917" stroke-width="2" marker-end="url(#a)"/>

  <rect x="40" y="644" width="820" height="88" fill="#ecfeff" stroke="#0e7490" stroke-width="2" rx="6"/>
  <text x="56" y="668" fill="#0e7490" font-size="13" font-weight="700">F · DEMO fail-closed · D4 15/15</text>
  <text x="56" y="690" fill="#1c1917" font-size="12">Khớp SHA + lineage cache · UI RGB ‖ Skeleton · GT tách Top-k</text>
  <text x="56" y="712" fill="#991b1b" font-size="12">Sai SHA / thiếu cache / clip ngoài → DỪNG, không fallback lịch sử</text>
  <line x1="450" y1="732" x2="450" y2="750" stroke="#1c1917" stroke-width="2" marker-end="url(#a)"/>

  <rect x="40" y="752" width="820" height="100" fill="#d1fae5" stroke="#047857" stroke-width="2" rx="6"/>
  <text x="56" y="776" fill="#064e3b" font-size="13" font-weight="700">G · OUTPUT</text>
  <text x="56" y="798" fill="#1c1917" font-size="12">Softmax 400 lớp → Top-1 / Top-5 → người xác nhận</text>
  <text x="56" y="818" fill="#166534" font-size="12" font-weight="700">Claim: B4-V6 outperform B3+MLP CTR trên protocol cục bộ = DEFENSE_GO</text>
  <text x="56" y="838" fill="#991b1b" font-size="11">Không: SOTA paper · pristine blind · real-time · production · dịch câu · E4 đề xuất</text>

  <text x="40" y="880" fill="#57534e" font-size="11">SoT: B4_CURRENT_REPORT.md · FINAL_RESULTS.json · Campaign DEFENSE_GO · b4_v6_ctr</text>
  <text x="40" y="900" fill="#57534e" font-size="11">Nguồn Mermaid chỉnh sửa: Docs/MSSTA_DEFENSE_FULL_FLOW_MERMAID.md</text>
</svg>
'''
    path = FIG / "m1_full_flow_v6.svg"
    path.write_text(svg, encoding="utf-8")
    (THESIS_FIG / "m1_full_flow_v6.svg").write_text(svg, encoding="utf-8")
    return path


def main():
    fig_m1_full_flow()
    fig_m2_b4v6()
    fig_m3_fsm()
    write_svg_m1()
    # copy svg companions for m2/m3 as note pointing to png
    for stem in ("m2_b4v6_internals", "m3_ops_fsm_v6"):
        note = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 60"><text x="10" y="35" font-size="14">See {stem}.png (raster companion)</text></svg>'
        (FIG / f"{stem}.svg").write_text(note, encoding="utf-8")
    print("Wrote figures to", FIG)
    print("Copied to", THESIS_FIG)


if __name__ == "__main__":
    main()
