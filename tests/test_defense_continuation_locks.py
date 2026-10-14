"""Locks for display names, the two protocols and the official title in the thesis TeX."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "Docs/thesis/VSL_MSSTA_REPORT.tex"


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def test_tex_macros_are_display_names():
    tex = TEX.read_text(encoding="utf-8")
    assert r"\newcommand{\Bfour}{Mô hình Chú ý Hợp nhất}" in tex
    assert r"\newcommand{\Bmlp}{Mốc Kết hợp Đơn giản}" in tex
    assert r"\newcommand{\Sfive}{Suy luận tối ưu}" in tex
    assert r"\newcommand{\EfourSyskd}{Chiến dịch Tối ưu Hệ thống}" in tex
    assert r"\newcommand{\Rzero}{Suy luận cơ sở}" in tex
    assert r"\newcommand{\Rone}{tính toán độ chính xác hỗn hợp}" in tex
    assert r"\newcommand{\Sthree}{gom vùng quan tâm theo lô}" in tex
    assert r"\newcommand{\SfiveSpeedup}{1{,}220}" in tex
    assert r"\newcommand{\SfiveCILow}{-0{,}015}" in tex
    assert r"\newcommand{\SfiveCIHigh}{+0{,}147}" in tex


def test_tex_tomtat_and_conclusion_use_story_names():
    tex = TEX.read_text(encoding="utf-8")
    tomtat = _between(tex, r"\chapter*{TÓM TẮT ĐỒ ÁN}", r"\chapter{MỞ ĐẦU}")
    assert "Mô hình Chú ý Hợp nhất" in tomtat
    assert "chỉ kết luận độ đúng được giữ nguyên" in tomtat
    assert "chưa dùng đường suy luận tối ưu" in tomtat
    assert "51,55" in tomtat
    assert "1,22" in tomtat
    assert "B4/" not in tomtat
    assert "selected=S5" not in tomtat
    assert "e4_ready" not in tomtat
    ket = _between(tex, r"\chapter{KẾT LUẬN}", r"\chapter{HƯỚNG PHÁT TRIỂN}")
    assert "chứa 0" in ket
    assert "tăng tốc thực thi" in ket
    assert "xác nhận hồi cứu" in ket
    assert "Chỉ nhận dạng từ đơn" in ket
    assert "B4/" not in ket
    assert "selected=S5" not in ket
    assert "dịch câu liên tục" in tex


def test_tex_english_abstract_avoids_bare_b4v6_as_hero_name():
    tex = TEX.read_text(encoding="utf-8")
    assert r"\chapter*{Abstract}" not in tex
    assert "B4-V6" not in tex
    summary = _between(tex, r"\chapter*{TÓM TẮT ĐỒ ÁN}", r"\chapter{MỞ ĐẦU}")
    assert "Unified Attention" in summary
    assert "có chứa 0" in summary


def test_tex_and_report_use_official_thesis_title():
    tex = TEX.read_text(encoding="utf-8")
    assert r"\section{Phát biểu bài toán}" in tex
    assert r"\section{Tên đề tài}" not in tex
    assert "recognition from video" in tex
    assert r"tab:decuong-vs-sot" not in tex
    assert "không kết luận độ đúng tăng" in tex
