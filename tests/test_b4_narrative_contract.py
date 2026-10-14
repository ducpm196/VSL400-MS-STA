"""Primary narrative contract: the current report states the v6 Defense GO result."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8-sig")


def test_current_report_is_v6_defense_go_sot() -> None:
    report = _read("Docs/B4_CURRENT_REPORT.md")
    for token in ("DEFENSE_GO", "51,55", "84,81", "+9,33", "cache-backed offline"):
        assert token in report
    assert "Unified Attention" in report
    assert "REPRODUCIBILITY.md" in report
    assert "E4 is" in report or "E4_ACCURACY_NO_GO" in report
    assert "realtime" not in report.lower()
