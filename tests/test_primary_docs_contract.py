from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_DOCS = (
    "Docs/B4_CURRENT_REPORT.md",
    "Docs/Research_Overview.md",
    "Docs/Strategy_Implementation.md",
    "Docs/phases/03_MS-STA.md",
    "Docs/phases/05_Evaluation.md",
)


def test_primary_docs_state_v6_defense_go_and_b4_v6() -> None:
    for relative in PRIMARY_DOCS:
        text = (ROOT / relative).read_text(encoding="utf-8-sig")
        assert "DEFENSE_GO" in text, relative
        assert "Unified Attention" in text or "Mô hình Chú ý Hợp nhất" in text, relative
        assert "51,96 ± 0,83%" not in text, relative
        assert "60,01 ± 0,96%" not in text, relative

    report = (ROOT / "Docs/B4_CURRENT_REPORT.md").read_text(encoding="utf-8-sig")
    assert "84,81" in report
    assert "51,55" in report
    assert "+9,33" in report or "9,33" in report
    assert "E4_ACCURACY_NO_GO" in report
