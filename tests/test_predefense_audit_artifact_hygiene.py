# -*- coding: utf-8 -*-
"""Hygiene checks for pre-defense audit Markdown/log artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs" / "results" / "project_predefense_audit"

# Disallow C0 controls except LF (\n). CR is not allowed in committed Markdown
# (files are eol=lf via .gitattributes). Tabs are forbidden in Markdown summaries.
FORBIDDEN_IN_MD = {i for i in range(32) if i != 10}


def _assert_no_bad_controls(path: Path, *, allow_tab: bool = False) -> None:
    data = path.read_bytes()
    # Must be valid UTF-8 without BOM
    assert not data.startswith(b"\xef\xbb\xbf"), f"BOM present: {path}"
    text = data.decode("utf-8")
    forbidden = set(FORBIDDEN_IN_MD)
    if allow_tab:
        forbidden.discard(9)
    bad = sorted({ord(ch) for ch in text if ord(ch) in forbidden})
    assert not bad, f"control chars {bad} in {path}"


def test_continuation_summary_markdown_hygiene() -> None:
    path = OUT / "PREDEFENSE_CONTINUATION_SUMMARY.md"
    _assert_no_bad_controls(path, allow_tab=False)
    text = path.read_text(encoding="utf-8")
    assert "\t" not in text
    assert "`DEFENSE_READY_SCIENTIFIC_AND_CORE_CODE`" in text
    assert "`audit/pre-defense-source-review`" in text
    assert "`test_manifest_validates`" in text
    assert "\\DEFENSE_READY" not in text
    # corrupted bell/tab escapes must not appear
    assert "\x07" not in text


def test_audit_summary_markdown_hygiene() -> None:
    path = OUT / "PREDEFENSE_AUDIT_SUMMARY.md"
    _assert_no_bad_controls(path, allow_tab=False)
    assert "\t" not in path.read_text(encoding="utf-8")


def test_writer_render_uses_backticks_not_backslash() -> None:
    from scripts.predefense_write_continuation_summary import render_continuation_summary

    text = render_continuation_summary(
        terminal="DEFENSE_READY_SCIENTIFIC_AND_CORE_CODE",
        date="2026-09-01T00:00:00Z",
        audit_data_commit="75334af644a1232d92ea85edccaf401418c3a0d0",
        audit_terminal_commit="f397769ef1596fdf7c466ec5fd036c4c47fcdeb1",
        branch="audit/pre-defense-source-review",
        split_status="PASS",
        metric_status="PASS",
    )
    assert "`DEFENSE_READY_SCIENTIFIC_AND_CORE_CODE`" in text
    assert "\\DEFENSE_READY" not in text
    assert "\x07" not in text
    assert "\t" not in text


def test_test_execution_raw_v2_is_utf8_text() -> None:
    path = OUT / "TEST_EXECUTION_RAW_V2.log"
    data = path.read_bytes()
    assert not data.startswith(b"\xff\xfe"), "still UTF-16 LE"
    assert b"\x00" not in data, "NUL bytes remain"
    text = data.decode("utf-8")
    assert "[100%]" in text


def test_release_terminal_v2_json_parse_and_commit_fields() -> None:
    path = OUT / "PREDEFENSE_RELEASE_TERMINAL_V2.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    git = obj["git"]
    assert git["audited_sha"].startswith("24be0a8")
    assert git["audit_data_commit"].startswith("75334af")
    assert git["audit_terminal_commit"].startswith("f397769")
    assert "branch_tip" not in git
    assert git["branch_tip_embedded"] is False
    assert git["branch_tip_resolution_command"] == "git rev-parse audit/pre-defense-source-review"
    assert git["accounting_is_non_scientific"] is True
    assert git["commits_before_accounting_record"] == 8
    assert len(git["commits_list_before_accounting_record"]) == 8
    assert "3bfe0dd" not in "".join(git["commits_list_before_accounting_record"])
    assert git["verified_clean_worktree_sha"].startswith("f8cd564")
    assert "scientific_core" in git["clean_verify_sha"]
    assert git["clean_verify_sha"]["scientific_core"].startswith("75334af")
    assert git["clean_verify_sha"]["artifact_hygiene"].startswith("f8cd564")
