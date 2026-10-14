# -*- coding: utf-8 -*-
"""Write pre-defense continuation summary with safe Markdown backticks (no escape corruption)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs" / "results" / "project_predefense_audit"

# Use chr(96) so generators never embed backslash-letter sequences that Python
# would interpret as escapes when authors mistakenly write \\`...\\`.
BT = chr(96)


def render_continuation_summary(
    *,
    terminal: str,
    date: str,
    audit_data_commit: str,
    audit_terminal_commit: str,
    branch: str,
    split_status: str,
    metric_status: str,
) -> str:
    """Return Markdown text. Callers must not wrap identifiers in backslash-backtick."""
    lines = [
        "# PRE-DEFENSE CONTINUATION SUMMARY",
        "",
        f"**Terminal:** {BT}{terminal}{BT}",
        f"**Date:** {date}",
        f"**Audit data commit:** {BT}{audit_data_commit}{BT}",
        f"**Audit terminal commit (pre-hygiene):** {BT}{audit_terminal_commit}{BT}",
        f"**Branch:** {BT}{branch}{BT} (not merged/pushed)",
        "",
        "## Five gates",
        "",
        "| Gate | Result |",
        "|---|---|",
        "| 1. Scope + release-critical line review | PASS (11/11 + support/tests) |",
        f"| 2. Split/leakage independent | **{split_status}** |",
        f"| 3. Metric recalc from raw dumps | **{metric_status}** |",
        "| 4. Defense-critical tests (FastAPI separated) | PASS + SKIP_POLICY production/raw-video |",
        "| 5. Clean worktree verify | PASS_WITH_EXTERNAL_ARTIFACT_SKIP |",
        "",
        "## Science locks",
        "",
        "B4 retained · E4 selected=null · e4_ready=false · test/D4/demo E4 not opened",
        "",
        "## Remaining manual",
        "",
        "- Student TeX cover fields",
        "- Production FastAPI hardening (P2)",
        f"- Optional: restore raw mp4 to green {BT}test_manifest_validates{BT}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    text = render_continuation_summary(
        terminal="DEFENSE_READY_SCIENTIFIC_AND_CORE_CODE",
        date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        audit_data_commit="75334af644a1232d92ea85edccaf401418c3a0d0",
        audit_terminal_commit="f397769ef1596fdf7c466ec5fd036c4c47fcdeb1",
        branch="audit/pre-defense-source-review",
        split_status="PASS",
        metric_status="PASS",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "PREDEFENSE_CONTINUATION_SUMMARY.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
