#!/usr/bin/env python3
"""Fail CI if historical campaign claims reappear in primary v6 SoT docs.

Scans README + Docs primary surfaces. Excludes historical_not_primary archives
and raw evidence trees that must remain for provenance.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

PRIMARY_GLOBS = [
    "README.md",
    "Docs/README.md",
    "Docs/B4_CURRENT_REPORT.md",
    "Docs/Research_Overview.md",
    "Docs/Strategy_Implementation.md",
    "Docs/MSSTA_END_TO_END_PIPELINE.md",
    "Docs/MSSTA_VSL400_Lock.md",
    "Docs/REPRODUCIBILITY.md",
    "Docs/references/VSL400_SOURCE_AUDIT.md",
    "Docs/phases/02_Backbone_and_Features.md",
    "Docs/phases/03_MS-STA.md",
    "Docs/phases/04_Deployment.md",
    "Docs/phases/05_Evaluation.md",
    "scripts/OPS.md",
    "src/deployment/backend_api.py",
    "src/deployment/app.py",
]

# Paths that may still contain historical numbers (provenance / archive).
SKIP_PREFIXES = (
    "Docs/_archive/",
    "Docs/results/ms_sta_defense_go/",
    "Docs/results/ms_sta_v5_defense_go/",
    "Docs/results/ms_sta_accuracy/",
    "Docs/results/ms_sta_efficiency/",
    "Docs/results/ms_sta_e1",
    "Docs/results/ms_sta_e2",
    "Docs/results/ms_sta_d4",
    "Docs/results/ms_sta_p0_claim_audit/",
    "Docs/reviews/",
)

_HISTORICAL_NEGATION = (
    "forbidden",
    "not primary",
    "không",
    "no-go",
    "validation_no_go",
    "historical_not_primary",
    "cấm",
    "do not",
    "must not",
    "thua",
    "fails",
    "fail",
    "historical",
    "rollback",
    "superseded",
    "archive",
    "provenance",
    "no defense",
    "không dùng",
    "refuse",
    "exits 2",
    "exit 2",
)

# Also forbid live E2 absolute KPI / stale v3 SoT / superseded v5 primary as live SoT.
FORBIDDEN = [
    (re.compile(r"current-best\s+B4", re.I), "current-best B4"),
    (re.compile(r"B4\s+vượt\s+B3", re.I), "B4 vượt B3"),
    (re.compile(r"B4R\s+vượt\s+trội", re.I), "B4R vượt trội"),
    (re.compile(r"B4\s*>\s*B3"), "B4>B3"),
    (re.compile(r"51[,.]96"), "51.96 historical fair-B4"),
    (re.compile(r"60[,.]01"), "60.01 historical B4-only"),
    (re.compile(r"40[,.]50"), "40.50 historical E1 B0"),
    (re.compile(r"49[,.]38"), "49.38 historical E1 B3"),
    (re.compile(r"23[,.]84"), "23.84 historical E2 B4"),
    (
        re.compile(
            r"(?i)(?:current-best|canonical reference)[^\n]{0,80}62[,.]29"
        ),
        "stale v3 62.29 as live SoT",
    ),
    (
        re.compile(r"(?i)Live SoT:\s*retrain-v3"),
        "stale Live SoT retrain-v3",
    ),
    (
        re.compile(r"(?i)Evidence:\s*`?Docs/results/ms_sta_retrain_v3/?`?\s*$"),
        "stale evidence path v3 as sole live",
    ),
    (re.compile(r"64[,.]75"), "v5 validation 64.75 as live primary"),
    (re.compile(r"36[,.]06"), "v5 test 36.06 as live primary"),
    (
        re.compile(r"(?i)ms_sta_v5_defense_go"),
        "v5 defense_go path as live primary",
    ),
    (
        re.compile(r"ms_sta_v5_b4_star_fixed"),
        "v5 engine id as live primary",
    ),
    (
        re.compile(
            r"(?i)(?:\bactive\b|\bproposed\b|\bprimary\b|\blive\b)[^\n]{0,80}b4_star_fixed"
            r"|b4_star_fixed[^\n]{0,80}(?:\bactive\b|\bproposed\b|\bprimary\b|\blive\b)"
        ),
        "stale b4_star_fixed as primary",
    ),
]

REQUIRED_IN_CURRENT_REPORT = [
    "DEFENSE_GO",
    "51,55",
    "84,81",
    "b4_v6_ctr",
    "ms_sta_v6_defense_go",
    "E4_ACCURACY_NO_GO",
]

REQUIRED_TOKENS_BY_FILE = {
    "scripts/OPS.md": ["DEFENSE_GO", "b4_v6_ctr", "demo_vsl400.ps1", "51,55", "retrospective"],
    "Docs/MSSTA_END_TO_END_PIPELINE.md": ["b4_v6_ctr", "B3+MLP", "cache-backed offline", "retrospective"],
    "Docs/MSSTA_VSL400_Lock.md": ["DEFENSE_GO", "51,55", "84,81"],
    "Docs/Research_Overview.md": ["DEFENSE_GO", "retrospective", "b4_v6_ctr"],
}


def _report_has_token(text: str, token: str) -> bool:
    if token in text:
        return True
    alt = token.replace(",", ".")
    return alt in text


def _iter_primary_files() -> list[Path]:
    out: list[Path] = []
    for rel in PRIMARY_GLOBS:
        p = _ROOT / rel
        if p.is_file():
            out.append(p)
    return out


def main() -> int:
    hits: list[str] = []
    report = _ROOT / "Docs/B4_CURRENT_REPORT.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8", errors="replace")
        for token in REQUIRED_IN_CURRENT_REPORT:
            if not _report_has_token(text, token):
                hits.append(f"Docs/B4_CURRENT_REPORT.md: missing required SoT token {token!r}")
        if "E4" not in text and "E4_ACCURACY_NO_GO" not in text:
            hits.append("Docs/B4_CURRENT_REPORT.md: missing required SoT token 'E4'")
    for rel, tokens in REQUIRED_TOKENS_BY_FILE.items():
        path = _ROOT / rel
        if not path.is_file():
            hits.append(f"{rel}: missing required primary file")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in tokens:
            if not _report_has_token(text, token):
                hits.append(f"{rel}: missing required SoT token {token!r}")
    for path in _iter_primary_files():
        rel = path.relative_to(_ROOT).as_posix()
        if any(rel.startswith(p) for p in SKIP_PREFIXES):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for rx, label in FORBIDDEN:
            for m in rx.finditer(text):
                start = max(0, m.start() - 120)
                window = text[start : m.end() + 80].lower()
                if any(neg in window for neg in _HISTORICAL_NEGATION):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel}:{line}: {label} -> {m.group(0)!r}")
    if hits:
        print("[FAIL] claim-lint primary SoT violations:")
        for h in hits:
            print(" ", h)
        return 1
    print("[OK] claim-lint: primary SoT aligned with MS-STA v6 Defense GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
