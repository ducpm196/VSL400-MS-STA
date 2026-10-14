# -*- coding: utf-8 -*-
"""Claim-lint for defense docs: forbid unlabeled live v5 KPIs outside archive."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = [
    (re.compile(r"64\s*[,.]\s*75"), "v5 val Top-1 64,75"),
    (re.compile(r"36\s*[,.]\s*06"), "v5 test Top-1 36,06"),
    (re.compile(r"(?<![0-9])5\s*[,.]\s*50\s*pp"), "v5 val delta +5,50 pp"),
    (re.compile(r"(?<![0-9])6\s*[,.]\s*97\s*pp"), "v5 test delta +6,97 pp"),
    (re.compile(r"11\s*/\s*11"), "v5 D4 11/11"),
    (re.compile(r"ms_sta_registry_v5_defense_go"), "v5 registry as active"),
    (re.compile(r"VALIDATION_LOCK_V2"), "v5 validation lock"),
]

SCAN = [
    ROOT / "Docs" / "thesis" / "VSL_MSSTA_REPORT.tex",
    ROOT / "Docs" / "MSSTA_DEFENSE_FULL_FLOW_MERMAID.md",
    ROOT / "Docs" / "B4_CURRENT_REPORT.md",
    ROOT / "Docs" / "README.md",
    ROOT / "Docs" / "thesis" / "README.md",
]


def line_ok(line: str) -> bool:
    low = line.lower()
    keys = (
        "rollback",
        "historical",
        "lịch sử",
        "histvfive",
        "not primary",
        "not live",
        "không phải sot",
        "không được trình bày",
        "archive",
        "không",
        "only mentioned",
        "rollback-only",
        "không phải sot hiện tại",
    )
    return any(k in low for k in keys) or "HistVfive" in line or "\\HistVfive" in line


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    hits = []
    thesis_dir = ROOT / "Docs" / "thesis"
    for path in SCAN:
        if not path.is_file():
            if not thesis_dir.exists() and thesis_dir in path.parents:
                # Public export: thesis files are not published (VSL400 Data Usage Agreement).
                continue
            hits.append((str(path.relative_to(ROOT)), 0, "missing file", ""))
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for rx, label in FORBIDDEN:
                if rx.search(line) and not line_ok(line):
                    if "Historical MS-STA v5" in line or "rollback-only" in line:
                        continue
                    hits.append((path.relative_to(ROOT).as_posix(), i, label, line.strip()[:100]))
    if hits:
        print("CLAIM-LINT FAIL")
        for p, i, label, snip in hits:
            print(f"  {p}:{i} [{label}] {snip}")
        return 1
    print("CLAIM-LINT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

