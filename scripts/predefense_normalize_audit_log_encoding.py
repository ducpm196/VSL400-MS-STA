# -*- coding: utf-8 -*-
"""Normalize predefense TEST_EXECUTION_RAW_V2.log to UTF-8 (no BOM) if UTF-16."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "Docs" / "results" / "project_predefense_audit" / "TEST_EXECUTION_RAW_V2.log"
META = ROOT / "Docs" / "results" / "project_predefense_audit" / "TEST_EXECUTION_LOG_ENCODING_FIX.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    raw = LOG.read_bytes()
    before = sha256_bytes(raw)
    encoding_detected = "unknown"
    text: str
    if raw.startswith(b"\xff\xfe") or (len(raw) >= 4 and raw[1] == 0 and raw[3] == 0):
        encoding_detected = "utf-16-le"
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xfe\xff"):
        encoding_detected = "utf-16-be"
        text = raw.decode("utf-16-be")
    else:
        encoding_detected = "utf-8"
        text = raw.decode("utf-8")
    # Normalize newlines to \n; preserve visible test output only
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    out = text.encode("utf-8")  # no BOM
    after = sha256_bytes(out)
    LOG.write_bytes(out)
    META.write_text(
        json.dumps(
            {
                "schema": "predefense_test_execution_log_encoding_fix_v1",
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "path": str(LOG.relative_to(ROOT)).replace("\\", "/"),
                "encoding_detected_before": encoding_detected,
                "encoding_after": "utf-8-no-bom",
                "sha256_before": before,
                "sha256_after": after,
                "reason": "PowerShell Tee-Object wrote UTF-16 LE; convert for git text/diff hygiene without changing test result text",
                "content_semantics_unchanged": True,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"before": before, "after": after, "encoding": encoding_detected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
