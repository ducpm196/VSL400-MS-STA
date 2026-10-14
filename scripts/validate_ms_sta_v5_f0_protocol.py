#!/usr/bin/env python3
"""Validate F0 protocol DRAFT (immutable) or PLAN documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from ms_sta_v5_f0_lib import (  # noqa: E402
    PROTOCOL_DRAFT_REL,
    PROTOCOL_PLAN_REL,
    load_json,
    validate_protocol_draft,
    validate_protocol_plan,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("draft", "plan"), required=True)
    ap.add_argument("--protocol", type=Path, default=None)
    args = ap.parse_args()
    default = PROTOCOL_DRAFT_REL if args.mode == "draft" else PROTOCOL_PLAN_REL
    path = args.protocol or (_ROOT / default)
    try:
        obj = load_json(path)
        result = (
            validate_protocol_draft(obj)
            if args.mode == "draft"
            else validate_protocol_plan(obj)
        )
        result["path"] = path.as_posix()
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
