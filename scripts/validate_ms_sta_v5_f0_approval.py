#!/usr/bin/env python3
"""Validate MS-STA v5 F0 approval — sealed identity fail-closed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from ms_sta_v5_f0_lib import (  # noqa: E402
    APPROVAL_REL,
    validate_approval,
    validate_approval_audit,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--approval", type=Path, default=_ROOT / APPROVAL_REL)
    ap.add_argument(
        "--diagnostic-content-only",
        action="store_true",
        help="Skip sealed-byte identity (for adversarial content unit tests only).",
    )
    ap.add_argument("--require-audit", action="store_true", default=True)
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()
    try:
        result = validate_approval(
            root=_ROOT,
            approval_path=args.approval,
            require_sealed_identity=not args.diagnostic_content_only,
        )
        if not args.skip_audit:
            result["approval_audit"] = validate_approval_audit(root=_ROOT)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
