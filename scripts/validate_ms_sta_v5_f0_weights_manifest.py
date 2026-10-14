#!/usr/bin/env python3
"""Validate F0 weights manifest with field-by-field plan matching + file SHA recompute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from ms_sta_v5_f0_lib import (  # noqa: E402
    PROTOCOL_PLAN_REL,
    WEIGHTS_REL,
    load_json,
    sha256_file,
    validate_weights_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=_ROOT / WEIGHTS_REL)
    ap.add_argument("--plan", type=Path, default=_ROOT / PROTOCOL_PLAN_REL)
    args = ap.parse_args()
    try:
        if not args.manifest.is_file():
            raise ValueError("weights_manifest_absent")
        if not args.plan.is_file():
            raise ValueError("protocol_plan_absent")
        plan = load_json(args.plan)
        result = validate_weights_manifest(
            load_json(args.manifest),
            root=_ROOT,
            plan=plan,
            plan_sha256=sha256_file(args.plan),
            recompute_files=True,
        )
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
