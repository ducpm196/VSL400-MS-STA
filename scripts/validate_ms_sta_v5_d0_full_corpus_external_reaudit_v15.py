#!/usr/bin/env python3
"""CLI validator for v15 external reaudit PASS artifact (does not write PASS).

Loads existing Docs/results/.../v15/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json
and runs validate_external_reaudit_pass_v15. Optional --check-readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
    evaluate_launch_readiness_v15,
    load_json,
    validate_external_reaudit_pass_v15,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_ROOT)
    ap.add_argument(
        "--check-readiness",
        action="store_true",
        help="Also run evaluate_launch_readiness_v15 after PASS validation",
    )
    args = ap.parse_args()
    root = args.root.resolve()

    path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    if not path.is_file():
        print(json.dumps({"ok": False, "error": "external_reaudit_v15_missing"}, indent=2))
        return 2

    try:
        obj = load_json(path)
        validate_external_reaudit_pass_v15(obj, root=root)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": f"external_reaudit_v15_unreadable:{exc}"}, indent=2))
        return 2

    out: dict = {
        "ok": True,
        "external_reaudit_rel": D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
        "audit_verdict": obj.get("audit_verdict"),
    }
    if args.check_readiness:
        readiness = evaluate_launch_readiness_v15(root=root)
        out["readiness"] = {
            "ready": readiness.get("ready"),
            "errors": readiness.get("errors"),
            "extraction_authorized": readiness.get("extraction_authorized"),
            "draft_friendly": readiness.get("draft_friendly"),
        }
        if not readiness.get("ready"):
            print(json.dumps(out, indent=2))
            return 3

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
