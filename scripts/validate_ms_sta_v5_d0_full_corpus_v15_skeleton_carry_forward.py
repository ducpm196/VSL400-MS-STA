#!/usr/bin/env python3
"""CLI: read-only v14 skeleton carry-forward verifier for D0 v15.

Does NOT copy/hardlink/mutate v14. Exit 0 only on PASS.
Optional --write-report uses atomic create-only under Docs/results/.../v15/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
    V14_SEALED_SKELETON_BINARY_ROOT_REL,
    atomic_create_json,
    verify_skeleton_carry_forward_from_v14,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_ROOT)
    ap.add_argument(
        "--sealed-binary-root-rel",
        default=V14_SEALED_SKELETON_BINARY_ROOT_REL,
        help="Sealed v14 binary root (read-only)",
    )
    ap.add_argument(
        "--sample-shas",
        action="store_true",
        help="Verify only first 64 sample SHAs (faster smoke); default verifies ALL",
    )
    ap.add_argument(
        "--write-report",
        action="store_true",
        help="Create-only audit report under Docs/results/ms_sta_v5_d0_full/v15/",
    )
    args = ap.parse_args()
    root = args.root.resolve()

    report = verify_skeleton_carry_forward_from_v14(
        root=root,
        sealed_binary_root_rel=str(args.sealed_binary_root_rel),
        verify_all_shas=not bool(args.sample_shas),
    )
    report["schema"] = "ms_sta_v5_d0_full_corpus_v15_skeleton_carry_forward_audit_v1"
    report["producer"] = "scripts/validate_ms_sta_v5_d0_full_corpus_v15_skeleton_carry_forward.py"

    print(
        json.dumps(
            {
                "ok": report["ok"],
                "source_root_rel": report.get("source_root_rel"),
                "sha_checked": report.get("sha_checked"),
                "sha_mismatches": report.get("sha_mismatches"),
                "errors": report.get("errors") or [],
                "binds": report.get("binds"),
            },
            indent=2,
        )
    )

    if args.write_report:
        out = root / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
        if out.is_file():
            prev = json.loads(out.read_text(encoding="utf-8"))
            prev_bind = ((prev.get("binds") or {}).get("v14_halt_closeout_sha256"))
            new_bind = ((report.get("binds") or {}).get("v14_halt_closeout_sha256"))
            prev_all = bool(prev.get("verify_all_shas"))
            new_all = bool(report.get("verify_all_shas"))
            # Remediation refresh: halt SHA advance, or upgrade sample→full SHA verify.
            refresh = False
            if prev_bind and new_bind and prev_bind != new_bind and prev.get("ok") is True:
                refresh = True
            if prev.get("ok") is True and (not prev_all) and new_all and report.get("ok") is True:
                refresh = True
            if refresh:
                out.unlink()
        sha = atomic_create_json(out, report)
        print(json.dumps({"written": True, "report_rel": D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL, "sha256": sha}, indent=2))

    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
