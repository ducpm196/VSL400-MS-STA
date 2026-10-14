#!/usr/bin/env python3
"""Phase-aware live smoke for D0 full-corpus launch readiness.

Not a unit test. Intended for external audit / operator preflight against a chosen root
(default: live repository). Exit 0 in report mode; nonzero only on unexpected exceptions.
Do not call this from hermetic pytest.

Binary-root reporting is approval-owned: resolve only from validated approval fields
(binary_root_rel or code_lock_git_revision). Never derive from current HEAD alone.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL,
    OUTPUT_VERSION,
    binary_root_rel,
    evaluate_launch_readiness,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def resolve_approval_owned_binary_root(
    status: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Resolve binary root only from approval-owned fields in readiness status.

    Prefer approval["binary_root_rel"] when it is a non-empty string.
    Else if approval["code_lock_git_revision"] is a 40-hex string, compute
    binary_root_rel(rev, root=root) with that revision.
    When approval is missing/invalid or fields cannot resolve a root, return
    nulls with binary_root_status="unknown".
    """
    approval = status.get("approval")
    if not isinstance(approval, dict):
        return {
            "binary_root_rel": None,
            "binary_root_exists": None,
            "binary_root_status": "unknown",
        }

    bin_rel: str | None = None
    owned = approval.get("binary_root_rel")
    if isinstance(owned, str) and owned.strip():
        bin_rel = owned
    else:
        rev = approval.get("code_lock_git_revision")
        if isinstance(rev, str) and _HEX40.match(rev):
            bin_rel = binary_root_rel(rev, root=root)

    if not bin_rel:
        return {
            "binary_root_rel": None,
            "binary_root_exists": None,
            "binary_root_status": "unknown",
        }

    return {
        "binary_root_rel": bin_rel,
        "binary_root_exists": (root / bin_rel).exists(),
        "binary_root_status": "ok",
    }


def build_report(*, root: Path = _ROOT) -> dict[str, Any]:
    status = evaluate_launch_readiness(root=root)
    root_fields = resolve_approval_owned_binary_root(status, root=root)
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL
    auth_path = root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL
    return {
        "ready": bool(status.get("ready")),
        "errors": list(status.get("errors") or []),
        "output_version": status.get("output_version", OUTPUT_VERSION),
        "campaign_token": status.get("campaign_token", CAMPAIGN_TOKEN),
        "binary_root_rel": root_fields["binary_root_rel"],
        "binary_root_exists": root_fields["binary_root_exists"],
        "binary_root_status": root_fields["binary_root_status"],
        "reaudit_path": D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL,
        "reaudit_path_exists": reaudit_path.is_file(),
        "post_auth_path": D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL,
        "post_auth_exists": auth_path.is_file(),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = _ROOT
    if argv:
        root = Path(argv[0]).resolve()
    try:
        report = build_report(root=root)
    except Exception as exc:  # noqa: BLE001 — report mode: only unexpected failures are nonzero
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
