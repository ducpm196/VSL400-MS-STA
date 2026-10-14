#!/usr/bin/env python3
"""Seal post-audit execution authorization v14 only when external re-audit PASS exists.

Refuses to write if v14 external re-audit companion is missing or not strict PASS.
Does not start extraction.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    D0_FULL_CORPUS_APPROVAL_V14_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL,
    OUTPUT_VERSION,
    atomic_create_json,
    load_json,
    sha256_file,
    validate_approval_v14,
    validate_execution_protocol_v14,
    validate_external_reaudit_pass,
    validate_post_audit_execution_authorization,
)
from ms_sta_v5_d0_lib import git_porcelain


def main() -> int:
    dirty = git_porcelain(_ROOT).strip()
    if dirty:
        raise SystemExit(f"dirty_worktree_forbidden:{dirty[:400]}")

    reaudit_path = _ROOT / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL
    if not reaudit_path.is_file():
        raise SystemExit("external_reaudit_v14_missing")
    reaudit = load_json(reaudit_path)
    try:
        validate_external_reaudit_pass(reaudit, root=_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V14_REL)
    protocol = load_json(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL)
    validate_approval_v14(approval, root=_ROOT)
    validate_execution_protocol_v14(protocol, root=_ROOT)

    if (_ROOT / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL).is_file():
        raise SystemExit("post_audit_auth_already_sealed")

    payload = {
        "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v14",
        "output_version": OUTPUT_VERSION,
        "campaign_token": CAMPAIGN_TOKEN,
        "launch_authorized": True,
        "authorized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval_v14_sha256": sha256_file(_ROOT / D0_FULL_CORPUS_APPROVAL_V14_REL),
        "execution_protocol_v14_sha256": sha256_file(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL),
        "external_reaudit_sha256": sha256_file(reaudit_path),
        "external_reaudit_rel": D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL,
        "code_lock_git_revision": approval["code_lock_git_revision"],
        "executable_tree_sha256": approval["executable_tree_sha256"],
        "binary_root_rel": approval["binary_root_rel"],
        "evidence_root_rel": approval["evidence_root_rel"],
        "notes": [
            "Post-audit execution authorization after strict external re-audit PASS (git-bound identity).",
            "Does not start extraction; runner still requires explicit --launch.",
            "Empty binary root; do not reuse prior diagnostic NPZ.",
        ],
    }
    validate_post_audit_execution_authorization(payload, root=_ROOT)
    auth_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL, payload)
    print(json.dumps({"ok": True, "post_audit_execution_auth_v14_sha256": auth_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
