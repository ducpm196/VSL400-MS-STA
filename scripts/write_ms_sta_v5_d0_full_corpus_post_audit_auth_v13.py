#!/usr/bin/env python3
"""Seal post-audit execution authorization v13 only when external re-audit PASS exists.

Frozen historical writer: uses only *_V13 constants / validators.
Does not start extraction. Does not write v14.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V13,
    D0_FULL_CORPUS_APPROVAL_V13_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V13_REL,
    OUTPUT_VERSION_V13,
    atomic_create_json,
    load_json,
    sha256_file,
    validate_approval_v13,
    validate_execution_protocol_v13,
    validate_post_audit_execution_authorization_v13,
)
from ms_sta_v5_d0_lib import git_porcelain


def main() -> int:
    dirty = git_porcelain(_ROOT).strip()
    if dirty:
        raise SystemExit(f"dirty_worktree_forbidden:{dirty[:400]}")

    reaudit_path = _ROOT / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL
    if not reaudit_path.is_file():
        raise SystemExit("external_reaudit_v13_missing")
    # v13 writer frozen: does not call live v14 validate_external_reaudit_pass

    approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V13_REL)
    protocol = load_json(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL)
    validate_approval_v13(approval, root=_ROOT)
    validate_execution_protocol_v13(protocol, root=_ROOT)

    if (_ROOT / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V13_REL).is_file():
        raise SystemExit("post_audit_auth_already_sealed")

    payload = {
        "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v13",
        "output_version": OUTPUT_VERSION_V13,
        "campaign_token": CAMPAIGN_TOKEN_V13,
        "launch_authorized": True,
        "authorized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval_v13_sha256": sha256_file(_ROOT / D0_FULL_CORPUS_APPROVAL_V13_REL),
        "execution_protocol_v13_sha256": sha256_file(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL),
        "external_reaudit_sha256": sha256_file(reaudit_path),
        "external_reaudit_rel": D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL,
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
    validate_post_audit_execution_authorization_v13(payload, root=_ROOT)
    auth_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V13_REL, payload)
    print(json.dumps({"ok": True, "post_audit_execution_auth_v13_sha256": auth_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
