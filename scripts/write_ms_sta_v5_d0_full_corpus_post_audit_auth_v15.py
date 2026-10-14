#!/usr/bin/env python3
"""Seal post-audit execution authorization v15 only when external re-audit PASS exists.

Clean worktree required. Create-only. Does not start extraction / M0 / train.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
    OUTPUT_VERSION_V15,
    atomic_create_json,
    load_json,
    sha256_file,
    validate_approval_v15,
    validate_execution_protocol_v15,
    validate_external_reaudit_pass_v15,
    validate_post_audit_execution_authorization_v15,
)
from ms_sta_v5_d0_lib import git_porcelain


def main() -> int:
    dirty = git_porcelain(_ROOT).strip()
    if dirty:
        raise SystemExit(f"dirty_worktree_forbidden:{dirty[:400]}")

    reaudit_path = _ROOT / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    if not reaudit_path.is_file():
        raise SystemExit("external_reaudit_v15_missing")
    reaudit = load_json(reaudit_path)
    try:
        validate_external_reaudit_pass_v15(reaudit, root=_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL)
    protocol = load_json(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL)
    validate_approval_v15(approval, root=_ROOT)
    validate_execution_protocol_v15(protocol, root=_ROOT)

    if (_ROOT / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL).is_file():
        raise SystemExit("post_audit_auth_v15_already_sealed")

    payload = {
        "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v15",
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "launch_authorized": True,
        "authorized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approval_v15_sha256": sha256_file(_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL),
        "execution_protocol_v15_sha256": sha256_file(_ROOT / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        "external_reaudit_sha256": sha256_file(reaudit_path),
        "external_reaudit_rel": D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
        "code_lock_git_revision": approval["code_lock_git_revision"],
        "executable_tree_sha256": approval["executable_tree_sha256"],
        "binary_root_rel": approval["binary_root_rel"],
        "evidence_root_rel": approval["evidence_root_rel"],
        "allow_m0": False,
        "allow_training": False,
        "allow_hpo": False,
        "allow_test": False,
        "allow_demo": False,
        "allow_auto_chain": False,
        "auto_chain_to_m0": False,
        "notes": [
            "Post-audit execution authorization after strict external re-audit PASS (v15).",
            "Does not start extraction; runner still requires explicit --launch.",
            "Does not authorize M0/train/HPO/test/demo/auto-chain.",
            "Empty binary root; do not reuse prior diagnostic NPZ.",
        ],
    }
    validate_post_audit_execution_authorization_v15(payload, root=_ROOT)
    auth_sha = atomic_create_json(_ROOT / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL, payload)
    print(json.dumps({"ok": True, "post_audit_execution_auth_v15_sha256": auth_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
