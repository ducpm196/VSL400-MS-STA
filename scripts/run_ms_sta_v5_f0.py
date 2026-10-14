#!/usr/bin/env python3
"""MS-STA v5 F0 runner — plan-lock lifecycle, dry-plan default, fail-closed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from ms_sta_v5_f0_lib import ENV_SPEC_REL, WEIGHTS_REL, preflight  # noqa: E402


def dry_plan_payload(pf: dict) -> dict:
    protocol = pf["protocol"]
    return {
        "schema": "ms_sta_v5_f0_dry_plan_v5",
        "ok": True,
        "action": "dry-plan",
        "protocol_mode": pf["protocol_mode"],
        "plan_lock_status": pf["plan_lock_status"],
        "plan_sha256": pf["plan_sha256"],
        "plan_lock_sha256": pf["plan_lock_sha256"],
        "plan_audit": pf.get("plan_audit"),
        "protocol_frozen": False,
        "gpu_ready": False,
        "download_ready": False,
        "realization_authorized": pf.get("realization_authorized"),
        "env_realization_ok": pf.get("env_realization_ok"),
        "approval": {
            "sha256": pf["approval"]["approval_sha256"],
            "scope": pf["approval"]["scope"],
            "one_shot": pf["approval"]["one_shot"],
        },
        "approval_audit": pf["approval_audit"],
        "source_resolution": protocol.get("source_resolution"),
        "lifecycle": protocol.get("lifecycle"),
        "runner": protocol.get("runner"),
        "bans": protocol.get("bans"),
        "measurement_contract": protocol.get("measurement_contract"),
        "weights_manifest": {"path": WEIGHTS_REL, "ok": pf["weights_ok"], "error": pf["weights_error"]},
        "isolated_env": {"spec": ENV_SPEC_REL, "install_authorized": False},
        "executable_tree_sha256": pf["executable_tree_sha256"],
        "worktree_dirty": pf["worktree_dirty"],
        "next_required": [
            "F0_FINAL_AUDIT_recorded",
            "STOP_no_d0_auto_chain",
            "separate_d0_authorization_only_after_review",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-plan", action="store_true")
    ap.add_argument(
        "--action",
        choices=(
            "dry-plan",
            "gpu",
            "run-gpu",
            "download-weights",
            "d0",
            "train",
            "hpo",
            "test",
            "accuracy",
        ),
        default=None,
    )
    args = ap.parse_args()
    action = "dry-plan" if args.dry_plan or args.action in (None, "dry-plan") else args.action
    try:
        pf = preflight(root=_ROOT, action=action, python_exe=sys.executable)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    if action == "dry-plan":
        print(json.dumps(dry_plan_payload(pf), indent=2))
        return 0
    print(
        json.dumps(
            {
                "schema": "ms_sta_v5_f0_refuse_v4",
                "ok": False,
                "action": action,
                "refused": True,
                "reasons": pf["refuse_reasons"],
                "plan_sha256": pf["plan_sha256"],
                "approval_sha256": pf["approval"]["approval_sha256"],
            },
            indent=2,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
