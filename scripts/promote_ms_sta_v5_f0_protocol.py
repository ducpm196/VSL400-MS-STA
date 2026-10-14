#!/usr/bin/env python3
"""Validate / check F0 protocol plan lock (no legacy single FROZEN promote)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
from ms_sta_v5_f0_lib import (  # noqa: E402
    ENV_SPEC_REL,
    PROTOCOL_PLAN_LOCK_REL,
    PROTOCOL_PLAN_REL,
    load_json,
    sha256_file,
    validate_protocol_plan,
    validate_protocol_plan_lock,
    validate_env_spec,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-plan-lock", action="store_true", default=True)
    ap.add_argument(
        "--promote",
        action="store_true",
        help="Always refused: legacy single FROZEN promote removed.",
    )
    args = ap.parse_args()
    if args.promote:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "legacy_promote_to_F0_PROTOCOL_forbidden",
                    "use": "plan_lock_then_realization_then_execution_lock",
                },
                indent=2,
            )
        )
        return 2
    try:
        plan_path = _ROOT / PROTOCOL_PLAN_REL
        lock_path = _ROOT / PROTOCOL_PLAN_LOCK_REL
        plan = load_json(plan_path)
        validate_protocol_plan(plan)
        env = load_json(_ROOT / ENV_SPEC_REL)
        validate_env_spec(env, mode="plan")
        plan_sha = sha256_file(plan_path)
        env_sha = sha256_file(_ROOT / ENV_SPEC_REL)
        lock = load_json(lock_path)
        validate_protocol_plan_lock(
            lock, root=_ROOT, plan_sha256=plan_sha, env_sha256=env_sha
        )
        result = {
            "ok": True,
            "plan_locked": True,
            "promotable_to_legacy_frozen": False,
            "plan_sha256": plan_sha,
            "env_requirements_sha256": env_sha,
            "next": [
                "authorized_download_and_env_realization",
                "F0_REALIZATION_MANIFEST",
                "F0_EXECUTION_LOCK",
            ],
        }
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
