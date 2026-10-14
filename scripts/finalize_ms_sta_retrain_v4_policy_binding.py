"""Finalize POLICY_FREEZE_BINDING.json after the immutable policy commit.

Two-step freeze contract:
  1) ``freeze_ms_sta_retrain_v4_policy.py`` writes RETRAIN_POLICY.json + pending binding
  2) Commit the policy file, then run this script to bind ``policy_commit`` and
     commit the binding in a separate commit.

Does not train and does not modify RETRAIN_POLICY.json bytes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import sha256_file
from src.training.retrain_v4 import (
    executable_tree_sha256,
    require_exact_str,
    require_freeze_binding,
    require_hex64,
)

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY.json"
DEFAULT_BINDING = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True).strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    ap.add_argument(
        "--policy-commit",
        type=str,
        default=None,
        help="Commit that introduced RETRAIN_POLICY.json (default: auto-detect via git log)",
    )
    args = ap.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    binding_path = args.binding if args.binding.is_absolute() else _ROOT / args.binding
    if not policy_path.is_file():
        print(f"[FAIL] missing policy: {policy_path}")
        return 2
    if not binding_path.is_file():
        print(f"[FAIL] missing binding: {binding_path}")
        return 2
    if _git_dirty():
        # Allow only the binding file itself to be dirty if we are about to rewrite it.
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True)
        lines = [ln for ln in dirty.splitlines() if ln.strip()]
        allowed = {
            "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json",
        }
        bad = []
        for ln in lines:
            path = ln[3:].strip().replace("\\", "/")
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path not in allowed:
                bad.append(path)
        if bad:
            print(f"[FAIL] clean worktree required except pending binding rewrite; dirty={bad[:8]}")
            return 2

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    policy_sha = sha256_file(policy_path)
    rel = str(policy_path.resolve().relative_to(_ROOT.resolve())).replace("\\", "/")
    try:
        require_exact_str(binding.get("policy_sha256"), policy_sha, label="binding.policy_sha256")
        require_hex64(binding.get("executable_tree_sha256"), label="binding.executable_tree_sha256")
        require_exact_str(
            binding.get("executable_tree_sha256"),
            executable_tree_sha256(_ROOT),
            label="live executable_tree_sha256",
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    policy_commit = (args.policy_commit or "").strip()
    if not policy_commit:
        # Prefer the most recent commit that touched the policy path.
        try:
            policy_commit = _git("log", "-n", "1", "--pretty=%H", "--", rel)
        except Exception:
            policy_commit = ""
    if not policy_commit:
        print("[FAIL] cannot resolve policy_commit; pass --policy-commit explicitly")
        return 2
    try:
        _git("rev-parse", "--verify", f"{policy_commit}^{{commit}}")
    except Exception:
        print(f"[FAIL] policy_commit={policy_commit!r} is not a git commit")
        return 2

    try:
        blob = subprocess.check_output(["git", "show", f"{policy_commit}:{rel}"], cwd=_ROOT)
    except Exception as exc:
        print(f"[FAIL] cannot read {rel} at {policy_commit}: {exc}")
        return 2
    import hashlib

    blob_sha = hashlib.sha256(blob).hexdigest()
    if blob_sha != policy_sha:
        print(
            f"[FAIL] policy bytes at {policy_commit} sha={blob_sha} != on-disk {policy_sha}; "
            "commit the immutable policy before finalize"
        )
        return 2

    binding["policy_commit"] = policy_commit
    binding["policy_path"] = rel
    binding["finalized_at"] = datetime.now(timezone.utc).isoformat()
    binding["note"] = (
        "policy_commit bound outside RETRAIN_POLICY.json; commit this binding file separately."
    )
    binding_path.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Validate the finalized binding against HEAD (must already include policy commit).
    head = _git("rev-parse", "HEAD")
    try:
        require_freeze_binding(
            binding,
            root=_ROOT,
            policy_path=policy_path,
            policy_sha=policy_sha,
            head_rev=head,
        )
    except ValueError as exc:
        print(f"[FAIL] finalized binding failed verification: {exc}")
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "binding": str(binding_path.relative_to(_ROOT)).replace("\\", "/"),
                "policy_commit": policy_commit,
                "policy_sha256": policy_sha,
                "next": "git add POLICY_FREEZE_BINDING.json && commit binding separately",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
