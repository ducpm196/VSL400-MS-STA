"""Fail-closed freeze helper for retrain-v4 immutable policy.

Does NOT train. Expects a reviewed freeze **candidate** (from
``promote_ms_sta_retrain_v4_policy_candidate.py``) or an already-promoted draft
payload. Copies into RETRAIN_POLICY.json only after validator + executable tree
bindings pass.

Code-lock model:
  - ``executable_code_lock`` = parent *code* git revision (tip before policy commit)
  - ``executable_tree_sha256`` = content hash of ``src/`` + ``scripts/`` (policy excluded)
  - ``policy_commit`` / policy SHA are bound *outside* the policy file in
    POLICY_FREEZE_BINDING.json (no self-referential policy commit inside policy)
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
    require_code_lock_fields,
    require_frozen_v4_policy,
)

DEFAULT_DRAFT = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json"
DEFAULT_OUT = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY.json"
DEFAULT_BINDING = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _repo_rel_or_fail(path: Path, *, label: str) -> str:
    """Frozen policy / binding must live inside the repo."""
    try:
        return path.resolve().relative_to(_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside repo root, got {path}") from exc


def _git_rev() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True)
    return bool(out.strip())


def _git_rev_exists(rev: str) -> bool:
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--draft",
        type=Path,
        default=DEFAULT_DRAFT,
        help="Freeze candidate JSON (may be absolute path outside repo, e.g. %%TEMP%%)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--binding-out", type=Path, default=DEFAULT_BINDING)
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Required to replace an existing RETRAIN_POLICY.json",
    )
    args = ap.parse_args()

    draft_path = args.draft if args.draft.is_absolute() else _ROOT / args.draft
    out_path = args.out if args.out.is_absolute() else _ROOT / args.out
    binding_path = (
        args.binding_out if args.binding_out.is_absolute() else _ROOT / args.binding_out
    )
    if not draft_path.is_file():
        print(f"[FAIL] missing draft: {draft_path}")
        return 2
    if out_path.exists() and not args.allow_overwrite:
        print(f"[FAIL] refuse overwrite existing {out_path} without --allow-overwrite")
        return 2
    if _git_dirty():
        print("[FAIL] clean worktree required before freeze")
        return 2

    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    # Strip self-referential policy_commit if present so validator can apply tree lock.
    code_lock = dict(draft.get("code_lock") or {})
    code_lock.pop("policy_commit", None)
    draft["code_lock"] = code_lock

    try:
        policy_rel = _repo_rel_or_fail(out_path, label="--out")
        binding_rel = _repo_rel_or_fail(binding_path, label="--binding-out")
        require_code_lock_fields(code_lock)
        require_frozen_v4_policy(draft, root=_ROOT)
    except ValueError as exc:
        print(f"[FAIL] freeze validator rejected candidate: {exc}")
        print(
            "[FAIL] Run scripts/promote_ms_sta_retrain_v4_policy_candidate.py "
            "--out $env:TEMP\\MS_STA_RETRAIN_V4_POLICY_CANDIDATE.json first, keep the "
            "worktree clean, then freeze with --draft pointing at that absolute path."
        )
        return 2

    head = _git_rev()
    parent = str(code_lock.get("executable_code_lock") or "").strip()
    if not _git_rev_exists(parent):
        print(f"[FAIL] executable_code_lock={parent!r} is not a resolvable git commit")
        return 2
    # Parent code tip must equal current HEAD at freeze time (policy not yet committed).
    if parent != head:
        print(
            f"[FAIL] executable_code_lock={parent!r} != HEAD {head}; "
            "freeze against the parent code tip, then commit policy separately"
        )
        return 2

    tree_actual = executable_tree_sha256(_ROOT)
    tree_expected = str(code_lock.get("executable_tree_sha256") or "")
    if tree_actual != tree_expected:
        print(
            f"[FAIL] executable_tree_sha256 mismatch: actual={tree_actual} "
            f"expected={tree_expected}"
        )
        return 2

    text = json.dumps(draft, ensure_ascii=False, indent=2) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    digest = sha256_file(out_path)

    binding = {
        "schema": "ms_sta_retrain_v4_policy_freeze_binding_v1",
        "policy_path": policy_rel,
        "policy_sha256": digest,
        "executable_code_lock": parent,
        "executable_tree_sha256": tree_actual,
        "policy_commit": "pending_commit_that_adds_immutable_policy_only",
        "candidate_source": _display_path(draft_path),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Step 1 complete. Next: commit RETRAIN_POLICY.json, then run "
            "scripts/finalize_ms_sta_retrain_v4_policy_binding.py and commit the binding "
            "separately. Do not embed policy_commit inside RETRAIN_POLICY.json."
        ),
    }
    binding_path.write_text(
        json.dumps(binding, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": policy_rel,
                "binding": binding_rel,
                "candidate_source": _display_path(draft_path),
                "sha256": digest,
                "executable_code_lock": parent,
                "executable_tree_sha256": tree_actual,
                "note": "Immutable policy written; bind policy_commit outside after git commit.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
