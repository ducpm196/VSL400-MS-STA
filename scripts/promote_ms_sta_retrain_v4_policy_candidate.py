"""Deterministic draft → freeze-candidate promotion for retrain-v4.

Does NOT freeze and does NOT set train live. Writes a validated candidate JSON
that ``freeze_ms_sta_retrain_v4_policy.py`` can consume.

Canonical runbook writes the candidate **outside** the repo (e.g. ``%TEMP%``)
so the worktree stays clean for freeze:

  python scripts/promote_ms_sta_retrain_v4_policy_candidate.py --out $env:TEMP\\...json
  python scripts/freeze_ms_sta_retrain_v4_policy.py --draft $env:TEMP\\...json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v4 import (
    executable_tree_sha256,
    promote_draft_to_freeze_candidate,
    require_frozen_v4_policy,
)

DEFAULT_DRAFT = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json"
DEFAULT_OUT = Path(tempfile.gettempdir()) / "MS_STA_RETRAIN_V4_POLICY_CANDIDATE.json"


def _git_rev() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()


def _display_path(path: Path) -> str:
    """Repo-relative when possible; otherwise absolute. Never raises."""
    try:
        return path.resolve().relative_to(_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _is_outside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(_ROOT.resolve())
        return False
    except ValueError:
        return True


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Candidate path (may be outside the repo; default: %%TEMP%%/MS_STA_RETRAIN_V4_POLICY_CANDIDATE.json)",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Required to replace an existing candidate file",
    )
    ap.add_argument(
        "--skip-validate",
        action="store_true",
        help="Write candidate without require_frozen_v4_policy (debug only)",
    )
    args = ap.parse_args()

    draft_path = args.draft if args.draft.is_absolute() else _ROOT / args.draft
    out_path = args.out if args.out.is_absolute() else (_ROOT / args.out)
    if not draft_path.is_file():
        print(f"[FAIL] missing draft: {draft_path}")
        return 2
    if out_path.exists() and not args.allow_overwrite:
        print(f"[FAIL] refuse overwrite existing {out_path} without --allow-overwrite")
        return 2

    draft_text = draft_path.read_text(encoding="utf-8")
    draft = json.loads(draft_text)
    head = _git_rev()
    tree = executable_tree_sha256(_ROOT)
    try:
        candidate = promote_draft_to_freeze_candidate(
            draft, head_rev=head, executable_tree_sha=tree
        )
        if not args.skip_validate:
            require_frozen_v4_policy(candidate, root=_ROOT)
    except ValueError as exc:
        print(f"[FAIL] promotion rejected: {exc}")
        return 2

    # Draft must remain untouched on disk.
    if draft_path.read_text(encoding="utf-8") != draft_text:
        print("[FAIL] promotion mutated draft file on disk")
        return 2

    _atomic_write_text(out_path, json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ok": True,
                "out": _display_path(out_path),
                "out_absolute": str(out_path.resolve()),
                "outside_repo": _is_outside_repo(out_path),
                "executable_code_lock": head,
                "executable_tree_sha256": tree,
                "schema": candidate.get("schema"),
                "status": candidate.get("status"),
                "train_allowed": candidate.get("train_allowed"),
                "note": (
                    "Candidate only. Keep worktree clean, then freeze with "
                    "scripts/freeze_ms_sta_retrain_v4_policy.py --draft <absolute-candidate>."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
