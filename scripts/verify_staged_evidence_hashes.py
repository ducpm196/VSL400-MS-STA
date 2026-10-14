"""Verify staged (index) blobs match working-tree + FINAL_RESULTS/EVIDENCE_INDEX.

Exit 0 only when git show :path SHA == worktree SHA == expected binding.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _index_bytes(rel: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "show", f":{rel}"], cwd=_ROOT)
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    final = json.loads(
        (_ROOT / "Docs/results/ms_sta_defense_go/FINAL_RESULTS.json").read_text(encoding="utf-8-sig")
    )
    idx = json.loads(
        (_ROOT / "Docs/results/ms_sta_defense_go/EVIDENCE_INDEX.json").read_text(encoding="utf-8-sig")
    )
    expected: dict[str, str] = {}
    for block in (final.get("e1_v2"), final.get("e2"), final.get("d4")):
        if block and block.get("path") and block.get("sha256"):
            expected[block["path"]] = block["sha256"]
    for item in idx.get("tracked_evidence") or []:
        if item.get("path") and item.get("sha256"):
            expected[item["path"]] = item["sha256"]
    # Always include policy
    expected.setdefault(
        "Docs/results/ms_sta_e1_policy/E1_POLICY.json",
        "12cf1176c94c43b9bef87d1da65cb5c911961a0afccedf2414f09fed963ae53a",
    )

    fails = 0
    for rel, exp in sorted(expected.items()):
        wt_path = _ROOT / rel
        wt = _sha_bytes(wt_path.read_bytes()) if wt_path.is_file() else None
        ix_b = _index_bytes(rel)
        ix = _sha_bytes(ix_b) if ix_b is not None else None
        ok = wt == ix == exp
        print(("OK  " if ok else "FAIL"), rel)
        print(f"     expect={exp[:16]} wt={(wt or '-')[:16]} index={(ix or '-')[:16]}")
        if not ok:
            fails += 1
    print(f"{len(expected) - fails}/{len(expected)} bindings OK")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
