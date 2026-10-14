#!/usr/bin/env python3
"""Phase-0 worktree manifest for V5-E2 preflight."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True)
    paths: list[str] = []
    for line in out.splitlines():
        p = line[3:].strip().strip('"')
        if p.startswith("Docs/_tmp"):
            continue
        if p.endswith("/"):
            continue
        fp = _ROOT / p
        if fp.is_file():
            paths.append(p.replace("\\", "/"))
    for p in (
        "Docs/reviews/CURSOR_MS_STA_V5_E2_LONG_TASK.md",
        "Docs/reviews/MS_STA_V5_E2_ARCHITECTURE_AUDIT.md",
        "Docs/reviews/MS_STA_V5_SKELETON_STABILITY_REVIEW.md",
    ):
        if (_ROOT / p).is_file() and p not in paths:
            paths.append(p)
    rows = []
    h = hashlib.sha256()
    for p in sorted(set(paths)):
        b = (_ROOT / p).read_bytes()
        d = hashlib.sha256(b).hexdigest()
        h.update(p.encode())
        h.update(d.encode())
        rows.append({"path": p, "sha256": d, "bytes": len(b)})
    man = {
        "schema": "ms_sta_v5_worktree_manifest_v1",
        "purpose": "phase0_preflight_before_e2",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aggregate_sha256": h.hexdigest(),
        "n_files": len(rows),
        "files": rows,
    }
    out_dir = _ROOT / "Docs/results/ms_sta_v5_e2_preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(man, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    (out_dir / "WORKTREE_MANIFEST.json").write_bytes(raw)
    print(json.dumps({"aggregate_sha256": h.hexdigest(), "n_files": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
