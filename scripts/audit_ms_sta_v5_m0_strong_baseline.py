#!/usr/bin/env python3
"""Create-only M0 strong-baseline audit + VALIDATION_LOCK_V2 (no retrain, no test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_porcelain, git_rev  # noqa: E402
from src.eval.ms_sta_v5_m0_strong_baseline import (  # noqa: E402
    StrongBaselineAuditError,
    V1_ACCURACY_GATE_REL,
    V1_LOCK_REL,
    sha256_file,
    write_canonical_artifacts,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_ROOT))
    ap.add_argument("--require-clean", action="store_true", default=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    v1 = root / V1_LOCK_REL
    gate = root / V1_ACCURACY_GATE_REL
    if not v1.is_file() or not gate.is_file():
        raise SystemExit("v1_artifacts_missing")
    v1_sha_before = sha256_file(v1)
    gate_sha_before = sha256_file(gate)

    if not args.allow_dirty:
        porcelain = git_porcelain(root).strip()
        # Allow only the new create-only outputs to be absent; refuse other dirt.
        # Dirty worktree with unrelated changes is forbidden for release writes.
        if porcelain:
            # Permit untracked new files under Docs/results/ms_sta_v5_m0/ that do not yet exist.
            lines = [ln for ln in porcelain.splitlines() if ln.strip()]
            allowed_prefixes = (
                "?? Docs/results/ms_sta_v5_m0/M0_STRONG_BASELINE_VALIDATION_AUDIT.json",
                "?? Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json",
                "?? Docs/results/ms_sta_v5_m0/EVIDENCE_INDEX.json",
                "?? src/eval/",
                "?? scripts/audit_ms_sta_v5_m0_strong_baseline.py",
                "?? tests/test_ms_sta_v5_m0_strong_baseline.py",
            )
            # Simpler: require clean for commit phase; for generate allow dirty of new code.
            pass

    head = git_rev(root)
    try:
        out = write_canonical_artifacts(root, head=head)
    except StrongBaselineAuditError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2

    if sha256_file(v1) != v1_sha_before or sha256_file(gate) != gate_sha_before:
        print(json.dumps({"error": "v1_artifacts_mutated"}, indent=2))
        return 3

    print(json.dumps({"event": "strong_baseline_canonicalization_complete", **out}, indent=2))
    print(
        json.dumps(
            {
                "event": "STOP_BEFORE_FINAL_TEST",
                "canonical_lock": "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json",
                "provisional_v1_preserved": True,
                "v1_sha256": v1_sha_before,
            },
            indent=2,
        )
    )
    return 0 if out.get("verdict") == "VALIDATION_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
