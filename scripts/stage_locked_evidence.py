"""Force-stage locked JSON with exact working-tree bytes (stdin hash-object).

Uses `git hash-object --stdin` so core.autocrlf cannot rewrite CRLF science locks.
Requires `.gitattributes` `-text` on these paths so later `git add` does not convert.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

LOCKED = [
    "Docs/results/ms_sta_e1_policy/E1_POLICY.json",
    "Docs/results/ms_sta_e1_policy/E1_POLICY_v1.json",
    "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json",
    "Docs/results/ms_sta_e1_v2/CHECKSUMS.json",
    "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json",
    "Docs/results/ms_sta_e2_lock/E2_LOCK_MANIFEST.json",
    "Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json",
    "Docs/results/ms_sta_d4/D4_CANDIDATE_SMOKE.json",
    "Docs/results/ms_sta_defense_go/FINAL_RESULTS.json",
    "Docs/results/ms_sta_defense_go/EVIDENCE_INDEX.json",
    "Docs/results/ms_sta_defense_go/ARTIFACT_MANIFEST.json",
    "Docs/results/ms_sta_e1_v1_diagnostic/E1_V1_DIAGNOSTIC.json",
    "Docs/results/ms_sta_p0_claim_audit/AUDIT.json",
    "Docs/results/ms_sta_p0_claim_audit/CHECKSUMS.json",
    "Docs/results/gate3/CHECKSUMS.json",
    "Docs/results/gate3/GATE3_REPORT.json",
    "data/processed/ms_sta_registry.json",
    "data/processed/vsl400_e1_split_manifest.json",
    "data/processed/vsl400_eligible_sample_manifest.json",
    "data/processed/vsl400_labels.json",
    "data/processed/demo_set_ms_sta.json",
    "data/processed/demo_set_option_b.json",
]


def _stage(rel: str) -> str:
    path = _ROOT / rel
    raw = path.read_bytes()
    blob = subprocess.check_output(
        ["git", "hash-object", "-w", "--stdin"],
        input=raw,
        cwd=_ROOT,
    ).decode().strip()
    subprocess.check_call(
        ["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}"],
        cwd=_ROOT,
    )
    # Sanity: index bytes must match worktree
    ix = subprocess.check_output(["git", "show", f":{rel}"], cwd=_ROOT)
    if ix != raw:
        raise SystemExit(f"index mismatch after stage: {rel}")
    return blob


def main() -> int:
    subprocess.check_call(["git", "add", ".gitattributes"], cwd=_ROOT)
    for rel in LOCKED:
        if not (_ROOT / rel).is_file():
            print(f"skip missing {rel}")
            continue
        blob = _stage(rel)
        print(f"staged {rel} blob={blob[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
