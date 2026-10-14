# -*- coding: utf-8 -*-
"""Create B4Q_V1_BASELINE_LOCK.json (create-only) after verifying frozen hashes."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.b4q.invariants import (  # noqa: E402
    load_policy,
    sha256_file,
    verify_b4_checkpoint,
    verify_slowfast_checkpoint,
    verify_split_hashes,
)

OUT = ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_BASELINE_LOCK.json"


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"refuse_overwrite:{OUT}")
    pol = load_policy()
    b4 = verify_b4_checkpoint(pol)
    sf = verify_slowfast_checkpoint(pol)
    verify_split_hashes(pol)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema": "ms_sta_v6_b4q_v1_baseline_lock_v1",
        "status": "BASELINE_LOCK_PASS",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "git_branch": branch,
        "policy_rel": "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json",
        "policy_sha256": sha256_file(ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json"),
        "b4_checkpoint_sha256": b4,
        "slowfast_checkpoint_sha256": sf,
        "split_manifest_sha256": pol["split"]["manifest_sha256"],
        "eligible_manifest_sha256": pol["split"]["eligible_manifest_sha256"],
        "labels_sha256": pol["split"]["labels_sha256"],
        "feature_roots_read_only": pol["feature_roots_read_only"],
        "regions_order_canonical": pol["rgb_token_v3"]["regions_order_canonical"],
        "r4c_redirect_rel": pol["r4c_redirect_rel"],
        "r4c_redirect_sha256": sha256_file(ROOT / pol["r4c_redirect_rel"]),
        "e4_ready": False,
        "note": "Lock binds hashes before candidate metrics; stage profile remeasure is a separate create-only artifact.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("BASELINE_LOCK_OK", OUT, sha256_file(OUT)[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
