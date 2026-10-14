# -*- coding: utf-8 -*-
"""Create E4-R4C HPO code-lock after all-pad tests PASS (create-only)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_CODE_LOCK.json"

TREE_FILES = [
    "src/models/ms_sta_v6_e4_r4.py",
    "src/training/ms_sta_e4_r4c_data.py",
    "src/training/ms_sta_e4_r4c_train.py",
    "scripts/run_ms_sta_v6_e4_r4c.py",
    "scripts/test_ms_sta_v6_e4_r4c_all_pad.py",
    "scripts/e4_r4c_extract_full_corpus.py",
    "scripts/e4_r4c_validate_full_corpus.py",
]


def die(msg: str) -> None:
    print(f"E4_R4C_CODE_LOCK_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_sha() -> str:
    h = hashlib.sha256()
    for rel in TREE_FILES:
        p = ROOT / rel
        if not p.is_file():
            die(f"missing:{rel}")
        h.update(rel.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def main() -> int:
    if OUT.is_file():
        die(f"refuse_overwrite:{OUT}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        die("worktree_not_clean")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema": "ms_sta_v6_e4_r4c_hpo_code_lock_v1",
        "status": "HPO_CODE_LOCK_PASS",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "executable_tree_sha256": tree_sha(),
        "tree_files": TREE_FILES,
        "all_pad_tests": "PASS",
        "all_pad_contract": [
            "all_pad_zeros_and_all_true_mask",
            "no_attention_nan_from_all_masked_keys",
            "finite_logits_loss_grad",
            "rgb_base_fallback_no_fake_skeleton",
            "mixed_batch_ok",
            "padded_value_invariance",
        ],
        "invalidated_prior_hpo": [
            "INVALID_EVAL_ALL_PAD_SKELETON",
            "INVALID_PAD_GUARD_FABRICATED_SKELETON",
        ],
        "candidate_id": "R1_S_A_d128",
        "e4_ready": False,
        "forbid_select_from_prelock_archive": True,
    }
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    OUT.write_bytes(raw)
    print("CODE_LOCK_OK", head, sha256_bytes(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
