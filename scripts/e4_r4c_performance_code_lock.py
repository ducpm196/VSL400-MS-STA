# -*- coding: utf-8 -*-
"""Create E4-R4C performance-only code-lock (create-only; after loader decision).

Requires clean worktree. Parent: E4_R4C_HPO_CODE_LOCK.json.
Does not change scientific policy.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_PERFORMANCE_CODE_LOCK.json"
HPO_LOCK = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_CODE_LOCK.json"
PARITY = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_PERFORMANCE_PARITY.json"
BENCH = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_DATALOADER_BENCHMARK.json"
TEACHER_PARITY = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_TEACHER_LOGITS_PARITY.json"
TEACHER_CACHE = ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/teacher_logits_confirm_train.npz"

TREE_FILES = [
    "src/models/ms_sta_v6_e4_r4.py",
    "src/training/ms_sta_e4_r4c_data.py",
    "src/training/ms_sta_e4_r4c_train.py",
    "scripts/run_ms_sta_v6_e4_r4c.py",
    "scripts/test_ms_sta_v6_e4_r4c_all_pad.py",
    "scripts/e4_r4c_extract_full_corpus.py",
    "scripts/e4_r4c_validate_full_corpus.py",
    "scripts/precompute_e4_r4c_teacher_logits.py",
    "scripts/e4_r4c_teacher_logits_parity.py",
    "scripts/e4_r4c_dataloader_benchmark.py",
    "scripts/e4_r4c_performance_parity.py",
    "scripts/e4_r4c_performance_code_lock.py",
]

POLICY_FILES = [
    "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_POLICY.json",
    "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json",
    "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_POST_HPO_BOUNDARY_POLICY.json",
]


def die(msg: str) -> None:
    print(f"E4_R4C_PERF_LOCK_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(files: list[str]) -> str:
    h = hashlib.sha256()
    for rel in files:
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
    if not HPO_LOCK.is_file():
        die("missing_hpo_lock")
    if not PARITY.is_file():
        die("missing_performance_parity")
    if not BENCH.is_file():
        die("missing_dataloader_benchmark")
    if not TEACHER_PARITY.is_file():
        die("missing_teacher_parity")
    if not TEACHER_CACHE.is_file():
        die("missing_teacher_cache")

    hpo = json.loads(HPO_LOCK.read_text(encoding="utf-8"))
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    teacher_parity = json.loads(TEACHER_PARITY.read_text(encoding="utf-8"))
    if hpo.get("status") != "HPO_CODE_LOCK_PASS":
        die("hpo_lock_not_pass")
    if teacher_parity.get("gate_c") != "PASS":
        die("teacher_parity_not_pass")
    if parity.get("gate_d_parity") != "PASS":
        die("performance_parity_gate_not_pass")
    decision = parity.get("decision") or {}
    if "num_workers" not in decision or "pin_memory" not in decision:
        die("parity_decision_incomplete")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema": "ms_sta_v6_e4_r4c_performance_code_lock_v1",
        "status": "PERFORMANCE_CODE_LOCK_PASS",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "parent_hpo_code_lock_rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_CODE_LOCK.json",
        "parent_hpo_code_lock_sha256": sha256_file(HPO_LOCK),
        "parent_hpo_git_head": hpo.get("git_head"),
        "parent_hpo_executable_tree_sha256": hpo.get("executable_tree_sha256"),
        "executable_tree_sha256": tree_sha(TREE_FILES),
        "tree_files": TREE_FILES,
        "frozen_policy_sha256": {rel: sha256_file(ROOT / rel) for rel in POLICY_FILES},
        "teacher_logits_cache_rel": "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/teacher_logits_confirm_train.npz",
        "teacher_logits_cache_sha256": sha256_file(TEACHER_CACHE),
        "teacher_logits_parity_sha256": sha256_file(TEACHER_PARITY),
        "dataloader_benchmark_sha256": sha256_file(BENCH),
        "performance_parity_sha256": sha256_file(PARITY),
        "dataloader": {
            "num_workers": int(decision["num_workers"]),
            "pin_memory": bool(decision["pin_memory"]),
            "decision_action": decision.get("action"),
            "decision_reason": decision.get("reason"),
        },
        "scientific_policy_unchanged": True,
        "forbid_allow_unlocked": True,
        "e4_ready": False,
    }
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    OUT.write_bytes(raw)
    print("PERF_CODE_LOCK_OK", head, hashlib.sha256(raw).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
