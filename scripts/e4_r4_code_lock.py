# -*- coding: utf-8 -*-
"""E4-R4 code-lock / preflight — bind SHAs; never write B4 artifacts."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs" / "results" / "ms_sta_v6_e4_r4" / "E4_R4_CODE_LOCK.json"
POLICY = ROOT / "Docs" / "results" / "ms_sta_v6_e4_r4_policy" / "E4_R4_POLICY.json"

TEACHER_SHA = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
CKPT = ROOT / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def executable_tree_sha() -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for root in ("src", "scripts"):
        for p in sorted((ROOT / root).rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() in {".pyc", ".pyo"} or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(ROOT).as_posix()
            h.update(rel.encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
            n += 1
    return h.hexdigest(), n


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    ckpt_sha = sha256_file(CKPT)
    if ckpt_sha != TEACHER_SHA:
        print("FAIL teacher checkpoint SHA mismatch", ckpt_sha)
        return 1
    if ckpt_sha != policy["teacher"]["checkpoint_sha256"]:
        print("FAIL policy teacher SHA mismatch")
        return 1

    tree_sha, n_files = executable_tree_sha()
    head = git_head()
    binds = {
        "split_manifest": sha256_file(ROOT / policy["split_manifest_rel"]),
        "eligible_manifest": sha256_file(ROOT / policy["eligible_manifest_rel"]),
        "active_registry_pointer": sha256_file(ROOT / policy["active_registry_pointer_rel"]),
        "defense_registry": sha256_file(ROOT / policy["defense_registry_rel"]),
        "final_results": sha256_file(ROOT / policy["final_results_rel"]),
        "policy": sha256_file(POLICY),
    }
    for key, expected in [
        ("split_manifest", policy["split_manifest_sha256"]),
        ("eligible_manifest", policy["eligible_manifest_sha256"]),
        ("active_registry_pointer", policy["active_registry_pointer_sha256"]),
        ("defense_registry", policy["defense_registry_sha256"]),
        ("final_results", policy["final_results_sha256"]),
    ]:
        if binds[key] != expected:
            print(f"FAIL {key} sha", binds[key], "!=", expected)
            return 1

    # Import layout module (semantic invariants).
    sys.path.insert(0, str(ROOT))
    from src.data_prep.e4_r4_joint_layouts import LAYOUT_S_A, LAYOUT_S_B

    payload = {
        "schema": "ms_sta_v6_e4_r4_code_lock_v1",
        "campaign": "E4-R4",
        "locked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "branch": "research/ms-sta-v6-e4-r4-raw-compact",
        "teacher_checkpoint_rel": policy["teacher"]["checkpoint_rel"],
        "teacher_checkpoint_sha256": ckpt_sha,
        "b4_write_forbidden": True,
        "e4_ready": False,
        "executable_tree_sha256": tree_sha,
        "executable_tree_file_count": n_files,
        "binds": binds,
        "policy_rel": "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4_POLICY.json",
        "layout_s_a_n_joints": len(LAYOUT_S_A),
        "layout_s_b_n_joints": len(LAYOUT_S_B),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.platform(),
        },
        "status": "CODE_LOCK_PASS",
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("CODE_LOCK_PASS", OUT)
    print("head", head)
    print("tree", tree_sha)
    print("S_A", len(LAYOUT_S_A), "S_B", len(LAYOUT_S_B))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
