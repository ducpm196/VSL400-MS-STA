# -*- coding: utf-8 -*-
"""Generate E4_R4_PHASE2B_CODE_LOCK.json after producers/validators/tests exist."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4_PHASE2B_CODE_LOCK.json"
POLICY = ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4_POLICY.json"
FREEZE = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4_PILOT_SAMPLE_FREEZE.json"
TEACHER_SHA = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
CKPT = ROOT / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"

REQUIRED_PRODUCERS = (
    "scripts/e4_r4_extract_pilot.py",
    "scripts/e4_r4_validate_pilot.py",
    "scripts/e4_r4_qa_pilot_overlays.py",
    "scripts/e4_r4_freeze_pilot_samples.py",
    "src/data_prep/e4_r4_joint_layouts.py",
    "src/data_prep/e4_r4_rgb_contract.py",
    "src/data_prep/e4_r4_skel_compact.py",
    "src/data_prep/e4_r4_rgb_pack.py",
    "tests/test_e4_r4_joint_layouts.py",
    "tests/test_e4_r4_phase2b_contracts.py",
)


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


def die(msg: str) -> None:
    print(f"PHASE2B_CODE_LOCK_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        die("refuse_lock_on_dirty_tree_commit_code_first")

    for rel in REQUIRED_PRODUCERS:
        if not (ROOT / rel).is_file():
            die(f"missing_required:{rel}")

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if "eligible_sample_manifest_sha256" not in freeze:
        die("freeze_missing_eligible_sha")

    ckpt_sha = sha256_file(CKPT)
    if ckpt_sha != TEACHER_SHA:
        die(f"teacher_sha_mismatch:{ckpt_sha}")

    # Import fail-closed contracts
    sys.path.insert(0, str(ROOT))
    from src.data_prep.e4_r4_joint_layouts import LAYOUT_S_A, LAYOUT_S_B, MEDIAPIPE_POSE_SELECTED
    from src.data_prep.e4_r4_rgb_contract import validate_rgb_budget_table

    if MEDIAPIPE_POSE_SELECTED[3][2] == "mouth_center":
        die("pose9_still_named_mouth_center")
    budget = validate_rgb_budget_table()

    src = policy.get("skeleton_candidates", {}).get("source", "")
    if "read_only_or_reextract" in str(src):
        die("policy_still_allows_ambiguous_skel_source")
    if src != "sealed_75joint_observations_read_only_compact_transform":
        die(f"bad_skel_source:{src}")

    tree_sha, n_files = executable_tree_sha()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    payload = {
        "schema": "ms_sta_v6_e4_r4_phase2b_code_lock_v1",
        "campaign": "E4-R4",
        "phase": "2b",
        "locked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "branch": "research/ms-sta-v6-e4-r4-raw-compact",
        "teacher_checkpoint_sha256": ckpt_sha,
        "executable_tree_sha256": tree_sha,
        "executable_tree_file_count": n_files,
        "policy_sha256": sha256_file(POLICY),
        "pilot_freeze_sha256": sha256_file(FREEZE),
        "eligible_sample_manifest_sha256": freeze["eligible_sample_manifest_sha256"],
        "split_manifest_sha256": freeze["split_manifest_sha256"],
        "layout_s_a_n_joints": len(LAYOUT_S_A),
        "layout_s_b_n_joints": len(LAYOUT_S_B),
        "rgb_budget": budget,
        "pose_landmark_9_name": MEDIAPIPE_POSE_SELECTED[3][2],
        "skeleton_source": src,
        "b4_write_forbidden": True,
        "e4_ready": False,
        "extraction_authorized_after_this_lock": True,
        "platform": {"python": sys.version.split()[0], "system": platform.platform()},
        "required_producers": list(REQUIRED_PRODUCERS),
        "status": "PHASE2B_CODE_LOCK_PASS",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print("PHASE2B_CODE_LOCK_PASS", OUT)
    print("head", head)
    print("tree", tree_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
