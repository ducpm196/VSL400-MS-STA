# -*- coding: utf-8 -*-
"""E4-R4 Phase-2b pilot extraction — sealed Skeleton compact + RGB pack.

NO extraction unless --require-phase2b-code-lock is satisfied (or --allow-unlocked for dry tests).
Never writes B4 paths / registry / FINAL_RESULTS / D4 / demo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.e4_r4_joint_layouts import validate_layout_invariants
from src.data_prep.e4_r4_rgb_contract import validate_rgb_budget_table
from src.data_prep.e4_r4_rgb_pack import (
    pack_rgb_candidate,
    read_selected_frames,
    sample_frame_indices,
    write_rgb_npz_atomic,
)
from src.data_prep.e4_r4_skel_compact import compact_one_sample
from src.data_prep.raw_video_resolve import RawVideoResolver
from src.models.feature_versioning import sha256_file, write_json_atomic

POLICY = ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4_POLICY.json"
FREEZE = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4_PILOT_SAMPLE_FREEZE.json"
PHASE2B_LOCK = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4_PHASE2B_CODE_LOCK.json"
META = ROOT / "metadata.csv"
B4_FORBIDDEN = (
    ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
    ROOT / "data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70",
    ROOT / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json",
    ROOT / "data/processed/ms_sta_registry_active.json",
)


def die(msg: str) -> None:
    print(f"E4_R4_EXTRACT_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_is_ancestor(ancestor: str, tip: str = "HEAD") -> bool:
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, tip],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.returncode == 0


def executable_tree_sha() -> str:
    h = hashlib.sha256()
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
    return h.hexdigest()


def load_metadata_rows() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with META.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def ensure_not_writing_b4(path: Path) -> None:
    resolved = path.resolve()
    for forbidden in B4_FORBIDDEN:
        try:
            resolved.relative_to(forbidden.resolve())
            die(f"refuses_write_under_b4_path:{path}")
        except ValueError:
            continue
        except FileNotFoundError:
            continue


def build_contract(policy: dict, freeze: dict, out_root: Path, layouts: list[str], rgb_cands: list[str]) -> dict:
    skel_src = ROOT / policy["b4_feature_roots_read_only"]["skeleton_root_rel"]
    return {
        "schema": "ms_sta_v6_e4_r4_phase2b_run_contract_v1",
        "kind": "e4_r4_pilot_extract",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pilot_freeze_sha256": sha256_file(FREEZE),
        "pilot_train_ids_sha256": freeze["pilot_train_ids_sha256"],
        "pilot_val_ids_sha256": freeze["pilot_val_ids_sha256"],
        "qa_contact_ids_sha256": freeze["qa_contact_ids_sha256"],
        "policy_sha256": sha256_file(POLICY),
        "metadata_sha256": sha256_file(META),
        "split_manifest_sha256": freeze["split_manifest_sha256"],
        "eligible_sample_manifest_sha256": freeze.get("eligible_sample_manifest_sha256"),
        "b4_skeleton_root_rel": policy["b4_feature_roots_read_only"]["skeleton_root_rel"],
        "b4_skeleton_root_sha_note": "directory_read_only_not_hashed_as_tree",
        "source_skeleton_mode": "sealed_75joint_observations_read_only_compact_transform",
        "layouts": layouts,
        "rgb_candidates": rgb_cands,
        "git_revision": git_head(),
        "executable_tree_sha256": executable_tree_sha(),
        "out_root_rel": out_root.relative_to(ROOT).as_posix(),
        "e4_ready": False,
        "forbid_b4_mutation": True,
    }


def require_resume_match(path: Path, expected: dict) -> None:
    if not path.is_file():
        die(f"missing_run_contract_for_resume:{path}")
    prev = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "pilot_freeze_sha256",
        "policy_sha256",
        "metadata_sha256",
        "split_manifest_sha256",
        "eligible_sample_manifest_sha256",
        "layouts",
        "rgb_candidates",
        "git_revision",
        "executable_tree_sha256",
        "source_skeleton_mode",
    )
    for k in keys:
        if prev.get(k) != expected.get(k):
            die(f"resume_identity_mismatch:{k}:{prev.get(k)}!={expected.get(k)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--layouts", default="S_A,S_B")
    ap.add_argument("--rgb-candidates", default="R0,R1,R2,R3")
    ap.add_argument("--require-phase2b-code-lock", action="store_true")
    ap.add_argument("--allow-unlocked", action="store_true", help="tests only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="optional cap for smoke tests")
    args = ap.parse_args()

    validate_layout_invariants()
    validate_rgb_budget_table()

    if not POLICY.is_file() or not FREEZE.is_file():
        die("missing_policy_or_freeze")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))

    if "eligible_sample_manifest_sha256" not in freeze:
        die("freeze_missing_eligible_sample_manifest_sha256_regenerate_freeze")

    if args.require_phase2b_code_lock:
        if not PHASE2B_LOCK.is_file():
            die("missing_phase2b_code_lock")
        lock = json.loads(PHASE2B_LOCK.read_text(encoding="utf-8"))
        if lock.get("status") != "PHASE2B_CODE_LOCK_PASS":
            die("phase2b_code_lock_not_pass")
        lock_head = str(lock.get("git_head") or "")
        head = git_head()
        if lock_head != head and not git_is_ancestor(lock_head, head):
            die(f"phase2b_lock_git_mismatch:{lock_head}!={head}")
        if lock.get("executable_tree_sha256") != executable_tree_sha():
            die("phase2b_lock_tree_mismatch")
    elif not args.allow_unlocked:
        die("refuse_extract_without_phase2b_code_lock_use_--require-phase2b-code-lock")

    skel_src = ROOT / policy["b4_feature_roots_read_only"]["skeleton_root_rel"] / "skeleton_v3"
    if not skel_src.is_dir():
        die(f"missing_sealed_skel_root:{skel_src}")

    out_root = args.out_root if args.out_root.is_absolute() else (ROOT / args.out_root)
    ensure_not_writing_b4(out_root)
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    rgb_cands = [x.strip() for x in args.rgb_candidates.split(",") if x.strip()]
    for cid in rgb_cands:
        if cid not in {"R0", "R1", "R2", "R3"}:
            die(f"bad_rgb_candidate:{cid}")

    ids = list(freeze["pilot_train_ids"]) + list(freeze["pilot_val_ids"])
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    out_root.mkdir(parents=True, exist_ok=True)
    contract = build_contract(policy, freeze, out_root, layouts, rgb_cands)
    contract_path = out_root / "_run_contract.json"
    if args.resume:
        require_resume_match(contract_path, contract)
    else:
        if any(out_root.glob("**/*.npz")) and contract_path.is_file():
            die("out_root_not_empty_use_--resume_or_new_versioned_root")
        write_json_atomic(contract_path, contract)

    meta_rows = load_metadata_rows()
    resolver = RawVideoResolver(ROOT)
    timing: list[dict] = []

    for sid in ids:
        row = meta_rows.get(sid)
        if row is None:
            die(f"missing_metadata:{sid}")
        sealed = skel_src / f"{sid}.npz"
        if not sealed.is_file():
            die(f"missing_sealed_skel:{sid}")

        # Skeleton compact
        for layout in layouts:
            skel_out = out_root / f"skel_{layout}" / f"{sid}.npz"
            ensure_not_writing_b4(skel_out)
            if args.resume and skel_out.is_file():
                continue
            t0 = time.perf_counter()
            info = compact_one_sample(sealed, skel_out, layout)
            timing.append({"id": sid, "kind": f"skel_{layout}", "sec": time.perf_counter() - t0, **info})

        # RGB pack aligned to sealed frames
        with np.load(sealed, allow_pickle=False) as z:
            sealed_fi = np.asarray(z["original_frame_index"], dtype=np.int32)
            sealed_ts = np.asarray(z["timestamps_sec"], dtype=np.float64)
            # Prefer raw_landmarks xy (image-normalized); fallback joints_raw
            if "raw_landmarks" in z.files:
                lm = np.asarray(z["raw_landmarks"], dtype=np.float32)[..., :2]
            else:
                lm = np.asarray(z["joints_raw"], dtype=np.float32)[..., :2]
            miss = np.asarray(z["missing_mask_prefill"], dtype=bool)
            if "effective_missing" in z.files:
                miss = miss | np.asarray(z["effective_missing"], dtype=bool)

        vpath = resolver.resolve(row)
        if vpath is None or not vpath.is_file():
            die(f"missing_raw_video:{sid}")
        raw_sha = sha256_file(vpath)

        # Probe frame count cheaply
        cap = cv2.VideoCapture(str(vpath))
        n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if n_native <= 0:
            die(f"bad_frame_count:{sid}")

        for cid in rgb_cands:
            rgb_out = out_root / f"rgb_{cid}" / f"{sid}.npz"
            ensure_not_writing_b4(rgb_out)
            if args.resume and rgb_out.is_file():
                continue
            t0 = time.perf_counter()
            t_max = 32
            fi = sample_frame_indices(n_native, t_max, sealed_fi)
            # Map sealed timestamps for selected frames
            fi_to_ts = {int(a): float(b) for a, b in zip(sealed_fi.tolist(), sealed_ts.tolist())}
            ts = np.asarray([fi_to_ts.get(int(x), float(x) / 25.0) for x in fi.tolist()], dtype=np.float64)
            # landmarks rows corresponding to sealed indices closest to fi
            row_map = []
            for fidx in fi.tolist():
                # exact match preferred
                matches = np.where(sealed_fi == int(fidx))[0]
                if matches.size:
                    row_map.append(int(matches[0]))
                else:
                    row_map.append(int(np.argmin(np.abs(sealed_fi.astype(np.int64) - int(fidx)))))
            row_map_a = np.asarray(row_map, dtype=np.int64)
            frames, fps = read_selected_frames(str(vpath), fi)
            arrays = pack_rgb_candidate(
                candidate_id=cid,
                frames_rgb=frames,
                landmarks_xy=lm[row_map_a],
                missing_mask=miss[row_map_a],
                frame_indices=fi,
                timestamps_sec=ts,
                fps=fps,
            )
            arrays["video_id"] = np.asarray(sid)
            arrays["raw_video_sha256"] = np.asarray(raw_sha)
            arrays["sealed_skel_sha256"] = np.asarray(sha256_file(sealed))
            write_rgb_npz_atomic(rgb_out, arrays)
            timing.append(
                {
                    "id": sid,
                    "kind": f"rgb_{cid}",
                    "sec": time.perf_counter() - t0,
                    "bytes": rgb_out.stat().st_size,
                    "T": int(arrays["sequence_length"]),
                }
            )

    summary = {
        "schema": "ms_sta_v6_e4_r4_pilot_extract_summary_v1",
        "n_ids": len(ids),
        "layouts": layouts,
        "rgb_candidates": rgb_cands,
        "timing": timing,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "qa_label": "MODEL_ASSISTED_QA",
        "pilot_val_role": "representation_qa_only_forbid_architecture_selection",
    }
    write_json_atomic(out_root / "E4_R4_PILOT_EXTRACT_SUMMARY.json", summary)
    print("PILOT_EXTRACT_OK", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
