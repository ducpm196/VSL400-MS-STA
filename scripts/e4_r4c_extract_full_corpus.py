# -*- coding: utf-8 -*-
"""E4-R4C full-corpus label-free extract: R1 + S_A only.

Requires R4B feasibility lock PASS. Never writes B4 paths / registry / FINAL_RESULTS / D4 / demo.
HPO/confirm loaders must still restrict to train/validation IDs even though extraction may cover eligible.
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

POLICY = ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_POLICY.json"
TRAIN_POLICY = ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json"
LOCK = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4B_FEASIBILITY_LOCK.json"
PHASE2B_LOCK = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4_PHASE2B_CODE_LOCK.json"
META = ROOT / "metadata.csv"
B4_FORBIDDEN = (
    ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
    ROOT / "data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70",
    ROOT / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json",
    ROOT / "data/processed/ms_sta_registry_active.json",
)


def die(msg: str) -> None:
    print(f"E4_R4C_EXTRACT_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def executable_tree_sha() -> str:
    h = hashlib.sha256()
    for root in ("src/data_prep", "src/models/ms_sta_v6_e4_r4.py", "scripts/e4_r4c_extract_full_corpus.py"):
        p = ROOT / root
        if p.is_file():
            files = [p]
        else:
            files = sorted(x for x in p.rglob("*") if x.is_file() and x.suffix == ".py" and "__pycache__" not in x.parts)
        for fp in files:
            rel = fp.relative_to(ROOT).as_posix()
            h.update(rel.encode())
            h.update(b"\0")
            h.update(fp.read_bytes())
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


def require_locks(policy: dict) -> None:
    if not LOCK.is_file():
        die("missing_r4b_feasibility_lock")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("verdict") != "E4_R4B_STAGE_MATCHED_PASS":
        die("feasibility_lock_not_pass")
    if lock.get("promoted_candidate_ids") != ["R1_S_A_d128"]:
        die("feasibility_lock_promoted_mismatch")
    if not PHASE2B_LOCK.is_file():
        die("missing_phase2b_code_lock")
    if sha256_file(ROOT / policy["eligible_manifest_rel"]) != policy["eligible_manifest_sha256"]:
        die("eligible_manifest_sha_mismatch")
    if sha256_file(ROOT / policy["split_manifest_rel"]) != policy["split_manifest_sha256"]:
        die("split_manifest_sha_mismatch")
    teacher = ROOT / policy["teacher"]["checkpoint_rel"]
    if sha256_file(teacher) != policy["teacher"]["checkpoint_sha256"]:
        die("teacher_checkpoint_sha_mismatch")


def build_contract(policy: dict, out_root: Path, ids: list[str]) -> dict:
    return {
        "schema": "ms_sta_v6_e4_r4c_full_corpus_run_contract_v1",
        "kind": "e4_r4c_full_corpus_extract_r1_sa",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy_sha256": sha256_file(POLICY),
        "train_policy_sha256": sha256_file(TRAIN_POLICY),
        "feasibility_lock_sha256": sha256_file(LOCK),
        "phase2b_code_lock_sha256": sha256_file(PHASE2B_LOCK),
        "eligible_manifest_sha256": policy["eligible_manifest_sha256"],
        "split_manifest_sha256": policy["split_manifest_sha256"],
        "metadata_sha256": sha256_file(META),
        "candidate_id": "R1_S_A_d128",
        "rgb_candidates": ["R1"],
        "layouts": ["S_A"],
        "n_ids": len(ids),
        "ids_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "git_revision": git_head(),
        "executable_tree_sha256": executable_tree_sha(),
        "out_root_rel": out_root.relative_to(ROOT).as_posix(),
        "label_free": True,
        "test_labels_locked": True,
        "e4_ready": False,
        "forbid_b4_mutation": True,
    }


def require_resume_match(path: Path, expected: dict) -> None:
    if not path.is_file():
        die(f"missing_run_contract_for_resume:{path}")
    prev = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "policy_sha256",
        "train_policy_sha256",
        "feasibility_lock_sha256",
        "eligible_manifest_sha256",
        "split_manifest_sha256",
        "metadata_sha256",
        "candidate_id",
        "rgb_candidates",
        "layouts",
        "ids_sha256",
        "executable_tree_sha256",
    )
    for k in keys:
        if prev.get(k) != expected.get(k):
            die(f"resume_identity_mismatch:{k}:{prev.get(k)}!={expected.get(k)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    validate_layout_invariants()
    validate_rgb_budget_table()
    if not POLICY.is_file() or not TRAIN_POLICY.is_file():
        die("missing_r4c_policy")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN" or policy.get("candidate_id") != "R1_S_A_d128":
        die("policy_not_frozen_or_wrong_candidate")
    require_locks(policy)

    eligible = json.loads((ROOT / policy["eligible_manifest_rel"]).read_text(encoding="utf-8"))
    ids = [str(x) for x in eligible["eligible_ids"]]
    ids = sorted(set(ids))
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    tree = executable_tree_sha()[:12]
    if args.out_root is None:
        out_root = ROOT / f"data/features/ms_sta_v6_e4_r4c_r1_sa_{tree}"
    else:
        out_root = args.out_root if args.out_root.is_absolute() else (ROOT / args.out_root)
    ensure_not_writing_b4(out_root)

    skel_src = ROOT / policy["b4_feature_roots_read_only"]["skeleton_root_rel"] / "skeleton_v3"
    if not skel_src.is_dir():
        die(f"missing_sealed_skel_root:{skel_src}")

    out_root.mkdir(parents=True, exist_ok=True)
    contract = build_contract(policy, out_root, ids)
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
    failed: list[dict] = []
    t_start = time.perf_counter()

    for i, sid in enumerate(ids, 1):
        row = meta_rows.get(sid)
        if row is None:
            die(f"missing_metadata:{sid}")
        sealed = skel_src / f"{sid}.npz"
        if not sealed.is_file():
            die(f"missing_sealed_skel:{sid}")

        skel_out = out_root / "skel_S_A" / f"{sid}.npz"
        rgb_out = out_root / "rgb_R1" / f"{sid}.npz"
        ensure_not_writing_b4(skel_out)
        ensure_not_writing_b4(rgb_out)

        try:
            if not (args.resume and skel_out.is_file()):
                t0 = time.perf_counter()
                info = compact_one_sample(sealed, skel_out, "S_A")
                timing.append({"id": sid, "kind": "skel_S_A", "sec": time.perf_counter() - t0, **info})

            if not (args.resume and rgb_out.is_file()):
                with np.load(sealed, allow_pickle=False) as z:
                    sealed_fi = np.asarray(z["original_frame_index"], dtype=np.int32)
                    sealed_ts = np.asarray(z["timestamps_sec"], dtype=np.float64)
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
                cap = cv2.VideoCapture(str(vpath))
                n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if n_native <= 0:
                    die(f"bad_frame_count:{sid}")

                t0 = time.perf_counter()
                fi = sample_frame_indices(n_native, 32, sealed_fi)
                fi_to_ts = {int(a): float(b) for a, b in zip(sealed_fi.tolist(), sealed_ts.tolist())}
                ts = np.asarray([fi_to_ts.get(int(x), float(x) / 25.0) for x in fi.tolist()], dtype=np.float64)
                row_map = []
                for fidx in fi.tolist():
                    matches = np.where(sealed_fi == int(fidx))[0]
                    if matches.size:
                        row_map.append(int(matches[0]))
                    else:
                        row_map.append(int(np.argmin(np.abs(sealed_fi.astype(np.int64) - int(fidx)))))
                row_map_a = np.asarray(row_map, dtype=np.int64)
                frames, fps = read_selected_frames(str(vpath), fi)
                arrays = pack_rgb_candidate(
                    candidate_id="R1",
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
                arrays["policy_sha256"] = np.asarray(contract["policy_sha256"])
                arrays["feasibility_lock_sha256"] = np.asarray(contract["feasibility_lock_sha256"])
                write_rgb_npz_atomic(rgb_out, arrays)
                timing.append(
                    {
                        "id": sid,
                        "kind": "rgb_R1",
                        "sec": time.perf_counter() - t0,
                        "bytes": rgb_out.stat().st_size,
                        "T": int(arrays["sequence_length"]),
                    }
                )
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — fail closed: any failed sample stops campaign
            failed.append({"id": sid, "error": repr(exc)})
            die(f"sample_failed:{sid}:{exc!r}")

        if args.progress_every and i % args.progress_every == 0:
            elapsed = time.perf_counter() - t_start
            rate = i / max(elapsed, 1e-9)
            eta = (len(ids) - i) / max(rate, 1e-9)
            print(f"progress {i}/{len(ids)} rate={rate:.2f}/s eta_h={eta/3600:.2f}", flush=True)

    summary = {
        "schema": "ms_sta_v6_e4_r4c_full_corpus_extract_summary_v1",
        "n_ids": len(ids),
        "n_failed": len(failed),
        "failed": failed,
        "layouts": ["S_A"],
        "rgb_candidates": ["R1"],
        "out_root_rel": out_root.relative_to(ROOT).as_posix(),
        "contract_sha256": sha256_file(contract_path),
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_sec": time.perf_counter() - t_start,
        "e4_ready": False,
    }
    write_json_atomic(out_root / "E4_R4C_FULL_CORPUS_EXTRACT_SUMMARY.json", summary)
    # lightweight timing sidecar (may be large)
    write_json_atomic(out_root / "E4_R4C_FULL_CORPUS_EXTRACT_TIMING.json", {"n": len(timing), "timing": timing[-200:]})
    print("FULL_CORPUS_EXTRACT_OK", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
