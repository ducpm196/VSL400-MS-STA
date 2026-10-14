#!/usr/bin/env python3
"""MS-STA v5 D0-PILOT-R6 — detector recovery (C0–C3) + skel/RGB re-extract.

Train/val pilot only. Preferred arm C3. No full-corpus / M0 / train B4*.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_R1_SAMPLE_SHA256,
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R4_PILOT_AUDIT_SHA256,
    EXPECTED_R5_PILOT_AUDIT_SHA256,
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R6_AMENDMENT_REL,
    PILOT_R1_SAMPLE_REL,
    PILOT_R4_OUT_DIR_REL,
    PILOT_R5_AUDIT_REL,
    PILOT_R5_INDEPENDENT_REAUDIT_REL,
    PILOT_R5_VISUAL_FREEZE_REL,
    PILOT_R5_VISUAL_MANUAL_REL,
    PILOT_R6_AUDIT_REL,
    PILOT_R6_COMPLETION_REL,
    PILOT_R6_DIAGNOSTIC_REL,
    PILOT_R6_HOLDOUT_FREEZE_REL,
    PILOT_R6_OUT_DIR_REL,
    PILOT_R6_OVERLAYS_INDEX_REL,
    PILOT_R6_REPORT_REL,
    PILOT_R6_RUN_CONTRACT_REL,
    PILOT_R6_VISUAL_FREEZE_REL,
    PILOT_R6_VISUAL_MANUAL_REL,
    PILOT_R6_VISUAL_REL,
    executable_tree_sha256,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_r5_independent_reaudit,
    validate_r5_visual_freeze_list,
    validate_r6_detector_amendment,
    validate_r6_freeze_lists,
    validate_visual_qa_manual,
    write_json,
)

EXTRACT_PY = _ROOT / "_venvs/extract/Scripts/python.exe"
F0_PY = _ROOT / "_venvs/f0_v5/Scripts/python.exe"
REGIONS_LOCAL = ("face", "left_hand", "right_hand")
R6_EXECUTABLE_PATHS = [
    "src/data_prep/hand_recovery_v6.py",
    "src/data_prep/extract_and_smooth.py",
    "src/data_prep/skel_v3.py",
    "src/models/rgb_token_v3.py",
    "scripts/ms_sta_v5_d0_lib.py",
    "scripts/run_ms_sta_v5_d0_r6_pilot.py",
]


def _code_lock_bind() -> dict[str, Any]:
    allowed = ("Docs/results/ms_sta_v5_d0_r6/", "Docs/results/ms_sta_v5_d0_r6")
    dirty_lines = []
    for line in git_porcelain(_ROOT).splitlines():
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        norm = path.replace("\\", "/")
        if any(norm == p.rstrip("/") or norm.startswith(p) for p in allowed):
            continue
        dirty_lines.append(line)
    dirty = "\n".join(dirty_lines).strip()
    rev = git_rev(_ROOT)
    return {
        "code_lock_git_revision": rev,
        "executable_tree_sha256": executable_tree_sha256(R6_EXECUTABLE_PATHS, root=_ROOT),
        "worktree_dirty": bool(dirty),
        "porcelain": dirty[:2000] if dirty else "",
    }


def _require_clean(expected_rev: str | None = None) -> dict[str, Any]:
    bind = _code_lock_bind()
    if bind["worktree_dirty"]:
        raise RuntimeError(f"dirty_worktree_forbidden:{bind['porcelain'][:500]}")
    if expected_rev is not None and bind["code_lock_git_revision"] != expected_rev:
        raise RuntimeError(f"code_lock_mismatch:{expected_rev}:{bind['code_lock_git_revision']}")
    if len(bind["code_lock_git_revision"]) != 40:
        raise RuntimeError("code_lock_git_revision_must_be_full_sha40")
    return bind


def _save_skel_npz(out_dir: Path, sample_id: str, payload: dict[str, Any], recovery: dict[str, Any]) -> str:
    path = out_dir / "skel_v3" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Encode observation sources as uint8: 0=missing, 1=holistic, 2=hands_fallback
    def enc(arr):
        m = {"missing": 0, "holistic": 1, "hands_fallback": 2}
        return np.asarray([m[str(x)] for x in arr], dtype=np.uint8)

    np.savez_compressed(
        path,
        coordinates=payload["coordinates"],
        joints_raw=payload["joints_raw"],
        velocity=payload["velocity"],
        bone_vectors=payload["bone_vectors"],
        missing_mask_prefill=payload["missing_mask_prefill"],
        validity=payload["validity"],
        confidence=payload["confidence"],
        original_frame_index=payload["original_frame_index"],
        timestamps_sec=payload["timestamps_sec"],
        padding_mask=payload["padding_mask"],
        joint_part=payload["joint_part"],
        joint_chirality=payload["joint_chirality"],
        bone_edges=payload["bone_edges"],
        sequence_length=np.asarray([payload["sequence_length"]], dtype=np.int32),
        raw_landmarks=recovery["raw_landmarks"],
        effective_landmarks=recovery["effective_landmarks"],
        raw_holistic_missing=recovery["raw_holistic_missing"],
        effective_missing=recovery["effective_missing"],
        observation_source_left_hand=enc(recovery["observation_source_left_hand"]),
        observation_source_right_hand=enc(recovery["observation_source_right_hand"]),
        wrist_assignment_distance_left=np.asarray(
            [(-1.0 if v is None else float(v)) for v in recovery["wrist_assignment_distance_left"]],
            dtype=np.float32,
        ),
        wrist_assignment_distance_right=np.asarray(
            [(-1.0 if v is None else float(v)) for v in recovery["wrist_assignment_distance_right"]],
            dtype=np.float32,
        ),
    )
    meta = {
        "arm": recovery["arm"],
        "pose_guided_crop_left": recovery["pose_guided_crop_left"],
        "pose_guided_crop_right": recovery["pose_guided_crop_right"],
        "fallback_rejection_reason_left": recovery["fallback_rejection_reason_left"],
        "fallback_rejection_reason_right": recovery["fallback_rejection_reason_right"],
        "observation_source_enum": {"0": "missing", "1": "holistic", "2": "hands_fallback"},
    }
    (out_dir / "skel_v3" / f"{sample_id}.recovery.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def _save_rgb_npz(out_dir: Path, sample_id: str, payload: dict[str, Any]) -> str:
    path = out_dir / "rgb_token_v3" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {f"token_{k}": v for k, v in payload["tokens"].items()}
    kwargs.update({f"missing_{k}": v for k, v in payload["missing_region_mask"].items()})
    kwargs["pathway_slow"] = payload["pathway_global_tokens"]["slow"]
    kwargs["pathway_fast"] = payload["pathway_global_tokens"]["fast"]
    kwargs["original_frame_index"] = payload["original_frame_index"]
    meta = {
        "windows": payload["windows"],
        "fps": payload["fps"],
        "n_native_frames": payload["n_native_frames"],
        "hook": payload["hook"],
        "roi_method": payload["roi_method"],
        "regions": payload["regions"],
        "window_starts": payload.get("window_starts"),
        "window_size": payload.get("window_size"),
        "window_stride": payload.get("window_stride"),
        "backbone": payload.get("backbone"),
        "operator_order": payload.get("operator_order"),
        "temporal_aggregation": payload.get("temporal_aggregation"),
        "forbid_temporal_mean_before_roi_align": payload.get("forbid_temporal_mean_before_roi_align"),
        "forbid_zero_dilution_in_temporal_mean": payload.get("forbid_zero_dilution_in_temporal_mean"),
        "joint_layout": payload.get("joint_layout"),
        "feature_map_shapes_example": (payload["windows"][0]["feature_map_shapes"] if payload["windows"] else {}),
        "recovery_arm": "C3",
    }
    meta_path = out_dir / "rgb_token_v3" / f"{sample_id}.meta.json"
    meta_path.write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    np.savez_compressed(path, **kwargs)
    return sha256_file(path)


def build_freeze_lists(*, overwrite: bool = False) -> dict[str, Any]:
    """Freeze R5 primary windows + holdout risk windows BEFORE any R6 algorithm run."""
    r5_freeze = load_json(_ROOT / PILOT_R5_VISUAL_FREEZE_REL)
    validate_r5_visual_freeze_list(r5_freeze)
    primary_path = _ROOT / PILOT_R6_VISUAL_FREEZE_REL
    holdout_path = _ROOT / PILOT_R6_HOLDOUT_FREEZE_REL
    if primary_path.is_file() and holdout_path.is_file() and not overwrite:
        primary = load_json(primary_path)
        holdout = load_json(holdout_path)
        validate_r6_freeze_lists(primary, holdout, r5_freeze=r5_freeze)
        return {"primary": primary, "holdout": holdout}

    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    samples = {s["sample_id"]: s for s in sample_set["samples"]}
    r5_keys = {(w["sample_id"], int(w["window_start_idx"])) for w in r5_freeze["windows"]}
    # R5 FAIL inspections → must be covered by primary freeze
    r5_manual = load_json(_ROOT / PILOT_R5_VISUAL_MANUAL_REL)
    fail_keys = {
        (r["sample_id"], int(r["window_start_idx"]))
        for r in r5_manual["inspections"]
        if r.get("verdict") == "FAIL"
    }

    primary = {
        "schema": "ms_sta_v5_d0_r6_visual_qa_freeze_list_v1",
        "frozen_before_algorithm": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_r5_freeze_rel": PILOT_R5_VISUAL_FREEZE_REL,
        "selection_rule": "exact_copy_of_r5_seven_windows",
        "n_windows": len(r5_freeze["windows"]),
        "r5_fail_keys_covered": sorted([list(k) for k in fail_keys]),
        "windows": [
            {
                **w,
                "r6_role": "primary_r5_carryforward",
            }
            for w in r5_freeze["windows"]
        ],
    }

    # Holdout: hand-risk windows from R4 metas not in R5 freeze
    rows = []
    out_r4 = _ROOT / PILOT_R4_OUT_DIR_REL
    for sid, s in samples.items():
        meta = json.loads((out_r4 / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        for wi, w in enumerate(meta["windows"]):
            key = (sid, int(w["start_idx"]))
            if key in r5_keys:
                continue
            rv = w["region_validity"]
            lh = float(rv["left_hand"]["valid_fraction"])
            rh = float(rv["right_hand"]["valid_fraction"])
            hand_min = min(lh, rh)
            if hand_min >= 1.0 - 1e-12:
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "split": s["split"],
                    "window_index": wi,
                    "window_start_idx": int(w["start_idx"]),
                    "center_frame": int(w["center_frame"]),
                    "valid_fraction_by_region": {
                        "face": float(rv["face"]["valid_fraction"]),
                        "left_hand": lh,
                        "right_hand": rh,
                    },
                    "min_hand_valid_fraction": hand_min,
                    "stratum": "holdout_hand_partial",
                }
            )
    rows = sorted(rows, key=lambda r: (r["min_hand_valid_fraction"], r["sample_id"], r["window_start_idx"]))
    # Keep a small holdout: lowest hand vf train + one val if available
    hold_sel = []
    for r in rows:
        if r["split"] == "train" and len([x for x in hold_sel if x["split"] == "train"]) < 3:
            hold_sel.append(r)
        elif r["split"] == "val" and len([x for x in hold_sel if x["split"] == "val"]) < 1:
            hold_sel.append(r)
        if len(hold_sel) >= 4:
            break
    if len(hold_sel) < 2:
        raise RuntimeError("holdout_too_few")
    holdout = {
        "schema": "ms_sta_v5_d0_r6_holdout_risk_freeze_list_v1",
        "frozen_before_algorithm": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_rule": "hand_partial_windows_outside_r5_freeze_sorted_by_min_hand_vf",
        "n_windows": len(hold_sel),
        "windows": hold_sel,
    }
    validate_r6_freeze_lists(primary, holdout, r5_freeze=r5_freeze)
    write_json(primary_path, primary, overwrite=True)
    write_json(holdout_path, holdout, overwrite=True)
    return {"primary": primary, "holdout": holdout}


def write_run_contract() -> str:
    validate_r5_independent_reaudit(load_json(_ROOT / PILOT_R5_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r6_detector_amendment(load_json(_ROOT / FULL_PROTOCOL_R6_AMENDMENT_REL), root=_ROOT)
    bind = _require_clean()
    obj = {
        "schema": "ms_sta_v5_d0_r6_pilot_run_contract_v1",
        "status": "ACTIVE",
        "campaign": "D0_PILOT_R6",
        "scope": "DETECTOR_RECOVERY_PILOT_TRAIN_VAL_ONLY",
        "preferred_arm": "C3",
        "diagnostic_arms": ["C0", "C1", "C2", "C3"],
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "worktree_dirty": False,
        "r5_pilot_audit_sha256": EXPECTED_R5_PILOT_AUDIT_SHA256,
        "r5_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R5_INDEPENDENT_REAUDIT_REL),
        "r6_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R6_AMENDMENT_REL),
        "canonical_full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "pilot_sample_sha256": EXPECTED_R1_SAMPLE_SHA256,
        "parent_r4_outputs": PILOT_R4_OUT_DIR_REL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "forbid_full_corpus": True,
        "forbid_push": True,
        "forbid_m0": True,
        "forbid_train_b4": True,
    }
    path = _ROOT / PILOT_R6_RUN_CONTRACT_REL
    if path.is_file():
        prev = load_json(path)
        if prev.get("status") == "CLOSED":
            raise RuntimeError("r6_run_contract_already_closed")
        return sha256_file(path)
    return write_json(path, obj, overwrite=False)


def _extract_one(sample: dict[str, Any], out_dir: Path, arm: str) -> dict[str, Any]:
    from src.data_prep.hand_recovery_v6 import HandRecoveryEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    engine = HandRecoveryEngine(arm=arm, model_complexity=0, store_stride=2)  # type: ignore[arg-type]
    try:
        rec = engine.process_video(_ROOT / sample["video_rel"])
    finally:
        engine.close()
    payload = build_skel_v3_arrays(
        joints_raw=rec["effective_landmarks"],
        raw_pose=rec["raw_pose"],
        missing_prefill=rec["effective_missing"],
        original_frame_index=rec["original_frame_index"],
        timestamps_sec=rec["timestamps_sec"],
    )
    payload["joints_raw"] = np.asarray(rec["effective_landmarks"], dtype=np.float32)
    digest = _save_skel_npz(out_dir, sample["sample_id"], payload, rec)
    return {
        "sample_id": sample["sample_id"],
        "ok": True,
        "arm": arm,
        "skel_sha256": digest,
        "n_frames": int(rec["original_frame_index"].shape[0]),
        "fallback_lh": int(np.sum(np.asarray(rec["observation_source_left_hand"]) == "hands_fallback")),
        "fallback_rh": int(np.sum(np.asarray(rec["observation_source_right_hand"]) == "hands_fallback")),
    }


def run_diagnostic_matrix(*, sample_ids: list[str] | None = None) -> dict[str, Any]:
    """Run C0–C3 on FAIL-risk samples; C0 missing must bit-match R4."""
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    samples = {s["sample_id"]: s for s in sample_set["samples"]}
    if sample_ids is None:
        sample_ids = ["016267", "006239", "011721", "015417", "000000"]
    diag_dir = _ROOT / PILOT_R6_OUT_DIR_REL / "diagnostics"
    rows = []
    for sid in sample_ids:
        sample = samples[sid]
        r4 = np.load(_ROOT / PILOT_R4_OUT_DIR_REL / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        arm_stats = {}
        for arm in ("C0", "C1", "C2", "C3"):
            out = diag_dir / arm
            row = _extract_one(sample, out, arm)
            sk = np.load(out / "skel_v3" / f"{sid}.npz", allow_pickle=False)
            if arm == "C0":
                # Compare raw holistic missing to R4 missing_mask_prefill
                raw = sk["raw_holistic_missing"]
                ref = r4["missing_mask_prefill"]
                # Align by original_frame_index intersection
                idx_a = {int(i): n for n, i in enumerate(sk["original_frame_index"].tolist())}
                idx_b = {int(i): n for n, i in enumerate(r4["original_frame_index"].tolist())}
                common = sorted(set(idx_a) & set(idx_b))
                if not common:
                    bit_exact = False
                else:
                    aa = np.stack([raw[idx_a[i]] for i in common])
                    bb = np.stack([ref[idx_b[i]] for i in common])
                    bit_exact = bool(np.array_equal(aa, bb))
                row["raw_holistic_bit_exact_vs_r4"] = bit_exact
                row["n_common_frames"] = len(common)
            arm_stats[arm] = row
        rows.append({"sample_id": sid, "arms": arm_stats})
    summary = {
        "schema": "ms_sta_v5_d0_r6_diagnostic_matrix_v1",
        "preferred_arm": "C3",
        "samples": rows,
        "c0_all_bit_exact": all(
            bool(r["arms"]["C0"].get("raw_holistic_bit_exact_vs_r4")) for r in rows
        ),
    }
    write_json(_ROOT / PILOT_R6_DIAGNOSTIC_REL, summary, overwrite=True)
    return summary


def run_skel_phase(*, resume: bool = False) -> dict[str, Any]:
    bind = _code_lock_bind()
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    out_dir = _ROOT / PILOT_R6_OUT_DIR_REL
    results = []
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "skel_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True, "skel_sha256": sha256_file(dest)})
            continue
        results.append(_extract_one(s, out_dir, "C3"))
    summary = {
        "phase": "skel",
        "arm": "C3",
        "ok": all(r.get("ok") for r in results),
        "results": results,
        "provenance": {
            "code_lock_git_revision": bind["code_lock_git_revision"],
            "executable_tree_sha256": bind["executable_tree_sha256"],
            "worktree_dirty": bind["worktree_dirty"],
        },
    }
    write_json(out_dir / "SKEL_PHASE_SUMMARY.json", summary, overwrite=True)
    return summary


def run_rgb_phase(*, resume: bool = False) -> dict[str, Any]:
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    bind = _code_lock_bind()
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    out_dir = _ROOT / PILOT_R6_OUT_DIR_REL
    visual_dir = _ROOT / PILOT_R6_VISUAL_REL
    visual_dir.mkdir(parents=True, exist_ok=True)
    extractor = FrozenSlowFastBlock4(
        checkpoint_path=_ROOT / EXPECTED_SLOWFAST_CKPT_REL,
        expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
        device="cpu",
    )
    results = []
    overlays_index: list[dict[str, Any]] = []
    freeze = load_json(_ROOT / PILOT_R6_VISUAL_FREEZE_REL)
    holdout = load_json(_ROOT / PILOT_R6_HOLDOUT_FREEZE_REL)
    qa_keys = {
        (w["sample_id"], int(w["window_start_idx"]))
        for w in freeze["windows"] + holdout["windows"]
    }
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "rgb_token_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True, "rgb_sha256": sha256_file(dest)})
            continue
        skel = np.load(out_dir / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        landmarks_by_frame = {}
        missing_by_frame = {}
        raw = skel["joints_raw"]
        miss = skel["missing_mask_prefill"]
        for i, fi in enumerate(skel["original_frame_index"].tolist()):
            landmarks_by_frame[int(fi)] = raw[i]
            missing_by_frame[int(fi)] = miss[i]
        overlay_dir = visual_dir / sid
        payload = extract_rgb_token_v3_for_video(
            video_path=_ROOT / s["video_rel"],
            extractor=extractor,
            landmarks_by_frame=landmarks_by_frame,
            missing_by_frame=missing_by_frame,
            skel_frame_index=skel["original_frame_index"],
            skel_timestamps=skel["timestamps_sec"],
            overlay_dir=overlay_dir,
            sample_id=sid,
        )
        digest = _save_rgb_npz(out_dir, sid, payload)
        # Collect overlays only for freeze/holdout windows
        for ov in payload.get("overlays") or []:
            # parse window start from filename ..._wXXXX_
            name = Path(ov["path"]).name
            try:
                wpart = [p for p in name.split("_") if p.startswith("w") and p[1:].isdigit()][0]
                wstart = int(wpart[1:])
            except Exception:
                continue
            if (sid, wstart) not in qa_keys:
                continue
            overlays_index.append(
                {
                    "sample_id": sid,
                    "kind": ov["kind"],
                    "path": ov["path"],
                    "sha256": ov["sha256"],
                    "center_frame": ov.get("center_frame"),
                    "native_frame": ov.get("native_frame"),
                    "landmark_frame": ov.get("landmark_frame"),
                    "window_start_idx": wstart,
                    "boxes": ov.get("boxes"),
                    "missing_regions": ov.get("missing_regions"),
                    "n_tiles": ov.get("n_tiles"),
                    "slice_native_frames": ov.get("slice_native_frames"),
                    "tiles": ov.get("tiles"),
                }
            )
        results.append({"sample_id": sid, "ok": True, "rgb_sha256": digest, "n_windows": len(payload["windows"])})
    summary = {
        "phase": "rgb",
        "ok": all(r.get("ok") for r in results),
        "results": results,
        "overlays_index": overlays_index,
        "provenance": {
            "code_lock_git_revision": bind["code_lock_git_revision"],
            "executable_tree_sha256": bind["executable_tree_sha256"],
            "backbone": extractor.load_meta,
        },
    }
    write_json(out_dir / "RGB_PHASE_SUMMARY.json", summary, overwrite=True)
    write_json(
        _ROOT / PILOT_R6_OVERLAYS_INDEX_REL,
        {"ok": True, "n_overlays": len(overlays_index), "overlays_index": overlays_index},
        overwrite=True,
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=["freeze", "contract", "diagnostics", "skel", "rgb", "all"],
        default="freeze",
    )
    ap.add_argument("--overwrite-freeze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    validate_r5_independent_reaudit(load_json(_ROOT / PILOT_R5_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r6_detector_amendment(load_json(_ROOT / FULL_PROTOCOL_R6_AMENDMENT_REL), root=_ROOT)
    if sha256_file(_ROOT / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        raise RuntimeError("canonical_protocol_drift")
    if sha256_file(_ROOT / PILOT_R5_AUDIT_REL) != EXPECTED_R5_PILOT_AUDIT_SHA256:
        raise RuntimeError("r5_audit_drift")

    if args.phase in {"freeze", "all"}:
        out = build_freeze_lists(overwrite=args.overwrite_freeze)
        print(json.dumps({"ok": True, "n_primary": out["primary"]["n_windows"], "n_holdout": out["holdout"]["n_windows"]}, indent=2))
        if args.phase == "freeze":
            return 0
    if args.phase in {"contract", "all"}:
        print(write_run_contract())
        if args.phase == "contract":
            return 0
    if args.phase in {"diagnostics", "all"}:
        # diagnostics require extract env mediapipe
        if sys.executable.lower().replace("\\", "/") != str(EXTRACT_PY).lower().replace("\\", "/"):
            print("[warn] diagnostics should run under _venvs/extract for mediapipe")
        out = run_diagnostic_matrix()
        print(json.dumps({"ok": True, "c0_all_bit_exact": out["c0_all_bit_exact"]}, indent=2))
        if args.phase == "diagnostics":
            return 0 if out["c0_all_bit_exact"] else 1
    if args.phase in {"skel", "all"}:
        out = run_skel_phase(resume=args.resume)
        print(json.dumps({"ok": out["ok"], "n": len(out["results"])}, indent=2))
        if args.phase == "skel":
            return 0 if out["ok"] else 1
    if args.phase in {"rgb", "all"}:
        out = run_rgb_phase(resume=args.resume)
        print(json.dumps({"ok": out["ok"], "n_overlays": len(out["overlays_index"])}, indent=2))
        if args.phase == "rgb":
            return 0 if out["ok"] else 1
    if args.phase == "all":
        print("[orchestrate] STOP_BEFORE_MANUAL_QA: write D0_R6_VISUAL_QA_MANUAL.json then finalize separately")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
