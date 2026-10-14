#!/usr/bin/env python3
"""MS-STA v5 D0-PILOT-R7 — dual-source Skeleton / RGB pose-prior ROI.

Finalizer never synthesizes manual verdicts. Primary+holdout zero-fail required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_R1_SAMPLE_SHA256,
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R6_PILOT_AUDIT_SHA256,
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R7_AMENDMENT_REL,
    PILOT_R1_SAMPLE_REL,
    PILOT_R4_OUT_DIR_REL,
    PILOT_R6_AUDIT_REL,
    PILOT_R6_INDEPENDENT_REAUDIT_REL,
    PILOT_R7_AUDIT_REL,
    PILOT_R7_COMPLETION_REL,
    PILOT_R7_DIAGNOSTIC_REL,
    PILOT_R7_OUT_DIR_REL,
    PILOT_R7_OVERLAYS_INDEX_REL,
    PILOT_R7_REPORT_REL,
    PILOT_R7_RUN_CONTRACT_REL,
    PILOT_R7_VISUAL_FREEZE_REL,
    PILOT_R7_VISUAL_MANUAL_REL,
    PILOT_R7_VISUAL_REL,
    executable_tree_sha256,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_r6_independent_reaudit,
    validate_r7_dual_source_amendment,
    validate_r7_pilot_report,
    validate_r7_visual_freeze_list,
    validate_r7_visual_qa_manual,
    write_json,
)

R7_EXECUTABLE_PATHS = [
    "src/data_prep/hand_recovery_v6.py",
    "src/data_prep/extract_and_smooth.py",
    "src/data_prep/skel_v3.py",
    "src/models/rgb_token_v3.py",
    "scripts/ms_sta_v5_d0_lib.py",
    "scripts/run_ms_sta_v5_d0_r6_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_finalize.py",
    "scripts/build_ms_sta_v5_d0_r7_freeze.py",
    "scripts/write_ms_sta_v5_d0_r7_manual_human.py",
]


def _code_lock_bind() -> dict[str, Any]:
    allowed = ("Docs/results/ms_sta_v5_d0_r7/",)
    dirty = []
    for line in git_porcelain(_ROOT).splitlines():
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        norm = path.replace("\\", "/")
        if any(norm == p.rstrip("/") or norm.startswith(p) for p in allowed):
            continue
        dirty.append(line)
    dirty_s = "\n".join(dirty).strip()
    rev = git_rev(_ROOT)
    return {
        "code_lock_git_revision": rev,
        "executable_tree_sha256": executable_tree_sha256(R7_EXECUTABLE_PATHS, root=_ROOT),
        "worktree_dirty": bool(dirty_s),
        "porcelain": dirty_s[:2000],
    }


def _require_clean(expected: str | None = None) -> dict[str, Any]:
    bind = _code_lock_bind()
    if bind["worktree_dirty"]:
        raise RuntimeError(f"dirty_worktree_forbidden:{bind['porcelain'][:500]}")
    if expected is not None and bind["code_lock_git_revision"] != expected:
        raise RuntimeError(f"code_lock_mismatch:{expected}:{bind['code_lock_git_revision']}")
    if len(bind["code_lock_git_revision"]) != 40:
        raise RuntimeError("code_lock_must_be_full_sha40")
    return bind


def write_run_contract() -> str:
    validate_r6_independent_reaudit(load_json(_ROOT / PILOT_R6_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r7_dual_source_amendment(load_json(_ROOT / FULL_PROTOCOL_R7_AMENDMENT_REL), root=_ROOT)
    freeze = load_json(_ROOT / PILOT_R7_VISUAL_FREEZE_REL)
    validate_r7_visual_freeze_list(freeze)
    bind = _require_clean()
    obj = {
        "schema": "ms_sta_v5_d0_r7_pilot_run_contract_v1",
        "status": "ACTIVE",
        "campaign": "D0_PILOT_R7",
        "scope": "DUAL_SOURCE_SKELETON_RGB_ROI_PILOT_TRAIN_VAL_ONLY",
        "code_lock_git_revision": bind["code_lock_git_revision"],
        "executable_tree_sha256": bind["executable_tree_sha256"],
        "worktree_dirty": False,
        "r6_pilot_audit_sha256": EXPECTED_R6_PILOT_AUDIT_SHA256,
        "r6_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R6_INDEPENDENT_REAUDIT_REL),
        "r7_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R7_AMENDMENT_REL),
        "canonical_full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "pilot_sample_sha256": EXPECTED_R1_SAMPLE_SHA256,
        "freeze_sha256": sha256_file(_ROOT / PILOT_R7_VISUAL_FREEZE_REL),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "forbid_full_corpus": True,
        "forbid_push": True,
        "finalizer_must_not_synthesize_manual": True,
    }
    path = _ROOT / PILOT_R7_RUN_CONTRACT_REL
    if path.is_file() and load_json(path).get("status") == "CLOSED":
        raise RuntimeError("r7_contract_closed")
    return write_json(path, obj, overwrite=True)


def _save_skel(out_dir: Path, sample_id: str, payload: dict[str, Any], recovery: dict[str, Any]) -> str:
    path = out_dir / "skel_v3" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)

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
        "handedness_hard_reject_disabled_for_non_mirrored_input": True,
    }
    (out_dir / "skel_v3" / f"{sample_id}.recovery.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return sha256_file(path)


def _save_rgb(out_dir: Path, sample_id: str, payload: dict[str, Any]) -> str:
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
        "enable_pose_wrist_prior": payload.get("enable_pose_wrist_prior"),
        "dual_source": {
            "skeleton": ["holistic", "hands_fallback", "missing"],
            "rgb_roi": ["hand_landmarks", "pose_wrist_prior", "missing"],
        },
    }
    (out_dir / "rgb_token_v3" / f"{sample_id}.meta.json").write_bytes(
        (json.dumps(meta, indent=2) + "\n").encode("utf-8")
    )
    np.savez_compressed(path, **kwargs)
    return sha256_file(path)


def _extract_skel_one(sample: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    from src.data_prep.hand_recovery_v6 import HandRecoveryEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    engine = HandRecoveryEngine(arm="C3", model_complexity=0, store_stride=2)
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
    digest = _save_skel(out_dir, sample["sample_id"], payload, rec)
    return {"sample_id": sample["sample_id"], "ok": True, "skel_sha256": digest, "arm": "C3"}


def run_c0_bit_exact_all16() -> dict[str, Any]:
    from src.data_prep.hand_recovery_v6 import HandRecoveryEngine

    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    rows = []
    all_ok = True
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        r4 = np.load(_ROOT / PILOT_R4_OUT_DIR_REL / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        eng = HandRecoveryEngine(arm="C0", model_complexity=0, store_stride=2)
        try:
            rec = eng.process_video(_ROOT / s["video_rel"])
        finally:
            eng.close()
        idx_a = {int(i): n for n, i in enumerate(rec["original_frame_index"].tolist())}
        idx_b = {int(i): n for n, i in enumerate(r4["original_frame_index"].tolist())}
        common = sorted(set(idx_a) & set(idx_b))
        aa = np.stack([rec["raw_holistic_missing"][idx_a[i]] for i in common])
        bb = np.stack([r4["missing_mask_prefill"][idx_b[i]] for i in common])
        bit_exact = bool(np.array_equal(aa, bb))
        all_ok = all_ok and bit_exact
        rows.append({"sample_id": sid, "bit_exact": bit_exact, "n_common_frames": len(common)})
    summary = {
        "schema": "ms_sta_v5_d0_r7_diagnostic_matrix_v1",
        "c0_scope": "all_16_pilot_samples",
        "c0_all_bit_exact": all_ok,
        "n_samples": len(rows),
        "samples": rows,
    }
    write_json(_ROOT / PILOT_R7_DIAGNOSTIC_REL, summary, overwrite=True)
    return summary


def run_skel(*, resume: bool = False) -> dict[str, Any]:
    bind = _code_lock_bind()
    out_dir = _ROOT / PILOT_R7_OUT_DIR_REL
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    results = []
    for s in sample_set["samples"]:
        dest = out_dir / "skel_v3" / f"{s['sample_id']}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": s["sample_id"], "ok": True, "resumed": True})
            continue
        results.append(_extract_skel_one(s, out_dir))
    summary = {
        "phase": "skel",
        "ok": all(r.get("ok") for r in results),
        "results": results,
        "provenance": {
            "code_lock_git_revision": bind["code_lock_git_revision"],
            "executable_tree_sha256": bind["executable_tree_sha256"],
        },
    }
    write_json(out_dir / "SKEL_PHASE_SUMMARY.json", summary, overwrite=True)
    return summary


def run_rgb(*, resume: bool = False) -> dict[str, Any]:
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    bind = _code_lock_bind()
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    out_dir = _ROOT / PILOT_R7_OUT_DIR_REL
    visual_dir = _ROOT / PILOT_R7_VISUAL_REL
    visual_dir.mkdir(parents=True, exist_ok=True)
    freeze = load_json(_ROOT / PILOT_R7_VISUAL_FREEZE_REL)
    qa_keys = {(w["sample_id"], int(w["window_start_idx"])) for w in freeze["windows"]}
    for n in freeze.get("truly_absent_negative_controls") or []:
        if n.get("kind") == "unit_backed":
            continue
        if n.get("sample_id") is None:
            continue
        qa_keys.add((n["sample_id"], int(n["window_start_idx"])))

    extractor = FrozenSlowFastBlock4(
        checkpoint_path=_ROOT / EXPECTED_SLOWFAST_CKPT_REL,
        expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
        device="cpu",
    )
    results = []
    overlays_index: list[dict[str, Any]] = []
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "rgb_token_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True})
            continue
        skel = np.load(out_dir / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        landmarks_by_frame = {}
        missing_by_frame = {}
        # Skeleton effective landmarks/missing (may be missing while RGB uses pose prior)
        for i, fi in enumerate(skel["original_frame_index"].tolist()):
            landmarks_by_frame[int(fi)] = skel["joints_raw"][i]
            missing_by_frame[int(fi)] = skel["missing_mask_prefill"][i]
        payload = extract_rgb_token_v3_for_video(
            video_path=_ROOT / s["video_rel"],
            extractor=extractor,
            landmarks_by_frame=landmarks_by_frame,
            missing_by_frame=missing_by_frame,
            skel_frame_index=skel["original_frame_index"],
            skel_timestamps=skel["timestamps_sec"],
            overlay_dir=visual_dir / sid,
            sample_id=sid,
            enable_pose_wrist_prior=True,
        )
        digest = _save_rgb(out_dir, sid, payload)
        for ov in payload.get("overlays") or []:
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
                    "rgb_roi_sources": ov.get("rgb_roi_sources"),
                    "n_tiles": ov.get("n_tiles"),
                    "slice_native_frames": ov.get("slice_native_frames"),
                    "tiles": ov.get("tiles"),
                }
            )
        results.append({"sample_id": sid, "ok": True, "rgb_sha256": digest})
    summary = {
        "phase": "rgb",
        "ok": all(r.get("ok") for r in results),
        "enable_pose_wrist_prior": True,
        "results": results,
        "overlays_index": overlays_index,
        "provenance": {
            "code_lock_git_revision": bind["code_lock_git_revision"],
            "executable_tree_sha256": bind["executable_tree_sha256"],
        },
    }
    write_json(out_dir / "RGB_PHASE_SUMMARY.json", summary, overwrite=True)
    write_json(
        _ROOT / PILOT_R7_OVERLAYS_INDEX_REL,
        {"ok": True, "n_overlays": len(overlays_index), "overlays_index": overlays_index},
        overwrite=True,
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=["contract", "c0_all16", "skel", "rgb", "all"],
        default="contract",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    validate_r6_independent_reaudit(load_json(_ROOT / PILOT_R6_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r7_dual_source_amendment(load_json(_ROOT / FULL_PROTOCOL_R7_AMENDMENT_REL), root=_ROOT)
    if sha256_file(_ROOT / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        raise RuntimeError("canonical_protocol_drift")
    if sha256_file(_ROOT / PILOT_R6_AUDIT_REL) != EXPECTED_R6_PILOT_AUDIT_SHA256:
        raise RuntimeError("r6_audit_drift")

    if args.phase in {"contract", "all"}:
        print(write_run_contract())
        if args.phase == "contract":
            return 0
    if args.phase in {"c0_all16", "all"}:
        out = run_c0_bit_exact_all16()
        print(json.dumps({"ok": out["c0_all_bit_exact"], "n": out["n_samples"]}, indent=2))
        if args.phase == "c0_all16":
            return 0 if out["c0_all_bit_exact"] else 1
    if args.phase in {"skel", "all"}:
        out = run_skel(resume=args.resume)
        print(json.dumps({"ok": out["ok"], "n": len(out["results"])}, indent=2))
        if args.phase == "skel":
            return 0 if out["ok"] else 1
    if args.phase in {"rgb", "all"}:
        out = run_rgb(resume=args.resume)
        print(json.dumps({"ok": out["ok"], "n_overlays": len(out["overlays_index"])}, indent=2))
        if args.phase == "rgb":
            return 0 if out["ok"] else 1
    if args.phase == "all":
        print(f"[STOP] Write human manual {PILOT_R7_VISUAL_MANUAL_REL} then run run_ms_sta_v5_d0_r7_finalize.py")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
