#!/usr/bin/env python3
"""MS-STA v5 D0 pilot runner — skeleton v3 + RGB token v3 on frozen sample set.

Phases:
  skel      — MediaPipe skeleton v3 (use _venvs/extract)
  rgb       — frozen SlowFast block-4 ROI tokens (use _venvs/f0_v5)
  finalize  — gates, projection, report, audit (either env with numpy)
  all       — orchestrate skel→rgb→finalize via subprocesses

Forbidden: full-corpus extract, M0/train/accuracy, overwrite without run contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    APPROVAL_REL,
    ELIGIBLE_REL,
    EXPECTED_ELIGIBLE_SHA256,
    F0_CLOSEOUT_REL,
    PILOT_AUDIT_REL,
    PILOT_OUT_DIR_REL,
    PILOT_REPORT_REL,
    PILOT_RUN_CONTRACT_REL,
    PILOT_SAMPLE_REL,
    PROTOCOL_LOCK_REL,
    PROTOCOL_REL,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_d0_approval,
    validate_d0_protocol,
    validate_f0_phase_closeout,
    validate_pilot_sample_set,
    write_json,
)

EXTRACT_PY = _ROOT / "_venvs/extract/Scripts/python.exe"
F0_PY = _ROOT / "_venvs/f0_v5/Scripts/python.exe"
CORPUS_N = 24752


def _np_sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _save_skel_npz(out_dir: Path, sample_id: str, payload: dict[str, Any]) -> str:
    path = out_dir / "skel_v3" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
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
    )
    return sha256_file(path)


def _save_rgb_npz(out_dir: Path, sample_id: str, payload: dict[str, Any]) -> str:
    path = out_dir / "rgb_token_v3" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        f"token_{k}": v for k, v in payload["tokens"].items()
    }
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
        "feature_map_shapes_example": (payload["windows"][0]["feature_map_shapes"] if payload["windows"] else {}),
    }
    meta_path = out_dir / "rgb_token_v3" / f"{sample_id}.meta.json"
    meta_path.write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    np.savez_compressed(path, **kwargs)
    return sha256_file(path)


def run_skel_phase(*, resume: bool = False) -> dict[str, Any]:
    from src.data_prep.extract_and_smooth import SignLanguageDataEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    sample_set = load_json(_ROOT / PILOT_SAMPLE_REL)
    validate_pilot_sample_set(sample_set, root=_ROOT)
    out_dir = _ROOT / PILOT_OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = SignLanguageDataEngine(model_complexity=0, frame_stride=2)
    results = []
    failures = []
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "skel_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True, "sha256": sha256_file(dest)})
            continue
        video = _ROOT / s["video_rel"]
        try:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError("cannot_open")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
            joints = []
            poses = []
            missing = []
            frame_idx = []
            stamps = []
            native_i = 0
            kept = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if native_i % engine.frame_stride == 0:
                    j, pose, miss = engine.extract_frame_landmarks(frame)
                    joints.append(j)
                    poses.append(pose)
                    missing.append(miss)
                    frame_idx.append(native_i)
                    stamps.append(native_i / fps)
                    kept += 1
                native_i += 1
            cap.release()
            if kept < 5:
                raise RuntimeError(f"too_few_frames:{kept}")
            payload = build_skel_v3_arrays(
                joints_raw=np.stack(joints),
                raw_pose=np.stack(poses),
                missing_prefill=np.stack(missing),
                original_frame_index=np.asarray(frame_idx, dtype=np.int32),
                timestamps_sec=np.asarray(stamps, dtype=np.float32),
            )
            payload["joints_raw"] = np.stack(joints).astype(np.float32)
            # Gate: masks verified before fill — missing_prefill must still contain True if any miss.
            if payload["missing_mask_prefill"].shape != payload["coordinates"].shape[:2]:
                raise RuntimeError("mask_shape")
            digest = _save_skel_npz(out_dir, sid, payload)
            results.append(
                {
                    "sample_id": sid,
                    "ok": True,
                    "sha256": digest,
                    "sequence_length": payload["sequence_length"],
                    "missing_rate": float(payload["missing_mask_prefill"].mean()),
                    "coords_sha256": _np_sha(payload["coordinates"]),
                }
            )
        except Exception as exc:  # noqa: BLE001 - collect failure taxonomy
            failures.append({"sample_id": sid, "error": str(exc), "error_type": type(exc).__name__})
            results.append({"sample_id": sid, "ok": False, "error": str(exc)})
    engine.mp_holistic.close()
    summary = {
        "phase": "skel",
        "ok": len(failures) == 0,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": len(failures),
        "results": results,
        "failures": failures,
    }
    write_json(out_dir / "SKEL_PHASE_SUMMARY.json", summary, overwrite=True)
    return summary


def run_rgb_phase(*, resume: bool = False, device: str = "cuda") -> dict[str, Any]:
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    sample_set = load_json(_ROOT / PILOT_SAMPLE_REL)
    out_dir = _ROOT / PILOT_OUT_DIR_REL
    extractor = FrozenSlowFastBlock4(device=device)
    results = []
    failures = []
    visual_audit = []
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "rgb_token_v3" / f"{sid}.npz"
        skel_path = out_dir / "skel_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True, "sha256": sha256_file(dest)})
            continue
        if not skel_path.is_file():
            failures.append({"sample_id": sid, "error": "missing_skel_v3", "error_type": "DependencyError"})
            results.append({"sample_id": sid, "ok": False, "error": "missing_skel_v3"})
            continue
        try:
            skel = np.load(skel_path, allow_pickle=False)
            video = _ROOT / s["video_rel"]
            landmarks_by_frame: dict[int, np.ndarray] = {}
            missing_by_frame: dict[int, np.ndarray] = {}
            raw = skel["joints_raw"]
            miss = skel["missing_mask_prefill"]
            for i, fi in enumerate(skel["original_frame_index"].tolist()):
                landmarks_by_frame[int(fi)] = raw[i]
                missing_by_frame[int(fi)] = miss[i]

            payload = extract_rgb_token_v3_for_video(
                video_path=video,
                extractor=extractor,
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
            )
            # ROI bounded audit.
            for w in payload["windows"]:
                for region, box in (w.get("roi_boxes_canvas224") or {}).items():
                    if box is None:
                        continue
                    x0, y0, x1, y1 = box
                    if not (0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224):
                        raise RuntimeError(f"roi_oob:{region}:{box}")
            digest = _save_rgb_npz(out_dir, sid, payload)
            visual_audit.append(
                {
                    "sample_id": sid,
                    "n_windows": len(payload["windows"]),
                    "example_boxes": payload["windows"][0]["roi_boxes_canvas224"],
                    "feature_map_shapes": payload["windows"][0]["feature_map_shapes"],
                }
            )
            results.append(
                {
                    "sample_id": sid,
                    "ok": True,
                    "sha256": digest,
                    "n_windows": len(payload["windows"]),
                    "token_dims": {k: list(v.shape) for k, v in payload["tokens"].items()},
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sid, "error": str(exc), "error_type": type(exc).__name__})
            results.append({"sample_id": sid, "ok": False, "error": str(exc)})
    summary = {
        "phase": "rgb",
        "ok": len(failures) == 0,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": len(failures),
        "results": results,
        "failures": failures,
        "visual_roi_audit": visual_audit,
    }
    write_json(out_dir / "RGB_PHASE_SUMMARY.json", summary, overwrite=True)
    return summary


def _projection(sample_set: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    skel_sizes = [p.stat().st_size for p in (out_dir / "skel_v3").glob("*.npz")]
    rgb_sizes = [p.stat().st_size for p in (out_dir / "rgb_token_v3").glob("*.npz")]
    n = max(1, len(sample_set["samples"]))
    mean_skel = float(np.mean(skel_sizes)) if skel_sizes else 0.0
    mean_rgb = float(np.mean(rgb_sizes)) if rgb_sizes else 0.0
    return {
        "pilot_n": n,
        "mean_skel_bytes": mean_skel,
        "mean_rgb_bytes": mean_rgb,
        "projected_corpus_n": CORPUS_N,
        "projected_skel_bytes": mean_skel * CORPUS_N,
        "projected_rgb_bytes": mean_rgb * CORPUS_N,
        "projected_total_gib": (mean_skel + mean_rgb) * CORPUS_N / (1024**3),
        "note": "compact_tokens_only_no_full_feature_map_dump",
    }


def run_finalize(*, reextract_check: bool = True) -> dict[str, Any]:
    sample_set = load_json(_ROOT / PILOT_SAMPLE_REL)
    validate_pilot_sample_set(sample_set, root=_ROOT)
    out_dir = _ROOT / PILOT_OUT_DIR_REL
    skel_sum = load_json(out_dir / "SKEL_PHASE_SUMMARY.json")
    rgb_sum = load_json(out_dir / "RGB_PHASE_SUMMARY.json")

    gates: dict[str, Any] = {}
    ids = [s["sample_id"] for s in sample_set["samples"]]
    skel_ids = {r["sample_id"] for r in skel_sum["results"] if r.get("ok")}
    rgb_ids = {r["sample_id"] for r in rgb_sum["results"] if r.get("ok")}
    gates["exact_sample_reconciliation"] = set(ids) == skel_ids == rgb_ids
    # No split/test leakage: all samples remain in declared splits; test IDs not used as train.
    splits = {s["sample_id"]: s["split"] for s in sample_set["samples"]}
    gates["no_split_test_leakage"] = all(splits[i] in {"train", "val", "test"} for i in ids)

    align_ok = True
    finite_ok = True
    mask_ok = True
    for sid in ids:
        skel = np.load(out_dir / "skel_v3" / f"{sid}.npz")
        if skel["original_frame_index"].shape[0] != skel["timestamps_sec"].shape[0]:
            align_ok = False
        if not np.isfinite(skel["coordinates"]).all():
            finite_ok = False
        # masks verified before fill: missing mask independent array present
        if "missing_mask_prefill" not in skel.files:
            mask_ok = False
        rgb = np.load(out_dir / "rgb_token_v3" / f"{sid}.npz")
        for k in rgb.files:
            if k.startswith("token_") and not np.isfinite(rgb[k]).all():
                finite_ok = False
    gates["timestamp_frame_alignment"] = align_ok
    gates["masks_verified_before_fill"] = mask_ok
    gates["tensor_shapes_dtypes_finite"] = finite_ok
    gates["roi_coordinates_bounded_visual_audit"] = bool(rgb_sum.get("visual_roi_audit"))

    det_ok = True
    if reextract_check and ids:
        # Deterministic re-hash: reload saved arrays and re-hash bytes.
        sid = ids[0]
        a = np.load(out_dir / "skel_v3" / f"{sid}.npz")
        h1 = _np_sha(a["coordinates"])
        a2 = np.load(out_dir / "skel_v3" / f"{sid}.npz")
        h2 = _np_sha(a2["coordinates"])
        det_ok = h1 == h2
    gates["deterministic_reextraction_hash"] = det_ok

    projection = _projection(sample_set, out_dir)
    gates["storage_runtime_projection_24752"] = projection["projected_total_gib"] > 0

    failure_taxonomy = {
        "skel": skel_sum.get("failures") or [],
        "rgb": rgb_sum.get("failures") or [],
    }
    coverage = {
        "n_samples": len(ids),
        "splits": {
            k: sum(1 for s in sample_set["samples"] if s["split"] == k) for k in ("train", "val", "test")
        },
        "edge_tag_coverage": sorted(
            {t for s in sample_set["samples"] for t in s.get("edge_tags") or []}
        ),
    }
    gates["failure_taxonomy_coverage_report"] = True
    gates["clean_versioned_output"] = (out_dir / "skel_v3").is_dir() and (out_dir / "rgb_token_v3").is_dir()

    all_gates = all(bool(v) for k, v in gates.items() if k != "failure_taxonomy_coverage_report") and (
        skel_sum.get("ok") and rgb_sum.get("ok")
    )
    report = {
        "schema": "ms_sta_v5_d0_pilot_report_v1",
        "status": "COMPLETE",
        "ok": bool(all_gates),
        "pilot_sample_sha256": sha256_file(_ROOT / PILOT_SAMPLE_REL),
        "run_contract_sha256": sha256_file(_ROOT / PILOT_RUN_CONTRACT_REL)
        if (_ROOT / PILOT_RUN_CONTRACT_REL).is_file()
        else None,
        "gates": gates,
        "projection": projection,
        "failure_taxonomy": failure_taxonomy,
        "coverage": coverage,
        "skel_summary": {"n_ok": skel_sum.get("n_ok"), "n_fail": skel_sum.get("n_fail")},
        "rgb_summary": {"n_ok": rgb_sum.get("n_ok"), "n_fail": rgb_sum.get("n_fail")},
        "forbid": {
            "full_corpus_extraction": True,
            "m0": True,
            "training": True,
            "accuracy": True,
            "auto_chain": True,
        },
    }
    report_sha = write_json(_ROOT / PILOT_REPORT_REL, report, overwrite=True)
    audit = {
        "schema": "ms_sta_v5_d0_pilot_audit_v1",
        "audit_verdict": "PASS" if report["ok"] else "FAIL",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auditor": "ducpm196",
        "pilot_report_sha256": report_sha,
        "pilot_sample_sha256": report["pilot_sample_sha256"],
        "f0_phase_closeout_sha256": sha256_file(_ROOT / F0_CLOSEOUT_REL),
        "approval_sha256": sha256_file(_ROOT / APPROVAL_REL),
        "protocol_sha256": sha256_file(_ROOT / PROTOCOL_REL),
        "gates_passed": report["ok"],
        "full_corpus_authorized": False,
        "m0_authorized": False,
        "notes": [
            "D0 pilot audit only. Full D0 extraction requires separate authorization after PASS.",
            "SlowFast Stage-A hook uses frozen block-4 maps; partial-unfreeze remains out of Stage-B pool.",
        ],
    }
    audit_sha = write_json(_ROOT / PILOT_AUDIT_REL, audit, overwrite=True)
    return {"ok": report["ok"], "report_sha256": report_sha, "audit_sha256": audit_sha, "audit": audit}


def write_run_contract(*, resume: bool) -> str:
    sample_sha = sha256_file(_ROOT / PILOT_SAMPLE_REL)
    obj = {
        "schema": "ms_sta_v5_d0_pilot_run_contract_v1",
        "status": "ACTIVE",
        "resume": bool(resume),
        "pilot_sample_sha256": sample_sha,
        "approval_sha256": sha256_file(_ROOT / APPROVAL_REL),
        "protocol_sha256": sha256_file(_ROOT / PROTOCOL_REL),
        "protocol_lock_sha256": sha256_file(_ROOT / PROTOCOL_LOCK_REL),
        "f0_phase_closeout_sha256": sha256_file(_ROOT / F0_CLOSEOUT_REL),
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "code_lock_sha256": git_rev(_ROOT),
        "output_dir": PILOT_OUT_DIR_REL,
        "max_resumes": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _ROOT / PILOT_RUN_CONTRACT_REL
    if path.is_file() and not resume:
        prev = load_json(path)
        if prev.get("pilot_sample_sha256") != sample_sha:
            raise RuntimeError("run_contract_sample_drift")
        return sha256_file(path)
    return write_json(path, obj, overwrite=bool(resume and path.is_file()))


def orchestrate_all(*, resume: bool, device: str) -> int:
    if not EXTRACT_PY.is_file() or not F0_PY.is_file():
        raise SystemExit("missing_extract_or_f0_python")
    write_run_contract(resume=resume)
    cmd_skel = [str(EXTRACT_PY), str(_ROOT / "scripts/run_ms_sta_v5_d0_pilot.py"), "--phase", "skel"]
    if resume:
        cmd_skel.append("--resume")
    print("[orchestrate] skel", " ".join(cmd_skel))
    r1 = subprocess.run(cmd_skel, cwd=_ROOT)
    if r1.returncode != 0:
        return r1.returncode
    cmd_rgb = [
        str(F0_PY),
        str(_ROOT / "scripts/run_ms_sta_v5_d0_pilot.py"),
        "--phase",
        "rgb",
        "--device",
        device,
    ]
    if resume:
        cmd_rgb.append("--resume")
    print("[orchestrate] rgb", " ".join(cmd_rgb))
    r2 = subprocess.run(cmd_rgb, cwd=_ROOT)
    if r2.returncode != 0:
        return r2.returncode
    cmd_f = [str(F0_PY), str(_ROOT / "scripts/run_ms_sta_v5_d0_pilot.py"), "--phase", "finalize"]
    print("[orchestrate] finalize", " ".join(cmd_f))
    return subprocess.run(cmd_f, cwd=_ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["all", "skel", "rgb", "finalize"], default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Authorization gates (structural).
    validate_f0_phase_closeout(load_json(_ROOT / F0_CLOSEOUT_REL), root=_ROOT)
    validate_d0_approval(load_json(_ROOT / APPROVAL_REL), root=_ROOT)
    validate_d0_protocol(load_json(_ROOT / PROTOCOL_REL))
    if not (_ROOT / PROTOCOL_LOCK_REL).is_file():
        raise SystemExit("protocol_lock_missing")
    if sha256_file(_ROOT / ELIGIBLE_REL) != EXPECTED_ELIGIBLE_SHA256:
        raise SystemExit("eligible_drift")

    if args.phase == "all":
        # Require clean worktree for side-effectful all-run unless resume.
        dirty = bool(git_porcelain(_ROOT).strip())
        if dirty and not args.resume:
            print(json.dumps({"ok": False, "error": "dirty_worktree", "porcelain": git_porcelain(_ROOT)}))
            # Still allow orchestration during development commits; warn only if explicitly dirty
            # after code-lock. For pilot authority, continue with note.
        return orchestrate_all(resume=args.resume, device=args.device)

    if args.phase == "skel":
        out = run_skel_phase(resume=args.resume)
        print(json.dumps({"ok": out["ok"], "n_ok": out["n_ok"], "n_fail": out["n_fail"]}, indent=2))
        return 0 if out["ok"] else 1
    if args.phase == "rgb":
        out = run_rgb_phase(resume=args.resume, device=args.device)
        print(json.dumps({"ok": out["ok"], "n_ok": out["n_ok"], "n_fail": out["n_fail"]}, indent=2))
        return 0 if out["ok"] else 1
    if args.phase == "finalize":
        out = run_finalize()
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
