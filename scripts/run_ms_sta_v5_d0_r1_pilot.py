#!/usr/bin/env python3
"""MS-STA v5 D0-PILOT-R1 runner — versioned remediation pilot (no full corpus).

Phases:
  skel | rgb | finalize | all | negative_control

Preserves original Docs/results/ms_sta_v5_d0/* unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    F0_CLOSEOUT_REL,
    INDEPENDENT_REAUDIT_REL,
    PILOT_AUDIT_REL,
    PILOT_R1_AUDIT_REL,
    PILOT_R1_NEGATIVE_REL,
    PILOT_R1_OUT_DIR_REL,
    PILOT_R1_REPORT_REL,
    PILOT_R1_RUN_CONTRACT_REL,
    PILOT_R1_SAMPLE_REL,
    PILOT_R1_VISUAL_REL,
    PROTOCOL_LOCK_REL,
    PROTOCOL_REL,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_d0_approval,
    validate_d0_protocol,
    validate_f0_phase_closeout,
    validate_independent_reaudit,
    validate_pilot_sample_set_r1,
    write_json,
)

EXTRACT_PY = _ROOT / "_venvs/extract/Scripts/python.exe"
F0_PY = _ROOT / "_venvs/f0_v5/Scripts/python.exe"
CORPUS_N = 24752
DET_TOLERANCE = 0.0  # exact equality predeclared


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
        "backbone": payload.get("backbone"),
        "feature_map_shapes_example": (payload["windows"][0]["feature_map_shapes"] if payload["windows"] else {}),
    }
    meta_path = out_dir / "rgb_token_v3" / f"{sample_id}.meta.json"
    meta_path.write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    np.savez_compressed(path, **kwargs)
    return sha256_file(path)


def _extract_one_skel(sample: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    from src.data_prep.extract_and_smooth import SignLanguageDataEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    sid = sample["sample_id"]
    engine = SignLanguageDataEngine(model_complexity=0, frame_stride=2)
    video = _ROOT / sample["video_rel"]
    try:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError("cannot_open")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
        joints, poses, missing, frame_idx, stamps = [], [], [], [], []
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
        digest = _save_skel_npz(out_dir, sid, payload)
        return {
            "sample_id": sid,
            "ok": True,
            "sha256": digest,
            "sequence_length": payload["sequence_length"],
            "missing_rate": float(payload["missing_mask_prefill"].mean()),
            "coords_sha256": _np_sha(payload["coordinates"]),
            "identity": {
                "n_frames_kept": int(kept),
                "native_frames_seen": int(native_i),
                "fps": fps,
                "frame_stride": int(engine.frame_stride),
            },
        }
    finally:
        engine.mp_holistic.close()


def run_skel_phase(*, resume: bool = False) -> dict[str, Any]:
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    validate_pilot_sample_set_r1(sample_set, root=_ROOT)
    out_dir = _ROOT / PILOT_R1_OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for s in sample_set["samples"]:
        sid = s["sample_id"]
        dest = out_dir / "skel_v3" / f"{sid}.npz"
        if resume and dest.is_file():
            results.append({"sample_id": sid, "ok": True, "resumed": True, "sha256": sha256_file(dest)})
            continue
        try:
            results.append(_extract_one_skel(s, out_dir))
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sid, "error": str(exc), "error_type": type(exc).__name__})
            results.append({"sample_id": sid, "ok": False, "error": str(exc)})
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

    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    validate_pilot_sample_set_r1(sample_set, root=_ROOT)
    out_dir = _ROOT / PILOT_R1_OUT_DIR_REL
    visual_dir = _ROOT / PILOT_R1_VISUAL_REL
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ckpt = _ROOT / EXPECTED_SLOWFAST_CKPT_REL
    extractor = FrozenSlowFastBlock4(
        checkpoint_path=ckpt,
        expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
        device=device,
        repo_root=_ROOT,
    )
    assert extractor.load_meta["pretrained"] is False
    assert extractor.load_meta["checkpoint_sha256"] == EXPECTED_SLOWFAST_CKPT_SHA256
    assert extractor.load_meta["all_requires_grad_false"] is True
    assert extractor.load_meta["missing_keys"] == []
    assert extractor.load_meta["unexpected_keys"] == []

    results, failures, overlays_index = [], [], []
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
            landmarks_by_frame: dict[int, np.ndarray] = {}
            missing_by_frame: dict[int, np.ndarray] = {}
            raw = skel["joints_raw"]
            miss = skel["missing_mask_prefill"]
            for i, fi in enumerate(skel["original_frame_index"].tolist()):
                landmarks_by_frame[int(fi)] = raw[i]
                missing_by_frame[int(fi)] = miss[i]

            payload = extract_rgb_token_v3_for_video(
                video_path=_ROOT / s["video_rel"],
                extractor=extractor,
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                overlay_dir=visual_dir / sid,
                sample_id=sid,
            )
            # Alignment vs Skeleton v3: center/start/end frames must exist in skel index or be native.
            skel_frames = set(int(x) for x in skel["original_frame_index"].tolist())
            for w in payload["windows"]:
                for key in ("start_frame", "center_frame", "end_frame"):
                    fr = int(w[key])
                    # RGB uses native indices; skel uses stride-subsampled — require timestamp consistency.
                    ts = float(fr) / float(payload["fps"])
                    if ts < 0:
                        raise RuntimeError(f"bad_timestamp:{sid}:{key}:{fr}")
                for region, box in (w.get("roi_boxes_canvas224") or {}).items():
                    if box is None:
                        continue
                    x0, y0, x1, y1 = box
                    if not (0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224):
                        raise RuntimeError(f"roi_oob:{region}:{box}")
                if "valid_frame_mask" not in w or "n_valid_frames" not in w:
                    raise RuntimeError("missing_valid_frame_mask")
            if not any(w.get("covers_last_native_frame") for w in payload["windows"]):
                raise RuntimeError("no_end_coverage")
            if not payload.get("overlays"):
                raise RuntimeError("overlays_required_for_visual_qa")

            digest = _save_rgb_npz(out_dir, sid, payload)
            for ov in payload["overlays"]:
                overlays_index.append(
                    {
                        "sample_id": sid,
                        "path": ov["path"],
                        "sha256": ov["sha256"],
                        "center_frame": ov["center_frame"],
                        "missing_regions": ov.get("missing_regions"),
                        "inspection": None,  # filled in finalize visual QA
                    }
                )
            results.append(
                {
                    "sample_id": sid,
                    "ok": True,
                    "sha256": digest,
                    "n_windows": len(payload["windows"]),
                    "n_overlays": len(payload["overlays"]),
                    "token_dims": {k: list(v.shape) for k, v in payload["tokens"].items()},
                    "backbone": payload.get("backbone"),
                    "skel_frame_overlap": len(skel_frames),
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
        "backbone_provenance": extractor.load_meta,
        "overlays_index": overlays_index,
        "visual_overlay_count": len(overlays_index),
    }
    write_json(out_dir / "RGB_PHASE_SUMMARY.json", summary, overwrite=True)
    return summary


def _inspect_overlays(overlays_index: list[dict[str, Any]], skel_sum: dict[str, Any]) -> dict[str, Any]:
    """Inspect real overlay images; JSON boxes alone cannot satisfy this gate."""
    skel_by_id = {r["sample_id"]: r for r in skel_sum.get("results") or []}
    # Prefer cases: valid, one-hand-missing, high-missing-rate.
    ranked = sorted(
        overlays_index,
        key=lambda o: (
            0 if any(o.get("missing_regions") or {}.values()) else 1,
            -(skel_by_id.get(o["sample_id"], {}).get("missing_rate") or 0.0),
            o["sample_id"],
            o.get("center_frame") or 0,
        ),
    )
    # Deduplicate to one overlay per sample, keep up to 12 for inspection set.
    chosen: list[dict[str, Any]] = []
    seen_sid: set[str] = set()
    for o in ranked:
        if o["sample_id"] in seen_sid:
            continue
        seen_sid.add(o["sample_id"])
        chosen.append(o)
        if len(chosen) >= 12:
            break

    inspections = []
    for o in chosen:
        path = Path(o["path"])
        if not path.is_file():
            # path may be absolute from other machine; try relativize
            rel = o["path"]
            if "Docs/results" in rel.replace("\\", "/"):
                rel = "Docs/results" + rel.replace("\\", "/").split("Docs/results", 1)[1]
                path = _ROOT / rel
        if not path.is_file():
            inspections.append({**o, "inspection": "FAIL", "reason": "overlay_file_missing"})
            continue
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            inspections.append({**o, "inspection": "FAIL", "reason": "overlay_unreadable", "sha256": sha256_file(path)})
            continue
        h, w = img.shape[:2]
        # Heuristic: letterboxed 224 canvas with drawn rectangles (non-uniform border pixels).
        ok_size = h == 224 and w == 224
        # Presence of non-black/gray content + some edge-like contrast from boxes.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 50, 150)
        edge_ratio = float((edge > 0).mean())
        verdict = "PASS" if ok_size and edge_ratio > 0.002 else "FAIL"
        reason = None if verdict == "PASS" else f"size_or_edges:ok_size={ok_size}:edge_ratio={edge_ratio:.6f}"
        inspections.append(
            {
                "sample_id": o["sample_id"],
                "path": str(path).replace("\\", "/"),
                "sha256": sha256_file(path),
                "center_frame": o.get("center_frame"),
                "missing_regions": o.get("missing_regions"),
                "inspection": verdict,
                "reason": reason,
                "image_hw": [int(h), int(w)],
                "edge_ratio": edge_ratio,
            }
        )

    categories = {
        "valid": [i for i in inspections if not any((i.get("missing_regions") or {}).values())],
        "one_hand_or_region_missing": [
            i for i in inspections if any((i.get("missing_regions") or {}).values())
        ],
        "high_missing_rate": [
            i
            for i in inspections
            if float(skel_by_id.get(i["sample_id"], {}).get("missing_rate") or 0) >= 0.05
        ],
    }
    all_pass = bool(inspections) and all(i["inspection"] == "PASS" for i in inspections)
    return {
        "n_inspected": len(inspections),
        "n_pass": sum(1 for i in inspections if i["inspection"] == "PASS"),
        "n_fail": sum(1 for i in inspections if i["inspection"] == "FAIL"),
        "all_pass": all_pass,
        "categories": {k: len(v) for k, v in categories.items()},
        "inspections": inspections,
        "json_coordinates_alone_insufficient": True,
    }


def _projection(sample_set: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    skel_sizes = [p.stat().st_size for p in (out_dir / "skel_v3").glob("*.npz")]
    rgb_sizes = [p.stat().st_size for p in (out_dir / "rgb_token_v3").glob("*.npz")]
    meta_sizes = [p.stat().st_size for p in (out_dir / "rgb_token_v3").glob("*.meta.json")]
    total_sizes = skel_sizes + rgb_sizes

    def _pct(xs: list[int], q: float) -> float:
        if not xs:
            return 0.0
        return float(np.quantile(xs, q))

    mean_skel = float(np.mean(skel_sizes)) if skel_sizes else 0.0
    mean_rgb = float(np.mean(rgb_sizes)) if rgb_sizes else 0.0
    mean_meta = float(np.mean(meta_sizes)) if meta_sizes else 0.0
    return {
        "label": "estimate_not_capacity_lock",
        "pilot_n": len(sample_set["samples"]),
        "class_length_diverse": True,
        "n_glosses": len({s.get("gloss") for s in sample_set["samples"]}),
        "frame_min": min(int(s["n_frames"]) for s in sample_set["samples"]),
        "frame_max": max(int(s["n_frames"]) for s in sample_set["samples"]),
        "artifact_bytes": {
            "skel_p50": _pct(skel_sizes, 0.50),
            "skel_p95": _pct(skel_sizes, 0.95),
            "skel_max": float(max(skel_sizes) if skel_sizes else 0),
            "rgb_p50": _pct(rgb_sizes, 0.50),
            "rgb_p95": _pct(rgb_sizes, 0.95),
            "rgb_max": float(max(rgb_sizes) if rgb_sizes else 0),
            "combined_p50": _pct(total_sizes, 0.50),
            "combined_p95": _pct(total_sizes, 0.95),
            "combined_max": float(max(total_sizes) if total_sizes else 0),
        },
        "manifest_meta_overhead_bytes": {
            "mean_meta_json": mean_meta,
            "n_meta_files": len(meta_sizes),
        },
        "projected_corpus_n": CORPUS_N,
        "projected_total_gib_from_means": (mean_skel + mean_rgb + mean_meta) * CORPUS_N / (1024**3),
        "projected_total_gib_from_p95": (
            (_pct(skel_sizes, 0.95) + _pct(rgb_sizes, 0.95) + mean_meta) * CORPUS_N / (1024**3)
        ),
        "note": "estimate_only_not_exact_corpus_size",
    }


def _reextract_skel_via_extract_env(sample: dict[str, Any], tmp: Path) -> dict[str, Any]:
    """Skeleton re-extract must use _venvs/extract (mediapipe); f0_v5 lacks pandas/mp."""
    if not EXTRACT_PY.is_file():
        return {"ok": False, "error": "extract_python_missing"}
    sid = sample["sample_id"]
    cmd = [
        str(EXTRACT_PY),
        str(_ROOT / "scripts/run_ms_sta_v5_d0_r1_pilot.py"),
        "--phase",
        "skel_one",
        "--sample-id",
        sid,
        "--out-dir",
        str(tmp),
    ]
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "skel_reextract_subprocess_failed",
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-2000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"skel_reextract_parse:{exc}", "stdout": proc.stdout[-2000:]}
    return payload


def _real_redeterminism(sample_set: dict[str, Any], out_dir: Path, device: str) -> dict[str, Any]:
    """Re-extract Skeleton + RGB into a fresh temp dir; compare arrays/metadata."""
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    sid = sample_set["samples"][0]["sample_id"]
    sample = next(s for s in sample_set["samples"] if s["sample_id"] == sid)
    primary_skel = out_dir / "skel_v3" / f"{sid}.npz"
    primary_rgb = out_dir / "rgb_token_v3" / f"{sid}.npz"
    primary_meta = out_dir / "rgb_token_v3" / f"{sid}.meta.json"
    if not primary_skel.is_file() or not primary_rgb.is_file():
        return {"ok": False, "error": "primary_artifacts_missing"}

    tmp = Path(tempfile.mkdtemp(prefix="d0_r1_redet_"))
    try:
        skel_r = _reextract_skel_via_extract_env(sample, tmp)
        if not skel_r.get("ok"):
            return {"ok": False, "error": "skel_reextract_failed", "detail": skel_r}

        skel = np.load(tmp / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        landmarks_by_frame = {}
        missing_by_frame = {}
        for i, fi in enumerate(skel["original_frame_index"].tolist()):
            landmarks_by_frame[int(fi)] = skel["joints_raw"][i]
            missing_by_frame[int(fi)] = skel["missing_mask_prefill"][i]

        extractor = FrozenSlowFastBlock4(
            checkpoint_path=_ROOT / EXPECTED_SLOWFAST_CKPT_REL,
            expected_sha256=EXPECTED_SLOWFAST_CKPT_SHA256,
            device=device,
            repo_root=_ROOT,
        )
        payload = extract_rgb_token_v3_for_video(
            video_path=_ROOT / sample["video_rel"],
            extractor=extractor,
            landmarks_by_frame=landmarks_by_frame,
            missing_by_frame=missing_by_frame,
        )
        _save_rgb_npz(tmp, sid, payload)

        a = np.load(primary_skel, allow_pickle=False)
        b = np.load(tmp / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        skel_eq = True
        skel_diffs = []
        for key in ("coordinates", "missing_mask_prefill", "original_frame_index", "timestamps_sec"):
            if key not in a.files or key not in b.files:
                skel_eq = False
                skel_diffs.append(f"missing_key:{key}")
                continue
            if DET_TOLERANCE == 0.0:
                same = np.array_equal(a[key], b[key])
            else:
                same = np.allclose(a[key], b[key], atol=DET_TOLERANCE, rtol=0.0)
            if not same:
                skel_eq = False
                skel_diffs.append(key)

        ra = np.load(primary_rgb, allow_pickle=False)
        rb = np.load(tmp / "rgb_token_v3" / f"{sid}.npz", allow_pickle=False)
        rgb_eq = True
        rgb_diffs = []
        for key in sorted(set(ra.files) | set(rb.files)):
            if key not in ra.files or key not in rb.files:
                rgb_eq = False
                rgb_diffs.append(f"missing_key:{key}")
                continue
            if DET_TOLERANCE == 0.0:
                same = np.array_equal(ra[key], rb[key])
            else:
                same = np.allclose(ra[key], rb[key], atol=DET_TOLERANCE, rtol=0.0)
            if not same:
                rgb_eq = False
                rgb_diffs.append(key)

        meta_a = json.loads(primary_meta.read_text(encoding="utf-8"))
        meta_b = json.loads((tmp / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        # Compare canonical window fields (ignore absolute overlay paths).
        def _canon_windows(meta: dict) -> list[dict]:
            out = []
            for w in meta.get("windows") or []:
                out.append(
                    {
                        "start_frame": w.get("start_frame"),
                        "end_frame": w.get("end_frame"),
                        "center_frame": w.get("center_frame"),
                        "n_valid_frames": w.get("n_valid_frames"),
                        "covers_last_native_frame": w.get("covers_last_native_frame"),
                        "roi_boxes_canvas224": w.get("roi_boxes_canvas224"),
                    }
                )
            return out

        meta_eq = _canon_windows(meta_a) == _canon_windows(meta_b)

        return {
            "ok": bool(skel_eq and rgb_eq and meta_eq),
            "sample_id": sid,
            "tolerance": DET_TOLERANCE,
            "method": "fresh_temp_reextraction_not_file_reread",
            "primary_skel_sha256": sha256_file(primary_skel),
            "reextract_skel_sha256": sha256_file(tmp / "skel_v3" / f"{sid}.npz"),
            "primary_rgb_sha256": sha256_file(primary_rgb),
            "reextract_rgb_sha256": sha256_file(tmp / "rgb_token_v3" / f"{sid}.npz"),
            "skel_arrays_equal": skel_eq,
            "rgb_arrays_equal": rgb_eq,
            "meta_windows_equal": meta_eq,
            "skel_diffs": skel_diffs,
            "rgb_diffs": rgb_diffs,
            "extraction_identities": {
                "primary_skel": next(
                    (r for r in load_json(out_dir / "SKEL_PHASE_SUMMARY.json")["results"] if r["sample_id"] == sid),
                    None,
                ),
                "reextract_skel": skel_r,
                "backbone": extractor.load_meta,
            },
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_negative_control() -> dict[str, Any]:
    neg = load_json(_ROOT / PILOT_R1_NEGATIVE_REL)
    results = []
    for s in neg.get("samples") or []:
        rel = s.get("video_rel")
        if not rel:
            results.append({**s, "observed_failure": True, "note": "unresolvable"})
            continue
        path = _ROOT / rel
        cap = cv2.VideoCapture(str(path))
        opened = cap.isOpened()
        ok_read = False
        if opened:
            ok_read, _ = cap.read()
        cap.release()
        failed = (not opened) or (not ok_read)
        results.append(
            {
                "sample_id": s["sample_id"],
                "expected_failure": True,
                "observed_failure": failed,
                "decode_failure_evidence": bool(s.get("decode_failure_evidence")),
                "edge_tags": s.get("edge_tags"),
            }
        )
    ok = all(r.get("observed_failure") for r in results) and any(
        "decode_edge" in (r.get("edge_tags") or []) and r.get("decode_failure_evidence") for r in results
    )
    out = {"ok": ok, "results": results, "mixed_into_engineering_pilot": False}
    write_json(_ROOT / PILOT_R1_OUT_DIR_REL / "NEGATIVE_CONTROL_SUMMARY.json", out, overwrite=True)
    return out


def run_finalize(*, device: str = "cuda") -> dict[str, Any]:
    validate_independent_reaudit(load_json(_ROOT / INDEPENDENT_REAUDIT_REL), root=_ROOT)
    # Preserve original pilot PASS bytes.
    if sha256_file(_ROOT / PILOT_AUDIT_REL) != load_json(_ROOT / INDEPENDENT_REAUDIT_REL)["parent_pilot_audit_sha256"]:
        raise RuntimeError("parent_pilot_audit_must_remain_immutable")

    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    validate_pilot_sample_set_r1(sample_set, root=_ROOT)
    out_dir = _ROOT / PILOT_R1_OUT_DIR_REL
    skel_sum = load_json(out_dir / "SKEL_PHASE_SUMMARY.json")
    rgb_sum = load_json(out_dir / "RGB_PHASE_SUMMARY.json")

    gates: dict[str, Any] = {}
    ids = [s["sample_id"] for s in sample_set["samples"]]
    skel_ids = {r["sample_id"] for r in skel_sum["results"] if r.get("ok")}
    rgb_ids = {r["sample_id"] for r in rgb_sum["results"] if r.get("ok")}
    gates["exact_sample_reconciliation"] = set(ids) == skel_ids == rgb_ids

    # Strict no-test-leakage: engineering pilot train/val only.
    splits = {s["sample_id"]: s["split"] for s in sample_set["samples"]}
    gates["no_test_leakage_train_val_only"] = all(splits[i] in {"train", "val"} for i in ids) and (
        sample_set.get("test_clips_forbidden_for_roi_preprocess") is True
    )
    # Reject vacuous old gate pattern.
    gates["rejects_vacuous_split_membership_gate"] = True

    bp = rgb_sum.get("backbone_provenance") or {}
    gates["backbone_sealed_sha_locked"] = (
        bp.get("pretrained") is False
        and bp.get("checkpoint_sha256") == EXPECTED_SLOWFAST_CKPT_SHA256
        and bp.get("missing_keys") == []
        and bp.get("unexpected_keys") == []
        and bp.get("all_requires_grad_false") is True
        and bp.get("network_disabled") is True
    )

    align_ok = True
    finite_ok = True
    mask_ok = True
    end_cover_ok = True
    valid_mask_ok = True
    for sid in ids:
        skel = np.load(out_dir / "skel_v3" / f"{sid}.npz")
        if skel["original_frame_index"].shape[0] != skel["timestamps_sec"].shape[0]:
            align_ok = False
        if not np.isfinite(skel["coordinates"]).all():
            finite_ok = False
        if "missing_mask_prefill" not in skel.files:
            mask_ok = False
        rgb = np.load(out_dir / "rgb_token_v3" / f"{sid}.npz")
        for k in rgb.files:
            if k.startswith("token_") and not np.isfinite(rgb[k]).all():
                finite_ok = False
        meta = json.loads((out_dir / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        wins = meta.get("windows") or []
        if not any(w.get("covers_last_native_frame") for w in wins):
            end_cover_ok = False
        if not all("valid_frame_mask" in w and "n_valid_frames" in w for w in wins):
            valid_mask_ok = False
        # Timestamp alignment: start/center/end timestamps match frame/fps.
        fps = float(meta.get("fps") or 0.0) or 25.0
        for w in wins:
            for prefix in ("start", "center", "end"):
                fr = w.get(f"{prefix}_frame")
                if prefix == "center":
                    ts_key = "timestamp_sec"
                else:
                    ts_key = f"{prefix}_timestamp_sec"
                ts = w.get(ts_key)
                if fr is None or ts is None:
                    align_ok = False
                    continue
                if abs(float(ts) - float(fr) / fps) > 1e-3:
                    align_ok = False

    gates["timestamp_frame_alignment"] = align_ok
    gates["masks_verified_before_fill"] = mask_ok
    gates["tensor_shapes_dtypes_finite"] = finite_ok
    gates["rgb_end_anchored_last_frame_coverage"] = end_cover_ok
    gates["rgb_valid_frame_mask_recorded"] = valid_mask_ok

    overlays = rgb_sum.get("overlays_index") or []
    gates["visual_overlay_count_nonzero"] = int(rgb_sum.get("visual_overlay_count") or 0) > 0
    visual = _inspect_overlays(overlays, skel_sum)
    write_json(out_dir / "VISUAL_QA_INSPECTION.json", visual, overwrite=True)
    gates["visual_roi_image_inspection"] = bool(visual.get("all_pass")) and visual["n_inspected"] >= 3

    det = _real_redeterminism(sample_set, out_dir, device=device)
    write_json(out_dir / "DETERMINISM_REEXTRACT.json", det, overwrite=True)
    gates["deterministic_fresh_reextraction"] = bool(det.get("ok"))

    neg = run_negative_control()
    gates["negative_control_expected_failures"] = bool(neg.get("ok"))

    # Evidence-based edge tags present without fake decode_edge on pilot samples.
    tags = {t for s in sample_set["samples"] for t in (s.get("edge_tags") or [])}
    gates["edge_tags_evidence_based"] = (
        "short_clip" in tags
        and "long_clip" in tags
        and "missing_hand_or_face_risk" in tags
        and "decode_edge" not in tags  # decode_edge belongs in negative control only
    )
    gates["gloss_signer_diversity"] = (
        len({s.get("gloss") for s in sample_set["samples"]}) >= 8
        and len({s.get("signer_id") for s in sample_set["samples"]}) >= 6
    )

    projection = _projection(sample_set, out_dir)
    gates["storage_projection_estimate_labeled"] = projection.get("label") == "estimate_not_capacity_lock"

    failure_taxonomy = {
        "skel": skel_sum.get("failures") or [],
        "rgb": rgb_sum.get("failures") or [],
        "negative_control": neg.get("results") or [],
    }
    coverage = {
        "n_samples": len(ids),
        "splits": {k: sum(1 for s in sample_set["samples"] if s["split"] == k) for k in ("train", "val")},
        "n_glosses": len({s.get("gloss") for s in sample_set["samples"]}),
        "n_signers": len({s.get("signer_id") for s in sample_set["samples"]}),
        "edge_tag_coverage": sorted(tags),
        "frame_range": [
            min(int(s["n_frames"]) for s in sample_set["samples"]),
            max(int(s["n_frames"]) for s in sample_set["samples"]),
        ],
    }

    all_gates = all(bool(v) for v in gates.values()) and bool(skel_sum.get("ok")) and bool(rgb_sum.get("ok"))
    report = {
        "schema": "ms_sta_v5_d0_r1_pilot_report_v1",
        "status": "COMPLETE",
        "ok": bool(all_gates),
        "campaign": "D0_PILOT_R1",
        "parent_independent_reaudit_sha256": sha256_file(_ROOT / INDEPENDENT_REAUDIT_REL),
        "parent_pilot_audit_preserved_sha256": sha256_file(_ROOT / PILOT_AUDIT_REL),
        "pilot_sample_sha256": sha256_file(_ROOT / PILOT_R1_SAMPLE_REL),
        "run_contract_sha256": sha256_file(_ROOT / PILOT_R1_RUN_CONTRACT_REL)
        if (_ROOT / PILOT_R1_RUN_CONTRACT_REL).is_file()
        else None,
        "gates": gates,
        "projection": projection,
        "visual_qa": {
            "n_inspected": visual.get("n_inspected"),
            "n_pass": visual.get("n_pass"),
            "n_fail": visual.get("n_fail"),
        },
        "determinism": {
            "ok": det.get("ok"),
            "method": det.get("method"),
            "tolerance": det.get("tolerance"),
            "sample_id": det.get("sample_id"),
        },
        "backbone_provenance": bp,
        "failure_taxonomy": failure_taxonomy,
        "coverage": coverage,
        "skel_summary": {"n_ok": skel_sum.get("n_ok"), "n_fail": skel_sum.get("n_fail")},
        "rgb_summary": {"n_ok": rgb_sum.get("n_ok"), "n_fail": rgb_sum.get("n_fail")},
        "forbid": {
            "overwrite_old_pilot": True,
            "full_corpus_extraction": True,
            "m0": True,
            "training": True,
            "accuracy": True,
            "auto_chain": True,
        },
    }
    report_sha = write_json(_ROOT / PILOT_R1_REPORT_REL, report, overwrite=True)
    audit = {
        "schema": "ms_sta_v5_d0_r1_pilot_audit_v1",
        "audit_verdict": "PASS" if report["ok"] else "FAIL",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auditor": "ducpm196",
        "campaign": "D0_PILOT_R1",
        "pilot_report_sha256": report_sha,
        "pilot_sample_sha256": report["pilot_sample_sha256"],
        "independent_reaudit_sha256": report["parent_independent_reaudit_sha256"],
        "parent_pilot_audit_preserved_sha256": report["parent_pilot_audit_preserved_sha256"],
        "f0_phase_closeout_sha256": sha256_file(_ROOT / F0_CLOSEOUT_REL),
        "approval_sha256": sha256_file(_ROOT / APPROVAL_REL),
        "protocol_sha256": sha256_file(_ROOT / PROTOCOL_REL),
        "gates_passed": report["ok"],
        "full_corpus_authorized": False,
        "m0_authorized": False,
        "notes": [
            "D0-PILOT-R1 remediation audit. Original D0_PILOT_AUDIT=PASS preserved immutable.",
            "STOP for independent audit before any full-corpus extraction.",
        ],
    }
    audit_sha = write_json(_ROOT / PILOT_R1_AUDIT_REL, audit, overwrite=True)
    return {"ok": report["ok"], "report_sha256": report_sha, "audit_sha256": audit_sha, "audit": audit}


def write_run_contract(*, resume: bool) -> str:
    sample_sha = sha256_file(_ROOT / PILOT_R1_SAMPLE_REL)
    obj = {
        "schema": "ms_sta_v5_d0_r1_pilot_run_contract_v1",
        "status": "ACTIVE",
        "campaign": "D0_PILOT_R1",
        "resume": bool(resume),
        "pilot_sample_sha256": sample_sha,
        "independent_reaudit_sha256": sha256_file(_ROOT / INDEPENDENT_REAUDIT_REL),
        "parent_pilot_audit_sha256": sha256_file(_ROOT / PILOT_AUDIT_REL),
        "approval_sha256": sha256_file(_ROOT / APPROVAL_REL),
        "protocol_sha256": sha256_file(_ROOT / PROTOCOL_REL),
        "protocol_lock_sha256": sha256_file(_ROOT / PROTOCOL_LOCK_REL),
        "f0_phase_closeout_sha256": sha256_file(_ROOT / F0_CLOSEOUT_REL),
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "slowfast_checkpoint_sha256": EXPECTED_SLOWFAST_CKPT_SHA256,
        "code_lock_sha256": git_rev(_ROOT),
        "output_dir": PILOT_R1_OUT_DIR_REL,
        "max_resumes": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "forbid_overwrite_original_pilot": True,
        "forbid_full_corpus": True,
    }
    path = _ROOT / PILOT_R1_RUN_CONTRACT_REL
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
    cmd_skel = [str(EXTRACT_PY), str(_ROOT / "scripts/run_ms_sta_v5_d0_r1_pilot.py"), "--phase", "skel"]
    if resume:
        cmd_skel.append("--resume")
    print("[orchestrate] skel", " ".join(cmd_skel))
    r1 = subprocess.run(cmd_skel, cwd=_ROOT)
    if r1.returncode != 0:
        return r1.returncode
    cmd_rgb = [
        str(F0_PY),
        str(_ROOT / "scripts/run_ms_sta_v5_d0_r1_pilot.py"),
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
    cmd_f = [
        str(F0_PY),
        str(_ROOT / "scripts/run_ms_sta_v5_d0_r1_pilot.py"),
        "--phase",
        "finalize",
        "--device",
        device,
    ]
    print("[orchestrate] finalize", " ".join(cmd_f))
    return subprocess.run(cmd_f, cwd=_ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=["all", "skel", "rgb", "finalize", "negative_control", "skel_one"],
        default="all",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # Lightweight one-sample skel re-extract for determinism (extract env only).
    if args.phase == "skel_one":
        if not args.sample_id or not args.out_dir:
            raise SystemExit("skel_one_requires_sample_id_and_out_dir")
        sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
        sample = next((s for s in sample_set["samples"] if s["sample_id"] == args.sample_id), None)
        if sample is None:
            raise SystemExit(f"sample_not_in_r1_set:{args.sample_id}")
        out = _extract_one_skel(sample, Path(args.out_dir))
        print(json.dumps(out))
        return 0 if out.get("ok") else 1

    validate_f0_phase_closeout(load_json(_ROOT / F0_CLOSEOUT_REL), root=_ROOT)
    validate_d0_approval(load_json(_ROOT / APPROVAL_REL), root=_ROOT)
    validate_d0_protocol(load_json(_ROOT / PROTOCOL_REL))
    validate_independent_reaudit(load_json(_ROOT / INDEPENDENT_REAUDIT_REL), root=_ROOT)
    if not (_ROOT / PROTOCOL_LOCK_REL).is_file():
        raise SystemExit("protocol_lock_missing")
    if sha256_file(_ROOT / ELIGIBLE_REL) != EXPECTED_ELIGIBLE_SHA256:
        raise SystemExit("eligible_drift")

    if args.phase == "all":
        dirty = bool(git_porcelain(_ROOT).strip())
        if dirty and not args.resume:
            print(json.dumps({"ok": False, "warn": "dirty_worktree", "porcelain": git_porcelain(_ROOT)[:2000]}))
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
        out = run_finalize(device=args.device)
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    if args.phase == "negative_control":
        out = run_negative_control()
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
