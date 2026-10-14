#!/usr/bin/env python3
"""MS-STA v5 D0-PILOT-R5 QA remediation — stratified freeze + contact identity.

Reuses sealed R4 tensors (no re-extract). Regenerates visual overlays only for the
frozen QA windows with per-tile identity binding. Zero-FAIL manual acceptance.
"""

from __future__ import annotations

import argparse
import json
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
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R5_AMENDMENT_REL,
    PILOT_R1_SAMPLE_REL,
    PILOT_R4_AUDIT_REL,
    PILOT_R4_INDEPENDENT_REAUDIT_REL,
    PILOT_R4_OUT_DIR_REL,
    PILOT_R5_AUDIT_REL,
    PILOT_R5_COMPLETION_REL,
    PILOT_R5_OVERLAYS_INDEX_REL,
    PILOT_R5_REPORT_REL,
    PILOT_R5_RUN_CONTRACT_REL,
    PILOT_R5_VISUAL_FREEZE_REL,
    PILOT_R5_VISUAL_MANUAL_REL,
    PILOT_R5_VISUAL_REL,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    validate_r4_independent_reaudit,
    validate_r5_qa_amendment,
    validate_r5_visual_freeze_list,
    validate_visual_qa_manual,
    write_json,
)

REGIONS_LOCAL = ("face", "left_hand", "right_hand")


def _code_lock_bind() -> dict[str, Any]:
    allowed = ("Docs/results/ms_sta_v5_d0_r5/", "Docs/results/ms_sta_v5_d0_r5")
    dirty_lines = []
    for line in git_porcelain(_ROOT).splitlines():
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        norm = path.replace("\\", "/")
        if any(norm == p.rstrip("/") or norm.startswith(p) for p in allowed):
            continue
        dirty_lines.append(line)
    dirty = "\n".join(dirty_lines).strip()
    return {
        "code_lock_sha256": git_rev(_ROOT),
        "worktree_dirty": bool(dirty),
        "porcelain": dirty[:2000] if dirty else "",
    }


def _require_clean(expected: str | None = None) -> dict[str, Any]:
    bind = _code_lock_bind()
    if bind["worktree_dirty"]:
        raise RuntimeError(f"dirty_worktree_forbidden:{bind['porcelain'][:500]}")
    if expected is not None and bind["code_lock_sha256"] != expected:
        raise RuntimeError(f"code_lock_mismatch:{expected}:{bind['code_lock_sha256']}")
    return bind


def _enumerate_windows(sample_set: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = _ROOT / PILOT_R4_OUT_DIR_REL
    meta_by = {s["sample_id"]: s for s in sample_set["samples"]}
    rows: list[dict[str, Any]] = []
    for sid, s in meta_by.items():
        meta = json.loads((out_dir / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        for wi, w in enumerate(meta["windows"]):
            rv = w["region_validity"]
            vfs = {r: float(rv[r]["valid_fraction"]) for r in REGIONS_LOCAL}
            min_vf = min(vfs.values())
            rows.append(
                {
                    "sample_id": sid,
                    "split": s["split"],
                    "signer_id": s["signer_id"],
                    "window_index": wi,
                    "window_start_idx": int(w["start_idx"]),
                    "center_frame": int(w["center_frame"]),
                    "start_frame": int(w["start_frame"]),
                    "valid_fraction_by_region": vfs,
                    "min_valid_fraction": min_vf,
                    "partial_missing": any(bool(rv[r]["partial_missing"]) for r in REGIONS_LOCAL),
                }
            )
    return rows


def build_freeze_list(*, overwrite: bool = False) -> dict[str, Any]:
    """Freeze stratified QA windows from R4 metas BEFORE any image review."""
    path = _ROOT / PILOT_R5_VISUAL_FREEZE_REL
    if path.is_file() and not overwrite:
        obj = load_json(path)
        validate_r5_visual_freeze_list(obj)
        return obj

    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    rows = _enumerate_windows(sample_set)
    by_key = {(r["sample_id"], r["window_start_idx"]): r for r in rows}

    def pick_one(cands: list[dict[str, Any]], *, prefer_split: str | None = None) -> dict[str, Any]:
        if prefer_split:
            pref = [c for c in cands if c["split"] == prefer_split]
            if pref:
                cands = pref
        if not cands:
            raise RuntimeError("freeze_no_candidates")
        return sorted(cands, key=lambda c: (c["sample_id"], c["window_start_idx"]))[0]

    full = [r for r in rows if r["min_valid_fraction"] >= 1.0 - 1e-12]
    mid = [r for r in rows if 0.5 <= r["min_valid_fraction"] < 1.0 - 1e-12]
    low = [r for r in rows if 0.0 < r["min_valid_fraction"] < 0.5]
    worst_face = sorted(rows, key=lambda r: r["valid_fraction_by_region"]["face"])[0]
    worst_lh = sorted(rows, key=lambda r: r["valid_fraction_by_region"]["left_hand"])[0]
    worst_rh = sorted(rows, key=lambda r: r["valid_fraction_by_region"]["right_hand"])[0]
    mandatory = by_key[("015417", 16)]
    if abs(mandatory["valid_fraction_by_region"]["face"] - 0.15) > 1e-9:
        raise RuntimeError("mandatory_015417_w0016_face_vf_not_0_15")

    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any], stratum: str) -> None:
        item = {
            **row,
            "stratum": stratum,
            "qa_kinds_required": ["center_overlay", "temporal_contact_sheet"],
        }
        # de-dup by sample/window but keep first stratum label in notes
        key = (item["sample_id"], item["window_start_idx"])
        if any((x["sample_id"], x["window_start_idx"]) == key for x in selected):
            for x in selected:
                if (x["sample_id"], x["window_start_idx"]) == key:
                    x.setdefault("also_strata", []).append(stratum)
            return
        selected.append(item)

    add(pick_one(full, prefer_split="train"), "full_valid_control_vf_eq_1")
    add(pick_one(mid, prefer_split="train"), "partial_mid_0_5_le_vf_lt_1")
    add(pick_one(low, prefer_split="train"), "partial_low_0_lt_vf_lt_0_5")
    add(worst_face, "worst_face")
    add(worst_lh, "worst_left_hand")
    add(worst_rh, "worst_right_hand")
    add(mandatory, "mandatory_015417_w0016_face_vf_0_15")
    # Ensure val coverage
    val_rows = [r for r in rows if r["split"] == "val"]
    if not any(x["split"] == "val" for x in selected):
        add(pick_one(val_rows), "val_split")
    else:
        # mark existing val window
        for x in selected:
            if x["split"] == "val":
                x.setdefault("also_strata", []).append("val_split")
                break
    if not any(x["split"] == "train" for x in selected):
        raise RuntimeError("freeze_missing_train")
    for x in selected:
        if x["split"] == "train":
            x.setdefault("also_strata", []).append("train_split")
            break

    obj = {
        "schema": "ms_sta_v5_d0_r5_visual_qa_freeze_list_v1",
        "frozen_before_review": True,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_r4_outputs": PILOT_R4_OUT_DIR_REL,
        "selection_rule": "objective_from_region_validity_before_image_review",
        "n_windows": len(selected),
        "windows": selected,
        "strata_covered": sorted(
            {
                *(w["stratum"] for w in selected),
                *(s for w in selected for s in (w.get("also_strata") or [])),
            }
        ),
        "notes": [
            "Frozen prior to manual image inspection.",
            "Must include 015417/w0016 (face vf=0.15) and stratified partial/full/worst/train/val.",
        ],
    }
    validate_r5_visual_freeze_list(obj)
    write_json(path, obj, overwrite=True)
    return obj


def _sample_rows(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) <= k:
        return rows
    idxs = np.linspace(0, len(rows) - 1, num=k).astype(int).tolist()
    return [rows[i] for i in idxs]


def regen_visuals_for_freeze() -> dict[str, Any]:
    """Regenerate center + contact overlays with per-tile identity; do not touch R4 npz."""
    from src.models.rgb_token_v3 import (
        INPUT_SIZE,
        REGIONS,
        _boxes_for_native_frame,
        _local_frame_for_feat_t,
        _resize_letterbox,
        build_temporal_contact_sheet,
        draw_roi_overlay,
        nearest_skel_landmark,
        read_video_frames_native,
        region_boxes_from_landmarks,
        landmark_xy_to_resized,
        WINDOW_SIZE,
    )
    freeze = load_json(_ROOT / PILOT_R5_VISUAL_FREEZE_REL)
    validate_r5_visual_freeze_list(freeze)
    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    samples = {s["sample_id"]: s for s in sample_set["samples"]}
    out_dir = _ROOT / PILOT_R4_OUT_DIR_REL
    visual_dir = _ROOT / PILOT_R5_VISUAL_REL
    visual_dir.mkdir(parents=True, exist_ok=True)

    overlays_index: list[dict[str, Any]] = []
    for win in freeze["windows"]:
        sid = win["sample_id"]
        sample = samples[sid]
        skel = np.load(out_dir / "skel_v3" / f"{sid}.npz", allow_pickle=False)
        meta = json.loads((out_dir / "rgb_token_v3" / f"{sid}.meta.json").read_text(encoding="utf-8"))
        w = meta["windows"][int(win["window_index"])]
        if int(w["start_idx"]) != int(win["window_start_idx"]):
            raise RuntimeError("freeze_window_index_drift")

        # Seal: R4 token bytes must remain unchanged.
        tok_path = out_dir / "rgb_token_v3" / f"{sid}.npz"
        tok_sha = sha256_file(tok_path)

        landmarks_by_frame: dict[int, np.ndarray] = {}
        missing_by_frame: dict[int, np.ndarray] = {}
        raw = skel["joints_raw"]
        miss = skel["missing_mask_prefill"]
        for i, fi in enumerate(skel["original_frame_index"].tolist()):
            landmarks_by_frame[int(fi)] = raw[i]
            missing_by_frame[int(fi)] = miss[i]

        frames_native, frame_idx, fps = read_video_frames_native(_ROOT / sample["video_rel"])
        resized = []
        metas = []
        for fr in frames_native:
            canvas, m = _resize_letterbox(fr, INPUT_SIZE)
            resized.append(canvas)
            metas.append(m)
        frames_u8 = np.stack(resized, axis=0)
        start = int(w["start_idx"])
        end = int(w["end_idx"]) + 1
        center = int(w["center_idx"])
        n_valid = int(w["n_valid_frames"])
        window_size = int(meta.get("window_size") or WINDOW_SIZE)
        shapes = w["feature_map_shapes"]
        t_slow = int(shapes["slow"][2])
        t_fast = int(shapes["fast"][2])

        # Center overlay
        align = nearest_skel_landmark(
            center_frame=int(frame_idx[center]),
            fps=fps,
            skel_frame_index=skel["original_frame_index"],
            skel_timestamps=skel["timestamps_sec"],
        )
        lm = landmarks_by_frame[int(align["landmark_frame"])]
        mm = missing_by_frame[int(align["landmark_frame"])]
        xy = landmark_xy_to_resized(lm[:, :2], metas[center])
        boxes_center = region_boxes_from_landmarks(xy, mm)
        bgr = cv2.cvtColor(frames_u8[center], cv2.COLOR_RGB2BGR)
        vis = draw_roi_overlay(bgr, boxes_center)
        sid_dir = visual_dir / sid
        sid_dir.mkdir(parents=True, exist_ok=True)
        center_path = sid_dir / f"{sid}_w{start:04d}_c{center:04d}.png"
        cv2.imwrite(str(center_path), vis)

        # Rebuild temporal tiles with boxes
        contact_rows: list[dict[str, Any]] = []
        for pathway, t_feat in (("slow", t_slow), ("fast", t_fast)):
            for t_idx in range(t_feat):
                local = _local_frame_for_feat_t(
                    pathway=pathway, t_idx=t_idx, t_feat=t_feat, n_valid=n_valid, window_size=window_size
                )
                native = int(frame_idx[min(start + local, frames_u8.shape[0] - 1)])
                boxes_t, align_t, _nr, _sources, _pr = _boxes_for_native_frame(
                    native_frame=native,
                    canvas_meta=metas[min(start + local, frames_u8.shape[0] - 1)],
                    landmarks_by_frame=landmarks_by_frame,
                    missing_by_frame=missing_by_frame,
                    skel_frame_index=skel["original_frame_index"],
                    skel_timestamps=skel["timestamps_sec"],
                    fps=fps,
                )
                region_missing = {r: boxes_t[r] is None for r in REGIONS}
                contact_rows.append(
                    {
                        "pathway": pathway,
                        "t_idx": t_idx,
                        "local_frame": local,
                        "native_frame": native,
                        "landmark_frame": int(align_t["landmark_frame"]),
                        "boxes": {
                            k: (None if v is None else [float(x) for x in v.tolist()])
                            for k, v in boxes_t.items()
                        },
                        "region_missing": region_missing,
                    }
                )
        slow_rows = [r for r in contact_rows if r["pathway"] == "slow"]
        fast_rows = [r for r in contact_rows if r["pathway"] == "fast"]
        sheet_rows = _sample_rows(slow_rows, 4) + _sample_rows(fast_rows, 4)
        tiles = [
            {
                "pathway": r["pathway"],
                "t_idx": int(r["t_idx"]),
                "local_frame": int(r["local_frame"]),
                "native_frame": int(r["native_frame"]),
                "landmark_frame": int(r["landmark_frame"]),
                "boxes": r["boxes"],
                "region_missing": r["region_missing"],
            }
            for r in sheet_rows
        ]
        sheet = build_temporal_contact_sheet(
            frames_u8=frames_u8[start:end],
            slice_overlays=sheet_rows,
            cols=4,
        )
        sheet_path = sid_dir / f"{sid}_w{start:04d}_temporal_contact.png"
        cv2.imwrite(str(sheet_path), sheet)

        missing_regions = {r: boxes_center[r] is None for r in REGIONS_LOCAL}
        center_boxes = {
            k: (None if v is None else [float(x) for x in v.tolist()]) for k, v in boxes_center.items()
        }
        overlays_index.append(
            {
                "sample_id": sid,
                "kind": "center_overlay",
                "path": str(center_path).replace("\\", "/"),
                "sha256": sha256_file(center_path),
                "center_frame": int(frame_idx[center]),
                "native_frame": int(frame_idx[center]),
                "landmark_frame": int(align["landmark_frame"]),
                "window_start_idx": start,
                "window_index": int(win["window_index"]),
                "min_valid_fraction": float(win["min_valid_fraction"]),
                "valid_fraction_by_region": win["valid_fraction_by_region"],
                "stratum": win["stratum"],
                "split": win["split"],
                "boxes": center_boxes,
                "missing_regions": missing_regions,
                "parent_r4_token_sha256": tok_sha,
            }
        )
        overlays_index.append(
            {
                "sample_id": sid,
                "kind": "temporal_contact_sheet",
                "path": str(sheet_path).replace("\\", "/"),
                "sha256": sha256_file(sheet_path),
                "center_frame": int(frame_idx[center]),
                "native_frame": int(frame_idx[center]),
                "landmark_frame": int(align["landmark_frame"]),
                "window_start_idx": start,
                "window_index": int(win["window_index"]),
                "min_valid_fraction": float(win["min_valid_fraction"]),
                "valid_fraction_by_region": win["valid_fraction_by_region"],
                "stratum": win["stratum"],
                "split": win["split"],
                "boxes": center_boxes,
                "missing_regions": missing_regions,
                "n_tiles": len(tiles),
                "slice_native_frames": [int(t["native_frame"]) for t in tiles],
                "tiles": tiles,
                "parent_r4_token_sha256": tok_sha,
            }
        )

    summary = {
        "phase": "regen_visual_qa",
        "campaign": "D0_PILOT_R5",
        "ok": True,
        "n_overlays": len(overlays_index),
        "n_windows": len(freeze["windows"]),
        "reextract_tensors": False,
        "overlays_index": overlays_index,
    }
    write_json(_ROOT / PILOT_R5_OVERLAYS_INDEX_REL, summary, overwrite=True)
    return summary


def write_run_contract() -> str:
    bind = _require_clean()
    validate_r4_independent_reaudit(load_json(_ROOT / PILOT_R4_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r5_qa_amendment(load_json(_ROOT / FULL_PROTOCOL_R5_AMENDMENT_REL), root=_ROOT)
    if sha256_file(_ROOT / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        raise RuntimeError("canonical_protocol_must_remain_r4_bytes")
    obj = {
        "schema": "ms_sta_v5_d0_r5_pilot_run_contract_v1",
        "status": "ACTIVE",
        "campaign": "D0_PILOT_R5",
        "scope": "QA_REMEDIATION_ONLY",
        "reextract_forbidden": True,
        "parent_r4_outputs": PILOT_R4_OUT_DIR_REL,
        "r4_pilot_audit_sha256": EXPECTED_R4_PILOT_AUDIT_SHA256,
        "r4_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R4_INDEPENDENT_REAUDIT_REL),
        "r5_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R5_AMENDMENT_REL),
        "canonical_full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "pilot_sample_sha256": EXPECTED_R1_SAMPLE_SHA256,
        "code_lock_sha256": bind["code_lock_sha256"],
        "worktree_dirty": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "forbid_full_corpus": True,
        "forbid_push": True,
    }
    path = _ROOT / PILOT_R5_RUN_CONTRACT_REL
    if path.is_file():
        prev = load_json(path)
        if prev.get("status") == "CLOSED":
            raise RuntimeError("r5_run_contract_already_closed")
        return sha256_file(path)
    return write_json(path, obj, overwrite=False)


def run_finalize() -> dict[str, Any]:
    validate_r4_independent_reaudit(load_json(_ROOT / PILOT_R4_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r5_qa_amendment(load_json(_ROOT / FULL_PROTOCOL_R5_AMENDMENT_REL), root=_ROOT)
    contract = load_json(_ROOT / PILOT_R5_RUN_CONTRACT_REL)
    bind = _require_clean(expected=contract.get("code_lock_sha256"))
    freeze = load_json(_ROOT / PILOT_R5_VISUAL_FREEZE_REL)
    validate_r5_visual_freeze_list(freeze)
    overlays_doc = load_json(_ROOT / PILOT_R5_OVERLAYS_INDEX_REL)
    overlays = overlays_doc["overlays_index"]
    manual_path = _ROOT / PILOT_R5_VISUAL_MANUAL_REL
    if not manual_path.is_file():
        raise RuntimeError(f"missing_manual:{PILOT_R5_VISUAL_MANUAL_REL}")
    manual = load_json(manual_path)
    validate_visual_qa_manual(
        manual,
        overlays_index=overlays,
        require_zero_fail=True,
        require_temporal_contact=True,
        require_r5_identity=True,
    )

    # Prove R4 tensors untouched for freeze samples.
    token_seal_ok = True
    for o in overlays:
        sid = o["sample_id"]
        p = _ROOT / PILOT_R4_OUT_DIR_REL / "rgb_token_v3" / f"{sid}.npz"
        if sha256_file(p) != o.get("parent_r4_token_sha256"):
            token_seal_ok = False

    gates = {
        "r4_pilot_audit_preserved": sha256_file(_ROOT / PILOT_R4_AUDIT_REL) == EXPECTED_R4_PILOT_AUDIT_SHA256,
        "r4_independent_fail_bound": True,
        "canonical_protocol_immutable_at_r4_sha": sha256_file(_ROOT / FULL_PROTOCOL_REL)
        == EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "r5_amendment_valid": True,
        "freeze_list_valid": True,
        "r4_tokens_untouched": token_seal_ok,
        "visual_qa_manual_r5_identity": True,
        "visual_qa_manual_zero_fail": True,
        "single_clean_code_lock": bind["worktree_dirty"] is False,
    }
    ok = all(bool(v) for v in gates.values())
    report = {
        "schema": "ms_sta_v5_d0_r5_pilot_report_v1",
        "status": "COMPLETE",
        "ok": ok,
        "campaign": "D0_PILOT_R5",
        "scope": "QA_REMEDIATION_ONLY",
        "gates": gates,
        "freeze_list_sha256": sha256_file(_ROOT / PILOT_R5_VISUAL_FREEZE_REL),
        "overlays_index_sha256": sha256_file(_ROOT / PILOT_R5_OVERLAYS_INDEX_REL),
        "visual_qa_manual_sha256": sha256_file(manual_path),
        "parent_r4_pilot_audit_sha256": EXPECTED_R4_PILOT_AUDIT_SHA256,
        "parent_r4_independent_reaudit_sha256": sha256_file(_ROOT / PILOT_R4_INDEPENDENT_REAUDIT_REL),
        "r5_amendment_sha256": sha256_file(_ROOT / FULL_PROTOCOL_R5_AMENDMENT_REL),
        "canonical_full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "code_lock_sha256": bind["code_lock_sha256"],
        "n_freeze_windows": len(freeze["windows"]),
        "n_manual_inspections": len(manual.get("inspections") or []),
        "forbid": {
            "full_corpus": True,
            "m0": True,
            "train": True,
            "push": True,
            "overwrite_r4_evidence": True,
        },
    }
    report_sha = write_json(_ROOT / PILOT_R5_REPORT_REL, report, overwrite=True)
    audit = {
        "schema": "ms_sta_v5_d0_r5_pilot_audit_v1",
        "audit_verdict": "PASS" if ok else "FAIL",
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auditor": "ducpm196",
        "campaign": "D0_PILOT_R5",
        "pilot_report_sha256": report_sha,
        "code_lock_sha256": bind["code_lock_sha256"],
        "gates_passed": ok,
        "full_corpus_authorized": False,
        "notes": [
            "R5 QA remediation only. R4 PASS audit preserved; R4 independent FAIL bound.",
            "Canonical D0_FULL_PROTOCOL.json left at R4 bytes; R5 uses amendment companion.",
            "STOP for independent audit before push/full-corpus authorization.",
        ],
    }
    audit_sha = write_json(_ROOT / PILOT_R5_AUDIT_REL, audit, overwrite=True)
    contract["status"] = "CLOSED"
    contract["closed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    contract["closed_by"] = PILOT_R5_COMPLETION_REL
    write_json(_ROOT / PILOT_R5_RUN_CONTRACT_REL, contract, overwrite=True)
    completion = {
        "schema": "ms_sta_v5_d0_r5_pilot_completion_v1",
        "status": "CLOSED",
        "closed_at": contract["closed_at"],
        "campaign": "D0_PILOT_R5",
        "pilot_report_sha256": report_sha,
        "pilot_audit_sha256": audit_sha,
        "pilot_audit_verdict": audit["audit_verdict"],
        "immutable": True,
        "full_corpus_authorized": False,
        "next": "INDEPENDENT_R5_AUDIT_REQUIRED_BEFORE_PUSH_OR_FULL_CORPUS",
    }
    completion_sha = write_json(_ROOT / PILOT_R5_COMPLETION_REL, completion, overwrite=True)
    return {
        "ok": ok,
        "report_sha256": report_sha,
        "audit_sha256": audit_sha,
        "completion_sha256": completion_sha,
        "audit": audit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase",
        choices=["freeze", "contract", "regen_visual", "finalize", "all"],
        default="all",
    )
    ap.add_argument("--overwrite-freeze", action="store_true")
    args = ap.parse_args()

    validate_r4_independent_reaudit(load_json(_ROOT / PILOT_R4_INDEPENDENT_REAUDIT_REL), root=_ROOT)
    validate_r5_qa_amendment(load_json(_ROOT / FULL_PROTOCOL_R5_AMENDMENT_REL), root=_ROOT)

    if args.phase in {"freeze", "all"}:
        out = build_freeze_list(overwrite=args.overwrite_freeze)
        print(json.dumps({"ok": True, "n_windows": out["n_windows"]}, indent=2))
        if args.phase == "freeze":
            return 0
    if args.phase in {"contract", "all"}:
        print(write_run_contract())
        if args.phase == "contract":
            return 0
    if args.phase in {"regen_visual", "all"}:
        out = regen_visuals_for_freeze()
        print(json.dumps({"ok": out["ok"], "n_overlays": out["n_overlays"]}, indent=2))
        if args.phase == "regen_visual":
            return 0
    if args.phase in {"finalize", "all"}:
        if args.phase == "all":
            print(
                f"[orchestrate] STOP_BEFORE_FINALIZE: write {PILOT_R5_VISUAL_MANUAL_REL} then --phase finalize"
            )
            return 2
        out = run_finalize()
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
