#!/usr/bin/env python3
"""Standalone C3 skeleton parity verifier for D0 full-corpus (no pytest/torch import).

Runnable as:
  _venvs/extract/Scripts/python.exe scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py
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

REPORT_SCHEMA = "ms_sta_v5_d0_full_corpus_c3_parity_report_v1"
PARITY_KEYS = (
    "coordinates",
    "missing_mask_prefill",
    "original_frame_index",
    "effective_landmarks",
    "raw_landmarks",
    "effective_missing",
    "raw_holistic_missing",
    "observation_source_left_hand",
    "observation_source_right_hand",
)
OBS_SOURCE_MAP = {"missing": 0, "holistic": 1, "hands_fallback": 2}


def _encode_observation_source(values: list[str]) -> np.ndarray:
    return np.asarray([OBS_SOURCE_MAP[str(x)] for x in values], dtype=np.uint8)


def verify_c3_parity(*, root: Path, sample_ids: list[str] | None = None) -> dict[str, Any]:
    from ms_sta_v5_d0_lib import (
        PILOT_R1_SAMPLE_REL,
        PILOT_R7C_OUT_DIR_REL,
        git_rev,
        load_json,
        sha256_file,
    )
    from src.data_prep.hand_recovery_v6 import HandRecoveryEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    sample_set = load_json(root / PILOT_R1_SAMPLE_REL)
    samples = sample_set["samples"]
    if sample_ids is not None:
        wanted = set(sample_ids)
        samples = [s for s in samples if s["sample_id"] in wanted]
    mismatches: list[str] = []
    checked = 0
    skipped = 0
    for s in samples:
        sid = s["sample_id"]
        frozen = root / PILOT_R7C_OUT_DIR_REL / "skel_v3" / f"{sid}.npz"
        video = root / s["video_rel"]
        if not frozen.is_file():
            skipped += 1
            mismatches.append(f"{sid}:frozen_missing")
            continue
        if not video.is_file():
            skipped += 1
            mismatches.append(f"{sid}:video_missing")
            continue
        eng = HandRecoveryEngine(arm="C3", model_complexity=0, store_stride=2)
        try:
            rec = eng.process_video(video)
        finally:
            eng.close()
        payload = build_skel_v3_arrays(
            joints_raw=rec["effective_landmarks"],
            raw_pose=rec["raw_pose"],
            missing_prefill=rec["effective_missing"],
            original_frame_index=rec["original_frame_index"],
            timestamps_sec=rec["timestamps_sec"],
        )
        ref = np.load(frozen, allow_pickle=False)
        checked += 1
        got_map = {
            "coordinates": payload["coordinates"],
            "missing_mask_prefill": payload["missing_mask_prefill"],
            "original_frame_index": payload["original_frame_index"],
            "effective_landmarks": rec["effective_landmarks"],
            "raw_landmarks": rec["raw_landmarks"],
            "effective_missing": rec["effective_missing"],
            "raw_holistic_missing": rec["raw_holistic_missing"],
            "observation_source_left_hand": _encode_observation_source(rec["observation_source_left_hand"]),
            "observation_source_right_hand": _encode_observation_source(rec["observation_source_right_hand"]),
        }
        for key in PARITY_KEYS:
            if key not in ref.files:
                mismatches.append(f"{sid}:missing_key:{key}")
                continue
            if not np.array_equal(got_map[key], ref[key]):
                mismatches.append(f"{sid}:array_mismatch:{key}")
    ok = checked > 0 and not mismatches
    return {
        "schema": REPORT_SCHEMA,
        "ok": ok,
        "checked": checked,
        "skipped": skipped,
        "n_samples": len(samples),
        "mismatches": mismatches,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "skeleton_arm": "C3",
        "code_lock_git_revision": git_rev(root),
        "sample_set_sha256": sha256_file(root / PILOT_R1_SAMPLE_REL),
        "frozen_ref_dir_rel": PILOT_R7C_OUT_DIR_REL,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    ap.add_argument("--sample-id", action="append", default=None, dest="sample_ids")
    args = ap.parse_args()
    report = verify_c3_parity(root=args.root, sample_ids=args.sample_ids)
    raw = json.dumps(report, indent=2)
    print(raw)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(raw + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
