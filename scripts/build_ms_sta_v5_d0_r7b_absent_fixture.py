#!/usr/bin/env python3
"""Build empirical truly-absent wrist fixture (blank video + all-missing skel)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

from ms_sta_v5_d0_lib import PILOT_R7B_FIXTURE_ABSENT_REL, sha256_file, write_json  # noqa: E402
from src.models.rgb_token_v3 import region_boxes_with_rgb_sources  # noqa: E402

N_JOINTS = 75
T = 32
H = W = 64


def main() -> int:
    out = _ROOT / PILOT_R7B_FIXTURE_ABSENT_REL
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / "absent_wrist.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 25.0, (W, H))
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for _ in range(T):
        vw.write(frame)
    vw.release()

    joints = np.zeros((T, N_JOINTS, 2), dtype=np.float32)
    missing = np.ones((T, N_JOINTS), dtype=bool)
    np.savez_compressed(
        out / "absent_wrist_skel.npz",
        joints_raw=joints,
        missing_mask_prefill=missing,
        original_frame_index=np.arange(T, dtype=np.int32),
        timestamps_sec=np.arange(T, dtype=np.float64) / 25.0,
    )

    boxes, sources, reasons = region_boxes_with_rgb_sources(
        joints[0], missing[0], enable_pose_wrist_prior=True
    )
    proof = {
        "schema": "ms_sta_v5_d0_r7b_truly_absent_wrist_fixture_v1",
        "sample_id": "NEG_ABSENT_WRIST",
        "video_rel": f"{PILOT_R7B_FIXTURE_ABSENT_REL}/absent_wrist.mp4",
        "skel_rel": f"{PILOT_R7B_FIXTURE_ABSENT_REL}/absent_wrist_skel.npz",
        "video_sha256": sha256_file(video_path),
        "skel_sha256": sha256_file(out / "absent_wrist_skel.npz"),
        "expected": {
            "left_hand_box": None,
            "right_hand_box": None,
            "left_hand_source": "missing",
            "right_hand_source": "missing",
            "left_hand_reason": "pose_wrist_missing",
            "right_hand_reason": "pose_wrist_missing",
        },
        "observed": {
            "left_hand_box": None if boxes["left_hand"] is None else boxes["left_hand"].tolist(),
            "right_hand_box": None if boxes["right_hand"] is None else boxes["right_hand"].tolist(),
            "left_hand_source": sources["left_hand"],
            "right_hand_source": sources["right_hand"],
            "left_hand_reason": reasons["left_hand"],
            "right_hand_reason": reasons["right_hand"],
        },
    }
    if (
        proof["observed"]["left_hand_box"] is not None
        or proof["observed"]["right_hand_box"] is not None
        or proof["observed"]["left_hand_source"] != "missing"
        or proof["observed"]["right_hand_source"] != "missing"
    ):
        raise RuntimeError("absent_fixture_pose_prior_must_not_emit_box")
    write_json(out / "PROOF.json", proof, overwrite=True)
    print(json.dumps({"ok": True, "fixture": PILOT_R7B_FIXTURE_ABSENT_REL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
