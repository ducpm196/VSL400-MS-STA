#!/usr/bin/env python3
"""Build versioned R7C truly-absent wrist fixture (required digests + ambiguity recompute)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

from ms_sta_v5_d0_lib import (  # noqa: E402
    PILOT_R7B_FIXTURE_ABSENT_REL,
    PILOT_R7C_FIXTURE_ABSENT_REL,
    R7C_AMBIGUITY_ROI_LOCK,
    sha256_file,
    validate_r7c_absent_wrist_fixture_proof,
    write_json,
)
from src.models.rgb_token_v3 import region_boxes_with_rgb_sources  # noqa: E402


def main() -> int:
    src = _ROOT / PILOT_R7B_FIXTURE_ABSENT_REL
    out = _ROOT / PILOT_R7C_FIXTURE_ABSENT_REL
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / "absent_wrist.mp4"
    skel_path = out / "absent_wrist_skel.npz"
    shutil.copy2(src / "absent_wrist.mp4", video_path)
    shutil.copy2(src / "absent_wrist_skel.npz", skel_path)

    import numpy as np

    data = np.load(skel_path)
    boxes, sources, reasons = region_boxes_with_rgb_sources(
        data["joints_raw"][0],
        data["missing_mask_prefill"][0],
        enable_pose_wrist_prior=False,
        enable_ambiguity_aware_prior=True,
        ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
    )
    proof = {
        "schema": "ms_sta_v5_d0_r7c_truly_absent_wrist_fixture_v1",
        "sample_id": "NEG_ABSENT_WRIST_R7C",
        "video_rel": f"{PILOT_R7C_FIXTURE_ABSENT_REL}/absent_wrist.mp4",
        "skel_rel": f"{PILOT_R7C_FIXTURE_ABSENT_REL}/absent_wrist_skel.npz",
        "video_sha256": sha256_file(video_path),
        "skel_sha256": sha256_file(skel_path),
        "ambiguity_aware_prior": True,
        "ambiguity_lock": dict(R7C_AMBIGUITY_ROI_LOCK),
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
        raise RuntimeError("r7c_absent_fixture_ambiguity_prior_must_not_emit_box")
    write_json(out / "PROOF.json", proof, overwrite=True)
    validate_r7c_absent_wrist_fixture_proof(proof, root=_ROOT)
    print(json.dumps({"ok": True, "fixture": PILOT_R7C_FIXTURE_ABSENT_REL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
