# -*- coding: utf-8 -*-
"""MODEL_ASSISTED_QA overlays — semantic joint labels on >=8 pilot clips."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.e4_r4_joint_layouts import (
    FACE_LOCAL_BY_NAME,
    MOUTH_ANCHOR_FACE_NAMES,
    POSE_LOCAL_BY_NAME,
    layout_indices,
)
from src.data_prep.joint_layout_v3 import FACE, POSE
from src.data_prep.raw_video_resolve import RawVideoResolver

LABEL_PAIRS = (
    ("left_shoulder", "pose"),
    ("right_shoulder", "pose"),
    ("left_wrist", "pose"),
    ("right_wrist", "pose"),
    ("lip_outer_left", "face"),
    ("lip_outer_right", "face"),
    ("lip_outer_bottom", "face"),
    ("lip_inner_lower", "face"),
)


def die(msg: str) -> None:
    print(f"E4_R4_QA_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, default=ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4_PILOT_SAMPLE_FREEZE.json")
    ap.add_argument("--n-clips", type=int, default=8)
    ap.add_argument("--skel-layout", default="S_B")
    args = ap.parse_args()

    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    qa_ids = list(freeze["qa_contact_ids"])[: max(8, args.n_clips)]
    if len(qa_ids) < 8:
        die(f"need_at_least_8_qa_ids_got_{len(qa_ids)}")

    meta = {}
    with (ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meta[str(row["video_id"])] = row
    resolver = RawVideoResolver(ROOT)
    overlay_dir = out_root / "qa_overlays_semantic"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Map semantic name -> compact layout index
    full_idxs = list(layout_indices(args.skel_layout))
    name_to_full = {}
    for name in POSE_LOCAL_BY_NAME:
        name_to_full[name] = POSE.start + POSE_LOCAL_BY_NAME[name]
    for name in FACE_LOCAL_BY_NAME:
        name_to_full[name] = FACE.start + FACE_LOCAL_BY_NAME[name]

    written = []
    for sid in qa_ids[: args.n_clips]:
        skel_path = out_root / f"skel_{args.skel_layout}" / f"{sid}.npz"
        if not skel_path.is_file():
            # allow sealed fallback for overlay-only dry runs
            sealed = (
                ROOT
                / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3"
                / f"{sid}.npz"
            )
            if not sealed.is_file():
                die(f"missing_skel_for_qa:{sid}")
            with np.load(sealed, allow_pickle=False) as z:
                coords75 = np.asarray(z["raw_landmarks"] if "raw_landmarks" in z.files else z["joints_raw"])
                fi = int(np.asarray(z["original_frame_index"])[min(5, len(z["original_frame_index"]) - 1)])
                xy = coords75[min(5, coords75.shape[0] - 1), :, :2]
                use_full = True
        else:
            with np.load(skel_path, allow_pickle=False) as z:
                coords = np.asarray(z["coordinates"])
                src_idx = np.asarray(z["source_joint_indices"], dtype=np.int32)
                fi = int(np.asarray(z["original_frame_index"])[min(5, coords.shape[0] - 1)])
                # compact coords are normalized — for overlay prefer sealed raw
                sealed = (
                    ROOT
                    / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3"
                    / f"{sid}.npz"
                )
                with np.load(sealed, allow_pickle=False) as zs:
                    coords75 = np.asarray(zs["raw_landmarks"] if "raw_landmarks" in zs.files else zs["joints_raw"])
                    trow = min(5, coords75.shape[0] - 1)
                    xy = coords75[trow, :, :2]
                use_full = True

        row = meta.get(sid)
        if row is None:
            die(f"missing_meta:{sid}")
        vpath = resolver.resolve(row)
        if vpath is None or not vpath.is_file():
            die(f"missing_video:{sid}")
        cap = cv2.VideoCapture(str(vpath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            die(f"cannot_read_frame:{sid}:{fi}")
        h, w = bgr.shape[:2]
        for name, _kind in LABEL_PAIRS:
            j = name_to_full[name]
            px = int(np.clip(xy[j, 0] * w, 0, w - 1))
            py = int(np.clip(xy[j, 1] * h, 0, h - 1))
            cv2.circle(bgr, (px, py), 4, (0, 255, 0), -1)
            cv2.putText(bgr, name, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        # Explicit mouth anchors presence check
        for name in MOUTH_ANCHOR_FACE_NAMES:
            if name not in name_to_full:
                die(f"mouth_anchor_missing_in_map:{name}")
        out_png = overlay_dir / f"{sid}_semantic.png"
        cv2.imwrite(str(out_png), bgr)
        written.append(out_png.name)

    report = {
        "schema": "ms_sta_v6_e4_r4_model_assisted_qa_v1",
        "qa_label": "MODEL_ASSISTED_QA",
        "forbid_label": "human-reviewed",
        "n_overlays": len(written),
        "overlays": written,
        "semantic_labels": [n for n, _ in LABEL_PAIRS],
        "mouth_anchors": list(MOUTH_ANCHOR_FACE_NAMES),
        "note": "Automated overlay for representation QA; not human-reviewed.",
    }
    (out_root / "E4_R4_MODEL_ASSISTED_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if len(written) < 8:
        die(f"fewer_than_8_overlays:{len(written)}")
    print("QA_OVERLAYS_OK", len(written), overlay_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
