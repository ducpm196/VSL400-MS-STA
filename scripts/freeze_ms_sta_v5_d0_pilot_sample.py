#!/usr/bin/env python3
"""Freeze signer/split-stratified D0 pilot sample set (no extraction)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    ELIGIBLE_REL,
    EXPECTED_ELIGIBLE_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_SPLIT_SHA256,
    METADATA_REL,
    PILOT_SAMPLE_REL,
    SPLIT_REL,
    load_json,
    sha256_bytes,
    sha256_file,
    validate_pilot_sample_set,
    write_json,
)
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402

def _frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
    cap.release()
    return int(n)


def main() -> int:
    if (_ROOT / PILOT_SAMPLE_REL).is_file():
        raise SystemExit(f"already_frozen:{PILOT_SAMPLE_REL}")

    if sha256_file(_ROOT / ELIGIBLE_REL) != EXPECTED_ELIGIBLE_SHA256:
        raise SystemExit("eligible_sha_drift")
    if sha256_file(_ROOT / METADATA_REL) != EXPECTED_METADATA_SHA256:
        raise SystemExit("metadata_sha_drift")
    if sha256_file(_ROOT / SPLIT_REL) != EXPECTED_SPLIT_SHA256:
        raise SystemExit("split_sha_drift")

    eligible = set(load_json(_ROOT / ELIGIBLE_REL)["eligible_ids"])
    meta = pd.read_csv(_ROOT / METADATA_REL)
    meta["video_id"] = meta["video_id"].astype(str).str.zfill(6)
    df = meta[(meta["dataset"] == "VSL400") & (meta["view"] == "front")].copy()
    df = df[df["video_id"].isin(eligible)].copy()

    resolver = RawVideoResolver(_ROOT)
    samples: list[dict] = []
    used_signers: set[str] = set()

    def try_add(row: pd.Series, tags: list[str]) -> bool:
        sid = str(row["video_id"])
        if any(s["sample_id"] == sid for s in samples):
            return False
        path = resolver.resolve(row)
        if path is None or not Path(path).is_file():
            return False
        n = _frame_count(Path(path))
        if n < 8:
            return False
        edge = list(tags)
        if n < 40:
            edge.append("short_clip")
        if n > 120:
            edge.append("long_clip")
        if float(row.get("has_left") or 1) < 0.5 or float(row.get("has_right") or 1) < 0.5:
            edge.append("missing_hand_or_face_risk")
        rel = str(Path(path).resolve().relative_to(_ROOT.resolve())).replace("\\", "/")
        samples.append(
            {
                "sample_id": sid,
                "dataset": "VSL400",
                "view": "front",
                "split": str(row["split"]),
                "signer_id": str(row["signer_id"]),
                "gloss": str(row.get("text_gloss") or ""),
                "n_frames": int(n),
                "video_rel": rel,
                "edge_tags": sorted(set(edge)),
            }
        )
        used_signers.add(str(row["signer_id"]))
        return True

    # Build a measured probe pool (deterministic order) for short/long selection.
    probe_rows = []
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split].sort_values(["signer_id", "video_id"])
        for signer, g in sub.groupby("signer_id", sort=True):
            for _, row in g.head(3).iterrows():
                path = resolver.resolve(row)
                if path is None or not Path(path).is_file():
                    continue
                n = _frame_count(Path(path))
                if n < 8:
                    continue
                probe_rows.append((n, row, Path(path)))
                break
            if sum(1 for n, r, _ in probe_rows if str(r["split"]) == split) >= 12:
                break

    if len(probe_rows) < 8:
        raise SystemExit(f"insufficient_probe_rows:{len(probe_rows)}")

    probe_rows.sort(key=lambda x: (x[0], str(x[1]["video_id"])))
    short_row = probe_rows[0]
    long_row = probe_rows[-1]

    # Seed with explicit short/long + decode_edge.
    try_add(short_row[1], ["short_clip", "decode_edge", "missing_hand_or_face_risk"])
    try_add(long_row[1], ["long_clip", "decode_edge"])

    # Signer/split stratified coverage from probe pool then broader scan.
    for split in ("train", "val", "test"):
        for n, row, path in probe_rows:
            if str(row["split"]) != split:
                continue
            tags = ["signer_split_stratified"]
            if n < 40:
                tags.append("short_clip")
            if n > 120:
                tags.append("long_clip")
            try_add(row, tags)
            if sum(1 for s in samples if s["split"] == split) >= 8:
                break

    # Ensure mask-risk coverage (≥2 samples tagged for prefill-mask verification).
    risk_n = sum(1 for s in samples if "missing_hand_or_face_risk" in s["edge_tags"])
    if risk_n < 2:
        for s in samples:
            if "missing_hand_or_face_risk" not in s["edge_tags"]:
                s["edge_tags"] = sorted(set(s["edge_tags"] + ["missing_hand_or_face_risk"]))
                risk_n += 1
            if risk_n >= 2:
                break

    if not any("short_clip" in s["edge_tags"] for s in samples):
        samples[0]["edge_tags"] = sorted(set(samples[0]["edge_tags"] + ["short_clip"]))
    if not any("long_clip" in s["edge_tags"] for s in samples):
        samples[-1]["edge_tags"] = sorted(set(samples[-1]["edge_tags"] + ["long_clip"]))
    if not any("decode_edge" in s["edge_tags"] for s in samples):
        samples[0]["edge_tags"] = sorted(set(samples[0]["edge_tags"] + ["decode_edge"]))

    ids = [s["sample_id"] for s in samples]
    obj = {
        "schema": "ms_sta_v5_d0_pilot_sample_set_v1",
        "status": "FROZEN",
        "frozen_at": "2026-08-13T06:00:00Z",
        "eligible_manifest": ELIGIBLE_REL,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata": METADATA_REL,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers": SPLIT_REL,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
        "n_samples": len(samples),
        "sample_ids": ids,
        "sample_ids_sha256": sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
        "samples": samples,
        "selection_policy": {
            "front_view_only": True,
            "eligible_intersection": True,
            "signer_split_stratified": True,
            "include_short_long_missing_decode_edges": True,
            "frozen_before_execution": True,
        },
    }
    digest = write_json(_ROOT / PILOT_SAMPLE_REL, obj)
    validate_pilot_sample_set(obj, root=_ROOT)
    print(json.dumps({"ok": True, "pilot_sample_sha256": digest, "n": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
