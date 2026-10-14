#!/usr/bin/env python3
"""Freeze D0-PILOT-R1 engineering sample set (train/val only; evidence-based tags).

Does not overwrite Docs/results/ms_sta_v5_d0/D0_PILOT_SAMPLE_SET.json.
Negative-control corrupt/ineligible clips go to a separate set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
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
    PILOT_R1_NEGATIVE_REL,
    PILOT_R1_SAMPLE_REL,
    SPLIT_REL,
    load_json,
    sha256_bytes,
    sha256_file,
    validate_pilot_sample_set_r1,
    write_json,
)
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402


def _probe_decode_ok(path: Path, *, min_frames: int = 8) -> tuple[bool, int, str | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0, "cannot_open"
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        n += 1
        if n > 5000:
            break
    cap.release()
    if n < min_frames:
        return False, n, "too_few_frames"
    return True, int(n), None


def _probe_missing_rate(path: Path, *, max_frames: int = 24, stride: int = 2) -> float:
    from src.data_prep.extract_and_smooth import SignLanguageDataEngine

    engine = SignLanguageDataEngine(model_complexity=0, frame_stride=stride)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        engine.mp_holistic.close()
        raise RuntimeError(f"cannot_open:{path}")
    rates: list[float] = []
    native_i = 0
    kept = 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if native_i % stride == 0:
            _j, _pose, miss = engine.extract_frame_landmarks(frame)
            rates.append(float(np.asarray(miss, dtype=bool).mean()))
            kept += 1
        native_i += 1
    cap.release()
    engine.mp_holistic.close()
    if not rates:
        return 1.0
    return float(np.mean(rates))


def _resolve_row(resolver: RawVideoResolver, row: pd.Series) -> dict | None:
    path = resolver.resolve(row)
    if path is None or not Path(path).is_file():
        return None
    ok, n, err = _probe_decode_ok(Path(path))
    if not ok:
        return None
    rel = str(Path(path).resolve().relative_to(_ROOT.resolve())).replace("\\", "/")
    return {
        "sample_id": str(row["video_id"]),
        "dataset": "VSL400",
        "view": "front",
        "split": str(row["split"]),
        "signer_id": str(row["signer_id"]),
        "gloss": str(row.get("text_gloss") or ""),
        "n_frames": int(n),
        "video_rel": rel,
        "decode_ok": True,
        "decode_failure_evidence": False,
    }


def main() -> int:
    if (_ROOT / PILOT_R1_SAMPLE_REL).is_file():
        raise SystemExit(f"already_frozen:{PILOT_R1_SAMPLE_REL}")

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
    df = df[df["split"].isin(["train", "val"])].copy()
    df = df.sort_values(["text_gloss", "signer_id", "video_id"])

    resolver = RawVideoResolver(_ROOT)

    # Phase 1: measure a gloss-diverse length pool (one clip per gloss, rotating signers).
    pool: list[dict] = []
    used_ids: set[str] = set()
    signer_counts: dict[str, int] = {}
    for gloss, g in df.groupby("text_gloss", sort=True):
        # Prefer under-used signers within this gloss.
        g2 = g.copy()
        g2["_sc"] = g2["signer_id"].map(lambda s: signer_counts.get(str(s), 0))
        g2 = g2.sort_values(["_sc", "signer_id", "video_id"])
        for _, row in g2.iterrows():
            item = _resolve_row(resolver, row)
            if item is None:
                continue
            if item["sample_id"] in used_ids:
                continue
            pool.append(item)
            used_ids.add(item["sample_id"])
            signer_counts[item["signer_id"]] = signer_counts.get(item["signer_id"], 0) + 1
            break
        if len(pool) >= 40:
            break

    # Phase 2: add length extremes by scanning more clips if needed.
    if len(pool) < 20:
        raise SystemExit(f"insufficient_gloss_pool:{len(pool)}")

    # Extra length probe: sample every Nth eligible train/val row for short/long tails.
    extra: list[dict] = []
    step = max(1, len(df) // 80)
    for i, (_, row) in enumerate(df.iloc[::step].iterrows()):
        if len(extra) >= 40:
            break
        sid = str(row["video_id"])
        if sid in used_ids:
            continue
        item = _resolve_row(resolver, row)
        if item is None:
            continue
        extra.append(item)
        used_ids.add(sid)

    measured = pool + extra
    frames = sorted(p["n_frames"] for p in measured)
    q20 = float(np.quantile(frames, 0.20))
    q80 = float(np.quantile(frames, 0.80))
    measured_sorted = sorted(measured, key=lambda p: (p["n_frames"], p["sample_id"]))

    # Force include true quantile extremes from measured pool.
    candidates: list[dict] = []
    seen: set[str] = set()

    def _push(p: dict) -> None:
        if p["sample_id"] in seen:
            return
        candidates.append(p)
        seen.add(p["sample_id"])

    _push(measured_sorted[0])
    _push(measured_sorted[-1])
    # Add a few near-q20 / near-q80
    for p in measured_sorted:
        if p["n_frames"] <= q20:
            _push(p)
        if len([c for c in candidates if c["n_frames"] <= q20]) >= 3:
            break
    for p in reversed(measured_sorted):
        if p["n_frames"] >= q80:
            _push(p)
        if len([c for c in candidates if c["n_frames"] >= q80]) >= 3:
            break

    # Fill remaining with gloss/signer diversity from gloss pool first.
    for p in sorted(pool, key=lambda x: (x["gloss"], x["signer_id"], x["sample_id"])):
        _push(p)
        if len(candidates) >= 28:
            break

    # Probe missing rates on extremes + mid subset.
    probe_list = (
        [candidates[0], candidates[-1]]
        + candidates[len(candidates) // 3 : len(candidates) // 3 + 2]
        + [c for c in candidates if c["n_frames"] <= q20][:2]
        + [c for c in candidates if c["n_frames"] >= q80][:2]
    )
    missing_rates: dict[str, float] = {}
    for p in probe_list:
        if p["sample_id"] in missing_rates:
            continue
        missing_rates[p["sample_id"]] = _probe_missing_rate(_ROOT / p["video_rel"])

    selected: list[dict] = []
    used_sel: set[str] = set()
    used_gloss: set[str] = set()
    used_signers: set[str] = set()

    def _add(p: dict) -> bool:
        if p["sample_id"] in used_sel:
            return False
        n = int(p["n_frames"])
        length_q = float(sum(1 for f in frames if f <= n) / max(1, len(frames)))
        tags = ["train_val_engineering"]
        if n <= q20:
            tags.append("short_clip")
        if n >= q80:
            tags.append("long_clip")
        mr = missing_rates.get(p["sample_id"])
        if mr is not None and mr >= 0.05:
            tags.append("missing_hand_or_face_risk")
        selected.append(
            {
                **p,
                "edge_tags": sorted(set(tags)),
                "length_quantile": round(length_q, 4),
                "length_q20": q20,
                "length_q80": q80,
                "measured_missing_rate": None if mr is None else round(float(mr), 6),
            }
        )
        used_sel.add(p["sample_id"])
        used_gloss.add(p["gloss"])
        used_signers.add(p["signer_id"])
        return True

    # Prefer new gloss + signer while keeping short/long.
    shorts = [c for c in candidates if c["n_frames"] <= q20]
    longs = [c for c in candidates if c["n_frames"] >= q80]
    if shorts:
        _add(shorts[0])
    if longs:
        _add(longs[-1])

    for p in sorted(candidates, key=lambda x: (x["gloss"], x["signer_id"], x["sample_id"])):
        if len(selected) >= 16:
            break
        if p["gloss"] in used_gloss and len(used_gloss) < 8:
            continue
        if p["signer_id"] in used_signers and len(used_signers) < 6:
            continue
        _add(p)

    for p in candidates:
        if len(selected) >= 16:
            break
        _add(p)

    # Ensure measured missing-risk tag exists.
    if not any("missing_hand_or_face_risk" in s["edge_tags"] for s in selected):
        for s in selected:
            if s["sample_id"] in missing_rates:
                continue
            mr = _probe_missing_rate(_ROOT / s["video_rel"], max_frames=32)
            s["measured_missing_rate"] = round(float(mr), 6)
            missing_rates[s["sample_id"]] = mr
            if mr >= 0.05:
                s["edge_tags"] = sorted(set(s["edge_tags"] + ["missing_hand_or_face_risk"]))
                break
        else:
            for s in selected:
                mr = _probe_missing_rate(_ROOT / s["video_rel"], max_frames=40)
                s["measured_missing_rate"] = round(float(mr), 6)
                if mr >= 0.05:
                    s["edge_tags"] = sorted(set(s["edge_tags"] + ["missing_hand_or_face_risk"]))
                    break

    if not any("short_clip" in s["edge_tags"] for s in selected) or not any(
        "long_clip" in s["edge_tags"] for s in selected
    ):
        raise SystemExit("quantile_tags_missing_after_selection")

    ids = [s["sample_id"] for s in selected]
    obj = {
        "schema": "ms_sta_v5_d0_r1_pilot_sample_set_v1",
        "status": "FROZEN",
        "frozen_at": "2026-08-13T07:30:00Z",
        "engineering_scope": "train_val_only",
        "test_clips_forbidden_for_roi_preprocess": True,
        "parent_independent_reaudit": "Docs/results/ms_sta_v5_d0/D0_PILOT_INDEPENDENT_REAUDIT.json",
        "eligible_manifest": ELIGIBLE_REL,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata": METADATA_REL,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers": SPLIT_REL,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
        "n_samples": len(selected),
        "sample_ids": ids,
        "sample_ids_sha256": sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
        "samples": selected,
        "length_quantiles": {
            "q20": q20,
            "q80": q80,
            "min": int(min(frames)),
            "max": int(max(frames)),
            "measured_pool_n": len(measured),
        },
        "selection_policy": {
            "front_view_only": True,
            "train_val_only": True,
            "eligible_intersection": True,
            "gloss_diverse_pool": True,
            "length_tags_from_quantiles": True,
            "missing_tags_from_measured_masks": True,
            "decode_edge_only_with_failure_evidence": True,
            "never_relabel_ordinary_as_decode_edge": True,
            "frozen_before_execution": True,
            "does_not_overwrite_original_pilot": True,
        },
    }
    digest = write_json(_ROOT / PILOT_R1_SAMPLE_REL, obj)
    validate_pilot_sample_set_r1(obj, root=_ROOT)

    # Separate negative-control set.
    neg_samples = []
    row_019 = meta[meta["video_id"] == "019442"]
    if len(row_019):
        row = row_019.iloc[0]
        path = resolver.resolve(row)
        entry = {
            "sample_id": "019442",
            "reason": "expected_failure_negative_control",
            "eligible": "019442" in eligible,
            "split": str(row.get("split")),
            "decode_failure_evidence": None,
            "video_rel": None,
        }
        if path is not None and Path(path).is_file():
            ok, n, err = _probe_decode_ok(Path(path), min_frames=8)
            entry["video_rel"] = str(Path(path).resolve().relative_to(_ROOT.resolve())).replace("\\", "/")
            entry["decode_ok"] = ok
            entry["n_frames"] = n
            entry["decode_error"] = err
            entry["decode_failure_evidence"] = (not ok)
            entry["edge_tags"] = ["decode_edge"] if not ok else ["ineligible_or_corrupt_control"]
        else:
            entry["decode_ok"] = False
            entry["decode_failure_evidence"] = True
            entry["decode_error"] = "missing_or_unresolvable"
            entry["edge_tags"] = ["decode_edge"]
        neg_samples.append(entry)

    inject_dir = _ROOT / "Docs/results/ms_sta_v5_d0_r1/negative_controls"
    inject_dir.mkdir(parents=True, exist_ok=True)
    inject_path = inject_dir / "injected_corrupt_empty.mp4"
    if not inject_path.is_file():
        inject_path.write_bytes(b"")
    ok_i, n_i, err_i = _probe_decode_ok(inject_path, min_frames=2)
    neg_samples.append(
        {
            "sample_id": "INJECTED_CORRUPT_EMPTY",
            "reason": "injected_corrupt_decode_edge",
            "eligible": False,
            "split": None,
            "video_rel": str(inject_path.resolve().relative_to(_ROOT.resolve())).replace("\\", "/"),
            "decode_ok": ok_i,
            "n_frames": n_i,
            "decode_error": err_i or "empty_file",
            "decode_failure_evidence": True,
            "edge_tags": ["decode_edge"],
        }
    )

    neg = {
        "schema": "ms_sta_v5_d0_r1_negative_control_set_v1",
        "status": "FROZEN",
        "note": "Separate from engineering pilot; not used for ROI/preprocess decisions",
        "samples": neg_samples,
        "forbid_use_in_roi_preprocess": True,
    }
    if (_ROOT / PILOT_R1_NEGATIVE_REL).is_file():
        neg_sha = sha256_file(_ROOT / PILOT_R1_NEGATIVE_REL)
    else:
        neg_sha = write_json(_ROOT / PILOT_R1_NEGATIVE_REL, neg)

    print(
        json.dumps(
            {
                "ok": True,
                "pilot_r1_sample_sha256": digest,
                "negative_control_sha256": neg_sha,
                "n": len(selected),
                "n_glosses": len({s["gloss"] for s in selected}),
                "n_signers": len({s["signer_id"] for s in selected}),
                "frame_range": [min(s["n_frames"] for s in selected), max(s["n_frames"] for s in selected)],
                "q20": q20,
                "q80": q80,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
