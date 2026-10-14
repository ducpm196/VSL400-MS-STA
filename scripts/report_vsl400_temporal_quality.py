"""Full-scan quality report for versioned RGB temporal feature dirs (pre-promote)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import (
    ELIGIBLE_MANIFEST_REL,
    load_eligible_sample_manifest,
    load_json,
    sha256_file,
)

FEAT_DIM = 2304
MIN_L = 2
EXPECTED_DTYPE = "float32"


def main() -> int:
    ap = argparse.ArgumentParser(description="VSL400 temporal RGB full-scan quality report")
    ap.add_argument(
        "--temporal-dir",
        type=Path,
        required=True,
        help="Versioned temporal feature directory containing *.npy",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="End-of-build temporal _manifest.json (optional cross-check)",
    )
    ap.add_argument(
        "--sample-manifest",
        type=Path,
        default=_ROOT / ELIGIBLE_MANIFEST_REL,
    )
    ap.add_argument(
        "--sample-check",
        type=int,
        default=0,
        help="Max files to deep-scan (0 = all eligible IDs — required for promote)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "logs" / "extract_temporal" / "quality_temporal.json",
    )
    args = ap.parse_args()

    temporal_dir = args.temporal_dir
    if not temporal_dir.is_absolute():
        temporal_dir = _ROOT / temporal_dir
    if not temporal_dir.is_dir():
        print(f"[FAIL] temporal dir missing: {temporal_dir}")
        return 2

    pack = load_eligible_sample_manifest(args.sample_manifest, root=_ROOT)
    eligible_ids = [str(x) for x in pack["eligible_ids"]]
    eligible_set = set(eligible_ids)
    meta = pd.read_csv(_ROOT / "metadata.csv")
    vsl = meta[meta["dataset"] == "VSL400"].copy()
    vsl["video_id"] = vsl["video_id"].astype(str)
    elig_rows = vsl[vsl["video_id"].isin(eligible_set)]

    present: list[str] = []
    missing: list[str] = []
    for vid in eligible_ids:
        p = temporal_dir / f"{vid}.npy"
        (present if p.is_file() else missing).append(vid)

    by_split: dict[str, dict] = {}
    for split_name, g in elig_rows.groupby(elig_rows["split"].astype(str)):
        ids = g["video_id"].astype(str).tolist()
        hit = sum((temporal_dir / f"{vid}.npy").is_file() for vid in ids)
        by_split[str(split_name)] = {
            "n": int(len(ids)),
            "hit": int(hit),
            "coverage": hit / max(len(ids), 1),
            "missing": int(len(ids) - hit),
        }

    if args.sample_check and args.sample_check > 0 and len(present) > args.sample_check:
        rng = np.random.default_rng(42)
        sample_ids = list(rng.choice(present, size=args.sample_check, replace=False))
        full_scan = False
    else:
        sample_ids = list(present)
        full_scan = True

    bad_shape = 0
    short_l = 0
    nan_files = 0
    inf_files = 0
    all_zero_files = 0
    dtype_mismatch = 0
    l_lens: list[int] = []
    dtypes: Counter[str] = Counter()
    abs_maxes: list[float] = []
    bad_preview: list[dict] = []

    for vid in sample_ids:
        arr = np.load(temporal_dir / f"{vid}.npy")
        dt = str(arr.dtype)
        dtypes[dt] += 1
        if arr.ndim != 2 or arr.shape[1] != FEAT_DIM:
            bad_shape += 1
            if len(bad_preview) < 20:
                bad_preview.append({"video_id": vid, "issue": "bad_shape", "shape": list(arr.shape)})
            continue
        L = int(arr.shape[0])
        l_lens.append(L)
        if L < MIN_L:
            short_l += 1
            if len(bad_preview) < 20:
                bad_preview.append({"video_id": vid, "issue": "L_lt_min", "L": L})
        if dt != EXPECTED_DTYPE:
            dtype_mismatch += 1
            if len(bad_preview) < 20:
                bad_preview.append({"video_id": vid, "issue": "dtype", "dtype": dt})
        if np.isnan(arr).any():
            nan_files += 1
            if len(bad_preview) < 20:
                bad_preview.append({"video_id": vid, "issue": "nan"})
        if np.isinf(arr).any():
            inf_files += 1
            if len(bad_preview) < 20:
                bad_preview.append({"video_id": vid, "issue": "inf"})
        if np.all(arr == 0):
            all_zero_files += 1
            if len(bad_preview) < 20:
                bad_preview.append({"video_id": vid, "issue": "all_zero"})
        finite = arr[np.isfinite(arr)]
        if finite.size:
            abs_maxes.append(float(np.max(np.abs(finite))))

    npy_on_disk = sum(1 for _ in temporal_dir.glob("*.npy"))
    dtype_ok = len(dtypes) == 1 and EXPECTED_DTYPE in dtypes
    split_ok = all(
        by_split.get(s, {}).get("n") == by_split.get(s, {}).get("hit")
        for s in ("train", "val", "test")
    ) and not missing

    man_cross: dict | None = None
    if args.manifest is not None:
        mpath = args.manifest if args.manifest.is_absolute() else _ROOT / args.manifest
        man = load_json(mpath)
        man_cross = {
            "manifest": str(mpath),
            "manifest_sha256": sha256_file(mpath),
            "saved": man.get("saved"),
            "failed": man.get("failed"),
            "L_min": man.get("L_min"),
            "L_max": man.get("L_max"),
            "L_mean": man.get("L_mean"),
            "n_sample_ids": man.get("n_sample_ids"),
            "git_revision": man.get("git_revision"),
            "matches_present": (
                man.get("saved") is not None and int(man.get("saved")) == len(present)
            ),
            "matches_failed_zero": (
                man.get("failed") is not None and int(man.get("failed")) == 0
            ),
            "L_min_matches_scan": (
                man.get("L_min") is not None
                and l_lens
                and int(man["L_min"]) == int(min(l_lens))
            ),
            "L_max_matches_scan": (
                man.get("L_max") is not None
                and l_lens
                and int(man["L_max"]) == int(max(l_lens))
            ),
        }

    quality_ok = (
        bad_shape == 0
        and short_l == 0
        and nan_files == 0
        and inf_files == 0
        and all_zero_files == 0
        and dtype_ok
        and dtype_mismatch == 0
        and split_ok
        and len(present) == len(eligible_ids)
        and npy_on_disk == len(present)
        and (man_cross is None or (man_cross["matches_present"] and man_cross["matches_failed_zero"]))
    )
    report = {
        "gate": "temporal_full_scan",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "temporal_dir": str(temporal_dir),
        "eligible_n": len(eligible_ids),
        "present": len(present),
        "missing": len(missing),
        "npy_on_disk": npy_on_disk,
        "coverage": len(present) / max(len(eligible_ids), 1),
        "by_split": by_split,
        "full_scan": full_scan,
        "sample_checked": len(sample_ids),
        "shape_contract": "(L, 2304)",
        "min_temporal_l": MIN_L,
        "expected_dtype": EXPECTED_DTYPE,
        "dtypes_observed": dict(dtypes),
        "dtype_unified": dtype_ok,
        "bad_shape": bad_shape,
        "short_l": short_l,
        "nan_files": nan_files,
        "inf_files": inf_files,
        "all_zero_files": all_zero_files,
        "dtype_mismatch": dtype_mismatch,
        "L_min": min(l_lens) if l_lens else None,
        "L_max": max(l_lens) if l_lens else None,
        "L_mean": float(np.mean(l_lens)) if l_lens else None,
        "abs_max_observed": float(max(abs_maxes)) if abs_maxes else None,
        "sample_manifest": str(args.sample_manifest),
        "sample_manifest_sha256": pack["sha256"],
        "sample_id_sha256": pack["raw"].get("sample_id_sha256"),
        "manifest_crosscheck": man_cross,
        "missing_preview": missing[:20],
        "bad_preview": bad_preview,
        "defense_note": (
            "Raw RGB temporal L is typically 2-3 SlowFast windows; train/eval pads to "
            "fixed_L=8 with validity mask. Do not claim eight real temporal segments."
        ),
        "pass": bool(full_scan and quality_ok),
        "pass_note": (
            None
            if full_scan
            else "Quality subsample only — not valid for temporal promote; re-run with --sample-check 0"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_sha = sha256_file(args.out)
    checksums = {
        "updated_at": report["updated_at"],
        "quality_report": str(args.out),
        "quality_report_sha256": report_sha,
        "temporal_dir": str(temporal_dir),
        "sample_manifest_sha256": pack["sha256"],
        "pass": report["pass"],
        "full_scan": full_scan,
        "present": len(present),
        "eligible_n": len(eligible_ids),
    }
    if man_cross:
        checksums["manifest_sha256"] = man_cross["manifest_sha256"]
    checksums_path = args.out.with_name(args.out.stem + "_CHECKSUMS.json")
    checksums_path.write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[temporal] present={len(present)}/{len(eligible_ids)} "
        f"npy_on_disk={npy_on_disk} by_split={by_split}"
    )
    print(
        f"[temporal] scan full={full_scan} n={len(sample_ids)} "
        f"bad_shape={bad_shape} short_l={short_l} nan={nan_files} inf={inf_files} "
        f"all_zero={all_zero_files} dtype_ok={dtype_ok} "
        f"L=[{report['L_min']},{report['L_max']}] mean={report['L_mean']}"
    )
    print(f"[temporal] report -> {args.out}")
    print(f"[temporal] checksums -> {checksums_path} sha256={report_sha}")
    if report["pass"]:
        print("[temporal] PASS")
        return 0
    print("[temporal] FAIL — do NOT promote until full-scan PASS")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
