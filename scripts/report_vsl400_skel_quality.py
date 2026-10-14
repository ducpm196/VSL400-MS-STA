"""Gate D1 — VSL400 skeleton coverage + quality report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_features import resolve_skel_dir

SKEL_DIR = resolve_skel_dir(_ROOT)
# Reject extreme scale that indicates shoulder-norm blow-up (~1e6).
OUTLIER_ABS_MAX = 50.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Option B Gate D1 skeleton quality report")
    ap.add_argument("--metadata", type=Path, default=_ROOT / "metadata.csv")
    ap.add_argument("--skel-dir", type=Path, default=SKEL_DIR)
    ap.add_argument(
        "--sample-check",
        type=int,
        default=0,
        help="Max files to deep-scan (0 = all present files — required for D1 pass claim)",
    )
    ap.add_argument("--min-coverage", type=float, default=0.95, help="Pass threshold on VSL400 coverage")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "logs" / "gate_d1" / "skel_quality_report.json",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    v = df[df["dataset"] == "VSL400"].copy()
    vids = v["video_id"].astype(str).tolist()
    present = []
    missing = []
    for vid in vids:
        p = args.skel_dir / f"{vid}.npy"
        (present if p.is_file() else missing).append(vid)

    coverage = len(present) / max(len(vids), 1)
    by_split = {}
    for split_name, g in v.groupby(v["split"].astype(str)):
        ids = g["video_id"].astype(str)
        hit = sum((args.skel_dir / f"{vid}.npy").is_file() for vid in ids)
        by_split[split_name] = {"n": int(len(ids)), "hit": int(hit), "coverage": hit / max(len(ids), 1)}

    # Quality deep-scan: default = all present (honest D1). Optional subsample for smoke.
    if args.sample_check and args.sample_check > 0 and len(present) > args.sample_check:
        rng = np.random.default_rng(42)
        sample_ids = list(rng.choice(present, size=args.sample_check, replace=False))
        full_scan = False
    else:
        sample_ids = list(present)
        full_scan = True

    bad_shape = 0
    nan_files = 0
    inf_files = 0
    zero_frame_files = 0
    outlier_scale_files = 0
    t_lens: list[int] = []
    abs_maxes: list[float] = []
    for vid in sample_ids:
        arr = np.load(args.skel_dir / f"{vid}.npy")
        if arr.ndim != 3 or arr.shape[1:] != (75, 3):
            bad_shape += 1
            continue
        t_lens.append(int(arr.shape[0]))
        if np.isnan(arr).any():
            nan_files += 1
        if np.isinf(arr).any():
            inf_files += 1
        # zero frame = all joints zero
        if np.any(np.all(arr == 0, axis=(1, 2))):
            zero_frame_files += 1
        finite = arr[np.isfinite(arr)]
        if finite.size:
            amax = float(np.max(np.abs(finite)))
            abs_maxes.append(amax)
            if amax > OUTLIER_ABS_MAX:
                outlier_scale_files += 1

    quality_ok = (
        bad_shape == 0
        and nan_files == 0
        and inf_files == 0
        and zero_frame_files == 0
        and outlier_scale_files == 0
    )
    report = {
        "gate": "D1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "skel_dir": str(args.skel_dir),
        "vsl400_n": len(vids),
        "present": len(present),
        "missing": len(missing),
        "coverage": coverage,
        "by_split": by_split,
        "full_scan": full_scan,
        "sample_checked": len(sample_ids),
        "bad_shape": bad_shape,
        "nan_files": nan_files,
        "inf_files": inf_files,
        "zero_frame_files": zero_frame_files,
        "outlier_scale_files": outlier_scale_files,
        "outlier_abs_max_threshold": OUTLIER_ABS_MAX,
        "abs_max_observed": float(max(abs_maxes)) if abs_maxes else None,
        "t_min": min(t_lens) if t_lens else None,
        "t_max": max(t_lens) if t_lens else None,
        "t_mean": float(np.mean(t_lens)) if t_lens else None,
        "min_coverage_required": args.min_coverage,
        "pass": bool(coverage >= args.min_coverage and quality_ok and full_scan),
        "pass_note": (
            None
            if full_scan
            else "Quality subsample only — not valid for D1 PASS; re-run with --sample-check 0"
        ),
        "missing_preview": missing[:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[D1] coverage={coverage:.4f} present={len(present)}/{len(vids)} by_split={by_split}")
    print(
        f"[D1] scan full={full_scan} n={len(sample_ids)} "
        f"bad_shape={bad_shape} nan={nan_files} inf={inf_files} "
        f"zero_frames={zero_frame_files} outlier_scale={outlier_scale_files}"
    )
    print(f"[D1] report -> {args.out}")
    if report["pass"]:
        print("[D1] PASS")
        return 0
    print("[D1] FAIL - coverage/quality incomplete; do NOT claim D1 until full-scan PASS")
    return 1


if __name__ == "__main__":
    sys.exit(main())
