"""Build canonical VSL400 eligible-ID manifest (exclude proven-bad clips before corpus jobs)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import (
    rel_posix,
    sha256_file,
    sha256_id_list,
    write_json_atomic,
)

# Gate3 / decode evidence: 019442 cannot open (corrupt mpeg4).
DEFAULT_EXCLUSIONS = [
    {
        "video_id": "019442",
        "dataset": "VSL400",
        "split": "test",
        "reason": "decode_failure",
        "evidence": [
            {
                "source": "Docs/results/gate3/GATE3_REPORT_pre_schema_v2.json",
                "error": "Không mở được video: .../019442.mp4",
            },
            {
                "source": "Docs/results/gate3/GATE3_REPORT.json",
                "note": "Gate3 schema-v2 KPI n=6165 skip=1 (019442)",
                "eval_payload_sha256": "c4172c77a9f634b1710e348ebf6df1fe3ff344ae2caf3ae8f2ec802bee01d731",
            },
        ],
    }
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=_ROOT / "metadata.csv")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "data" / "processed" / "vsl400_eligible_sample_manifest.json",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    v = df[df["dataset"] == "VSL400"].copy()
    excl_ids = {e["video_id"] for e in DEFAULT_EXCLUSIONS}
    present_excl = sorted(set(v["video_id"].astype(str)) & excl_ids)
    missing_excl = sorted(excl_ids - set(v["video_id"].astype(str)))

    eligible = v[~v["video_id"].astype(str).isin(excl_ids)].copy()
    by_split = {
        str(s): int(n)
        for s, n in eligible.groupby(eligible["split"].astype(str))["video_id"].count().items()
    }
    ids = eligible["video_id"].astype(str).tolist()
    # Stable order for hash: sort
    ids_sorted = sorted(ids)

    meta_arg = args.metadata if args.metadata.is_absolute() else _ROOT / args.metadata
    try:
        meta_rel = rel_posix(_ROOT, meta_arg)
    except ValueError:
        raise SystemExit(f"[FAIL] --metadata must be inside repo root for tracked artifact: {meta_arg}")

    payload = {
        "schema": "vsl400_eligible_sample_manifest_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VSL400",
        "metadata": meta_rel,
        "metadata_sha256": sha256_file(meta_arg),
        "eligible_ids": ids_sorted,
        "eligible_n": len(ids_sorted),
        "eligible_by_split": by_split,
        "sample_id_sha256": sha256_id_list(ids_sorted),
        "excluded": [e for e in DEFAULT_EXCLUSIONS if e["video_id"] in present_excl],
        "excluded_n": len(present_excl),
        "excluded_missing_from_metadata": missing_excl,
        "kpi_note": (
            "Full SI test KPI uses eligible test IDs only (historically n=6165 with excluded=1). "
            "Never silently skip without recording exclusion evidence."
        ),
    }
    out = args.out if args.out.is_absolute() else _ROOT / args.out
    write_json_atomic(out, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out),
                "eligible_n": payload["eligible_n"],
                "excluded_n": payload["excluded_n"],
                "sample_id_sha256": payload["sample_id_sha256"],
                "by_split": by_split,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
