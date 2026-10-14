"""Build Option B demo set from VSL400 val (never test KPI)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "data" / "processed" / "demo_set_option_b.json"


def main() -> int:
    df = pd.read_csv(_ROOT / "metadata.csv")
    v = df[(df["dataset"] == "VSL400") & (df["split"] == "val")].copy()
    # Prefer clips with RGB cache
    rgb = _ROOT / "data" / "features" / "isolated_sf2304"
    v = v[v["video_id"].astype(str).map(lambda x: (rgb / f"{x}.npy").is_file())]
    # Pick 3 scenarios later filled by eval; seed stable sample of 30
    sample = v.sample(n=min(30, len(v)), random_state=42)
    payload = {
        "gate": "D4",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "demo_set from VSL400 val only; disjoint from test KPI",
        "n": int(len(sample)),
        "video_ids": sample["video_id"].astype(str).tolist(),
        "scenarios": {
            "top1_correct": None,
            "top5_assist": None,
        },
        "note": "Fill after E2; defense UI uses Top-1 success + original Top-5 assist.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[D4] demo set n={payload['n']} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
