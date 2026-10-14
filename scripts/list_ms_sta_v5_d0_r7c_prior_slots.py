#!/usr/bin/env python3
"""List R7C prior-tile slots for genuine human review (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import PILOT_R7C_OVERLAYS_INDEX_REL, RGB_PRIOR_SOURCES, load_json  # noqa: E402


def main() -> int:
    overlays = load_json(_ROOT / PILOT_R7C_OVERLAYS_INDEX_REL)["overlays_index"]
    rows = []
    for o in overlays:
        if o["kind"] == "center_overlay":
            src = o.get("rgb_roi_sources") or {}
            for hand in ("left_hand", "right_hand"):
                if src.get(hand) in RGB_PRIOR_SOURCES:
                    rows.append(
                        {
                            "kind": "center",
                            "sample_id": o["sample_id"],
                            "window_start_idx": o["window_start_idx"],
                            "hand": hand,
                            "source": src[hand],
                            "native_frame": o.get("native_frame"),
                            "path": o["path"],
                            "boxes": (o.get("boxes") or {}).get(hand),
                        }
                    )
        else:
            for t in o.get("tiles") or []:
                tsrc = t.get("rgb_roi_sources") or {}
                for hand in ("left_hand", "right_hand"):
                    if tsrc.get(hand) in RGB_PRIOR_SOURCES:
                        rows.append(
                            {
                                "kind": "tile",
                                "sample_id": o["sample_id"],
                                "window_start_idx": o["window_start_idx"],
                                "pathway": t.get("pathway"),
                                "t_idx": t.get("t_idx"),
                                "native_frame": t.get("native_frame"),
                                "hand": hand,
                                "source": tsrc[hand],
                                "path": o["path"],
                                "boxes": (t.get("boxes") or {}).get(hand),
                            }
                        )
    print(json.dumps({"n": len(rows), "slots": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
