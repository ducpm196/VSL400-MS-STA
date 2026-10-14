#!/usr/bin/env python3
"""FORBIDDEN producer: must never write human manual verdict fields.

Retained in the executable tree so code-lock can prove the PASS-filling path is disabled.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "FORBIDDEN: fill_ms_sta_v5_d0_r7c_h_manual.py must not write human verdict fields.\n"
        "Use the null H1 template and enter each prior_tile_judgment after visual inspection.\n"
        "Optional merge helper: scripts/merge_ms_sta_v5_d0_r7c_h1_human_judgments.py "
        "(requires a complete human judgments file; invents nothing).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
