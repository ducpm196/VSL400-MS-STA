"""Backward-compatible wrapper — prefer summarize_ms_sta_gates.py. """

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_ms_sta_gates import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
