#!/usr/bin/env python3
"""Streamlit AppTest smoke for E4 appendix (no live API required for fail path)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def main() -> int:
    from streamlit.testing.v1 import AppTest

    health = {
        "ready": True,
        "engine": "ms_sta_v6_b4_v6_ctr",
        "e4_ready": False,
        "ms_sta_ready": True,
    }

    with mock.patch("requests.get") as get:
        resp = mock.Mock()
        resp.json.return_value = health
        resp.status_code = 200
        get.return_value = resp
        at = AppTest.from_file(
            str(_ROOT / "src/deployment/pages/2_E4_Efficiency_Appendix.py"),
            default_timeout=30,
        )
        at.run()

    assert not at.exception, at.exception
    # Fail-closed banner / success decision must mention FEASIBILITY_NO_GO.
    texts = " ".join(str(x.value) for x in list(at.markdown) + list(at.error) + list(at.success))
    assert "E4_R3_FEASIBILITY_NO_GO" in texts or any(
        "E4_R3_FEASIBILITY_NO_GO" in str(e.value) for e in at.error
    )
    assert any("e4_ready" in str(e.value) or "e4_ready=false" in texts for e in list(at.error) + list(at.success) + list(at.markdown)) or "e4_ready=false" in texts
    print({"event": "e4_apptest_ok", "error_count": len(at.error), "success_count": len(at.success)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
