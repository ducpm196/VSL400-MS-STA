"""Provenance refuse-existing / binding shape tests for E4-R2."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.ms_sta_e4_r2_provenance import refuse_existing_outputs


def test_refuse_existing_outputs(tmp_path: Path) -> None:
    refuse_existing_outputs(tmp_path)  # empty ok
    (tmp_path / "SUMMARY.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refuse_existing_outputs"):
        refuse_existing_outputs(tmp_path)
