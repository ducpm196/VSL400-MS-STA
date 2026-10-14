# -*- coding: utf-8 -*-
"""Unit tests for E4_B4SC_V1 E1 schedule lock (no GPU required)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V1_E1_SCHEDULE_LOCK.json"
POL = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"


def test_e1_schedule_lock_factorial_frozen():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["status"] == "FROZEN_BEFORE_E1_METRICS"
    assert lock["e4_ready"] is False
    assert set(lock["factorial_arms"]) == {"E0", "E1a", "E1b", "E1c"}
    assert lock["factorial_arms"]["E0"] == {
        "precision": "FP32",
        "decode_pipeline": "original",
        "q2_amp": False,
        "batched_windows": False,
    }
    assert lock["bindings"]["n_samples"] == 24
    assert len(lock["bindings"]["sample_ids"]) == 24
    assert lock["measurement_schedule"]["measured_rounds_per_sample_per_arm"] == 5
    assert lock["statistics"]["n_boot"] == 10000
    assert lock["forbid_open_E2_E3_E4_E5_in_this_task"] is True
    assert "self" not in json.dumps(lock).lower() or "self-SHA" in lock.get("note", "")


def test_e1_policy_still_blocks_slow_open():
    pol = json.loads(POL.read_text(encoding="utf-8"))
    assert pol["e4_ready"] is False
    assert pol["amendment_id"] == "SCIENCE_REVIEW_V1"
    assert pol["slow_width_ladder_locked"]["E2_slow_widths"] == [224, 448, 896, 1792]
