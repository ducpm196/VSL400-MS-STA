# -*- coding: utf-8 -*-
"""S5 definition/schedule/ROI-order/schema tests (no GPU metrics)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "Docs/results/e4_system_kd_v1"
ROI = ["global", "left_hand", "right_hand", "face"]


def test_s5_definition_lock_s3_only():
    d = json.loads((ART / "E4_SYSKD_V1_S5_DEFINITION_LOCK.json").read_text(encoding="utf-8"))
    assert d["s5_definition"]["formula"] == "S5 = R1 + S3_ROI_batching"
    assert "S1_pinned_transfer" in d["s5_definition"]["excludes"]
    assert "S2_decode_overlap" in d["s5_definition"]["excludes"]
    assert "S4_cuda_streams" in d["s5_definition"]["excludes"]
    assert d["s5_definition"]["not_combined_multi_arm"] is True
    assert d["roi_regions_order_canonical_code"] == ROI
    assert d["selected"] is None
    assert d["e4_ready"] is False


def test_s5_schedule_lock_triple_latin():
    s = json.loads((ART / "E4_SYSKD_V1_S5_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    assert s["n_samples"] == 24
    assert s["measured_rounds"] == 5
    assert s["expected_efficiency_rows"] == 360
    assert s["arms"] == ["R0", "R1", "S5"]
    assert s["triple_arm_order_by_round"]["1"] == ["R0", "R1", "S5"]
    assert s["triple_arm_order_by_round"]["2"] == ["R1", "S5", "R0"]
    assert s["triple_arm_order_by_round"]["3"] == ["S5", "R0", "R1"]
    assert s["triple_arm_order_by_round"]["4"] == ["R0", "S5", "R1"]
    assert s["triple_arm_order_by_round"]["5"] == ["R1", "R0", "S5"]
    assert s["validation"]["n"] == 2629
    assert s["validation"]["forbid_test"] is True


def test_set_extra_coverage_parentheses():
    """Regression: `|` binds looser than `-` — must parenthesize union before difference."""
    val = {"a", "b"}
    r1 = {"a", "b"}
    s5 = {"a", "b"}
    wrong = sorted(set(r1) | set(s5) - set(val))  # noqa: intentionally wrong pattern
    right = sorted((set(r1) | set(s5)) - set(val))
    assert right == []
    assert wrong == sorted(r1)  # documents the bug that caused ACCURACY_NO_GO false fail
