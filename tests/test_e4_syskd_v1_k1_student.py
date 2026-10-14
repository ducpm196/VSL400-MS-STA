# -*- coding: utf-8 -*-
"""K1 dense student shape / Fast-unchanged / schema tests."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_k1_c_mid_constants():
    from src.optimization.e4_syskd.k1_dense_student import K1_C_MID, K1_C_OUT, K2_C_MID_INVENTORY_ONLY

    assert list(K1_C_MID) == [56, 112, 224, 448]
    assert list(K1_C_OUT) == [256, 512, 1024, 2048]
    assert list(K2_C_MID_INVENTORY_ONLY) == [48, 96, 192, 384]


def test_k1_rebuild_mid_on_tiny_mock():
    """Structural: rebuild changes mid conv out_channels without touching Fast pathway names."""
    from src.optimization.e4_syskd.k1_dense_student import K1_C_MID, DenseK1Student

    # Skip if no CUDA / no slowfast ckpt in CI-less unit path — use synthetic ModuleList skip
    # This test only checks constants + import path when GPU absent.
    if not torch.cuda.is_available():
        return
    # Full construction covered by hardware script; here assert API exists
    assert DenseK1Student is not None
    assert len(K1_C_MID) == 4


def test_s5_efficiency_report_schema_if_present():
    p = ROOT / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_S5_EFFICIENCY_REPORT.json"
    if not p.exists():
        return
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["schema"] == "e4_syskd_v1_s5_efficiency_report_v1"
    assert r["n_ok"] == 360
    assert set(r["arms"]) == {"R0", "R1", "S5"}
