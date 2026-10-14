# -*- coding: utf-8 -*-
"""Unit tests for E4_B4SC narrow SlowFast (GPU optional for shape-only via skip)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_training_policy_frozen():
    pol = json.loads((ROOT / "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V1_E2E3_TRAINING_POLICY.json").read_text(encoding="utf-8"))
    assert pol["status"] == "FROZEN_BEFORE_TRAINING"
    assert pol["decision_lock"]["scientific_baseline"] == "E0"
    assert pol["decision_lock"]["training_system_base"] == "E1a"
    assert pol["decision_lock"]["forbid_batched_window"] is True
    assert pol["architecture"]["E2_slow_widths"] == [224, 448, 896, 1792]
    assert pol["architecture"]["E3_slow_widths"] == [192, 384, 768, 1536]


def test_channel_topk_deterministic():
    from src.optimization.e4_b4sc.narrow_slowfast import select_topk_indices

    s = np.array([0.1, 0.5, 0.5, 0.2], dtype=np.float64)
    a = select_topk_indices(s, 2)
    b = select_topk_indices(s, 2)
    assert np.array_equal(a, b)
    assert set(a.tolist()) == {1, 2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_e2_fast_invariance_and_widths():
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4
    from src.optimization.e4_b4sc.narrow_slowfast import E2_WIDTHS, NarrowSlowFastBlock4, compute_channel_plan

    pol = json.loads((ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    if torch.cuda.get_device_name(0) != pol["scientific_environment_lock"]["required_gpu_name"]:
        pytest.skip("gpu mismatch")
    teacher = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    plan = compute_channel_plan(teacher, E2_WIDTHS)
    student = NarrowSlowFastBlock4(teacher, plan, device=device)
    slow = torch.randn(1, 3, 8, 224, 224, device=device)
    fast = torch.randn(1, 3, 32, 224, 224, device=device)
    with torch.inference_mode():
        tm = teacher.forward_maps(slow, fast)
        sm = student.forward_maps(slow, fast)
    assert tuple(sm.slow.shape) == (1, 1792, 8, 7, 7)
    assert tuple(sm.fast.shape) == tuple(tm.fast.shape)
    assert torch.equal(sm.fast, tm.fast)
