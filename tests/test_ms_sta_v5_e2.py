"""Unit tests for MS-STA v5-E2 early reducer and score formula."""

from __future__ import annotations

import math

import pytest
import torch

from src.models.ms_sta_v5 import score_element_reduction
from src.models.ms_sta_v5_e2 import MotionAwareEarlyReducer, build_ms_sta_v5_e2


def _toy_batch(*, b: int = 2, t: int = 17, l: int = 8) -> dict:
    return {
        "token_global": torch.randn(b, l, 2304),
        "token_left_hand": torch.randn(b, l, 2304),
        "token_right_hand": torch.randn(b, l, 2304),
        "token_face": torch.randn(b, l, 2304),
        "rgb_pad_mask": torch.zeros(b, l, dtype=torch.bool),
        "coordinates": torch.randn(b, t, 75, 3),
        "velocity": torch.randn(b, t, 75, 3),
        "validity": torch.ones(b, t, 75),
        "skel_pad_mask": torch.zeros(b, t, dtype=torch.bool),
        "label": torch.zeros(b, dtype=torch.long),
    }


def test_score_element_reduction_t32_and_t33() -> None:
    assert score_element_reduction(32, 4) == pytest.approx(0.9375)
    expected_33 = 1.0 - (math.ceil(33 / 4) ** 2) / (33**2)
    assert score_element_reduction(33, 4) == pytest.approx(expected_33)
    t = torch.tensor([32, 33], dtype=torch.long)
    red = score_element_reduction(t, 4)
    assert isinstance(red, torch.Tensor)
    assert float(red[0]) == pytest.approx(0.9375)


def test_motion_reducer_picks_max_energy_valid_frame() -> None:
    red = MotionAwareEarlyReducer(stride=2, d_model=16)
    coords = torch.zeros(1, 4, 75, 3)
    velocity = torch.zeros(1, 4, 75, 3)
    # Frame 1 has higher motion than frame 0 in first window.
    velocity[0, 0, :, 0] = 1.0
    velocity[0, 1, :, 0] = 5.0
    velocity[0, 2, :, 0] = 2.0
    velocity[0, 3, :, 0] = 9.0
    validity = torch.ones(1, 4, 75)
    pad = torch.zeros(1, 4, dtype=torch.bool)
    coords[0, 1, 0, 0] = 11.0
    coords[0, 3, 0, 0] = 33.0
    c_r, _v_r, _val_r, pooled_pad, part_summary, stats = red(coords, velocity, validity, pad)
    assert c_r.shape == (1, 2, 75, 3)
    assert float(c_r[0, 0, 0, 0]) == pytest.approx(11.0)
    assert float(c_r[0, 1, 0, 0]) == pytest.approx(33.0)
    assert not bool(pooled_pad.any())
    assert part_summary.shape == (1, 16)
    assert int(stats["t_opt"][0]) == 2


def test_motion_reducer_all_missing_block_becomes_pad() -> None:
    red = MotionAwareEarlyReducer(stride=2, d_model=8)
    coords = torch.randn(1, 2, 75, 3)
    velocity = torch.randn(1, 2, 75, 3)
    validity = torch.zeros(1, 2, 75)
    pad = torch.ones(1, 2, dtype=torch.bool)
    _c, _v, _val, pooled_pad, _ps, stats = red(coords, velocity, validity, pad)
    assert bool(pooled_pad.all())
    assert int(stats["t_valid"][0]) == 0


def test_motion_reducer_deterministic_and_no_nan() -> None:
    torch.manual_seed(0)
    red = MotionAwareEarlyReducer(stride=4, d_model=8)
    batch = (
        torch.randn(2, 7, 75, 3),
        torch.randn(2, 7, 75, 3),
        torch.ones(2, 7, 75),
        torch.tensor([[False] * 5 + [True, True], [False] * 7]),
    )
    out1 = red(*batch)
    out2 = red(*batch)
    for a, b in zip(out1[:5], out2[:5]):
        assert torch.equal(a, b)
        assert torch.isfinite(a).all()


@pytest.mark.parametrize("variant", ["e2_s2", "e2_lite_s2", "e2_lite_s4", "e1_late_s4"])
def test_e2_forward_shapes_gradient(variant: str) -> None:
    model = build_ms_sta_v5_e2(variant, dropout=0.0)
    batch = _toy_batch(t=17 if "s4" in variant or variant == "e1_late_s4" else 9)
    out = model(batch)
    assert out["logits"].shape == (2, 400)
    assert torch.isfinite(out["logits"]).all()
    assert "skel_complexity" in out
    out["logits"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_e2_lite_has_fewer_params_than_full() -> None:
    full = build_ms_sta_v5_e2("e2_s2")
    lite = build_ms_sta_v5_e2("e2_lite_s2")
    n_full = sum(p.numel() for p in full.parameters())
    n_lite = sum(p.numel() for p in lite.parameters())
    assert n_lite < n_full


def test_teacher_logits_detached_in_kl_helper() -> None:
    from src.training.ms_sta_v5_train import _kl_logits

    s = torch.randn(4, 10, requires_grad=True)
    t = torch.randn(4, 10, requires_grad=True)
    loss = _kl_logits(s, t.detach(), 2.0)
    loss.backward()
    assert s.grad is not None
    assert t.grad is None
