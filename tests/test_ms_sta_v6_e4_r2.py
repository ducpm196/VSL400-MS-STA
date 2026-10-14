"""Unit tests for MS-STA E4-R2 architecture pilots A–E."""

from __future__ import annotations

import torch

from src.models.ms_sta_v6_e4_r2 import (
    ArmE4R2,
    arm_flags,
    build_ms_sta_v6_e4_r2,
    count_parameters,
)


def _toy_batch(
    *,
    b: int = 2,
    t: int = 16,
    l: int = 8,
    all_missing: bool = False,
    partial_missing: bool = False,
    pad_last: bool = False,
) -> dict:
    batch = {
        "token_global": torch.randn(b, l, 2304),
        "token_left_hand": torch.randn(b, l, 2304),
        "token_right_hand": torch.randn(b, l, 2304),
        "token_face": torch.randn(b, l, 2304),
        "missing_global": torch.zeros(b, l, dtype=torch.bool),
        "missing_left_hand": torch.zeros(b, l, dtype=torch.bool),
        "missing_right_hand": torch.zeros(b, l, dtype=torch.bool),
        "missing_face": torch.zeros(b, l, dtype=torch.bool),
        "rgb_pad_mask": torch.zeros(b, l, dtype=torch.bool),
        "coordinates": torch.randn(b, t, 75, 3),
        "velocity": torch.randn(b, t, 75, 3),
        "validity": torch.ones(b, t, 75),
        "skel_pad_mask": torch.zeros(b, t, dtype=torch.bool),
        "label": torch.zeros(b, dtype=torch.long),
    }
    if all_missing:
        for r in ("global", "left_hand", "right_hand", "face"):
            batch[f"missing_{r}"] = torch.ones(b, l, dtype=torch.bool)
        batch["validity"] = torch.zeros(b, t, 75)
    if partial_missing:
        batch["missing_face"] = torch.ones(b, l, dtype=torch.bool)
        batch["validity"][:, :, 12:33] = 0.0  # face joints
    if pad_last:
        batch["rgb_pad_mask"][:, -2:] = True
        batch["skel_pad_mask"][:, -3:] = True
    return batch


def test_e4_r2_arms_forward_shapes_and_finite() -> None:
    for arm in ("A", "B", "C", "D", "E"):
        m = build_ms_sta_v6_e4_r2(arm, dropout=0.0)
        out = m(_toy_batch())
        assert out["logits"].shape == (2, 400)
        assert out["h_base"].shape == (2, 256)
        assert out["u"].shape == (2, 256)
        assert torch.isfinite(out["logits"]).all()
        out["logits"].sum().backward()
        assert count_parameters(m) > 0
        flags = arm_flags(arm)  # type: ignore[arg-type]
        assert isinstance(flags, dict)


def test_e4_r2_all_missing_and_partial_and_padding() -> None:
    for arm in ("A", "B", "C", "D", "E"):
        m = build_ms_sta_v6_e4_r2(arm, dropout=0.0)
        for kwargs in (
            {"all_missing": True},
            {"partial_missing": True},
            {"pad_last": True},
        ):
            out = m(_toy_batch(**kwargs))
            assert torch.isfinite(out["logits"]).all(), (arm, kwargs)


def test_e4_r2_region_reduce_shortens_cross_tokens() -> None:
    m_b = build_ms_sta_v6_e4_r2("B", dropout=0.0)
    batch = _toy_batch(l=8)
    rgb_ctx, rgb_pad, rgb_flat = m_b._encode_rgb(batch)
    assert rgb_ctx.shape == (2, 256)
    assert rgb_flat.shape == (2, 8, 256)  # L, not 4L
    assert rgb_pad is not None and rgb_pad.shape == (2, 8)

    m_a = build_ms_sta_v6_e4_r2("A", dropout=0.0)
    _, _, rgb_flat_a = m_a._encode_rgb(batch)
    assert rgb_flat_a.shape == (2, 32, 256)  # 4L


def test_e4_r2_motion_reducer_fixed_length() -> None:
    m = build_ms_sta_v6_e4_r2("C", dropout=0.0, skel_keep_t=10)
    batch = _toy_batch(t=16)
    skel_seq, skel_pool, skel_pad = m._encode_skel(batch)
    assert skel_seq.shape[1] == 10
    assert skel_pool.shape == (2, 256)
    assert skel_pad is not None and skel_pad.shape == (2, 10)


def test_e4_r2_arm_flag_matrix() -> None:
    expected: dict[ArmE4R2, tuple[bool, bool, bool, bool]] = {
        "A": (True, False, False, False),
        "B": (False, True, False, False),
        "C": (False, False, True, False),
        "D": (False, False, False, True),
        "E": (True, True, True, True),
    }
    for arm, vals in expected.items():
        f = arm_flags(arm)
        assert (
            f["shared_low_rank_rgb"],
            f["region_reduce_4l_to_l"],
            f["motion_aware_skel_temporal"],
            f["separable_skel_graph_temporal"],
        ) == vals
