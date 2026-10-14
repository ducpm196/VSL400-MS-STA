"""Unit tests for MS-STA v6-E4 width/layer students."""

from __future__ import annotations

import torch

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters as count_v6
from src.models.ms_sta_v6_e4 import build_ms_sta_v6_e4, count_parameters


def _toy_batch(*, b: int = 2, t: int = 16, l: int = 8) -> dict:
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


def test_e4_variants_smaller_than_teacher() -> None:
    teacher_n = count_v6(build_ms_sta_v6("b4_v6_ctr"))
    batch = _toy_batch()
    for v in ("e4_w192", "e4_w192_t1", "e4_w160_t1"):
        m = build_ms_sta_v6_e4(v, dropout=0.0)
        out = m(batch)
        assert out["logits"].shape == (2, 400)
        assert torch.isfinite(out["logits"]).all()
        out["logits"].sum().backward()
        assert count_parameters(m) < teacher_n
