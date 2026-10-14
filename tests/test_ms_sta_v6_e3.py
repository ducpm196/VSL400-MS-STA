"""Unit tests for MS-STA v6-E3 dual-stream student."""

from __future__ import annotations

import torch

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters as count_v6
from src.models.ms_sta_v6_e3 import build_ms_sta_v6_e3, count_parameters


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


def test_e3_variants_forward_and_smaller_than_teacher() -> None:
    teacher_n = count_v6(build_ms_sta_v6("b4_v6_ctr"))
    batch = _toy_batch()
    for v in ("e3_lite_s2", "e3_lite_s4", "e3_balanced_s2"):
        m = build_ms_sta_v6_e3(v, dropout=0.0)
        out = m(batch)
        assert out["logits"].shape == (2, 400)
        assert torch.isfinite(out["logits"]).all()
        out["logits"].sum().backward()
        assert count_parameters(m) < teacher_n
