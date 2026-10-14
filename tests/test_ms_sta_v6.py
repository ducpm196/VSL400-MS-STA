"""Unit tests for MS-STA v6 Stage-A matrix + CTR-GC-lite."""

from __future__ import annotations

import torch

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters
from src.models.ms_sta_v6_skel import CTRGCLiteEncoder, build_adjacency_75


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


def test_adjacency_75_symmetric_normalized() -> None:
    a = build_adjacency_75()
    assert a.shape == (75, 75)
    assert torch.allclose(a, a.T, atol=1e-5)
    assert float(a.diag().min()) > 0


def test_ctr_encoder_shapes() -> None:
    enc = CTRGCLiteEncoder(d_model=64, layers=1)
    coords = torch.randn(2, 9, 75, 3)
    vel = torch.randn(2, 9, 75, 3)
    val = torch.ones(2, 9, 75)
    out = enc(coords, vel, val, None)
    assert out.shape == (2, 9, 64)
    assert torch.isfinite(out).all()


def test_v6_matrix_forward_and_param_near_match() -> None:
    batch = _toy_batch()
    params = {}
    for v in (
        "rgb_strong",
        "b3_strong",
        "b3_mlp_strong",
        "b3_mlp_ctr",
        "b3_ctr_concat",
        "b4_v6",
        "b4_v6_ctr",
        "b4_v6_bi",
    ):
        m = build_ms_sta_v6(v, dropout=0.0, d_model=128, temporal_layers=1)
        out = m(batch)
        assert out["logits"].shape == (2, 400)
        assert torch.isfinite(out["logits"]).all()
        out["logits"].sum().backward()
        params[v] = count_parameters(m)
        if v in {"b3_mlp_ctr", "b3_ctr_concat"}:
            assert m.use_ctr and not m.use_cross
    # Near-matched: mlp vs b4 within ~25% (finite engineering target, not equality).
    ratio = params["b3_mlp_strong"] / max(params["b4_v6"], 1)
    assert 0.75 <= ratio <= 1.25
    # CTR-matched control should sit near teacher param count (no cross arms).
    assert params["b3_mlp_ctr"] < params["b4_v6_ctr"]
    assert params["b3_ctr_concat"] < params["b4_v6_ctr"]
