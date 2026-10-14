"""Shape checks for MS-STA word-level building blocks (VSL400 B4 path)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.ms_sta_wordlevel import (
    CrossStreamAttention,
    JointSpatialAttention,
    MSSTAWordLevel,
    StreamGate,
    TemporalTransformer,
)
from src.models.rgb_backbone import SLOWFAST_FEAT_DIM


def test_joint_spatial_attention_shape() -> None:
    x = torch.randn(2, 8, 75, 3)
    out, attn = JointSpatialAttention()(x)
    assert out.shape == (2, 8, 64)
    assert attn.shape == (2, 8, 75)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_temporal_transformer_pooling() -> None:
    x = torch.randn(2, 16, 256)
    tokens, pooled, attn = TemporalTransformer()(x)
    assert tokens.shape == (2, 16, 256)
    assert pooled.shape == (2, 256)
    assert attn.shape == (2, 1, 16)


def test_cross_and_gate_shapes() -> None:
    q = torch.randn(2, 16, 256)
    kv = torch.randn(2, 8, 256)
    fused, weights = CrossStreamAttention()(q, kv)
    assert fused.shape == (2, 16, 256)
    assert weights.shape[0] == 2
    gated, w = StreamGate()(torch.randn(2, 256), torch.randn(2, 256))
    assert gated.shape == (2, 256)
    assert w.shape == (2, 2)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_ms_sta_b4_forward_and_stride4() -> None:
    rgb = torch.randn(2, 3, SLOWFAST_FEAT_DIM)
    skel = torch.randn(2, 64, 75, 3)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[:, 2] = True  # last window padded/invalid

    model = MSSTAWordLevel(variant="b4", skel_temporal_stride=1)
    logits = model(rgb, skel, rgb_key_padding_mask=mask)
    assert logits.shape == (2, 400)

    model4 = MSSTAWordLevel(variant="b4", skel_temporal_stride=4)
    logits4 = model4(rgb, skel, rgb_key_padding_mask=mask)
    assert logits4.shape == (2, 400)


if __name__ == "__main__":
    test_joint_spatial_attention_shape()
    test_temporal_transformer_pooling()
    test_cross_and_gate_shapes()
    test_ms_sta_b4_forward_and_stride4()
    print("[OK] MS-STA word-level shape checks passed.")
