"""Gate M2 — MS-STA forward / shape / attention / temporal-order tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_wordlevel import (
    JointSpatialAttention,
    MSSTAWordLevel,
    SinusoidalPositionalEncoding,
    build_ms_sta,
)


@pytest.mark.parametrize("variant", ["b0", "b1", "b2", "b3", "b4", "b4r", "b5"])
def test_ms_sta_forward_shapes(variant: str) -> None:
    model = build_ms_sta(variant, num_classes=400)  # type: ignore[arg-type]
    model.eval()
    b, t, l = 2, 32, 4
    rgb = torch.randn(b, l, 2304)
    skel = torch.randn(b, t, 75, 3)
    with torch.no_grad():
        if variant in ("b0", "b1"):
            logits = model(rgb=rgb)
        elif variant == "b2":
            logits = model(skel=skel)
        else:
            logits = model(rgb=rgb, skel=skel)
    assert logits.shape == (b, 400)


def test_b4r_zero_delta_head_matches_b3_at_init() -> None:
    """Default zero-init last Linear of cls: logits == B3 at init (delta output is 0)."""
    torch.manual_seed(0)
    b3 = build_ms_sta("b3", num_classes=40).eval()
    torch.manual_seed(0)
    b4r = build_ms_sta("b4r", num_classes=40, residual_init="zero_delta_head").eval()
    with torch.no_grad():
        b4r.concat_cls.load_state_dict(b3.concat_cls.state_dict())
        assert float(b4r.residual_scale.item()) == 1.0
        last = b4r.cls[-1]
        assert torch.count_nonzero(last.weight).item() == 0
        assert torch.count_nonzero(last.bias).item() == 0
        rgb = torch.randn(2, 4, 2304)
        skel = torch.randn(2, 16, 75, 3)
        a = b3(rgb=rgb, skel=skel)
        b, attn = b4r(rgb=rgb, skel=skel, return_attention=True)
    assert torch.allclose(a, b, atol=1e-5)
    assert attn["residual_init"] == "zero_delta_head"
    assert float(attn["residual_scale"].item()) == 1.0


def test_b4r_zero_scale_legacy_matches_b3() -> None:
    """Legacy alpha=0 ablation still equals B3 but is not the default."""
    torch.manual_seed(0)
    b3 = build_ms_sta("b3", num_classes=40).eval()
    torch.manual_seed(0)
    b4r = build_ms_sta("b4r", num_classes=40, residual_init="zero_scale").eval()
    with torch.no_grad():
        b4r.concat_cls.load_state_dict(b3.concat_cls.state_dict())
        assert float(b4r.residual_scale.item()) == 0.0
        rgb = torch.randn(2, 4, 2304)
        skel = torch.randn(2, 16, 75, 3)
        a = b3(rgb=rgb, skel=skel)
        b, attn = b4r(rgb=rgb, skel=skel, return_attention=True)
    assert torch.allclose(a, b, atol=1e-5)
    assert attn["residual_init"] == "zero_scale"


def test_b4r_zero_delta_head_first_step_grads() -> None:
    """At init: cls delta head + B3 get grads; cross/gate wait until last Linear leaves zero."""
    torch.manual_seed(7)
    model = build_ms_sta("b4r", num_classes=40, residual_init="zero_delta_head")
    model.train()
    rgb = torch.randn(2, 4, 2304)
    skel = torch.randn(2, 16, 75, 3)
    y = torch.randint(0, 40, (2,))
    logits = model(rgb=rgb, skel=skel)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert float(model.cls[-1].weight.grad.abs().sum().item()) > 0.0
    assert float(model.concat_cls[-1].weight.grad.abs().sum().item()) > 0.0
    assert model.residual_scale.grad is not None
    assert float(model.cross.attn.in_proj_weight.grad.abs().sum().item()) == 0.0
    assert float(model.gate.fc[0].weight.grad.abs().sum().item()) == 0.0

    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    opt.step()
    opt.zero_grad()
    logits2 = model(rgb=rgb, skel=skel)
    loss2 = torch.nn.functional.cross_entropy(logits2, y)
    loss2.backward()
    assert float(model.cross.attn.in_proj_weight.grad.abs().sum().item()) > 0.0
    assert float(model.gate.fc[0].weight.grad.abs().sum().item()) > 0.0


def test_b4r_zero_scale_first_step_blocks_attention_grads() -> None:
    """Documents the blocked legacy design: alpha=0 yields no attention-path grads."""
    torch.manual_seed(7)
    model = build_ms_sta("b4r", num_classes=40, residual_init="zero_scale")
    model.train()
    rgb = torch.randn(2, 4, 2304)
    skel = torch.randn(2, 16, 75, 3)
    y = torch.randint(0, 40, (2,))
    logits = model(rgb=rgb, skel=skel)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert model.cls[-1].weight.grad is None or float(model.cls[-1].weight.grad.abs().sum()) == 0.0
    assert model.cross.attn.in_proj_weight.grad is None or float(
        model.cross.attn.in_proj_weight.grad.abs().sum()
    ) == 0.0
    assert model.gate.fc[0].weight.grad is None or float(model.gate.fc[0].weight.grad.abs().sum()) == 0.0
    assert model.concat_cls[-1].weight.grad is not None
    assert float(model.concat_cls[-1].weight.grad.abs().sum().item()) > 0.0


def test_b4r_param_count_matches_b4_plus_one_scale() -> None:
    """zero_delta_head reuses cls — no duplicate delta_cls head."""
    b4 = build_ms_sta("b4", num_classes=400)
    b4r = build_ms_sta("b4r", num_classes=400, residual_init="zero_delta_head")
    n_b4 = sum(p.numel() for p in b4.parameters())
    n_b4r = sum(p.numel() for p in b4r.parameters())
    assert n_b4r == n_b4 + 1  # residual_scale scalar only
    assert not any(k.startswith("delta_cls.") for k in b4r.state_dict())


def test_legacy_b4_state_dict_excludes_residual_modules() -> None:
    b4 = build_ms_sta("b4", num_classes=20)
    keys = b4.state_dict().keys()
    assert "residual_scale" not in keys
    assert not any(k.startswith("delta_cls.") for k in keys)
    b4r = build_ms_sta("b4r", num_classes=20, residual_init="zero_delta_head")
    assert "residual_scale" in b4r.state_dict()
    assert not any(k.startswith("delta_cls.") for k in b4r.state_dict())


def test_b4r_rejects_unknown_residual_init() -> None:
    with pytest.raises(ValueError, match="residual_init"):
        build_ms_sta("b4r", residual_init="bogus")


def test_b1_rejects_pooled_l1() -> None:
    model = build_ms_sta("b1", num_classes=400)
    with pytest.raises(ValueError, match="temporal length"):
        model(rgb=torch.randn(2, 2304))


def test_b4_rejects_rgb_l1() -> None:
    model = build_ms_sta("b4", num_classes=400)
    with pytest.raises(ValueError, match="valid L"):
        model(rgb=torch.randn(1, 1, 2304), skel=torch.randn(1, 16, 75, 3))


def test_positional_encoding_changes_with_order() -> None:
    pe = SinusoidalPositionalEncoding(32, max_len=16)
    x = torch.randn(1, 8, 32)
    y = pe(x)
    # Reversed sequence must differ after PE (order sensitivity)
    y_rev = pe(torch.flip(x, dims=[1]))
    assert not torch.allclose(y, torch.flip(y_rev, dims=[1]), atol=1e-6)


def test_spatial_attention_not_uniform() -> None:
    torch.manual_seed(0)
    layer = JointSpatialAttention()
    # Force scorer to prefer large |coord| joints (deterministic, not init-luck)
    with torch.no_grad():
        layer.projection[0].weight.copy_(torch.eye(64, 3))
        layer.projection[0].bias.zero_()
        layer.w_s.weight.copy_(torch.ones(1, 64))
    x = torch.zeros(1, 4, 75, 3)
    x[:, :, 0, :] = 5.0
    with torch.no_grad():
        _, attn = layer(x)
        assert attn.shape == (1, 4, 75)
        assert float(attn[0, 0, 0].detach()) > float(attn[0, 0, 1].detach())
        assert float(attn[0, 0].std().detach()) > 1e-4


def test_ms_sta_attention_pack_real_spatial() -> None:
    model = build_ms_sta("b4", num_classes=400)
    with torch.no_grad():
        logits, attn = model(
            rgb=torch.randn(1, 4, 2304),
            skel=torch.randn(1, 24, 75, 3),
            return_attention=True,
        )
        assert logits.shape == (1, 400)
        assert "spatial_joints" in attn
        assert attn["spatial_joints"].shape[-1] == 75
        # Not forced uniform 1/75
        assert float(attn["spatial_joints"][0, 0].std().detach()) > 1e-6
        assert attn["rgb_len"] == 4
        assert "cross" in attn
        assert attn["cross"].shape[-1] == 4  # RGB key length L


def test_skeleton_stride4_reduces_temporal_tokens_only() -> None:
    model = build_ms_sta("b4", num_classes=400, skel_temporal_stride=4).eval()
    with torch.no_grad():
        logits, attn = model(
            rgb=torch.randn(1, 4, 2304),
            skel=torch.randn(1, 64, 75, 3),
            return_attention=True,
        )
    assert logits.shape == (1, 400)
    assert attn["spatial_joints"].shape == (1, 64, 75)
    assert attn["skel_len"] == 16
    assert attn["temporal_skel"].shape[-1] == 16
    assert attn["cross"].shape[-2:] == (16, 4)


def test_skeleton_stride1_preserves_default_state_and_output() -> None:
    torch.manual_seed(9)
    default = build_ms_sta("b4", num_classes=20).eval()
    torch.manual_seed(9)
    explicit = build_ms_sta("b4", num_classes=20, skel_temporal_stride=1).eval()
    assert default.state_dict().keys() == explicit.state_dict().keys()
    rgb = torch.randn(1, 3, 2304)
    skel = torch.randn(1, 12, 75, 3)
    with torch.no_grad():
        a = default(rgb=rgb, skel=skel)
        b = explicit(rgb=rgb, skel=skel)
    assert torch.equal(a, b)


def test_skeleton_stride_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        build_ms_sta("b4", skel_temporal_stride=0)


def test_ms_sta_missing_stream_errors() -> None:
    model = build_ms_sta("b4", num_classes=400)
    with pytest.raises(ValueError):
        model(rgb=torch.randn(1, 4, 2304))
    with pytest.raises(ValueError):
        model(skel=torch.randn(1, 16, 75, 3))
    with pytest.raises(ValueError):
        model()


def test_load_b0_head_from_isolated_state() -> None:
    from src.models.isolated_sign_classifier import IsolatedSignHead

    head = IsolatedSignHead(num_classes=400).eval()
    model = build_ms_sta("b0", num_classes=400).eval()
    model.load_b0_head(head.state_dict())
    x = torch.randn(2, 2304)
    with torch.no_grad():
        a, _ = head(x)
        b = model(rgb=x)
    assert torch.allclose(a, b, atol=1e-5)
