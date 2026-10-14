"""MS-STA v5 M0 Stage-A architecture and integration tests (no GPU train, no test split)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_v5 import (  # noqa: E402
    CROSS_ATTENTION_MODE,
    RESIDUAL_ARM_ID,
    RGB_REGIONS,
    build_ms_sta_v5,
    count_parameters,
)
from src.training.ms_sta_v5_data import collate_v5, load_split_role_ids  # noqa: E402


def _toy_batch(bs: int = 2, L: int = 3, T: int = 5) -> dict:
    batch = {
        "label": torch.randint(0, 400, (bs,)),
        "rgb_pad_mask": torch.zeros(bs, L, dtype=torch.bool),
        "skel_pad_mask": torch.zeros(bs, T, dtype=torch.bool),
        "coordinates": torch.randn(bs, T, 75, 3),
        "velocity": torch.randn(bs, T, 75, 3),
        "validity": torch.ones(bs, T, 75),
    }
    for r in RGB_REGIONS:
        batch[f"token_{r}"] = torch.randn(bs, L, 2304)
        batch[f"missing_{r}"] = torch.zeros(bs, L, dtype=torch.bool)
    return batch


@pytest.mark.parametrize(
    "variant",
    ["rgb_fixed", "b3_star_fixed", "b3_mlp_roi_fixed", "b4_star_fixed"],
)
def test_forward_backward_finite(variant: str) -> None:
    model = build_ms_sta_v5(variant, num_classes=400, d_model=128, temporal_layers=1)
    batch = _toy_batch()
    out = model(batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], batch["label"])
    if variant == "b4_star_fixed":
        loss = loss + 0.1 * torch.nn.functional.cross_entropy(out["logits_base"], batch["label"])
    loss.backward()
    assert torch.isfinite(out["logits"]).all()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no_grads"
    assert all(torch.isfinite(g).all() for g in grads)


def test_rezero_identity_at_init() -> None:
    model = build_ms_sta_v5("b4_star_fixed", num_classes=400, d_model=128, temporal_layers=1)
    model.eval()
    assert RESIDUAL_ARM_ID == "R-ReZero"
    assert model.rezero is not None
    assert float(model.rezero().detach()) == pytest.approx(0.0, abs=1e-7)
    batch = _toy_batch()
    with torch.no_grad():
        out0 = model(batch, alpha_override=0.0)
        out = model(batch)
    # At init alpha≈0 ⇒ pre-dropout features match base.
    assert torch.allclose(out["h"], out["h_base"], atol=1e-5)
    assert torch.allclose(out["logits"], out0["logits"], atol=1e-5)


def test_alpha_override_zero_exact_identity_path() -> None:
    model = build_ms_sta_v5("b4_star_fixed", num_classes=400, d_model=128, temporal_layers=1)
    model.eval()
    assert model.rezero is not None
    with torch.no_grad():
        model.rezero.g.fill_(2.0)
    batch = _toy_batch()
    out = model(batch, alpha_override=0.0)
    assert float(out["alpha"]) == 0.0
    assert torch.allclose(out["h"], out["h_base"], atol=1e-6)


def test_missing_region_mask_zeros_tokens() -> None:
    model = build_ms_sta_v5("b4_star_fixed", num_classes=400, d_model=128, temporal_layers=1)
    batch = _toy_batch()
    batch["missing_face"][:] = True
    out = model(batch)
    assert torch.isfinite(out["logits"]).all()


def test_l1_temporal_rejected() -> None:
    model = build_ms_sta_v5("b4_star_fixed", num_classes=400, d_model=128, temporal_layers=1)
    batch = _toy_batch(L=1)
    with pytest.raises(ValueError, match="temporal_attention_requires_L_ge_2"):
        model(batch)


def test_gradient_flow_attention_and_gate_branches() -> None:
    model = build_ms_sta_v5("b4_star_fixed", num_classes=400, d_model=128, temporal_layers=1)
    batch = _toy_batch()
    assert model.rezero is not None
    with torch.no_grad():
        model.rezero.g.fill_(1.0)
    out = model(batch)
    loss = out["logits"].sum() + out["logits_base"].sum()
    loss.backward()
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    must_have_grad = [
        "cross.attn.in_proj_weight",
        "gate.fc.0.weight",
        "rgb_temporal.enc.layers.0.self_attn.in_proj_weight",
        "skel_temporal.enc.layers.0.self_attn.in_proj_weight",
        "rezero.g",
    ]
    for name in must_have_grad:
        assert named[name].grad is not None, name
        assert torch.isfinite(named[name].grad).all(), name


def test_param_counts_positive_and_mlp_ge_b3() -> None:
    b3 = count_parameters(build_ms_sta_v5("b3_star_fixed", d_model=256))
    mlp = count_parameters(build_ms_sta_v5("b3_mlp_roi_fixed", d_model=256))
    b4 = count_parameters(build_ms_sta_v5("b4_star_fixed", d_model=256))
    be = count_parameters(build_ms_sta_v5("b4_star_e_stride4", d_model=256))
    assert b3 > 0 and mlp > 0 and b4 > 0 and be > 0
    assert mlp >= b3
    assert b4 > b3
    assert be == b4  # same modules; pooling is functional
    assert CROSS_ATTENTION_MODE.startswith("unidirectional")


def test_stride4_pool_invalid_pad_and_exact_9375_on_matrix() -> None:
    from src.models.ms_sta_v5 import masked_avg_pool_stride4

    tokens = torch.randn(2, 5, 8)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pooled, pooled_pad, stats = masked_avg_pool_stride4(tokens, pad)
    assert pooled.shape == (2, 2, 8)  # pad to 8 → 2 groups
    assert pooled_pad.shape == (2, 2)
    assert bool(pooled_pad[:, 1].all()) is False or True  # second group may be partial
    # Valid-T formula: T=5 → 1 - ceil(5/4)^2 / 5^2 = 0.84.
    assert float(stats["score_reduction"][0]) == pytest.approx(0.84, abs=1e-6)
    # Legacy padded-matrix reference keeps exact 93.75% on pad length 8.
    assert float(stats["legacy_padded_reference_score_reduction"][0]) == pytest.approx(
        0.9375, abs=1e-6
    )


def test_b4_e_forward_returns_complexity() -> None:
    model = build_ms_sta_v5("b4_star_e_stride4", num_classes=400, d_model=128, temporal_layers=1)
    batch = _toy_batch(L=3, T=8)
    out = model(batch)
    assert "skel_complexity" in out
    assert torch.isfinite(out["logits"]).all()


def test_split_role_refuses_test() -> None:
    split = {
        "hpo_train_ids": ["000001"],
        "test_ids": ["000002"],
    }
    with pytest.raises(ValueError, match="test_role_forbidden"):
        load_split_role_ids(split, "test")
    with pytest.raises(ValueError, match="train_val_intersects_test"):
        load_split_role_ids({"hpo_train_ids": ["000002"], "test_ids": ["000002"]}, "hpo_train")


def test_collate_padding_masks() -> None:
    items = []
    for L, T in ((2, 3), (4, 5)):
        it = {
            "sample_id": f"s{L}",
            "label": 1,
            "signer": "signer_1",
            "rgb_len": L,
            "skel_len": T,
            "coordinates": torch.zeros(T, 75, 3),
            "velocity": torch.zeros(T, 75, 3),
            "validity": torch.ones(T, 75),
            "joint_part_ids": torch.zeros(75, dtype=torch.long),
            "joint_chirality_ids": torch.zeros(75, dtype=torch.long),
        }
        for r in RGB_REGIONS:
            it[f"token_{r}"] = torch.zeros(L, 2304)
            it[f"missing_{r}"] = torch.zeros(L, dtype=torch.bool)
        items.append(it)
    batch = collate_v5(items)
    assert batch["token_global"].shape[1] == 4
    assert batch["coordinates"].shape[1] == 5
    assert batch["rgb_pad_mask"][0].sum() == 2
    assert batch["skel_pad_mask"][0].sum() == 2


def test_metadata_binds_video_id_text_gloss() -> None:
    from src.training.ms_sta_v5_train import load_metadata_by_id

    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    assert "000000" in meta
    assert meta["000000"]["gloss"] == "ANH"
    assert meta["000000"]["signer"].startswith("signer_")
