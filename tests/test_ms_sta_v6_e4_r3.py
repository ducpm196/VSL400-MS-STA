"""Contract tests for MS-STA E4-R3 arms A/B/C (Phase 2)."""

from __future__ import annotations

import hashlib

import torch

from src.models.ms_sta_v6_e4_r3 import (
    E4R3_SEMANTIC_GROUPS,
    ArmE4R3,
    SemanticJointPool,
    arm_flags,
    assert_semantic_groups_policy,
    build_ms_sta_v6_e4_r3,
    checkpoint_identity,
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
        batch["validity"][:, :, 12:33] = 0.0
    if pad_last:
        batch["rgb_pad_mask"][:, -2:] = True
        batch["skel_pad_mask"][:, -3:] = True
    return batch


def test_semantic_groups_unique_and_full_coverage() -> None:
    assert_semantic_groups_policy()
    names = [n for n, _ in E4R3_SEMANTIC_GROUPS]
    assert len(names) == len(set(names))
    covered = [i for _, idxs in E4R3_SEMANTIC_GROUPS for i in idxs]
    assert sorted(covered) == list(range(75))
    assert len(covered) == 75


def test_e4_r3_arms_forward_shapes_dtype_and_finite() -> None:
    for arm in ("A", "B", "C"):
        m = build_ms_sta_v6_e4_r3(arm, dropout=0.0)
        out = m(_toy_batch())
        assert out["logits"].dtype == torch.float32
        assert out["logits"].shape == (2, 400)
        assert out["h_base"].shape == (2, 256)
        assert out["u"].shape == (2, 256)
        assert out["gate_weights"].shape[0] == 2
        assert torch.isfinite(out["logits"]).all()
        assert count_parameters(m) > 0


def test_e4_r3_all_missing_partial_and_padding() -> None:
    for arm in ("A", "B", "C"):
        m = build_ms_sta_v6_e4_r3(arm, dropout=0.0)
        for kwargs in (
            {"all_missing": True},
            {"partial_missing": True},
            {"pad_last": True},
        ):
            out = m(_toy_batch(**kwargs))
            assert torch.isfinite(out["logits"]).all(), (arm, kwargs)


def test_e4_r3_deterministic_forward() -> None:
    torch.manual_seed(0)
    m = build_ms_sta_v6_e4_r3("C", dropout=0.0).eval()
    batch = _toy_batch()
    with torch.inference_mode():
        a = m(batch)["logits"].clone()
        b = m(batch)["logits"].clone()
    assert torch.equal(a, b)


def test_e4_r3_rgb_compact_shortens_cross_tokens() -> None:
    batch = _toy_batch(l=8)
    m_a = build_ms_sta_v6_e4_r3("A", dropout=0.0)
    _, rgb_pad, rgb_flat = m_a._encode_rgb(batch)
    assert rgb_flat.shape == (2, 8, 256)
    assert rgb_pad is not None and rgb_pad.shape == (2, 8)

    m_b = build_ms_sta_v6_e4_r3("B", dropout=0.0)
    _, _, rgb_flat_b = m_b._encode_rgb(batch)
    assert rgb_flat_b.shape == (2, 32, 256)


def test_e4_r3_skel_compact_super_joints() -> None:
    m = build_ms_sta_v6_e4_r3("B", dropout=0.0)
    batch = _toy_batch(t=16)
    coords, vel, val = m.joint_pool(batch["coordinates"], batch["velocity"], batch["validity"])
    assert coords.shape == (2, 16, 5, 3)
    assert vel.shape == (2, 16, 5, 3)
    assert val.shape == (2, 16, 5)
    skel_seq, skel_pool, skel_pad = m._encode_skel(batch)
    assert skel_seq.shape == (2, 16, 256)
    assert skel_pool.shape == (2, 256)
    assert skel_pad is not None and skel_pad.shape == (2, 16)


def test_e4_r3_missing_joints_do_not_invent_observations() -> None:
    pool = SemanticJointPool()
    b, t = 1, 4
    coords = torch.randn(b, t, 75, 3)
    velocity = torch.randn(b, t, 75, 3)
    validity = torch.ones(b, t, 75)
    validity[:, :, 33:54] = 0.0  # left hand all missing
    c, v, val = pool(coords, velocity, validity)
    left_idx = [n for n, _ in E4R3_SEMANTIC_GROUPS].index("left_hand")
    assert torch.all(val[:, :, left_idx] == 0.0)
    assert torch.all(c[:, :, left_idx, :] == 0.0)
    assert torch.all(v[:, :, left_idx, :] == 0.0)


def test_e4_r3_padded_tokens_do_not_affect_pooled_output() -> None:
    m = build_ms_sta_v6_e4_r3("C", dropout=0.0).eval()
    base = _toy_batch(pad_last=True)
    mutated = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    mutated["token_global"][:, -2:] = torch.randn_like(mutated["token_global"][:, -2:])
    mutated["coordinates"][:, -3:] = torch.randn_like(mutated["coordinates"][:, -3:])
    mutated["velocity"][:, -3:] = torch.randn_like(mutated["velocity"][:, -3:])
    with torch.inference_mode():
        a = m(base)["logits"]
        b = m(mutated)["logits"]
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_e4_r3_gradients_reach_reducers_cross_and_gate() -> None:
    m = build_ms_sta_v6_e4_r3("C", dropout=0.0)
    out = m(_toy_batch())
    out["logits"].sum().backward()
    assert m.rgb_shared.down.weight.grad is not None
    assert m.region_reduce.score.weight.grad is not None
    assert any(p.grad is not None for p in m.skel_encoder.parameters())
    assert any(p.grad is not None for p in m.cross.parameters())
    assert any(p.grad is not None for p in m.gate.parameters())


def test_e4_r3_arm_flags_and_param_recompute() -> None:
    expected: dict[ArmE4R3, tuple[bool, bool]] = {
        "A": (True, False),
        "B": (False, True),
        "C": (True, True),
    }
    counts = {}
    for arm, vals in expected.items():
        f = arm_flags(arm)
        assert (f["rgb_compact"], f["skel_compact"]) == vals
        counts[arm] = count_parameters(build_ms_sta_v6_e4_r3(arm, dropout=0.0))
    assert counts["A"] > 0 and counts["B"] > 0 and counts["C"] > 0
    # Full C is not necessarily max params (shared compact paths differ), but recomputed stable.
    assert counts["A"] == count_parameters(build_ms_sta_v6_e4_r3("A", dropout=0.0))


def test_e4_r3_checkpoint_identity_binds_arm_config_policy_tree() -> None:
    ident = checkpoint_identity(
        arm="C",
        config={"d_model": 256, "rgb_rank": 384},
        policy_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
    )
    assert ident["arm"] == "C"
    assert ident["variant"] == "e4r3_C"
    assert ident["policy_sha256"] == "a" * 64
    assert ident["executable_tree_sha256"] == "b" * 64
    assert ident["forbid_grayscale_replicated_3ch"] is True
    raw = str(ident).encode()
    assert hashlib.sha256(raw).hexdigest()


def test_e4_r3_cuda_fp32_and_amp_if_available() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    m = build_ms_sta_v6_e4_r3("C", dropout=0.0).to(device).eval()
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in _toy_batch().items()}
    with torch.inference_mode():
        out_fp32 = m(batch)
        assert out_fp32["logits"].is_cuda
        assert torch.isfinite(out_fp32["logits"]).all()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out_amp = m(batch)
        assert torch.isfinite(out_amp["logits"].float()).all()
