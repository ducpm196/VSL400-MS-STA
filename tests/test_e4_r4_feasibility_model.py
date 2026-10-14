"""Unit checks for E4-R4 feasibility model contracts (CPU-safe shapes)."""

from __future__ import annotations

import torch

from src.models.ms_sta_v6_e4_r4 import build_ms_sta_v6_e4_r4, count_parameters


def _batch(rgb_id: str, n_joints: int, device="cpu"):
    t = 8
    if rgb_id == "R2":
        rgb = torch.randn(1, t, 2, 64, 64, 3, device=device)
    elif rgb_id == "R3":
        rgb = torch.randn(1, t, 64, 64, 1, device=device)
    else:
        rgb = torch.randn(1, t, 64, 64, 3, device=device)
    return {
        "rgb_tensor": rgb,
        "temporal_valid": torch.ones(1, t, dtype=torch.bool, device=device),
        "coordinates": torch.randn(1, t, n_joints, 3, device=device),
        "velocity": torch.randn(1, t, n_joints, 3, device=device),
        "validity": torch.ones(1, t, n_joints, device=device),
        "skel_pad_mask": torch.zeros(1, t, dtype=torch.bool, device=device),
    }


def test_e4_r4_logits_shape_and_finite():
    for rgb, skel, nj in (("R0", "S_A", 30), ("R1", "S_B", 35), ("R2", "S_A", 30), ("R3", "S_B", 35)):
        m = build_ms_sta_v6_e4_r4(rgb_id=rgb, skel_layout=skel, d_model=64, temporal_layers=1, ctr_layers=1)
        m.eval()
        # override n_joints mismatch: S_A is 30, S_B 35 — use model's n_joints
        nj = m.n_joints
        b = _batch(rgb, nj)
        with torch.inference_mode():
            out = m(b)
        assert out["logits"].shape == (1, 400)
        assert torch.isfinite(out["logits"]).all()
        assert count_parameters(m) > 0


def test_pad_invariance_smoke():
    m = build_ms_sta_v6_e4_r4(rgb_id="R0", skel_layout="S_A", d_model=64, temporal_layers=1, ctr_layers=1)
    m.eval()
    b = _batch("R0", m.n_joints)
    b["skel_pad_mask"][0, -2:] = True
    b["temporal_valid"][0, -2:] = False
    with torch.inference_mode():
        a = m(b)["logits"].clone()
        b2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in b.items()}
        b2["coordinates"][:, -2:] = 99.0
        c = m(b2)["logits"]
    assert torch.allclose(a, c, atol=1e-4)
