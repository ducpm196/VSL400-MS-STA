# -*- coding: utf-8 -*-
"""All-pad skeleton contract tests for E4-R4C (RGB/base fallback, no fake joints)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ms_sta_v6_e4_r4 import build_ms_sta_v6_e4_r4


def _batch(B: int, T: int = 32, H: int = 160, W: int = 160, J: int = 30, *, skel_all_pad: torch.Tensor) -> dict:
    """skel_all_pad: bool tensor (B,)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = torch.randn(B, T, H, W, 3, device=device)
    tvalid = torch.ones(B, T, dtype=torch.bool, device=device)
    coords = torch.randn(B, T, J, 3, device=device)
    vel = torch.randn(B, T, J, 3, device=device)
    validity = torch.ones(B, T, J, device=device)
    spad = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        if bool(skel_all_pad[i]):
            coords[i].zero_()
            vel[i].zero_()
            validity[i].zero_()
            spad[i] = True
    return {
        "rgb_tensor": rgb,
        "temporal_valid": tvalid,
        "coordinates": coords,
        "velocity": vel,
        "validity": validity,
        "skel_pad_mask": spad,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ms_sta_v6_e4_r4(rgb_id="R1", skel_layout="S_A", d_model=128).to(device)
    model.train()

    # 1) Pure all-pad batch: zeros + all-True mask, finite logits/loss/grad, RGB/base path
    skel_all = torch.ones(4, dtype=torch.bool)
    batch = _batch(4, skel_all_pad=skel_all)
    assert bool(batch["skel_pad_mask"].all())
    assert float(batch["coordinates"].abs().sum()) == 0.0
    out = model(batch)
    assert bool(out["skel_all_pad"].all())
    assert torch.isfinite(out["logits"]).all(), "logits_not_finite_all_pad"
    loss = torch.nn.functional.cross_entropy(out["logits"], torch.zeros(4, dtype=torch.long, device=device))
    assert torch.isfinite(loss), "loss_not_finite_all_pad"
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "grad_not_finite_all_pad"
    model.zero_grad(set_to_none=True)

    # 2) Changing padded skeleton values must not change all-pad outputs
    model.eval()
    with torch.no_grad():
        b0 = _batch(2, skel_all_pad=torch.tensor([True, True]))
        logits0 = model(b0)["logits"].clone()
        b1 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b0.items()}
        b1["coordinates"] = torch.randn_like(b1["coordinates"])
        b1["velocity"] = torch.randn_like(b1["velocity"])
        b1["validity"] = torch.rand_like(b1["validity"])
        # mask remains all-True
        assert bool(b1["skel_pad_mask"].all())
        logits1 = model(b1)["logits"]
        assert torch.allclose(logits0, logits1, atol=1e-5, rtol=1e-5), "padded_region_affects_all_pad_output"

    # 3) Mixed valid + all-pad batch works; all-pad matches RGB/base-only reference
    with torch.no_grad():
        mix_flag = torch.tensor([False, True, False, True])
        bm = _batch(4, skel_all_pad=mix_flag)
        out_m = model(bm)
        assert torch.isfinite(out_m["logits"]).all()
        assert bool((out_m["skel_all_pad"] == mix_flag.to(device)).all())
        # Reference: force those rows to all-pad zeros and compare
        bref = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bm.items()}
        for i in range(4):
            if bool(mix_flag[i]):
                bref["coordinates"][i].zero_()
                bref["velocity"][i].zero_()
                bref["validity"][i].zero_()
                bref["skel_pad_mask"][i] = True
        out_ref = model(bref)
        assert torch.allclose(out_m["logits"][mix_flag], out_ref["logits"][mix_flag], atol=1e-5, rtol=1e-5)

    # 4) Gate weights on all-pad rows are RGB-only [1,0]
    with torch.no_grad():
        b = _batch(3, skel_all_pad=torch.ones(3, dtype=torch.bool))
        gw = model(b)["gate_weights"]
        expect = torch.tensor([1.0, 0.0], device=device)
        assert torch.allclose(gw, expect.expand_as(gw), atol=1e-6)

    print("ALL_PAD_TESTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
