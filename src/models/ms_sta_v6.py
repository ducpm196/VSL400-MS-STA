"""MS-STA v6 Stage-A contrast matrix (finite teacher search on sealed features).

Variants: rgb_strong, b3_strong, b3_mlp_strong, b4_v6 (+ optional ctr/bi flags).
Stage-B RGB backbones are wired later via feature-root swap — not in this module.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from src.models.ms_sta_v5 import (
    CHIRALITY_VOCAB,
    PART_VOCAB,
    RGB_DIM,
    RGB_REGIONS,
    CrossSkelToRgb,
    JointGraphEncoder,
    ReZeroGate,
    StreamGate,
    TemporalTransformer,
)
from src.models.ms_sta_v6_skel import CTRGCLiteEncoder

VariantV6 = Literal[
    "rgb_strong",
    "b3_strong",
    "b3_mlp_strong",
    "b3_mlp_ctr",
    "b3_ctr_concat",
    "b4_v6",
    "b4_v6_ctr",
    "b4_v6_bi",
    "b4_v6_ctr_bi",
]


class CrossBidirectional(nn.Module):
    """Symmetric Skel↔RGB updates (predeclared alternative to one-way)."""

    def __init__(self, d_model: int = 256, nhead: int = 4) -> None:
        super().__init__()
        self.s2r = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.r2s = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)

    def forward(
        self,
        skel_tokens: torch.Tensor,
        rgb_tokens: torch.Tensor,
        *,
        skel_pad: torch.Tensor | None = None,
        rgb_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_out, _ = self.s2r(skel_tokens, rgb_tokens, rgb_tokens, key_padding_mask=rgb_pad, need_weights=False)
        r_out, _ = self.r2s(rgb_tokens, skel_tokens, skel_tokens, key_padding_mask=skel_pad, need_weights=False)
        if skel_pad is not None:
            s_out = s_out.masked_fill(skel_pad.unsqueeze(-1), 0.0)
        if rgb_pad is not None:
            r_out = r_out.masked_fill(rgb_pad.unsqueeze(-1), 0.0)
        return self.norm_s(skel_tokens + s_out), self.norm_r(rgb_tokens + r_out)


class MSSTAv6(nn.Module):
    def __init__(
        self,
        variant: VariantV6,
        *,
        num_classes: int = 400,
        d_model: int = 256,
        nhead: int = 4,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        alpha_max: float = 1.0,
        gate_entropy_reg: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in {
            "rgb_strong",
            "b3_strong",
            "b3_mlp_strong",
            "b3_mlp_ctr",
            "b3_ctr_concat",
            "b4_v6",
            "b4_v6_ctr",
            "b4_v6_bi",
            "b4_v6_ctr_bi",
        }:
            raise ValueError(f"unknown_v6_variant:{variant}")
        self.variant = variant
        self.d_model = d_model
        self.gate_entropy_reg = float(gate_entropy_reg)
        self.use_skel = variant != "rgb_strong"
        self.use_cross = variant.startswith("b4_v6")
        self.use_ctr = "ctr" in variant
        self.use_bi = variant.endswith("_bi") or variant == "b4_v6_ctr_bi"
        # CTR-matched controls: same skeleton encoder as b4_v6_ctr, no cross/gate/residual.
        self.use_mlp_fuse = variant in {"b3_mlp_strong", "b3_mlp_ctr"}

        self.rgb_proj = nn.ModuleDict({r: nn.Linear(RGB_DIM, d_model) for r in RGB_REGIONS})
        self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)
        self.rgb_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)

        if self.use_skel:
            if self.use_ctr:
                self.skel_encoder = CTRGCLiteEncoder(d_model, layers=2, dropout=dropout)
            else:
                self.skel_encoder = JointGraphEncoder(d_model=d_model)
            self.skel_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)
        else:
            self.skel_encoder = None
            self.skel_temporal = None

        if self.use_cross:
            if self.use_bi:
                self.cross = CrossBidirectional(d_model, nhead)
            else:
                self.cross = CrossSkelToRgb(d_model, nhead)
            self.gate = StreamGate(d_model)
            self.rezero = ReZeroGate(alpha_max=alpha_max)
            self.update_ln = nn.LayerNorm(d_model)
        else:
            self.cross = None
            self.gate = None
            self.rezero = None
            self.update_ln = None

        fuse_in = d_model if not self.use_skel else d_model * 2
        if self.use_mlp_fuse:
            self.base_fuse = nn.Sequential(
                nn.Linear(fuse_in, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.base_fuse = nn.Sequential(
                nn.Linear(fuse_in, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.cls = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

        part_ids = torch.tensor(
            [PART_VOCAB.get(p, 0) for p in (["pose"] * 12 + ["face"] * 21 + ["left_hand"] * 21 + ["right_hand"] * 21)],
            dtype=torch.long,
        )
        chir_ids = torch.tensor(
            [CHIRALITY_VOCAB.get(c, 0) for c in (["none"] * 12 + ["none"] * 21 + ["left"] * 21 + ["right"] * 21)],
            dtype=torch.long,
        )
        self.register_buffer("default_part_ids", part_ids, persistent=False)
        self.register_buffer("default_chirality_ids", chir_ids, persistent=False)

    def _pool_masked(self, x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        if pad is None:
            return x.mean(dim=1)
        w = (~pad).float().unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def _encode_rgb(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        rgb_pad = batch.get("rgb_pad_mask")
        regions = []
        for i, r in enumerate(RGB_REGIONS):
            tok = self.rgb_proj[r](batch[f"token_{r}"])
            miss = batch.get(f"missing_{r}")
            if miss is not None:
                tok = tok.masked_fill(miss.unsqueeze(-1), 0.0)
            tok = tok + self.region_emb.weight[i]
            if rgb_pad is not None:
                tok = tok.masked_fill(rgb_pad.unsqueeze(-1), 0.0)
            regions.append(tok)
        global_tok = regions[0]
        rgb_fused = torch.stack(regions, dim=2).mean(dim=2)
        _seq, rgb_ctx = self.rgb_temporal(rgb_fused, rgb_pad)
        rgb_flat = torch.cat(regions, dim=1)
        rgb_flat_pad = rgb_pad.repeat(1, len(RGB_REGIONS)) if rgb_pad is not None else None
        return rgb_ctx, rgb_flat_pad, rgb_flat

    def forward(self, batch: dict[str, Any], *, alpha_override: float | None = None) -> dict[str, torch.Tensor]:
        rgb_ctx, rgb_flat_pad, rgb_flat = self._encode_rgb(batch)
        rgb_pad = batch.get("rgb_pad_mask")
        rgb_pool = self._pool_masked(
            self.rgb_proj["global"](batch["token_global"]) + self.region_emb.weight[0],
            rgb_pad,
        )
        # Prefer temporal RGB context as pool for stronger RGB path.
        rgb_pool = rgb_ctx

        if not self.use_skel:
            h_base = self.base_fuse(rgb_pool)
            h = self.dropout(h_base)
            logits = self.cls(h)
            return {"logits": logits, "logits_base": logits, "h": h_base, "h_base": h_base, "alpha": torch.zeros((), device=h.device)}

        skel_pad = batch.get("skel_pad_mask")
        coords = batch["coordinates"]
        velocity = batch["velocity"]
        validity = batch["validity"]
        if self.use_ctr:
            assert isinstance(self.skel_encoder, CTRGCLiteEncoder)
            skel_frames = self.skel_encoder(coords, velocity, validity, skel_pad)
        else:
            assert isinstance(self.skel_encoder, JointGraphEncoder)
            skel_frames = self.skel_encoder(
                coords,
                velocity,
                validity,
                batch.get("joint_part_ids", self.default_part_ids),
                batch.get("joint_chirality_ids", self.default_chirality_ids),
            )
            if skel_pad is not None:
                skel_frames = skel_frames.masked_fill(skel_pad.unsqueeze(-1), 0.0)

        assert self.skel_temporal is not None
        skel_seq, skel_pool = self.skel_temporal(skel_frames, skel_pad)
        h_base = self.base_fuse(torch.cat([rgb_pool, skel_pool], dim=-1))

        if not self.use_cross:
            h = self.dropout(h_base)
            logits = self.cls(h)
            return {"logits": logits, "logits_base": logits, "h": h_base, "h_base": h_base, "alpha": torch.zeros((), device=h.device)}

        assert self.cross is not None and self.gate is not None and self.rezero is not None and self.update_ln is not None
        if self.use_bi:
            crossed_s, crossed_r = self.cross(skel_seq, rgb_flat, skel_pad=skel_pad, rgb_pad=rgb_flat_pad)
            crossed_pool = self._pool_masked(crossed_s, skel_pad)
            rgb_ctx2 = self._pool_masked(crossed_r, rgb_flat_pad)
            gated, gate_w = self.gate(rgb_ctx2, crossed_pool)
        else:
            crossed = self.cross(skel_seq, rgb_flat, skel_pad=skel_pad, rgb_pad=rgb_flat_pad)
            crossed_pool = self._pool_masked(crossed, skel_pad)
            gated, gate_w = self.gate(rgb_ctx, crossed_pool)
        u = self.update_ln(gated)
        alpha = self.rezero(override=alpha_override)
        h_pre = h_base + alpha * u
        h = self.dropout(h_pre)
        logits = self.cls(h)
        logits_base = self.cls(self.dropout(h_base))
        out = {
            "logits": logits,
            "logits_base": logits_base,
            "h": h_pre,
            "h_base": h_base,
            "alpha": alpha,
            "gate_weights": gate_w,
            "u": u,
        }
        if self.gate_entropy_reg > 0:
            # Encourage using both streams: high entropy of gate weights.
            p = gate_w.clamp_min(1e-8)
            ent = -(p * p.log()).sum(dim=-1).mean()
            out["gate_entropy"] = ent
            out["aux_reg"] = -self.gate_entropy_reg * ent
        return out


def build_ms_sta_v6(variant: str, **kwargs: Any) -> MSSTAv6:
    return MSSTAv6(variant, **kwargs)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
