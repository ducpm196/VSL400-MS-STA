"""MS-STA v6-E4 efficiency student — width/layer reduction (no temporal stride).

Teacher: Stage-A b4_v6_ctr. Distill logits only when d_model differs.
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
    ReZeroGate,
    StreamGate,
    TemporalTransformer,
)
from src.models.ms_sta_v6_skel import CTRGCLiteEncoder

VariantE4 = Literal["e4_w192", "e4_w192_t1", "e4_w160_t1"]


class MSSTAv6E4(nn.Module):
    """Compressed b4_v6_ctr: same topology, smaller width/depth."""

    def __init__(
        self,
        variant: VariantE4,
        *,
        num_classes: int = 400,
        dropout: float = 0.1,
        alpha_max: float = 1.0,
    ) -> None:
        super().__init__()
        if variant not in {"e4_w192", "e4_w192_t1", "e4_w160_t1"}:
            raise ValueError(f"unknown_e4_variant:{variant}")
        self.variant = variant
        if variant == "e4_w192":
            self.d_model, self.temporal_layers, self.ctr_layers = 192, 2, 2
        elif variant == "e4_w192_t1":
            self.d_model, self.temporal_layers, self.ctr_layers = 192, 1, 1
        else:
            self.d_model, self.temporal_layers, self.ctr_layers = 160, 1, 1
        self.nhead = 4
        d_model = self.d_model

        self.rgb_proj = nn.ModuleDict({r: nn.Linear(RGB_DIM, d_model) for r in RGB_REGIONS})
        self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)
        self.rgb_temporal = TemporalTransformer(d_model, self.nhead, self.temporal_layers, dropout)
        self.skel_encoder = CTRGCLiteEncoder(d_model, layers=self.ctr_layers, dropout=dropout)
        self.skel_temporal = TemporalTransformer(d_model, self.nhead, self.temporal_layers, dropout)
        self.cross = CrossSkelToRgb(d_model, self.nhead)
        self.gate = StreamGate(d_model)
        self.rezero = ReZeroGate(alpha_max=alpha_max)
        self.update_ln = nn.LayerNorm(d_model)
        self.base_fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout))
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
        rgb_fused = torch.stack(regions, dim=2).mean(dim=2)
        _seq, rgb_ctx = self.rgb_temporal(rgb_fused, rgb_pad)
        rgb_flat = torch.cat(regions, dim=1)
        rgb_flat_pad = rgb_pad.repeat(1, len(RGB_REGIONS)) if rgb_pad is not None else None
        return rgb_ctx, rgb_flat_pad, rgb_flat

    def forward(self, batch: dict[str, Any], *, alpha_override: float | None = None) -> dict[str, torch.Tensor]:
        rgb_ctx, rgb_flat_pad, rgb_flat = self._encode_rgb(batch)
        skel_pad = batch.get("skel_pad_mask")
        skel_frames = self.skel_encoder(batch["coordinates"], batch["velocity"], batch["validity"], skel_pad)
        skel_seq, skel_pool = self.skel_temporal(skel_frames, skel_pad)
        h_base = self.base_fuse(torch.cat([rgb_ctx, skel_pool], dim=-1))
        crossed = self.cross(skel_seq, rgb_flat, skel_pad=skel_pad, rgb_pad=rgb_flat_pad)
        crossed_pool = self._pool_masked(crossed, skel_pad)
        gated, gate_w = self.gate(rgb_ctx, crossed_pool)
        u = self.update_ln(gated)
        alpha = self.rezero(override=alpha_override)
        h_pre = h_base + alpha * u
        h = self.dropout(h_pre)
        logits = self.cls(h)
        logits_base = self.cls(self.dropout(h_base))
        return {
            "logits": logits,
            "logits_base": logits_base,
            "h": h_pre,
            "h_base": h_base,
            "alpha": alpha,
            "gate_weights": gate_w,
            "u": u,
        }


def build_ms_sta_v6_e4(variant: str, **kwargs: Any) -> MSSTAv6E4:
    return MSSTAv6E4(variant, **kwargs)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
