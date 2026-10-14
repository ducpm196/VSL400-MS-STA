"""MS-STA v6-E3 dual-stream student (RGB + Skeleton compressed).

Teacher: locked Stage-A b4_v6_ctr. Distill on train IDs only.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from src.models.ms_sta_v5_e2 import MotionAwareEarlyReducer
from src.models.ms_sta_v6_skel import CTRGCLiteEncoder

VariantE3 = Literal["e3_lite_s2", "e3_lite_s4", "e3_balanced_s2"]


class RgbTemporalReducer(nn.Module):
    """Cheap RGB window merge before temporal transformer (validity-aware avg)."""

    def __init__(self, stride: int) -> None:
        super().__init__()
        if stride < 2:
            raise ValueError("rgb_stride_must_ge_2")
        self.stride = int(stride)

    def forward(self, tok: torch.Tensor, pad: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # tok (B,L,C)
        bsz, length, dim = tok.shape
        pad_n = (self.stride - (length % self.stride)) % self.stride
        if pad_n:
            tok = F.pad(tok, (0, 0, 0, pad_n), value=0.0)
            if pad is None:
                pad = torch.zeros(bsz, length, dtype=torch.bool, device=tok.device)
            pad = F.pad(pad, (0, pad_n), value=True)
        elif pad is None:
            pad = torch.zeros(bsz, length, dtype=torch.bool, device=tok.device)
        length = tok.size(1)
        g = length // self.stride
        x = tok.view(bsz, g, self.stride, dim)
        w = (~pad).float().view(bsz, g, self.stride, 1)
        denom = w.sum(dim=2).clamp_min(1.0)
        pooled = (x * w).sum(dim=2) / denom
        pooled_pad = pad.view(bsz, g, self.stride).all(dim=2)
        return pooled, pooled_pad


class MSSTAv6E3(nn.Module):
    """Student: early skel reduce + CTR-lite + RGB temporal merge + lite width."""

    def __init__(
        self,
        variant: VariantE3,
        *,
        num_classes: int = 400,
        dropout: float = 0.1,
        alpha_max: float = 1.0,
    ) -> None:
        super().__init__()
        if variant not in {"e3_lite_s2", "e3_lite_s4", "e3_balanced_s2"}:
            raise ValueError(f"unknown_e3_variant:{variant}")
        self.variant = variant
        lite = variant.startswith("e3_lite")
        self.d_model = 160 if lite else 192
        self.temporal_layers = 1
        self.nhead = 4
        self.skel_stride = 4 if variant.endswith("s4") else 2
        self.rgb_stride = 2

        d_model = self.d_model
        self.rgb_proj = nn.ModuleDict({r: nn.Linear(RGB_DIM, d_model) for r in RGB_REGIONS})
        self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)
        self.rgb_reducer = RgbTemporalReducer(self.rgb_stride)
        self.rgb_temporal = TemporalTransformer(d_model, self.nhead, self.temporal_layers, dropout)

        self.early_reducer = MotionAwareEarlyReducer(self.skel_stride, d_model)
        self.skel_encoder = CTRGCLiteEncoder(d_model, layers=1, dropout=dropout)
        self.skel_temporal = TemporalTransformer(d_model, self.nhead, self.temporal_layers, dropout)
        self.part_fuse = nn.Linear(d_model * 2, d_model)

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

    def forward(self, batch: dict[str, Any], *, alpha_override: float | None = None) -> dict[str, torch.Tensor]:
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
        rgb_fused, rgb_pad2 = self.rgb_reducer(rgb_fused, rgb_pad)
        _rgb_seq, rgb_ctx = self.rgb_temporal(rgb_fused, rgb_pad2)
        rgb_flat = torch.cat(regions, dim=1)
        # Reduce flat RGB tokens with same temporal stride per region then cat.
        flat_parts = []
        flat_pads = []
        for tok in regions:
            t2, p2 = self.rgb_reducer(tok, rgb_pad)
            flat_parts.append(t2)
            flat_pads.append(p2)
        rgb_flat = torch.cat(flat_parts, dim=1)
        rgb_flat_pad = torch.cat(flat_pads, dim=1) if flat_pads[0] is not None else None

        coords, velocity, validity, skel_pad, part_summary, complexity = self.early_reducer(
            batch["coordinates"], batch["velocity"], batch["validity"], batch.get("skel_pad_mask")
        )
        skel_frames = self.skel_encoder(coords, velocity, validity, skel_pad)
        skel_seq, skel_pool = self.skel_temporal(skel_frames, skel_pad)
        skel_pool = self.part_fuse(torch.cat([skel_pool, part_summary], dim=-1))
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
            "skel_complexity": complexity,
        }


def build_ms_sta_v6_e3(variant: str, **kwargs: Any) -> MSSTAv6E3:
    return MSSTAv6E3(variant, **kwargs)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
