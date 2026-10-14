#!/usr/bin/env python3
"""MS-STA v5 Stage-A model family (B4*-fixed + matched controls).

Consumes RGB token v3 region tokens + Skeleton v3 arrays.
Does not edit legacy ms_sta_wordlevel B4/B4R. No test-split access here.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_prep.joint_layout_v3 import N_JOINTS

VariantV5 = Literal[
    "rgb_fixed",
    "b3_star_fixed",
    "b3_mlp_roi_fixed",
    "b4_star_fixed",
    "b4_star_e_stride4",
]
RGB_REGIONS = ("global", "left_hand", "right_hand", "face")
RGB_DIM = 2304
PART_VOCAB = {"pose": 0, "face": 1, "left_hand": 2, "right_hand": 3}
CHIRALITY_VOCAB = {"none": 0, "left": 1, "right": 2}
SKEL_STRIDE4 = 4

# Frozen M0 residual primary arm (R0 §4.1).
RESIDUAL_ARM_ID = "R-ReZero"
CROSS_ATTENTION_MODE = "unidirectional_skel_query_rgb_kv"


def score_element_reduction(t_valid: int | torch.Tensor, stride: int = SKEL_STRIDE4) -> float | torch.Tensor:
    """Actual Skeleton temporal-attention score reduction: 1 - ceil(T/s)^2 / T^2.

    Exact 0.9375 only when T is divisible by s (e.g. T=32,s=4). T=33 → ~0.9256.
    """
    if stride < 2:
        raise ValueError("stride_must_ge_2")
    if isinstance(t_valid, torch.Tensor):
        t = t_valid.to(torch.float32).clamp_min(1.0)
        t_opt = torch.ceil(t / float(stride))
        return 1.0 - (t_opt / t) ** 2
    t = max(int(t_valid), 1)
    t_opt = (t + stride - 1) // stride
    return 1.0 - (t_opt / t) ** 2


def masked_avg_pool_stride4(
    tokens: torch.Tensor,
    pad_mask: torch.Tensor | None,
    *,
    stride: int = SKEL_STRIDE4,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Masked average pool along time (B,T,D) with kernel=stride=4.

    Pads with *invalid* positions only (never repeats the final token).
    Output mask is invalid iff all source positions in the group are invalid.
    Returns (pooled_tokens, pooled_pad_mask, complexity_stats).

    Score accounting:
    - ``score_reduction`` / ``ref_score_elements`` use valid length T (formula
      ``1 - ceil(T/s)^2 / T^2``).
    - ``legacy_padded_reference_*`` retains the older padded-matrix baseline
      (do not overwrite sealed B4-E artifacts that used it).
    """
    if tokens.ndim != 3:
        raise ValueError(f"tokens_rank_invalid:{tokens.ndim}")
    bsz, t_len, dim = tokens.shape
    if stride < 2:
        raise ValueError("stride_must_ge_2")
    if pad_mask is None:
        t_valid = torch.full((bsz,), t_len, device=tokens.device, dtype=torch.long)
        pad_mask = torch.zeros(bsz, t_len, dtype=torch.bool, device=tokens.device)
    else:
        t_valid = (~pad_mask).sum(dim=1).to(torch.long)

    pad_n = (stride - (t_len % stride)) % stride
    if pad_n:
        tokens = F.pad(tokens, (0, 0, 0, pad_n), value=0.0)
        pad_mask = F.pad(pad_mask, (0, pad_n), value=True)  # True = invalid/pad

    t_pad = tokens.size(1)
    n_groups = t_pad // stride
    grouped = tokens.view(bsz, n_groups, stride, dim)
    grouped_pad = pad_mask.view(bsz, n_groups, stride)
    valid = (~grouped_pad).float().unsqueeze(-1)  # (B,G,S,1)
    denom = valid.sum(dim=2).clamp_min(1.0)
    pooled = (grouped * valid).sum(dim=2) / denom
    # Invalid group iff all sources invalid.
    pooled_pad = grouped_pad.all(dim=2)
    pooled = pooled.masked_fill(pooled_pad.unsqueeze(-1), 0.0)

    t_f = t_valid.to(torch.float32).clamp_min(1.0)
    t_opt_f = torch.ceil(t_f / float(stride))
    ref_scores = t_f ** 2
    opt_scores = t_opt_f ** 2
    reduction = 1.0 - (opt_scores / ref_scores)
    legacy_ref = torch.full((bsz,), float(t_pad * t_pad), device=tokens.device)
    legacy_opt = torch.full((bsz,), float(n_groups * n_groups), device=tokens.device)
    legacy_reduction = 1.0 - (legacy_opt / legacy_ref.clamp_min(1.0))
    stats = {
        "t_valid": t_valid,
        "t_opt": t_opt_f.to(torch.long),
        "t_matrix_legacy_padded": torch.full((bsz,), t_pad, device=tokens.device, dtype=torch.long),
        "ref_score_elements": ref_scores,
        "opt_score_elements": opt_scores,
        "score_reduction": reduction,
        "exact_93_75": bool(((t_valid % stride) == 0).all().item())
        and bool((reduction - 0.9375).abs().max().item() < 1e-6),
        "legacy_padded_reference_ref_score_elements": legacy_ref,
        "legacy_padded_reference_opt_score_elements": legacy_opt,
        "legacy_padded_reference_score_reduction": legacy_reduction,
    }
    return pooled, pooled_pad, stats



class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) < 2:
            raise ValueError("temporal_attention_requires_L_ge_2")
        h = self.norm(self.enc(self.pos(x), src_key_padding_mask=key_padding_mask))
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            pooled = (h * valid).sum(dim=1) / denom
        else:
            pooled = h.mean(dim=1)
        return h, pooled


class JointGraphEncoder(nn.Module):
    """Per-frame joint encode + spatial attention with validity mask."""

    def __init__(self, d_model: int = 256, d_joint: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(6, d_joint), nn.GELU())
        self.part_emb = nn.Embedding(4, d_joint)
        self.chirality_emb = nn.Embedding(3, d_joint)
        self.score = nn.Linear(d_joint, 1, bias=False)
        self.out = nn.Linear(d_joint, d_model)

    def forward(
        self,
        coords: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
        joint_part_ids: torch.Tensor,
        joint_chirality_ids: torch.Tensor,
    ) -> torch.Tensor:
        # coords/velocity: (B,T,N,3); validity: (B,T,N)
        x = torch.cat([coords, velocity], dim=-1)
        z = self.proj(x)
        z = z + self.part_emb(joint_part_ids).unsqueeze(0).unsqueeze(0)
        z = z + self.chirality_emb(joint_chirality_ids).unsqueeze(0).unsqueeze(0)
        scores = self.score(z) / math.sqrt(z.shape[-1])
        invalid = validity <= 0
        scores = scores.masked_fill(invalid.unsqueeze(-1), float("-inf"))
        # If an entire frame is invalid, fall back to zeros.
        all_bad = invalid.all(dim=-1, keepdim=True)
        attn = torch.softmax(scores, dim=-2)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.sum(attn * z, dim=-2)
        pooled = pooled.masked_fill(all_bad, 0.0)
        return self.out(pooled)  # (B,T,d)


class CrossSkelToRgb(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        skel_tokens: torch.Tensor,
        rgb_tokens: torch.Tensor,
        *,
        skel_pad: torch.Tensor | None = None,
        rgb_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.attn(
            skel_tokens,
            rgb_tokens,
            rgb_tokens,
            key_padding_mask=rgb_pad,
            need_weights=False,
        )
        if skel_pad is not None:
            out = out.masked_fill(skel_pad.unsqueeze(-1), 0.0)
        return self.norm(skel_tokens + out)


class StreamGate(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.fc(torch.cat([a, b], dim=-1)), dim=-1)
        return w[:, 0:1] * a + w[:, 1:2] * b, w


class ReZeroGate(nn.Module):
    """R-ReZero: alpha = alpha_max * tanh(g); g zero-init ⇒ alpha=0 at init."""

    def __init__(self, alpha_max: float = 1.0) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.g = nn.Parameter(torch.zeros(()))

    def forward(self, override: float | None = None) -> torch.Tensor:
        if override is not None:
            return torch.tensor(float(override), device=self.g.device, dtype=self.g.dtype)
        return self.alpha_max * torch.tanh(self.g)


class MSSTAv5StageA(nn.Module):
    def __init__(
        self,
        variant: VariantV5,
        *,
        num_classes: int = 400,
        d_model: int = 256,
        nhead: int = 4,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        alpha_max: float = 1.0,
        mlp_hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        if variant not in {
            "rgb_fixed",
            "b3_star_fixed",
            "b3_mlp_roi_fixed",
            "b4_star_fixed",
            "b4_star_e_stride4",
        }:
            raise ValueError(f"unknown_v5_variant:{variant}")
        self.variant = variant
        self.d_model = d_model
        self.residual_arm_id = RESIDUAL_ARM_ID
        self.cross_attention_mode = CROSS_ATTENTION_MODE
        self.skel_stride4 = variant == "b4_star_e_stride4"

        self.rgb_proj = nn.ModuleDict(
            {r: nn.Linear(RGB_DIM, d_model) for r in RGB_REGIONS}
        )
        self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)
        self.skel_encoder = JointGraphEncoder(d_model=d_model)
        needs_skel_temporal = variant in {
            "b3_star_fixed",
            "b3_mlp_roi_fixed",
            "b4_star_fixed",
            "b4_star_e_stride4",
        }
        self.skel_temporal = (
            TemporalTransformer(d_model, nhead, temporal_layers, dropout)
            if needs_skel_temporal
            else None
        )
        self.rgb_temporal = (
            TemporalTransformer(d_model, nhead, temporal_layers, dropout)
            if variant in {"b4_star_fixed", "b4_star_e_stride4"}
            else None
        )
        self.cross = (
            CrossSkelToRgb(d_model, nhead)
            if variant in {"b4_star_fixed", "b4_star_e_stride4"}
            else None
        )
        self.gate = StreamGate(d_model) if variant in {"b4_star_fixed", "b4_star_e_stride4"} else None
        self.rezero = (
            ReZeroGate(alpha_max=alpha_max)
            if variant in {"b4_star_fixed", "b4_star_e_stride4"}
            else None
        )
        self.update_ln = (
            nn.LayerNorm(d_model) if variant in {"b4_star_fixed", "b4_star_e_stride4"} else None
        )

        # B3* base head path: global RGB pool + skel pool
        self.base_fuse = (
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if needs_skel_temporal
            else None
        )
        # Param-matched MLP for ROI concat without attention (4 regions + skel)
        hidden = d_model * mlp_hidden_mult
        self.roi_mlp = (
            nn.Sequential(
                nn.Linear(d_model * 5, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if variant == "b3_mlp_roi_fixed"
            else None
        )
        self.cls = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Register default joint identity buffers (overwritten by batch if provided).
        part_ids = torch.tensor(
            [PART_VOCAB.get(p, 0) for p in (
                ["pose"] * 12 + ["face"] * 21 + ["left_hand"] * 21 + ["right_hand"] * 21
            )],
            dtype=torch.long,
        )
        chir_ids = torch.tensor(
            [CHIRALITY_VOCAB.get(c, 0) for c in (
                ["none"] * 12 + ["none"] * 21 + ["left"] * 21 + ["right"] * 21
            )],
            dtype=torch.long,
        )
        self.register_buffer("default_part_ids", part_ids, persistent=False)
        self.register_buffer("default_chirality_ids", chir_ids, persistent=False)

    def _pool_masked(self, tokens: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        if pad is None:
            return tokens.mean(dim=1)
        valid = (~pad).float().unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def _encode_rgb_regions(
        self,
        rgb: dict[str, torch.Tensor],
        rgb_pad: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for i, r in enumerate(RGB_REGIONS):
            tok = self.rgb_proj[r](rgb[f"token_{r}"])
            miss = rgb.get(f"missing_{r}")
            if miss is not None:
                tok = tok.masked_fill(miss.unsqueeze(-1), 0.0)
            tok = tok + self.region_emb.weight[i]
            out[r] = tok
        if rgb_pad is not None:
            for r in RGB_REGIONS:
                out[r] = out[r].masked_fill(rgb_pad.unsqueeze(-1), 0.0)
        return out

    def forward(
        self,
        batch: dict[str, Any],
        *,
        alpha_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        rgb_tokens = {
            f"token_{r}": batch[f"token_{r}"] for r in RGB_REGIONS
        }
        for r in RGB_REGIONS:
            if f"missing_{r}" in batch:
                rgb_tokens[f"missing_{r}"] = batch[f"missing_{r}"]
        rgb_pad = batch.get("rgb_pad_mask")
        skel_pad = batch.get("skel_pad_mask")
        coords = batch["coordinates"]
        velocity = batch["velocity"]
        validity = batch["validity"]
        part_ids = batch.get("joint_part_ids", self.default_part_ids)
        chir_ids = batch.get("joint_chirality_ids", self.default_chirality_ids)

        regions = self._encode_rgb_regions(rgb_tokens, rgb_pad)
        global_tok = regions["global"]
        b4_family = self.variant in {"b4_star_fixed", "b4_star_e_stride4"}
        if global_tok.size(1) < 2 and self.variant in {
            "b4_star_fixed",
            "b4_star_e_stride4",
            "b3_star_fixed",
            "b3_mlp_roi_fixed",
        }:
            if self.variant != "rgb_fixed":
                raise ValueError("temporal_attention_requires_L_ge_2")

        skel_frames = self.skel_encoder(coords, velocity, validity, part_ids, chir_ids)
        if skel_pad is not None:
            skel_frames = skel_frames.masked_fill(skel_pad.unsqueeze(-1), 0.0)

        complexity: dict[str, torch.Tensor] | None = None
        if self.skel_stride4:
            # Record original valid length before pad/pool.
            if skel_pad is not None:
                t_orig = (~skel_pad).sum(dim=1)
            else:
                t_orig = torch.full(
                    (skel_frames.size(0),),
                    skel_frames.size(1),
                    device=skel_frames.device,
                    dtype=torch.long,
                )
            skel_frames, skel_pad, complexity = masked_avg_pool_stride4(skel_frames, skel_pad)
            complexity = dict(complexity)
            complexity["t_orig_valid"] = t_orig
            complexity["reduction_from_valid_t"] = complexity["score_reduction"]

        rgb_pool = self._pool_masked(global_tok, rgb_pad)
        if self.variant == "rgb_fixed":
            h = self.dropout(rgb_pool)
            logits = self.cls(h)
            return {
                "logits": logits,
                "logits_base": logits,
                "h": h,
                "h_base": h,
                "alpha": torch.zeros((), device=h.device),
            }

        assert self.skel_temporal is not None and self.base_fuse is not None
        skel_seq, skel_pool = self.skel_temporal(skel_frames, skel_pad)
        h_base = self.base_fuse(torch.cat([rgb_pool, skel_pool], dim=-1))

        if self.variant == "b3_star_fixed":
            h = self.dropout(h_base)
            logits = self.cls(h)
            return {
                "logits": logits,
                "logits_base": logits,
                "h": h,
                "h_base": h_base,
                "alpha": torch.zeros((), device=h.device),
            }

        if self.variant == "b3_mlp_roi_fixed":
            assert self.roi_mlp is not None
            pools = [self._pool_masked(regions[r], rgb_pad) for r in RGB_REGIONS]
            feat = torch.cat(pools + [skel_pool], dim=-1)
            h = self.dropout(self.roi_mlp(feat))
            logits = self.cls(h)
            return {
                "logits": logits,
                "logits_base": self.cls(h_base.detach()),
                "h": h,
                "h_base": h_base,
                "alpha": torch.zeros((), device=h.device),
            }

        # b4_star_fixed / b4_star_e_stride4
        assert b4_family
        assert self.rgb_temporal is not None and self.cross is not None
        assert self.gate is not None and self.rezero is not None and self.update_ln is not None
        rgb_fused = torch.stack([regions[r] for r in RGB_REGIONS], dim=2).mean(dim=2)
        _rgb_seq, rgb_ctx = self.rgb_temporal(rgb_fused, rgb_pad)
        rgb_flat = torch.cat([regions[r] for r in RGB_REGIONS], dim=1)
        if rgb_pad is not None:
            rgb_flat_pad = rgb_pad.repeat(1, len(RGB_REGIONS))
        else:
            rgb_flat_pad = None
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
        if complexity is not None:
            out["skel_complexity"] = complexity
        return out


def build_ms_sta_v5(
    variant: VariantV5,
    num_classes: int = 400,
    **kwargs: Any,
) -> MSSTAv5StageA:
    return MSSTAv5StageA(variant, num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
