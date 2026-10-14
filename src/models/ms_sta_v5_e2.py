#!/usr/bin/env python3
"""MS-STA v5-E2 early motion-aware temporal reduction (student family).

Does not overwrite B4-E NO-GO. No test-split access.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_prep.joint_layout_v3 import N_JOINTS
from src.models.ms_sta_v5 import (
    CHIRALITY_VOCAB,
    CROSS_ATTENTION_MODE,
    PART_VOCAB,
    RESIDUAL_ARM_ID,
    RGB_DIM,
    RGB_REGIONS,
    CrossSkelToRgb,
    JointGraphEncoder,
    ReZeroGate,
    StreamGate,
    TemporalTransformer,
    masked_avg_pool_stride4,
    score_element_reduction,
)

VariantE2 = Literal["e2_s2", "e2_lite_s2", "e2_lite_s4", "e1_late_s4"]

# pose=12, face=21, LH=21, RH=21 (joint_layout_v3 order used in Stage-A buffers)
_PART_SLICES = (
    ("pose", slice(0, 12)),
    ("face", slice(12, 33)),
    ("left_hand", slice(33, 54)),
    ("right_hand", slice(54, 75)),
)


class MotionAwareEarlyReducer(nn.Module):
    """Reduce Skeleton length *before* JointGraphEncoder.

    Within each stride window, keep the frame with highest motion energy among
    valid frames (validity-aware). Also emits a cheap full-rate part summary.
    """

    def __init__(self, stride: int, d_model: int) -> None:
        super().__init__()
        if stride < 2:
            raise ValueError("stride_must_ge_2")
        self.stride = int(stride)
        self.part_proj = nn.Linear(3 * 4, d_model)

    def forward(
        self,
        coords: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
        skel_pad: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        # coords/velocity: (B,T,J,3); validity: (B,T,J); skel_pad True=invalid frame
        if coords.ndim != 4:
            raise ValueError(f"coords_rank_invalid:{coords.ndim}")
        bsz, t_len, n_j, _ = coords.shape
        if n_j != N_JOINTS:
            raise ValueError(f"n_joints_mismatch:{n_j}")
        if skel_pad is None:
            skel_pad = torch.zeros(bsz, t_len, dtype=torch.bool, device=coords.device)
            t_valid = torch.full((bsz,), t_len, device=coords.device, dtype=torch.long)
        else:
            t_valid = (~skel_pad).sum(dim=1).to(torch.long)

        # Motion energy per frame (validity-weighted).
        speed = velocity.norm(dim=-1)  # (B,T,J)
        joint_valid = validity > 0.5
        energy = (speed * joint_valid.float()).sum(dim=-1)  # (B,T)
        energy = energy.masked_fill(skel_pad, -1.0)

        pad_n = (self.stride - (t_len % self.stride)) % self.stride
        if pad_n:
            coords = F.pad(coords, (0, 0, 0, 0, 0, pad_n), value=0.0)
            velocity = F.pad(velocity, (0, 0, 0, 0, 0, pad_n), value=0.0)
            validity = F.pad(validity, (0, 0, 0, pad_n), value=0.0)
            skel_pad = F.pad(skel_pad, (0, pad_n), value=True)
            energy = F.pad(energy, (0, pad_n), value=-1.0)
        t_pad = coords.size(1)
        n_groups = t_pad // self.stride
        e_g = energy.view(bsz, n_groups, self.stride)
        # argmax over window; if all -1, pick 0 (group becomes pad).
        idx = e_g.argmax(dim=2)  # (B,G)
        c_g = coords.view(bsz * n_groups, self.stride, n_j, 3)
        v_g = velocity.view(bsz * n_groups, self.stride, n_j, 3)
        val_g = validity.view(bsz * n_groups, self.stride, n_j)
        pad_g = skel_pad.view(bsz, n_groups, self.stride)
        flat = torch.arange(bsz * n_groups, device=coords.device)
        take = idx.reshape(-1)
        coords_r = c_g[flat, take].view(bsz, n_groups, n_j, 3)
        velocity_r = v_g[flat, take].view(bsz, n_groups, n_j, 3)
        validity_r = val_g[flat, take].view(bsz, n_groups, n_j)
        # group pad if all frames invalid
        pooled_pad = pad_g.all(dim=2)

        # Full-rate part summary (cheap): mean valid coords per part over time.
        part_feats: list[torch.Tensor] = []
        frame_ok = (~skel_pad).float().unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
        for _name, sl in _PART_SLICES:
            c = coords[:, :, sl, :]
            jv = (validity[:, :, sl] > 0.5).float().unsqueeze(-1)
            w = frame_ok * jv
            denom = w.sum(dim=(1, 2)).clamp_min(1.0)
            part_feats.append((c * w).sum(dim=(1, 2)) / denom)
        part_cat = torch.cat(part_feats, dim=-1)  # (B,12)
        part_summary = self.part_proj(part_cat)

        red = score_element_reduction(t_valid, self.stride)
        if isinstance(red, float):
            red_t = torch.full((bsz,), red, device=coords.device)
        else:
            red_t = red
        stats = {
            "t_valid": t_valid,
            "t_opt": torch.ceil(t_valid.to(torch.float32) / float(self.stride)).to(torch.long),
            "score_reduction": red_t,
            "stride": torch.full((bsz,), self.stride, device=coords.device, dtype=torch.long),
        }
        return coords_r, velocity_r, validity_r, pooled_pad, part_summary, stats


class MSSTAv5E2(nn.Module):
    """B4* student with early reduction and optional lite width/layers."""

    def __init__(
        self,
        variant: VariantE2,
        *,
        num_classes: int = 400,
        dropout: float = 0.1,
        alpha_max: float = 1.0,
    ) -> None:
        super().__init__()
        if variant not in {"e2_s2", "e2_lite_s2", "e2_lite_s4", "e1_late_s4"}:
            raise ValueError(f"unknown_e2_variant:{variant}")
        self.variant = variant
        self.residual_arm_id = RESIDUAL_ARM_ID
        self.cross_attention_mode = CROSS_ATTENTION_MODE
        lite = variant.startswith("e2_lite")
        self.d_model = 192 if lite else 256
        self.temporal_layers = 1 if lite else 2
        self.nhead = 4
        if variant in {"e2_s2", "e2_lite_s2"}:
            self.stride = 2
            self.early = True
        elif variant == "e2_lite_s4":
            self.stride = 4
            self.early = True
        else:
            self.stride = 4
            self.early = False  # e1_late_s4

        d_model = self.d_model
        self.rgb_proj = nn.ModuleDict({r: nn.Linear(RGB_DIM, d_model) for r in RGB_REGIONS})
        self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)
        self.skel_encoder = JointGraphEncoder(d_model=d_model)
        self.skel_temporal = TemporalTransformer(d_model, self.nhead, self.temporal_layers, dropout)
        self.rgb_temporal = TemporalTransformer(d_model, self.nhead, self.temporal_layers, dropout)
        self.cross = CrossSkelToRgb(d_model, self.nhead)
        self.gate = StreamGate(d_model)
        self.rezero = ReZeroGate(alpha_max=alpha_max)
        self.update_ln = nn.LayerNorm(d_model)
        self.base_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.early_reducer = MotionAwareEarlyReducer(self.stride, d_model) if self.early else None
        self.part_fuse = nn.Linear(d_model * 2, d_model) if self.early else None

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

    def _encode_rgb_regions(self, rgb_tokens: dict[str, Any], rgb_pad: torch.Tensor | None) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for i, r in enumerate(RGB_REGIONS):
            tok = self.rgb_proj[r](rgb_tokens[f"token_{r}"])
            miss = rgb_tokens.get(f"missing_{r}")
            if miss is not None:
                tok = tok.masked_fill(miss.unsqueeze(-1), 0.0)
            tok = tok + self.region_emb.weight[i]
            out[r] = tok
        if rgb_pad is not None:
            for r in RGB_REGIONS:
                out[r] = out[r].masked_fill(rgb_pad.unsqueeze(-1), 0.0)
        return out

    def forward(self, batch: dict[str, Any], *, alpha_override: float | None = None) -> dict[str, torch.Tensor]:
        rgb_tokens = {f"token_{r}": batch[f"token_{r}"] for r in RGB_REGIONS}
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
        if global_tok.size(1) < 2:
            raise ValueError("temporal_attention_requires_L_ge_2")

        complexity: dict[str, torch.Tensor] | None = None
        part_summary = None
        if self.early:
            assert self.early_reducer is not None
            coords, velocity, validity, skel_pad, part_summary, complexity = self.early_reducer(
                coords, velocity, validity, skel_pad
            )
            skel_frames = self.skel_encoder(coords, velocity, validity, part_ids, chir_ids)
            if skel_pad is not None:
                skel_frames = skel_frames.masked_fill(skel_pad.unsqueeze(-1), 0.0)
        else:
            skel_frames = self.skel_encoder(coords, velocity, validity, part_ids, chir_ids)
            if skel_pad is not None:
                skel_frames = skel_frames.masked_fill(skel_pad.unsqueeze(-1), 0.0)
            skel_frames, skel_pad, complexity = masked_avg_pool_stride4(
                skel_frames, skel_pad, stride=self.stride
            )

        rgb_pool = self._pool_masked(global_tok, rgb_pad)
        skel_seq, skel_pool = self.skel_temporal(skel_frames, skel_pad)
        if part_summary is not None and self.part_fuse is not None:
            skel_pool = self.part_fuse(torch.cat([skel_pool, part_summary], dim=-1))
        h_base = self.base_fuse(torch.cat([rgb_pool, skel_pool], dim=-1))

        rgb_fused = torch.stack([regions[r] for r in RGB_REGIONS], dim=2).mean(dim=2)
        _rgb_seq, rgb_ctx = self.rgb_temporal(rgb_fused, rgb_pad)
        rgb_flat = torch.cat([regions[r] for r in RGB_REGIONS], dim=1)
        rgb_flat_pad = rgb_pad.repeat(1, len(RGB_REGIONS)) if rgb_pad is not None else None
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


def build_ms_sta_v5_e2(variant: str, **kwargs: Any) -> MSSTAv5E2:
    return MSSTAv5E2(variant=variant, **kwargs)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
