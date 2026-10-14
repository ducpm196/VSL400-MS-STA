"""MS-STA v6 E4-R2 architecture pilots (research branch only).

Finite arms A–E for accuracy-preserving measured-latency optimization.
Does not mutate historical e4_w192 or B4-V6 teacher checkpoints.
External inputs remain RGB token v3 + Skeleton v3 caches (identical to B4).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_prep.joint_layout_v3 import N_JOINTS
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
from src.models.ms_sta_v6_skel import CTRGCLiteBlock, CTRGCLiteEncoder

ArmE4R2 = Literal["A", "B", "C", "D", "E"]

# Keep teacher width for architectural pilots (axis search: topology before width).
_DEFAULT_D = 256
_DEFAULT_NHEAD = 4
_DEFAULT_TEMPORAL_LAYERS = 2
_DEFAULT_RGB_RANK = 384
_DEFAULT_SKEL_KEEP_T = 16


class SharedLowRankRgbProj(nn.Module):
    """One shared low-rank projection over stacked [B,L,4,2304] regions."""

    def __init__(self, d_model: int = _DEFAULT_D, rank: int = _DEFAULT_RGB_RANK) -> None:
        super().__init__()
        self.rank = int(rank)
        self.down = nn.Linear(RGB_DIM, self.rank, bias=True)
        self.up = nn.Linear(self.rank, d_model, bias=True)
        self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: (B, L, 4, 2304) → (B, L, 4, d)
        b, l, r, _ = stacked.shape
        flat = stacked.reshape(b * l * r, RGB_DIM)
        h = self.up(F.gelu(self.down(flat))).view(b, l, r, -1)
        emb = self.region_emb.weight.view(1, 1, r, -1)
        return h + emb


class ValidityAwareRegionReduce(nn.Module):
    """Soft validity-aware gate: 4 region tokens → 1 fused token per time step."""

    def __init__(self, d_model: int = _DEFAULT_D) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        regions: torch.Tensor,
        *,
        missing: torch.Tensor | None,
        rgb_pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # regions: (B, L, 4, d); missing: (B, L, 4) True=missing
        scores = self.score(regions).squeeze(-1)  # (B, L, 4)
        if missing is not None:
            scores = scores.masked_fill(missing, float("-inf"))
        if rgb_pad is not None:
            scores = scores.masked_fill(rgb_pad.unsqueeze(-1), float("-inf"))
        all_bad = ~torch.isfinite(scores).any(dim=-1, keepdim=True)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        fused = (attn.unsqueeze(-1) * regions).sum(dim=2)  # (B, L, d)
        if rgb_pad is not None:
            fused = fused.masked_fill(rgb_pad.unsqueeze(-1), 0.0)
        fused = fused.masked_fill(all_bad, 0.0)
        return fused, rgb_pad


class MotionAwareTemporalReducer(nn.Module):
    """Fixed-length motion-aware frame selection (keep endpoints + high motion)."""

    def __init__(self, keep_t: int = _DEFAULT_SKEL_KEEP_T) -> None:
        super().__init__()
        if keep_t < 2:
            raise ValueError("keep_t_must_be_ge_2")
        self.keep_t = int(keep_t)

    def forward(
        self,
        frames: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
        skel_pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # frames: (B,T,d); velocity (B,T,J,3); validity (B,T,J)
        b, t, d = frames.shape
        keep = min(self.keep_t, t)
        if keep == t:
            return frames, skel_pad

        motion = (velocity.norm(dim=-1) * (validity > 0.5).float()).sum(dim=-1)  # (B,T)
        if skel_pad is not None:
            motion = motion.masked_fill(skel_pad, -1.0)
        # Boost endpoints so they always survive top-k when valid.
        boost = motion.detach().amax(dim=1, keepdim=True).clamp_min(0.0) + 1.0
        scores = motion.clone()
        scores[:, 0] = scores[:, 0] + boost.squeeze(1)
        scores[:, -1] = scores[:, -1] + boost.squeeze(1)

        idx = scores.topk(keep, dim=1).indices  # (B, keep)
        idx, _ = idx.sort(dim=1)
        gather_idx = idx.unsqueeze(-1).expand(b, keep, d)
        reduced = torch.gather(frames, 1, gather_idx)

        new_pad: torch.Tensor | None = None
        if skel_pad is not None:
            new_pad = torch.gather(skel_pad, 1, idx)
            # If a selected slot was pad (rare under boost), mark pad.
            reduced = reduced.masked_fill(new_pad.unsqueeze(-1), 0.0)
        return reduced, new_pad


class SeparableGraphTemporalBlock(nn.Module):
    """Spatial CTR-lite then depthwise temporal conv (EfficientGCN-inspired)."""

    def __init__(self, channels: int, *, dropout: float = 0.1, k_t: int = 3) -> None:
        super().__init__()
        self.spatial = CTRGCLiteBlock(channels, dropout=dropout)
        pad = k_t // 2
        self.temporal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(k_t, 1),
            padding=(pad, 0),
            groups=channels,
            bias=False,
        )
        self.temporal_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J)
        y = self.spatial(x)
        y = self.temporal_pw(self.temporal(y))
        y = self.bn(y)
        return self.drop(F.gelu(y + x))


class SeparableSkelEncoder(nn.Module):
    """Replace CTR×N + TemporalTransformer with separable graph-temporal stack."""

    def __init__(
        self,
        d_model: int = _DEFAULT_D,
        *,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(6, d_model)
        self.blocks = nn.ModuleList(
            [SeparableGraphTemporalBlock(d_model, dropout=dropout) for _ in range(layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)
        self.register_buffer("_n_joints", torch.tensor(N_JOINTS), persistent=False)

    def forward(
        self,
        coords: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
        skel_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([coords, velocity], dim=-1)
        h = self.in_proj(x)
        if validity is not None:
            h = h * (validity > 0.5).float().unsqueeze(-1)
        h = h.permute(0, 3, 1, 2).contiguous()
        for blk in self.blocks:
            h = blk(h)
        h = h.permute(0, 2, 3, 1).contiguous()
        w = (validity > 0.5).float().unsqueeze(-1)
        pooled = (h * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        if skel_pad is not None:
            pooled = pooled.masked_fill(skel_pad.unsqueeze(-1), 0.0)
        return self.out_ln(pooled)


class MSSTAv6E4R2(nn.Module):
    """B4-topology teacher-compatible student with optional E4-R2 pilots."""

    def __init__(
        self,
        arm: ArmE4R2,
        *,
        num_classes: int = 400,
        d_model: int = _DEFAULT_D,
        nhead: int = _DEFAULT_NHEAD,
        temporal_layers: int = _DEFAULT_TEMPORAL_LAYERS,
        dropout: float = 0.1,
        alpha_max: float = 1.0,
        rgb_rank: int = _DEFAULT_RGB_RANK,
        skel_keep_t: int = _DEFAULT_SKEL_KEEP_T,
        ctr_layers: int = 2,
    ) -> None:
        super().__init__()
        if arm not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"unknown_e4_r2_arm:{arm}")
        self.arm = arm
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.use_shared_rgb = arm in {"A", "E"}
        self.use_region_reduce = arm in {"B", "E"}
        self.use_motion_reduce = arm in {"C", "E"}
        self.use_separable_skel = arm in {"D", "E"}

        if self.use_shared_rgb:
            self.rgb_shared = SharedLowRankRgbProj(d_model, rank=rgb_rank)
            self.rgb_proj = None
            self.region_emb = None
        else:
            self.rgb_shared = None
            self.rgb_proj = nn.ModuleDict({r: nn.Linear(RGB_DIM, d_model) for r in RGB_REGIONS})
            self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)

        self.region_reduce = ValidityAwareRegionReduce(d_model) if self.use_region_reduce else None
        self.rgb_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)

        if self.use_separable_skel:
            self.skel_encoder: nn.Module = SeparableSkelEncoder(
                d_model, layers=ctr_layers, dropout=dropout
            )
            self.skel_temporal: nn.Module | None = None
        else:
            self.skel_encoder = CTRGCLiteEncoder(d_model, layers=ctr_layers, dropout=dropout)
            self.skel_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)

        self.motion_reducer = (
            MotionAwareTemporalReducer(keep_t=skel_keep_t) if self.use_motion_reduce else None
        )

        self.cross = CrossSkelToRgb(d_model, nhead)
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

    def _stack_rgb(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        toks = torch.stack([batch[f"token_{r}"] for r in RGB_REGIONS], dim=2)  # (B,L,4,2304)
        miss_list = [batch.get(f"missing_{r}") for r in RGB_REGIONS]
        if any(m is not None for m in miss_list):
            zeros = torch.zeros_like(toks[..., 0], dtype=torch.bool)
            missing = torch.stack(
                [m if m is not None else zeros for m in miss_list],
                dim=2,
            )
        else:
            missing = None
        return toks, missing

    def _encode_rgb(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        rgb_pad = batch.get("rgb_pad_mask")
        stacked, missing = self._stack_rgb(batch)

        if self.use_shared_rgb:
            assert self.rgb_shared is not None
            regions = self.rgb_shared(stacked)  # (B,L,4,d)
            if missing is not None:
                regions = regions.masked_fill(missing.unsqueeze(-1), 0.0)
            if rgb_pad is not None:
                regions = regions.masked_fill(rgb_pad.unsqueeze(-1).unsqueeze(-1), 0.0)
        else:
            assert self.rgb_proj is not None and self.region_emb is not None
            parts = []
            for i, r in enumerate(RGB_REGIONS):
                tok = self.rgb_proj[r](batch[f"token_{r}"])
                miss = batch.get(f"missing_{r}")
                if miss is not None:
                    tok = tok.masked_fill(miss.unsqueeze(-1), 0.0)
                tok = tok + self.region_emb.weight[i]
                if rgb_pad is not None:
                    tok = tok.masked_fill(rgb_pad.unsqueeze(-1), 0.0)
                parts.append(tok)
            regions = torch.stack(parts, dim=2)

        rgb_fused = regions.mean(dim=2)
        _seq, rgb_ctx = self.rgb_temporal(rgb_fused, rgb_pad)

        if self.use_region_reduce:
            assert self.region_reduce is not None
            rgb_for_cross, rgb_cross_pad = self.region_reduce(
                regions, missing=missing, rgb_pad=rgb_pad
            )
        else:
            b, l, r, d = regions.shape
            rgb_for_cross = regions.reshape(b, l * r, d)
            rgb_cross_pad = rgb_pad.repeat(1, len(RGB_REGIONS)) if rgb_pad is not None else None

        return rgb_ctx, rgb_cross_pad, rgb_for_cross

    def _encode_skel(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        skel_pad = batch.get("skel_pad_mask")
        coords = batch["coordinates"]
        velocity = batch["velocity"]
        validity = batch["validity"]
        skel_frames = self.skel_encoder(coords, velocity, validity, skel_pad)

        if self.use_motion_reduce:
            assert self.motion_reducer is not None
            skel_frames, skel_pad = self.motion_reducer(skel_frames, velocity, validity, skel_pad)

        if self.use_separable_skel:
            # Separable encoder already mixes temporal; pool directly.
            skel_seq = skel_frames
            skel_pool = self._pool_masked(skel_seq, skel_pad)
        else:
            assert self.skel_temporal is not None
            skel_seq, skel_pool = self.skel_temporal(skel_frames, skel_pad)
        return skel_seq, skel_pool, skel_pad

    def forward(self, batch: dict[str, Any], *, alpha_override: float | None = None) -> dict[str, torch.Tensor]:
        rgb_ctx, rgb_flat_pad, rgb_flat = self._encode_rgb(batch)
        skel_seq, skel_pool, skel_pad = self._encode_skel(batch)
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


def build_ms_sta_v6_e4_r2(arm: str, **kwargs: Any) -> MSSTAv6E4R2:
    return MSSTAv6E4R2(arm, **kwargs)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def arm_flags(arm: ArmE4R2) -> dict[str, bool]:
    return {
        "shared_low_rank_rgb": arm in {"A", "E"},
        "region_reduce_4l_to_l": arm in {"B", "E"},
        "motion_aware_skel_temporal": arm in {"C", "E"},
        "separable_skel_graph_temporal": arm in {"D", "E"},
    }
