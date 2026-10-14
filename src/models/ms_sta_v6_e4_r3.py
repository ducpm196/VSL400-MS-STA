"""MS-STA v6 E4-R3 architecture (research branch only).

Arms:
  A — RGB compact: low-rank proj + validity-aware 4→1 region fusion
  B — Skeleton compact: semantic joint pooling + separable graph-temporal
  C — Full: A + B

External inputs remain sealed RGB token v3 + Skeleton v3 (identical to B4).
Grayscale / corpus re-extract is forbidden by policy.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_prep.joint_layout_v3 import N_JOINTS
from src.models.ms_sta_v5 import (
    RGB_DIM,
    RGB_REGIONS,
    CrossSkelToRgb,
    ReZeroGate,
    StreamGate,
    TemporalTransformer,
)
from src.models.ms_sta_v6_skel import CTRGCLiteEncoder

ArmE4R3 = Literal["A", "B", "C"]

_DEFAULT_D = 256
_DEFAULT_NHEAD = 4
_DEFAULT_TEMPORAL_LAYERS = 2
_DEFAULT_RGB_RANK = 384
_DEFAULT_CTR_LAYERS = 2

# Policy-locked semantic groups: unique coverage of all 75 joints.
E4R3_SEMANTIC_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("upper_torso", (0, 1, 2, 3, 4, 5)),
    ("upper_arms", (6, 7, 8, 9, 10, 11)),
    ("face", tuple(range(12, 33))),
    ("left_hand", tuple(range(33, 54))),
    ("right_hand", tuple(range(54, 75))),
)


def assert_semantic_groups_policy() -> None:
    names = [n for n, _ in E4R3_SEMANTIC_GROUPS]
    if len(names) != len(set(names)):
        raise AssertionError("semantic_group_names_not_unique")
    covered: list[int] = []
    for _, idxs in E4R3_SEMANTIC_GROUPS:
        covered.extend(idxs)
    if len(covered) != len(set(covered)):
        raise AssertionError("semantic_group_indices_overlap")
    if sorted(covered) != list(range(N_JOINTS)):
        raise AssertionError("semantic_group_coverage_not_exact_0_74")


assert_semantic_groups_policy()


class SharedLowRankRgbProj(nn.Module):
    def __init__(self, d_model: int = _DEFAULT_D, rank: int = _DEFAULT_RGB_RANK) -> None:
        super().__init__()
        self.rank = int(rank)
        self.down = nn.Linear(RGB_DIM, self.rank, bias=True)
        self.up = nn.Linear(self.rank, d_model, bias=True)
        self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        b, l, r, _ = stacked.shape
        flat = stacked.reshape(b * l * r, RGB_DIM)
        h = self.up(F.gelu(self.down(flat))).view(b, l, r, -1)
        return h + self.region_emb.weight.view(1, 1, r, -1)


class ValidityAwareRegionReduce(nn.Module):
    """4 region tokens → 1 fused token per time step (mask-aware)."""

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
        scores = self.score(regions).squeeze(-1)
        if missing is not None:
            scores = scores.masked_fill(missing, float("-inf"))
        if rgb_pad is not None:
            scores = scores.masked_fill(rgb_pad.unsqueeze(-1), float("-inf"))
        all_bad = ~torch.isfinite(scores).any(dim=-1, keepdim=True)
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        fused = (attn.unsqueeze(-1) * regions).sum(dim=2)
        if rgb_pad is not None:
            fused = fused.masked_fill(rgb_pad.unsqueeze(-1), 0.0)
        return fused.masked_fill(all_bad, 0.0), rgb_pad


class SemanticJointPool(nn.Module):
    """Mask-aware pooling of 75 joints → fixed semantic super-joints."""

    def __init__(self) -> None:
        super().__init__()
        self.group_names = [n for n, _ in E4R3_SEMANTIC_GROUPS]
        self.n_super = len(E4R3_SEMANTIC_GROUPS)
        # Constant index buffers (not parameters).
        for name, idxs in E4R3_SEMANTIC_GROUPS:
            self.register_buffer(
                f"idx_{name}",
                torch.tensor(idxs, dtype=torch.long),
                persistent=False,
            )

    def forward(
        self,
        coords: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # coords/vel: (B,T,75,3); validity: (B,T,75)
        pooled_c = []
        pooled_v = []
        pooled_valid = []
        for name, _ in E4R3_SEMANTIC_GROUPS:
            idx = getattr(self, f"idx_{name}")
            c = coords.index_select(2, idx)
            v = velocity.index_select(2, idx)
            w = (validity.index_select(2, idx) > 0.5).float()
            denom = w.sum(dim=2, keepdim=True).clamp_min(1.0)
            c_p = (c * w.unsqueeze(-1)).sum(dim=2) / denom
            v_p = (v * w.unsqueeze(-1)).sum(dim=2) / denom
            # Never treat all-missing as a real observation.
            all_miss = w.sum(dim=2) < 0.5
            c_p = c_p.masked_fill(all_miss.unsqueeze(-1), 0.0)
            v_p = v_p.masked_fill(all_miss.unsqueeze(-1), 0.0)
            pooled_c.append(c_p)
            pooled_v.append(v_p)
            pooled_valid.append((~all_miss).float())
        coords_s = torch.stack(pooled_c, dim=2)  # (B,T,S,3)
        vel_s = torch.stack(pooled_v, dim=2)
        val_s = torch.stack(pooled_valid, dim=2)  # (B,T,S)
        return coords_s, vel_s, val_s


class SeparableGraphTemporalBlock(nn.Module):
    """Learned residual adjacency over semantic super-joints + depthwise temporal."""

    def __init__(self, channels: int, n_joints: int, *, dropout: float = 0.1, k_t: int = 3) -> None:
        super().__init__()
        self.n_joints = int(n_joints)
        eye = torch.eye(self.n_joints)
        prior = torch.ones(self.n_joints, self.n_joints) - eye
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1.0)
        prior = prior + eye
        deg = prior.sum(dim=1).clamp_min(1e-6)
        d_inv = deg.pow(-0.5)
        prior = d_inv.unsqueeze(1) * prior * d_inv.unsqueeze(0)
        self.register_buffer("shared_a", prior, persistent=False)
        self.delta_a = nn.Parameter(torch.zeros(self.n_joints, self.n_joints))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 8), channels, 1),
            nn.Sigmoid(),
        )
        self.bn_s = nn.BatchNorm2d(channels)
        pad = k_t // 2
        self.temporal = nn.Conv2d(
            channels, channels, kernel_size=(k_t, 1), padding=(pad, 0), groups=channels, bias=False
        )
        self.temporal_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn_t = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.shared_a + torch.tanh(self.delta_a)
        a = 0.5 * (a + a.transpose(0, 1))
        y = torch.einsum("bctj,jk->bctk", x, a)
        y = self.conv(y)
        y = y * self.channel_attn(y)
        y = self.bn_s(y)
        y = self.drop(F.gelu(y + x))
        z = self.temporal_pw(self.temporal(y))
        z = self.bn_t(z)
        return self.drop(F.gelu(z + y))


class SeparableSkelEncoderCompact(nn.Module):
    """In-proj + separable graph-temporal on semantic super-joints."""

    def __init__(
        self,
        d_model: int = _DEFAULT_D,
        *,
        n_joints: int,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(6, d_model)
        self.blocks = nn.ModuleList(
            [
                SeparableGraphTemporalBlock(d_model, n_joints, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        coords: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
        skel_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([coords, velocity], dim=-1)
        h = self.in_proj(x)
        h = h * (validity > 0.5).float().unsqueeze(-1)
        # Zero padded frames before spatial/temporal mix so pad values cannot leak.
        if skel_pad is not None:
            h = h.masked_fill(skel_pad.unsqueeze(-1).unsqueeze(-1), 0.0)
        h = h.permute(0, 3, 1, 2).contiguous()
        for blk in self.blocks:
            h = blk(h)
        h = h.permute(0, 2, 3, 1).contiguous()
        w = (validity > 0.5).float().unsqueeze(-1)
        if skel_pad is not None:
            w = w.masked_fill(skel_pad.unsqueeze(-1).unsqueeze(-1), 0.0)
        pooled = (h * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        if skel_pad is not None:
            pooled = pooled.masked_fill(skel_pad.unsqueeze(-1), 0.0)
        return self.out_ln(pooled)


class MSSTAv6E4R3(nn.Module):
    def __init__(
        self,
        arm: ArmE4R3,
        *,
        num_classes: int = 400,
        d_model: int = _DEFAULT_D,
        nhead: int = _DEFAULT_NHEAD,
        temporal_layers: int = _DEFAULT_TEMPORAL_LAYERS,
        dropout: float = 0.1,
        alpha_max: float = 1.0,
        rgb_rank: int = _DEFAULT_RGB_RANK,
        ctr_layers: int = _DEFAULT_CTR_LAYERS,
    ) -> None:
        super().__init__()
        if arm not in {"A", "B", "C"}:
            raise ValueError(f"unknown_e4_r3_arm:{arm}")
        self.arm = arm
        self.d_model = int(d_model)
        self.use_rgb_compact = arm in {"A", "C"}
        self.use_skel_compact = arm in {"B", "C"}

        if self.use_rgb_compact:
            self.rgb_shared = SharedLowRankRgbProj(d_model, rank=rgb_rank)
            self.region_reduce = ValidityAwareRegionReduce(d_model)
            self.rgb_proj = None
            self.region_emb = None
        else:
            self.rgb_shared = None
            self.region_reduce = None
            self.rgb_proj = nn.ModuleDict({r: nn.Linear(RGB_DIM, d_model) for r in RGB_REGIONS})
            self.region_emb = nn.Embedding(len(RGB_REGIONS), d_model)

        self.rgb_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)

        if self.use_skel_compact:
            self.joint_pool = SemanticJointPool()
            self.skel_encoder: nn.Module = SeparableSkelEncoderCompact(
                d_model,
                n_joints=len(E4R3_SEMANTIC_GROUPS),
                layers=ctr_layers,
                dropout=dropout,
            )
            self.skel_temporal: nn.Module | None = None
        else:
            self.joint_pool = None
            self.skel_encoder = CTRGCLiteEncoder(d_model, layers=ctr_layers, dropout=dropout)
            self.skel_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)

        self.cross = CrossSkelToRgb(d_model, nhead)
        self.gate = StreamGate(d_model)
        self.rezero = ReZeroGate(alpha_max=alpha_max)
        self.update_ln = nn.LayerNorm(d_model)
        self.base_fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _pool_masked(self, x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        if pad is None:
            return x.mean(dim=1)
        w = (~pad).float().unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def _stack_rgb(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        toks = torch.stack([batch[f"token_{r}"] for r in RGB_REGIONS], dim=2)
        miss_list = [batch.get(f"missing_{r}") for r in RGB_REGIONS]
        if any(m is not None for m in miss_list):
            zeros = torch.zeros_like(toks[..., 0], dtype=torch.bool)
            missing = torch.stack([m if m is not None else zeros for m in miss_list], dim=2)
        else:
            missing = None
        return toks, missing

    def _encode_rgb(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        rgb_pad = batch.get("rgb_pad_mask")
        stacked, missing = self._stack_rgb(batch)

        if self.use_rgb_compact:
            assert self.rgb_shared is not None and self.region_reduce is not None
            regions = self.rgb_shared(stacked)
            if missing is not None:
                regions = regions.masked_fill(missing.unsqueeze(-1), 0.0)
            if rgb_pad is not None:
                regions = regions.masked_fill(rgb_pad.unsqueeze(-1).unsqueeze(-1), 0.0)
            rgb_for_cross, rgb_cross_pad = self.region_reduce(
                regions, missing=missing, rgb_pad=rgb_pad
            )
            _seq, rgb_ctx = self.rgb_temporal(rgb_for_cross, rgb_pad)
            return rgb_ctx, rgb_cross_pad, rgb_for_cross

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

        if self.use_skel_compact:
            assert self.joint_pool is not None
            coords, velocity, validity = self.joint_pool(coords, velocity, validity)
            skel_frames = self.skel_encoder(coords, velocity, validity, skel_pad)
            skel_pool = self._pool_masked(skel_frames, skel_pad)
            return skel_frames, skel_pool, skel_pad

        skel_frames = self.skel_encoder(coords, velocity, validity, skel_pad)
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


def build_ms_sta_v6_e4_r3(arm: str, **kwargs: Any) -> MSSTAv6E4R3:
    return MSSTAv6E4R3(arm, **kwargs)  # type: ignore[arg-type]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def arm_flags(arm: ArmE4R3) -> dict[str, bool]:
    return {
        "rgb_compact": arm in {"A", "C"},
        "skel_compact": arm in {"B", "C"},
    }


def checkpoint_identity(
    *,
    arm: str,
    config: dict[str, Any],
    policy_sha256: str,
    executable_tree_sha256: str,
) -> dict[str, Any]:
    """Exact arm/config/policy/tree SHA identity for exported checkpoints."""
    return {
        "schema": "ms_sta_v6_e4_r3_checkpoint_identity_v1",
        "campaign": "E4-R3",
        "arm": arm,
        "variant": f"e4r3_{arm}",
        "config": dict(config),
        "policy_sha256": policy_sha256,
        "executable_tree_sha256": executable_tree_sha256,
        "forbid_grayscale_replicated_3ch": True,
    }
