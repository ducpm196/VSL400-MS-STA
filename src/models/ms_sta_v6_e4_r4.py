"""MS-STA v6 E4-R4 feasibility models (untrained screening only).

Consumes compact RGB tensors (R0–R3) + compact Skeleton (S_A/S_B).
Preserves spatial/temporal/cross attention + stream gate (policy F arm).
No Python assert in production paths.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_prep.e4_r4_joint_layouts import layout_indices
from src.models.ms_sta_v5 import CrossSkelToRgb, ReZeroGate, StreamGate, TemporalTransformer


def _build_compact_adjacency(n_joints: int) -> torch.Tensor:
    a = torch.eye(n_joints, dtype=torch.float32)
    for i in range(n_joints - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    # Mild global coupling for stability on short layouts.
    a = a + 0.05
    deg = a.sum(dim=1).clamp_min(1e-6)
    d = deg.pow(-0.5)
    return d.unsqueeze(1) * a * d.unsqueeze(0)


class CompactCTRBlock(nn.Module):
    def __init__(self, channels: int, n_joints: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("shared_a", _build_compact_adjacency(n_joints), persistent=False)
        self.delta_a = nn.Parameter(torch.zeros(n_joints, n_joints))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,T,J
        a = self.shared_a + torch.tanh(self.delta_a)
        a = 0.5 * (a + a.transpose(0, 1))
        y = torch.einsum("bctj,jk->bctk", x, a)
        y = self.conv(y)
        y = self.bn(y)
        return self.drop(F.gelu(y + x))


class CompactSkelEncoder(nn.Module):
    def __init__(self, n_joints: int, d_model: int, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.in_proj = nn.Linear(6, d_model)  # xyz + vel xyz
        self.blocks = nn.ModuleList([CompactCTRBlock(d_model, n_joints, dropout) for _ in range(layers)])
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        coords: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
        pad: torch.Tensor | None,
    ) -> torch.Tensor:
        # coords/velocity: B,T,J,3
        x = torch.cat([coords, velocity], dim=-1)
        h = self.in_proj(x)  # B,T,J,D
        h = h * validity.unsqueeze(-1).clamp(0, 1)
        h = h.permute(0, 3, 1, 2).contiguous()  # B,D,T,J
        for blk in self.blocks:
            h = blk(h)
        h = h.permute(0, 2, 3, 1).contiguous()  # B,T,J,D
        pooled = (h * validity.unsqueeze(-1)).sum(dim=2) / validity.sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        out = self.out(pooled)
        if pad is not None:
            out = out.masked_fill(pad.unsqueeze(-1), 0.0)
        return out


class CompactRgbEncoder(nn.Module):
    """Single-backbone-pass RGB encoder for R0/R1/R2/R3 canvases."""

    def __init__(self, *, in_ch: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(dropout)

    def forward_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: B*T, C, H, W  OR  B*T*V, C, H, W
        x = self.stem(frames).flatten(1)
        return self.drop(x)


class MSSTAv6E4R4(nn.Module):
    def __init__(
        self,
        *,
        rgb_id: str,
        skel_layout: str,
        num_classes: int = 400,
        d_model: int = 128,
        nhead: int = 4,
        temporal_layers: int = 1,
        ctr_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if rgb_id not in {"R0", "R1", "R2", "R3"}:
            raise ValueError(f"bad_rgb_id:{rgb_id}")
        if skel_layout not in {"S_A", "S_B"}:
            raise ValueError(f"bad_skel:{skel_layout}")
        self.rgb_id = rgb_id
        self.skel_layout = skel_layout
        self.d_model = d_model
        self.n_joints = len(layout_indices(skel_layout))
        in_ch = 1 if rgb_id == "R3" else 3
        self.rgb_encoder = CompactRgbEncoder(in_ch=in_ch, d_model=d_model, dropout=dropout)
        self.rgb_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)
        self.skel_encoder = CompactSkelEncoder(self.n_joints, d_model, layers=ctr_layers, dropout=dropout)
        self.skel_temporal = TemporalTransformer(d_model, nhead, temporal_layers, dropout)
        self.cross = CrossSkelToRgb(d_model, nhead)
        self.gate = StreamGate(d_model)
        self.rezero = ReZeroGate(alpha_max=1.0)
        self.update_ln = nn.LayerNorm(d_model)
        self.base_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(d_model, num_classes)
        self.drop = nn.Dropout(dropout)

    def _pool(self, x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        if pad is None:
            return x.mean(dim=1)
        w = (~pad).float().unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def _encode_rgb(self, rgb: torch.Tensor, temporal_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # rgb layouts:
        # R0/R1/R3: B,T,H,W,C
        # R2: B,T,2,H,W,C
        pad = ~temporal_valid
        if self.rgb_id == "R2":
            if rgb.ndim != 6:
                raise RuntimeError(f"r2_bad_ndim:{tuple(rgb.shape)}")
            b, t, v, h, w, c = rgb.shape
            if v != 2:
                raise RuntimeError("r2_views_ne_2")
            flat = rgb.permute(0, 1, 2, 5, 3, 4).reshape(b * t * v, c, h, w)
            tok = self.rgb_encoder.forward_frames(flat).view(b, t, v, -1).mean(dim=2)
        else:
            if rgb.ndim != 5:
                raise RuntimeError(f"rgb_bad_ndim:{tuple(rgb.shape)}")
            b, t, h, w, c = rgb.shape
            flat = rgb.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
            tok = self.rgb_encoder.forward_frames(flat).view(b, t, -1)
        tok = tok.masked_fill(pad.unsqueeze(-1), 0.0)
        ctx, pooled = self.rgb_temporal(tok, pad)
        return ctx, pad

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        rgb = batch["rgb_tensor"]
        tvalid = batch["temporal_valid"]
        if tvalid.dtype != torch.bool:
            tvalid = tvalid.bool()
        rgb_ctx, rgb_pad = self._encode_rgb(rgb, tvalid)
        rgb_pool = self._pool(rgb_ctx, rgb_pad)

        coords = batch["coordinates"]
        velocity = batch["velocity"]
        validity = batch["validity"]
        skel_pad = batch.get("skel_pad_mask")
        if skel_pad is None:
            skel_pad = torch.zeros(coords.shape[:2], dtype=torch.bool, device=coords.device)
        elif skel_pad.dtype != torch.bool:
            skel_pad = skel_pad.bool()

        # All-pad skeleton: keep zeros + all-True mask; do NOT invent observed joints.
        # Fall back to RGB/base branch (zero skeleton contribution).
        skel_all_pad = skel_pad.all(dim=1)  # (B,)
        zeros_skel = torch.zeros_like(rgb_pool)

        if bool((~skel_all_pad).any()):
            # Transformer forbids all-True key masks; use a safe mask only for the call,
            # then hard-zero outputs for all-pad rows so padded values cannot affect them.
            safe_pad = skel_pad.clone()
            if bool(skel_all_pad.any()):
                safe_pad[skel_all_pad, 0] = False
            skel_frames = self.skel_encoder(coords, velocity, validity, skel_pad)
            skel_seq, skel_pool = self.skel_temporal(skel_frames, safe_pad)
            if bool(skel_all_pad.any()):
                skel_seq = skel_seq.masked_fill(skel_all_pad[:, None, None], 0.0)
                skel_pool = skel_pool.masked_fill(skel_all_pad[:, None], 0.0)
            crossed = self.cross(skel_seq, rgb_ctx, skel_pad=skel_pad, rgb_pad=rgb_pad)
            crossed_pool = self._pool(crossed, skel_pad)
            if bool(skel_all_pad.any()):
                crossed_pool = crossed_pool.masked_fill(skel_all_pad[:, None], 0.0)
        else:
            skel_pool = zeros_skel
            crossed_pool = zeros_skel

        h_base = self.base_fuse(torch.cat([rgb_pool, skel_pool], dim=-1))
        gated, gate_w = self.gate(rgb_pool, crossed_pool)
        u = self.update_ln(gated)
        alpha = self.rezero()
        h = self.drop(h_base + alpha * u)
        # All-pad samples: RGB/base only (no cross-update from empty skeleton).
        if bool(skel_all_pad.any()):
            h_rgb_only = self.drop(h_base)
            h = torch.where(skel_all_pad[:, None], h_rgb_only, h)
            gate_w = gate_w.clone()
            gate_w[skel_all_pad] = gate_w.new_tensor([1.0, 0.0])

        logits = self.cls(h)
        return {
            "logits": logits,
            "logits_base": self.cls(self.drop(h_base)),
            "h": h,
            "gate_weights": gate_w,
            "alpha": alpha,
            "skel_all_pad": skel_all_pad,
        }


def build_ms_sta_v6_e4_r4(*, rgb_id: str, skel_layout: str, **kwargs: Any) -> MSSTAv6E4R4:
    return MSSTAv6E4R4(rgb_id=rgb_id, skel_layout=skel_layout, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
