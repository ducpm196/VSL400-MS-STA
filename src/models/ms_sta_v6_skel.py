"""MS-STA v6 Skeleton graph helpers (75-joint CTR-GC-lite adjacency).

Does not copy NTU topology. Edges come from joint_layout_v3 bone list plus
intra-part fully-connected pose hubs and wrist–pose bridges.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_prep.joint_layout_v3 import (
    FACE,
    LEFT_HAND,
    N_JOINTS,
    POSE,
    RIGHT_HAND,
    bone_edge_list,
)


def build_adjacency_75(*, self_loops: bool = True) -> torch.Tensor:
    """Return dense adjacency (J,J) float32 for VSL 75-joint layout."""
    a = torch.zeros(N_JOINTS, N_JOINTS, dtype=torch.float32)
    for i, j in bone_edge_list():
        a[i, j] = 1.0
        a[j, i] = 1.0
    # Pose hub: connect shoulders/hips more densely within pose block.
    for i in range(POSE.start, POSE.stop):
        for j in range(POSE.start, POSE.stop):
            if i != j:
                a[i, j] = max(float(a[i, j]), 0.25)
    # Wrist bridges: LH/RH root (index 0 in hand) to pose wrists if present.
    # Pose indices 9/10 are common wrist-ish endpoints in local bone list.
    for hand, wrist_pose in ((LEFT_HAND, 9), (RIGHT_HAND, 10)):
        hi = hand.start
        if 0 <= wrist_pose < N_JOINTS:
            a[hi, wrist_pose] = 1.0
            a[wrist_pose, hi] = 1.0
    # Face root to pose nose/head proxy (pose joint 0).
    a[FACE.start, 0] = 1.0
    a[0, FACE.start] = 1.0
    if self_loops:
        a += torch.eye(N_JOINTS)
    # Symmetric normalize D^{-1/2} A D^{-1/2}
    deg = a.sum(dim=1).clamp_min(1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)


class CTRGCLiteBlock(nn.Module):
    """Channel-wise topology refinement (lite): shared A + learned residual."""

    def __init__(self, channels: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.shared_a = nn.Parameter(build_adjacency_75(), requires_grad=False)
        self.delta_a = nn.Parameter(torch.zeros(N_JOINTS, N_JOINTS))
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), 1),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 8), channels, 1),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J)
        a = self.shared_a + torch.tanh(self.delta_a)
        a = 0.5 * (a + a.transpose(0, 1))
        # Graph conv over joints: x @ A
        y = torch.einsum("bctj,jk->bctk", x, a)
        y = self.conv(y)
        y = y * self.channel_attn(y)
        y = self.bn(y)
        return self.drop(F.gelu(y + x))


class CTRGCLiteEncoder(nn.Module):
    """Encode coords+velocity with 1–2 CTR-GC-lite layers → (B,T,d_model)."""

    def __init__(self, d_model: int = 256, *, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(6, d_model)
        self.blocks = nn.ModuleList([CTRGCLiteBlock(d_model, dropout=dropout) for _ in range(layers)])
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        coords: torch.Tensor,
        velocity: torch.Tensor,
        validity: torch.Tensor,
        skel_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # coords/velocity (B,T,J,3)
        x = torch.cat([coords, velocity], dim=-1)
        h = self.in_proj(x)  # (B,T,J,C)
        if validity is not None:
            h = h * (validity > 0.5).float().unsqueeze(-1)
        h = h.permute(0, 3, 1, 2).contiguous()  # (B,C,T,J)
        for blk in self.blocks:
            h = blk(h)
        h = h.permute(0, 2, 3, 1).contiguous()  # (B,T,J,C)
        # Validity-aware joint pool → frame token
        w = (validity > 0.5).float().unsqueeze(-1)
        pooled = (h * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        if skel_pad is not None:
            pooled = pooled.masked_fill(skel_pad.unsqueeze(-1), 0.0)
        return self.out_ln(pooled)
