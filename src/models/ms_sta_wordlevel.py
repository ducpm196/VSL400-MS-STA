"""Option B MS-STA word-level classifier (VSL400, 400-way CE).

Requires RGB temporal tokens (B, L, 2304) with L>=2 for temporal/cross claims.
Skeleton: (B, T, 75, 3). Spatial attention returns learned joint weights.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rgb_backbone import SLOWFAST_FEAT_DIM

# b4r: residual MS-STA on B3 concat (v4 hypothesis; not part of retrain-v3 matrix)
Variant = Literal["b0", "b1", "b2", "b3", "b4", "b4r", "b5"]


def _as_tokens_2304(rgb: torch.Tensor) -> torch.Tensor:
    """Normalize RGB to (B, L, 2304). Reject silent L=1 inflation from pooled vectors in train path."""
    if rgb.ndim == 2:
        # (B, 2304) — allowed only for B0 pooled path
        return rgb.unsqueeze(1)
    if rgb.ndim == 3 and rgb.shape[-1] == SLOWFAST_FEAT_DIM:
        return rgb
    raise ValueError(f"RGB shape invalid: {tuple(rgb.shape)} (expect B,2304 or B,L,2304)")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class JointSpatialAttention(nn.Module):
    """Learned per-joint weights; returns pooled features + attention (B, T, N)."""

    def __init__(self, in_features: int = 3, d_feat: int = 64) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.projection = nn.Sequential(nn.Linear(in_features, d_feat), nn.ReLU())
        self.w_s = nn.Linear(d_feat, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(x)
        scores = self.w_s(z) / math.sqrt(self.d_feat)
        attn = torch.softmax(scores, dim=-2)  # (B, T, N, 1)
        pooled = torch.sum(attn * z, dim=-2)  # (B, T, d_feat)
        return pooled, attn.squeeze(-1)


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm(self.enc(self.pos(x), src_key_padding_mask=key_padding_mask))
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            q = (h * valid).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
        else:
            q = h.mean(dim=1, keepdim=True)
        scores = torch.matmul(q, h.transpose(1, 2)) / math.sqrt(h.shape[-1])
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, h).squeeze(1)
        return h, pooled, attn


class StreamGate(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, rgb: torch.Tensor, skel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.fc(torch.cat([rgb, skel], dim=-1)), dim=-1)
        return w[:, 0:1] * rgb + w[:, 1:2] * skel, w


class CrossStreamAttention(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, weights = self.attn(
            q_tokens,
            kv_tokens,
            kv_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return self.norm(out + q_tokens), weights


class MSSTAWordLevel(nn.Module):
    def __init__(
        self,
        num_classes: int = 400,
        *,
        variant: Variant = "b4",
        d_model: int = 256,
        dropout: float = 0.2,
        disable_spatial: bool = False,
        disable_temporal: bool = False,
        min_rgb_len_for_temporal: int = 2,
        skel_temporal_stride: int = 1,
        residual_init: str = "zero_delta_head",
    ) -> None:
        super().__init__()
        self.variant: Variant = variant
        self.d_model = d_model
        self.disable_spatial = bool(disable_spatial)
        self.disable_temporal = bool(disable_temporal)
        self.min_rgb_len_for_temporal = int(min_rgb_len_for_temporal)
        self.skel_temporal_stride = int(skel_temporal_stride)
        if self.skel_temporal_stride < 1:
            raise ValueError("skel_temporal_stride must be >= 1")
        if variant == "b5" and not disable_spatial and not disable_temporal:
            self.disable_spatial = True

        self.rgb_proj = nn.Linear(SLOWFAST_FEAT_DIM, d_model)
        self.spatial = JointSpatialAttention(in_features=3, d_feat=64)
        self.skel_joint_mean_proj = nn.Linear(3, 64)
        self.skel_proj = nn.Linear(64, d_model)
        self.rgb_temporal = TemporalTransformer(d_model=d_model, dropout=dropout)
        self.skel_temporal = TemporalTransformer(d_model=d_model, dropout=dropout)
        self.cross = CrossStreamAttention(d_model=d_model)
        self.gate = StreamGate(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        head = lambda: nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.cls = head()
        self.b0_cls = head()
        self.concat_cls = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )
        # b4r residual modes (v4 hypothesis; not in retrain-v3 matrix):
        # - zero_delta_head (default): B3 + scale * ZeroInitLastLinear(cls(gated)); scale init 1.0
        #   Reuses `cls` (same as B4) — last Linear zeroed so param count matches B4.
        # - zero_scale (legacy ablation): B3 + 0 * cls(gated); attention gets no first-step grad
        if variant == "b4r":
            mode = str(residual_init or "zero_delta_head").strip().lower()
            if mode not in ("zero_delta_head", "zero_scale"):
                raise ValueError(
                    f"unsupported residual_init={residual_init!r}; "
                    "expected 'zero_delta_head' or 'zero_scale'"
                )
            self.residual_init = mode
            if mode == "zero_delta_head":
                last = self.cls[-1]
                if not isinstance(last, nn.Linear):
                    raise TypeError(
                        f"b4r zero_delta_head expects cls[-1] to be nn.Linear, got {type(last)!r}"
                    )
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
                self.residual_scale = nn.Parameter(torch.ones(()))
            else:
                self.residual_scale = nn.Parameter(torch.zeros(()))
        else:
            self.residual_init = "none"
            self.register_parameter("residual_scale", None)

    def _encode_rgb(
        self,
        rgb: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.rgb_proj(_as_tokens_2304(rgb))
        l_len = tokens.shape[1]
        if key_padding_mask is not None:
            valid_counts = (~key_padding_mask).sum(dim=1)
            min_valid = int(valid_counts.min().item())
        else:
            min_valid = l_len
        if self.variant == "b0" or self.disable_temporal or min_valid < self.min_rgb_len_for_temporal:
            if key_padding_mask is not None:
                valid = (~key_padding_mask).float().unsqueeze(-1)
                denom = valid.sum(dim=1).clamp_min(1.0)
                pooled = (tokens * valid).sum(dim=1) / denom
            else:
                pooled = tokens.mean(dim=1)
            attn = torch.ones(tokens.shape[0], 1, l_len, device=tokens.device) / max(l_len, 1)
            return tokens, pooled, attn
        tok, pooled, attn = self.rgb_temporal(tokens, key_padding_mask=key_padding_mask)
        return tok, pooled, attn

    def _encode_skel(
        self, skel: torch.Tensor, *, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if skel.ndim != 4 or skel.shape[-2:] != (75, 3):
            raise ValueError(f"Skeleton shape invalid: {tuple(skel.shape)} (expect B,T,75,3)")
        if self.disable_spatial:
            mean_xyz = skel.mean(dim=2)
            h = torch.relu(self.skel_joint_mean_proj(mean_xyz))
            joint_attn = torch.ones(batch, skel.shape[1], 75, device=device) / 75.0
        else:
            h, joint_attn = self.spatial(skel)
        tokens = self.skel_proj(h)
        if self.skel_temporal_stride > 1:
            tokens = F.avg_pool1d(
                tokens.transpose(1, 2),
                kernel_size=self.skel_temporal_stride,
                stride=self.skel_temporal_stride,
                ceil_mode=True,
            ).transpose(1, 2)
        if self.disable_temporal or self.variant == "b0":
            pooled = tokens.mean(dim=1)
            t_attn = torch.ones(batch, 1, tokens.shape[1], device=device) / max(tokens.shape[1], 1)
            return tokens, pooled, t_attn, joint_attn
        tok, pooled, t_attn = self.skel_temporal(tokens)
        return tok, pooled, t_attn, joint_attn

    def forward(
        self,
        rgb: torch.Tensor | None = None,
        skel: torch.Tensor | None = None,
        *,
        rgb_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if rgb is None and skel is None:
            raise ValueError("Need at least one stream (rgb or skel)")

        attn_pack: dict = {}
        device = (rgb if rgb is not None else skel).device  # type: ignore[union-attr]
        batch = (rgb if rgb is not None else skel).shape[0]  # type: ignore[union-attr]

        rgb_tok = rgb_pool = None
        rgb_valid_len = 0
        if rgb is not None:
            rgb_tok, rgb_pool, rgb_attn = self._encode_rgb(
                rgb, key_padding_mask=rgb_key_padding_mask
            )
            if rgb_key_padding_mask is not None:
                rgb_valid_len = int((~rgb_key_padding_mask).sum(dim=1).min().item())
            else:
                rgb_valid_len = int(rgb_tok.shape[1])
            attn_pack["temporal_rgb"] = rgb_attn
            attn_pack["rgb_len"] = rgb_valid_len
            attn_pack["rgb_pad_len"] = int(rgb_tok.shape[1])

        skel_tok = skel_pool = None
        if skel is not None:
            skel_tok, skel_pool, skel_attn, joint_attn = self._encode_skel(
                skel, batch=batch, device=device
            )
            attn_pack["temporal_skel"] = skel_attn
            attn_pack["spatial_joints"] = joint_attn
            attn_pack["skel_len"] = int(skel_tok.shape[1])

        variant = self.variant
        if variant == "b0":
            if rgb_pool is None:
                raise ValueError("B0 requires RGB")
            logits = self.b0_cls(rgb_pool)
        elif variant == "b1":
            if rgb_pool is None:
                raise ValueError("B1 requires RGB")
            if rgb_valid_len < self.min_rgb_len_for_temporal:
                raise ValueError(
                    f"B1 requires RGB temporal length >= {self.min_rgb_len_for_temporal} "
                    f"(got valid L={rgb_valid_len}). Build isolated_sf_temporal cache with L>=2."
                )
            logits = self.cls(self.dropout(rgb_pool))
        elif variant == "b2":
            if skel_pool is None:
                raise ValueError("B2 requires skeleton")
            logits = self.cls(self.dropout(skel_pool))
        elif variant == "b3":
            if rgb_pool is None or skel_pool is None:
                raise ValueError("B3 requires both streams")
            logits = self.concat_cls(self.dropout(torch.cat([rgb_pool, skel_pool], dim=-1)))
        elif variant in ("b4", "b4r", "b5"):
            if rgb_tok is None or skel_tok is None or rgb_pool is None or skel_pool is None:
                raise ValueError(f"{variant} requires both streams")
            if rgb_valid_len < self.min_rgb_len_for_temporal:
                raise ValueError(
                    f"{variant} requires RGB valid L>={self.min_rgb_len_for_temporal} "
                    f"(got valid L={rgb_valid_len}); refuse padded L=1 fakes."
                )
            fused_tok, cross_w = self.cross(
                skel_tok, rgb_tok, key_padding_mask=rgb_key_padding_mask
            )
            attn_pack["cross"] = cross_w
            # Pool fused tokens over skeleton time only (skel has no pad here).
            fused_pool = fused_tok.mean(dim=1)
            gated, gate_w = self.gate(rgb_pool, fused_pool)
            attn_pack["stream_gate"] = gate_w
            if variant == "b4r":
                base = self.concat_cls(self.dropout(torch.cat([rgb_pool, skel_pool], dim=-1)))
                delta = self.cls(self.dropout(gated))
                scale = self.residual_scale
                if scale is None:
                    raise RuntimeError("b4r requires residual_scale parameter")
                base_n = base.detach().norm(dim=-1)
                delta_n = delta.detach().norm(dim=-1)
                ratio_n = delta_n / base_n.clamp_min(1e-8)
                attn_pack["residual_scale"] = scale.detach()
                attn_pack["residual_init"] = self.residual_init
                # Per-sample tensors for unbiased mean/std; scalars kept for display.
                attn_pack["base_logits_norm_per_sample"] = base_n.detach()
                attn_pack["delta_logits_norm_per_sample"] = delta_n.detach()
                attn_pack["delta_over_base_ratio_per_sample"] = ratio_n.detach()
                attn_pack["base_logits_norm"] = base_n.mean().detach()
                attn_pack["delta_logits_norm"] = delta_n.mean().detach()
                attn_pack["delta_over_base_ratio"] = ratio_n.mean().detach()
                logits = base + scale * delta
            else:
                logits = self.cls(self.dropout(gated))
        else:
            raise ValueError(f"unsupported MS-STA variant: {variant!r}")

        if return_attention:
            return logits, attn_pack
        return logits

    def load_b0_head(
        self,
        state: dict,
        *,
        classifier: bool = True,
        classifier_target: str = "b0_cls",
    ) -> None:
        """Load Gate3 head weights.

        classifier=False -> rgb_proj only
        classifier=True + target b0_cls -> rgb_proj + b0_cls (B0-E1)
        classifier=True + target cls -> rgb_proj + cls (B1/B4/B5 compatible)
        """
        if "proj.weight" in state:
            self.rgb_proj.weight.data.copy_(state["proj.weight"])
            self.rgb_proj.bias.data.copy_(state["proj.bias"])
        if not classifier:
            return
        mapped = {k[len("cls.") :]: v for k, v in state.items() if k.startswith("cls.")}
        if not mapped:
            return
        if classifier_target == "b0_cls":
            self.b0_cls.load_state_dict(mapped, strict=True)
        elif classifier_target == "cls":
            self.cls.load_state_dict(mapped, strict=True)
        else:
            raise ValueError(
                f"unsupported classifier_target={classifier_target!r}; "
                "expected 'b0_cls' or 'cls' (concat_cls is not Gate3-compatible)"
            )


def build_ms_sta(variant: Variant, num_classes: int = 400, **kwargs) -> MSSTAWordLevel:
    disable_spatial = kwargs.pop("disable_spatial", variant == "b5")
    disable_temporal = kwargs.pop("disable_temporal", False)
    residual_init = kwargs.pop("residual_init", "zero_delta_head")
    return MSSTAWordLevel(
        num_classes=num_classes,
        variant=variant,
        disable_spatial=disable_spatial,
        disable_temporal=disable_temporal,
        residual_init=residual_init,
        **kwargs,
    )
