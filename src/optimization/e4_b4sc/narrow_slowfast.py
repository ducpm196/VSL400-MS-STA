# -*- coding: utf-8 -*-
"""E4_B4SC narrow SlowFast: structured Slow-channel compression with Fast frozen."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.rgb_token_v3 import PathwayMaps


TEACHER_SLOW_WIDTHS = (256, 512, 1024, 2048)
FAST_WIDTHS = (32, 64, 128, 256)
FUSION_OUT = (16, 64, 128, 256)  # after stems/stages 0..3
STEM_SLOW = 64
STEM_FAST = 8


def _sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stage_inner(width: int) -> int:
    return width // 4


@dataclass(frozen=True)
class ArmWidths:
    name: str
    slow_widths: tuple[int, int, int, int]

    @property
    def final_slow(self) -> int:
        return int(self.slow_widths[-1])

    @property
    def token_dim(self) -> int:
        return int(self.final_slow + FAST_WIDTHS[-1])


E2_WIDTHS = ArmWidths("E2", (224, 448, 896, 1792))
E3_WIDTHS = ArmWidths("E3", (192, 384, 768, 1536))


def select_topk_indices(score: np.ndarray, k: int) -> np.ndarray:
    if k > score.size:
        raise ValueError(f"topk_overflow:{k}>{score.size}")
    # stable: higher score first; ties by lower index
    order = np.lexsort((np.arange(score.size), -score))
    idx = np.sort(order[:k].astype(np.int64))
    return idx


def bn_gamma_scores(module: nn.Module) -> np.ndarray | None:
    if isinstance(module, nn.BatchNorm3d) and module.weight is not None:
        return module.weight.detach().float().cpu().numpy().astype(np.float64)
    return None


def conv_out_l1_scores(conv: nn.Conv3d) -> np.ndarray:
    w = conv.weight.detach().float().cpu().numpy()
    return np.abs(w).reshape(w.shape[0], -1).sum(axis=1).astype(np.float64)


def replace_conv3d(
    old: nn.Conv3d,
    *,
    out_idx: np.ndarray | None,
    in_idx: np.ndarray | None,
) -> nn.Conv3d:
    out_ch = old.out_channels if out_idx is None else int(len(out_idx))
    in_ch = old.in_channels if in_idx is None else int(len(in_idx))
    new = nn.Conv3d(
        in_ch,
        out_ch,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=1 if old.groups == 1 else old.groups,  # only support groups=1 prune
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        w = old.weight.detach().cpu()
        if in_idx is not None:
            w = w[:, torch.as_tensor(in_idx, dtype=torch.long), ...]
        if out_idx is not None:
            w = w[torch.as_tensor(out_idx, dtype=torch.long), ...]
        new.weight.copy_(w)
        if old.bias is not None and new.bias is not None:
            b = old.bias.detach().cpu()
            if out_idx is not None:
                b = b[torch.as_tensor(out_idx, dtype=torch.long)]
            new.bias.copy_(b)
    return new


def replace_bn3d(old: nn.BatchNorm3d, idx: np.ndarray | None) -> nn.BatchNorm3d:
    n = old.num_features if idx is None else int(len(idx))
    new = nn.BatchNorm3d(
        n,
        eps=old.eps,
        momentum=old.momentum,
        affine=old.affine,
        track_running_stats=old.track_running_stats,
    )
    with torch.no_grad():
        if idx is None:
            if old.affine:
                new.weight.copy_(old.weight.detach().cpu())
                new.bias.copy_(old.bias.detach().cpu())
            if old.track_running_stats:
                new.running_mean.copy_(old.running_mean.detach().cpu())
                new.running_var.copy_(old.running_var.detach().cpu())
                new.num_batches_tracked.copy_(old.num_batches_tracked.detach().cpu())
        else:
            t = torch.as_tensor(idx, dtype=torch.long)
            if old.affine:
                new.weight.copy_(old.weight.detach().cpu()[t])
                new.bias.copy_(old.bias.detach().cpu()[t])
            if old.track_running_stats:
                new.running_mean.copy_(old.running_mean.detach().cpu()[t])
                new.running_var.copy_(old.running_var.detach().cpu()[t])
                new.num_batches_tracked.copy_(old.num_batches_tracked.detach().cpu())
    return new


def _get(mod: nn.Module, path: str) -> nn.Module:
    cur: Any = mod
    for p in path.split("."):
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur


def _set(mod: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = mod
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def compute_channel_plan(
    teacher: nn.Module,
    arm: ArmWidths,
    *,
    activation_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Deterministic channel indices for Slow stages from BN/conv scores (+ optional act)."""
    plan: dict[str, Any] = {
        "arm": arm.name,
        "slow_widths": list(arm.slow_widths),
        "stages": {},
    }
    # Stage block index in SlowFast: blocks.1..4 are ResStages for pathway 0
    for si, (t_w, s_w) in enumerate(zip(TEACHER_SLOW_WIDTHS, arm.slow_widths)):
        block_i = si + 1
        stage_path = f"blocks.{block_i}.multipathway_blocks.0"
        stage = _get(teacher, stage_path)
        # score output channels from last residual's norm_c (or conv_c L1)
        last = stage.res_blocks[-1]
        bn = last.branch2.norm_c
        scores = bn_gamma_scores(bn)
        if scores is None:
            scores = conv_out_l1_scores(last.branch2.conv_c)
        if activation_scores and f"stage{si}_out" in activation_scores:
            scores = scores + 1e-6 * activation_scores[f"stage{si}_out"]
        out_idx = select_topk_indices(scores, s_w)

        # inner bottleneck channels: from first block conv_b / norm_b size = t_w//4
        inner_t = stage_inner(t_w)
        inner_s = stage_inner(s_w)
        bn_b = stage.res_blocks[0].branch2.norm_b
        inner_scores = bn_gamma_scores(bn_b)
        if inner_scores is None:
            inner_scores = conv_out_l1_scores(stage.res_blocks[0].branch2.conv_b)
        inner_idx = select_topk_indices(inner_scores, inner_s)

        plan["stages"][str(si)] = {
            "teacher_out": t_w,
            "student_out": s_w,
            "out_idx": out_idx.tolist(),
            "out_idx_sha256": _sha_arr(out_idx),
            "teacher_inner": inner_t,
            "student_inner": inner_s,
            "inner_idx": inner_idx.tolist(),
            "inner_idx_sha256": _sha_arr(inner_idx),
            "block_index": block_i,
        }
    return plan


def apply_channel_plan(teacher_blocks: nn.ModuleList, plan: dict[str, Any]) -> nn.ModuleList:
    """Clone teacher blocks and prune Slow pathway according to plan; Fast untouched."""
    student = copy.deepcopy(teacher_blocks)
    # previous stage out indices for Slow portion of next stage input
    prev_out_idx: np.ndarray | None = None  # None => stem (all 64)

    for si in range(4):
        st = plan["stages"][str(si)]
        block_i = int(st["block_index"])
        out_idx = np.asarray(st["out_idx"], dtype=np.int64)
        inner_idx = np.asarray(st["inner_idx"], dtype=np.int64)
        # teacher_blocks is ModuleList of blocks.0..4 — index directly
        stage = student[block_i].multipathway_blocks[0]

        # Build input index for Slow features entering this stage.
        # Input layout: [slow_prev | fusion_channels]
        fusion_c = FUSION_OUT[si]
        if prev_out_idx is None:
            slow_in = STEM_SLOW
            slow_in_idx = np.arange(STEM_SLOW, dtype=np.int64)
        else:
            slow_in = int(len(prev_out_idx))
            slow_in_idx = prev_out_idx
        # fusion channels are appended; keep all
        in_idx = np.concatenate(
            [slow_in_idx, np.arange(slow_in, slow_in + fusion_c, dtype=np.int64)]
        )
        # After previous prune, module already has reduced slow channels; fusion still full.
        # When si>0, the tensor channels are already [len(prev_out)|fusion], so in_idx for
        # *current tensor* is identity 0..slow_in+fusion-1, but weights were teacher-sized.
        # We prune from teacher-sized weights using teacher-space indices:
        if prev_out_idx is None:
            teacher_in_idx = in_idx  # 0..79
        else:
            # teacher input was [256/512/...] + fusion; map student slow channels via prev_out_idx
            teacher_slow = TEACHER_SLOW_WIDTHS[si - 1]
            teacher_in_idx = np.concatenate(
                [
                    prev_out_idx.astype(np.int64),
                    np.arange(teacher_slow, teacher_slow + fusion_c, dtype=np.int64),
                ]
            )

        for bi, block in enumerate(stage.res_blocks):
            in_idx_teacher = teacher_in_idx if bi == 0 else out_idx
            if getattr(block, "branch1_conv", None) is not None:
                _set(
                    block,
                    "branch1_conv",
                    replace_conv3d(block.branch1_conv, out_idx=out_idx, in_idx=in_idx_teacher),
                )
                if getattr(block, "branch1_norm", None) is not None:
                    _set(block, "branch1_norm", replace_bn3d(block.branch1_norm, out_idx))

            _set(
                block,
                "branch2.conv_a",
                replace_conv3d(block.branch2.conv_a, out_idx=inner_idx, in_idx=in_idx_teacher),
            )
            _set(block, "branch2.norm_a", replace_bn3d(block.branch2.norm_a, inner_idx))
            _set(
                block,
                "branch2.conv_b",
                replace_conv3d(block.branch2.conv_b, out_idx=inner_idx, in_idx=inner_idx),
            )
            _set(block, "branch2.norm_b", replace_bn3d(block.branch2.norm_b, inner_idx))
            _set(
                block,
                "branch2.conv_c",
                replace_conv3d(block.branch2.conv_c, out_idx=out_idx, in_idx=inner_idx),
            )
            _set(block, "branch2.norm_c", replace_bn3d(block.branch2.norm_c, out_idx))

        prev_out_idx = out_idx

    return student


class NarrowSlowFastBlock4(nn.Module):
    """Student SlowFast Block4 with narrowed Slow pathway; Fast copied exactly."""

    def __init__(
        self,
        teacher: nn.Module,
        plan: dict[str, Any],
        *,
        device: torch.device | str = "cuda",
        requires_grad_slow: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.arm = plan["arm"]
        self.plan = plan
        self.load_meta = dict(getattr(teacher, "load_meta", {}) or {})
        self.load_meta["e4_b4sc_arm"] = str(plan.get("arm"))
        self.load_meta["slow_widths"] = list(plan.get("slow_widths") or [])
        self.blocks = apply_channel_plan(teacher.blocks, plan)
        # freeze fast pathway + fusion + stem; train Slow res stages (blocks 1..4 pathway 0)
        for name, p in self.named_parameters():
            # names look like "1.multipathway_blocks.0...." (ModuleList indices)
            top = name.split(".", 1)[0]
            is_stem = top == "0"
            is_fast = ".multipathway_blocks.1." in name or name.endswith("multipathway_blocks.1")
            is_fusion = "multipathway_fusion" in name
            if is_stem or is_fast or is_fusion:
                p.requires_grad = False
            else:
                p.requires_grad = bool(requires_grad_slow)
        self.to(self.device)

    def forward_maps(self, slow: torch.Tensor, fast: torch.Tensor) -> PathwayMaps:
        maps, _ = self.forward_maps_with_stages(slow, fast)
        return maps

    def forward_maps_with_stages(
        self, slow: torch.Tensor, fast: torch.Tensor
    ) -> tuple[PathwayMaps, list[torch.Tensor]]:
        """Return final maps and Slow stage outputs *before* lateral fusion concat.

        After each ResStage block (indices 1..4), MultiPathWayWithFuse may append
        Fast→Slow fusion channels. We slice to the locked Slow stage width so
        StageKDAdapters see pure Slow channels.
        """
        widths = tuple(int(self.plan["stages"][str(i)]["student_out"]) for i in range(4))
        x: Any = [slow.to(self.device), fast.to(self.device)]
        slow_feats: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= 1 and isinstance(x, (list, tuple)):
                w = widths[i - 1]
                feat = x[0]
                if feat.shape[1] < w:
                    raise RuntimeError(f"slow_feat_narrower_than_plan:{feat.shape[1]}<{w}")
                slow_feats.append(feat[:, :w])
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise RuntimeError(f"unexpected_block4_output:{type(x)}")
        if len(slow_feats) != 4:
            raise RuntimeError(f"expected_4_slow_stage_feats:{len(slow_feats)}")
        return PathwayMaps(slow=x[0][:, : widths[-1]] if x[0].shape[1] >= widths[-1] else x[0], fast=x[1]), slow_feats

    def count_params(self) -> dict[str, int]:
        slow = fast = fusion = stem = 0
        for name, p in self.named_parameters():
            n = int(p.numel())
            top = name.split(".", 1)[0]
            if "multipathway_blocks.1" in name:
                fast += n
            elif "multipathway_fusion" in name:
                fusion += n
            elif top == "0":
                stem += n
            else:
                slow += n
        return {
            "slow_params": slow,
            "fast_params": fast,
            "fusion_params": fusion,
            "stem_params": stem,
            "total": slow + fast + fusion + stem,
            "trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }


class RoiToHeadProjection(nn.Module):
    """Deployable per-region linear map student_token_dim → 2304."""

    def __init__(self, in_dim: int, out_dim: int = 2304, regions: tuple[str, ...] = ("global", "left_hand", "right_hand", "face")):
        super().__init__()
        self.regions = regions
        self.proj = nn.ModuleDict({r: nn.Linear(in_dim, out_dim, bias=True) for r in regions})
        for r in regions:
            nn.init.eye_(self.proj[r].weight[:, : min(in_dim, out_dim)])
            if in_dim < out_dim:
                with torch.no_grad():
                    self.proj[r].weight[:, in_dim:] = 0
            nn.init.zeros_(self.proj[r].bias)

    def forward(self, tokens: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {r: self.proj[r](tokens[r]) for r in self.regions}


class StageKDAdapters(nn.Module):
    """TRAIN_ONLY adapters: student stage feat → teacher channels."""

    def __init__(self, student_widths: tuple[int, ...], teacher_widths: tuple[int, ...] = TEACHER_SLOW_WIDTHS):
        super().__init__()
        self.adapters = nn.ModuleList(
            [nn.Conv3d(s, t, kernel_size=1, bias=False) for s, t in zip(student_widths, teacher_widths)]
        )
        for a, s, t in zip(self.adapters, student_widths, teacher_widths):
            nn.init.zeros_(a.weight)
            with torch.no_grad():
                k = min(s, t)
                a.weight[:k, :k, 0, 0, 0] = torch.eye(k)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [a(f) for a, f in zip(self.adapters, feats)]
