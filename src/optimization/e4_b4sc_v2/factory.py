# -*- coding: utf-8 -*-
"""E4_B4SC_V2 shared SlowFast factory: exact clone, identity transplant, future S5."""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.rgb_token_v3 import PathwayMaps
from src.optimization.e4_b4sc.narrow_slowfast import (
    FUSION_OUT,
    TEACHER_SLOW_WIDTHS,
    _get,
    apply_channel_plan,
    stage_inner,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    arr = t.detach().float().cpu().contiguous().numpy()
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def module_param_buffer_hashes(module: nn.Module) -> dict[str, Any]:
    params = {}
    buffers = {}
    for name, p in sorted(module.named_parameters(), key=lambda x: x[0]):
        params[name] = {
            "shape": list(p.shape),
            "sha256": tensor_sha256(p.data),
            "requires_grad": bool(p.requires_grad),
        }
    for name, b in sorted(module.named_buffers(), key=lambda x: x[0]):
        buffers[name] = {"shape": list(b.shape), "sha256": tensor_sha256(b.data)}
    return {
        "param_count": len(params),
        "buffer_count": len(buffers),
        "params": params,
        "buffers": buffers,
        "params_manifest_sha256": sha256_bytes(
            json_dumps_stable([{k: v["sha256"] for k, v in params.items()}]).encode()
        ),
        "buffers_manifest_sha256": sha256_bytes(
            json_dumps_stable([{k: v["sha256"] for k, v in buffers.items()}]).encode()
        ),
    }


def json_dumps_stable(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compare_hash_maps(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    pa, pb = a["params"], b["params"]
    ba, bb = a["buffers"], b["buffers"]
    missing_p = sorted(set(pa) - set(pb))
    extra_p = sorted(set(pb) - set(pa))
    mismatch_p = sorted(
        k for k in set(pa) & set(pb) if pa[k]["sha256"] != pb[k]["sha256"] or pa[k]["shape"] != pb[k]["shape"]
    )
    missing_b = sorted(set(ba) - set(bb))
    extra_b = sorted(set(bb) - set(ba))
    mismatch_b = sorted(
        k for k in set(ba) & set(bb) if ba[k]["sha256"] != bb[k]["sha256"] or ba[k]["shape"] != bb[k]["shape"]
    )
    return {
        "params_keys_equal": not missing_p and not extra_p,
        "buffers_keys_equal": not missing_b and not extra_b,
        "params_sha_equal": not mismatch_p,
        "buffers_sha_equal": not mismatch_b,
        "missing_params": missing_p[:20],
        "extra_params": extra_p[:20],
        "mismatch_params": mismatch_p[:20],
        "missing_buffers": missing_b[:20],
        "extra_buffers": extra_b[:20],
        "mismatch_buffers": mismatch_b[:20],
        "pass": not (missing_p or extra_p or mismatch_p or missing_b or extra_b or mismatch_b),
    }


def identity_channel_plan() -> dict[str, Any]:
    """100% retention plan: every out/inner index is identity arange."""
    plan: dict[str, Any] = {
        "arm": "C1_IDENTITY",
        "slow_widths": list(TEACHER_SLOW_WIDTHS),
        "c_mid": [stage_inner(w) for w in TEACHER_SLOW_WIDTHS],
        "retention_fraction": 1.0,
        "stages": {},
    }
    for si, t_w in enumerate(TEACHER_SLOW_WIDTHS):
        inner = stage_inner(t_w)
        out_idx = np.arange(t_w, dtype=np.int64)
        inner_idx = np.arange(inner, dtype=np.int64)
        plan["stages"][str(si)] = {
            "teacher_out": t_w,
            "student_out": t_w,
            "out_idx": out_idx.tolist(),
            "out_idx_sha256": sha256_bytes(np.ascontiguousarray(out_idx).tobytes()),
            "teacher_inner": inner,
            "student_inner": inner,
            "inner_idx": inner_idx.tolist(),
            "inner_idx_sha256": sha256_bytes(np.ascontiguousarray(inner_idx).tobytes()),
            "block_index": si + 1,
            "is_identity": True,
        }
    return plan


def s5_mid_plan(
    teacher: nn.Module | None = None,
    c_mid: tuple[int, int, int, int] = (64, 128, 256, 448),
    *,
    allow_prefix_stub: bool = False,
) -> dict[str, Any]:
    """Keep C_out; compress only last-stage C_mid via BN-gamma (+ conv L1 tie-break).

    Prefix stub indices are forbidden for formal S5 unless allow_prefix_stub=True (unit tests only).
    """
    from src.optimization.e4_b4sc.narrow_slowfast import (
        bn_gamma_scores,
        conv_out_l1_scores,
        select_topk_indices,
    )

    if len(c_mid) != 4:
        raise ValueError("c_mid_len")
    base = [stage_inner(w) for w in TEACHER_SLOW_WIDTHS]
    plan: dict[str, Any] = {
        "arm": "V2_S5",
        "slow_widths": list(TEACHER_SLOW_WIDTHS),
        "c_mid": list(c_mid),
        "channel_method": "bn_gamma_l1_then_conv_l1_tiebreak",
        "stages": {},
    }
    for si, t_w in enumerate(TEACHER_SLOW_WIDTHS):
        out_idx = np.arange(t_w, dtype=np.int64)  # C_out identity always in V2
        inner_t = base[si]
        inner_s = int(c_mid[si])
        if inner_s > inner_t:
            raise ValueError(f"c_mid_expand_forbidden:{inner_s}>{inner_t}")
        if inner_s == inner_t:
            inner_idx = np.arange(inner_t, dtype=np.int64)
            method = "identity"
        else:
            if teacher is None:
                if not allow_prefix_stub:
                    raise ValueError("s5_mid_plan_requires_teacher_for_nonidentity_mid")
                inner_idx = np.arange(inner_s, dtype=np.int64)
                method = "prefix_stub_test_only"
            else:
                block_i = si + 1
                stage = teacher.blocks[block_i].multipathway_blocks[0]
                bn_b = stage.res_blocks[0].branch2.norm_b
                scores = bn_gamma_scores(bn_b)
                if scores is None:
                    scores = conv_out_l1_scores(stage.res_blocks[0].branch2.conv_b)
                # tie-break: add tiny conv L1 so equal gamma ranks are deterministic
                conv_scores = conv_out_l1_scores(stage.res_blocks[0].branch2.conv_b)
                scores = scores.astype(np.float64) + 1e-12 * conv_scores.astype(np.float64)
                inner_idx = select_topk_indices(scores, inner_s)
                method = "bn_gamma_l1_then_conv_l1_tiebreak"
        plan["stages"][str(si)] = {
            "teacher_out": t_w,
            "student_out": t_w,
            "out_idx": out_idx.tolist(),
            "out_idx_sha256": sha256_bytes(np.ascontiguousarray(out_idx).tobytes()),
            "teacher_inner": inner_t,
            "student_inner": inner_s,
            "inner_idx": inner_idx.tolist(),
            "inner_idx_sha256": sha256_bytes(np.ascontiguousarray(inner_idx).tobytes()),
            "block_index": si + 1,
            "compress_mid_only": inner_s < inner_t,
            "inner_selection_method": method,
        }
    return plan


class ExactWidthClone(nn.Module):
    """Full-width SlowFast Block4 clone through the V2 factory path (no compression)."""

    def __init__(self, teacher: nn.Module, *, device: torch.device | str = "cuda") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.arm = "C0_EXACT"
        self.blocks = copy.deepcopy(teacher.blocks)
        self.load_meta = dict(getattr(teacher, "load_meta", {}) or {})
        self.load_meta["e4_b4sc_v2_arm"] = self.arm
        self.to(self.device)

    def forward_maps(self, slow: torch.Tensor, fast: torch.Tensor) -> PathwayMaps:
        maps, _ = self.forward_maps_with_stages(slow, fast)
        return maps

    def forward_maps_with_stages(
        self, slow: torch.Tensor, fast: torch.Tensor
    ) -> tuple[PathwayMaps, list[torch.Tensor]]:
        x: Any = [slow.to(self.device), fast.to(self.device)]
        slow_feats: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= 1 and isinstance(x, (list, tuple)):
                w = int(TEACHER_SLOW_WIDTHS[i - 1])
                feat = x[0]
                slow_feats.append(feat[:, :w])
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise RuntimeError(f"unexpected_clone_output:{type(x)}")
        if len(slow_feats) != 4:
            raise RuntimeError(f"expected_4_slow_stages:{len(slow_feats)}")
        return PathwayMaps(slow=x[0][:, : TEACHER_SLOW_WIDTHS[-1]], fast=x[1]), slow_feats


class TransplantedSlowFast(nn.Module):
    """Student built by applying a channel plan (identity or mid-compression) to teacher blocks."""

    def __init__(
        self,
        teacher: nn.Module,
        plan: dict[str, Any],
        *,
        device: torch.device | str = "cuda",
        requires_grad_slow: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.arm = str(plan.get("arm"))
        self.plan = plan
        self.blocks = apply_channel_plan(teacher.blocks, plan)
        self.load_meta = dict(getattr(teacher, "load_meta", {}) or {})
        self.load_meta["e4_b4sc_v2_arm"] = self.arm
        for name, p in self.named_parameters():
            top = name.split(".", 1)[0]
            is_stem = top == "0"
            is_fast = ".multipathway_blocks.1." in name
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
        x: Any = [slow.to(self.device), fast.to(self.device)]
        slow_feats: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= 1 and isinstance(x, (list, tuple)):
                w = int(self.plan["slow_widths"][i - 1])
                feat = x[0]
                if feat.shape[1] < w:
                    raise RuntimeError(f"slow_feat_narrow:{feat.shape[1]}<{w}")
                slow_feats.append(feat[:, :w])
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise RuntimeError(f"unexpected_tx_output:{type(x)}")
        if len(slow_feats) != 4:
            raise RuntimeError(f"expected_4_slow_stages:{len(slow_feats)}")
        return PathwayMaps(slow=x[0][:, : int(self.plan["slow_widths"][-1])], fast=x[1]), slow_feats


def build_v2_slowfast(
    teacher: nn.Module,
    arm: str,
    *,
    device: torch.device | str = "cuda",
    requires_grad_slow: bool = False,
    s5_c_mid: tuple[int, int, int, int] = (64, 128, 256, 448),
    eval_mode: bool = True,
) -> nn.Module:
    """Shared factory for C0 exact clone, C1 identity transplant, and later S5."""
    arm_u = arm.upper()
    if arm_u in ("C0", "C0_EXACT", "EXACT_CLONE"):
        m: nn.Module = ExactWidthClone(teacher, device=device)
    elif arm_u in ("C1", "C1_IDENTITY", "IDENTITY"):
        plan = identity_channel_plan()
        m = TransplantedSlowFast(teacher, plan, device=device, requires_grad_slow=requires_grad_slow)
    elif arm_u in ("V2_S5", "S5"):
        plan = s5_mid_plan(teacher, s5_c_mid, allow_prefix_stub=False)
        m = TransplantedSlowFast(teacher, plan, device=device, requires_grad_slow=requires_grad_slow)
    else:
        raise ValueError(f"unknown_v2_arm:{arm}")
    if eval_mode:
        m.eval()
    return m


def assert_identity_plan(plan: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for si in range(4):
        st = plan["stages"][str(si)]
        out = np.asarray(st["out_idx"], dtype=np.int64)
        inn = np.asarray(st["inner_idx"], dtype=np.int64)
        if not np.array_equal(out, np.arange(st["teacher_out"])):
            issues.append(f"stage{si}_out_not_identity")
        if not np.array_equal(inn, np.arange(st["teacher_inner"])):
            issues.append(f"stage{si}_inner_not_identity")
        if int(st["student_out"]) != int(st["teacher_out"]):
            issues.append(f"stage{si}_C_out_changed")
        if int(st["student_inner"]) != int(st["teacher_inner"]):
            issues.append(f"stage{si}_C_mid_changed")
    return issues
