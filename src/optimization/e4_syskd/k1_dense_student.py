# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 K1 dense compact Slow student (hardware controls only; no training).

Dense C_mid resize — NOT V2 hard channel selection / transplant / factorized conv_b.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.models.rgb_token_v3 import PathwayMaps
from src.optimization.e4_b4sc.narrow_slowfast import TEACHER_SLOW_WIDTHS, stage_inner

K1_C_MID = (56, 112, 224, 448)
K1_C_OUT = TEACHER_SLOW_WIDTHS  # (256, 512, 1024, 2048)
K2_C_MID_INVENTORY_ONLY = (48, 96, 192, 384)
FAST_WIDTHS = (32, 64, 128, 256)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    arr = t.detach().float().cpu().contiguous().numpy()
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def module_param_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters(), key=lambda x: x[0]):
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().float().cpu().numpy()).tobytes())
    for name, b in sorted(module.named_buffers(), key=lambda x: x[0]):
        h.update(name.encode())
        h.update(np.ascontiguousarray(b.detach().float().cpu().numpy()).tobytes())
    return h.hexdigest()


def _new_conv3d(old: nn.Conv3d, in_ch: int, out_ch: int) -> nn.Conv3d:
    new = nn.Conv3d(
        in_ch,
        out_ch,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=1,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    return new


def _new_bn3d(old: nn.BatchNorm3d, num_features: int) -> nn.BatchNorm3d:
    new = nn.BatchNorm3d(
        num_features,
        eps=old.eps,
        momentum=old.momentum,
        affine=old.affine,
        track_running_stats=old.track_running_stats,
    )
    if new.affine:
        nn.init.ones_(new.weight)
        nn.init.zeros_(new.bias)
    if new.track_running_stats:
        new.running_mean.zero_()
        new.running_var.fill_(1)
        new.num_batches_tracked.zero_()
    return new


def _set(parent: nn.Module, dotted: str, value: nn.Module) -> None:
    parts = dotted.split(".")
    obj = parent
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    name = parts[-1]
    if name.isdigit():
        obj[int(name)] = value
    else:
        setattr(obj, name, value)


def rebuild_dense_mid_blocks(teacher_blocks: nn.ModuleList, c_mid: tuple[int, ...]) -> nn.ModuleList:
    """Deepcopy teacher; replace Slow bottleneck mid channels with dense random modules."""
    if len(c_mid) != 4:
        raise ValueError("c_mid_len")
    student = copy.deepcopy(teacher_blocks)
    for si, mid in enumerate(c_mid):
        block_i = si + 1
        stage = student[block_i].multipathway_blocks[0]
        c_out = int(TEACHER_SLOW_WIDTHS[si])
        for block in stage.res_blocks:
            # branch1 (projection) keeps C_out — untouched if present
            ca = block.branch2.conv_a
            # in_channels already match stage input (stem/fusion or prior C_out)
            in_ch = int(ca.in_channels)
            _set(block, "branch2.conv_a", _new_conv3d(ca, in_ch, int(mid)))
            _set(block, "branch2.norm_a", _new_bn3d(block.branch2.norm_a, int(mid)))
            _set(block, "branch2.conv_b", _new_conv3d(block.branch2.conv_b, int(mid), int(mid)))
            _set(block, "branch2.norm_b", _new_bn3d(block.branch2.norm_b, int(mid)))
            _set(block, "branch2.conv_c", _new_conv3d(block.branch2.conv_c, int(mid), c_out))
            # norm_c stays C_out — keep teacher weights for interface stability on BN stats shape
            if int(block.branch2.norm_c.num_features) != c_out:
                _set(block, "branch2.norm_c", _new_bn3d(block.branch2.norm_c, c_out))
    return student


def inventory_params(module: nn.Module) -> dict[str, Any]:
    n = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"param_count": int(n), "trainable_param_count": int(trainable)}


class DenseK1Student(nn.Module):
    """K1 light student: C_mid compressed densily; C_out / Fast / fusion interfaces retained."""

    def __init__(
        self,
        teacher: nn.Module,
        *,
        c_mid: tuple[int, int, int, int] = K1_C_MID,
        device: torch.device | str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.arm = "K1_DENSE"
        self.c_mid = tuple(int(x) for x in c_mid)
        self.c_out = tuple(int(x) for x in K1_C_OUT)
        self.load_meta = dict(getattr(teacher, "load_meta", {}) or {})
        self.load_meta["e4_syskd_student"] = self.arm
        self.load_meta["c_mid"] = list(self.c_mid)
        self.load_meta["c_out"] = list(self.c_out)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        # isolate RNG for rebuild
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.blocks = rebuild_dense_mid_blocks(teacher.blocks, self.c_mid)
        # freeze Fast + stem + fusion (hardware study; no train)
        for name, p in self.named_parameters():
            top = name.split(".", 1)[0]
            is_stem = top == "0"
            is_fast = ".multipathway_blocks.1." in name or name.endswith("multipathway_blocks.1")
            is_fusion = "multipathway_fusion" in name
            if is_stem or is_fast or is_fusion:
                p.requires_grad = False
            else:
                p.requires_grad = False  # no training in this task
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
                w = self.c_out[i - 1]
                feat = x[0]
                if feat.shape[1] < w:
                    raise RuntimeError(f"slow_feat_narrower_than_c_out:{feat.shape[1]}<{w}")
                slow_feats.append(feat[:, :w])
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise RuntimeError(f"unexpected_k1_output:{type(x)}")
        if len(slow_feats) != 4:
            raise RuntimeError(f"expected_4_slow_stages:{len(slow_feats)}")
        return PathwayMaps(slow=x[0][:, : self.c_out[-1]], fast=x[1]), slow_feats

    def stage_mid_channels(self) -> list[int]:
        out = []
        for si in range(4):
            stage = self.blocks[si + 1].multipathway_blocks[0]
            out.append(int(stage.res_blocks[0].branch2.conv_b.out_channels))
        return out


def verify_fast_unchanged(teacher: nn.Module, student: nn.Module) -> dict[str, Any]:
    t_hashes = {}
    s_hashes = {}
    for name, p in teacher.named_parameters():
        if ".multipathway_blocks.1." in name or name.endswith("multipathway_blocks.1"):
            t_hashes[name] = tensor_sha256(p.data)
    for name, p in student.named_parameters():
        if ".multipathway_blocks.1." in name or name.endswith("multipathway_blocks.1"):
            s_hashes[name] = tensor_sha256(p.data)
    missing = sorted(set(t_hashes) - set(s_hashes))
    extra = sorted(set(s_hashes) - set(t_hashes))
    mismatch = sorted(k for k in set(t_hashes) & set(s_hashes) if t_hashes[k] != s_hashes[k])
    return {
        "fast_param_count_teacher": len(t_hashes),
        "fast_param_count_student": len(s_hashes),
        "missing": missing[:10],
        "extra": extra[:10],
        "mismatch": mismatch[:10],
        "pass": not missing and not extra and not mismatch,
        "teacher_fast_manifest_sha256": sha256_bytes(
            json_stable([{k: t_hashes[k] for k in sorted(t_hashes)}]).encode()
        ),
        "student_fast_manifest_sha256": sha256_bytes(
            json_stable([{k: s_hashes[k] for k in sorted(s_hashes)}]).encode()
        ),
    }


def json_stable(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def teacher_mid_channels(teacher: nn.Module) -> list[int]:
    out = []
    for si in range(4):
        stage = teacher.blocks[si + 1].multipathway_blocks[0]
        out.append(int(stage.res_blocks[0].branch2.conv_b.out_channels))
    return out


def architecture_inventory(teacher: nn.Module, student: DenseK1Student) -> dict[str, Any]:
    return {
        "schema": "e4_syskd_v1_k1_architecture_inventory_v1",
        "teacher": {
            "c_mid": teacher_mid_channels(teacher),
            "c_out": list(TEACHER_SLOW_WIDTHS),
            "expected_c_mid_default": [stage_inner(w) for w in TEACHER_SLOW_WIDTHS],
            "params": inventory_params(teacher),
        },
        "k1": {
            "c_mid": list(student.c_mid),
            "c_out": list(student.c_out),
            "observed_stage_mid": student.stage_mid_channels(),
            "params": inventory_params(student),
            "construction": "dense_rebuild_random_init_mid_bottleneck",
            "forbidden": [
                "hard_channel_selection",
                "teacher_transplant",
                "v2_channel_indices",
                "factorized_conv_b",
                "trained_weights",
            ],
        },
        "k2_inventory_only": {
            "c_mid": list(K2_C_MID_INVENTORY_ONLY),
            "opened": False,
            "note": "Parameter inventory only; no formal hardware benchmark until K1 policy path.",
        },
        "fast_widths_unchanged": list(FAST_WIDTHS),
    }
