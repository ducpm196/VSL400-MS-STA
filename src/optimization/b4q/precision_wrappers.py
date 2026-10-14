# -*- coding: utf-8 -*-
"""B4Q precision wrappers — change execution dtype only; keep B4 semantics."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn

from src.models.rgb_token_v3 import FrozenSlowFastBlock4, PathwayMaps


@contextmanager
def maybe_autocast(enabled: bool, dtype: torch.dtype = torch.float16) -> Iterator[None]:
    if enabled and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


class AmpRoiFp32SlowFast(nn.Module):
    """SlowFast conv under AMP FP16; pathway maps cast to FP32 before ROI/reductions."""

    def __init__(self, base: FrozenSlowFastBlock4, *, channels_last: bool = False) -> None:
        super().__init__()
        self.base = base
        self.channels_last = bool(channels_last)
        self.eval()

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def checkpoint_sha256(self) -> str:
        return self.base.checkpoint_sha256

    @property
    def load_meta(self) -> dict[str, Any]:
        return self.base.load_meta

    @torch.no_grad()
    def forward_maps(self, slow: torch.Tensor, fast: torch.Tensor) -> PathwayMaps:
        if self.channels_last:
            # pytorchvideo pathways are 5D; channels_last_3d for volumetric tensors
            slow = slow.to(memory_format=torch.channels_last_3d)
            fast = fast.to(memory_format=torch.channels_last_3d)
        with torch.inference_mode(), maybe_autocast(True, torch.float16):
            maps = self.base.forward_maps(slow, fast)
        # ROIAlign / masked mean stay FP32
        return PathwayMaps(slow=maps.slow.float(), fast=maps.fast.float())


def head_forward(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    amp_fp16: bool,
    inference_mode: bool = True,
) -> torch.Tensor:
    ctx = torch.inference_mode() if inference_mode else torch.no_grad()
    with ctx:
        with maybe_autocast(amp_fp16, torch.float16):
            out = model(batch)
        logits = out["logits"].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non_finite_logits")
        return logits
