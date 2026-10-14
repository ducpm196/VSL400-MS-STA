# -*- coding: utf-8 -*-
"""S1 pinned/preallocated H2D wrapper for SlowFast window tensors."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.rgb_token_v3 import PathwayMaps


class PinnedTransferSlowFast(nn.Module):
    """Reusable pinned-host + preallocated GPU buffers around an inner extractor.

    Does not change compute graph values when copies are exact.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.device = getattr(inner, "device", torch.device("cuda"))
        self._slow_gpu: torch.Tensor | None = None
        self._fast_gpu: torch.Tensor | None = None
        self._slow_pin: torch.Tensor | None = None
        self._fast_pin: torch.Tensor | None = None
        self.stats = {"h2d_calls": 0, "reallocs": 0}

    @property
    def blocks(self):
        return self.inner.blocks

    @property
    def checkpoint_sha256(self) -> str:
        return self.inner.checkpoint_sha256

    @property
    def load_meta(self) -> dict:
        return self.inner.load_meta

    def _ensure(self, slow: torch.Tensor, fast: torch.Tensor) -> None:
        if (
            self._slow_gpu is None
            or tuple(self._slow_gpu.shape) != tuple(slow.shape)
            or self._slow_gpu.dtype != slow.dtype
        ):
            self._slow_gpu = torch.empty(slow.shape, device=self.device, dtype=slow.dtype)
            self._fast_gpu = torch.empty(fast.shape, device=self.device, dtype=fast.dtype)
            self._slow_pin = torch.empty(slow.shape, dtype=slow.dtype, pin_memory=True)
            self._fast_pin = torch.empty(fast.shape, dtype=fast.dtype, pin_memory=True)
            self.stats["reallocs"] += 1

    @torch.no_grad()
    def forward_maps(self, slow: torch.Tensor, fast: torch.Tensor) -> PathwayMaps:
        if slow.is_cuda or fast.is_cuda:
            # Already on device — fall through without changing path.
            return self.inner.forward_maps(slow, fast)
        self._ensure(slow, fast)
        assert self._slow_pin is not None and self._fast_pin is not None
        assert self._slow_gpu is not None and self._fast_gpu is not None
        self._slow_pin.copy_(slow)
        self._fast_pin.copy_(fast)
        self._slow_gpu.copy_(self._slow_pin, non_blocking=True)
        self._fast_gpu.copy_(self._fast_pin, non_blocking=True)
        self.stats["h2d_calls"] += 1
        # Inner FrozenSlowFastBlock4.forward_maps does .to(device) again — bypass by
        # calling blocks directly if available.
        if hasattr(self.inner, "blocks"):
            torch.cuda.current_stream().synchronize()  # ensure copies complete before compute
            x: Any = [self._slow_gpu, self._fast_gpu]
            for block in self.inner.blocks:
                x = block(x)
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                raise RuntimeError(f"unexpected_block4_output:{type(x)}")
            return PathwayMaps(slow=x[0], fast=x[1])
        return self.inner.forward_maps(self._slow_gpu, self._fast_gpu)
