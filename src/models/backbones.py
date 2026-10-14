"""Stage-1 backbone placeholders and SlowFast builder for RGB caching."""

from __future__ import annotations

import torch.nn as nn


def build_slowfast_extractor(device: str = "cuda"):
    """Real SlowFast-R50 feature extractor (inference-only, for RGB feature caches)."""
    from src.models.rgb_backbone import SlowFastFeatureExtractor

    return SlowFastFeatureExtractor(device=device)


def build_slowfast_placeholder(num_classes: int = 100) -> nn.Module:
    """Legacy placeholder — use build_slowfast_extractor() for production RGB cache."""
    return nn.Sequential(
        nn.Linear(3 * 32 * 224 * 224, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )


def build_stgcn_placeholder(num_classes: int = 100) -> nn.Module:
    """Skeleton MLP scaffold used by Stage-1 (`train_stage1.py`)."""
    return nn.Sequential(
        nn.Linear(75 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes),
    )
