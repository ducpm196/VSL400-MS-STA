# -*- coding: utf-8 -*-
"""Unit tests for E4_SYSKD S3 batched ROI and S1 pinned wrapper shapes."""
from __future__ import annotations

import numpy as np
import torch

from src.models.rgb_token_v3 import REGIONS, _roi_align_token, _roi_align_tokens_batched


def test_roi_batch_matches_sequential_order_and_values() -> None:
    torch.manual_seed(0)
    feat = torch.randn(1, 8, 16, 16)
    boxes = {
        "global": np.asarray([10.0, 10.0, 100.0, 100.0], dtype=np.float32),
        "left_hand": np.asarray([20.0, 30.0, 80.0, 90.0], dtype=np.float32),
        "right_hand": None,
        "face": np.asarray([40.0, 20.0, 120.0, 110.0], dtype=np.float32),
    }
    assert tuple(REGIONS) == ("global", "left_hand", "right_hand", "face")
    batched_tok, batched_miss = _roi_align_tokens_batched(feat, boxes, regions=REGIONS)
    for r in REGIONS:
        seq_tok, seq_miss = _roi_align_token(feat, boxes[r])
        assert bool(batched_miss[r]) == bool(seq_miss)
        if seq_miss:
            assert np.allclose(batched_tok[r], 0.0)
        else:
            assert np.allclose(batched_tok[r], seq_tok, rtol=0, atol=1e-5)


def test_roi_batch_rejects_wrong_region_order() -> None:
    feat = torch.randn(1, 4, 8, 8)
    boxes = {r: None for r in REGIONS}
    try:
        _roi_align_tokens_batched(feat, boxes, regions=("global", "face", "left_hand", "right_hand"))
        assert False, "expected order mismatch"
    except RuntimeError as exc:
        assert "roi_region_order_mismatch" in str(exc)
