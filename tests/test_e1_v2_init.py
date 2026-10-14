"""E1-v2 init: Gate3 classifier loads into self.cls for compatible variants."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from src.models.isolated_sign_classifier import DEFAULT_CKPT
from src.models.ms_sta_wordlevel import build_ms_sta
from src.training.e1_policy_enforce import (
    INIT_GATE3_PROJ_COMPAT_CLS,
    apply_initialization,
)


pytestmark = pytest.mark.requires_artifacts


def test_compatible_classifier_loads_into_cls() -> None:
    ckpt_path = _ROOT / DEFAULT_CKPT
    assert ckpt_path.is_file()
    model = build_ms_sta("b1", num_classes=400)
    before = model.cls[3].weight.detach().clone()
    mode = apply_initialization(
        model,
        init_mode=INIT_GATE3_PROJ_COMPAT_CLS,
        gate3_ckpt_path=ckpt_path,
    )
    assert mode == INIT_GATE3_PROJ_COMPAT_CLS
    assert not torch.allclose(model.cls[3].weight.detach(), before)
    gate = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = gate["head"] if isinstance(gate, dict) and "head" in gate else gate
    assert torch.allclose(model.cls[3].weight.detach(), head["cls.3.weight"])
    assert torch.allclose(model.rgb_proj.weight.detach(), head["proj.weight"])


def test_b0_still_targets_b0_cls() -> None:
    ckpt_path = _ROOT / DEFAULT_CKPT
    model = build_ms_sta("b0", num_classes=400)
    apply_initialization(
        model,
        init_mode="gate3_rgb_proj_and_classifier",
        gate3_ckpt_path=ckpt_path,
    )
    gate = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = gate["head"] if isinstance(gate, dict) and "head" in gate else gate
    assert torch.allclose(model.b0_cls[3].weight.detach(), head["cls.3.weight"])
