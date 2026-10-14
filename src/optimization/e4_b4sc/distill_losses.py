# -*- coding: utf-8 -*-
"""Distillation losses for E4_B4SC E2/E3 (policy-locked coefficients)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_logit_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(temperature)
    log_p = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)


def stage_feature_loss(student_adapted: list[torch.Tensor], teacher_feats: list[torch.Tensor]) -> torch.Tensor:
    losses = []
    for s, t in zip(student_adapted, teacher_feats):
        s_n = F.normalize(s.flatten(2), dim=1)
        t_n = F.normalize(t.flatten(2).detach(), dim=1)
        losses.append(F.mse_loss(s_n, t_n))
    return torch.stack(losses).mean()


def roi_region_loss(
    student_tokens: dict[str, torch.Tensor],
    teacher_tokens: dict[str, torch.Tensor],
    regions: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    per = {}
    losses = []
    for r in regions:
        s = F.normalize(student_tokens[r].flatten(1), dim=-1)
        t = F.normalize(teacher_tokens[r].flatten(1).detach(), dim=-1)
        # cosine distance
        loss = (1.0 - (s * t).sum(dim=-1)).mean()
        per[r] = float(loss.detach().cpu())
        losses.append(loss)
    return torch.stack(losses).mean(), per


def temporal_delta_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """student/teacher: (B, L, C) token sequences."""
    if student.shape[1] < 2:
        return student.new_zeros(())
    ds = student[:, 1:] - student[:, :-1]
    dt = teacher[:, 1:] - teacher[:, :-1]
    ds = F.normalize(ds.flatten(2), dim=-1)
    dt = F.normalize(dt.flatten(2).detach(), dim=-1)
    return F.mse_loss(ds, dt)
