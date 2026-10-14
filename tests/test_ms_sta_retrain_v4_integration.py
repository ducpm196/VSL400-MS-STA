"""Integration fixtures for retrain-v4 contracts (no CUDA train, no freeze)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.ms_sta_wordlevel import build_ms_sta
from src.training.retrain_v4 import (
    aux_loss_amendment_required,
    build_optimizer_and_scheduler,
    checkpoint_hyperparameters,
    confirm_plan,
    hpo_plan,
    load_policy,
    refuse_train_until_frozen,
    select_configs_per_variant,
    should_early_stop,
)
from src.training.retrain_v3 import promote_candidate_checkpoint

_ROOT = Path(__file__).resolve().parents[1]
DRAFT = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json"


def test_v4_plans_selection_and_aux_halt() -> None:
    policy = load_policy(DRAFT)
    assert refuse_train_until_frozen(policy)
    trials = hpo_plan(policy)
    assert len(trials) == 48
    assert all(t["early_stopping"] for t in trials)
    assert all(t["phase"] == "hpo" for t in trials)
    b4r = [t for t in trials if t["variant"] == "b4r"]
    assert all(t["residual_init"] == "zero_delta_head" for t in b4r)

    # Synthetic completed HPO rows: each variant one config wins.
    rows = []
    for variant in ("b0", "b3", "b4", "b4r"):
        for cfg, top1 in (("lr3e-4_wd0.01", 50.0), ("lr1e-4_wd0.01", 40.0)):
            for seed in (11, 29):
                rows.append(
                    {
                        "variant": variant,
                        "config_id": cfg,
                        "seed": seed,
                        "best_val_top1": top1 + (1.0 if variant == "b4r" and cfg.startswith("lr3") else 0.0),
                        "best_val_top5": top1 + 5.0,
                    }
                )
    # Make b4r lose to b3 on the winning config to trigger aux halt.
    for seed in (11, 29):
        for r in rows:
            if r["variant"] == "b4r" and r["config_id"] == "lr3e-4_wd0.01" and r["seed"] == seed:
                r["best_val_top1"] = 45.0
            if r["variant"] == "b3" and r["config_id"] == "lr3e-4_wd0.01" and r["seed"] == seed:
                r["best_val_top1"] = 50.0

    selected = select_configs_per_variant(rows)
    assert selected["b3"] == "lr3e-4_wd0.01"
    assert selected["b4r"] == "lr3e-4_wd0.01"
    aux = aux_loss_amendment_required(rows, selected)
    assert aux["triggered"] is True
    assert aux["action_if_triggered"] == "halt_campaign_for_policy_amendment"

    conf = confirm_plan(policy, selected_config_by_variant=selected)
    assert len(conf) == 12
    assert all(t["early_stopping"] is False for t in conf)
    assert all(t["patience"] is None for t in conf)


def test_v4_adamw_scheduler_residual_init_and_promote(tmp_path: Path) -> None:
    assert should_early_stop(bad_epochs=5, patience=5, early_stopping=True) is True
    assert should_early_stop(bad_epochs=100, patience=5, early_stopping=False) is False
    assert should_early_stop(bad_epochs=5, patience=None, early_stopping=True) is False

    model = build_ms_sta("b4r", num_classes=8, residual_init="zero_delta_head")
    opt, sched, meta = build_optimizer_and_scheduler(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
        epochs=2,
        steps_per_epoch=4,
        torch_module=torch,
    )
    assert meta["optimizer"] == "AdamW"
    assert meta["scheduler"] == "linear_warmup_1_epoch_then_cosine"
    assert abs(opt.param_groups[0]["lr"] - 3e-4 * 0.01) < 1e-12
    for _ in range(4):
        opt.step()
        sched.step()
    # After warmup epoch, LR should leave the start_factor regime.
    assert opt.param_groups[0]["lr"] > 3e-4 * 0.01

    hp = checkpoint_hyperparameters(
        epochs=16,
        batch_size=32,
        lr=3e-4,
        weight_decay=0.01,
        label_smoothing=0.05,
        grad_clip=1.0,
        patience=None,
        early_stopping=False,
        residual_init="zero_delta_head",
        skel_temporal_stride=1,
    )
    assert hp["residual_init"] == "zero_delta_head"
    assert hp["early_stopping"] is False
    assert hp["scheduler"] == "linear_warmup_1_epoch_then_cosine"

    cand = tmp_path / "cand.pth"
    final = tmp_path / "cand__final.pth"
    torch.save(
        {
            "checkpoint_status": "candidate",
            "residual_init": "zero_delta_head",
            "selection": "val_top1",
            "model": model.state_dict(),
        },
        cand,
    )
    promo = promote_candidate_checkpoint(cand, final)
    assert promo["final_sha256"]
    assert promo["source_candidate_sha256"]
    loaded = torch.load(final, map_location="cpu", weights_only=False)
    assert loaded["checkpoint_status"] == "final_selected_checkpoint"
    assert loaded["residual_init"] == "zero_delta_head"


def test_v4_draft_policy_still_refuses_train_phase() -> None:
    import subprocess
    import sys

    script = _ROOT / "scripts/run_ms_sta_retrain_v4.py"
    phase = subprocess.run(
        [sys.executable, str(script), "--phase", "hpo"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert phase.returncode == 2
    assert "train_allowed" in phase.stdout

    dry = subprocess.run(
        [sys.executable, str(script), "--dry-plan"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode == 0
    payload = json.loads(dry.stdout)
    assert payload["hpo"]["runs_total"] == 48
    assert payload["confirm"]["n_runs"] == 12
    assert payload["confirm"]["early_stopping_allowed"] is False
    assert payload["train_refused"]
