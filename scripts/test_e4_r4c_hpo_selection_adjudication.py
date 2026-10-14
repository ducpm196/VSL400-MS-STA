# -*- coding: utf-8 -*-
"""Unit tests for E4-R4C HPO grouped selection adjudication."""
from __future__ import annotations

import math
import unittest

from scripts.e4_r4c_hpo_boundary_audit_and_adjudicate import (
    group_hparams,
    rank_groups,
    validate_group_seeds,
)


def _row(lr: float, wd: float, seed: int, t1: float, t5: float, cid: str | None = None) -> dict:
    return {
        "config_id": cid or f"c_lr{lr}_wd{wd}_s{seed}",
        "lr": lr,
        "weight_decay": wd,
        "seed": seed,
        "val_top1": t1,
        "val_top5": t5,
    }


class TestHpoAdjudication(unittest.TestCase):
    def test_grouping_and_rank_mean_top1(self) -> None:
        rows = [
            _row(3e-4, 1e-2, 11, 0.2447, 0.55),
            _row(3e-4, 1e-2, 29, 0.2461, 0.55),
            _row(3e-4, 1e-3, 11, 0.2314, 0.52),
            _row(3e-4, 1e-3, 29, 0.2263, 0.52),
        ]
        ranked = rank_groups(group_hparams(rows))
        self.assertEqual(ranked[0]["lr"], 3e-4)
        self.assertEqual(ranked[0]["weight_decay"], 1e-2)
        self.assertAlmostEqual(ranked[0]["mean_top1"], (0.2447 + 0.2461) / 2, places=6)

    def test_missing_seed_raises(self) -> None:
        rows = [_row(3e-4, 1e-2, 11, 0.2, 0.5)]
        with self.assertRaises(ValueError):
            validate_group_seeds(rows)

    def test_duplicate_seed_raises(self) -> None:
        rows = [
            _row(3e-4, 1e-2, 11, 0.2, 0.5, "a"),
            _row(3e-4, 1e-2, 11, 0.21, 0.5, "b"),
        ]
        with self.assertRaises(ValueError):
            validate_group_seeds(rows)

    def test_non_finite_raises(self) -> None:
        rows = [
            _row(3e-4, 1e-2, 11, float("nan"), 0.5),
            _row(3e-4, 1e-2, 29, 0.2, 0.5),
        ]
        with self.assertRaises(ValueError):
            validate_group_seeds(rows)

    def test_tie_break_stable_config_id(self) -> None:
        rows = [
            _row(1e-4, 1e-2, 11, 0.20, 0.40, "zzz_s11"),
            _row(1e-4, 1e-2, 29, 0.20, 0.40, "zzz_s29"),
            _row(1e-4, 1e-3, 11, 0.20, 0.40, "aaa_s11"),
            _row(1e-4, 1e-3, 29, 0.20, 0.40, "aaa_s29"),
        ]
        ranked = rank_groups(group_hparams(rows))
        # equal mean top1/top5 → stable_config_id ascending
        self.assertEqual(ranked[0]["stable_config_id"], "aaa_s11")

    def test_material_mismatch_detection(self) -> None:
        rows = [
            _row(3e-4, 1e-3, 11, 0.30, 0.60),
            _row(3e-4, 1e-3, 29, 0.30, 0.60),
            _row(3e-4, 1e-2, 11, 0.20, 0.50),
            _row(3e-4, 1e-2, 29, 0.20, 0.50),
        ]
        winner = rank_groups(group_hparams(rows))[0]
        self.assertEqual(winner["weight_decay"], 1e-3)
        selection_wd = 1e-2
        self.assertFalse(math.isclose(winner["weight_decay"], selection_wd))


if __name__ == "__main__":
    unittest.main()
