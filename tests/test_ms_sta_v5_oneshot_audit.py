"""Negative / integrity tests for v5 one-shot audit (no locked-artifact mutation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.ms_sta_v5_oneshot_audit import OneshotAuditError, load_preds_strict, audit_oneshot

_ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_sample_id_rejected(tmp_path: Path) -> None:
    row = {
        "sample_id": "000001",
        "signer": "signer_24",
        "label": 1,
        "pred": 1,
        "correct": True,
        "top5_hit": True,
        "top5": [1, 2, 3, 4, 5],
    }
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"rows": [row, dict(row)]}), encoding="utf-8")
    with pytest.raises(OneshotAuditError, match="duplicate_sample_id"):
        load_preds_strict(path)


def test_top5_flag_mismatch_rejected(tmp_path: Path) -> None:
    row = {
        "sample_id": "000001",
        "signer": "signer_24",
        "label": 9,
        "pred": 1,
        "correct": False,
        "top5_hit": True,  # label 9 not in top5
        "top5": [1, 2, 3, 4, 5],
    }
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    with pytest.raises(OneshotAuditError, match="top5_flag_mismatch"):
        load_preds_strict(path)


def test_correct_flag_mismatch_rejected(tmp_path: Path) -> None:
    row = {
        "sample_id": "000001",
        "signer": "signer_24",
        "label": 1,
        "pred": 2,
        "correct": True,
        "top5_hit": False,
        "top5": [2, 3, 4, 5, 6],
    }
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    with pytest.raises(OneshotAuditError, match="correct_flag_mismatch"):
        load_preds_strict(path)


@pytest.mark.requires_artifacts
def test_live_oneshot_audit_pass() -> None:
    audit = audit_oneshot(_ROOT)
    assert audit["status"] == "AUDIT_PASS"
    assert audit["b4_minus_mlp"]["discordant_top1"]["b4_only_correct"] == 935
    assert audit["b4_minus_mlp"]["signer_27_top5_delta_pp"] == pytest.approx(-3.0)
    assert audit["b4_minus_mlp"]["top1_improved_all_signers"] is True
    assert audit["b4_minus_mlp"]["top5_improved_all_signers"] is False


def test_completion_artifact_immutable_exists() -> None:
    path = _ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_COMPLETION.json"
    assert path.is_file()
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert obj["status"] == "COMPLETE"
    assert obj["defense_ready"] is False
    assert obj["pristine_blind_claim"] is False
