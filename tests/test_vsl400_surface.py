"""Unit tests for VSL400 eval sampling helpers and API readiness contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_eval_helpers():
    path = _ROOT / "scripts" / "evaluate_vsl400.py"
    spec = importlib.util.spec_from_file_location("evaluate_vsl400", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_sample_modes_cover_classes():
    mod = _load_eval_helpers()
    rows = []
    for gloss_i in range(10):
        for j in range(5):
            rows.append(
                {
                    "text_gloss": f"G{gloss_i}",
                    "signer_id": f"signer_{j}",
                    "video_id": f"{gloss_i:03d}{j}",
                }
            )
    df = pd.DataFrame(rows)
    first = mod._sample_test(df, 8, "first", 42)
    assert len(first) == 8
    assert list(first["video_id"]) == list(df.iloc[:8]["video_id"])

    stratified = mod._sample_test(df, 10, "stratified", 42)
    assert len(stratified) == 10
    assert stratified["text_gloss"].nunique() == 10

    rnd = mod._sample_test(df, 12, "random", 42)
    assert len(rnd) == 12


def test_health_not_ready_without_classifier(monkeypatch):
    import src.deployment.backend_api as api

    client = TestClient(api.app)
    monkeypatch.setattr(api, "isolated_clf", None)
    monkeypatch.setattr(api, "_metadata_df", None)
    monkeypatch.setattr(api, "_resolver", None)
    monkeypatch.setattr(api, "_video_index", None)
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["status"] == "not_ready"


def test_predict_rejects_removed_modes():
    from src.deployment import backend_api as api

    client = TestClient(api.app)
    resp = client.post(
        "/predict_clip",
        json={"video_id": "x", "mode": "assist"},
    )
    assert resp.status_code == 400
