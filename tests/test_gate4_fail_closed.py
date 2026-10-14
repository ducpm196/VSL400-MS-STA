"""Gate 4 automated fail-closed checks + resolver view safety."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_health_503_when_not_ready(monkeypatch):
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


def test_predict_sanitized_500_on_corrupt_video(monkeypatch, tmp_path):
    """Corrupt/unreadable video must not leak absolute paths to the client."""
    import src.deployment.backend_api as api
    from src.models.isolated_sign_classifier import IsolatedSignClassifier

    class _BoomClf:
        num_classes = 400

        def predict(self, path, top_k=3):
            raise RuntimeError(f"moov atom not found: {path}")

        def predict_windows(self, path, stride=16, conf_min=0.12):
            raise RuntimeError(f"moov atom not found: {path}")

    bad = tmp_path / "019442.mp4"
    bad.write_bytes(b"not-a-video")

    client = TestClient(api.app)
    monkeypatch.setattr(api, "isolated_clf", _BoomClf())
    monkeypatch.setattr(api, "_metadata_df", pd.DataFrame([{"video_id": "019442", "dataset": "VSL400"}]))
    monkeypatch.setattr(api, "_video_index", {"019442": 0})

    class _Res:
        def resolve(self, row, view=None):
            return bad

    monkeypatch.setattr(api, "_resolver", _Res())
    resp = client.post("/predict_clip", json={"video_id": "019442", "mode": "isolated", "top_k": 3})
    assert resp.status_code == 500
    detail = str(resp.json().get("detail", ""))
    assert "019442.mp4" not in detail
    assert str(tmp_path) not in detail
    assert "Inference failed" in detail


def test_gloss_llm_fail_closed_without_gemini_key(monkeypatch):
    from src.deployment import gloss_llm

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    # Force missing key path
    monkeypatch.setattr(gloss_llm, "gemini_api_key", lambda: None)
    out = gloss_llm.compose_vietnamese_from_glosses(["XIN_CHAO"], vietnamese_words=["xin chào"])
    assert out["ok"] is False
    assert out.get("suggestion_vietnamese_text") is None


def test_vsl400_resolve_no_front_fallback_for_side_views(tmp_path):
    from src.data_prep.raw_video_resolve import RawVideoResolver

    root = tmp_path
    vsl = root / "data" / "raw" / "VSL-Cropped" / "raw" / "raw" / "VSL400" / "Part_1" / "split_1"
    front = vsl / "front_view"
    front.mkdir(parents=True)
    (front / "abc.mp4").write_bytes(b"x")
    # intentionally no left_view/abc.mp4

    resolver = RawVideoResolver(root)
    row = {"dataset": "VSL400", "video_id": "abc", "signer_id": "signer_1"}
    assert resolver.resolve(row, view="front") is not None
    assert resolver.resolve(row, view="left") is None
    assert resolver.resolve(row, view="right") is None
    # Mapping protocol must also accept legacy Series-like objects when present.
    row_series = pd.Series(row)
    assert resolver.resolve(row_series, view="front") is not None
    assert resolver.resolve(row_series, view="left") is None


def test_train_default_final_test_is_off():
    text = (_ROOT / "src" / "training" / "train_vsl400_wordlevel.py").read_text(encoding="utf-8")
    # Ensure default is validation-only (Gate 3 owns the test pass).
    assert '"--final-test"' in text or "'--final-test'" in text
    assert "default=False" in text
    assert "deferred_to_evaluate_vsl400" in text
