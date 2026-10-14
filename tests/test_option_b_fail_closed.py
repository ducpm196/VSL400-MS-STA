"""Fail-closed Option B: MS-STA request, E2 unlock, RGB temporal contract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_features import (
    FIXED_RGB_L,
    MIN_TEMPORAL_L,
    fix_rgb_length,
    to_rgb_tensor,
)


def _load_eval_module():
    path = _ROOT / "scripts" / "evaluate_ms_sta_matrix.py"
    spec = importlib.util.spec_from_file_location("evaluate_ms_sta_matrix", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fix_rgb_rejects_l1() -> None:
    with pytest.raises(ValueError, match="raw temporal L=1"):
        fix_rgb_length(np.random.randn(1, 2304).astype(np.float32), FIXED_RGB_L)


def test_fix_rgb_length_pad_mask_not_last_repeat() -> None:
    short = np.random.randn(3, 2304).astype(np.float32)
    fixed, valid = fix_rgb_length(short, FIXED_RGB_L)
    assert fixed.shape == (FIXED_RGB_L, 2304)
    assert valid.dtype == bool
    assert int(valid.sum()) == 3
    assert np.allclose(fixed[:3], short)
    # Pads are zeros — not repeated last token
    assert np.allclose(fixed[3:], 0.0)
    assert not np.allclose(fixed[-1], short[-1])

    long = np.random.randn(20, 2304).astype(np.float32)
    fixed2, valid2 = fix_rgb_length(long, FIXED_RGB_L)
    assert fixed2.shape == (FIXED_RGB_L, 2304)
    assert bool(valid2.all())


def test_to_rgb_tensor_pooled_keeps_l1() -> None:
    pooled = np.random.randn(2304).astype(np.float32)
    t, mask = to_rgb_tensor(pooled[None, :], fixed_l=None)
    assert t.shape == (1, 2304)
    assert mask is None


def test_ms_sta_409_when_requested_but_unavailable(monkeypatch, tmp_path) -> None:
    import src.deployment.backend_api as api

    class _OkClf:
        num_classes = 400
        id_to_gloss = {0: "A"}
        id_to_vietnamese = {0: "a"}

        def predict(self, path, top_k=3):
            return [{"gloss": "A", "vietnamese": "a", "probability": 1.0, "class_id": 0}]

    vid = tmp_path / "x.mp4"
    vid.write_bytes(b"fake")

    client = TestClient(api.app)
    monkeypatch.setattr(api, "isolated_clf", _OkClf())
    monkeypatch.setattr(api, "_ms_sta_model", None)
    monkeypatch.setattr(api, "_ms_sta_variant", None)
    monkeypatch.setattr(api, "_metadata_df", pd.DataFrame([{"video_id": "x", "dataset": "VSL400"}]))
    monkeypatch.setattr(api, "_video_index", {"x": 0})

    class _Res:
        def resolve(self, row, view=None):
            return vid

    monkeypatch.setattr(api, "_resolver", _Res())
    resp = client.post(
        "/predict_clip",
        json={"video_id": "x", "mode": "isolated", "top_k": 1, "use_ms_sta": True},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "defense" in detail.lower() or "candidate" in detail.lower() or "NO-GO" in detail or "not loaded" in detail.lower()


def test_evaluate_refuses_e2_without_confirm(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps({"defense_ready": False, "e1_pass": False, "e1_selected": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_ms_sta_matrix.py", "--split", "test", "--limit", "0", "--device", "cpu", "--variants", "b0"],
    )
    assert ev.main() == 2


def test_evaluate_refuses_e2_on_val_split(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_selected": True,
                "evaluation_locked": True,
                "active_variant": "b4",
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": "abc",
                "defense_ready": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ms_sta_matrix.py",
            "--gate-label",
            "E2",
            "--split",
            "val",
            "--limit",
            "0",
            "--confirm-e2",
            "--device",
            "cpu",
            "--variants",
            "b0",
        ],
    )
    assert ev.main() == 2


def test_evaluate_refuses_legacy_e1_pass_without_e1_selected(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_pass": True,
                "e2_unlocked": True,
                "e1_selected": False,
                "evaluation_locked": False,
                "active_variant": "b4",
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": "abc",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ms_sta_matrix.py",
            "--split",
            "test",
            "--limit",
            "0",
            "--confirm-e2",
            "--device",
            "cpu",
        ],
    )
    assert ev.main() == 2


def test_evaluate_refuses_pilot_dir_as_e2(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps({"e1_selected": True, "evaluation_locked": True, "defense_ready": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    pilot = tmp_path / "ms_sta_pilot_nogo"
    pilot.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ms_sta_matrix.py",
            "--split",
            "test",
            "--limit",
            "0",
            "--confirm-e2",
            "--ckpts-dir",
            str(pilot),
            "--allow-pilot-eval",
            "--device",
            "cpu",
            "--variants",
            "b0",
        ],
    )
    assert ev.main() == 2


def test_evaluate_e2_requires_checkpoint_and_sha(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_selected": True,
                "evaluation_locked": True,
                "active_variant": "b4",
                "checkpoint": None,
                "checkpoint_sha256": None,
                "defense_ready": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ms_sta_matrix.py",
            "--split",
            "test",
            "--limit",
            "0",
            "--confirm-e2",
            "--device",
            "cpu",
            "--variants",
            "b0,b1,b2,b3,b4,b5",
        ],
    )
    assert ev.main() == 2


def test_evaluate_e2_requires_eligible_manifest_sha(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_selected": True,
                "evaluation_locked": True,
                "active_variant": "b4",
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": "abc",
                "defense_ready": False,
                # missing eligible_sample_manifest_sha256
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ms_sta_matrix.py",
            "--split",
            "test",
            "--limit",
            "0",
            "--confirm-e2",
            "--device",
            "cpu",
        ],
    )
    assert ev.main() == 2


def test_evaluate_e2_requires_both_payload_lifecycle_flags(tmp_path, monkeypatch) -> None:
    """Payload must have BOTH e1_selected and evaluation_locked (not OR)."""
    ev = _load_eval_module()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    path = ckpt_dir / "b4_seed42.pth"
    # Minimal torch save without importing torch at module level if unavailable
    import torch

    torch.save(
        {
            "model": {},
            "variant": "b4",
            "e1_selected": True,
            # evaluation_locked intentionally missing / false
            "evaluation_locked": False,
        },
        path,
    )
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_selected": True,
                "evaluation_locked": True,
                "active_variant": "b4",
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": digest,
                "defense_ready": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ms_sta_matrix.py",
            "--split",
            "test",
            "--limit",
            "0",
            "--confirm-e2",
            "--ckpts-dir",
            str(ckpt_dir),
            "--device",
            "cpu",
        ],
    )
    assert ev.main() == 2


def test_evaluate_e2_refuses_without_active_variant(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_selected": True,
                "evaluation_locked": True,
                "active_variant": None,
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": "abc",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_ms_sta_matrix.py",
            "--split",
            "test",
            "--limit",
            "0",
            "--confirm-e2",
            "--device",
            "cpu",
        ],
    )
    assert ev.main() == 2


def test_clean_d1_requires_output_dir() -> None:
    path = _ROOT / "src" / "data_prep" / "extract_and_smooth.py"
    text = path.read_text(encoding="utf-8")
    assert "--clean-d1" in text
    assert "--output-dir" in text
    assert "--sample-manifest" in text
    assert "_run_contract.json" in text or "write_run_contract" in text


def test_temporal_build_requires_output_dir() -> None:
    text = (_ROOT / "scripts" / "build_rgb_temporal_cache.py").read_text(encoding="utf-8")
    assert "sample-manifest" in text
    assert "write_run_contract" in text
    assert "require_resume_contract" in text
