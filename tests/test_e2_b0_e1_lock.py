"""E2 lock: B0-E1 required; Gate3 substitution refused; validate-lock has no test KPI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_eval_module():
    path = _ROOT / "scripts" / "evaluate_ms_sta_matrix.py"
    spec = importlib.util.spec_from_file_location("evaluate_ms_sta_matrix", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_refuse_gate3_as_b0_helper() -> None:
    ev = _load_eval_module()
    # Basename contract — does not require the .pth bytes on disk
    err = ev._refuse_gate3_as_b0(Path("data/processed/vsl400_classifier.pth"))
    assert err is not None
    assert "Gate3" in err or "DEFAULT_CKPT" in err or "demo" in err.lower()


def test_e2_refuses_missing_b0_checkpoint(tmp_path, monkeypatch) -> None:
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
                # b0_checkpoint intentionally missing
                "eligible_sample_manifest_sha256": "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522",
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
            "--ckpts-dir",
            str(tmp_path),
        ],
    )
    assert ev.main() == 2


@pytest.mark.requires_artifacts
def test_e2_refuses_gate3_basename_as_b0(tmp_path, monkeypatch) -> None:
    ev = _load_eval_module()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    # Point b0_checkpoint at Gate3 demo basename — must fail.
    gate3 = _ROOT / "data" / "processed" / "vsl400_classifier.pth"
    if not gate3.is_file():
        pytest.skip("Gate3 checkpoint missing")
    linked = ckpt_dir / "vsl400_classifier.pth"
    linked.write_bytes(gate3.read_bytes())
    # Minimal active ckpt so failure is specifically Gate3-B0
    import torch

    torch.save(
        {
            "model": {},
            "variant": "b4",
            "seed": 42,
            "e1_selected": True,
            "evaluation_locked": True,
            "schema_version": 2,
            "sample_manifest_sha256": "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522",
            "source_checkpoint_sha256": "x" * 64,
        },
        ckpt_dir / "b4_seed42.pth",
    )
    digest_b4 = __import__("hashlib").sha256((ckpt_dir / "b4_seed42.pth").read_bytes()).hexdigest()
    digest_b0 = __import__("hashlib").sha256(linked.read_bytes()).hexdigest()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_pass": True,
                "e1_selected": True,
                "evaluation_locked": True,
                "active_variant": "b4",
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": digest_b4,
                "b0_checkpoint": "vsl400_classifier.pth",
                "b0_checkpoint_sha256": digest_b0,
                "eligible_sample_manifest_sha256": "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522",
                "defense_ready": False,
                "e2_pass": False,
                "status": "NO-GO",
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
            "--ckpts-dir",
            str(ckpt_dir),
        ],
    )
    assert ev.main() == 2


def test_e2_refuses_allow_gate3_flag(tmp_path, monkeypatch) -> None:
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
                "b0_checkpoint": "b0_seed42.pth",
                "b0_checkpoint_sha256": "def",
                "eligible_sample_manifest_sha256": "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522",
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
            "--allow-gate3-b0",
            "--device",
            "cpu",
        ],
    )
    assert ev.main() == 2


def test_validate_lock_reports_errors_without_test_kpi(tmp_path, monkeypatch, capsys) -> None:
    ev = _load_eval_module()
    reg = tmp_path / "ms_sta_registry.json"
    reg.write_text(
        json.dumps(
            {
                "e1_pass": False,
                "e1_selected": False,
                "evaluation_locked": False,
                "defense_ready": False,
                "e2_pass": False,
                "status": "NO-GO",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ev, "REGISTRY", reg)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_ms_sta_matrix.py", "--validate-lock", "--ckpts-dir", str(tmp_path), "--device", "cpu"],
    )
    rc = ev.main()
    assert rc == 2
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert payload["gate"] == "VALIDATE_LOCK"
    assert payload["test_kpi_computed"] is False
    assert payload["ok"] is False
