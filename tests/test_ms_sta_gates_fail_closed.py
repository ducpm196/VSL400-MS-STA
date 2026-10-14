"""Fail-closed MS-STA gate summarizer with active-registry authority."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_summarizer():
    path = _ROOT / "scripts" / "summarize_ms_sta_gates.py"
    spec = importlib.util.spec_from_file_location("summarize_ms_sta_gates", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.requires_artifacts
def test_summarizer_defense_go_on_full_workspace() -> None:
    mod = _load_summarizer()
    assert mod.main() == 0
    status = json.loads((_ROOT / "logs/ms_sta_gates/STATUS.json").read_text(encoding="utf-8"))
    assert status["verdict"] == "DEFENSE_GO"
    assert status["defense_ready"] is True
    assert status["demo_allowed"] is True
    assert status.get("current_best_variant") == "b4_v6_ctr"
    assert status.get("e4_ready") is False
    assert status["contradictions"] == []
    # E4 efficiency gate is intentionally False (appendix NO-GO).
    for name, gate in status["gates"].items():
        if name == "e4_efficiency":
            assert gate["pass"] is False
        else:
            assert gate["pass"] is True

def test_summarizer_refuses_historical_go_without_active_pointer(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "proj"
    hist = _ROOT / "data/processed/ms_sta_registry.json"
    if not hist.is_file():
        pytest.skip("missing historical registry")
    dest = root / "data/processed/ms_sta_registry.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hist, dest)
    mod = _load_summarizer()
    monkeypatch.setattr(mod, "_ROOT", root)
    assert mod.main() == 2
    status = json.loads((root / "logs/ms_sta_gates/STATUS.json").read_text(encoding="utf-8"))
    assert status["verdict"] == "NO-GO"
    assert any("active" in c.lower() or "historical" in c.lower() for c in status["contradictions"])
    assert status["verdict"] != "GO"


def test_summarizer_binds_active_no_go_registry(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "proj"
    for rel in (
        "data/processed/ms_sta_registry_active.json",
        "data/processed/ms_sta_registry_v4_validation_no_go.json",
        "Docs/results/ms_sta_retrain_v4/CONFIRM_VALIDATION_REPORT.json",
        "Docs/results/ms_sta_retrain_v4/VALIDATION_METRICS.json",
        "Docs/results/ms_sta_retrain_v4/VALIDATION_NO_GO.json",
    ):
        src = _ROOT / rel
        if not src.is_file():
            pytest.skip(f"missing {rel}")
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    # Rewrite pointer sha to match copied registry bytes.
    reg = root / "data/processed/ms_sta_registry_v4_validation_no_go.json"
    ptr = {
        "schema": "ms_sta_registry_active_pointer_v1",
        "registry_path": "data/processed/ms_sta_registry_v4_validation_no_go.json",
        "registry_sha256": _sha(reg),
        "campaign_status": "VALIDATION_NO_GO",
        "defense_ready": False,
        "demo_allowed": False,
        "historical_fallback_disabled": True,
    }
    (root / "data/processed/ms_sta_registry_active.json").write_text(
        json.dumps(ptr, indent=2) + "\n", encoding="utf-8"
    )
    mod = _load_summarizer()
    monkeypatch.setattr(mod, "_ROOT", root)
    assert mod.main() == 2
    status = json.loads((root / "logs/ms_sta_gates/STATUS.json").read_text(encoding="utf-8"))
    assert status["verdict"] == "VALIDATION_NO_GO"
    assert status["registry_source"] == "active"
    assert status.get("closeout_campaign") == "retrain_v4"


def test_immutable_historical_e2_d4_sha_still_tracked() -> None:
    """Raw historical evidence bytes remain for provenance (not live verdict)."""
    assert _sha(_ROOT / "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json").startswith("408fc077")
    assert _sha(_ROOT / "Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json").startswith("f5375f22")
    assert _sha(_ROOT / "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json").startswith("69aed785")
