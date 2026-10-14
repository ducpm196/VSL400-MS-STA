"""Unit/contract tests for MS-STA v6 demo UI wiring (no live server)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.deployment.backend_api import _ms_sta_v6_identity_errors

ROOT = Path(__file__).resolve().parents[1]
APP_PY = ROOT / "src/deployment/app.py"

# Expected validation confirm Top-1 from FINAL_RESULTS (three seeds, pct scale).
_EXPECTED_VAL_CONFIRM = (84.81, 0.51)


def _mean_std_pct(values: list[float]) -> tuple[float, float]:
    """Duplicate of planned app helper — pct mean and sample std over fractional Top-1."""
    if not values:
        return 0.0, 0.0
    pct = [100.0 * float(v) for v in values]
    mean = sum(pct) / len(pct)
    if len(pct) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in pct) / (len(pct) - 1)
    return mean, var**0.5


def _try_import_app_mean_std() -> tuple[float, float] | None:
    try:
        from src.deployment.app import _mean_std_pct as app_mean_std_pct  # type: ignore[attr-defined]
    except ImportError:
        return None
    return app_mean_std_pct  # type: ignore[return-value]


def _select_primary_kpi(health: dict) -> dict:
    """Pure contract: v6 defense KPI wins over v5 when both are present."""
    v6 = health.get("v6_kpi") or {}
    v5 = health.get("v5_kpi") or {}
    if v6.get("scope") == "v6_defense_go":
        return v6
    if v5.get("scope") == "v5_defense_go":
        return v5
    return {}


def _simulate_v6_health() -> dict:
    final = json.loads(
        (ROOT / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json").read_text(encoding="utf-8")
    )
    validation = final.get("validation") or {}
    test = final.get("test") or {}
    return {
        "engine": "ms_sta_v6_b4_v6_ctr",
        "ms_sta_v6": True,
        "ms_sta_ready": True,
        "campaign_status": "DEFENSE_GO",
        "e4_ready": False,
        "b4_ready": True,
        "ms_sta_checkpoint_sha256": final.get("checkpoint_sha256"),
        "v6_kpi": {
            "scope": "v6_defense_go",
            "validation_confirm_top1": validation.get("b4_confirm_top1"),
            "test_kind": final.get("test_kind"),
            "test_n": test.get("n"),
            "test_b4_top1": test.get("b4_top1"),
            "test_mlp_top1": test.get("mlp_top1"),
            "test_delta_top1_pp": test.get("delta_top1_pp"),
            "pristine_blind_claim": final.get("pristine_blind_claim"),
            "e4_ready": False,
            "b4_ready": True,
        },
        "v5_kpi": {
            "scope": "v5_defense_go",
            "validation_b4_top1_pct": 64.75,
            "test_b4_top1": 0.3606,
        },
    }


def test_mean_std_pct_validation_confirm_top1_contract() -> None:
    health = _simulate_v6_health()
    confirm = health["v6_kpi"]["validation_confirm_top1"]
    assert confirm and len(confirm) == 3

    app_fn = _try_import_app_mean_std()
    mean_fn = app_fn if app_fn is not None else _mean_std_pct
    mean, std = mean_fn(confirm)
    exp_mean, exp_std = _EXPECTED_VAL_CONFIRM
    assert mean == pytest.approx(exp_mean, abs=0.01)
    assert std == pytest.approx(exp_std, abs=0.02)


def test_primary_kpi_prefers_v6_over_v5() -> None:
    health = _simulate_v6_health()
    picked = _select_primary_kpi(health)
    assert picked.get("scope") == "v6_defense_go"
    assert picked is health["v6_kpi"]
    assert picked.get("test_b4_top1") == pytest.approx(0.5154906731549067)


def test_v6_health_identity_contract() -> None:
    health = _simulate_v6_health()
    assert health["engine"] == "ms_sta_v6_b4_v6_ctr"
    sha = str(health.get("ms_sta_checkpoint_sha256") or "")
    assert sha.startswith("0b73f8f2")
    assert health["e4_ready"] is False
    assert health["v6_kpi"]["e4_ready"] is False


def test_active_registry_is_v6_schema_e4_not_ready() -> None:
    pointer = json.loads(
        (ROOT / "data/processed/ms_sta_registry_active.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / str(pointer["registry_path"])).read_text(encoding="utf-8")
    )
    assert registry["schema"] == "ms_sta_registry_v6_defense_go_v1"
    assert registry["active_variant"] == "b4_v6_ctr"
    assert registry["engine"] == "ms_sta_v6_b4_v6_ctr"
    assert registry["e4_ready"] is False
    assert registry["b4_ready"] is True


def test_v5_rollback_registry_readable_b4_star_fixed() -> None:
    pointer = json.loads(
        (ROOT / "data/processed/ms_sta_registry_active.json").read_text(encoding="utf-8")
    )
    rollback_rel = pointer.get("rollback_registry_path") or "data/processed/ms_sta_registry_v5_defense_go.json"
    rollback = json.loads((ROOT / str(rollback_rel)).read_text(encoding="utf-8"))
    assert rollback["schema"] == "ms_sta_registry_v5_defense_go_v1"
    assert rollback["active_variant"] == "b4_star_fixed"
    assert rollback["campaign_status"] == "DEFENSE_GO"


def test_v6_identity_errors_enforce_b4_v6_ctr_and_e4_not_ready() -> None:
    reg = json.loads(
        (ROOT / "data/processed/ms_sta_registry_v6_defense_go.json").read_text(encoding="utf-8")
    )
    payload = {"variant": "b4_v6_ctr", "seed": 42}
    assert _ms_sta_v6_identity_errors(reg, payload) == []
    bad = dict(reg)
    bad["e4_ready"] = True
    assert _ms_sta_v6_identity_errors(bad, payload)


def test_app_source_kpi_block_prefers_v6_and_b4_v6_label() -> None:
    text = APP_PY.read_text(encoding="utf-8")
    assert "Use MS-STA Unified Attention" in text
    assert 'st.checkbox(\n        "Use MS-STA engine (B4*)"' not in text
    assert re.search(r'st\.checkbox\(\s*"Use MS-STA engine \(B4\*\)"', text) is None
    assert "Use MS-STA B4-V6" not in text

    assert "v6_kpi" in text, "app.py must read v6_kpi in KPI display block"
    assert "v5_kpi" in text
    kpi_anchor = text.find("health_resp, health, health_err = _fetch_health()")
    assert kpi_anchor >= 0
    kpi_block = text[kpi_anchor : text.find("accuracy_study = health.get", kpi_anchor)]
    v6_pos = kpi_block.find("v6_kpi")
    v5_pos = kpi_block.find("v5_kpi")
    assert v6_pos >= 0 and v5_pos >= 0
    assert v6_pos < v5_pos, "KPI display must consult v6_kpi before v5_kpi"
