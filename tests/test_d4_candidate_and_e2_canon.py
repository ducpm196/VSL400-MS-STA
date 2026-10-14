"""D4 candidate load vs defense GO — fail-closed SHA + lifecycle gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_candidate_load_requires_operator_flag(monkeypatch, ms_sta_tmp_checkpoint) -> None:
    import src.deployment.backend_api as api

    src, digest, _payload = ms_sta_tmp_checkpoint
    ckpt_dir = src.parent
    reg = {
        "defense_ready": False,
        "e1_pass": True,
        "e1_selected": True,
        "evaluation_locked": True,
        "e2_pass": True,
        "d4_pass": False,
        "d1_pass": True,
        "d4_candidate_allowed": False,
        "active_variant": "b4",
        "checkpoint": src.name,
        "checkpoint_sha256": digest,
        "ckpts_dir": str(ckpt_dir),
    }
    monkeypatch.delenv("ALLOW_MS_STA_CANDIDATE", raising=False)
    monkeypatch.setattr(api, "_resolve_ms_sta_ckpt_dir", lambda r: ckpt_dir)
    model, variant, mode = api._load_ms_sta_from_registry(reg)
    assert model is None and mode is None

    reg["d4_candidate_allowed"] = True
    model, variant, mode = api._load_ms_sta_from_registry(reg)
    assert model is not None
    assert variant == "b4"
    assert mode == "candidate"


def test_candidate_refuses_sha_mismatch(monkeypatch, ms_sta_tmp_checkpoint) -> None:
    import src.deployment.backend_api as api

    src, _digest, _payload = ms_sta_tmp_checkpoint
    ckpt_dir = src.parent
    reg = {
        "defense_ready": False,
        "e1_selected": True,
        "evaluation_locked": True,
        "e2_pass": True,
        "d1_pass": True,
        "d4_candidate_allowed": True,
        "active_variant": "b4",
        "checkpoint": src.name,
        "checkpoint_sha256": "0" * 64,
        "ckpts_dir": str(ckpt_dir),
    }
    monkeypatch.setattr(api, "_resolve_ms_sta_ckpt_dir", lambda r: ckpt_dir)
    model, variant, mode = api._load_ms_sta_from_registry(reg)
    assert model is None and mode is None


def test_defense_requires_payload_defense_ready(monkeypatch, ms_sta_tmp_checkpoint) -> None:
    import src.deployment.backend_api as api

    src, digest, _payload = ms_sta_tmp_checkpoint
    ckpt_dir = src.parent
    reg = {
        "defense_ready": True,
        "e1_pass": True,
        "e2_pass": True,
        "d4_pass": True,
        "d1_pass": True,
        "active_variant": "b4",
        "checkpoint": src.name,
        "checkpoint_sha256": digest,
    }
    monkeypatch.setattr(api, "_resolve_ms_sta_ckpt_dir", lambda r: ckpt_dir)
    # Fixture payload has defense_ready=false → defense path must refuse
    model, variant, mode = api._load_ms_sta_from_registry(reg)
    assert model is None and mode is None


def test_canonical_e2_report_exists_and_binds_source() -> None:
    rep = json.loads(
        (_ROOT / "Docs" / "results" / "ms_sta_e2" / "E2_CANONICAL_REPORT.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert rep["scientific_pass"] is True
    # Immutable E2-time artifact may still say NO-GO; post-D4 SoT is FINAL_RESULTS.
    assert rep["defense_status"] in {"NO-GO", "GO"}
    assert rep["n"] == 6165
    assert abs(rep["summary"]["delta_top1_pp"] - 7.283) < 0.01
    assert rep["bootstrap"]["top1_delta"]["seed"] == 42
    assert rep["mcnemar_top1"]["exact_two_sided_p"] < 1e-30
    assert rep["encoding_audit"]["gold_utf8_verified"] is True
    assert (
        hashlib.sha256(
            (_ROOT / "Docs" / "results" / "ms_sta_e2" / "E2_CANONICAL_REPORT.json").read_bytes()
        ).hexdigest()
        == "408fc077d92746dc043dc3d457575701b22c7bf4f19b50484450df6b95e31e79"
    )
    src = _ROOT / rep["source_artifact"]
    if not src.is_file():
        pytest.skip("E2 raw rows artifact not present (release download required)")
    assert _sha(src) == rep["source_artifact_sha256"]


def test_final_results_is_post_d4_companion() -> None:
    final = json.loads(
        (_ROOT / "Docs" / "results" / "ms_sta_defense_go" / "FINAL_RESULTS.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert final["defense_status"] == "GO"
    assert final["e2"]["sha256"].startswith("408fc077")
    assert final["d4"]["sha256"].startswith("f5375f22")
    assert final["e1_v2"]["sha256"].startswith("69aed785")
    assert "derived_from" in final
    e2 = json.loads(
        (_ROOT / "Docs" / "results" / "ms_sta_e2" / "E2_CANONICAL_REPORT.json").read_text(
            encoding="utf-8-sig"
        )
    )
    # Companion may advance defense status without rewriting E2 bytes.
    assert final["e2"]["artifact_defense_status_at_e2"] == e2.get("defense_status")


def test_registry_defense_go_after_d4() -> None:
    reg = json.loads((_ROOT / "data" / "processed" / "ms_sta_registry.json").read_text(encoding="utf-8"))
    assert reg["e2_pass"] is True
    assert reg["e1_selected"] is True
    assert reg["evaluation_locked"] is True
    assert reg["defense_ready"] is True
    assert reg["d4_pass"] is True
    assert reg["status"] == "GO"
    assert reg["d4_candidate_allowed"] is False
    assert reg["checkpoint"] == "b4_seed42_defense.pth"
    assert reg["ckpts_dir"] == "data/processed/ms_sta_defense"
    assert reg["defense_promotion"]["weights_unchanged_from_e2_lock"] is True
    ckpt = _ROOT / "data/processed/ms_sta_defense/b4_seed42_defense.pth"
    if not ckpt.is_file():
        pytest.skip("defense checkpoint not present (release download required)")


def test_d4_acceptance_report_all_pass() -> None:
    rep = json.loads(
        (_ROOT / "Docs" / "results" / "ms_sta_d4" / "D4_ACCEPTANCE_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert rep["all_scenarios_pass"] is True
    assert rep["d4_pass"] is True
    assert rep["post_promotion_smoke"]["pass"] is True
    assert rep["demo_smoke_wording"]["not_kpi"] is True
    assert rep["demo_smoke_wording"]["no_realtime_arbitrary_video_claim"] is True


def test_final_results_binds_e1_e2_d4() -> None:
    final = json.loads(
        (_ROOT / "Docs" / "results" / "ms_sta_defense_go" / "FINAL_RESULTS.json").read_text(
            encoding="utf-8"
        )
    )
    assert final["defense_status"] == "GO"
    assert final["active_variant"] == "b4"
    assert abs(final["e2"]["delta_top1_pp"] - 7.283) < 0.01
    assert final["e1_v2"]["consistent_improvement_pass"] is True
    assert final["no_rerun_e2"] is True
