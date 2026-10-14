"""E1 policy freeze guards — B0 identity + seed rule must be locked before training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
POLICY = _ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY.json"
POLICY_V1 = _ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY_v1.json"
SPLIT = _ROOT / "data" / "processed" / "vsl400_e1_split_manifest.json"
REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
DIAG = _ROOT / "Docs" / "results" / "ms_sta_e1_v1_diagnostic" / "E1_V1_DIAGNOSTIC.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e1_policy_frozen_and_bound() -> None:
    assert POLICY.is_file()
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert policy["status"] == "FROZEN"
    assert policy["defense_status"] == "NO-GO"
    assert policy["schema"] == "option_b_e1_policy_v2"
    assert policy["b0_identity"]["b0_e1"]["retrain_per_seed"] is True
    assert policy["b0_identity"]["b0_gate3"]["sha256"].startswith("26d866c3")
    assert "e2_b0_counterpart_after_seeing_results" in policy["b0_identity"]["b0_gate3"]["forbidden_uses"]
    seeds = policy["seeds"]
    assert seeds["list"] == [42, 123, 456]
    assert seeds["primary_seed"] == 42
    assert seeds["rule"] == "primary_seed_predeclared_for_e2"
    assert policy["checkpoint_selection"]["within_run_metric"] == "val_top1"
    assert policy["checkpoint_selection"]["e1_must_not_compute_test_kpi"] is True
    assert policy["variants"]["required"] == ["b0", "b1", "b3", "b4", "b5"]
    assert "b2" in policy["variants"]["diagnostic_optional"]
    assert policy["variant_controls"]["b5"]["disable_spatial"] is True
    assert policy["variant_controls"]["b4"]["disable_temporal"] is False
    assert policy["variant_controls"]["b1"]["init"] == "gate3_rgb_proj_and_compatible_classifier"
    assert policy["variant_controls"]["b3"]["head_only_warmup_epochs"] == 1
    assert policy["collapse_guard"]["enabled"] is True
    assert policy["artifact_dirs"]["checkpoints"] == "data/processed/ms_sta_e1_v2"
    assert policy["artifact_dirs"]["status"] == "logs/gate_e1_v2"
    assert policy["runtime_locks"]["fail_if_cuda_unavailable"] is True
    assert policy["output_contract"]["refuse_existing_checkpoint"] is True

    assert SPLIT.is_file()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert split["n_train"] == 15992
    assert split["n_val"] == 2595
    assert split["n_test"] == 6165
    assert _sha(SPLIT) == policy["data_locks"]["e1_split_manifest_sha256"]

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert reg["status"] == "GO"
    assert reg["e1_policy"]["status"] == "FROZEN"
    assert reg["e1_policy"]["sha256"] == _sha(POLICY)
    assert reg["e1_policy"]["primary_seed"] == 42
    assert reg["e1_policy"]["b0_e1_retrain_per_seed"] is True
    assert reg["e1_policy"]["trainer_enforcement"] is True
    assert reg["e1_policy"]["schema"] == "option_b_e1_policy_v2"
    assert reg["active_temporal_dir"] == "data/features/isolated_sf_temporal_v2_8741189c657a"
    assert reg["defense_ready"] is True
    assert reg["d4_pass"] is True
    assert reg["e2_pass"] is True
    assert reg["d4_candidate_allowed"] is False
    assert reg["checkpoint"] == "b4_seed42_defense.pth"


def test_e1_v1_preserved_and_classified() -> None:
    assert POLICY_V1.is_file()
    v1 = json.loads(POLICY_V1.read_text(encoding="utf-8"))
    assert v1["schema"] == "option_b_e1_policy_v1.1"
    assert _sha(POLICY_V1) == "30c454d4781750ee778c7935bb151035bdc9b48445a6458db83723bc95501b66"
    assert DIAG.is_file()
    diag = json.loads(DIAG.read_text(encoding="utf-8"))
    assert diag["classification"] == "failed_optimization_configuration"
    assert diag["not_a_negative_architecture_result"] is True
    assert diag["e2_forbidden"] is True


def test_e1_policy_has_no_mojibake_signatures() -> None:
    sigs = [
        bytes([0xC3, 0xA2, 0xE2, 0x82, 0xAC, 0xE2, 0x80, 0x9D]),
        bytes([0xC3, 0x83, 0xC2]),
    ]
    for path in (POLICY, POLICY_V1, REGISTRY, SPLIT, DIAG):
        data = path.read_bytes()
        for sig in sigs:
            assert sig not in data, f"mojibake signature in {path}"
