"""Fail-closed E1 trainer enforcement tests (policy v2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))

from src.training.e1_policy_enforce import (
    INIT_GATE3_PROJ_COMPAT_CLS,
    check_collapse_guard,
    enforce_cuda_device,
    enforce_data_and_runtime_locks,
    refuse_free_ablation_flags,
    refuse_output_collision,
    resolve_variant_control,
    resolve_warmup,
)

POLICY = json.loads(
    (_ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY.json").read_text(encoding="utf-8")
)
REGISTRY = json.loads((_ROOT / "data" / "processed" / "ms_sta_registry.json").read_text(encoding="utf-8"))


def test_variant_controls_locked_v2() -> None:
    assert POLICY["schema"] == "option_b_e1_policy_v2"
    b4 = resolve_variant_control(POLICY, "b4")
    assert b4["disable_spatial"] is False
    assert b4["disable_temporal"] is False
    assert b4["init"] == INIT_GATE3_PROJ_COMPAT_CLS
    b5 = resolve_variant_control(POLICY, "b5")
    assert b5["disable_spatial"] is True
    assert b5["init"] == INIT_GATE3_PROJ_COMPAT_CLS
    b1 = resolve_variant_control(POLICY, "b1")
    assert b1["init"] == INIT_GATE3_PROJ_COMPAT_CLS
    b0 = resolve_variant_control(POLICY, "b0")
    assert b0["init"] == "gate3_rgb_proj_and_classifier"
    b3 = resolve_variant_control(POLICY, "b3")
    assert b3["init"] == "gate3_rgb_proj_only"
    assert int(b3["head_only_warmup_epochs"]) == 1
    b2 = resolve_variant_control(POLICY, "b2")
    assert b2["init"] == "random"
    for v in ("b0", "b1", "b2", "b3", "b4", "b5"):
        assert resolve_variant_control(POLICY, v)["disable_temporal"] is False


def test_warmup_b3_only() -> None:
    w3 = resolve_warmup(POLICY, "b3")
    assert w3["epochs"] == 1
    assert w3["modules"] == ["concat_cls"]
    w4 = resolve_warmup(POLICY, "b4")
    assert w4["epochs"] == 0


def test_collapse_guard_triggers() -> None:
    with pytest.raises(ValueError, match="collapse_guard"):
        check_collapse_guard(
            POLICY,
            epoch=2,
            train_loss=5.99,
            train_top1=0.3,
            allow_pilot=False,
        )
    # healthy train signal passes
    check_collapse_guard(
        POLICY,
        epoch=2,
        train_loss=2.0,
        train_top1=35.0,
        allow_pilot=False,
    )
    # epoch 1 not checked yet
    check_collapse_guard(
        POLICY,
        epoch=1,
        train_loss=5.99,
        train_top1=0.3,
        allow_pilot=False,
    )


def test_refuse_free_ablation_flags() -> None:
    with pytest.raises(ValueError, match="refuses free ablation"):
        refuse_free_ablation_flags(
            warm_b0=False, disable_spatial=False, disable_temporal=True, allow_pilot=False
        )
    with pytest.raises(ValueError, match="refuses free ablation"):
        refuse_free_ablation_flags(
            warm_b0=True, disable_spatial=False, disable_temporal=False, allow_pilot=False
        )
    refuse_free_ablation_flags(
        warm_b0=True, disable_spatial=True, disable_temporal=True, allow_pilot=True
    )


def test_fail_closed_cuda() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        enforce_cuda_device(
            POLICY,
            requested_device="cuda",
            cuda_available=False,
            allow_pilot=False,
        )
    with pytest.raises(ValueError, match="refuses device"):
        enforce_cuda_device(
            POLICY,
            requested_device="cpu",
            cuda_available=True,
            allow_pilot=False,
        )
    assert (
        enforce_cuda_device(
            POLICY,
            requested_device="cuda",
            cuda_available=True,
            allow_pilot=False,
        )
        == "cuda"
    )


def _require_e1_runtime_artifacts() -> Path:
    labels = _ROOT / "data" / "processed" / "vsl400_labels.json"
    temporal = _ROOT / "data/features/isolated_sf_temporal_v2_8741189c657a/_manifest.json"
    skel = _ROOT / "data/features/local_skel_v2_66fc5f01ee415fa1/_manifest_VSL400_all.json"
    if not labels.is_file() or not temporal.is_file() or not skel.is_file():
        pytest.skip("E1 runtime artifacts missing (labels/feature manifests)")
    return labels


@pytest.mark.requires_artifacts
def test_runtime_hash_mismatch_labels(tmp_path: Path) -> None:
    bad_policy = json.loads(json.dumps(POLICY))
    bad_policy["data_locks"]["labels_sha256"] = "0" * 64
    labels = _require_e1_runtime_artifacts()
    with pytest.raises(ValueError, match="labels sha mismatch"):
        enforce_data_and_runtime_locks(
            bad_policy,
            root=_ROOT,
            registry=REGISTRY,
            labels_path=labels,
            split_manifest_sha=POLICY["data_locks"]["e1_split_manifest_sha256"],
            eligible_manifest_sha=POLICY["data_locks"]["eligible_sample_manifest_sha256"],
            pooled_dir=_ROOT / "data/features/isolated_sf2304",
            temporal_dir=_ROOT / "data/features/isolated_sf_temporal_v2_8741189c657a",
            skel_dir=_ROOT / "data/features/local_skel_v2_66fc5f01ee415fa1",
            allow_pilot=False,
        )


@pytest.mark.requires_artifacts
def test_runtime_hash_mismatch_temporal_manifest() -> None:
    bad_policy = json.loads(json.dumps(POLICY))
    bad_policy["runtime_locks"]["temporal_manifest_sha256"] = "1" * 64
    labels = _require_e1_runtime_artifacts()
    with pytest.raises(ValueError, match="temporal_manifest sha mismatch"):
        enforce_data_and_runtime_locks(
            bad_policy,
            root=_ROOT,
            registry=REGISTRY,
            labels_path=labels,
            split_manifest_sha=POLICY["data_locks"]["e1_split_manifest_sha256"],
            eligible_manifest_sha=POLICY["data_locks"]["eligible_sample_manifest_sha256"],
            pooled_dir=_ROOT / "data/features/isolated_sf2304",
            temporal_dir=_ROOT / "data/features/isolated_sf_temporal_v2_8741189c657a",
            skel_dir=_ROOT / "data/features/local_skel_v2_66fc5f01ee415fa1",
            allow_pilot=False,
        )


@pytest.mark.requires_artifacts
def test_active_dir_mismatch() -> None:
    labels = _require_e1_runtime_artifacts()
    with pytest.raises(ValueError, match="active_temporal_dir mismatch"):
        enforce_data_and_runtime_locks(
            POLICY,
            root=_ROOT,
            registry=REGISTRY,
            labels_path=labels,
            split_manifest_sha=POLICY["data_locks"]["e1_split_manifest_sha256"],
            eligible_manifest_sha=POLICY["data_locks"]["eligible_sample_manifest_sha256"],
            pooled_dir=_ROOT / "data/features/isolated_sf2304",
            temporal_dir=_ROOT / "data/features/isolated_sf_temporal",
            skel_dir=_ROOT / "data/features/local_skel_v2_66fc5f01ee415fa1",
            allow_pilot=False,
        )


def test_refuse_output_collision(tmp_path: Path) -> None:
    ckpt = tmp_path / "b4_seed42.pth"
    status = tmp_path / "b4_seed42.json"
    ckpt.write_bytes(b"x")
    with pytest.raises(ValueError, match="checkpoint already exists"):
        refuse_output_collision(
            checkpoint_path=ckpt,
            status_path=status,
            allow_overwrite=False,
            allow_pilot=False,
        )
    status.write_text("{}", encoding="utf-8")
    ckpt.unlink()
    with pytest.raises(ValueError, match="status already exists"):
        refuse_output_collision(
            checkpoint_path=ckpt,
            status_path=status,
            allow_overwrite=False,
            allow_pilot=False,
        )
    refuse_output_collision(
        checkpoint_path=ckpt,
        status_path=status,
        allow_overwrite=True,
        allow_pilot=True,
    )


def test_artifact_dirs_are_v2_not_v1() -> None:
    art = POLICY["artifact_dirs"]
    assert art["checkpoints"] == "data/processed/ms_sta_e1_v2"
    assert art["status"] == "logs/gate_e1_v2"
    assert POLICY["e1_v1_preserve"]["checkpoints"] == "data/processed/ms_sta"
    assert POLICY["e1_v1_preserve"]["status"] == "logs/gate_e1"


@pytest.mark.requires_artifacts
def test_canonical_runtime_locks_pass_on_disk() -> None:
    labels = _require_e1_runtime_artifacts()
    prov = enforce_data_and_runtime_locks(
        POLICY,
        root=_ROOT,
        registry=REGISTRY,
        labels_path=labels,
        split_manifest_sha=POLICY["data_locks"]["e1_split_manifest_sha256"],
        eligible_manifest_sha=POLICY["data_locks"]["eligible_sample_manifest_sha256"],
        pooled_dir=_ROOT / "data/features/isolated_sf2304",
        temporal_dir=_ROOT / "data/features/isolated_sf_temporal_v2_8741189c657a",
        skel_dir=_ROOT / "data/features/local_skel_v2_66fc5f01ee415fa1",
        allow_pilot=False,
    )
    assert prov["backbone_checkpoint_sha256"].startswith("454f39e1")
    assert prov["cuda_provenance_sha256"].startswith("e4c2032d")
    assert prov["gate3_checkpoint_sha256"].startswith("26d866c3")
