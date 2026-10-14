"""Fail-closed E1 policy enforcement helpers (pure checks; no training side effects)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.feature_versioning import load_json, sha256_file

INIT_GATE3_PROJ_CLS = "gate3_rgb_proj_and_classifier"
INIT_GATE3_PROJ_COMPAT_CLS = "gate3_rgb_proj_and_compatible_classifier"
INIT_GATE3_PROJ_ONLY = "gate3_rgb_proj_only"
INIT_RANDOM = "random"
KNOWN_INIT_MODES = {
    INIT_GATE3_PROJ_CLS,
    INIT_GATE3_PROJ_COMPAT_CLS,
    INIT_GATE3_PROJ_ONLY,
    INIT_RANDOM,
}


def require_frozen_policy(policy: dict[str, Any], *, allow_pilot: bool) -> None:
    if policy.get("status") != "FROZEN" and not allow_pilot:
        raise ValueError("E1 policy status != FROZEN")


def resolve_variant_control(policy: dict[str, Any], variant: str) -> dict[str, Any]:
    controls = (policy.get("variant_controls") or {}).get(variant)
    if not isinstance(controls, dict):
        raise ValueError(f"E1 policy missing variant_controls[{variant}]")
    required = (
        "disable_spatial",
        "disable_temporal",
        "require_skel",
        "require_temporal_rgb",
        "init",
    )
    missing = [k for k in required if k not in controls]
    if missing:
        raise ValueError(f"variant_controls[{variant}] missing keys: {missing}")
    return controls


def refuse_free_ablation_flags(
    *,
    warm_b0: bool,
    disable_spatial: bool,
    disable_temporal: bool,
    allow_pilot: bool,
) -> None:
    """Canonical E1 derives ablation/init from policy — CLI must not set these."""
    if allow_pilot:
        return
    bad = []
    if warm_b0:
        bad.append("--warm-b0")
    if disable_spatial:
        bad.append("--disable-spatial")
    if disable_temporal:
        bad.append("--disable-temporal")
    if bad:
        raise ValueError(
            "canonical E1 refuses free ablation/init CLI flags "
            f"{bad}; configuration is derived from E1_POLICY.variant_controls"
        )


def enforce_hyperparams(
    policy: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    allow_pilot: bool,
) -> None:
    if allow_pilot:
        return
    hp = policy.get("train_hyperparameters") or {}
    checks = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience_early_stopping": patience,
    }
    for key, val in checks.items():
        if key in hp and hp[key] != val:
            raise ValueError(f"frozen E1 policy {key}={hp[key]} != cli {val}")
    seeds = (policy.get("seeds") or {}).get("list") or []
    if seeds and seed not in seeds:
        raise ValueError(f"seed={seed} not in frozen E1 seeds {seeds}")


def enforce_cuda_device(
    policy: dict[str, Any],
    *,
    requested_device: str,
    cuda_available: bool,
    allow_pilot: bool,
) -> str:
    """Return resolved device string; fail-closed when policy expects CUDA."""
    runtime = policy.get("runtime_locks") or {}
    expected = str(
        runtime.get("device_expected")
        or (policy.get("train_hyperparameters") or {}).get("device_expected")
        or "cuda"
    ).lower()
    req = str(requested_device).lower()
    fail_closed = runtime.get("fail_if_cuda_unavailable", True)
    if allow_pilot:
        if req.startswith("cuda") and not cuda_available:
            return "cpu"
        return req
    if expected.startswith("cuda") or req.startswith("cuda"):
        if fail_closed and not cuda_available:
            raise ValueError(
                "canonical E1 requires CUDA (device_expected=cuda) but torch.cuda.is_available() is False"
            )
        if not req.startswith("cuda"):
            raise ValueError(f"canonical E1 refuses device={requested_device}; expected cuda")
        return "cuda"
    return req


def _expect_sha(label: str, expected: str | None, actual: str) -> None:
    if expected and expected != actual:
        raise ValueError(f"{label} sha mismatch: actual={actual} expected={expected}")


def enforce_data_and_runtime_locks(
    policy: dict[str, Any],
    *,
    root: Path,
    registry: dict[str, Any],
    labels_path: Path,
    split_manifest_sha: str | None,
    eligible_manifest_sha: str | None,
    pooled_dir: Path,
    temporal_dir: Path,
    skel_dir: Path,
    allow_pilot: bool,
) -> dict[str, str]:
    """Validate locked hashes/dirs before DataLoader construction. Returns provenance bundle."""
    if allow_pilot:
        return {}
    locks = policy.get("data_locks") or {}
    runtime = policy.get("runtime_locks") or {}
    lineage = policy.get("lineage") or {}
    feats = registry.get("features") or {}
    skel_ev = feats.get("skel_promotion_evidence") or {}
    temporal_ev = feats.get("temporal_promotion_evidence") or {}

    labels_sha = sha256_file(labels_path)
    _expect_sha("labels", locks.get("labels_sha256"), labels_sha)
    if split_manifest_sha is not None:
        _expect_sha("e1_split_manifest", locks.get("e1_split_manifest_sha256"), split_manifest_sha)
    if eligible_manifest_sha is not None:
        _expect_sha(
            "eligible_sample_manifest",
            locks.get("eligible_sample_manifest_sha256"),
            eligible_manifest_sha,
        )

    def _rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            return str(p).replace("\\", "/")

    expect_skel = runtime.get("active_skel_dir") or lineage.get("active_skel_dir")
    expect_temporal = runtime.get("active_temporal_dir") or lineage.get("active_temporal_dir")
    expect_pooled = runtime.get("pooled_dir") or (feats.get("pooled_dir") if feats else None) or "data/features/isolated_sf2304"
    if expect_skel and _rel(skel_dir) != str(expect_skel).replace("\\", "/"):
        raise ValueError(f"active_skel_dir mismatch: {_rel(skel_dir)} != {expect_skel}")
    if expect_temporal and _rel(temporal_dir) != str(expect_temporal).replace("\\", "/"):
        raise ValueError(f"active_temporal_dir mismatch: {_rel(temporal_dir)} != {expect_temporal}")
    if expect_pooled and _rel(pooled_dir) != str(expect_pooled).replace("\\", "/"):
        raise ValueError(f"pooled_dir mismatch: {_rel(pooled_dir)} != {expect_pooled}")

    skel_man_sha = skel_ev.get("manifest_sha256")
    temporal_man_sha = temporal_ev.get("manifest_sha256")
    _expect_sha(
        "skel_manifest",
        runtime.get("skel_manifest_sha256") or lineage.get("skel_manifest_sha256"),
        str(skel_man_sha or ""),
    )
    _expect_sha(
        "temporal_manifest",
        runtime.get("temporal_manifest_sha256") or lineage.get("temporal_manifest_sha256"),
        str(temporal_man_sha or ""),
    )

    cuda_rel = runtime.get("cuda_provenance") or "logs/preflight/cuda_train_canonical/PROVENANCE.json"
    cuda_path = root / str(cuda_rel)
    if not cuda_path.is_file():
        raise ValueError(f"missing CUDA provenance: {cuda_path}")
    cuda_sha = sha256_file(cuda_path)
    _expect_sha("cuda_provenance", runtime.get("cuda_provenance_sha256") or lineage.get("cuda_provenance_sha256"), cuda_sha)
    cuda = load_json(cuda_path)
    if cuda.get("pass") is not True:
        raise ValueError("CUDA provenance pass!=true")
    backbone = (cuda.get("backbone_checkpoint_identity") or {}).get("sha256")
    expect_bb = runtime.get("backbone_checkpoint_sha256") or lineage.get("backbone_checkpoint_sha256")
    _expect_sha("backbone_checkpoint", expect_bb, str(backbone or ""))
    pip_sha = cuda.get("pip_freeze_sha256")
    expect_pip = runtime.get("pip_freeze_sha256")
    if expect_pip:
        _expect_sha("pip_freeze", expect_pip, str(pip_sha or ""))

    gate3 = (policy.get("initialization") or {}).get("gate3_checkpoint_sha256") or (
        (policy.get("b0_identity") or {}).get("b0_gate3") or {}
    ).get("sha256")
    gate3_rel = (policy.get("initialization") or {}).get("gate3_checkpoint") or (
        (policy.get("b0_identity") or {}).get("b0_gate3") or {}
    ).get("path")
    if gate3_rel:
        gate3_path = root / str(gate3_rel)
        if not gate3_path.is_file():
            raise ValueError(f"missing Gate3 checkpoint: {gate3_path}")
        gate3_sha = sha256_file(gate3_path)
        _expect_sha("gate3_checkpoint", gate3, gate3_sha)
    else:
        gate3_sha = ""

    return {
        "labels_sha256": labels_sha,
        "cuda_provenance_sha256": cuda_sha,
        "backbone_checkpoint_sha256": str(backbone or ""),
        "pip_freeze_sha256": str(pip_sha or ""),
        "skel_manifest_sha256": str(skel_man_sha or ""),
        "temporal_manifest_sha256": str(temporal_man_sha or ""),
        "gate3_checkpoint_sha256": gate3_sha,
    }


def refuse_output_collision(
    *,
    checkpoint_path: Path,
    status_path: Path,
    allow_overwrite: bool,
    allow_pilot: bool,
) -> None:
    if allow_overwrite and allow_pilot:
        return
    if checkpoint_path.is_file():
        raise ValueError(
            f"checkpoint already exists (refuse overwrite): {checkpoint_path}. "
            "Use empty artifact dir or --force-overwrite with --allow-unfrozen-pilot"
        )
    if status_path.is_file():
        raise ValueError(
            f"status already exists (refuse overwrite): {status_path}. "
            "Use empty status dir or --force-overwrite with --allow-unfrozen-pilot"
        )


def resolve_warmup(policy: dict[str, Any], variant: str) -> dict[str, Any]:
    """Return head-only warmup plan derived from variant_controls / warmup block."""
    controls = (policy.get("variant_controls") or {}).get(variant) or {}
    epochs = int(controls.get("head_only_warmup_epochs") or 0)
    modules = list(controls.get("head_only_modules") or [])
    if epochs > 0 and not modules:
        warm = (policy.get("warmup") or {}).get(variant) or {}
        modules = list(warm.get("modules") or [])
    if epochs < 0:
        raise ValueError(f"head_only_warmup_epochs must be >=0 for {variant}")
    return {"epochs": epochs, "modules": modules}


def set_head_only_trainable(model: Any, *, modules: list[str], enabled: bool) -> None:
    """Freeze all params except named head modules when enabled; else unfreeze all."""
    if not enabled:
        for p in model.parameters():
            p.requires_grad = True
        return
    if not modules:
        raise ValueError("head-only warmup enabled but modules list empty")
    for p in model.parameters():
        p.requires_grad = False
    for name in modules:
        mod = getattr(model, name, None)
        if mod is None:
            raise ValueError(f"unknown head-only module: {name}")
        for p in mod.parameters():
            p.requires_grad = True


def check_collapse_guard(
    policy: dict[str, Any],
    *,
    epoch: int,
    train_loss: float,
    train_top1: float,
    allow_pilot: bool,
) -> None:
    """Fail-closed if train metrics still near chance after configured epoch."""
    if allow_pilot:
        return
    guard = policy.get("collapse_guard") or {}
    if not guard.get("enabled", False):
        return
    check_after = int(guard.get("check_after_epoch") or 2)
    if epoch < check_after:
        return
    if epoch > check_after:
        return
    max_loss = float(guard.get("max_train_loss") or 5.5)
    min_top1 = float(guard.get("min_train_top1_pct") or 1.0)
    collapsed = train_loss > max_loss and train_top1 < min_top1
    if collapsed and str(guard.get("action") or "fail_run") == "fail_run":
        raise ValueError(
            f"E1 collapse_guard failed after epoch {epoch}: "
            f"train_loss={train_loss:.4f}>{max_loss} and "
            f"train_top1={train_top1:.4f}%<{min_top1}% "
            f"(near ln(400) / chance — optimization collapse)"
        )


def apply_initialization(
    model: Any,
    *,
    init_mode: str,
    gate3_ckpt_path: Path,
) -> str:
    """Apply policy init mode; returns applied mode label."""
    if init_mode == INIT_RANDOM:
        return INIT_RANDOM
    if init_mode not in KNOWN_INIT_MODES:
        raise ValueError(f"unknown init mode: {init_mode}")
    import torch

    ckpt = torch.load(gate3_ckpt_path, map_location="cpu", weights_only=False)
    head = ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt
    if init_mode == INIT_GATE3_PROJ_CLS:
        model.load_b0_head(head, classifier=True, classifier_target="b0_cls")
        return INIT_GATE3_PROJ_CLS
    if init_mode == INIT_GATE3_PROJ_COMPAT_CLS:
        model.load_b0_head(head, classifier=True, classifier_target="cls")
        return INIT_GATE3_PROJ_COMPAT_CLS
    model.load_b0_head(head, classifier=False)
    return INIT_GATE3_PROJ_ONLY
