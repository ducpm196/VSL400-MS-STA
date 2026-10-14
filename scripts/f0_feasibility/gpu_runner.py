"""Offline CUDA feasibility for four F0 backbone candidates (remediation-7).

- Exact classifier allowlists / VideoMAE loading_info
- SlowFast: Kinetics-400 strict load then replace head -> 100
- Plan-driven measurement modes with truthful grad-ckpt requested/supported/applied
- Restore initial state before each mode
- CPU forbidden
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    CANDIDATE_IDS,
    CANDIDATE_MEASUREMENT,
    CLASSIFIER_KEY_ALLOWLIST,
    LAYOUT_CONTRACT,
    SLOWFAST_TRAINABLE_BLOCKS,
    any_module_prefix,
    measurement_modes_for,
)
from .preprocess import synthetic_batch_for

MEASUREMENT = {
    "batch_size_first": 1,
    "optimizer_step_required": True,
    "require_finite_loss_and_grads": True,
    "warmup_steps": 2,
    "timed_steps": 10,
    "amp_logged_separately": True,
    "grad_checkpointing_logged_separately": True,
    "record_peak_allocated_vram_bytes": True,
    "record_peak_reserved_vram_bytes": True,
    "record_min_free_vram_bytes": True,
    "record_step_latency_ms_distribution": True,
    "checkpoint_save_load_bit_exact": True,
    "output_tensor_contract_required": True,
    "forbid_single_step_latency_claim": True,
}


def _require_torch():
    try:
        import torch  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch_required_for_gpu_feasibility") from exc
    return torch


def require_cuda_device(device: str) -> str:
    dev = str(device or "").strip().lower()
    if not (dev == "cuda" or dev.startswith("cuda:")):
        raise ValueError("device_not_cuda")
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_not_available")
    return dev


def candidate_snapshot_dir(weights_root: Path, candidate_id: str) -> Path:
    path = weights_root / candidate_id
    if not path.is_dir():
        raise FileNotFoundError(f"snapshot_dir_missing:{candidate_id}")
    return path


def list_required_local_files(snapshot_contract: dict[str, Any], candidate_id: str) -> list[str]:
    csnap = (snapshot_contract.get("candidates") or {}).get(candidate_id) or {}
    return [str(f["local_name"]) for f in (csnap.get("files") or [])]


def assert_local_snapshot_complete(snapshot_dir: Path, required_names: list[str]) -> None:
    missing = [n for n in required_names if not (snapshot_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"snapshot_incomplete:{snapshot_dir.name}:{missing}")


def _extract_state_dict(state: Any) -> dict[str, Any]:
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
        return state["model_state"]
    if isinstance(state, dict):
        return state
    raise RuntimeError("checkpoint_state_dict_invalid")


def load_state_dict_with_allowlist(
    model: Any,
    state: dict[str, Any],
    *,
    allowed_keys: frozenset[str],
) -> dict[str, Any]:
    """Strict except exact allowlisted classifier keys."""
    filtered = _extract_state_dict(state)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    bad_missing = [k for k in missing if k not in allowed_keys]
    bad_unexpected = [k for k in unexpected if k not in allowed_keys]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"checkpoint_key_mismatch:missing={bad_missing}:unexpected={bad_unexpected}"
        )
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "allowed_keys": sorted(allowed_keys),
    }


def _normalize_loading_keys(items: Any) -> set[str]:
    """Normalize Transformers loading-info entries (str or (key, ckpt_shape, model_shape))."""
    out: set[str] = set()
    for item in items or []:
        if isinstance(item, tuple):
            if not item:
                raise RuntimeError("videomae_loading_info_empty_tuple")
            out.add(str(item[0]))
        else:
            out.add(str(item))
    return out


def assert_videomae_loading_info(
    info: dict[str, Any],
    *,
    allowed: frozenset[str],
    require_exact_classifier_mismatch: bool = True,
) -> dict[str, Any]:
    err_msgs = info.get("error_msgs") or []
    if err_msgs:
        raise RuntimeError(f"videomae_loading_error_msgs:{err_msgs}")
    mismatched = _normalize_loading_keys(info.get("mismatched_keys"))
    missing = _normalize_loading_keys(info.get("missing_keys"))
    unexpected = _normalize_loading_keys(info.get("unexpected_keys"))
    bad = (mismatched | missing | unexpected) - set(allowed)
    if bad:
        raise RuntimeError(f"videomae_loading_info_non_classifier:{sorted(bad)}")
    if require_exact_classifier_mismatch and mismatched != set(allowed):
        raise RuntimeError(
            "videomae_classifier_mismatch_set_invalid:"
            f"got={sorted(mismatched)}:expected={sorted(allowed)}"
        )
    return {
        "mismatched_keys": sorted(mismatched),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "allowed_keys": sorted(allowed),
        "error_msgs": [],
    }


def replace_slowfast_head(model: Any, num_classes: int) -> dict[str, Any]:
    torch = _require_torch()
    if not hasattr(model, "blocks") or len(model.blocks) < 1:
        raise RuntimeError("slowfast_missing_blocks_for_head")
    head = model.blocks[-1]
    if not hasattr(head, "proj") or not hasattr(head.proj, "in_features"):
        raise RuntimeError("slowfast_missing_proj_head")
    in_features = int(head.proj.in_features)
    old_out = int(getattr(head.proj, "out_features", -1))
    head.proj = torch.nn.Linear(in_features, num_classes)
    if int(head.proj.out_features) != num_classes:
        raise RuntimeError("slowfast_head_replace_failed")
    return {
        "head_attr": "blocks[-1].proj",
        "in_features": in_features,
        "old_out_features": old_out,
        "new_out_features": num_classes,
    }


def replace_swin_head(model: Any, num_classes: int) -> dict[str, Any]:
    torch = _require_torch()
    if getattr(model, "head", None) is None or not hasattr(model.head, "in_features"):
        raise RuntimeError("swin_missing_head")
    in_features = int(model.head.in_features)
    model.head = torch.nn.Linear(in_features, num_classes)
    return {
        "head_attr": "head",
        "in_features": in_features,
        "new_out_features": num_classes,
    }


def apply_slowfast_partial_unfreeze(model: Any) -> dict[str, Any]:
    frozen = list(SLOWFAST_TRAINABLE_BLOCKS["frozen"])
    trainable = list(SLOWFAST_TRAINABLE_BLOCKS["trainable"])
    if not hasattr(model, "blocks"):
        raise RuntimeError("slowfast_missing_blocks_attr")
    for idx, block in enumerate(model.blocks):
        requires = f"blocks.{idx}" in trainable
        if f"blocks.{idx}" in frozen:
            requires = False
        for p in block.parameters():
            p.requires_grad = requires
    identities = trainable_param_identities(model)
    for prefix in frozen:
        if any_module_prefix(identities, prefix):
            raise RuntimeError(f"slowfast_frozen_block_still_trainable:{prefix}")
    for prefix in trainable:
        if not any_module_prefix(identities, prefix):
            raise RuntimeError(f"slowfast_trainable_block_not_found:{prefix}")
    return {"frozen": frozen, "trainable": trainable, "trainable_param_count": len(identities)}


def install_swin_grad_checkpointing_api(model: Any) -> None:
    """Provide HF-style enable/disable that actually checkpoints Swin features."""
    torch = _require_torch()
    import torch.utils.checkpoint as torch_checkpoint  # noqa: WPS433 - direct import required

    if not hasattr(torch_checkpoint, "checkpoint"):
        raise RuntimeError("torch_utils_checkpoint_unavailable")
    if hasattr(model, "gradient_checkpointing_enable") and hasattr(
        model, "gradient_checkpointing_disable"
    ):
        # Still ensure invocation counter exists for F0 adapter-compatible models.
        if not hasattr(model, "_f0_ckpt_invocations"):
            model._f0_ckpt_invocations = 0
        return
    if not hasattr(model, "features"):
        raise RuntimeError("swin_missing_features_for_grad_checkpointing")

    model._f0_grad_ckpt = False
    model._f0_ckpt_invocations = 0
    orig_features_forward = model.features.forward

    def _features_forward(x):
        if not getattr(model, "_f0_grad_ckpt", False):
            return orig_features_forward(x)
        out = x
        for module in model.features:
            model._f0_ckpt_invocations = int(getattr(model, "_f0_ckpt_invocations", 0)) + 1
            out = torch_checkpoint.checkpoint(module, out, use_reentrant=False)
        return out

    model.features.forward = _features_forward

    def _enable() -> None:
        model._f0_grad_ckpt = True
        model._f0_ckpt_invocations = 0

    def _disable() -> None:
        model._f0_grad_ckpt = False

    model.gradient_checkpointing_enable = _enable  # type: ignore[attr-defined]
    model.gradient_checkpointing_disable = _disable  # type: ignore[attr-defined]


def set_grad_checkpointing(model: Any, *, requested: bool) -> dict[str, Any]:
    """Return truthful requested/supported/applied; never lie that unsupported was applied."""
    meta = {"requested": bool(requested), "supported": False, "applied": False}
    if not requested:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            meta["supported"] = True
        meta["applied"] = False
        return meta
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        meta["supported"] = True
        # Prefer explicit flag when present (Swin F0 adapter).
        if hasattr(model, "_f0_grad_ckpt"):
            meta["applied"] = bool(model._f0_grad_ckpt)
        else:
            meta["applied"] = True
        return meta
    return meta


def build_model_offline(
    candidate_id: str,
    snapshot_dir: Path,
) -> tuple[Any, Callable[[Any, dict[str, Any]], Any], dict[str, Any]]:
    torch = _require_torch()
    spec = CANDIDATE_MEASUREMENT[candidate_id]
    num_dummy = int(spec["num_classes_dummy"])
    allowed = CLASSIFIER_KEY_ALLOWLIST[candidate_id]
    load_meta: dict[str, Any] = {}

    if candidate_id.startswith("videomae"):
        from transformers import VideoMAEForVideoClassification  # noqa: WPS433

        model, info = VideoMAEForVideoClassification.from_pretrained(
            str(snapshot_dir),
            local_files_only=True,
            num_labels=num_dummy,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
        load_meta["loader"] = "transformers_local_files_only"
        load_meta["loading_info"] = assert_videomae_loading_info(info, allowed=allowed)
        load_meta["head"] = {
            "head_attr": "classifier",
            "new_out_features": num_dummy,
        }

        def _fwd(m, batch):
            return m(pixel_values=batch["pixel_values"]).logits

        return model, _fwd, load_meta

    if candidate_id == "video_swin_tiny":
        from torchvision.models.video import swin3d_t  # noqa: WPS433

        model = swin3d_t(weights=None)
        state = torch.load(snapshot_dir / "swin3d_t-7615ae03.pth", map_location="cpu")
        load_meta = load_state_dict_with_allowlist(model, state, allowed_keys=allowed)
        load_meta["head"] = replace_swin_head(model, num_dummy)
        install_swin_grad_checkpointing_api(model)
        load_meta["grad_checkpointing_api"] = "f0_features_checkpoint_adapter"

        def _fwd(m, batch):
            return m(batch["video"])

        return model, _fwd, load_meta

    if candidate_id == "slowfast_r50_partial_unfreeze":
        try:
            from pytorchvideo.models.hub import slowfast_r50  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pytorchvideo_required_for_slowfast") from exc

        model = slowfast_r50(pretrained=False)
        ckpt = torch.load(snapshot_dir / "SLOWFAST_8x8_R50.pyth", map_location="cpu")
        # Kinetics-400 must load with empty allowlist (strict identity).
        load_meta = load_state_dict_with_allowlist(model, ckpt, allowed_keys=frozenset())
        load_meta["head"] = replace_slowfast_head(model, num_dummy)
        load_meta["trainable_blocks"] = apply_slowfast_partial_unfreeze(model)

        def _fwd(m, batch):
            return m([batch["slow"], batch["fast"]])

        return model, _fwd, load_meta

    raise ValueError(f"unknown_candidate:{candidate_id}")


def synthetic_torch_batch(candidate_id: str, device: Any) -> dict[str, Any]:
    torch = _require_torch()
    meta = synthetic_batch_for(candidate_id)
    out: dict[str, Any] = {}
    if candidate_id.startswith("videomae"):
        shape = meta["pixel_values_ntchw"]["shape"]
        out["pixel_values"] = torch.randn(*shape, device=device)
    elif candidate_id == "video_swin_tiny":
        shape = meta["video_cthw"]["shape"]
        out["video"] = torch.randn(*shape, device=device)
    elif candidate_id == "slowfast_r50_partial_unfreeze":
        out["slow"] = torch.randn(*meta["slow_path_ncthw"]["shape"], device=device)
        out["fast"] = torch.randn(*meta["fast_path_ncthw"]["shape"], device=device)
    else:
        raise ValueError(f"unknown_candidate:{candidate_id}")
    return out


def trainable_param_identities(model: Any) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _finite(t: Any) -> bool:
    torch = _require_torch()
    return bool(torch.isfinite(t).all().item())


def assert_output_contract(logits: Any, *, batch_size: int, num_classes: int) -> None:
    torch = _require_torch()
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("output_not_tensor")
    if tuple(logits.shape) != (batch_size, num_classes):
        raise RuntimeError(f"output_shape_invalid:{tuple(logits.shape)}")
    if logits.dtype not in (torch.float16, torch.float32):
        raise RuntimeError(f"output_dtype_invalid:{logits.dtype}")


def checkpoint_roundtrip_bit_exact(model: Any, path: Path) -> bool:
    torch = _require_torch()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(model.state_dict(), path)
    loaded = torch.load(path, map_location="cpu")
    model.load_state_dict(loaded, strict=True)
    after = model.state_dict()
    for k, v in before.items():
        if not torch.equal(v, after[k].detach().cpu()):
            return False
    return True


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_state_dict(model: Any, state: dict[str, Any]) -> None:
    torch = _require_torch()
    # Move clones to model device.
    device = next(model.parameters()).device
    payload = {k: v.to(device) for k, v in state.items()}
    model.load_state_dict(payload, strict=True)


def _run_mode(
    *,
    candidate_id: str,
    model: Any,
    forward_fn: Callable,
    batch: dict[str, Any],
    device: str,
    meas: dict[str, Any],
    use_amp: bool,
    grad_checkpointing_requested: bool,
    num_classes: int,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    torch = _require_torch()
    restore_state_dict(model, initial_state)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    ckpt_meta = set_grad_checkpointing(model, requested=grad_checkpointing_requested)
    if grad_checkpointing_requested and not ckpt_meta["applied"]:
        raise RuntimeError(
            f"grad_checkpointing_requested_but_not_applied:{candidate_id}:{ckpt_meta}"
        )

    model.train()
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for _ in range(int(meas["warmup_steps"])):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = forward_fn(model, batch)
            assert_output_contract(
                logits, batch_size=int(meas["batch_size_first"]), num_classes=num_classes
            )
            loss = logits.float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # Restore again so timed steps start from identical initial weights.
    restore_state_dict(model, initial_state)
    set_grad_checkpointing(model, requested=grad_checkpointing_requested)
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    latencies_ms: list[float] = []
    min_free: int | None = None
    last_loss = None
    observed_shape: list[int] | None = None
    observed_dtype: str | None = None
    for _ in range(int(meas["timed_steps"])):
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = forward_fn(model, batch)
            torch.cuda.synchronize()
            free_mid_fwd, _ = torch.cuda.mem_get_info()
            assert_output_contract(
                logits, batch_size=int(meas["batch_size_first"]), num_classes=num_classes
            )
            observed_shape = [int(x) for x in logits.shape]
            observed_dtype = str(logits.dtype).replace("torch.", "")
            loss = logits.float().pow(2).mean()
        if not _finite(loss):
            raise RuntimeError(f"non_finite_loss:{candidate_id}")
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        free_mid_bwd, _ = torch.cuda.mem_get_info()
        for p in model.parameters():
            if p.grad is not None and not _finite(p.grad):
                raise RuntimeError(f"non_finite_grad:{candidate_id}")
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        step_min = min(int(free_before), int(free_mid_fwd), int(free_mid_bwd), int(free_after))
        min_free = step_min if min_free is None else min(min_free, step_min)
        last_loss = float(loss.detach().cpu())

    lat_sorted = sorted(latencies_ms)
    inv_count = int(getattr(model, "_f0_ckpt_invocations", 0) or 0)
    if (
        candidate_id == "video_swin_tiny"
        and ckpt_meta.get("applied") is True
        and inv_count < 1
    ):
        raise RuntimeError("swin_grad_checkpoint_not_invoked")
    return {
        "precision": "amp_fp16" if use_amp else "fp32",
        "grad_checkpointing": {
            "requested": ckpt_meta["requested"],
            "supported": ckpt_meta["supported"],
            "applied": ckpt_meta["applied"],
        },
        "checkpoint_invocation_count": inv_count,
        "last_loss": last_loss,
        "observed_output": {
            "shape": observed_shape,
            "dtype": observed_dtype,
        },
        "step_latency_ms_distribution": {
            "n": len(lat_sorted),
            "p50_ms": lat_sorted[len(lat_sorted) // 2],
            "p90_ms": lat_sorted[max(0, int(len(lat_sorted) * 0.9) - 1)],
            "mean_ms": sum(lat_sorted) / len(lat_sorted),
            "all_ms": lat_sorted,
        },
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "min_free_vram_bytes": min_free,
    }


def run_candidate_feasibility(
    candidate_id: str,
    *,
    snapshot_dir: Path,
    device: str = "cuda",
    work_dir: Path | None = None,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import gc

    torch = _require_torch()
    device = require_cuda_device(device)
    meas = measurement or MEASUREMENT
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError(f"unknown_candidate:{candidate_id}")

    model = None
    batch = None
    try:
        num_classes = int(CANDIDATE_MEASUREMENT[candidate_id]["num_classes_dummy"])
        model, forward_fn, load_meta = build_model_offline(candidate_id, snapshot_dir)
        model = model.to(device)
        params = trainable_param_identities(model)
        if not params:
            raise RuntimeError(f"no_trainable_params:{candidate_id}")

        initial_state = clone_state_dict(model)
        batch = synthetic_torch_batch(candidate_id, device)
        modes = []
        for mode in measurement_modes_for(candidate_id):
            modes.append(
                _run_mode(
                    candidate_id=candidate_id,
                    model=model,
                    forward_fn=forward_fn,
                    batch=batch,
                    device=device,
                    meas=meas,
                    use_amp=bool(mode["use_amp"]),
                    grad_checkpointing_requested=bool(mode["grad_checkpointing_requested"]),
                    num_classes=num_classes,
                    initial_state=initial_state,
                )
            )

        work_dir = work_dir or (snapshot_dir / "_f0_ckpt_tmp")
        work_dir.mkdir(parents=True, exist_ok=True)
        restore_state_dict(model, initial_state)
        ckpt_path = work_dir / f"{candidate_id}_roundtrip.pt"
        bit_exact = checkpoint_roundtrip_bit_exact(model, ckpt_path)

        return {
            "candidate_id": candidate_id,
            "ok": True,
            "device": device,
            "trainable_param_count": len(params),
            "trainable_param_identities_sample": params[:16],
            "load_meta": load_meta,
            "modes": modes,
            "checkpoint_save_load_bit_exact": bit_exact,
            "offline_only": True,
            "layout": LAYOUT_CONTRACT[candidate_id]["layout"],
            "output_tensor_contract": {"shape": [1, num_classes], "dtype": "float16|float32"},
        }
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            del batch
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def run_all_candidates_offline(
    *,
    weights_root: Path,
    snapshot_contract: dict[str, Any],
    device: str = "cuda",
    work_dir: Path | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import gc

    from .contracts import assert_candidate_measurement_matches_plan

    torch = _require_torch()
    device = require_cuda_device(device)
    if plan is not None:
        assert_candidate_measurement_matches_plan(plan)

    results: list[dict[str, Any]] = []
    for cid in CANDIDATE_IDS:
        try:
            snap_dir = candidate_snapshot_dir(weights_root, cid)
            required = list_required_local_files(snapshot_contract, cid)
            assert_local_snapshot_complete(snap_dir, required)
            results.append(
                run_candidate_feasibility(
                    cid,
                    snapshot_dir=snap_dir,
                    device=device,
                    work_dir=None if work_dir is None else work_dir / cid,
                )
            )
        except Exception as exc:  # noqa: BLE001 - per-candidate structured failure
            results.append(
                {
                    "candidate_id": cid,
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "device": device,
                }
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    if [r.get("candidate_id") for r in results] != list(CANDIDATE_IDS):
        raise RuntimeError("candidate_report_incomplete")
    return {
        "schema": "ms_sta_v5_f0_feasibility_cuda_report_v1",
        "ok": all(bool(r.get("ok")) for r in results),
        "device": device,
        "candidates": results,
        "candidate_count": len(results),
        "measurement_contract": MEASUREMENT,
        "network_fetch": False,
        "cpu_forbidden": True,
        "oneshot_required": True,
        "oneshot": True,
    }


def implementation_surface() -> dict[str, Any]:
    return {
        "module": "scripts/f0_feasibility/gpu_runner.py",
        "candidates": list(CANDIDATE_IDS),
        "entrypoints": [
            "build_model_offline",
            "run_candidate_feasibility",
            "run_all_candidates_offline",
            "load_state_dict_with_allowlist",
            "assert_videomae_loading_info",
            "install_swin_grad_checkpointing_api",
            "replace_slowfast_head",
            "set_grad_checkpointing",
            "measurement_modes_for",
            "require_cuda_device",
        ],
        "offline_only": True,
        "cpu_forbidden": True,
        "forbid_from_pretrained_network": True,
        "measurement_keys": sorted(MEASUREMENT.keys()),
        "precision_modes": ["fp32", "amp_fp16"],
        "grad_checkpointing_truthful": True,
        "classifier_allowlists": {k: sorted(v) for k, v in CLASSIFIER_KEY_ALLOWLIST.items()},
    }


# Re-export for tests / runner.
__all__ = [
    "assert_videomae_loading_info",
    "build_model_offline",
    "implementation_surface",
    "load_state_dict_with_allowlist",
    "measurement_modes_for",
    "replace_slowfast_head",
    "require_cuda_device",
    "run_all_candidates_offline",
    "run_candidate_feasibility",
    "set_grad_checkpointing",
]
