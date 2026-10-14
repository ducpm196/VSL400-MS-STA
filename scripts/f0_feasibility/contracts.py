"""F0 feasibility contracts and plan-derived measurement matrices (no GPU)."""

from __future__ import annotations

from typing import Any

CANDIDATE_IDS = (
    "slowfast_r50_partial_unfreeze",
    "videomae_vit_small",
    "videomae_vit_base",
    "video_swin_tiny",
)

SLOWFAST_TRAINABLE_BLOCKS = {
    "mode": "partial_unfreeze",
    "frozen": ["blocks.0", "blocks.1", "blocks.2"],
    "trainable": ["blocks.3", "blocks.4", "blocks.5"],
}


def matches_module_prefix(name: str, prefix: str) -> bool:
    """True when ``name`` is exactly ``prefix`` or a nested child under that module.

    Substring checks are forbidden: ``blocks.0`` must not match
    ``blocks.3.multipathway_blocks.0.res_blocks.0``.
    """
    return name == prefix or name.startswith(prefix + ".")


def any_module_prefix(names: list[str] | tuple[str, ...] | set[str], prefix: str) -> bool:
    return any(matches_module_prefix(name, prefix) for name in names)

# Exact classifier key allowlists (no broad substring matching).
CLASSIFIER_KEY_ALLOWLIST = {
    "videomae_vit_small": frozenset({"classifier.weight", "classifier.bias"}),
    "videomae_vit_base": frozenset({"classifier.weight", "classifier.bias"}),
    "video_swin_tiny": frozenset({"head.weight", "head.bias"}),
    "slowfast_r50_partial_unfreeze": frozenset(),  # Kinetics-400 must load strict
}

# Mirrors sealed F0_PROTOCOL_PLAN.json candidate measurement fields.
CANDIDATE_MEASUREMENT = {
    "slowfast_r50_partial_unfreeze": {
        "precision_modes": ("fp32", "amp_fp16"),
        "grad_checkpointing": False,
        "num_classes_pretrained": 400,
        "num_classes_dummy": 100,
    },
    "videomae_vit_small": {
        "precision_modes": ("fp32", "amp_fp16"),
        "grad_checkpointing": True,
        "num_classes_pretrained": 400,
        "num_classes_dummy": 100,
    },
    "videomae_vit_base": {
        "precision_modes": ("fp32", "amp_fp16"),
        "grad_checkpointing": True,
        "num_classes_pretrained": 400,
        "num_classes_dummy": 100,
    },
    "video_swin_tiny": {
        "precision_modes": ("fp32", "amp_fp16"),
        "grad_checkpointing": True,
        "num_classes_pretrained": 400,
        "num_classes_dummy": 100,
    },
}

LAYOUT_CONTRACT = {
    "slowfast_r50_partial_unfreeze": {
        "layout": "CTHW_dual_pathway",
        "paths": ("slow_path_ncthw", "fast_path_ncthw"),
    },
    "videomae_vit_small": {
        "layout": "NTCHW",
        "paths": ("pixel_values_ntchw",),
        "runtime_assert": "processor_pixel_values_shape_NTCHW",
    },
    "videomae_vit_base": {
        "layout": "NTCHW",
        "paths": ("pixel_values_ntchw",),
        "runtime_assert": "processor_pixel_values_shape_NTCHW",
    },
    "video_swin_tiny": {
        "layout": "CTHW",
        "paths": ("video_cthw",),
    },
}

FEASIBILITY_REPORT_REL = "Docs/results/ms_sta_v5_f0/F0_FEASIBILITY_REPORT.json"
FEASIBILITY_COMPLETION_REL = "Docs/results/ms_sta_v5_f0/F0_FEASIBILITY_COMPLETION.json"
SEAL_CONSUMPTION_REL = "Docs/results/ms_sta_v5_f0/F0_EXECUTION_SEAL_CONSUMPTION.json"
EXECUTION_ATTEMPT_REL = "Docs/results/ms_sta_v5_f0/F0_EXECUTION_ATTEMPT.json"
FEASIBILITY_STAGING_DIR_REL = "Docs/results/ms_sta_v5_f0/.f0_feasibility_staging"
FEASIBILITY_READY_REL = "Docs/results/ms_sta_v5_f0/F0_FEASIBILITY_READY.json"
FEASIBILITY_ARTIFACT_PREFIX = "F0"


def configure_feasibility_artifacts(*, campaign: str = "f0") -> None:
    """Point oneshot publish/verify paths at F0 or versioned F0-R1 artifacts."""
    from .campaign_paths import campaign_paths

    global FEASIBILITY_REPORT_REL
    global FEASIBILITY_COMPLETION_REL
    global SEAL_CONSUMPTION_REL
    global EXECUTION_ATTEMPT_REL
    global FEASIBILITY_STAGING_DIR_REL
    global FEASIBILITY_READY_REL
    global FEASIBILITY_ARTIFACT_PREFIX

    paths = campaign_paths(campaign)
    FEASIBILITY_REPORT_REL = paths.feasibility_report_rel
    FEASIBILITY_COMPLETION_REL = paths.feasibility_completion_rel
    SEAL_CONSUMPTION_REL = paths.seal_consumption_rel
    EXECUTION_ATTEMPT_REL = paths.execution_attempt_rel
    FEASIBILITY_STAGING_DIR_REL = paths.feasibility_staging_dir_rel
    FEASIBILITY_READY_REL = paths.feasibility_ready_rel
    FEASIBILITY_ARTIFACT_PREFIX = paths.artifact_prefix


def assert_candidate_measurement_matches_plan(plan: dict[str, Any]) -> None:
    """Full measurement/input/output contract check vs locked protocol plan."""
    by_id = {c["id"]: c for c in (plan.get("candidates") or [])}
    plan_meas = plan.get("measurement_contract") or {}
    required_meas_keys = {
        "batch_size_first",
        "optimizer_step_required",
        "require_finite_loss_and_grads",
        "warmup_steps",
        "timed_steps",
        "amp_logged_separately",
        "grad_checkpointing_logged_separately",
        "record_peak_allocated_vram_bytes",
        "record_peak_reserved_vram_bytes",
        "record_min_free_vram_bytes",
        "record_step_latency_ms_distribution",
        "checkpoint_save_load_bit_exact",
        "output_tensor_contract_required",
        "forbid_single_step_latency_claim",
    }
    missing = sorted(required_meas_keys - set(plan_meas.keys()))
    if missing:
        raise ValueError("plan_measurement_contract_missing:" + ",".join(missing))

    for cid, spec in CANDIDATE_MEASUREMENT.items():
        pc = by_id.get(cid)
        if not pc:
            raise ValueError(f"plan_missing_candidate:{cid}")
        if list(pc.get("precision_modes") or []) != list(spec["precision_modes"]):
            raise ValueError(f"{cid}:precision_modes_drift_from_plan")
        if pc.get("grad_checkpointing") is not bool(spec["grad_checkpointing"]):
            raise ValueError(f"{cid}:grad_checkpointing_drift_from_plan")
        layout = (pc.get("input") or {}).get("layout")
        if layout != LAYOUT_CONTRACT[cid]["layout"]:
            raise ValueError(f"{cid}:input_layout_drift_from_plan")
        ost = pc.get("optimizer_step_tensor_contract") or {}
        if list(ost.get("inputs") or []) != list(LAYOUT_CONTRACT[cid]["paths"]):
            raise ValueError(f"{cid}:optimizer_inputs_drift_from_plan")
        outputs = str(ost.get("outputs") or "")
        if "C_dummy" not in outputs and "logits" not in outputs:
            raise ValueError(f"{cid}:optimizer_outputs_contract_missing")
        tb = pc.get("trainable_blocks") or {}
        if cid == "slowfast_r50_partial_unfreeze":
            if list(tb.get("frozen") or []) != list(SLOWFAST_TRAINABLE_BLOCKS["frozen"]):
                raise ValueError(f"{cid}:trainable_frozen_drift_from_plan")
            if list(tb.get("trainable") or []) != list(SLOWFAST_TRAINABLE_BLOCKS["trainable"]):
                raise ValueError(f"{cid}:trainable_blocks_drift_from_plan")
            if tb.get("mode") != SLOWFAST_TRAINABLE_BLOCKS["mode"]:
                raise ValueError(f"{cid}:trainable_mode_drift_from_plan")
        elif not isinstance(tb.get("trainable"), list) or not tb.get("trainable"):
            raise ValueError(f"{cid}:trainable_blocks_missing")


def measurement_modes_for(candidate_id: str) -> list[dict[str, Any]]:
    """Build precision × grad-ckpt matrix from sealed plan intent."""
    spec = CANDIDATE_MEASUREMENT[candidate_id]
    grad_flags = [False]
    if spec["grad_checkpointing"] is True:
        grad_flags.append(True)
    modes = []
    for prec in spec["precision_modes"]:
        for grad in grad_flags:
            modes.append(
                {
                    "precision": prec,
                    "grad_checkpointing_requested": grad,
                    "use_amp": prec == "amp_fp16",
                }
            )
    return modes


def feasibility_plan(*, allow_gpu: bool = False) -> dict[str, Any]:
    return {
        "schema": "ms_sta_v5_f0_feasibility_plan_v1",
        "ok": True,
        "gpu_authorized": False,
        "allow_gpu_requested": allow_gpu,
        "candidates": list(CANDIDATE_IDS),
        "measurement_by_candidate": {
            cid: {
                **CANDIDATE_MEASUREMENT[cid],
                "modes": measurement_modes_for(cid),
            }
            for cid in CANDIDATE_IDS
        },
        "steps_per_candidate": [
            "build_synthetic_batch_under_layout_contract",
            "assert_preprocess_shapes",
            "strict_or_exact_classifier_local_checkpoint_load",
            "replace_dummy_head_100",
            "verify_trainable_block_identities",
            "restore_initial_state_per_mode",
            "forward_fp32_amp_matrix",
            "truthful_grad_checkpointing_requested_supported_applied",
            "backward_optimizer_step",
            "finite_loss_and_grads",
            "min_free_vram_across_steps",
            "checkpoint_bit_exact_roundtrip",
            "atomic_oneshot_report_completion_seal_consumption",
        ],
        "forbid": [
            "accuracy_metrics",
            "network_weight_download_inside_gpu_run",
            "from_pretrained_hub_fetch",
            "cpu_device_for_gpu_report",
            "mutate_train_or_demo_env",
            "ignore_refuse_reasons",
            "reuse_execution_seal",
            "report_grad_ckpt_applied_when_unsupported",
        ],
        "status": "IMPLEMENTATION_PRESENT_AWAITING_EXECUTION_SEAL",
        "watchdog_policy": {
            "max_restarts_before_attempt": 2,
            "after_attempt": "heal_or_verify_only_no_inference",
        },
        "oneshot_artifacts": [
            EXECUTION_ATTEMPT_REL,
            FEASIBILITY_READY_REL,
            FEASIBILITY_REPORT_REL,
            FEASIBILITY_COMPLETION_REL,
            SEAL_CONSUMPTION_REL,
        ],
    }
