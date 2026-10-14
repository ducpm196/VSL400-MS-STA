"""Create-only bootstrap artifacts for E4_SYSKD_V1 (policy draft phase)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(r"D:/TTNT/DNTA/VSL_E4_SYSKD_V1")
SRC = pathlib.Path(r"D:/TTNT/DNTA/VSL_Translation_Project")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256(path)


def resolve(rel: str) -> pathlib.Path:
    for base in (ROOT, SRC):
        p = base / rel
        if p.exists():
            return p
    return ROOT / rel


def main() -> int:
    os.chdir(ROOT)
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    branch = "research/e4-system-kd-v1"
    base_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    main_sha = "1a37beb2d7c5df8b4ee8bc5026175356006ae60c"
    if base_sha != main_sha:
        print(f"BLOCKED: worktree HEAD {base_sha} != expected main SHA {main_sha}", file=sys.stderr)
        return 2

    bindings = {
        "b4_checkpoint_rel": "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
        "b4_checkpoint_sha256": sha256(
            resolve("Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt")
        ),
        "slowfast_checkpoint_rel": (
            "Docs/results/ms_sta_v5_f0/weights/slowfast_r50_partial_unfreeze/SLOWFAST_8x8_R50.pyth"
        ),
        "split_manifest_rel": "data/processed/vsl400_retrain_v4_split_manifest.json",
        "split_manifest_sha256": sha256(resolve("data/processed/vsl400_retrain_v4_split_manifest.json")),
        "labels_rel": "data/processed/vsl400_labels.json",
        "labels_sha256": sha256(resolve("data/processed/vsl400_labels.json")),
        "eligible_manifest_rel": "data/processed/vsl400_eligible_sample_manifest.json",
        "eligible_manifest_sha256": sha256(resolve("data/processed/vsl400_eligible_sample_manifest.json")),
        "active_registry_pointer_rel": "data/processed/ms_sta_registry_active.json",
        "active_registry_pointer_sha256": sha256(resolve("data/processed/ms_sta_registry_active.json")),
        "defense_registry_rel": "data/processed/ms_sta_registry_v6_defense_go.json",
        "defense_registry_sha256": sha256(resolve("data/processed/ms_sta_registry_v6_defense_go.json")),
        "gate_a_schedule_rel": "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json",
        "gate_a_schedule_sha256": sha256(
            resolve("Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json")
        ),
        "e1_schedule_lock_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V1_E1_SCHEDULE_LOCK.json",
        "v2_s5_policy_lock_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_POLICY_LOCK.json",
    }
    sf = resolve(bindings["slowfast_checkpoint_rel"])
    bindings["slowfast_checkpoint_sha256"] = (
        sha256(sf) if sf.exists() else "SOURCE_GAP_FILE_MISSING"
    )

    cap = {
        "schema": "e4_syskd_v1_capability_inventory_v1",
        "campaign": "E4_SYSKD_V1",
        "created_at": now,
        "status": "CREATE_ONLY",
        "environment": {
            "os": platform.platform(),
            "python_locked": str(SRC / "_venvs/train_cuda/Scripts/python.exe"),
            "torch": "2.5.1+cu121",
            "cuda_available": True,
            "gpu_name": "NVIDIA GeForce RTX 2060",
            "cudnn": 90100,
            "driver_smi": "577.00",
            "cuda_toolkit_env": os.environ.get("CUDA_PATH", ""),
            "disk_free_gb_repo_host": round(shutil.disk_usage(str(ROOT)).free / 1e9, 2),
        },
        "optional_deps": {
            "tensorrt": "NOT_AVAILABLE",
            "onnxruntime": "NOT_AVAILABLE",
            "triton": "NOT_AVAILABLE",
            "torchao": "NOT_AVAILABLE",
            "nvidia.dali": "NOT_AVAILABLE",
            "install_policy": "do_not_install_in_this_campaign",
        },
        "arms_feasibility_preview": {
            "S1_pinned_transfer": "LIKELY_AVAILABLE",
            "S2_decode_overlap": "LIKELY_AVAILABLE",
            "S3_roi_batching": "BLOCKED_UNTIL_TOLERANCE_FREEZE",
            "S4_cuda_stream_graph": "CONDITIONAL",
            "Q4_tensorrt": "NOT_RUN_ENVIRONMENT_UNAVAILABLE",
            "Q5_int8_torchao": "NOT_RUN_ENVIRONMENT_UNAVAILABLE",
        },
        "historical_immutable": {
            "parent_terminal": "E4_CONTROLLED_NO_GO_B4_RETAINED",
            "selected": None,
            "e4_ready": False,
            "b4_retained": True,
        },
        "bindings": bindings,
    }

    policy = {
        "schema": "e4_syskd_v1_policy_draft_v1",
        "campaign": "E4_SYSKD_V1",
        "status": "POLICY_DRAFT_AWAITING_FREEZE",
        "immutable": False,
        "created_at": now,
        "branch": branch,
        "base_commit_sha": base_sha,
        "e4_ready": False,
        "selected": None,
        "b4_source_of_truth": True,
        "namespaces": {
            "policy_dir": "Docs/results/e4_system_kd_v1_policy/",
            "artifacts_dir": "Docs/results/e4_system_kd_v1/",
            "registry_dir": "Docs/results/e4_system_kd_v1_registry/",
        },
        "parent_locks_immutable": {
            "E4_FINAL": "Docs/results/ms_sta_v6_e4_final/E4_FINAL_CONTROLLED_NO_GO.json",
            "B4Q_POLICY": "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json",
            "B4SC_V1_POLICY": "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json",
            "B4SC_V2_POLICY": "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V2_POLICY.json",
            "V2_FIDELITY_GATE": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_FIDELITY_GATE.json",
            "V2_S5_POLICY_LOCK": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_POLICY_LOCK.json",
            "E1_SCHEDULE_LOCK": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V1_E1_SCHEDULE_LOCK.json",
        },
        "forbid": {
            "reopen_v1_output_width_narrowing": True,
            "reopen_v2_hard_channel_transplant": True,
            "reopen_v3_factorized_conv_b_rank_448_480": True,
            "uniform_fast_frame_reduction": True,
            "drop_face_hand_rois": True,
            "change_joint_subset": True,
            "change_split": True,
            "use_test_to_select_architecture": True,
            "merge_speedups_across_schedules": True,
            "set_tolerance_after_seeing_metrics": True,
            "mutate_b4_checkpoint": True,
            "open_locked_test_d4_demo_until_promotion": True,
            "install_new_optional_deps": True,
            "term_nguoi_ky_for_signer_id": True,
        },
        "terminology": {
            "signer_id": "mã người thực hiện ký hiệu (mã người ra dấu)",
            "per_signer_accuracy": "độ chính xác theo từng người thực hiện ký hiệu",
            "signer_regression": "suy giảm hiệu năng trên từng người thực hiện ký hiệu",
            "signer_disjoint_split": "phân chia dữ liệu tách biệt theo người thực hiện ký hiệu",
        },
        "architecture_invariants": {
            "C_out": [256, 512, 1024, 2048],
            "fast_widths": [32, 64, 128, 256],
            "roi_regions_order": ["global", "left_hand", "right_hand", "face"],
            "roi_token_dim": 2304,
            "n_joints": 75,
            "require_coordinates_velocity_validity": True,
            "cross_attention": "Skeleton-query → RGB-key/value one-way",
            "stream_gate": True,
            "bounded_residual": True,
            "n_classes": 400,
        },
        "student_candidates": {
            "K1_light": {
                "C_mid": [56, 112, 224, 448],
                "note": "trained compact Slow internal bottleneck; NOT V2 hard transplant",
            },
            "K2_75pct": {
                "C_mid": [48, 96, 192, 384],
                "requires": ["K1_accuracy_pass"],
            },
            "init": "standard_or_reconstruction_locked_before_metrics; no teacher-equivalent-at-init claim",
            "forbid_v2_transplant_fidelity_as_init_requirement": True,
            "hardware_feasibility_before_train": True,
        },
        "comparators": {
            "R0": "B4 E0 FP32 sequential decode canonical",
            "R1": "B4 AMP Q2/E1a remeasured in E4_SYSKD harness",
            "system_arms": ["S1", "S2", "S3", "S4", "S5"],
            "compression_arms": ["K0", "K1", "K2", "K3", "K4"],
            "factorial_attribution_required": True,
        },
        "accuracy_gate": {
            "source": "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json#accuracy_gate",
            "paired_vs": "R0",
            "eval_role": "val",
            "n_boot": 10000,
            "bootstrap_seed": 42,
            "hierarchical_ci95_lower_top1_pp_min": -0.3,
            "delta_macro_f1_pp_min": -0.5,
            "delta_top5_pp_min": -0.3,
            "per_signer_delta_top1_pp_min": -1.0,
            "forbid_nan_inf": True,
            "require_prediction_dump": True,
            "note": "Copied from frozen B4SC V1 accuracy_gate BEFORE SYSKD metrics.",
        },
        "efficiency_gate": {
            "source": "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json#efficiency_gate",
            "e2e_median_speedup_min": 1.15,
            "e2e_speedup_ci95_lower_min": 1.08,
            "forbid_p90_regression": True,
            "forbid_vram_increase_vs_r0": True,
            "accuracy_before_efficiency_ranking": True,
            "batch_primary": 1,
            "full_pipeline_rounds_min": 5,
            "sample_schedule": "24 IDs from B4Q Gate A / E1 schedule lock; remasured interleaved",
            "warmup_gpu_stages_min": 100,
            "measured_trials_gpu_stages_min": 500,
        },
        "system_parity_gate_draft": {
            "source_parity_bounds": (
                "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json"
                "#parity_thresholds_declared_before_metrics"
            ),
            "exact_frame_sha_required_for_transfer_decode_arms": True,
            "exact_sample_window_order": True,
            "logit_max_abs_fp16_vs_fp32": 0.5,
            "logit_mean_abs_fp16_vs_fp32": 0.05,
            "require_finite": True,
            "require_topk1_agreement_rate_min": 0.95,
            "token_shape_exact": True,
            "region_order_exact": True,
            "roi_batching_numeric_tolerance": {
                "status": "SOURCE_GAP",
                "blocks": ["S3", "policy_freeze_if_S3_in_scope"],
                "reason": (
                    "No a-priori locked tolerance for batched ROI vs sequential ROI under "
                    "cuDNN reorder. Must propose+freeze before S3 metrics."
                ),
                "proposed_before_metrics_candidate": {
                    "logit_max_abs": 1e-4,
                    "logit_mean_abs": 1e-5,
                    "require_topk1_agreement_rate_min": 1.0,
                    "motivation": "historical E1 batched-window numeric drift ~1e-5; NOT authorizing post-hoc",
                },
            },
        },
        "kd_supporting_diagnostics": {
            "source_fidelity_gate": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_FIDELITY_GATE.json",
            "source_temporal_diff": (
                "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_POLICY_LOCK.json"
                "#temporal_diff_cosine_min"
            ),
            "role": "KD directional/supporting diagnostics for trained student; NOT transplant init gate",
            "teacher_student_pred_agreement_min_on_train_calib": 0.90,
            "roi_fidelity_cosine_min": {
                "left_hand": 0.85,
                "right_hand": 0.85,
                "face": 0.85,
                "global": 0.85,
            },
            "temporal_diff_cosine_min": 0.8,
            "temporal_diff_status": "LOCKED_FROM_V2_S5_POLICY_LOCK",
        },
        "hardware_feasibility_pretrain_rule": {
            "student_K1_must_benchmark_full_SlowFast_ROI_e2e_before_train": True,
            "if_amdahl_upper_bound_lt_1_15x": "E4_SYSKD_V1_STUDENT_HARDWARE_NO_GO",
        },
        "confirmation_seeds": [42, 123, 456],
        "test_opening_rule": "Only after confirmation PASS + TEST_OPENING_AUTHORIZATION.json",
        "stop_rules": {
            "baseline_repro_fail": "E4_SYSKD_V1_BASELINE_REPRO_NO_GO",
            "system_fail": "E4_SYSKD_V1_SYSTEM_NO_GO",
            "student_hw_fail": "E4_SYSKD_V1_STUDENT_HARDWARE_NO_GO",
            "kd_mechanics_fail": "E4_SYSKD_V1_KD_MECHANICS_NO_GO",
            "k1_accuracy_fail": "E4_SYSKD_V1_K1_ACCURACY_NO_GO",
        },
        "source_gaps_blocking_freeze": [
            {
                "id": "ROI_BATCHING_NUMERIC_TOLERANCE",
                "path": "system_parity_gate_draft.roi_batching_numeric_tolerance",
                "options": [
                    "Propose+accept candidate tolerances then freeze",
                    "Exclude S3/S5-with-S3 from V1 scope and freeze without S3",
                ],
            }
        ],
        "bindings": bindings,
    }

    status = {
        "schema": "e4_syskd_v1_campaign_status_v1",
        "campaign": "E4_SYSKD_V1",
        "created_at": now,
        "updated_at": now,
        "phase": "CAPTURE_POLICY_DRAFT",
        "status": "POLICY_DRAFT_AWAITING_FREEZE",
        "selected": None,
        "e4_ready": False,
        "b4_retained": True,
        "branch": branch,
        "base_commit_sha": base_sha,
        "worktree": str(ROOT),
        "main_repo_dirty_note": (
            "Primary repo main may have unrelated HTML/TeX edits; "
            "campaign uses clean worktree at base SHA."
        ),
        "next_allowed_action": (
            "Resolve ROI batching tolerance SOURCE_GAP (accept proposal or exclude S3), "
            "then freeze policy with commit `research(e4): freeze E4_SYSKD_V1 policy`. "
            "Do not run Gate A / student train until freeze."
        ),
        "opened": [],
        "not_opened": ["GateA", "S1-S5", "K0-K4", "confirmation", "test", "D4", "demo"],
    }

    policy_dir = ROOT / "Docs/results/e4_system_kd_v1_policy"
    art_dir = ROOT / "Docs/results/e4_system_kd_v1"
    reg_dir = ROOT / "Docs/results/e4_system_kd_v1_registry"
    for d in (policy_dir, art_dir, reg_dir):
        d.mkdir(parents=True, exist_ok=True)

    h_pol = write_json(policy_dir / "E4_SYSKD_V1_POLICY_DRAFT.json", policy)
    h_cap = write_json(art_dir / "E4_SYSKD_V1_CAPABILITY_INVENTORY.json", cap)
    h_st = write_json(art_dir / "E4_SYSKD_V1_CAMPAIGN_STATUS.json", status)
    h_reg = write_json(
        reg_dir / "CHECKPOINT_REGISTRY.json",
        {
            "schema": "e4_syskd_v1_registry_v1",
            "created_at": now,
            "selected": None,
            "e4_ready": False,
            "b4_retained": True,
            "checkpoints": [],
        },
    )

    ledger_path = art_dir / "E4_SYSKD_V1_RUN_LEDGER.jsonl"
    if ledger_path.exists():
        raise FileExistsError(ledger_path)
    entry = {
        "ts": now,
        "event": "campaign_bootstrap",
        "branch": branch,
        "base_commit_sha": base_sha,
        "status": "POLICY_DRAFT_AWAITING_FREEZE",
        "artifacts": {
            "policy_draft": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY_DRAFT.json",
            "policy_draft_sha256": h_pol,
            "capability": "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_CAPABILITY_INVENTORY.json",
            "capability_sha256": h_cap,
            "status": "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_CAMPAIGN_STATUS.json",
            "status_sha256": h_st,
            "registry": "Docs/results/e4_system_kd_v1_registry/CHECKPOINT_REGISTRY.json",
            "registry_sha256": h_reg,
        },
        "source_gaps": policy["source_gaps_blocking_freeze"],
    }
    ledger_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "policy_sha": h_pol,
                "cap_sha": h_cap,
                "status_sha": h_st,
                "reg_sha": h_reg,
                "b4_ckpt": bindings["b4_checkpoint_sha256"],
                "slowfast": bindings["slowfast_checkpoint_sha256"],
                "status": "POLICY_DRAFT_AWAITING_FREEZE",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
