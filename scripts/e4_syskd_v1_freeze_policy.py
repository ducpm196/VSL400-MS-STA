"""Freeze E4_SYSKD_V1 policy with approved S3 a-priori parity tolerances."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(r"D:/TTNT/DNTA/VSL_E4_SYSKD_V1")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, obj: dict, *, overwrite: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    # UTF-8 LF only — avoid Windows CRLF changing content hashes.
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def main() -> int:
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    draft_path = ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY_DRAFT.json"
    policy_path = ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"
    status_path = ROOT / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_CAMPAIGN_STATUS.json"
    ledger_path = ROOT / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_RUN_LEDGER.jsonl"
    meta_path = ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY_FREEZE_META.json"

    if policy_path.exists():
        print("BLOCKED: POLICY already exists (immutable create-only)", file=sys.stderr)
        return 2

    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    # Code-canonical ROI order (src/models/rgb_token_v3.py REGIONS / RGB_REGIONS).
    # Prose "global/face/left/right" is NOT used.
    code_roi_order = ["global", "left_hand", "right_hand", "face"]

    s3 = {
        "arm_id": "S3",
        "label": "R1 + vectorized/batched ROI extraction",
        "independent_optional_arm": True,
        "comparator": "R1",
        "forbid_compare_to_R0_for_parity": True,
        "parity_vs_efficiency_separation": True,
        "requires_same": [
            "b4_checkpoint",
            "amp_precision",
            "decoded_frame_bytes",
            "sample_window_order",
            "roi_coordinates",
            "roi_masks",
            "roi_region_order",
        ],
        "parity_probe": {
            "sample_schedule_rel": draft["bindings"]["gate_a_schedule_rel"],
            "sample_schedule_sha256": draft["bindings"]["gate_a_schedule_sha256"],
            "n_ids": 24,
            "forbid_replace_ids_after_seeing_diff": True,
            "deterministic_reference_order": True,
            "require_raw_rows": True,
            "require_cuda_synchronize_before_tensor_or_timing_read": True,
        },
        "roi_regions_order_canonical_code": code_roi_order,
        "roi_regions_order_note": (
            "Locked to code tuple in src/models/rgb_token_v3.py REGIONS / RGB_REGIONS; "
            "do not reorder to match prose global/face/left-hand/right-hand."
        ),
        "formal_parity_gates": {
            "frame_sha_exact_match_24_of_24": True,
            "window_frame_indices_exact": True,
            "roi_coordinates_exact": True,
            "roi_region_order_exact": code_roi_order,
            "roi_masks_exact": True,
            "shapes_dtypes_exact": True,
            "nan_inf_count_max": 0,
            "pre_softmax_logits_max_abs_diff": 1e-4,
            "pre_softmax_logits_mean_abs_diff": 1e-5,
            "top1_prediction_agreement": 1.0,
            "top5_label_set_agreement": 1.0,
            "top5_compares_label_sets_not_rank_order": True,
            "sample_output_order_exact": True,
            "role": "formal_policy_gate_not_diagnostic",
        },
        "supporting_diagnostics_no_threshold_after_run": [
            "roi_token_max_mean_abs_diff",
            "rgb_temporal_token_cosine",
            "logit_cosine",
            "top1_top2_margin",
            "n_samples_with_logit_margin_lt_max_observed_drift",
        ],
        "fail_rules": {
            "parity_fail_terminal": "S3_ROI_BATCHING_PARITY_NO_GO",
            "parity_fail_actions": [
                "reject_S3",
                "do_not_change_tolerance",
                "do_not_change_ids",
                "do_not_rerun_with_new_threshold",
                "do_not_override_with_aggregate_accuracy",
                "do_not_include_S3_in_S5",
                "continue_campaign_with_S1_S2_S4_and_student_hw",
            ],
            "parity_fail_does_not_auto_system_no_go": True,
            "efficiency_fail_after_parity_pass_terminal": "S3_ROI_BATCHING_EFFICIENCY_NO_GO",
            "efficiency_fail_excludes_from_S5": True,
        },
        "efficiency_after_parity_pass_only": {
            "metrics": [
                "e2e_median",
                "e2e_p90",
                "slowfast_roi_time",
                "kernel_count_if_available",
                "synchronization",
                "vram",
                "ci_bootstrap",
                "speedup_vs_R1",
            ],
            "speedup_vs_R0_is_supplemental_report_only": True,
            "forbid_promote_from_roi_microbench_alone": True,
            "must_improve_e2e_per_system_arm_policy": True,
        },
        "s5_inheritance": {
            "eligible_only_if_parity_and_efficiency_pass": True,
            "s5_must_rerun_own_parity_and_efficiency": True,
            "no_automatic_inheritance_of_S3_pass": True,
        },
        "approval": {
            "decision": "APPROVED_FREEZE_WITH_S3_ROI_BATCHING",
            "option": 1,
            "tolerance_approved_before_any_S3_metric": True,
            "observed_S3_diff_at_freeze": False,
            "thresholds_immutable_within_V1": True,
        },
    }

    policy = dict(draft)
    policy.update(
        {
            "schema": "e4_syskd_v1_policy_v1",
            "status": "FROZEN",
            "immutable": True,
            "frozen_at": now,
            "freeze_base_commit_sha": head,
            "draft_superseded": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY_DRAFT.json",
            "e4_ready": False,
            "selected": None,
            "b4_source_of_truth": True,
        }
    )
    # Close SOURCE_GAP by locking S3 parity.
    policy["system_parity_gate_draft"] = {
        **draft.get("system_parity_gate_draft", {}),
        "roi_batching_numeric_tolerance": {
            "status": "FROZEN",
            "source": "user_approval_option_1_before_S3_metrics",
            "formal_gates": s3["formal_parity_gates"],
            "comparator": "R1",
        },
    }
    policy["s3_roi_batching"] = s3
    policy["source_gaps_blocking_freeze"] = []
    policy["source_gaps_resolved"] = [
        {
            "id": "ROI_BATCHING_NUMERIC_TOLERANCE",
            "resolved_at": now,
            "resolution": "Option 1: freeze a-priori tolerances; keep S3 in campaign",
        }
    ]
    policy["architecture_invariants"]["roi_regions_order"] = code_roi_order
    policy["next_after_freeze"] = [
        "Gate A R0/R1 reproduction",
        "S1/S2/S3/S4 independent",
        "S5 only from parity+efficiency PASS arms",
        "Student full-architecture hardware feasibility",
        "Train K1 only if Amdahl upper bound can reach 1.15x",
        "Do not open test/D4/demo",
        "selected=null, e4_ready=false until promotion terminal",
    ]

    # Remove mutable draft-only fields that conflict with freeze semantics.
    policy.pop("status", None)
    policy["status"] = "FROZEN"

    h_policy = write_json(policy_path, policy, overwrite=False)

    # Update draft to pointer (mutable record of supersession).
    draft_update = dict(draft)
    draft_update.update(
        {
            "status": "SUPERSEDED_BY_FROZEN_POLICY",
            "immutable": False,
            "superseded_at": now,
            "frozen_policy_rel": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json",
            "frozen_policy_sha256": h_policy,
            "source_gaps_blocking_freeze": [],
            "s3_decision": "Option1_keep_S3_freeze_a_priori_tolerance",
        }
    )
    h_draft = write_json(draft_path, draft_update, overwrite=True)

    status.update(
        {
            "updated_at": now,
            "phase": "POLICY_FROZEN",
            "status": "POLICY_FROZEN_AWAITING_GATE_A",
            "selected": None,
            "e4_ready": False,
            "b4_retained": True,
            "frozen_policy_rel": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json",
            "frozen_policy_sha256": h_policy,
            "freeze_base_commit_sha": head,
            "s3_in_scope": True,
            "s3_parity_tolerance_frozen_before_metrics": True,
            "next_allowed_action": (
                "Run Gate A R0/R1 reproduction under frozen policy; "
                "do not open S3 until Gate A PASS."
            ),
            "opened": ["policy_freeze"],
            "not_opened": ["GateA", "S1", "S2", "S3", "S4", "S5", "K0-K4", "confirmation", "test", "D4", "demo"],
        }
    )
    h_status = write_json(status_path, status, overwrite=True)

    meta = {
        "schema": "e4_syskd_v1_policy_freeze_meta_v1",
        "created_at": now,
        "campaign": "E4_SYSKD_V1",
        "freeze_base_commit_sha": head,
        "policy_rel": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json",
        "policy_sha256": h_policy,
        "draft_sha256_after_supersede": h_draft,
        "status_sha256": h_status,
        "observed_S3_diff_at_freeze": False,
        "tolerance_approved_before_S3_metric": True,
        "roi_regions_order_canonical_code": code_roi_order,
        "note": "policy_sha256 is content hash of frozen POLICY.json; no self-SHA inside POLICY body.",
    }
    h_meta = write_json(meta_path, meta, overwrite=False)

    entry = {
        "ts": now,
        "event": "policy_freeze",
        "status": "POLICY_FROZEN_AWAITING_GATE_A",
        "decision": "APPROVED_FREEZE_WITH_S3_ROI_BATCHING",
        "freeze_base_commit_sha": head,
        "artifacts": {
            "policy": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json",
            "policy_sha256": h_policy,
            "draft": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY_DRAFT.json",
            "draft_sha256": h_draft,
            "status": "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_CAMPAIGN_STATUS.json",
            "status_sha256": h_status,
            "meta": "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY_FREEZE_META.json",
            "meta_sha256": h_meta,
        },
        "s3_formal_logit_gates": {"max_abs": 1e-4, "mean_abs": 1e-5, "top1_agree": 1.0, "top5_set_agree": 1.0},
        "source_gaps_remaining": [],
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Verify round-trip
    loaded = json.loads(policy_path.read_text(encoding="utf-8"))
    assert loaded["status"] == "FROZEN"
    assert loaded["immutable"] is True
    assert loaded["s3_roi_batching"]["formal_parity_gates"]["pre_softmax_logits_max_abs_diff"] == 1e-4
    assert loaded["s3_roi_batching"]["formal_parity_gates"]["pre_softmax_logits_mean_abs_diff"] == 1e-5
    assert loaded["s3_roi_batching"]["roi_regions_order_canonical_code"] == code_roi_order
    assert loaded["source_gaps_blocking_freeze"] == []
    assert sha256_file(policy_path) == h_policy

    print(
        json.dumps(
            {
                "status": "POLICY_FROZEN",
                "policy_sha256": h_policy,
                "draft_sha256": h_draft,
                "status_sha256": h_status,
                "meta_sha256": h_meta,
                "roi_order": code_roi_order,
                "selected": None,
                "e4_ready": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
