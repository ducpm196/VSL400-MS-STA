# -*- coding: utf-8 -*-
"""Create S5 definition + schedule locks BEFORE any S5 metrics (create-only)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "Docs/results/e4_system_kd_v1"
POL = ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"
BINDING = ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json"
RGB = ROOT / "src/models/rgb_token_v3.py"
SCHED = ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json"

EXPECTED_POLICY = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
EXPECTED_BINDING = "ed86157ddcbcb0dff8d1d454f9d572a409b9b5f50c4169b87fc16b34938a69eb"
EXPECTED_CKPT = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
ROI_ORDER = ["global", "left_hand", "right_hand", "face"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_lf(path: Path, obj: dict) -> str:
    if path.exists():
        raise SystemExit(f"exists:{path}")
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pol_sha = sha(POL)
    bind_sha = sha(BINDING)
    rgb_sha = sha(RGB)
    if pol_sha != EXPECTED_POLICY:
        raise SystemExit(f"policy_sha:{pol_sha}")
    if bind_sha != EXPECTED_BINDING:
        raise SystemExit(f"binding_sha:{bind_sha}")
    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    ids = list(sched["sample_ids"])
    if len(ids) != 24:
        raise SystemExit("n_ids")

    definition = {
        "schema": "e4_syskd_v1_s5_definition_lock_v1",
        "status": "FROZEN_BEFORE_S5_METRICS",
        "created_at": now,
        "source_commit_sha": head,
        "s5_definition": {
            "formula": "S5 = R1 + S3_ROI_batching",
            "includes": ["Q2_AMP_SlowFast_FP16_ROI_FP32", "batch_roi_regions=True"],
            "excludes": ["S1_pinned_transfer", "S2_decode_overlap", "S4_cuda_streams", "any_new_optimization"],
            "not_combined_multi_arm": True,
            "label": "formal_system_candidate_S3_only",
        },
        "comparators": {
            "parity_and_accuracy": "R1",
            "formal_promotion_efficiency": "R0",
            "causal_intermediate": "R1",
        },
        "roi_regions_order_canonical_code": ROI_ORDER,
        "implementation": {
            "file": "src/models/rgb_token_v3.py",
            "symbols": ["_roi_align_tokens_batched", "extract_rgb_token_v3_for_video.batch_roi_regions"],
            "file_sha256": rgb_sha,
            "s3_feat_commit": "4cb4583",
            "flag": "batch_roi_regions=True",
            "forbid_change_after_lock": True,
        },
        "hashes": {
            "policy_sha256": pol_sha,
            "binding_sha256": bind_sha,
            "checkpoint_sha256": EXPECTED_CKPT,
            "rgb_token_v3_sha256": rgb_sha,
        },
        "accuracy_gate_from_policy": {
            "paired_vs": "R1",
            "eval_role": "val",
            "n_val": 2629,
            "n_boot": 10000,
            "bootstrap_seed": 42,
            "hierarchical_ci95_lower_top1_pp_min": -0.3,
            "delta_macro_f1_pp_min": -0.5,
            "delta_top5_pp_min": -0.3,
            "per_signer_delta_top1_pp_min": -1.0,
            "forbid_nan_inf": True,
            "parity_logit_max_abs": 1e-4,
            "parity_logit_mean_abs": 1e-5,
            "top1_agreement": 1.0,
            "top5_label_set_agreement": 1.0,
        },
        "efficiency_gate_from_policy": {
            "e2e_median_speedup_min_vs_R0": 1.15,
            "e2e_speedup_ci95_lower_min": 1.08,
            "forbid_p90_regression": True,
            "forbid_vram_increase_vs_r0": True,
            "batch": 1,
        },
        "selected": None,
        "e4_ready": False,
        "note": "Locked before any S5 formal metrics.",
    }
    h_def = write_lf(ART / "E4_SYSKD_V1_S5_DEFINITION_LOCK.json", definition)

    schedule = {
        "schema": "e4_syskd_v1_s5_schedule_lock_v1",
        "status": "FROZEN_BEFORE_S5_METRICS",
        "created_at": now,
        "definition_lock_sha256": h_def,
        "source_commit_sha": head,
        "sample_ids": ids,
        "n_samples": 24,
        "measured_rounds": 5,
        "warmup_rounds_per_sample_per_arm": 1,
        "arms": ["R0", "R1", "S5"],
        "expected_efficiency_rows": 360,
        "triple_arm_order_by_round": {
            "1": ["R0", "R1", "S5"],
            "2": ["R1", "S5", "R0"],
            "3": ["S5", "R0", "R1"],
            "4": ["R0", "S5", "R1"],
            "5": ["R1", "R0", "S5"],
        },
        "validation": {
            "role": "val",
            "n": 2629,
            "split_key": "val_ids",
            "arms": ["R1", "S5"],
            "forbid_test": True,
        },
        "invalid_run_rules": {
            "keep_raw_row_on_error": True,
            "forbid_post_hoc_outlier_drop": True,
            "coverage_required_efficiency": "24x5x3",
            "coverage_required_validation": "2629x2",
        },
        "synchronization": {
            "cuda_synchronize_before_after_gpu": True,
            "e2e_wall_clock": True,
        },
        "timeout_sec_overall": 28800,
        "selected": None,
        "e4_ready": False,
    }
    h_sch = write_lf(ART / "E4_SYSKD_V1_S5_SCHEDULE_LOCK.json", schedule)
    print(json.dumps({"definition_sha": h_def, "schedule_sha": h_sch, "head": head}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
