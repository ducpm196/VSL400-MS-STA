#!/usr/bin/env python3
"""Package E4-R3 feasibility NO-GO closeout artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.training.retrain_v4 import executable_tree_sha256  # noqa: E402


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic(path: Path, obj: dict) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    out = _ROOT / "Docs/results/ms_sta_v6_e4_r3"
    feas = out / "E4_R3_FEASIBILITY.json"
    nogo = out / "E4_R3_FEASIBILITY_NO_GO.json"
    pre = out / "E4_R3_PREFLIGHT.json"
    pol = _ROOT / "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_POLICY.json"
    train_pol = _ROOT / "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_TRAIN_POLICY.json"
    feas_obj = json.loads(feas.read_text(encoding="utf-8"))

    git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_ROOT, text=True
    ).strip()
    tree = executable_tree_sha256(_ROOT)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    speedups = {k: v["speedup_vs_b4"] for k, v in feas_obj["arms"].items()}
    best_arm = max(speedups, key=speedups.get)

    closeout = {
        "schema": "ms_sta_v6_e4_r3_closeout_feasibility_no_go_v1",
        "campaign": "E4-R3",
        "status": "CLOSED_FEASIBILITY_NO_GO",
        "canonical": True,
        "closed_at": now,
        "branch": branch,
        "verdict": "E4_R3_FEASIBILITY_NO_GO",
        "e4_ready": False,
        "keep_b4_as_primary": True,
        "forbid_hpo": True,
        "forbid_confirm": True,
        "forbid_efficiency_benchmark": True,
        "forbid_oneshot_test": True,
        "forbid_d4": True,
        "forbid_registry_promotion": True,
        "hypothesis": "true_rgb_and_skeleton_token_reduction_for_measured_latency",
        "feasibility_gate": {
            "speedup_min_required": feas_obj["speedup_min_required"],
            "b4_median_ms": feas_obj["b4"]["median_sec"] * 1000.0,
            "speedups": speedups,
            "best_arm": best_arm,
            "best_speedup": speedups[best_arm],
            "promoted_arms": [],
        },
        "interpretation": [
            "Arm A (RGB compact) was fastest among E4-R3 arms but only ~+1.9% vs B4 median — below the locked 8% feasibility floor.",
            "Arms B/C (skeleton compact / full) were slower than B4 on batch-1 FP32 model-forward despite lower parameter counts.",
            "Parameter reduction alone did not yield the required model-forward latency gain under the locked microbench protocol.",
            "Grayscale replicated-3ch was not used (forbidden; would not reduce backbone FLOPs).",
        ],
        "phases": {
            "phase0_preflight_policy": "PASS",
            "phase1_architecture": "IMPLEMENTED_A_B_C",
            "phase2_contract_tests": "PASS",
            "phase2_feasibility": "FAIL_E4_R3_FEASIBILITY_NO_GO",
            "hpo": "NOT_OPENED",
            "confirm": "NOT_OPENED",
            "accuracy_gate": "NOT_OPENED",
            "efficiency": "NOT_OPENED",
            "oneshot_test": "NOT_OPENED",
            "d4": "NOT_OPENED",
        },
        "artifacts": {
            "preflight_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_PREFLIGHT.json",
            "policy_rel": "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_POLICY.json",
            "feasibility_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json",
            "feasibility_no_go_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json",
        },
        "primary_model_unchanged": {
            "variant": "b4_v6_ctr",
            "checkpoint_sha256": "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1",
            "active_registry_rel": "data/processed/ms_sta_registry_v6_defense_go.json",
            "e4_ready": False,
        },
        "claim_limits": [
            "E4-R3 closed at feasibility: no arm met ≥8% model-forward median speedup vs B4.",
            "HPO / confirm / accuracy / efficiency / oneshot / D4 were not opened.",
            "Feasibility microbench is screening-only; not a canonical efficiency claim.",
            "B4-V6 remains the primary defense model; e4_ready=false.",
            "Do not claim real-time, production, or end-to-end from model-only timing.",
        ],
        "git_revision_at_closeout_start": git_rev,
        "executable_tree_sha256_dirty_ok": tree,
        "starting_tip_required": "40fc707e9ad358ed47e9e095be6cea3f6085ddf3",
    }
    close_sha = atomic(out / "E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json", closeout)

    evidence = {
        "schema": "ms_sta_v6_e4_r3_evidence_index_v1",
        "campaign": "E4-R3",
        "status": "CLOSED_FEASIBILITY_NO_GO",
        "entries": [
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_PREFLIGHT.json",
                "sha256": sha_file(pre),
                "role": "phase0_preflight",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_POLICY.json",
                "sha256": sha_file(pol),
                "role": "immutable_policy",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_TRAIN_POLICY.json",
                "sha256": sha_file(train_pol),
                "role": "train_policy_draft_pre_cancel",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json",
                "sha256": sha_file(feas),
                "role": "feasibility_microbench",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json",
                "sha256": sha_file(nogo),
                "role": "feasibility_no_go",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json",
                "sha256": close_sha,
                "role": "closeout",
            },
        ],
    }
    atomic(out / "EVIDENCE_INDEX.json", evidence)

    status = {
        "schema": "ms_sta_v6_e4_r3_campaign_status_v1",
        "campaign": "E4-R3",
        "status": "CLOSED_FEASIBILITY_NO_GO",
        "e4_ready": False,
        "halt": True,
        "verdict": "E4_R3_FEASIBILITY_NO_GO",
        "closeout_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json",
        "evidence_index_rel": "Docs/results/ms_sta_v6_e4_r3/EVIDENCE_INDEX.json",
        "next": "none_campaign_closed",
        "b4_remains_primary": True,
    }
    atomic(out / "CAMPAIGN_STATUS.json", status)

    tp = json.loads(train_pol.read_text(encoding="utf-8"))
    tp["status"] = "CANCELLED_FEASIBILITY_NO_GO"
    tp["train_allowed"] = False
    tp["cancelled_reason"] = "E4_R3_FEASIBILITY_NO_GO"
    # refresh train policy sha in evidence after cancel — rewrite evidence entry after
    atomic(train_pol, tp)
    evidence["entries"][2]["sha256"] = sha_file(train_pol)
    evidence["entries"][2]["role"] = "train_policy_cancelled"
    atomic(out / "EVIDENCE_INDEX.json", evidence)

    print(
        json.dumps(
            {"closeout_sha": close_sha, "best_arm": best_arm, "speedups": speedups},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
