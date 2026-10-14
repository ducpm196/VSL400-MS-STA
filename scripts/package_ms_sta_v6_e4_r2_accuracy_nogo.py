#!/usr/bin/env python3
"""Package E4-R2 ACCURACY_NO_GO closeout + campaign STATUS (create-only scientific JSON)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

from src.eval.ms_sta_v5_m0_strong_baseline import load_json  # noqa: E402
from src.training.ms_sta_e4_r2_status import (  # noqa: E402
    build_campaign_status,
    write_campaign_status,
)
from src.training.retrain_v4 import executable_tree_sha256  # noqa: E402


def atomic_write_create_only(path: Path, obj: dict[str, Any]) -> str:
    if path.is_file():
        raise RuntimeError(f"refuse_overwrite:{path}")
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    out = _ROOT / "Docs/results/ms_sta_v6_e4_r2"
    gate = load_json(out / "ACCURACY_GATE.json")
    if gate.get("verdict") != "E4_R2_ACCURACY_NO_GO":
        raise SystemExit(f"unexpected_verdict:{gate.get('verdict')}")
    nogo = load_json(out / "E4_R2_ACCURACY_NO_GO.json")
    report = load_json(out / "CONFIRM_VALIDATION_REPORT.json")
    companion_path = out / "E4_R2_NO_GO_COMPANION_INVALID_COMPARATOR.json"
    companion = load_json(companion_path)

    closeout = {
        "schema": "ms_sta_v6_e4_r2_closeout_remediated_v1",
        "campaign": "E4-R2",
        "status": "CLOSED_ACCURACY_NO_GO",
        "canonical": True,
        "closed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": "research/ms-sta-v6-e4-r2",
        "verdict": "E4_R2_ACCURACY_NO_GO",
        "e4_ready": False,
        "keep_b4_as_primary": True,
        "forbid_efficiency_benchmark": True,
        "forbid_oneshot_test": True,
        "forbid_d4": True,
        "forbid_registry_promotion": True,
        "accuracy_gate_rel": "Docs/results/ms_sta_v6_e4_r2/ACCURACY_GATE.json",
        "accuracy_no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_ACCURACY_NO_GO.json",
        "confirm_validation_report_rel": "Docs/results/ms_sta_v6_e4_r2/CONFIRM_VALIDATION_REPORT.json",
        "confirm_completion_rel": "Docs/results/ms_sta_v6_e4_r2/CONFIRM_COMPLETION.json",
        "evidence_index_rel": "Docs/results/ms_sta_v6_e4_r2/EVIDENCE_INDEX.json",
        "hpo_audit_remediated_rel": "Docs/results/ms_sta_v6_e4_r2/HPO_AUDIT_REMEDIATED.json",
        "hpo_selected_remediated_rel": "Docs/results/ms_sta_v6_e4_r2/HPO_SELECTED_REMEDIATED.json",
        "selected_config_id": gate.get("selected_config_id"),
        "mean_delta_top1_pp": gate.get("mean_delta_top1_pp"),
        "hierarchical_ci95_top1_pp": gate.get("hierarchical_ci95_top1_pp"),
        "checks": gate.get("checks"),
        "historical_invalid_comparator": {
            "no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_NO_GO.json",
            "closeout_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_CLOSEOUT.json",
            "companion_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_NO_GO_COMPANION_INVALID_COMPARATOR.json",
            "note": "Preserved; not deleted; scientifically superseded by remediation + confirm gate.",
        },
        "phases": {
            "phase0_baseline": "PASS",
            "phase1_profiler": "PASS",
            "phase2_pilots": "PASS_SCREENING_ONLY",
            "phase3_4_hpo_wave1": "COMPLETE_REMEDIATED_SELECT",
            "confirm": "COMPLETE_3_OF_3",
            "accuracy_gate": "FAIL_E4_R2_ACCURACY_NO_GO",
            "efficiency": "NOT_OPENED",
            "oneshot_test": "NOT_OPENED",
            "d4": "NOT_OPENED",
        },
        "primary_model_unchanged": {
            "variant": "b4_v6_ctr",
            "checkpoint_sha256": "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1",
            "active_registry_rel": "data/processed/ms_sta_registry_v6_defense_go.json",
        },
        "claim_limits": [
            "E4-R2 failed frozen accuracy non-inferiority vs B4-V6 confirm.",
            "Canonical efficiency / oneshot / D4 were not opened.",
            "Pilot +84/+94% and HPO microbench remain screening-only.",
            "B4-V6 remains the primary defense model; e4_ready=false.",
        ],
        "git_revision": report.get("git_revision"),
        "executable_tree_sha256": report.get("executable_tree_sha256"),
    }
    c_sha = atomic_write_create_only(out / "E4_R2_CLOSEOUT_ACCURACY_NO_GO.json", closeout)

    # Companion: mark superseded by remediation audit (do not delete).
    companion_upd = dict(companion)
    companion_upd["remediation_required"] = False
    companion_upd["superseded_by"] = {
        "status": "E4_R2_ACCURACY_NO_GO",
        "accuracy_no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_ACCURACY_NO_GO.json",
        "closeout_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_CLOSEOUT_ACCURACY_NO_GO.json",
        "hpo_audit_remediated_rel": "Docs/results/ms_sta_v6_e4_r2/HPO_AUDIT_REMEDIATED.json",
        "note": "Invalid-comparator packaging retained as incident evidence.",
    }
    companion_upd["recorded_supersession_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    companion_path.write_text(
        json.dumps(companion_upd, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lock = load_json(out / "E4_R2_LOCK.json")
    lock["status"] = "E4_R2_ACCURACY_NO_GO"
    lock["scientific_status"] = "CONFIRM_COMPLETE — ACCURACY_NO_GO"
    lock["e4_ready"] = False
    lock["accuracy_closeout_rel"] = "Docs/results/ms_sta_v6_e4_r2/E4_R2_CLOSEOUT_ACCURACY_NO_GO.json"
    lock["accuracy_gate_rel"] = "Docs/results/ms_sta_v6_e4_r2/ACCURACY_GATE.json"
    lock["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.joinpath("E4_R2_LOCK.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    runs_root = _ROOT / "Docs/results/ms_sta_v6_e4_r2_runs_canonical"
    prev = load_json(runs_root / "STATUS.json") if (runs_root / "STATUS.json").is_file() else None
    confirm = (
        load_json(runs_root / "confirm" / "STATUS.json")
        if (runs_root / "confirm" / "STATUS.json").is_file()
        else {"status": "CONFIRM_COMPLETE"}
    )
    tree = executable_tree_sha256(_ROOT)
    shas = write_campaign_status(
        runs_root=runs_root,
        sealed_rel_root=out,
        payload=build_campaign_status(
            status="E4_R2_ACCURACY_NO_GO",
            phase="accuracy_closeout",
            git_revision=str(report.get("git_revision") or ""),
            executable_tree_sha256=str(report.get("executable_tree_sha256") or tree),
            prev=prev,
            extra={
                "accuracy_gate_rel": "Docs/results/ms_sta_v6_e4_r2/ACCURACY_GATE.json",
                "accuracy_no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_ACCURACY_NO_GO.json",
                "closeout_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_CLOSEOUT_ACCURACY_NO_GO.json",
                "forbid_efficiency_benchmark": True,
                "nogo_sha256_hint": nogo.get("status"),
                "packaging_closeout_sha256": c_sha,
                "note": "Supersedes stale E4_R2_HPO_NO_SELECT; historical packaging preserved.",
            },
        ),
        confirm=confirm,
    )
    print(
        json.dumps(
            {
                "event": "accuracy_nogo_packaged",
                "closeout_sha256": c_sha,
                "status_shas": shas,
                "executable_tree_sha256_now": tree,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
