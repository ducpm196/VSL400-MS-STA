# -*- coding: utf-8 -*-
"""Create E4-R4C post-HPO research redirect (create-only) + campaign status pointer."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_POST_HPO_RESEARCH_REDIRECT.json"
STATUS = ROOT / "Docs/results/ms_sta_v6_e4_r4/CAMPAIGN_STATUS.json"
CONFIRM_NOTE = ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/confirm/seed42/INCOMPLETE_USER_STOPPED.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, obj: dict) -> None:
    if path.is_file():
        raise SystemExit(f"refuse_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"refuse_overwrite:{OUT}")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    artifacts = {
        "HPO_INDEX.json": "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/HPO_INDEX.json",
        "HPO_SELECTION.json": "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/HPO_SELECTION.json",
        "E4_R4C_HPO_BOUNDARY_AUDIT.json": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_BOUNDARY_AUDIT.json",
        "E4_R4C_HPO_SELECTION_ADJUDICATION.json": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_SELECTION_ADJUDICATION.json",
        "E4_R4C_TEACHER_LOGITS_PARITY.json": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_TEACHER_LOGITS_PARITY.json",
        "E4_R4C_PERFORMANCE_CODE_LOCK.json": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_PERFORMANCE_CODE_LOCK.json",
        "E4_R4B_STAGE_BREAKDOWN_REPORT.json": "Docs/results/ms_sta_v6_e4_r4/E4_R4B_STAGE_BREAKDOWN_REPORT.json",
        "b4_v6_ctr_seed42_best.pt": "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
        "long_task_b4q": "Docs/CURSOR_LONG_TASK_E4_B4_PRESERVING_COMPRESSION.md",
    }
    hashes = {k: sha256_file(ROOT / rel) for k, rel in artifacts.items()}

    stage = json.loads((ROOT / artifacts["E4_R4B_STAGE_BREAKDOWN_REPORT.json"]).read_text(encoding="utf-8"))
    b4 = stage["baseline_b4_hot"]
    stages = b4["stage_medians_ms"]
    total = float(b4["total_median_ms"])
    head_ms = float(stages["model_forward"])
    rgb_ms = float(stages["rgb_representation"])
    decode_ms = float(stages["video_decode"])

    redirect = {
        "schema": "ms_sta_v6_e4_r4c_post_hpo_research_redirect_v1",
        "status": "STOPPED_POST_HPO_USER_REDIRECT_NO_PROMOTION",
        "created_at": now,
        "git_head": head,
        "git_branch": branch,
        "campaign_closed": "E4-R4C",
        "e4_ready": False,
        "b4_source_of_truth": True,
        "b4_variant": "b4_v6_ctr",
        "b4_checkpoint_sha256": hashes["b4_v6_ctr_seed42_best.pt"],
        "hpo": {
            "complete": True,
            "n_registered": 8,
            "n_done": 8,
            "hpo_index_sha256": hashes["HPO_INDEX.json"],
            "hpo_selection_sha256": hashes["HPO_SELECTION.json"],
            "boundary_audit_sha256": hashes["E4_R4C_HPO_BOUNDARY_AUDIT.json"],
            "adjudication_sha256": hashes["E4_R4C_HPO_SELECTION_ADJUDICATION.json"],
        },
        "gates": {
            "confirm": "NOT_OPENED_INCOMPLETE_USER_STOPPED",
            "accuracy": "NOT_OPENED",
            "efficiency": "NOT_OPENED",
            "final_test": "NOT_OPENED",
            "D4": "NOT_OPENED",
            "demo_cutover": "NOT_OPENED",
            "registry_promotion": "NOT_OPENED",
        },
        "confirm_partial_evidence": {
            "seed42_dir": "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/confirm/seed42",
            "had_best_pt_only": True,
            "summary_json": False,
            "history_json": False,
            "val_predictions_json": False,
            "live_log_rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_CONFIRM_LIVE.log",
            "note": "Confirm seed42 killed mid-run after user redirect; incomplete artifacts must not be used for accuracy claims.",
        },
        "claims": {
            "accuracy_failure": False,
            "accuracy_success": False,
            "efficiency_failure": False,
            "efficiency_success": False,
            "promotion": False,
            "reason": "Confirm/accuracy/efficiency gates never completed; R4C closed by user scientific redirect, not by gate verdict.",
        },
        "redirect_reason": {
            "primary": "Representation/model compression (R4C) reduces B4 invariants and is no longer aligned with thesis defense goals.",
            "bottleneck_evidence_source": "Docs/results/ms_sta_v6_e4_r4/E4_R4B_STAGE_BREAKDOWN_REPORT.json",
            "bottleneck_evidence_sha256": hashes["E4_R4B_STAGE_BREAKDOWN_REPORT.json"],
            "b4_total_median_ms": total,
            "stage_medians_ms": stages,
            "amdahl_head_only_max_e2e_fraction": head_ms / total,
            "amdahl_rgb_slowfast_roi_fraction": rgb_ms / total,
            "amdahl_decode_fraction": decode_ms / total,
            "interpretation": "Quantizing B4 head alone cannot yield meaningful end-to-end speedup; bottleneck is SlowFast/ROI then decode.",
        },
        "user_decision_provenance": {
            "decision": "Redirect E4 to B4-preserving deployment optimization (B4Q-V1)",
            "long_task_rel": "Docs/CURSOR_LONG_TASK_E4_B4_PRESERVING_COMPRESSION.md",
            "long_task_sha256": hashes["long_task_b4q"],
            "expected_sha256_user_claimed": "12B7A6AA17C6040585214578ADB03CFA27D23D94F3214EBD5350CCCE40A1A475",
            "sha256_match": hashes["long_task_b4q"].upper()
            == "12B7A6AA17C6040585214578ADB03CFA27D23D94F3214EBD5350CCCE40A1A475",
        },
        "next_campaign": {
            "id": "B4Q-V1",
            "presentation_name": "E4 — B4-preserving deployment optimization",
            "policy_rel": "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json",
            "status_rel": "Docs/results/ms_sta_v6_b4q/B4Q_V1_CAMPAIGN_STATUS.json",
            "branch_intended": "research/ms-sta-v6-b4q-preserving",
        },
        "artifact_sha256": hashes,
        "forbid_use_r4c_as_b4q_baseline": True,
        "forbid_promote_partial_confirm": True,
    }
    atomic_write(OUT, redirect)

    if not CONFIRM_NOTE.is_file():
        atomic_write(
            CONFIRM_NOTE,
            {
                "schema": "ms_sta_v6_e4_r4c_confirm_incomplete_v1",
                "status": "INCOMPLETE_USER_STOPPED",
                "seed": 42,
                "created_at": now,
                "git_head": head,
                "files_present": ["best.pt"],
                "files_missing": ["summary.json", "history.json", "val_predictions.json"],
                "do_not_use_for_accuracy_gate": True,
                "redirect_rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_POST_HPO_RESEARCH_REDIRECT.json",
            },
        )

    # Atomic campaign status with history (overwrite status pointer; keep prior body in history).
    prior = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    history = list(prior.get("history") or [])
    if prior:
        history.append({k: v for k, v in prior.items() if k != "history"})
    new_status = {
        "schema": "ms_sta_v6_e4_r4_campaign_status_v2",
        "campaign": "E4-R4C",
        "branch": branch,
        "phase": "post_hpo_research_redirect",
        "status": "STOPPED_POST_HPO_USER_REDIRECT_NO_PROMOTION",
        "e4_ready": False,
        "b4_frozen": True,
        "final_test": False,
        "hpo_complete_8_of_8": True,
        "confirm": "NOT_OPENED_INCOMPLETE_USER_STOPPED",
        "accuracy_gate": "NOT_OPENED",
        "efficiency_gate": "NOT_OPENED",
        "redirect_rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_POST_HPO_RESEARCH_REDIRECT.json",
        "redirect_sha256": sha256_file(OUT),
        "next_campaign": "B4Q-V1",
        "updated_at": now,
        "git_head": head,
        "history": history,
    }
    # Status pointer may update (not create-only); write atomically via temp+replace.
    raw = (json.dumps(new_status, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = STATUS.with_suffix(STATUS.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(STATUS)

    print("REDIRECT_OK", OUT)
    print("STATUS_OK", STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
