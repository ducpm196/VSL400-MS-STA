"""Versioned F0 / F0-R1 artifact path bundles (no overwrites across campaigns)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CampaignPaths:
    name: str
    results_dir_rel: str
    weights_manifest_rel: str
    conversion_evidence_rel: str
    realization_manifest_rel: str
    realization_manifest_audit_rel: str
    execution_approval_rel: str
    execution_lock_rel: str
    execution_audit_rel: str
    execution_seal_rel: str
    execution_attempt_rel: str
    feasibility_ready_rel: str
    feasibility_report_rel: str
    feasibility_completion_rel: str
    seal_consumption_rel: str
    feasibility_staging_dir_rel: str
    final_audit_rel: str
    gpu_tmp_rel: str
    artifact_prefix: str


F0_PATHS = CampaignPaths(
    name="f0",
    results_dir_rel="Docs/results/ms_sta_v5_f0",
    weights_manifest_rel="Docs/results/ms_sta_v5_f0/F0_WEIGHTS_MANIFEST.json",
    conversion_evidence_rel="Docs/results/ms_sta_v5_f0/F0_VIDEOMAE_SMALL_CONVERSION_EVIDENCE.json",
    realization_manifest_rel="Docs/results/ms_sta_v5_f0/F0_REALIZATION_MANIFEST.json",
    realization_manifest_audit_rel="Docs/results/ms_sta_v5_f0/F0_REALIZATION_MANIFEST_AUDIT.json",
    execution_approval_rel="Docs/results/ms_sta_v5_f0/F0_EXECUTION_APPROVAL.json",
    execution_lock_rel="Docs/results/ms_sta_v5_f0/F0_EXECUTION_LOCK.json",
    execution_audit_rel="Docs/results/ms_sta_v5_f0/F0_EXECUTION_AUDIT.json",
    execution_seal_rel="Docs/results/ms_sta_v5_f0/F0_EXECUTION_SEAL.json",
    execution_attempt_rel="Docs/results/ms_sta_v5_f0/F0_EXECUTION_ATTEMPT.json",
    feasibility_ready_rel="Docs/results/ms_sta_v5_f0/F0_FEASIBILITY_READY.json",
    feasibility_report_rel="Docs/results/ms_sta_v5_f0/F0_FEASIBILITY_REPORT.json",
    feasibility_completion_rel="Docs/results/ms_sta_v5_f0/F0_FEASIBILITY_COMPLETION.json",
    seal_consumption_rel="Docs/results/ms_sta_v5_f0/F0_EXECUTION_SEAL_CONSUMPTION.json",
    feasibility_staging_dir_rel="Docs/results/ms_sta_v5_f0/.f0_feasibility_staging",
    final_audit_rel="Docs/results/ms_sta_v5_f0/F0_FINAL_AUDIT.json",
    gpu_tmp_rel="Docs/results/ms_sta_v5_f0/evidence/gpu_feasibility_tmp",
    artifact_prefix="F0",
)

F0_R1_PATHS = CampaignPaths(
    name="f0_r1",
    results_dir_rel="Docs/results/ms_sta_v5_f0_r1",
    weights_manifest_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_WEIGHTS_MANIFEST.json",
    conversion_evidence_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_VIDEOMAE_SMALL_CONVERSION_EVIDENCE.json",
    realization_manifest_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_REALIZATION_MANIFEST.json",
    realization_manifest_audit_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_REALIZATION_MANIFEST_AUDIT.json",
    execution_approval_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_EXECUTION_APPROVAL.json",
    execution_lock_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_EXECUTION_LOCK.json",
    execution_audit_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_EXECUTION_AUDIT.json",
    execution_seal_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_EXECUTION_SEAL.json",
    execution_attempt_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_EXECUTION_ATTEMPT.json",
    feasibility_ready_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_FEASIBILITY_READY.json",
    feasibility_report_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_FEASIBILITY_REPORT.json",
    feasibility_completion_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_FEASIBILITY_COMPLETION.json",
    seal_consumption_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_EXECUTION_SEAL_CONSUMPTION.json",
    feasibility_staging_dir_rel="Docs/results/ms_sta_v5_f0_r1/.f0_r1_feasibility_staging",
    final_audit_rel="Docs/results/ms_sta_v5_f0_r1/F0_R1_FINAL_AUDIT.json",
    gpu_tmp_rel="Docs/results/ms_sta_v5_f0_r1/evidence/gpu_feasibility_tmp",
    artifact_prefix="F0_R1",
)


def campaign_paths(name: str) -> CampaignPaths:
    key = str(name or "f0").strip().lower().replace("-", "_")
    if key in {"f0", "original"}:
        return F0_PATHS
    if key in {"f0_r1", "r1"}:
        return F0_R1_PATHS
    raise ValueError(f"unknown_campaign:{name}")
