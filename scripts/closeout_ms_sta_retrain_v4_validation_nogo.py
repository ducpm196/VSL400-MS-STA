#!/usr/bin/env python3
"""Close retrain-v4 as VALIDATION_NO_GO — version evidence + cut over active registry.

Does not overwrite logs/ms_sta_retrain_v4/CONFIRM_VALIDATION_REPORT.json.
Does not create EVALUATION_LOCKED / test / v4e artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LOG = _ROOT / "logs" / "ms_sta_retrain_v4"
_OUT = _ROOT / "Docs" / "results" / "ms_sta_retrain_v4"
_POLICY = _ROOT / "Docs" / "results" / "ms_sta_retrain_v4_policy" / "RETRAIN_POLICY.json"
_CKPT = _ROOT / "data" / "processed" / "ms_sta_retrain_v4"
_ACTIVE = _ROOT / "data" / "processed" / "ms_sta_registry_active.json"
_REG = _ROOT / "data" / "processed" / "ms_sta_registry_v4_validation_no_go.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_rev() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True)
        .strip()
    )


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True)
    return bool(out.strip())


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return _sha(path)


def _copy_bitexact(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    s_sha, d_sha = _sha(src), _sha(dst)
    if s_sha != d_sha:
        raise SystemExit(f"[FAIL] bit-exact copy mismatch {src} -> {dst}")
    return d_sha


def _variant_means_from_status() -> dict[str, dict]:
    by: dict[str, list[tuple[int, float, float | None]]] = {}
    for p in sorted(_LOG.glob("confirm__*.json")):
        if "__final" in p.name:
            continue
        j = _load(p)
        hist = j.get("history") or []
        best = None
        for h in hist:
            v = h.get("val_top1")
            if v is None:
                continue
            if best is None or float(v) > float(best["val_top1"]):
                best = h
        parts = p.stem.split("__")
        variant = parts[1]
        seed = int([x for x in parts if x.startswith("seed")][0].replace("seed", ""))
        top1 = float(best["val_top1"]) if best else float(j["best_val_top1"])
        top5 = float(best["val_top5"]) if best and best.get("val_top5") is not None else None
        by.setdefault(variant, []).append((seed, top1, top5))
    out: dict[str, dict] = {}
    for variant, rows in by.items():
        t1 = [r[1] for r in rows]
        t5 = [r[2] for r in rows if r[2] is not None]
        out[variant] = {
            "n_seeds": len(rows),
            "top1_mean_pct": float(statistics.mean(t1)),
            "top1_sample_std_pct": float(statistics.stdev(t1)) if len(t1) > 1 else 0.0,
            "top5_mean_pct": float(statistics.mean(t5)) if t5 else None,
            "top5_sample_std_pct": float(statistics.stdev(t5)) if len(t5) > 1 else 0.0,
            "per_seed": [
                {"seed": s, "top1": t1v, "top5": t5v} for s, t1v, t5v in sorted(rows)
            ],
        }
    return out


def _final_checkpoint_table() -> dict[str, str]:
    table: dict[str, str] = {}
    finals = sorted(_CKPT.glob("confirm__*__final.pth"))
    if len(finals) != 12:
        raise SystemExit(f"[FAIL] expected 12 confirm finals, found {len(finals)}")
    for p in finals:
        rel = p.relative_to(_ROOT).as_posix()
        table[rel] = _sha(p)
    return table


def main() -> int:
    report_src = _LOG / "CONFIRM_VALIDATION_REPORT.json"
    manifest_src = _LOG / "per_sample_val" / "PER_SAMPLE_MANIFEST.json"
    run_state = _LOG / "RUN_STATE.json"
    fail_art = _LOG / "failures" / "confirm__b4__lr6e-4_wd0.05__seed123__stride1.json"
    for req in (report_src, manifest_src, run_state, _POLICY, fail_art):
        if not req.is_file():
            raise SystemExit(f"[FAIL] missing required artifact: {req}")

    # Hard refuse lock/test/v4e leakage into closeout.
    for banned in (
        _LOG / "EVALUATION_LOCKED.json",
        _OUT / "EVALUATION_LOCKED.json",
        _LOG / "TEST_REPORT.json",
        _ROOT / "Docs" / "results" / "ms_sta_retrain_v4e",
    ):
        if banned.exists():
            raise SystemExit(f"[FAIL] forbidden artifact present: {banned}")

    git_rev = _git_rev()
    dirty = _git_dirty()
    if dirty:
        raise SystemExit(
            "[FAIL] packager requires clean worktree "
            "(packager_preflight_git_dirty must be false). "
            "Commit tooling/docs first, then re-run closeout."
        )
    # Evaluator that produced the canonical report (do not regenerate report here).
    evaluator_git_revision = "f4e7a80107e9aba849a4809138c5ff6a93ba2319"
    packager_git_revision = git_rev

    report = _load(report_src)
    gates = report.get("architecture_gates") or {}
    if gates.get("architecture_gate_pass") is True:
        raise SystemExit("[FAIL] architecture_gate_pass=true; this closeout is NO-GO only")
    summary = report.get("summary") or {}

    now = datetime.now(timezone.utc).isoformat()
    policy_sha = _sha(_POLICY)
    report_sha = _sha(report_src)
    manifest_sha = _sha(manifest_src)
    split_sha = str((_load(run_state).get("split_manifest_sha256") or "")).strip()
    if not split_sha:
        raise SystemExit("[FAIL] RUN_STATE missing split_manifest_sha256")

    _OUT.mkdir(parents=True, exist_ok=True)
    report_dst_sha = _copy_bitexact(report_src, _OUT / "CONFIRM_VALIDATION_REPORT.json")
    if report_dst_sha != report_sha:
        raise SystemExit("[FAIL] report sha drift after copy")
    if report_sha != "149cfc5d6a0cd037998a08c65a12f0f93a0c5a9bcc20c9823eadd0e0a212a591":
        raise SystemExit(
            f"[FAIL] unexpected report sha {report_sha}; refuse silent regenerate"
        )

    # Manifest pointer + sha binding (rows stay under logs/; index is versioned).
    manifest_index = {
        "schema": "ms_sta_retrain_v4_per_sample_manifest_pointer_v1",
        "per_sample_manifest_rel": "logs/ms_sta_retrain_v4/per_sample_val/PER_SAMPLE_MANIFEST.json",
        "per_sample_manifest_sha256": manifest_sha,
        "n_val": int(report.get("n_samples") or 0),
        "copied_at": now,
        "note": "JSONL rows remain under logs/; this pointer is the versioned binding.",
    }
    manifest_index_sha = _write(_OUT / "PER_SAMPLE_MANIFEST_POINTER.json", manifest_index)

    variants = _variant_means_from_status()
    b3 = variants["b3"]
    b4r = variants["b4r"]
    metrics = {
        "schema": "ms_sta_retrain_v4_validation_metrics_v1",
        "split_role": "val",
        "val_n": int(report.get("n_samples") or 2629),
        "test_ids_touched": False,
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": git_rev,
        "evaluator_git_revision": evaluator_git_revision,
        "packager_git_revision": packager_git_revision,
        "packager_preflight_git_dirty": False,
        "evaluator_script": "scripts/evaluate_ms_sta_retrain_v4_validation.py",
        "confirm_validation_report_sha256": report_sha,
        "per_sample_manifest_sha256": manifest_sha,
        "created_at": now,
        "variants": variants,
        "canonical_reference": {
            "variant": "b3",
            "top1_mean_pct": b3["top1_mean_pct"],
            "top1_sample_std_pct": b3["top1_sample_std_pct"],
            "top5_mean_pct": b3["top5_mean_pct"],
            "top5_sample_std_pct": b3["top5_sample_std_pct"],
            "macro_f1_mean_pct": float(summary.get("macro_f1_b3")),
            "scope": "path_b_confirmation_validation_only",
        },
        "primary_proposal": {
            "variant": "b4r",
            "top1_mean_pct": b4r["top1_mean_pct"],
            "top1_sample_std_pct": b4r["top1_sample_std_pct"],
            "mean_delta_top1_pp": float(summary.get("mean_delta_top1_pp")),
            "architecture_gate_pass": False,
            "claim_language": gates.get("claim_language") or "improve_not_outperform",
        },
    }
    metrics_sha = _write(_OUT / "VALIDATION_METRICS.json", metrics)

    finals = _final_checkpoint_table()
    fail_sha = _sha(fail_art)
    nogo = {
        "schema": "ms_sta_retrain_v4_campaign_close_v1",
        "campaign_status": "VALIDATION_NO_GO",
        "architecture_passed": False,
        "defense_ready": False,
        "demo_allowed": False,
        "forbid_lock": True,
        "forbid_test": True,
        "forbid_v4e": True,
        "campaign_closed": True,
        "git_revision": git_rev,
        "evaluator_git_revision": evaluator_git_revision,
        "packager_git_revision": packager_git_revision,
        "packager_preflight_git_dirty": False,
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "confirm_validation_report_sha256": report_sha,
        "per_sample_manifest_sha256": manifest_sha,
        "validation_metrics_sha256": metrics_sha,
        "canonical_reference": {
            "variant": "b3",
            "top1_mean_pm_sample_std_pct": [
                b3["top1_mean_pct"],
                b3["top1_sample_std_pct"],
            ],
            "display": "64,49 +/- 0,58%",
            "scope": "path_b_confirmation_validation_only",
        },
        "b4r_summary": {
            "top1_mean_pm_sample_std_pct": [
                b4r["top1_mean_pct"],
                b4r["top1_sample_std_pct"],
            ],
            "mean_delta_top1_pp": float(summary.get("mean_delta_top1_pp")),
            "paired_ci95": {
                "low": float((report.get("paired_bootstrap") or {}).get("ci_low")),
                "high": float((report.get("paired_bootstrap") or {}).get("ci_high")),
            },
            "seed123_delta_top1_pp": float(
                (report.get("per_seed") or {}).get("123", {}).get("delta_top1_pp")
            ),
            "claim_language": "improve_not_outperform",
        },
        "final_checkpoint_sha256": finals,
        "resume_and_failure_evidence": {
            "resume_count": 1,
            "failure_artifact": (
                "logs/ms_sta_retrain_v4/failures/"
                "confirm__b4__lr6e-4_wd0.05__seed123__stride1.json"
            ),
            "failure_artifact_sha256": fail_sha,
            "windows_exit_code": 3221226505,
            "windows_status_hex": "0xC0000409",
            "resume_behavior": [
                "skipped_7_bitexact_complete_trials",
                "relaunched_incomplete_b4_seed123",
                "completed_remaining_5_trials",
                "did_not_overwrite_valid_checkpoints",
            ],
            "retain_failure_artifact": True,
        },
        "historical_not_primary": {
            "retrain_v3_b3_top1": "62,29 +/- 0,46%",
            "path": "Docs/results/ms_sta_retrain_v3/",
            "note": "v3 metrics are historical; live canonical reference is v4 B3.",
        },
        "closed_at": now,
    }
    nogo_sha = _write(_OUT / "VALIDATION_NO_GO.json", nogo)

    diagnosis = {
        "schema": "ms_sta_retrain_v4_diagnosis_b3_b4r_v1",
        "campaign_status": "VALIDATION_NO_GO",
        "created_at": now,
        "unit_of_analysis_primary": "seed_averaged_per_sample_correctness_n_2629",
        "seed_averaged_exclusive": {
            "n_videos": 2629,
            "b4r_better": 326,
            "b3_better": 303,
            "tie": 2000,
            "note": "Primary diagnostic unit. Do not present as 969 distinct videos.",
        },
        "seed_pair_exclusive_diagnostic_only": {
            "n_pairs": 7887,
            "b4r_only": 511,
            "b3_only": 458,
            "note": "Sum over 3 seeds x 2629 samples; diagnostic only, not primary unit.",
        },
        "residual_diagnostics_observed": {
            "residual_scale_end_of_train": [1.8209, 1.8303, 1.8361],
            "delta_over_base_ratio_end_of_train": [0.8682, 0.8883, 0.8340],
            "interpretation": "hypothesis_forming_only",
            "forbid_causal_claim": (
                "Do not conclude residual-too-strong or attention-underlearning as proven "
                "causes; these observations motivate registered v5 ablations only."
            ),
        },
        "literature_role": {
            "primary_precedent": ["ReZero", "LayerScale/CaiT"],
            "secondary_reference_only": [
                "AttnRes (2026 preprint)",
                "Review Residuals (2026 preprint)",
            ],
        },
    }
    diagnosis_sha = _write(_OUT / "DIAGNOSIS_B3_B4R.json", diagnosis)

    # CLOSEOUT_EVIDENCE must NOT embed its own SHA (unstable by construction).
    evidence = {
        "schema": "ms_sta_retrain_v4_closeout_evidence_v1",
        "campaign_status": "VALIDATION_NO_GO",
        "created_at": now,
        "evaluator_git_revision": evaluator_git_revision,
        "packager_git_revision": packager_git_revision,
        "packager_preflight_git_dirty": False,
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "confirm_validation_report_sha256": report_sha,
        "per_sample_manifest_sha256": manifest_sha,
        "validation_metrics_sha256": metrics_sha,
        "validation_no_go_sha256": nogo_sha,
        "diagnosis_sha256": diagnosis_sha,
        "per_sample_manifest_pointer_sha256": manifest_index_sha,
        "canonical_reference_variant": "b3",
        "canonical_reference_top1_mean_pct": b3["top1_mean_pct"],
        "forbid_defense_demo": True,
        "forbid_lock_test_v4e": True,
        "artifacts": {
            "Docs/results/ms_sta_retrain_v4/CONFIRM_VALIDATION_REPORT.json": report_sha,
            "Docs/results/ms_sta_retrain_v4/VALIDATION_METRICS.json": metrics_sha,
            "Docs/results/ms_sta_retrain_v4/VALIDATION_NO_GO.json": nogo_sha,
            "Docs/results/ms_sta_retrain_v4/DIAGNOSIS_B3_B4R.json": diagnosis_sha,
            "Docs/results/ms_sta_retrain_v4/PER_SAMPLE_MANIFEST_POINTER.json": manifest_index_sha,
            "logs/ms_sta_retrain_v4/per_sample_val/PER_SAMPLE_MANIFEST.json": manifest_sha,
            "logs/ms_sta_retrain_v4/failures/confirm__b4__lr6e-4_wd0.05__seed123__stride1.json": fail_sha,
        },
        "final_checkpoint_sha256": finals,
        "v5_not_started": True,
        "note": (
            "v4 closed VALIDATION_NO_GO. Live SoT = B3 v4 64.49+/-0.58%. "
            "B4R is a non-outperforming hypothesis. Do not train v5 or open test here. "
            "CLOSEOUT_EVIDENCE.json is bound externally via registry closeout_evidence_sha256."
        ),
    }
    evidence_sha = _write(_OUT / "CLOSEOUT_EVIDENCE.json", evidence)

    registry = {
        "schema": "ms_sta_registry_v4_validation_no_go_v1",
        "status": "VALIDATION_NO_GO",
        "campaign_status": "VALIDATION_NO_GO",
        "retrain_v4": True,
        "retrain_v3": False,
        "defense_ready": False,
        "demo_allowed": False,
        "e1_pass": False,
        "e1_selected": False,
        "e2_pass": False,
        "e2_unlocked": False,
        "evaluation_locked": False,
        "d4_pass": False,
        "architecture_passed": False,
        "active_variant": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "current_best_variant": "b3",
        "current_best_scope": "path_b_confirmation_validation_only",
        "current_best_metrics": {
            "top1_mean_pct": b3["top1_mean_pct"],
            "top1_sample_std_pct": b3["top1_sample_std_pct"],
            "top5_mean_pct": b3["top5_mean_pct"],
            "top5_sample_std_pct": b3["top5_sample_std_pct"],
            "macro_f1_mean_pct": float(summary.get("macro_f1_b3")),
            "display_top1": "64,49 +/- 0,58%",
            "source": "Docs/results/ms_sta_retrain_v4/VALIDATION_METRICS.json",
        },
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "confirm_validation_report_sha256": report_sha,
        "per_sample_manifest_sha256": manifest_sha,
        "validation_metrics_sha256": metrics_sha,
        "validation_no_go_sha256": nogo_sha,
        "closeout_evidence_sha256": evidence_sha,
        "evaluator_git_revision": evaluator_git_revision,
        "packager_git_revision": packager_git_revision,
        "packager_preflight_git_dirty": False,
        "git_revision": git_rev,
        "evidence": {
            "closeout_manifest": "Docs/results/ms_sta_retrain_v4/CLOSEOUT_EVIDENCE.json",
            "confirm_validation_report": "Docs/results/ms_sta_retrain_v4/CONFIRM_VALIDATION_REPORT.json",
            "validation_metrics": "Docs/results/ms_sta_retrain_v4/VALIDATION_METRICS.json",
            "validation_no_go": "Docs/results/ms_sta_retrain_v4/VALIDATION_NO_GO.json",
            "diagnosis_b3_b4r": "Docs/results/ms_sta_retrain_v4/DIAGNOSIS_B3_B4R.json",
            "per_sample_manifest_pointer": "Docs/results/ms_sta_retrain_v4/PER_SAMPLE_MANIFEST_POINTER.json",
        },
        "historical_not_primary": {
            "note": "v3 and earlier campaigns retained for provenance; not live SoT.",
            "retrain_v3": "Docs/results/ms_sta_retrain_v3/",
            "retrain_v3_registry": "data/processed/ms_sta_registry_v3_validation_no_go.json",
            "retrain_v3_b3_top1_display": "62,29 +/- 0,46%",
            "registry": "data/processed/ms_sta_registry.json",
            "defense_go": "Docs/results/ms_sta_defense_go/",
        },
        "forbid_claims": [
            "Defense GO",
            "current-best B4R / outperform",
            "B4R vượt trội B3",
            "93.75% as experimental success",
            "v3 62.29% as current-best",
            "open test / lock / v4e after v4 NO-GO",
        ],
        "created_at": now,
    }
    reg_sha = _write(_REG, registry)

    pointer = {
        "schema": "ms_sta_registry_active_pointer_v1",
        "registry_path": "data/processed/ms_sta_registry_v4_validation_no_go.json",
        "registry_sha256": reg_sha,
        "campaign_status": "VALIDATION_NO_GO",
        "defense_ready": False,
        "demo_allowed": False,
        "historical_fallback_disabled": True,
        "closeout_evidence_sha256": evidence_sha,
        "updated_at": now,
        "supersedes": "data/processed/ms_sta_registry_v3_validation_no_go.json",
    }
    _write(_ACTIVE, pointer)

    print(
        json.dumps(
            {
                "ok": True,
                "campaign_status": "VALIDATION_NO_GO",
                "report_sha256": report_sha,
                "manifest_sha256": manifest_sha,
                "closeout_evidence_sha256": evidence_sha,
                "registry_sha256": reg_sha,
                "evaluator_git_revision": evaluator_git_revision,
                "packager_git_revision": packager_git_revision,
                "packager_preflight_git_dirty": False,
                "canonical_b3_top1": b3["top1_mean_pct"],
                "out_dir": str(_OUT.relative_to(_ROOT).as_posix()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
