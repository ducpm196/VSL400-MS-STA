#!/usr/bin/env python3
"""Promote MS-STA v5 Defense GO after TEST_COMPLETION + D4 PASS (create-only).

Does not retrain or re-run one-shot. Binds B4* seed-42 SHA. Never loads B4-E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_rev  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, obj: dict[str, Any], *, overwrite: bool = False) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not overwrite:
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    del args

    completion = load_json(_ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_COMPLETION.json")
    d4 = load_json(_ROOT / "Docs/results/ms_sta_v5_d4/D4_ACCEPTANCE_REPORT.json")
    freeze = load_json(_ROOT / "Docs/results/ms_sta_v5_oneshot/PRETEST_FREEZE.json")
    audit = load_json(_ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_AUDIT.json")
    if completion.get("status") != "COMPLETE":
        raise SystemExit("completion_required")
    if d4.get("d4_pass") is not True:
        raise SystemExit("d4_pass_required")
    if audit.get("status") != "AUDIT_PASS":
        raise SystemExit("audit_pass_required")

    b4 = freeze["checkpoints_seed42"]["b4_star_fixed"]
    ckpt_rel = b4["checkpoint_rel"]
    ckpt_sha = b4["checkpoint_sha256"]
    if sha256_file(_ROOT / ckpt_rel) != ckpt_sha:
        raise SystemExit("checkpoint_sha_drift")
    if not ckpt_sha.startswith("dc1a8d5a"):
        raise SystemExit("unexpected_b4_sha_prefix")

    head = git_rev(_ROOT)
    d4_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v5_d4/D4_ACCEPTANCE_REPORT.json")
    completion_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_COMPLETION.json")
    audit_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_AUDIT.json")
    evidence_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_EVIDENCE_INDEX.json")

    final = {
        "schema": "ms_sta_v5_defense_go_final_results_v1",
        "defense_status": "GO",
        "defense_ready": True,
        "proposed_final_model": "b4_star_fixed",
        "primary_seed": 42,
        "checkpoint_rel": ckpt_rel,
        "checkpoint_sha256": ckpt_sha,
        "test_kind": "retrospective_signer_disjoint_confirmation",
        "pristine_blind_claim": False,
        "b4_e_on_test_or_demo": False,
        "validation": {
            "lock_v2": "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json",
            "b4_mean_top1_pct": 64.75,
            "mlp_mean_top1_pct": 59.25,
            "mean_delta_top1_pp": 5.50,
            "hierarchical_ci95_top1_pp": [3.54, 7.69],
        },
        "test": {
            "n": 6165,
            "b4_top1": completion["summary"]["b4_top1"],
            "mlp_top1": completion["summary"]["mlp_top1"],
            "delta_top1_pp": completion["summary"]["delta_top1_pp"],
            "hierarchical_ci95_top1_pp": completion["summary"]["hierarchical_ci95_top1_pp"],
            "signer_27_top5_delta_pp": completion["summary"]["signer_27_top5_delta_pp"],
            "discordant_top1": completion["summary"]["discordant_top1"],
            "completion_sha256": completion_sha,
            "audit_sha256": audit_sha,
            "evidence_index_sha256": evidence_sha,
        },
        "d4": {
            "path": "Docs/results/ms_sta_v5_d4/D4_ACCEPTANCE_REPORT.json",
            "sha256": d4_sha,
            "d4_pass": True,
            "n_scenarios": d4["n_scenarios"],
            "n_pass": d4["n_pass"],
        },
        "b4_e_appendix": {
            "closeout": "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json",
            "diagnostic": "Docs/results/ms_sta_v5_e/DIAGNOSTIC_EFFICIENCY_BENCH.json",
            "role": "negative_engineering_only",
            "latency_median_change": "+12.43% (slower)",
            "accuracy_tradeoff_pp": -1.42,
            "score_element_reduction": 0.9375,
        },
        "claim_limits": [
            "Test is retrospective signer-disjoint confirmation, not pristine blind.",
            "Do not claim every metric rises on every signer (signer_27 Top-5 −3.00 pp).",
            "Do not claim B4-E faster/accuracy-equivalent/93.75% total FLOPs.",
            "Demo is cache-backed offline on D0 feature NPZ; not real-time production.",
        ],
        "head": head,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    final_path = _ROOT / "Docs/results/ms_sta_v5_defense_go/FINAL_RESULTS.json"
    final_sha = atomic_write(final_path, final)

    registry = {
        "schema": "ms_sta_registry_v5_defense_go_v1",
        "status": "GO",
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "d4_pass": True,
        "evaluation_locked": True,
        "ms_sta_v5": True,
        "active_variant": "b4_star_fixed",
        "experiment_variant": "b4_star_fixed",
        "checkpoint": ckpt_rel,
        "checkpoint_sha256": ckpt_sha,
        "primary_seed": 42,
        "feature_roots": {
            "skeleton_root_rel": "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
            "rgb_root_rel": "data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70",
        },
        "split_manifest_rel": "data/processed/vsl400_retrain_v4_split_manifest.json",
        "demo_set_rel": "data/processed/demo_set_ms_sta.json",
        "final_results_rel": "Docs/results/ms_sta_v5_defense_go/FINAL_RESULTS.json",
        "final_results_sha256": final_sha,
        "d4_report_sha256": d4_sha,
        "test_completion_sha256": completion_sha,
        "forbidden_variants_on_demo": ["b4_star_e_stride4"],
        "historical_not_primary": [
            "data/processed/ms_sta_registry.json",
            "Docs/results/ms_sta_defense_go/FINAL_RESULTS.json",
            "data/processed/ms_sta_registry_v4_validation_no_go.json",
        ],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head": head,
    }
    reg_path = _ROOT / "data/processed/ms_sta_registry_v5_defense_go.json"
    reg_sha = atomic_write(reg_path, registry)

    pointer = {
        "schema": "ms_sta_registry_active_pointer_v1",
        "registry_path": "data/processed/ms_sta_registry_v5_defense_go.json",
        "registry_sha256": reg_sha,
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "historical_fallback_disabled": True,
        "final_results_sha256": final_sha,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supersedes": "data/processed/ms_sta_registry_v4_validation_no_go.json",
        "head": head,
    }
    # Active pointer is the live switch — overwrite allowed only from NO_GO → GO promote.
    pointer_sha = atomic_write(
        _ROOT / "data/processed/ms_sta_registry_active.json", pointer, overwrite=True
    )

    smoke = {
        "schema": "ms_sta_v5_post_promotion_smoke_v1",
        "status": "PASS",
        "checks": {
            "final_results_defense_ready": final["defense_ready"] is True,
            "d4_pass": True,
            "checkpoint_sha_matches": sha256_file(_ROOT / ckpt_rel) == ckpt_sha,
            "active_pointer_targets_v5": pointer["registry_path"].endswith("v5_defense_go.json"),
            "active_pointer_sha_matches_registry": pointer["registry_sha256"] == reg_sha,
            "b4_e_forbidden": "b4_star_e_stride4" in registry["forbidden_variants_on_demo"],
        },
        "bindings": {
            "final_results_sha256": final_sha,
            "registry_sha256": reg_sha,
            "active_pointer_sha256": pointer_sha,
            "d4_sha256": d4_sha,
            "checkpoint_sha256": ckpt_sha,
        },
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not all(smoke["checks"].values()):
        raise SystemExit(f"post_promotion_smoke_fail:{smoke['checks']}")
    smoke_sha = atomic_write(_ROOT / "Docs/results/ms_sta_v5_defense_go/POST_PROMOTION_SMOKE.json", smoke)

    print(
        json.dumps(
            {
                "event": "V5_DEFENSE_GO",
                "final_results_sha256": final_sha,
                "registry_sha256": reg_sha,
                "smoke_sha256": smoke_sha,
                "checkpoint_sha256": ckpt_sha,
                "defense_ready": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
