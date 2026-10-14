#!/usr/bin/env python3
"""Create-only canonicalize of MS-STA v5 one-shot test evidence (no re-inference)."""

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
from src.eval.ms_sta_v5_oneshot_audit import OneshotAuditError, audit_oneshot, sha256_file  # noqa: E402


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    try:
        audit = audit_oneshot(root)
    except OneshotAuditError as exc:
        print(json.dumps({"event": "AUDIT_FAIL", "error": str(exc)}, indent=2))
        return 2

    head = git_rev(root)
    audit["head_at_audit"] = head
    audit["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    oneshot = root / "Docs/results/ms_sta_v5_oneshot"
    audit_sha = atomic_write(oneshot / "TEST_AUDIT.json", audit)

    completion = {
        "schema": "ms_sta_v5_oneshot_test_completion_v1",
        "status": "COMPLETE",
        "immutable": True,
        "no_reinference": True,
        "defense_ready": False,
        "d4_required_before_defense_go": True,
        "proposed_final_model": "b4_star_fixed",
        "primary_seed": 42,
        "test_kind": "retrospective_signer_disjoint_confirmation",
        "pristine_blind_claim": False,
        "b4_e_on_test": False,
        "test_n": 6165,
        "primary_contrast": "b4_star_fixed_vs_b3_mlp_roi_fixed",
        "summary": {
            "b4_top1": audit["metrics_recomputed"]["b4_star_fixed"]["top1"],
            "mlp_top1": audit["metrics_recomputed"]["b3_mlp_roi_fixed"]["top1"],
            "delta_top1_pp": 100.0 * audit["b4_minus_mlp"]["delta_top1"],
            "hierarchical_ci95_top1_pp": audit["b4_minus_mlp"]["hierarchical_ci95_top1_pp"],
            "delta_top5_pp": 100.0 * audit["b4_minus_mlp"]["delta_top5"],
            "hierarchical_ci95_top5_pp": audit["b4_minus_mlp"]["hierarchical_ci95_top5_pp"],
            "delta_macro_f1_pp": 100.0 * audit["b4_minus_mlp"]["delta_macro_f1"],
            "top1_improved_all_signers": audit["b4_minus_mlp"]["top1_improved_all_signers"],
            "top5_improved_all_signers": audit["b4_minus_mlp"]["top5_improved_all_signers"],
            "signer_27_top5_delta_pp": audit["b4_minus_mlp"]["signer_27_top5_delta_pp"],
            "discordant_top1": audit["b4_minus_mlp"]["discordant_top1"],
        },
        "bindings": audit["bindings"],
        "test_audit_sha256": audit_sha,
        "head_at_completion": head,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    completion_sha = atomic_write(oneshot / "TEST_COMPLETION.json", completion)

    evidence = {
        "schema": "ms_sta_v5_oneshot_test_evidence_index_v1",
        "status": "LOCKED",
        "defense_ready": False,
        "proposed_final_model": "b4_star_fixed",
        "artifacts": {
            "TEST_REPORT.json": audit["bindings"]["test_report_sha256"],
            "TEST_AUDIT.json": audit_sha,
            "TEST_COMPLETION.json": completion_sha,
            "PRETEST_FREEZE.json": audit["bindings"]["pretest_freeze_sha256"],
            "ONE_SHOT_TEST_AUTHORIZATION.json": audit["bindings"]["oneshot_auth_sha256"],
            "split_manifest": audit["bindings"]["split_manifest_sha256"],
            "VALIDATION_LOCK_V2.json": audit["bindings"]["validation_lock_v2_sha256"],
            "V5_E_CLOSEOUT.json": audit["bindings"]["v5_e_closeout_sha256"],
            **{f"TEST_PREDS_{k}_seed42.json": v for k, v in audit["bindings"]["prediction_sha256"].items()},
            **{f"ckpt_{k}": v for k, v in audit["bindings"]["checkpoint_sha256"].items()},
        },
        "next_gate": "D4_ACCEPTANCE_WITH_LOCKED_B4_SEED42",
        "head_at_index": head,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    evidence_sha = atomic_write(oneshot / "TEST_EVIDENCE_INDEX.json", evidence)

    final_results = {
        "schema": "ms_sta_v5_final_results_v1",
        "status": "TEST_CANONICALIZED_AWAITING_D4",
        "defense_ready": False,
        "defense_status": "PENDING_D4",
        "proposed_final_model": "b4_star_fixed",
        "primary_seed": 42,
        "validation": {
            "lock": "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json",
            "sha256": audit["bindings"]["validation_lock_v2_sha256"],
            "b4_mean_top1_pct": 64.75,
            "mlp_mean_top1_pct": 59.25,
            "mean_delta_top1_pp": 5.50,
            "hierarchical_ci95_top1_pp": [3.54, 7.69],
            "note": "Multi-seed validation; separate table from test.",
        },
        "test": {
            "kind": "retrospective_signer_disjoint_confirmation",
            "n": 6165,
            "report": "Docs/results/ms_sta_v5_oneshot/TEST_REPORT.json",
            "completion": "Docs/results/ms_sta_v5_oneshot/TEST_COMPLETION.json",
            "completion_sha256": completion_sha,
            "audit_sha256": audit_sha,
            "evidence_index_sha256": evidence_sha,
            "b4_top1": completion["summary"]["b4_top1"],
            "mlp_top1": completion["summary"]["mlp_top1"],
            "delta_top1_pp": completion["summary"]["delta_top1_pp"],
            "hierarchical_ci95_top1_pp": completion["summary"]["hierarchical_ci95_top1_pp"],
            "checkpoint_b4_sha256": audit["bindings"]["checkpoint_sha256"]["b4_star_fixed"],
        },
        "b4_e_supplement": {
            "role": "negative_engineering_appendix_only",
            "closeout": "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json",
            "not_proposed_model": True,
            "not_on_final_test": True,
        },
        "head": head,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    final_sha = atomic_write(oneshot / "FINAL_RESULTS.json", final_results)

    print(
        json.dumps(
            {
                "event": "ONESHOT_CANONICALIZED",
                "audit_sha256": audit_sha,
                "completion_sha256": completion_sha,
                "evidence_index_sha256": evidence_sha,
                "final_results_sha256": final_sha,
                "b4_top1": completion["summary"]["b4_top1"],
                "delta_top1_pp": completion["summary"]["delta_top1_pp"],
                "signer_27_top5_delta_pp": completion["summary"]["signer_27_top5_delta_pp"],
                "discordant": completion["summary"]["discordant_top1"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
