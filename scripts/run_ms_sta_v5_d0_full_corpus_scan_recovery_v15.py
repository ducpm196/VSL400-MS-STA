#!/usr/bin/env python3
"""D0 v15 scan-only recovery — read-only dual-root scan + completion publication.

Preserves CLOSED_FAIL runtime SCAN_REPORT/COMPLETION/seal under the campaign binary.
Writes create-only attempt packs under Docs/results/.../v15/scan_recovery/attempts/<id>/.
Does not extract Skeleton/RGB. Does not resume watchdog. Does not open M0/train.

--execute requires dedicated scan-recovery external PASS + post-auth artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL,
    D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL,
    EVIDENCE_ROOT_V15_REL,
    OUTPUT_VERSION_V15,
    V14_SEALED_SKELETON_BINARY_ROOT_REL,
    atomic_create_json,
    build_scan_recovery_attempt_auth_binds,
    evaluate_scan_recovery_execute_readiness_v15,
    load_json,
    preflight_scan_recovery_execute,
    publish_completion_pack,
    resolve_scan_recovery_attempt_dir,
    runtime_dir,
    sha256_file,
    validate_approval_v15,
    validate_scan_recovery_attempt_auth_binds,
)
from scan_ms_sta_v5_d0_full_corpus import scan_full_corpus  # noqa: E402

RECOVERY_ROOT_REL = f"{EVIDENCE_ROOT_V15_REL}/scan_recovery"
RECOVERY_SEAL_NAME = "D0_FULL_CORPUS_SCAN_RECOVERY_COMPLETION_SEAL.json"


def _build_plan(
    *,
    execute: bool,
    attempt_id: str | None,
) -> tuple[dict, int]:
    companion = _ROOT / D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL
    if not companion.is_file():
        return {"ok": False, "error": "scan_recovery_companion_missing"}, 2
    companion_obj = load_json(companion)
    if companion_obj.get("status") != "QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD":
        return {"ok": False, "error": "scan_recovery_companion_status_invalid"}, 2

    approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL)
    validate_approval_v15(approval, root=_ROOT)
    if approval.get("output_version") != OUTPUT_VERSION_V15:
        return {"ok": False, "error": "approval_not_v15"}, 2

    rgb_rel = str(companion_obj.get("binary_root_rel") or "").replace("\\", "/")
    skel_rel = str(
        companion_obj.get("skeleton_source_root_rel") or V14_SEALED_SKELETON_BINARY_ROOT_REL
    ).replace("\\", "/")
    if not rgb_rel:
        return {"ok": False, "error": "scan_recovery_companion_missing_binary_root"}, 2
    rgb_root = (_ROOT / rgb_rel).resolve()
    skel_root = (_ROOT / skel_rel).resolve()
    if not rgb_root.is_dir() or not skel_root.is_dir():
        return {
            "ok": False,
            "error": "scan_recovery_tensor_root_missing",
            "rgb_root_rel": rgb_rel,
            "skeleton_root_rel": skel_rel,
        }, 2

    rgb_runtime = runtime_dir(rgb_root)
    skel_runtime = runtime_dir(skel_root)
    root_r = _ROOT.resolve()

    def _rel(p: Path) -> str:
        return str(p.resolve().relative_to(root_r)).replace("\\", "/")

    readiness = evaluate_scan_recovery_execute_readiness_v15(root=_ROOT)
    execution_ready = bool(readiness.get("execution_ready"))
    raw_attempt = attempt_id or time.strftime("attempt_%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        attempt, _attempt_rt, attempt_rel = resolve_scan_recovery_attempt_dir(
            raw_attempt, root=_ROOT
        )
    except ValueError as exc:
        return {
            "ok": False,
            "error": "scan_recovery_attempt_id_invalid",
            "detail": str(exc),
            "attempt_id": raw_attempt,
        }, 2

    plan = {
        "ok": True,
        "execute": bool(execute),
        "execution_ready": execution_ready,
        "blockers": list(readiness.get("errors") or []),
        "attempt_id": attempt,
        "attempt_rel": attempt_rel,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "approval_binary_root_rel": approval.get("binary_root_rel"),
        "skeleton_tensor_root_rel": skel_rel,
        "skeleton_manifest_runtime_rel": _rel(skel_runtime),
        "rgb_tensor_root_rel": rgb_rel,
        "rgb_manifest_runtime_rel": _rel(rgb_runtime),
        "recovery_runtime_rel": attempt_rel,
        "preserves_closed_fail_scan_report": True,
        "companion_sha256": sha256_file(companion),
        "r1_remediation_companion_rel": D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL,
        "r1_remediation_companion_present": (
            _ROOT / D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL
        ).is_file(),
        "notes": [
            "Tensor roots bound from QUALITY_SCAN companion (CLOSED_FAIL RGB + sealed v14 skel).",
            "Does not overwrite CLOSED_FAIL SCAN_REPORT/COMPLETION under binary runtime.",
            "Attempt packs are create-only; FAIL reports are immutable.",
            "Does not authorize M0/train/auto-chain/extraction.",
        ],
    }
    return plan, 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run recovery scan only when scan-recovery external PASS + post-auth exist.",
    )
    ap.add_argument(
        "--attempt-id",
        default=None,
        help="Create-only attempt identity under scan_recovery/attempts/<id>/",
    )
    args = ap.parse_args()

    plan, code = _build_plan(execute=bool(args.execute), attempt_id=args.attempt_id)
    if code != 0:
        print(json.dumps(plan, indent=2))
        return code

    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    if not plan.get("execution_ready"):
        print(
            json.dumps(
                {
                    **plan,
                    "ok": False,
                    "error": "scan_recovery_execute_not_authorized",
                    "blockers": plan.get("blockers"),
                },
                indent=2,
            )
        )
        return 2

    attempt_id, attempt_rt, attempt_rel = resolve_scan_recovery_attempt_dir(
        str(plan["attempt_id"]), root=_ROOT
    )
    plan["attempt_id"] = attempt_id
    plan["attempt_rel"] = attempt_rel
    plan["recovery_runtime_rel"] = attempt_rel

    # Containment proof for operators / auditors.
    attempts_root = (_ROOT / EVIDENCE_ROOT_V15_REL / "scan_recovery" / "attempts").resolve()
    inside = True
    try:
        attempt_rt.relative_to(attempts_root)
    except ValueError:
        inside = False
    if not inside:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "scan_recovery_attempt_outside_root",
                    "attempt_id": attempt_id,
                    "inside_attempt_root": False,
                },
                indent=2,
            )
        )
        return 2

    if attempt_rt.exists():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "scan_recovery_attempt_exists_create_only",
                    "attempt_rel": attempt_rel,
                },
                indent=2,
            )
        )
        return 2

    # Provenance gate BEFORE any attempt directory / ATTEMPT_STARTED write.
    try:
        preflight = preflight_scan_recovery_execute(root=_ROOT)
    except ValueError as exc:
        print(
            json.dumps(
                {
                    **plan,
                    "ok": False,
                    "execute": True,
                    "error": "scan_recovery_preflight_failed",
                    "detail": str(exc),
                },
                indent=2,
            )
        )
        return 2

    attempt_rt.mkdir(parents=True, exist_ok=False)

    companion = _ROOT / D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL
    companion_obj = load_json(companion)
    approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL)
    rgb_root = (_ROOT / str(plan["rgb_tensor_root_rel"])).resolve()
    skel_root = (_ROOT / str(plan["skeleton_tensor_root_rel"])).resolve()
    rgb_runtime = runtime_dir(rgb_root)
    skel_runtime = runtime_dir(skel_root)

    auth_binds = build_scan_recovery_attempt_auth_binds(root=_ROOT)
    auth_binds["companion_sha256"] = plan["companion_sha256"]
    auth_binds["preflight"] = {
        "execution_head": preflight.get("execution_head"),
        "code_lock_git_revision": preflight.get("code_lock_git_revision"),
        "executable_tree_sha256": preflight.get("executable_tree_sha256"),
    }
    validate_scan_recovery_attempt_auth_binds(auth_binds, root=_ROOT)

    started_payload = {
        "schema": "ms_sta_v5_d0_full_corpus_scan_recovery_attempt_started_v15",
        "attempt_id": attempt_id,
        "attempt_rel": attempt_rel,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "immutable": True,
        "phase": "pre_scan",
        **auth_binds,
    }
    validate_scan_recovery_attempt_auth_binds(started_payload, root=_ROOT)
    atomic_create_json(attempt_rt / "ATTEMPT_STARTED.json", started_payload)

    thr = dict(approval.get("quality_thresholds") or {})
    try:
        scan = scan_full_corpus(
            binary_root=rgb_root,
            runtime_dir=rgb_runtime,
            thresholds=thr,
            root=_ROOT,
            skeleton_tensor_root=skel_root,
            skeleton_manifest_runtime=skel_runtime,
            rgb_tensor_root=rgb_root,
            rgb_manifest_runtime=rgb_runtime,
            output_version=OUTPUT_VERSION_V15,
            write_artifacts=True,
            output_runtime_dir=attempt_rt,
        )
    except Exception as exc:  # noqa: BLE001
        failure_payload = {
            "schema": "ms_sta_v5_d0_full_corpus_scan_recovery_attempt_failure_v15",
            "attempt_id": attempt_id,
            "attempt_rel": attempt_rel,
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "immutable": True,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "phase": "scan_full_corpus",
            "attempt_started_sha256": sha256_file(attempt_rt / "ATTEMPT_STARTED.json"),
            **auth_binds,
        }
        validate_scan_recovery_attempt_auth_binds(failure_payload, root=_ROOT)
        try:
            atomic_create_json(attempt_rt / "ATTEMPT_FAILURE.json", failure_payload)
        except FileExistsError:
            pass
        print(
            json.dumps(
                {
                    **plan,
                    "ok": False,
                    "execute": True,
                    "error": "scan_recovery_attempt_exception",
                    "failure": failure_payload,
                },
                indent=2,
            )
        )
        return 2

    report_path = attempt_rt / "SCAN_REPORT.json"
    report_sha = sha256_file(report_path) if report_path.is_file() else None
    identity_payload = {
        "schema": "ms_sta_v5_d0_full_corpus_scan_recovery_attempt_identity_v15",
        "attempt_id": attempt_id,
        "attempt_rel": attempt_rel,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scan_report_sha256": report_sha,
        "immutable": True,
        "attempt_started_sha256": sha256_file(attempt_rt / "ATTEMPT_STARTED.json"),
        **auth_binds,
    }
    validate_scan_recovery_attempt_auth_binds(identity_payload, root=_ROOT)
    atomic_create_json(attempt_rt / "ATTEMPT_IDENTITY.json", identity_payload)

    quality_stats = load_json(attempt_rt / "QUALITY_STATS.json")
    verdict = (
        "D0_FULL_CORPUS_SCAN_COMPLETE_PASS"
        if scan.get("ok") is True
        else "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL"
    )
    completion = {
        "schema": "ms_sta_v5_d0_full_corpus_scan_recovery_completion_v15",
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "verdict": verdict,
        "closed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hard_stop": True,
        "auto_chain_to_m0": False,
        "m0_authorized": False,
        "scan_ok": bool(scan.get("ok")),
        "failed_count": int(scan.get("failed_count") or 0),
        "global_failure_count": int(scan.get("global_failure_count") or 0),
        "sample_failure_count": int(scan.get("sample_failure_count") or 0),
        "unique_failed_sample_count": int(scan.get("unique_failed_sample_count") or 0),
        "n_paired": int(scan.get("n_paired") or 0),
        "quality_stats_sha256": scan.get("quality_stats_sha256"),
        "binary_root_rel": plan["rgb_tensor_root_rel"],
        "skeleton_source_root_rel": plan["skeleton_tensor_root_rel"],
        "recovery": True,
        "attempt_id": attempt_id,
        "source_verify_stats": scan.get("source_verify_stats"),
        **auth_binds,
    }
    validate_scan_recovery_attempt_auth_binds(completion, root=_ROOT)
    result = publish_completion_pack(
        evidence=attempt_rt,
        runtime=attempt_rt,
        completion=completion,
        scan_report=scan,
        quality_stats=quality_stats,
    )
    seal_path = attempt_rt / "D0_FULL_CORPUS_COMPLETION_SEAL.json"
    if seal_path.is_file():
        # Seal is a copy of completion — re-validate auth binds after publish.
        seal_obj = load_json(seal_path)
        validate_scan_recovery_attempt_auth_binds(seal_obj, root=_ROOT)
        dest = attempt_rt / RECOVERY_SEAL_NAME
        if not dest.exists():
            seal_path.replace(dest)

    out = {
        **plan,
        "execute": True,
        "ok": bool(scan.get("ok")),
        "verdict": verdict,
        "n_paired": scan.get("n_paired"),
        "failed_count": scan.get("failed_count"),
        "global_failure_count": scan.get("global_failure_count"),
        "sample_failure_count": scan.get("sample_failure_count"),
        "unique_failed_sample_count": scan.get("unique_failed_sample_count"),
        "quality_eval_ok": (scan.get("quality_eval") or {}).get("ok"),
        "scan_report_sha256": report_sha,
        "quality_stats_sha256": scan.get("quality_stats_sha256"),
        "completion_sha256": sha256_file(attempt_rt / "COMPLETION.json"),
        "binds": scan.get("binds"),
        "auth_binds": auth_binds,
        "source_verify_stats": scan.get("source_verify_stats"),
        "result_verdict": result.get("verdict"),
        "closed_fail_companion_status": companion_obj.get("status"),
        "inside_attempt_root": True,
    }
    print(json.dumps(out, indent=2))
    return 0 if scan.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
