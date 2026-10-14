#!/usr/bin/env python3
"""D0 full-corpus runner (v14) — C3 HandRecovery parity, SHA-locked resume, no M0.

Phases: skeleton_v3 -> rgb_token_v3 -> quality_scan -> completion
Requires sealed approval v14, external re-audit PASS, and post-audit execution auth. Does not launch unless --launch is set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    CAMPAIGN_TOKEN_V15,
    CORPUS_N,
    D0_FULL_CORPUS_APPROVAL_V14_REL,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL,
    D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
    EXTRACT_PY_REL,
    F0_PY_REL,
    MAX_RESUMES,
    OUTPUT_VERSION,
    OUTPUT_VERSION_V15,
    PHASES_EXACT,
    RGB_MANIFEST_REQUIRED_FIELDS,
    RGB_REQUIRED_KEYS,
    SKEL_MANIFEST_REQUIRED_FIELDS,
    SKEL_REQUIRED_KEYS,
    append_jsonl,
    assert_psutil_available,
    atomic_create_json,
    atomic_replace_json,
    atomic_savez_compressed,
    binary_root_path,
    build_immutable_run_contract_payload,
    campaign_related_process_alive,
    canonical_sample_paths,
    commit_manifest_record,
    disk_preflight,
    evaluate_launch_readiness,
    evaluate_launch_readiness_v15,
    evidence_root_path,
    heal_or_resume_sample,
    load_eligible_identity,
    load_json,
    load_manifest_index,
    promote_file,
    publish_completion_pack,
    publish_deterministic_phase_failure_create_only,
    register_worker,
    rebuild_manifest_index,
    require_launch_readiness,
    require_launch_readiness_v15,
    require_manifest_fields,
    runtime_dir,
    sha256_file,
    staging_dir,
    unregister_worker,
    validate_approval_v14,
    validate_approval_v15,
    validate_execution_protocol_v14,
    validate_execution_protocol_v15,
    validate_external_reaudit_pass,
    validate_external_reaudit_pass_v15,
    validate_npz_keys,
    validate_post_audit_execution_authorization,
    validate_post_audit_execution_authorization_v15,
    verify_skeleton_carry_forward_from_v14,
    write_ready_descriptor,
    _process_identity,
)
from ms_sta_v5_d0_lib import (  # noqa: E402
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    R7C_AMBIGUITY_ROI_LOCK,
    sha256_file as _sha256_file,
)

OBS_SOURCE_MAP = {"missing": 0, "holistic": 1, "hands_fallback": 2}
SKEL_PHASE = "skeleton_v3"
RGB_PHASE = "rgb_token_v3"


def _enc_obs(arr: Any) -> np.ndarray:
    return np.asarray([OBS_SOURCE_MAP[str(x)] for x in arr], dtype=np.uint8)


def _load_metadata_maps(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    import csv

    by_id: dict[str, dict[str, str]] = {}
    split_by: dict[str, str] = {}
    signer_by: dict[str, str] = {}
    with (root / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["video_id"]
            by_id[sid] = row
            split_by[sid] = row.get("split") or "unknown"
            signer_by[sid] = row.get("signer_id") or "unknown"
    return by_id, split_by, signer_by


class DeterministicDependencyError(RuntimeError):
    """Phase-wide dependency failure — HALT; do not per-sample fan-out or watchdog-resume."""


def preflight_rgb_f0_runtime() -> None:
    """Fail-fast under the same interpreter that runs rgb_token_v3 (expected: _venvs/f0_v5)."""
    import importlib
    import sys

    required = (
        "numpy",
        "torch",
        "cv2",
        "src.data_prep.raw_video_resolve",
        "src.models.rgb_token_v3",
    )
    missing: list[str] = []
    pandas_already = "pandas" in sys.modules
    for mod in required:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:
            missing.append(f"{mod}:{exc.name or mod}")
    if (not pandas_already) and ("pandas" in sys.modules):
        missing.append("raw_video_resolve_or_rgb_pulled_pandas")
    if missing:
        raise DeterministicDependencyError(
            "rgb_f0_preflight_missing_deps:" + ",".join(missing)
        )


def _resolve_video(root: Path, row: dict[str, str]) -> Path:
    from src.data_prep.raw_video_resolve import RawVideoResolver

    resolver = RawVideoResolver(root)
    path = resolver.resolve(row)
    if path is None or not path.is_file():
        raise RuntimeError(f"video_missing:{row.get('video_id')}")
    return path


def _publish_deterministic_phase_failure(
    *,
    runtime: Path,
    phase: str,
    exc: BaseException,
    sample_id: str | None = None,
    n_ok: int = 0,
    n_fail: int = 0,
    n_resume: int = 0,
) -> dict[str, Any]:
    return publish_deterministic_phase_failure_create_only(
        runtime,
        phase,
        exc,
        n_ok=n_ok,
        n_fail=n_fail,
        n_resume=n_resume,
        sample_id=sample_id,
    )


def _resolve_skeleton_source_root(*, root: Path, binary: Path, approval: dict[str, Any] | None) -> Path:
    """Read skeleton from sealed source root when approval/env binds carry-forward."""
    env_rel = os.environ.get("D0_SKELETON_SOURCE_ROOT") or os.environ.get("D0_SKELETON_SOURCE_ROOT_REL")
    rel = None
    if approval:
        rel = approval.get("skeleton_source_root_rel")
    if env_rel:
        rel = env_rel
    if not rel:
        return binary
    path = Path(rel)
    if not path.is_absolute():
        path = root / path
    return path


def _skel_arrays(payload: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    return {
        "coordinates": payload["coordinates"],
        "joints_raw": payload["joints_raw"],
        "velocity": payload["velocity"],
        "bone_vectors": payload["bone_vectors"],
        "missing_mask_prefill": payload["missing_mask_prefill"],
        "validity": payload["validity"],
        "confidence": payload["confidence"],
        "original_frame_index": payload["original_frame_index"],
        "timestamps_sec": payload["timestamps_sec"],
        "padding_mask": payload["padding_mask"],
        "joint_part": payload["joint_part"],
        "joint_chirality": payload["joint_chirality"],
        "bone_edges": payload["bone_edges"],
        "sequence_length": np.asarray([payload["sequence_length"]], dtype=np.int32),
        "raw_landmarks": recovery["raw_landmarks"],
        "effective_landmarks": recovery["effective_landmarks"],
        "raw_holistic_missing": recovery["raw_holistic_missing"],
        "effective_missing": recovery["effective_missing"],
        "observation_source_left_hand": _enc_obs(recovery["observation_source_left_hand"]),
        "observation_source_right_hand": _enc_obs(recovery["observation_source_right_hand"]),
        "wrist_assignment_distance_left": np.asarray(
            [(-1.0 if v is None else float(v)) for v in recovery["wrist_assignment_distance_left"]],
            dtype=np.float32,
        ),
        "wrist_assignment_distance_right": np.asarray(
            [(-1.0 if v is None else float(v)) for v in recovery["wrist_assignment_distance_right"]],
            dtype=np.float32,
        ),
    }


def _skel_sidecar_meta(sample_id: str, recovery: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "arm": recovery["arm"],
        "pose_guided_crop_left": recovery["pose_guided_crop_left"],
        "pose_guided_crop_right": recovery["pose_guided_crop_right"],
        "fallback_rejection_reason_left": recovery["fallback_rejection_reason_left"],
        "fallback_rejection_reason_right": recovery["fallback_rejection_reason_right"],
        "handedness_hard_reject_disabled_for_non_mirrored_input": True,
    }


def _save_skel_atomic(
    out_dir: Path,
    sample_id: str,
    payload: dict[str, Any],
    recovery: dict[str, Any],
    *,
    manifest_row: dict[str, Any] | None = None,
    cleanup_staging: bool = True,
) -> tuple[Path, Path, str]:
    """Stage → validate → READY (optional) → promote finals. Used by tests and phase runner."""
    stage = staging_dir(out_dir, SKEL_PHASE, sample_id)
    stage.mkdir(parents=True, exist_ok=True)
    dest_final = out_dir / SKEL_PHASE / f"{sample_id}.npz"
    sidecar_final = out_dir / SKEL_PHASE / f"{sample_id}.recovery.json"
    stage_npz = stage / dest_final.name
    stage_side = stage / sidecar_final.name

    dig = atomic_savez_compressed(stage_npz, **_skel_arrays(payload, recovery))
    validate_npz_keys(stage_npz, SKEL_REQUIRED_KEYS)
    sidecar_sha = atomic_replace_json(stage_side, _skel_sidecar_meta(sample_id, recovery))

    if manifest_row is not None:
        row = dict(manifest_row)
        row["output_sha256"] = dig
        row["sidecar_sha256"] = sidecar_sha
        require_manifest_fields(row, SKEL_MANIFEST_REQUIRED_FIELDS)
        write_ready_descriptor(
            stage / "READY.json",
            {
                "manifest_row": row,
                "output_sha256": dig,
                "sidecar_sha256": sidecar_sha,
                "source_sha256": row.get("source_sha256"),
            },
        )

    promote_file(stage_npz, dest_final)
    promote_file(stage_side, sidecar_final)
    if cleanup_staging:
        shutil.rmtree(stage, ignore_errors=True)
    return dest_final, sidecar_final, dig


def _partial_masks_from_windows(windows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    regions = ("face", "left_hand", "right_hand")
    out: dict[str, list[bool]] = {r: [] for r in regions}
    for w in windows:
        rv = w.get("region_validity") or {}
        for r in regions:
            out[r].append(bool((rv.get(r) or {}).get("partial_missing")))
    return {r: np.asarray(v, dtype=bool) for r, v in out.items()}


def _rgb_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    arrays: dict[str, Any] = {f"token_{k}": v for k, v in payload["tokens"].items()}
    arrays.update({f"missing_{k}": np.asarray(v, dtype=bool) for k, v in payload["missing_region_mask"].items()})
    partial = _partial_masks_from_windows(payload.get("windows") or [])
    for r, arr in partial.items():
        arrays[f"partial_{r}"] = arr
    for r in ("face", "left_hand", "right_hand"):
        key = f"partial_{r}"
        miss_key = f"missing_{r}"
        if key not in arrays:
            n = int(np.asarray(arrays[miss_key]).shape[0]) if miss_key in arrays else 0
            arrays[key] = np.zeros((n,), dtype=bool)
    arrays["pathway_slow"] = payload["pathway_global_tokens"]["slow"]
    arrays["pathway_fast"] = payload["pathway_global_tokens"]["fast"]
    arrays["original_frame_index"] = payload["original_frame_index"]
    return arrays


def _rgb_sidecar_meta(sample_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "windows": payload["windows"],
        "fps": payload["fps"],
        "n_native_frames": payload["n_native_frames"],
        "hook": payload["hook"],
        "roi_method": payload["roi_method"],
        "regions": payload["regions"],
        "window_starts": payload.get("window_starts"),
        "window_size": payload.get("window_size"),
        "window_stride": payload.get("window_stride"),
        "backbone": payload.get("backbone"),
        "operator_order": payload.get("operator_order"),
        "temporal_aggregation": payload.get("temporal_aggregation"),
        "enable_pose_wrist_prior": payload.get("enable_pose_wrist_prior"),
        "enable_ambiguity_aware_prior": True,
        "ambiguity_roi_lock": dict(R7C_AMBIGUITY_ROI_LOCK),
        "dual_source": {
            "skeleton": ["holistic", "hands_fallback", "missing"],
            "rgb_roi": ["hand_landmarks", "pose_wrist_prior", "bimanual_shared_prior", "missing"],
        },
    }


def _save_rgb_atomic(
    out_dir: Path,
    sample_id: str,
    payload: dict[str, Any],
    *,
    manifest_row: dict[str, Any] | None = None,
    cleanup_staging: bool = True,
) -> tuple[Path, Path, str]:
    """Stage → validate → READY (optional) → promote finals. Used by tests and phase runner."""
    stage = staging_dir(out_dir, RGB_PHASE, sample_id)
    stage.mkdir(parents=True, exist_ok=True)
    dest_final = out_dir / RGB_PHASE / f"{sample_id}.npz"
    sidecar_final = out_dir / RGB_PHASE / f"{sample_id}.meta.json"
    stage_npz = stage / dest_final.name
    stage_side = stage / sidecar_final.name

    dig = atomic_savez_compressed(stage_npz, **_rgb_arrays(payload))
    validate_npz_keys(stage_npz, RGB_REQUIRED_KEYS)
    sidecar_sha = atomic_replace_json(stage_side, _rgb_sidecar_meta(sample_id, payload))

    if manifest_row is not None:
        row = dict(manifest_row)
        row["output_sha256"] = dig
        row["sidecar_sha256"] = sidecar_sha
        require_manifest_fields(row, RGB_MANIFEST_REQUIRED_FIELDS)
        write_ready_descriptor(
            stage / "READY.json",
            {
                "manifest_row": row,
                "output_sha256": dig,
                "sidecar_sha256": sidecar_sha,
                "source_sha256": row.get("source_sha256"),
                "skel_sha256": row.get("skel_sha256"),
            },
        )

    promote_file(stage_npz, dest_final)
    promote_file(stage_side, sidecar_final)
    if cleanup_staging:
        shutil.rmtree(stage, ignore_errors=True)
    return dest_final, sidecar_final, dig


def _ledger(runtime: Path, row: dict[str, Any]) -> None:
    append_jsonl(runtime / "FAILURE_LEDGER.jsonl", row)


def _update_mutable(runtime: Path, state: dict[str, Any]) -> None:
    atomic_replace_json(runtime / "RUNTIME_MUTABLE.json", state)


def _load_mutable(runtime: Path) -> dict[str, Any]:
    path = runtime / "RUNTIME_MUTABLE.json"
    if path.is_file():
        return load_json(path)
    return {
        "status": "RUNNING",
        "resumes_used": 0,
        "restart_count": 0,
        "orchestrator_pid": None,
        "campaign_token": CAMPAIGN_TOKEN,
        "phase": None,
    }


def _assert_contract_identity(
    prev: dict[str, Any],
    approval: dict[str, Any],
    *,
    root: Path,
    campaign_token: str = CAMPAIGN_TOKEN,
) -> None:
    if prev.get("campaign_token") != campaign_token:
        raise SystemExit(2)
    bin_rel = str(approval.get("binary_root_rel") or "").replace("\\", "/")
    if prev.get("binary_root_rel") != bin_rel:
        raise SystemExit(2)
    if campaign_token == CAMPAIGN_TOKEN_V15:
        if prev.get("approval_v15_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V15_REL):
            raise SystemExit(2)
        if prev.get("execution_protocol_v15_sha256") != sha256_file(
            root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
        ):
            raise SystemExit(2)
        if prev.get("external_reaudit_v15_sha256") != sha256_file(
            root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
        ):
            raise SystemExit(2)
        if prev.get("post_audit_execution_auth_v15_sha256") != sha256_file(
            root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
        ):
            raise SystemExit(2)
    else:
        if prev.get("approval_v14_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V14_REL):
            raise SystemExit(2)
        if prev.get("execution_protocol_v14_sha256") != sha256_file(
            root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL
        ):
            raise SystemExit(2)
        if prev.get("external_reaudit_v14_sha256") != sha256_file(
            root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL
        ):
            raise SystemExit(2)
        if prev.get("post_audit_execution_auth_v14_sha256") != sha256_file(
            root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL
        ):
            raise SystemExit(2)
    if prev.get("code_lock_git_revision") != approval["code_lock_git_revision"]:
        raise SystemExit(2)
    if prev.get("executable_tree_sha256") != approval["executable_tree_sha256"]:
        raise SystemExit(2)
    env_prev = (prev.get("env_provenance") or {}).get("f0_pip_freeze_sha256")
    if env_prev is not None:
        from ms_sta_v5_d0_full_corpus_lib import env_provenance

        env_now = env_provenance(root=root).get("f0_pip_freeze_sha256")
        if env_now != env_prev:
            raise SystemExit(2)


def _build_v15_immutable_run_contract_payload(
    *,
    approval: dict[str, Any],
    protocol: dict[str, Any],
    resume: bool,
    root: Path,
) -> dict[str, Any]:
    del protocol  # validated by launch gates; binds use disk SHAs
    return {
        "schema": "ms_sta_v5_d0_full_corpus_immutable_run_contract_v15",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resume": bool(resume),
        "output_version": OUTPUT_VERSION_V15,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "eligible_n": CORPUS_N,
        "approval_v15_sha256": sha256_file(root / D0_FULL_CORPUS_APPROVAL_V15_REL),
        "execution_protocol_v15_sha256": sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        "external_reaudit_v15_sha256": sha256_file(
            root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
        ),
        "post_audit_execution_auth_v15_sha256": sha256_file(
            root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
        ),
        "code_lock_git_revision": approval["code_lock_git_revision"],
        "executable_tree_sha256": approval["executable_tree_sha256"],
        "binary_root_rel": str(approval["binary_root_rel"]).replace("\\", "/"),
        "skeleton_source_root_rel": str(approval.get("skeleton_source_root_rel") or "").replace(
            "\\", "/"
        ),
        "carry_forward_required": bool(approval.get("carry_forward_required")),
        "max_resumes": MAX_RESUMES,
        "worker_registry_required": True,
        "sample_transaction_required": True,
        "atomic_manifest_required": True,
        "auto_chain_to_m0": False,
        "stop_after_pass": "D0_FULL_CORPUS_SCAN_COMPLETE_PASS",
        "stop_after_fail": "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL",
    }


def _require_launch_gates(
    *,
    resume: bool,
    root: Path,
    bound_worker: bool = False,
    campaign_token: str = CAMPAIGN_TOKEN,
) -> tuple[dict, dict, Path, Path]:
    if campaign_token not in (CAMPAIGN_TOKEN, CAMPAIGN_TOKEN_V15):
        raise RuntimeError("campaign_token_mismatch")
    assert_psutil_available()
    is_v15 = campaign_token == CAMPAIGN_TOKEN_V15
    if is_v15:
        if bound_worker:
            # Attach path: skip absent-or-empty binary gate (runtime already seeded).
            approval = load_json(root / D0_FULL_CORPUS_APPROVAL_V15_REL)
            protocol = load_json(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL)
            validate_approval_v15(approval, root=root)
            validate_execution_protocol_v15(protocol, root=root)
            reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
            auth_path = root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
            if not reaudit_path.is_file() or not auth_path.is_file():
                raise RuntimeError("bound_worker_missing_external_pass_or_post_auth")
            validate_external_reaudit_pass_v15(load_json(reaudit_path), root=root)
            validate_post_audit_execution_authorization_v15(load_json(auth_path), root=root)
        else:
            gate = require_launch_readiness_v15(root=root)
            approval = gate["approval"]
            protocol = gate["protocol"]
        assert approval is not None and protocol is not None
        bin_rel = str(approval.get("binary_root_rel") or "").replace("\\", "/")
        if not bin_rel:
            raise RuntimeError("approval_v15_binary_root_rel_missing")
        binary = (root / bin_rel).resolve()
    else:
        if bound_worker:
            approval = load_json(root / D0_FULL_CORPUS_APPROVAL_V14_REL)
            protocol = load_json(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL)
            validate_approval_v14(approval, root=root)
            validate_execution_protocol_v14(protocol, root=root)
            reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL
            auth_path = root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL
            if not reaudit_path.is_file() or not auth_path.is_file():
                raise RuntimeError("bound_worker_missing_external_pass_or_post_auth")
            validate_external_reaudit_pass(load_json(reaudit_path), root=root)
            validate_post_audit_execution_authorization(load_json(auth_path), root=root)
            binary = binary_root_path(approval["code_lock_git_revision"], root=root)
        else:
            gate = require_launch_readiness(root=root)
            approval = gate["approval"]
            protocol = gate["protocol"]
            assert approval is not None and protocol is not None
            binary = binary_root_path(approval["code_lock_git_revision"], root=root)
    if _sha256_file(root / EXPECTED_SLOWFAST_CKPT_REL) != EXPECTED_SLOWFAST_CKPT_SHA256:
        raise RuntimeError("backbone_sha_drift")
    disk = disk_preflight(float((approval.get("disk_preflight") or {}).get("min_free_gb") or 80), root=root)
    if not disk["ok"]:
        raise RuntimeError(f"disk_preflight_failed:{disk}")
    runtime = runtime_dir(binary)
    runtime.mkdir(parents=True, exist_ok=True)
    contract_path = runtime / "_run_contract.json"
    mutable = _load_mutable(runtime)

    if contract_path.is_file():
        prev = load_json(contract_path)
        _assert_contract_identity(prev, approval, root=root, campaign_token=campaign_token)
        if bound_worker:
            return approval, protocol, binary, runtime
        if not resume:
            raise RuntimeError("existing_output_without_resume")
        resumes = int(mutable.get("resumes_used") or 0)
        if resumes >= MAX_RESUMES:
            raise RuntimeError("max_resumes_exceeded")
        old_pid = int(mutable.get("orchestrator_pid") or 0)
        old_ct = mutable.get("orchestrator_create_time")
        try:
            old_ct_f = float(old_ct) if old_ct is not None else None
        except (TypeError, ValueError):
            old_ct_f = None
        if campaign_related_process_alive(
            old_pid,
            campaign_token=campaign_token,
            runtime=runtime,
            orchestrator_create_time=old_ct_f,
        ):
            raise RuntimeError(f"live_process_tree_blocks_resume:{old_pid}")
        mutable["resumes_used"] = resumes + 1
        mutable["status"] = "RESUMED"
        _bind_orchestrator_identity(mutable)
        mutable["campaign_token"] = campaign_token
        mutable["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _update_mutable(runtime, mutable)
    else:
        if bound_worker:
            raise RuntimeError("bound_worker_requires_existing_contract")
        if resume:
            raise RuntimeError("resume_without_existing_contract")
        if is_v15:
            payload = _build_v15_immutable_run_contract_payload(
                approval=approval, protocol=protocol, resume=False, root=root
            )
        else:
            payload = build_immutable_run_contract_payload(
                approval=approval, protocol=protocol, resume=False, root=root
            )
        payload.pop("orchestrator_pid", None)
        payload.pop("resumes_used", None)
        payload["status"] = "BOUND"
        atomic_create_json(contract_path, payload)
        mutable = {
            "status": "RUNNING",
            "resumes_used": 0,
            "restart_count": 0,
            "orchestrator_pid": os.getpid(),
            "orchestrator_create_time": None,
            "campaign_token": campaign_token,
            "phase": None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _bind_orchestrator_identity(mutable)
        _update_mutable(runtime, mutable)
    return approval, protocol, binary, runtime


def _heal_manifest_from_ready(
    *,
    binary: Path,
    runtime: Path,
    phase: str,
    sample_id: str,
    manifest_index: dict[str, dict[str, Any]],
) -> None:
    ready_path = staging_dir(binary, phase, sample_id) / "READY.json"
    ready = load_json(ready_path)
    man_row = dict(ready.get("manifest_row") or ready)
    record_sha = commit_manifest_record(runtime, phase, man_row)
    ready["manifest_record_sha256"] = record_sha
    write_ready_descriptor(ready_path, ready)
    manifest_index[sample_id] = man_row
    shutil.rmtree(staging_dir(binary, phase, sample_id), ignore_errors=True)


def run_skeleton_phase(*, binary: Path, runtime: Path, root: Path, resume: bool) -> dict[str, Any]:
    from src.data_prep.hand_recovery_v6 import HandRecoveryEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    identity = load_eligible_identity(root=root)
    meta_by, split_by, signer_by = _load_metadata_maps(root)
    engine = HandRecoveryEngine(arm="C3", model_complexity=0, store_stride=2)
    n_ok = n_fail = n_resume = 0
    manifest_index = load_manifest_index(runtime, SKEL_PHASE)
    try:
        for sid in identity["eligible_ids"]:
            row = meta_by.get(sid)
            if row is None:
                _ledger(runtime, {"sample_id": sid, "phase": SKEL_PHASE, "error": "missing_metadata"})
                n_fail += 1
                continue
            dest = binary / SKEL_PHASE / f"{sid}.npz"
            sidecar = binary / SKEL_PHASE / f"{sid}.recovery.json"
            ready_path = staging_dir(binary, SKEL_PHASE, sid) / "READY.json"
            try:
                video = _resolve_video(root, row)
                source_sha = sha256_file(video)
                skip_sha = heal_or_resume_sample(
                    dest=dest,
                    sidecar=sidecar,
                    ready_path=ready_path,
                    manifest_index=manifest_index,
                    sample_id=sid,
                    required_keys=SKEL_REQUIRED_KEYS,
                    required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
                    resume=resume,
                    source_sha=source_sha,
                )
                if skip_sha is not None:
                    if skip_sha.startswith("HEAL_MANIFEST:"):
                        _heal_manifest_from_ready(
                            binary=binary,
                            runtime=runtime,
                            phase=SKEL_PHASE,
                            sample_id=sid,
                            manifest_index=manifest_index,
                        )
                    n_resume += 1
                    n_ok += 1
                    continue
                rec = engine.process_video(video)
                payload = build_skel_v3_arrays(
                    joints_raw=rec["effective_landmarks"],
                    raw_pose=rec["raw_pose"],
                    missing_prefill=rec["effective_missing"],
                    original_frame_index=rec["original_frame_index"],
                    timestamps_sec=rec["timestamps_sec"],
                )
                payload["joints_raw"] = np.asarray(rec["effective_landmarks"], dtype=np.float32)
                output_rel, sidecar_rel = canonical_sample_paths(
                    root=root, binary=binary, phase=SKEL_PHASE, sample_id=sid
                )
                man_row = {
                    "sample_id": sid,
                    "split": split_by.get(sid),
                    "signer_id": signer_by.get(sid),
                    "source_sha256": source_sha,
                    "output_rel": output_rel,
                    "sidecar_rel": sidecar_rel,
                    "output_sha256": None,
                    "sidecar_sha256": None,
                    "producer": "scripts/run_ms_sta_v5_d0_full_corpus.py:skeleton_v3:HandRecoveryEngine:C3",
                }
                path, side_path, dig = _save_skel_atomic(
                    binary, sid, payload, rec, manifest_row=man_row, cleanup_staging=False
                )
                man_row["output_sha256"] = dig
                man_row["sidecar_sha256"] = sha256_file(side_path)
                require_manifest_fields(man_row, SKEL_MANIFEST_REQUIRED_FIELDS)
                if sid in manifest_index:
                    raise SystemExit(2)
                record_sha = commit_manifest_record(runtime, SKEL_PHASE, man_row)
                ready_path = staging_dir(binary, SKEL_PHASE, sid) / "READY.json"
                if ready_path.is_file():
                    ready_obj = load_json(ready_path)
                    ready_obj["manifest_record_sha256"] = record_sha
                    write_ready_descriptor(ready_path, ready_obj)
                manifest_index[sid] = man_row
                shutil.rmtree(staging_dir(binary, SKEL_PHASE, sid), ignore_errors=True)
                _ = path
                n_ok += 1
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                _ledger(
                    runtime,
                    {
                        "sample_id": sid,
                        "phase": SKEL_PHASE,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                n_fail += 1
    finally:
        try:
            engine.close()
        except Exception:
            pass
        rebuild_manifest_index(runtime, SKEL_PHASE)
    summary = {
        "phase": SKEL_PHASE,
        "arm": "C3",
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_resume": n_resume,
        "ok": n_fail == 0 and n_ok == CORPUS_N,
    }
    atomic_replace_json(runtime / "SKEL_PHASE_SUMMARY.json", summary)
    return summary


def run_rgb_phase(
    *,
    binary: Path,
    runtime: Path,
    root: Path,
    resume: bool,
    device: str,
    skeleton_source_root: Path | None = None,
    approval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    det_path = runtime / "DETERMINISTIC_PHASE_FAILURE.json"
    if det_path.is_file():
        existing = load_json(det_path)
        prev = {}
        summary_path = runtime / "RGB_PHASE_SUMMARY.json"
        if summary_path.is_file():
            try:
                prev = load_json(summary_path)
            except (OSError, json.JSONDecodeError):
                prev = {}
        return {
            "phase": RGB_PHASE,
            "n_ok": int(prev.get("n_ok") or existing.get("n_ok_at_halt") or 0),
            "n_fail": int(prev.get("n_fail") or existing.get("n_fail_at_halt") or 0),
            "n_resume": int(prev.get("n_resume") or existing.get("n_resume_at_halt") or 0),
            "ok": False,
            "deterministic_dependency_failure": True,
            "rerun_verify_only": True,
            "error": existing.get("error"),
            "error_type": existing.get("error_type"),
        }

    skel_root = skeleton_source_root or _resolve_skeleton_source_root(
        root=root, binary=binary, approval=approval
    )

    try:
        preflight_rgb_f0_runtime()
    except DeterministicDependencyError as exc:
        return _publish_deterministic_phase_failure(
            runtime=runtime, phase=RGB_PHASE, exc=exc, sample_id=None, n_ok=0, n_fail=0, n_resume=0
        )

    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    identity = load_eligible_identity(root=root)
    meta_by, split_by, signer_by = _load_metadata_maps(root)
    try:
        extractor = FrozenSlowFastBlock4(device=device)
    except ModuleNotFoundError as exc:
        return _publish_deterministic_phase_failure(
            runtime=runtime,
            phase=RGB_PHASE,
            exc=DeterministicDependencyError(str(exc)),
            sample_id=None,
            n_ok=0,
            n_fail=0,
            n_resume=0,
        )
    n_ok = n_fail = n_resume = 0
    manifest_index = load_manifest_index(runtime, RGB_PHASE)
    for sid in identity["eligible_ids"]:
        row = meta_by.get(sid)
        dest = binary / RGB_PHASE / f"{sid}.npz"
        sidecar = binary / RGB_PHASE / f"{sid}.meta.json"
        skel_path = skel_root / SKEL_PHASE / f"{sid}.npz"
        ready_path = staging_dir(binary, RGB_PHASE, sid) / "READY.json"
        try:
            if row is None:
                raise RuntimeError("missing_metadata")
            video = _resolve_video(root, row)
            source_sha = sha256_file(video)
            if not skel_path.is_file():
                raise RuntimeError("missing_skeleton")
            skel_sha = sha256_file(skel_path)
            skip_sha = heal_or_resume_sample(
                dest=dest,
                sidecar=sidecar,
                ready_path=ready_path,
                manifest_index=manifest_index,
                sample_id=sid,
                required_keys=RGB_REQUIRED_KEYS,
                required_manifest_fields=RGB_MANIFEST_REQUIRED_FIELDS,
                resume=resume,
                source_sha=source_sha,
                skel_sha=skel_sha,
            )
            if skip_sha is not None:
                if skip_sha.startswith("HEAL_MANIFEST:"):
                    _heal_manifest_from_ready(
                        binary=binary,
                        runtime=runtime,
                        phase=RGB_PHASE,
                        sample_id=sid,
                        manifest_index=manifest_index,
                    )
                n_resume += 1
                n_ok += 1
                continue
            skel = np.load(skel_path, allow_pickle=False)
            landmarks_by_frame: dict[int, np.ndarray] = {}
            missing_by_frame: dict[int, np.ndarray] = {}
            raw = skel["effective_landmarks"] if "effective_landmarks" in skel.files else skel["joints_raw"]
            miss = skel["effective_missing"] if "effective_missing" in skel.files else skel["missing_mask_prefill"]
            for i, fi in enumerate(skel["original_frame_index"].tolist()):
                landmarks_by_frame[int(fi)] = raw[i]
                missing_by_frame[int(fi)] = miss[i]
            payload = extract_rgb_token_v3_for_video(
                video_path=video,
                extractor=extractor,
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                enable_pose_wrist_prior=False,
                enable_ambiguity_aware_prior=True,
                ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
            )
            output_rel, sidecar_rel = canonical_sample_paths(
                root=root, binary=binary, phase=RGB_PHASE, sample_id=sid
            )
            man_row = {
                "sample_id": sid,
                "split": split_by.get(sid),
                "signer_id": signer_by.get(sid),
                "source_sha256": source_sha,
                "skel_sha256": skel_sha,
                "output_rel": output_rel,
                "sidecar_rel": sidecar_rel,
                "output_sha256": None,
                "sidecar_sha256": None,
                "producer": "scripts/run_ms_sta_v5_d0_full_corpus.py:rgb_token_v3",
            }
            path, meta_path, dig = _save_rgb_atomic(
                binary, sid, payload, manifest_row=man_row, cleanup_staging=False
            )
            man_row["output_sha256"] = dig
            man_row["sidecar_sha256"] = sha256_file(meta_path)
            require_manifest_fields(man_row, RGB_MANIFEST_REQUIRED_FIELDS)
            if sid in manifest_index:
                raise SystemExit(2)
            record_sha = commit_manifest_record(runtime, RGB_PHASE, man_row)
            ready_path = staging_dir(binary, RGB_PHASE, sid) / "READY.json"
            if ready_path.is_file():
                ready_obj = load_json(ready_path)
                ready_obj["manifest_record_sha256"] = record_sha
                write_ready_descriptor(ready_path, ready_obj)
            manifest_index[sid] = man_row
            shutil.rmtree(staging_dir(binary, RGB_PHASE, sid), ignore_errors=True)
            _ = path
            n_ok += 1
        except SystemExit:
            raise
        except ModuleNotFoundError as exc:
            return _publish_deterministic_phase_failure(
                runtime=runtime,
                phase=RGB_PHASE,
                exc=DeterministicDependencyError(str(exc)),
                sample_id=sid,
                n_ok=n_ok,
                n_fail=n_fail,
                n_resume=n_resume,
            )
        except DeterministicDependencyError as exc:
            return _publish_deterministic_phase_failure(
                runtime=runtime,
                phase=RGB_PHASE,
                exc=exc,
                sample_id=sid,
                n_ok=n_ok,
                n_fail=n_fail,
                n_resume=n_resume,
            )
        except Exception as exc:  # noqa: BLE001
            _ledger(
                runtime,
                {
                    "sample_id": sid,
                    "phase": RGB_PHASE,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            n_fail += 1
    rebuild_manifest_index(runtime, RGB_PHASE)
    try:
        skel_rel = str(skel_root.relative_to(root)).replace("\\", "/")
    except ValueError:
        skel_rel = str(skel_root)
    summary = {
        "phase": RGB_PHASE,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_resume": n_resume,
        "ok": n_fail == 0 and n_ok == CORPUS_N,
        "skeleton_source_root_rel": skel_rel,
    }
    atomic_replace_json(runtime / "RGB_PHASE_SUMMARY.json", summary)
    return summary


def run_quality_and_completion(
    *,
    binary: Path,
    runtime: Path,
    root: Path,
    approval: dict,
    skeleton_source_root: Path | None = None,
) -> dict[str, Any]:
    from scan_ms_sta_v5_d0_full_corpus import scan_full_corpus

    evidence = evidence_root_path(root=root)
    if (approval or {}).get("evidence_root_rel"):
        evidence = root / str(approval["evidence_root_rel"])
    evidence.mkdir(parents=True, exist_ok=True)
    seal_path = evidence / "D0_FULL_CORPUS_COMPLETION_SEAL.json"
    intent_path = runtime / "COMPLETION_INTENT.json"
    closed_at: str | None = None
    if seal_path.is_file():
        closed_at = load_json(seal_path).get("closed_at")
    elif intent_path.is_file():
        closed_at = load_json(intent_path).get("closed_at")

    skel_root = skeleton_source_root or _resolve_skeleton_source_root(
        root=root, binary=binary, approval=approval
    )
    out_ver = str(approval.get("output_version") or OUTPUT_VERSION)
    from ms_sta_v5_d0_full_corpus_lib import runtime_dir as _runtime_dir

    dual = skel_root.resolve() != binary.resolve()
    scan = scan_full_corpus(
        binary_root=binary,
        runtime_dir=runtime,
        thresholds=dict(approval.get("quality_thresholds") or {}),
        root=root,
        skeleton_source_root=skel_root if dual else None,
        skeleton_tensor_root=skel_root if dual else None,
        skeleton_manifest_runtime=_runtime_dir(skel_root) if dual else None,
        rgb_tensor_root=binary,
        rgb_manifest_runtime=runtime,
        output_version=out_ver,
    )
    quality_stats = load_json(runtime / "QUALITY_STATS.json")
    verdict = (
        "D0_FULL_CORPUS_SCAN_COMPLETE_PASS"
        if scan.get("ok") is True
        else "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL"
    )
    completion = {
        "schema": f"ms_sta_v5_d0_full_corpus_completion_{out_ver}",
        "output_version": out_ver,
        "verdict": verdict,
        "closed_at": closed_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        "binary_root_rel": str(binary.relative_to(root)).replace("\\", "/"),
    }
    result = publish_completion_pack(
        evidence=evidence,
        runtime=runtime,
        completion=completion,
        scan_report=scan,
        quality_stats=quality_stats,
    )
    mutable = _load_mutable(runtime)
    mutable["status"] = "CLOSED_PASS" if verdict.endswith("_PASS") else "CLOSED_FAIL"
    mutable["phase"] = "completion"
    mutable["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _update_mutable(runtime, mutable)
    return result


def _bind_orchestrator_identity(mutable: dict[str, Any]) -> None:
    mutable["orchestrator_pid"] = os.getpid()
    ident = _process_identity(os.getpid())
    mutable["orchestrator_create_time"] = float(ident["create_time"]) if ident else None


def _run_phase_worker(
    cmd: list[str],
    *,
    runtime: Path,
    phase: str,
    root: Path,
    campaign_token: str = CAMPAIGN_TOKEN,
) -> int:
    logs_dir = runtime / "logs" / phase
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stdout_path = logs_dir / f"{phase}_{stamp}_stdout.log"
    stderr_path = logs_dir / f"{phase}_{stamp}_stderr.log"
    with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
        proc = subprocess.Popen(cmd, cwd=root, stdout=out_f, stderr=err_f)
        register_worker(runtime, pid=proc.pid, phase=phase, campaign_token=campaign_token)
        try:
            code = int(proc.wait())
        finally:
            unregister_worker(runtime, proc.pid, campaign_token=campaign_token)
    if code != 0:
        rel_stdout = str(stdout_path.relative_to(runtime)).replace("\\", "/")
        rel_stderr = str(stderr_path.relative_to(runtime)).replace("\\", "/")
        failure = {
            "schema": "ms_sta_v5_d0_full_corpus_phase_failure_v1",
            "phase": phase,
            "exit_code": int(code),
            "campaign_token": campaign_token,
            "command": list(cmd),
            "command_identity": " ".join(str(x) for x in cmd)[:1000],
            "stdout_rel": rel_stdout,
            "stderr_rel": rel_stderr,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        # Versioned copy + latest pointer for watchdog fail-closed.
        atomic_replace_json(logs_dir / f"PHASE_FAILURE_{stamp}.json", failure)
        atomic_replace_json(runtime / "PHASE_FAILURE.json", failure)
    return code


def _verify_carry_forward_for_orchestrate(
    *,
    approval: dict[str, Any],
    root: Path,
) -> tuple[bool, str | None, list[str]]:
    """Return (ok, audit_sha256, errors). Prefer sealed audit + critical binds; else re-verify."""
    audit_path = root / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
    bind_sha = approval.get("skeleton_carry_forward_audit_sha256")
    if audit_path.is_file():
        try:
            audit = load_json(audit_path)
        except (OSError, json.JSONDecodeError) as exc:
            return False, None, [f"carry_forward_audit_unreadable:{exc}"]
        audit_sha = sha256_file(audit_path)
        errors: list[str] = []
        if audit.get("ok") is not True:
            errors.append("carry_forward_audit_not_ok")
        if audit.get("verify_all_shas") is not True:
            errors.append("carry_forward_audit_verify_all_shas")
        if audit.get("mutated") is True:
            errors.append("carry_forward_audit_mutated")
        n_ok = audit.get("summary_n_ok")
        if n_ok is None or int(n_ok) != CORPUS_N:
            errors.append(f"carry_forward_audit_n_ok:{n_ok}")
        if bind_sha is not None and bind_sha != audit_sha:
            errors.append("carry_forward_audit_sha_bind_mismatch")
        src = str(audit.get("source_root_rel") or "").replace("\\", "/")
        skel_src = str(approval.get("skeleton_source_root_rel") or "").replace("\\", "/")
        if skel_src and src and src != skel_src:
            errors.append("carry_forward_source_root_mismatch")
        if not errors:
            return True, audit_sha, []
        # Fall through to live verify when sealed audit fails critical binds.
    report = verify_skeleton_carry_forward_from_v14(root=root, verify_all_shas=True)
    if report.get("ok") is not True:
        return False, None, list(report.get("errors") or ["carry_forward_verify_failed"])
    audit_sha = sha256_file(audit_path) if audit_path.is_file() else None
    if bind_sha is not None and audit_sha is not None and bind_sha != audit_sha:
        return False, audit_sha, ["carry_forward_audit_sha_bind_mismatch_after_verify"]
    return True, audit_sha, []


def orchestrate(
    *,
    resume: bool,
    device: str,
    root: Path = _ROOT,
    campaign_token: str = CAMPAIGN_TOKEN,
) -> int:
    approval, _protocol, binary, runtime = _require_launch_gates(
        resume=resume, root=root, campaign_token=campaign_token
    )
    mutable = _load_mutable(runtime)
    mutable["status"] = "RUNNING"
    _bind_orchestrator_identity(mutable)
    mutable["campaign_token"] = campaign_token
    mutable["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _update_mutable(runtime, mutable)

    extract_py = root / EXTRACT_PY_REL
    f0_py = root / F0_PY_REL
    runner = str(root / "scripts/run_ms_sta_v5_d0_full_corpus.py")
    carry_forward = bool(approval.get("carry_forward_required"))

    if carry_forward:
        ok, audit_sha, errors = _verify_carry_forward_for_orchestrate(approval=approval, root=root)
        if not ok:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "carry_forward_verify_failed",
                        "errors": errors,
                        "campaign_token": campaign_token,
                    },
                    indent=2,
                )
            )
            return 2
        if not f0_py.is_file():
            raise RuntimeError("missing_isolated_python_envs")
        summary = {
            "schema": "ms_sta_v5_d0_full_corpus_v15_carry_forward_runtime_summary_v1",
            "ok": True,
            "n_ok": CORPUS_N,
            "n_fail": 0,
            "source": "read_only_carry_forward",
            "audit_sha256": audit_sha,
            "campaign_token": campaign_token,
            "output_version": OUTPUT_VERSION_V15,
            "binary_root_rel": str(approval.get("binary_root_rel") or "").replace("\\", "/"),
            "skeleton_source_root_rel": str(approval.get("skeleton_source_root_rel") or "").replace(
                "\\", "/"
            ),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_create_json(runtime / "CARRY_FORWARD_SUMMARY.json", summary)
        mutable["phase"] = "carry_forward_verified"
        _update_mutable(runtime, mutable)
    else:
        if not extract_py.is_file() or not f0_py.is_file():
            raise RuntimeError("missing_isolated_python_envs")
        cmd_skel = [
            str(extract_py),
            runner,
            "--phase",
            SKEL_PHASE,
            "--launch",
            "--bound-worker",
            "--campaign-token",
            campaign_token,
        ]
        if resume:
            cmd_skel.append("--resume")
        mutable["phase"] = SKEL_PHASE
        _update_mutable(runtime, mutable)
        r1 = _run_phase_worker(
            cmd_skel, runtime=runtime, phase=SKEL_PHASE, root=root, campaign_token=campaign_token
        )
        if r1 != 0:
            return r1

    cmd_rgb = [
        str(f0_py),
        runner,
        "--phase",
        RGB_PHASE,
        "--device",
        device,
        "--launch",
        "--bound-worker",
        "--campaign-token",
        campaign_token,
    ]
    if resume:
        cmd_rgb.append("--resume")
    mutable["phase"] = RGB_PHASE
    _update_mutable(runtime, mutable)
    r2 = _run_phase_worker(
        cmd_rgb, runtime=runtime, phase=RGB_PHASE, root=root, campaign_token=campaign_token
    )
    if r2 != 0:
        return r2

    cmd_scan = [
        str(f0_py),
        runner,
        "--phase",
        "quality_scan",
        "--launch",
        "--bound-worker",
        "--campaign-token",
        campaign_token,
    ]
    mutable["phase"] = "quality_scan"
    _update_mutable(runtime, mutable)
    return _run_phase_worker(
        cmd_scan, runtime=runtime, phase="quality_scan", root=root, campaign_token=campaign_token
    )


BOUND_WORKER_PHASES = (SKEL_PHASE, RGB_PHASE, "quality_scan")


def _exit_json(payload: dict[str, Any], code: int) -> int:
    print(json.dumps(payload, indent=2))
    return code


def _run_bound_phase(
    *,
    phase: str,
    approval: dict[str, Any],
    binary: Path,
    runtime: Path,
    root: Path,
    resume: bool,
    device: str,
    is_v15: bool,
) -> int:
    skel_root = _resolve_skeleton_source_root(root=root, binary=binary, approval=approval)
    if phase == SKEL_PHASE:
        if is_v15 and approval.get("carry_forward_required"):
            return _exit_json(
                {
                    "ok": False,
                    "error": "v15_skeleton_reextract_forbidden_carry_forward_required",
                    "skeleton_source_root_rel": str(skel_root.relative_to(root)).replace("\\", "/"),
                },
                2,
            )
        out = run_skeleton_phase(binary=binary, runtime=runtime, root=root, resume=resume)
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1
    if phase == RGB_PHASE:
        out = run_rgb_phase(
            binary=binary,
            runtime=runtime,
            root=root,
            resume=resume,
            device=device,
            skeleton_source_root=skel_root,
            approval=approval,
        )
        print(json.dumps(out, indent=2))
        if out.get("deterministic_dependency_failure"):
            return 3
        return 0 if out["ok"] else 1
    if phase in ("quality_scan", "completion"):
        out = run_quality_and_completion(
            binary=binary,
            runtime=runtime,
            root=root,
            approval=approval,
            skeleton_source_root=skel_root,
        )
        print(json.dumps(out, indent=2))
        return 0 if out["verdict"].endswith("_PASS") else 1
    return 2


def _main_bound_worker(args: argparse.Namespace) -> int:
    """Internal attach path: never re-run cold-start evaluate_launch_readiness*."""
    is_v15 = args.campaign_token == CAMPAIGN_TOKEN_V15
    is_v14 = args.campaign_token == CAMPAIGN_TOKEN
    if not (is_v14 or is_v15):
        return _exit_json({"ok": False, "error": "campaign_token_mismatch", "launch": False}, 2)
    if not args.launch:
        return _exit_json(
            {
                "ok": False,
                "error": "bound_worker_requires_launch",
                "launch": False,
            },
            2,
        )
    if args.phase not in BOUND_WORKER_PHASES:
        return _exit_json(
            {
                "ok": False,
                "error": "bound_worker_invalid_phase",
                "phase": args.phase,
                "allowed_phases": list(BOUND_WORKER_PHASES),
                "launch": False,
            },
            2,
        )
    approval_rel = D0_FULL_CORPUS_APPROVAL_V15_REL if is_v15 else D0_FULL_CORPUS_APPROVAL_V14_REL
    if not (_ROOT / approval_rel).is_file():
        return _exit_json(
            {
                "ok": False,
                "error": "approval_v15_missing" if is_v15 else "approval_v14_missing",
                "launch": False,
                "ready": False,
            },
            2,
        )
    try:
        approval, _protocol, binary, runtime = _require_launch_gates(
            resume=bool(args.resume),
            root=_ROOT,
            bound_worker=True,
            campaign_token=args.campaign_token,
        )
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        return _exit_json(
            {
                "ok": False,
                "error": "bound_worker_contract_identity_mismatch",
                "launch": False,
            },
            code if code else 2,
        )
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return _exit_json(
            {
                "ok": False,
                "error": "bound_worker_attach_failed",
                "detail": str(exc),
                "launch": False,
            },
            2,
        )
    return _run_bound_phase(
        phase=args.phase,
        approval=approval,
        binary=binary,
        runtime=runtime,
        root=_ROOT,
        resume=bool(args.resume),
        device=args.device,
        is_v15=is_v15,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["all", *PHASES_EXACT], default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--bound-worker",
        action="store_true",
        help="Attach to existing immutable _run_contract.json without creating/bumping resumes.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--campaign-token", default=CAMPAIGN_TOKEN)
    ap.add_argument(
        "--launch",
        action="store_true",
        help="Required to perform side-effectful extraction/scan. Absent => contract check only.",
    )
    args = ap.parse_args()

    is_v15 = args.campaign_token == CAMPAIGN_TOKEN_V15
    is_v14 = args.campaign_token == CAMPAIGN_TOKEN
    if not (is_v14 or is_v15):
        return _exit_json({"ok": False, "error": "campaign_token_mismatch", "launch": False}, 2)

    # PATH B — internal bound worker attach (never cold-start readiness).
    if args.bound_worker:
        return _main_bound_worker(args)

    # PATH A — orchestrator / readiness / direct non-bound phase launch.
    if is_v14:
        if not (_ROOT / D0_FULL_CORPUS_APPROVAL_V14_REL).is_file():
            return _exit_json(
                {"ok": False, "error": "approval_v14_missing", "launch": False, "ready": False},
                2,
            )
        readiness = evaluate_launch_readiness(root=_ROOT)
        out_ver = OUTPUT_VERSION
        camp = CAMPAIGN_TOKEN
    else:
        if not (_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL).is_file():
            return _exit_json(
                {"ok": False, "error": "approval_v15_missing", "launch": False, "ready": False},
                2,
            )
        readiness = evaluate_launch_readiness_v15(root=_ROOT)
        out_ver = OUTPUT_VERSION_V15
        camp = CAMPAIGN_TOKEN_V15

    if not args.launch:
        return _exit_json(
            {
                "ok": readiness["ready"],
                "ready": readiness["ready"],
                "launch": False,
                "errors": readiness["errors"],
                "message": (
                    "Launch gates satisfied. Pass --launch to start extraction."
                    if readiness["ready"]
                    else (
                        "Launch blocked: v15 draft — post-audit execution authorization required."
                        if is_v15
                        else "Launch blocked: external re-audit PASS and post-audit execution authorization required."
                    )
                ),
                "output_version": out_ver,
                "campaign_token": camp,
                "binary_root_rel": (readiness.get("approval") or {}).get("binary_root_rel"),
                "skeleton_source_root_rel": (readiness.get("approval") or {}).get(
                    "skeleton_source_root_rel"
                ),
                "phases": list(PHASES_EXACT),
                "skeleton_arm": "C3",
            },
            0 if readiness["ready"] else 2,
        )

    if is_v15:
        # Fail-closed: draft scaffolding never launches full RGB/extraction.
        approval = readiness.get("approval") or {}
        if not readiness["ready"] or approval.get("allow_full_corpus_extraction") is not True:
            return _exit_json(
                {
                    "ok": False,
                    "ready": False,
                    "launch": False,
                    "error": "v15_extraction_not_authorized",
                    "errors": readiness["errors"],
                    "status": approval.get("status"),
                    "allow_full_corpus_extraction": approval.get("allow_full_corpus_extraction"),
                },
                2,
            )

    approval = readiness.get("approval") or {}
    protocol = readiness.get("protocol") or {}
    if is_v14:
        validate_approval_v14(approval, root=_ROOT)
        validate_execution_protocol_v14(protocol, root=_ROOT)
    elif is_v15:
        validate_approval_v15(approval, root=_ROOT)
        validate_execution_protocol_v15(protocol, root=_ROOT)

    if args.phase == "all":
        return orchestrate(
            resume=args.resume,
            device=args.device,
            root=_ROOT,
            campaign_token=args.campaign_token,
        )

    # Direct non-bound phase launch — still requires cold-start empty-root readiness.
    approval, _protocol, binary, runtime = _require_launch_gates(
        resume=args.resume,
        root=_ROOT,
        bound_worker=False,
        campaign_token=args.campaign_token,
    )
    return _run_bound_phase(
        phase=args.phase,
        approval=approval,
        binary=binary,
        runtime=runtime,
        root=_ROOT,
        resume=bool(args.resume),
        device=args.device,
        is_v15=is_v15,
    )


if __name__ == "__main__":
    raise SystemExit(main())
