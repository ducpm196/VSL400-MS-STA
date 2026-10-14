"""One-shot feasibility claim / staging publish / resume-heal (remediation-11).

Seal consumption point is exclusive F0_EXECUTION_ATTEMPT (O_CREAT|O_EXCL).
Artifacts are staged immutably, sealed with READY, then promoted by copy.
Resume may heal promotion from READY+staging; never re-infers.
Report validation enforces architecture load-key/head identity and FP16/FP32 dtypes.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from . import contracts as _contracts
from .contracts import (
    CANDIDATE_IDS,
    CANDIDATE_MEASUREMENT,
    CLASSIFIER_KEY_ALLOWLIST,
    LAYOUT_CONTRACT,
    SLOWFAST_TRAINABLE_BLOCKS,
    any_module_prefix,
    measurement_modes_for,
)
from .gpu_runner import MEASUREMENT

_LATENCY_TOL = 1e-6
_ALLOWED_OUTPUT_DTYPES = frozenset({"float16", "float32"})
_DECLARED_OUTPUT_DTYPE = "float16|float32"
_EXPECTED_HEAD_ATTR = {
    "slowfast_r50_partial_unfreeze": "blocks[-1].proj",
    "videomae_vit_small": "classifier",
    "videomae_vit_base": "classifier",
    "video_swin_tiny": "head",
}


def _staging_names() -> tuple[str, str, str]:
    prefix = _contracts.FEASIBILITY_ARTIFACT_PREFIX
    return (
        f"{prefix}_FEASIBILITY_REPORT.json",
        f"{prefix}_FEASIBILITY_COMPLETION.json",
        f"{prefix}_EXECUTION_SEAL_CONSUMPTION.json",
    )


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_bytes(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def attempt_exists(*, root: Path) -> bool:
    return (root / _contracts.EXECUTION_ATTEMPT_REL).is_file()


def ready_exists(*, root: Path) -> bool:
    return (root / _contracts.FEASIBILITY_READY_REL).is_file()


def seal_already_consumed(*, root: Path) -> bool:
    return (root / _contracts.SEAL_CONSUMPTION_REL).is_file() or attempt_exists(root=root)


def refuse_if_seal_consumed(*, root: Path) -> list[str]:
    reasons: list[str] = []
    if attempt_exists(root=root):
        reasons.append("execution_attempt_already_claimed")
    if (root / _contracts.SEAL_CONSUMPTION_REL).is_file():
        reasons.append("execution_seal_already_consumed")
    return reasons


def attempt_sha256(*, root: Path) -> str:
    return _sha256_file(root / _contracts.EXECUTION_ATTEMPT_REL)


def claim_execution_attempt(
    *,
    root: Path,
    execution_seal_sha256: str,
    executable_tree_sha256: str,
) -> dict[str, Any]:
    """Exclusive create-if-absent claim. This is the seal-consume point before GPU."""
    path = root / _contracts.EXECUTION_ATTEMPT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": "ms_sta_v5_f0_execution_attempt_v1",
        "status": "CLAIMED",
        "execution_seal_sha256": execution_seal_sha256,
        "executable_tree_sha256": executable_tree_sha256,
        "oneshot": True,
        "gpu_inference_authorized_once": True,
        "reuse_forbidden": True,
        "watchdog": {
            "max_restarts_before_attempt": 2,
            "after_attempt": "heal_or_verify_only_no_inference",
        },
    }
    payload = _json_bytes(obj)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(str(path), flags)
    except FileExistsError as exc:
        raise RuntimeError("execution_attempt_already_claimed") from exc
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    digest = _sha256_bytes(payload)
    return {
        "ok": True,
        "attempt_sha256": digest,
        "attempt_rel": _contracts.EXECUTION_ATTEMPT_REL,
    }


def _finite_number(val: Any) -> bool:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _as_float(val: Any) -> float:
    return float(val)


def _latency_stats(all_ms: list[float]) -> dict[str, float]:
    lat_sorted = sorted(all_ms)
    n = len(lat_sorted)
    return {
        "p50_ms": lat_sorted[n // 2],
        "p90_ms": lat_sorted[max(0, int(n * 0.9) - 1)],
        "mean_ms": sum(lat_sorted) / n,
    }


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(_LATENCY_TOL, _LATENCY_TOL * max(abs(a), abs(b)))


def _normalize_dtype(val: Any) -> str:
    s = str(val or "").strip().lower()
    if s.startswith("torch."):
        s = s[len("torch.") :]
    return s


def _as_key_set(items: Any) -> set[str] | None:
    if items is None:
        return set()
    if not isinstance(items, list):
        return None
    out: set[str] = set()
    for item in items:
        if isinstance(item, tuple):
            if not item:
                return None
            out.add(str(item[0]))
        else:
            out.add(str(item))
    return out


def _validate_load_meta(cid: str, load_meta: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(load_meta, dict) or not load_meta:
        return [f"{cid}:load_meta_missing"]
    head = load_meta.get("head")
    expected_head = _EXPECTED_HEAD_ATTR[cid]
    if not isinstance(head, dict):
        errors.append(f"{cid}:load_meta_head_missing")
    else:
        if head.get("head_attr") != expected_head:
            errors.append(f"{cid}:load_meta_head_attr_invalid")
        if int(head.get("new_out_features") or -1) != int(
            CANDIDATE_MEASUREMENT[cid]["num_classes_dummy"]
        ):
            errors.append(f"{cid}:load_meta_head_classes")

    allowed = set(CLASSIFIER_KEY_ALLOWLIST[cid])
    allowed_sorted = sorted(allowed)

    if cid.startswith("videomae"):
        if load_meta.get("loader") != "transformers_local_files_only":
            errors.append(f"{cid}:load_meta_loader")
        info = load_meta.get("loading_info")
        if not isinstance(info, dict):
            errors.append(f"{cid}:load_meta_loading_info_missing")
        else:
            if list(info.get("error_msgs") or []) != []:
                errors.append(f"{cid}:load_meta_error_msgs_nonempty")
            if list(info.get("allowed_keys") or []) != allowed_sorted:
                errors.append(f"{cid}:load_meta_allowed_keys")
            mismatched = _as_key_set(info.get("mismatched_keys"))
            missing = _as_key_set(info.get("missing_keys"))
            unexpected = _as_key_set(info.get("unexpected_keys"))
            if mismatched is None or missing is None or unexpected is None:
                errors.append(f"{cid}:load_meta_keys_type_invalid")
            else:
                if mismatched != allowed:
                    errors.append(f"{cid}:load_meta_mismatched_keys")
                outside = (missing | unexpected | mismatched) - allowed
                if outside:
                    errors.append(f"{cid}:load_meta_keys_outside_allowlist")
    elif cid == "video_swin_tiny":
        if load_meta.get("grad_checkpointing_api") != "f0_features_checkpoint_adapter":
            errors.append(f"{cid}:load_meta_grad_ckpt_api")
        if "missing_keys" not in load_meta or "unexpected_keys" not in load_meta:
            errors.append(f"{cid}:load_meta_keys_missing")
        if list(load_meta.get("allowed_keys") or []) != allowed_sorted:
            errors.append(f"{cid}:load_meta_allowed_keys")
        missing = _as_key_set(load_meta.get("missing_keys"))
        unexpected = _as_key_set(load_meta.get("unexpected_keys"))
        if missing is None or unexpected is None:
            errors.append(f"{cid}:load_meta_keys_type_invalid")
        else:
            outside = (missing | unexpected) - allowed
            if outside:
                errors.append(f"{cid}:load_meta_keys_outside_allowlist")
    elif cid == "slowfast_r50_partial_unfreeze":
        if "missing_keys" not in load_meta or "unexpected_keys" not in load_meta:
            errors.append(f"{cid}:load_meta_keys_missing")
        if list(load_meta.get("allowed_keys") or []) != []:
            errors.append(f"{cid}:load_meta_allowed_keys_must_be_empty")
        missing = _as_key_set(load_meta.get("missing_keys"))
        unexpected = _as_key_set(load_meta.get("unexpected_keys"))
        if missing is None or unexpected is None:
            errors.append(f"{cid}:load_meta_keys_type_invalid")
        elif missing or unexpected:
            errors.append(f"{cid}:load_meta_keys_must_be_empty")
        tb = load_meta.get("trainable_blocks") or {}
        if list(tb.get("frozen") or []) != list(SLOWFAST_TRAINABLE_BLOCKS["frozen"]):
            errors.append(f"{cid}:load_meta_frozen_blocks")
        if list(tb.get("trainable") or []) != list(SLOWFAST_TRAINABLE_BLOCKS["trainable"]):
            errors.append(f"{cid}:load_meta_trainable_blocks")
    return errors

def _validate_trainable_identities(cid: str, c: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    count = c.get("trainable_param_count")
    sample = c.get("trainable_param_identities_sample")
    if not isinstance(count, int) or count < 1:
        errors.append(f"{cid}:trainable_param_count")
    if not isinstance(sample, list) or not sample or any(
        not isinstance(x, str) or not x for x in sample
    ):
        errors.append(f"{cid}:trainable_identities_invalid")
        return errors
    if isinstance(count, int) and count < len(sample):
        errors.append(f"{cid}:trainable_identities_count_mismatch")
    if cid == "slowfast_r50_partial_unfreeze":
        for prefix in SLOWFAST_TRAINABLE_BLOCKS["trainable"]:
            if not any_module_prefix(sample, prefix):
                # sample may truncate; require at least one trainable prefix when sample is full
                if isinstance(count, int) and count <= len(sample):
                    errors.append(f"{cid}:trainable_identities_missing_prefix:{prefix}")
        for prefix in SLOWFAST_TRAINABLE_BLOCKS["frozen"]:
            if any_module_prefix(sample, prefix):
                errors.append(f"{cid}:trainable_identities_frozen_present:{prefix}")
    return errors


def validate_feasibility_report(report: dict[str, Any]) -> list[str]:
    """Single report validator used by publish and resume (math-consistent)."""
    errors: list[str] = []
    if report.get("schema") != "ms_sta_v5_f0_feasibility_cuda_report_v1":
        errors.append("report_bad_schema")
    if report.get("oneshot") is not True:
        errors.append("report_oneshot_required")
    if report.get("network_fetch") is not False:
        errors.append("report_network_fetch_must_be_false")
    if report.get("cpu_forbidden") is not True:
        errors.append("report_cpu_forbidden_required")
    device = str(report.get("device") or "")
    if not (device == "cuda" or device.startswith("cuda:")):
        errors.append("report_device_not_cuda")

    meas = report.get("measurement_contract")
    if meas != MEASUREMENT:
        errors.append("report_measurement_contract_not_exact")

    cands = report.get("candidates")
    if not isinstance(cands, list) or len(cands) != len(CANDIDATE_IDS):
        errors.append("report_candidate_count")
        return errors
    if report.get("candidate_count") != len(CANDIDATE_IDS):
        errors.append("report_candidate_count_field_mismatch")
    ids = [c.get("candidate_id") for c in cands]
    if ids != list(CANDIDATE_IDS):
        errors.append("report_candidate_ids_mismatch")

    timed_steps = int(MEASUREMENT["timed_steps"])
    for c in cands:
        cid = c.get("candidate_id")
        if c.get("ok") is True:
            errors.extend(_validate_ok_candidate(c, timed_steps=timed_steps))
        elif c.get("ok") is False:
            if not c.get("error"):
                errors.append(f"{cid}:fail_missing_error")
            if not c.get("error_type"):
                errors.append(f"{cid}:fail_missing_error_type")
            if not (str(c.get("device") or "").startswith("cuda")):
                errors.append(f"{cid}:fail_device_not_cuda")
        else:
            errors.append(f"{cid}:ok_flag_invalid")

    if "ok" not in report:
        errors.append("report_missing_ok")
    elif report.get("ok") is not all(bool(c.get("ok")) for c in cands):
        errors.append("report_ok_inconsistent_with_candidates")
    return errors


def _validate_ok_candidate(c: dict[str, Any], *, timed_steps: int) -> list[str]:
    errors: list[str] = []
    cid = c.get("candidate_id")
    device = str(c.get("device") or "")
    if not (device == "cuda" or device.startswith("cuda:")):
        errors.append(f"{cid}:device_not_cuda")
    if c.get("offline_only") is not True:
        errors.append(f"{cid}:offline_only_required")
    if c.get("checkpoint_save_load_bit_exact") is not True:
        errors.append(f"{cid}:checkpoint_roundtrip_required")
    errors.extend(_validate_load_meta(cid, c.get("load_meta")))
    errors.extend(_validate_trainable_identities(cid, c))

    expected_layout = LAYOUT_CONTRACT[cid]["layout"]
    if c.get("layout") != expected_layout:
        errors.append(f"{cid}:layout_mismatch")

    expected_c = int(CANDIDATE_MEASUREMENT[cid]["num_classes_dummy"])
    expected_shape = [1, expected_c]
    out_c = c.get("output_tensor_contract") or {}
    if list(out_c.get("shape") or []) != expected_shape:
        errors.append(f"{cid}:output_shape_invalid")
    if out_c.get("dtype") != _DECLARED_OUTPUT_DTYPE:
        errors.append(f"{cid}:output_dtype_invalid")

    expected_modes = measurement_modes_for(cid)
    modes = c.get("modes") or []
    if len(modes) != len(expected_modes):
        errors.append(f"{cid}:mode_count_mismatch")
        return errors
    for got, exp in zip(modes, expected_modes):
        prec = "amp_fp16" if exp["use_amp"] else "fp32"
        if got.get("precision") != prec:
            errors.append(f"{cid}:precision_mismatch")
        gmeta = got.get("grad_checkpointing") or {}
        if bool(gmeta.get("requested")) != bool(exp["grad_checkpointing_requested"]):
            errors.append(f"{cid}:grad_ckpt_requested_mismatch")
        if exp["grad_checkpointing_requested"]:
            if gmeta.get("applied") is not True or gmeta.get("supported") is not True:
                errors.append(f"{cid}:grad_ckpt_not_applied")
            if cid == "video_swin_tiny":
                inv = got.get("checkpoint_invocation_count")
                if not isinstance(inv, int) or inv < 1:
                    errors.append(f"{cid}:checkpoint_invocation_count_required")

        loss = got.get("last_loss")
        if not _finite_number(loss) or _as_float(loss) < 0:
            errors.append(f"{cid}:last_loss_invalid")

        observed = got.get("observed_output") or {}
        if list(observed.get("shape") or []) != expected_shape:
            errors.append(f"{cid}:observed_output_shape_invalid")
        odt = _normalize_dtype(observed.get("dtype"))
        if odt not in _ALLOWED_OUTPUT_DTYPES:
            errors.append(f"{cid}:observed_output_dtype_invalid")

        dist = got.get("step_latency_ms_distribution") or {}
        if dist.get("n") != timed_steps:
            errors.append(f"{cid}:latency_n_mismatch")
        all_ms_raw = dist.get("all_ms")
        if not isinstance(all_ms_raw, list) or len(all_ms_raw) != timed_steps:
            errors.append(f"{cid}:latency_all_ms_missing")
        else:
            if any(not _finite_number(x) or _as_float(x) <= 0 for x in all_ms_raw):
                errors.append(f"{cid}:latency_non_positive")
            else:
                all_ms = [_as_float(x) for x in all_ms_raw]
                stats = _latency_stats(all_ms)
                for key in ("p50_ms", "p90_ms", "mean_ms"):
                    if not _finite_number(dist.get(key)):
                        errors.append(f"{cid}:latency_{key}_invalid")
                    elif not _close(_as_float(dist.get(key)), stats[key]):
                        errors.append(f"{cid}:latency_{key}_inconsistent")
                    elif _as_float(dist.get(key)) <= 0:
                        errors.append(f"{cid}:latency_{key}_non_positive")

        alloc = got.get("peak_allocated_vram_bytes")
        reserved = got.get("peak_reserved_vram_bytes")
        min_free = got.get("min_free_vram_bytes")
        if not isinstance(alloc, int) or alloc <= 0:
            errors.append(f"{cid}:peak_allocated_vram_bytes_invalid")
        if not isinstance(reserved, int) or reserved < 0:
            errors.append(f"{cid}:peak_reserved_vram_bytes_invalid")
        elif isinstance(alloc, int) and reserved < alloc:
            errors.append(f"{cid}:peak_reserved_lt_allocated")
        if not isinstance(min_free, int) or min_free <= 0:
            errors.append(f"{cid}:min_free_vram_bytes_invalid")
    return errors


def validate_feasibility_report_or_raise(report: dict[str, Any]) -> None:
    errors = validate_feasibility_report(report)
    if errors:
        raise ValueError("report_invalid:" + ";".join(errors))


def _promote_from_staging(*, root: Path) -> None:
    staging = root / _contracts.FEASIBILITY_STAGING_DIR_REL
    mapping = (
        (_staging_names()[0], _contracts.FEASIBILITY_REPORT_REL),
        (_staging_names()[1], _contracts.FEASIBILITY_COMPLETION_REL),
        (_staging_names()[2], _contracts.SEAL_CONSUMPTION_REL),
    )
    for src_name, dest_rel in mapping:
        src = staging / src_name
        if not src.is_file():
            raise RuntimeError(f"staging_missing:{src_name}")
        dest = root / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".promote_tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)


def heal_promotion_from_ready(*, root: Path) -> dict[str, Any]:
    """Copy finals from immutable staging using READY digests. No inference."""
    if not ready_exists(root=root):
        raise ValueError("heal_requires_ready")
    if not attempt_exists(root=root):
        raise ValueError("heal_requires_attempt")
    ready = json.loads((root / _contracts.FEASIBILITY_READY_REL).read_text(encoding="utf-8"))
    staging = root / _contracts.FEASIBILITY_STAGING_DIR_REL
    for name, key in (
        (_staging_names()[0], "report_sha256"),
        (_staging_names()[1], "completion_sha256"),
        (_staging_names()[2], "seal_consumption_sha256"),
    ):
        path = staging / name
        if not path.is_file():
            raise ValueError(f"heal_staging_missing:{name}")
        actual = _sha256_file(path)
        if actual != ready.get(key):
            raise ValueError(f"heal_staging_sha_mismatch:{name}")
    _promote_from_staging(root=root)
    return {"ok": True, "healed": True, "inference": False}


def publish_oneshot_artifacts(
    *,
    root: Path,
    report: dict[str, Any],
    execution_seal_sha256: str,
    executable_tree_sha256: str,
) -> dict[str, Any]:
    """Validate report, stage immutably, write READY, promote by copy."""
    if not attempt_exists(root=root):
        raise RuntimeError("execution_attempt_required_before_publish")
    if ready_exists(root=root) or (root / _contracts.SEAL_CONSUMPTION_REL).is_file():
        raise RuntimeError("execution_seal_already_consumed")

    report_obj = {
        **report,
        "schema": report.get("schema") or "ms_sta_v5_f0_feasibility_cuda_report_v1",
        "oneshot": True,
    }
    validate_feasibility_report_or_raise(report_obj)
    attempt_digest = attempt_sha256(root=root)

    staging = root / _contracts.FEASIBILITY_STAGING_DIR_REL
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    report_bytes = _json_bytes(report_obj)
    report_sha = _sha256_bytes(report_bytes)
    (staging / _staging_names()[0]).write_bytes(report_bytes)

    completion = {
        "schema": "ms_sta_v5_f0_feasibility_completion_v1",
        "status": "COMPLETE",
        "report_rel": _contracts.FEASIBILITY_REPORT_REL,
        "report_sha256": report_sha,
        "execution_seal_sha256": execution_seal_sha256,
        "executable_tree_sha256": executable_tree_sha256,
        "execution_attempt_sha256": attempt_digest,
        "oneshot": True,
    }
    completion_bytes = _json_bytes(completion)
    completion_sha = _sha256_bytes(completion_bytes)
    (staging / _staging_names()[1]).write_bytes(completion_bytes)

    consumption = {
        "schema": "ms_sta_v5_f0_execution_seal_consumption_v1",
        "status": "CONSUMED",
        "execution_seal_sha256": execution_seal_sha256,
        "report_sha256": report_sha,
        "completion_sha256": completion_sha,
        "executable_tree_sha256": executable_tree_sha256,
        "execution_attempt_rel": _contracts.EXECUTION_ATTEMPT_REL,
        "execution_attempt_sha256": attempt_digest,
        "oneshot": True,
        "reuse_forbidden": True,
    }
    consumption_bytes = _json_bytes(consumption)
    consumption_sha = _sha256_bytes(consumption_bytes)
    (staging / _staging_names()[2]).write_bytes(consumption_bytes)

    ready = {
        "schema": "ms_sta_v5_f0_feasibility_ready_v1",
        "status": "READY",
        "report_sha256": report_sha,
        "completion_sha256": completion_sha,
        "seal_consumption_sha256": consumption_sha,
        "execution_seal_sha256": execution_seal_sha256,
        "executable_tree_sha256": executable_tree_sha256,
        "execution_attempt_sha256": attempt_digest,
        "staging_dir": _contracts.FEASIBILITY_STAGING_DIR_REL,
        "promotion": "copy_from_staging",
    }
    ready_bytes = _json_bytes(ready)
    ready_path = root / _contracts.FEASIBILITY_READY_REL
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        fd = os.open(str(ready_path), flags)
    except FileExistsError as exc:
        raise RuntimeError("feasibility_ready_already_exists") from exc
    try:
        os.write(fd, ready_bytes)
    finally:
        os.close(fd)

    _promote_from_staging(root=root)

    return {
        "ok": True,
        "report_sha256": report_sha,
        "completion_sha256": completion_sha,
        "seal_consumption_sha256": consumption_sha,
        "ready_sha256": _sha256_bytes(ready_bytes),
        "execution_attempt_sha256": attempt_digest,
        "report_rel": _contracts.FEASIBILITY_REPORT_REL,
        "completion_rel": _contracts.FEASIBILITY_COMPLETION_REL,
        "seal_consumption_rel": _contracts.SEAL_CONSUMPTION_REL,
        "ready_rel": _contracts.FEASIBILITY_READY_REL,
    }


def verify_oneshot_artifacts(
    *,
    root: Path,
    execution_seal_sha256: str,
    executable_tree_sha256: str,
    allow_heal: bool = True,
) -> dict[str, Any]:
    """Resume: heal promotion if needed, then full seal/tree/report validation. No inference."""
    errors: list[str] = []
    if not execution_seal_sha256:
        raise ValueError("resume_requires_execution_seal_sha256")
    if not executable_tree_sha256:
        raise ValueError("resume_requires_executable_tree_sha256")
    if not attempt_exists(root=root):
        raise ValueError("oneshot_missing_execution_attempt")
    if not ready_exists(root=root):
        raise ValueError("oneshot_missing_ready")

    healed = False
    finals_missing = not all(
        (root / rel).is_file()
        for rel in (_contracts.FEASIBILITY_REPORT_REL, _contracts.FEASIBILITY_COMPLETION_REL, _contracts.SEAL_CONSUMPTION_REL)
    )
    if finals_missing:
        if not allow_heal:
            raise ValueError("oneshot_finals_missing_heal_disabled")
        heal_promotion_from_ready(root=root)
        healed = True

    attempt = json.loads((root / _contracts.EXECUTION_ATTEMPT_REL).read_text(encoding="utf-8"))
    attempt_digest = attempt_sha256(root=root)
    if attempt.get("schema") != "ms_sta_v5_f0_execution_attempt_v1":
        errors.append("attempt_bad_schema")
    if attempt.get("execution_seal_sha256") != execution_seal_sha256:
        errors.append("attempt_seal_sha_mismatch")
    if attempt.get("executable_tree_sha256") != executable_tree_sha256:
        errors.append("attempt_executable_tree_sha_mismatch")

    ready = json.loads((root / _contracts.FEASIBILITY_READY_REL).read_text(encoding="utf-8"))
    if ready.get("schema") != "ms_sta_v5_f0_feasibility_ready_v1" or ready.get("status") != "READY":
        errors.append("ready_invalid")
    if ready.get("execution_seal_sha256") != execution_seal_sha256:
        errors.append("ready_seal_sha_mismatch")
    if ready.get("executable_tree_sha256") != executable_tree_sha256:
        errors.append("ready_executable_tree_sha_mismatch")
    if ready.get("execution_attempt_sha256") != attempt_digest:
        errors.append("ready_attempt_sha_mismatch")

    report_path = root / _contracts.FEASIBILITY_REPORT_REL
    completion_path = root / _contracts.FEASIBILITY_COMPLETION_REL
    consumption_path = root / _contracts.SEAL_CONSUMPTION_REL
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors.extend(validate_feasibility_report(report))
    report_sha = _sha256_file(report_path)
    if ready.get("report_sha256") != report_sha:
        errors.append("ready_report_sha_mismatch")

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("schema") != "ms_sta_v5_f0_feasibility_completion_v1":
        errors.append("completion_bad_schema")
    if completion.get("status") != "COMPLETE" or completion.get("oneshot") is not True:
        errors.append("completion_status_invalid")
    if completion.get("report_sha256") != report_sha:
        errors.append("completion_report_sha_mismatch")
    if completion.get("execution_seal_sha256") != execution_seal_sha256:
        errors.append("completion_seal_sha_mismatch")
    if completion.get("executable_tree_sha256") != executable_tree_sha256:
        errors.append("completion_executable_tree_sha_mismatch")
    if completion.get("execution_attempt_sha256") != attempt_digest:
        errors.append("completion_attempt_sha_mismatch")
    completion_sha = _sha256_file(completion_path)
    if ready.get("completion_sha256") != completion_sha:
        errors.append("ready_completion_sha_mismatch")

    consumption = json.loads(consumption_path.read_text(encoding="utf-8"))
    if consumption.get("schema") != "ms_sta_v5_f0_execution_seal_consumption_v1":
        errors.append("consumption_bad_schema")
    if consumption.get("status") != "CONSUMED" or consumption.get("oneshot") is not True:
        errors.append("consumption_status_invalid")
    if consumption.get("report_sha256") != report_sha:
        errors.append("consumption_report_sha_mismatch")
    if consumption.get("completion_sha256") != completion_sha:
        errors.append("consumption_completion_sha_mismatch")
    if consumption.get("execution_seal_sha256") != execution_seal_sha256:
        errors.append("consumption_seal_sha_mismatch")
    if consumption.get("executable_tree_sha256") != executable_tree_sha256:
        errors.append("consumption_executable_tree_sha_mismatch")
    if consumption.get("execution_attempt_sha256") != attempt_digest:
        errors.append("consumption_attempt_sha_mismatch")
    consumption_sha = _sha256_file(consumption_path)
    if ready.get("seal_consumption_sha256") != consumption_sha:
        errors.append("ready_consumption_sha_mismatch")

    if errors:
        raise ValueError(";".join(errors))
    return {
        "ok": True,
        "verify_only": True,
        "healed": healed,
        "inference": False,
        "report_sha256": report_sha,
        "completion_sha256": completion_sha,
        "seal_consumption_sha256": consumption_sha,
        "execution_attempt_sha256": attempt_digest,
        "report_ok": report.get("ok"),
        "watchdog": attempt.get("watchdog"),
    }
