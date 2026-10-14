#!/usr/bin/env python3
"""Shared fail-closed helpers for MS-STA v5 F0 (remediation-6).

Lifecycle:
  PLAN_LOCK → PLAN_AUDIT → SOURCE_RESOLUTION
  → download/env realization → F0_ENV_REALIZATION + REALIZATION_MANIFEST
  → EXECUTION_APPROVAL → EXECUTION_LOCK → EXECUTION_AUDIT → EXECUTION_SEAL → GPU
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_APPROVAL_SHA256 = (
    "da0b047a3051c64b5b24509ec3725caf0d24e4675c4f7f469e4e0bb46d0e3c95"
)
EXPECTED_APPROVAL_COMMIT = "f6920ca2b51d5b2db88dd37e37d4fd8227a3f810"
EXPECTED_APPROVAL_AUDIT_SHA256 = (
    "2676302c942261af52ce0add00372020af07ea308d5b497decb49053f4eac1a8"
)
EXPECTED_APPROVAL_AUDIT_COMMIT = "a209082c1b8d38bee5728550b545bacbe51c1d4f"

# Sealed after remediation-3 plan rewrite (commit 1); PLAN_AUDIT seals in commit 2.
EXPECTED_PROTOCOL_PLAN_SHA256 = (
    "a74535260313f9eda0c9fdf91e968c8bf375f3fd859002b7fd0f6314ba7e3a65"
)
EXPECTED_PROTOCOL_PLAN_LOCK_SHA256 = (
    "7e54810dee4745abf0616a3e8bb5ced7770ebd72d826f1b41907cabe21a1dd41"
)
EXPECTED_REQUIREMENTS_LOCK_SHA256 = (
    "7742d3a25d86e1f328b6b931a2fc23f1fed632b95bb2e553fc30a976866005cb"
)
# Filled in PLAN_AUDIT seal commit; None means audit not yet required only when explicitly disabled.
EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256: str | None = "2da4c8a468a994441efe8dd74bf3ca6ec012250eb74331af1e4261879081b51b"
EXPECTED_PROTOCOL_PLAN_AUDIT_COMMIT: str | None = "19d60bc025f03fb5129d32b1d3c0f1cb7b182c88"

# Sealed companion contracts (remediation-4 bytes; rem-5 binds into auth chain).
EXPECTED_SNAPSHOT_CONTRACT_SHA256 = (
    "af6b2a0341edcd3fedce560a31e6064c7e787a8b7729373853f7e9c3f1e2cd3c"
)
EXPECTED_INDEX_POLICY_SHA256 = (
    "4eec73de67269eeccaade50b623f60908dc4dd0aeaf6aaffd4c71e4ec11c2634"
)

APPROVAL_REL = "Docs/results/ms_sta_v5_f0/F0_APPROVAL.json"
APPROVAL_AUDIT_REL = "Docs/results/ms_sta_v5_f0/F0_APPROVAL_AUDIT.json"
APPROVAL_AUDIT_LOCK_REL = "Docs/results/ms_sta_v5_f0/F0_APPROVAL_AUDIT_LOCK.json"
PROTOCOL_DRAFT_REL = "Docs/results/ms_sta_v5_f0/F0_PROTOCOL_DRAFT.json"
PROTOCOL_PLAN_REL = "Docs/results/ms_sta_v5_f0/F0_PROTOCOL_PLAN.json"
PROTOCOL_PLAN_LOCK_REL = "Docs/results/ms_sta_v5_f0/F0_PROTOCOL_PLAN_LOCK.json"
PROTOCOL_PLAN_AUDIT_REL = "Docs/results/ms_sta_v5_f0/F0_PROTOCOL_PLAN_AUDIT.json"
PROTOCOL_FROZEN_REL = "Docs/results/ms_sta_v5_f0/F0_PROTOCOL.json"
WEIGHTS_REL = "Docs/results/ms_sta_v5_f0/F0_WEIGHTS_MANIFEST.json"
WEIGHTS_DIR_REL = "Docs/results/ms_sta_v5_f0/weights"
REALIZATION_REL = "Docs/results/ms_sta_v5_f0/F0_REALIZATION_MANIFEST.json"
EXECUTION_LOCK_REL = "Docs/results/ms_sta_v5_f0/F0_EXECUTION_LOCK.json"
ENV_SPEC_REL = "Docs/results/ms_sta_v5_f0/F0_ISOLATED_ENV_REQUIREMENTS.json"
ENV_REALIZATION_REL = "Docs/results/ms_sta_v5_f0/F0_ENV_REALIZATION.json"
SNAPSHOT_CONTRACT_REL = "Docs/results/ms_sta_v5_f0/F0_CHECKPOINT_SNAPSHOT_CONTRACT.json"
INDEX_POLICY_REL = "Docs/results/ms_sta_v5_f0/F0_INDEX_POLICY.json"
EXECUTION_APPROVAL_REL = "Docs/results/ms_sta_v5_f0/F0_EXECUTION_APPROVAL.json"
EXECUTION_AUDIT_REL = "Docs/results/ms_sta_v5_f0/F0_EXECUTION_AUDIT.json"
EXECUTION_SEAL_REL = "Docs/results/ms_sta_v5_f0/F0_EXECUTION_SEAL.json"
EVIDENCE_DIR_REL = "Docs/results/ms_sta_v5_f0/evidence"
WHEELHOUSE_DIR_REL = "Docs/results/ms_sta_v5_f0/evidence/wheelhouse"
REQUIREMENTS_LOCK_REL = "Docs/results/ms_sta_v5_f0/F0_REQUIREMENTS.lock"
SCAFFOLD_FINAL_AUDIT_REL = "Docs/results/ms_sta_v5_f0/F0_SCAFFOLD_FINAL_AUDIT.json"
REALIZATION_APPROVAL_REL = "Docs/results/ms_sta_v5_f0/F0_REALIZATION_APPROVAL.json"
REALIZATION_APPROVAL_AUDIT_REL = "Docs/results/ms_sta_v5_f0/F0_REALIZATION_APPROVAL_AUDIT.json"
REALIZATION_MANIFEST_AUDIT_REL = "Docs/results/ms_sta_v5_f0/F0_REALIZATION_MANIFEST_AUDIT.json"
VIDEOMAE_SMALL_CONVERSION_EVIDENCE_REL = (
    "Docs/results/ms_sta_v5_f0/F0_VIDEOMAE_SMALL_CONVERSION_EVIDENCE.json"
)
EXPECTED_SCAFFOLD_FINAL_AUDIT_SHA256 = (
    "bbaefe577ea432a6d1894423885d957ff04a70dc11c7b60059dc41ef367f921e"
)
EXPECTED_REALIZATION_APPROVAL_BASE_COMMIT = (
    "82da583edb02f80e58b009342ddbf412988310ee"
)

# Exact f0 venv path fragments required in python provenance.
F0_VENV_EXEC_SUFFIXES = (
    "_venvs/f0_v5/Scripts/python.exe",
    "_venvs/f0_v5/bin/python",
    "_venvs/f0_v5/bin/python3",
)

# Exact basename required per evidence key (no aliasing / reuse).
ENV_EVIDENCE_BASENAMES = {
    "pip_check_log": "pip_check.log",
    "import_smoke_log": "import_smoke.log",
    "cuda_smoke_log": "cuda_smoke.log",
    "pip_freeze_txt": "pip_freeze.txt",
    "pip_freeze_hashes_json": "pip_freeze.hashes.json",
    "python_provenance_json": "python_provenance.json",
    "cuda_provenance_json": "cuda_provenance.json",
}
REALIZATION_EVIDENCE_BASENAMES = {
    "pip_check_log": "pip_check.log",
    "import_smoke_log": "import_smoke.log",
    "cuda_smoke_log": "cuda_smoke.log",
    "pip_freeze_txt": "pip_freeze.txt",
    "python_provenance_json": "python_provenance.json",
    "cuda_provenance_json": "cuda_provenance.json",
}
# Map top-level SHA fields → evidence key whose recomputed digest must match.
ENV_TOPLEVEL_SHA_BINDINGS = {
    "pip_check_output_sha256": "pip_check_log",
    "import_smoke_output_sha256": "import_smoke_log",
    "cuda_smoke_output_sha256": "cuda_smoke_log",
    "pip_freeze_sha256": "pip_freeze_txt",
}

EXPECTED_CANDIDATES = [
    "slowfast_r50_partial_unfreeze",
    "videomae_vit_small",
    "videomae_vit_base",
    "video_swin_tiny",
]

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_RE = re.compile(
    r"(DRAFT_PIN_AT_FREEZE|OPERATOR-PROVIDED|TBD|TODO|<[^>]+>|SOURCE_GAP)",
    re.IGNORECASE,
)
FORBIDDEN_ACCURACY_KEYS = frozenset(
    {
        "top1",
        "top5",
        "macro_f1",
        "macro-f1",
        "accuracy",
        "top-1",
        "top-5",
        "val_top1",
        "test_top1",
    }
)

EXPECTED_RUNNER = {
    "default_action": "dry-plan",
    "download_requires_plan_lock": True,
    "download_requires_plan_audit": True,
    "download_requires_source_resolution": True,
    "gpu_requires_execution_lock": True,
    "gpu_requires_weights_manifest_verified": True,
    "require_isolated_venv": True,
    "require_clean_worktree": True,
    "max_watchdog_resumes": 2,
    "stop_after": "F0_FEASIBILITY_REPORT",
    "forbid_legacy_F0_PROTOCOL_json": True,
}

EXPECTED_MEASUREMENT_CONTRACT = {
    "batch_size_first": 1,
    "optimizer_step_required": True,
    "require_finite_loss_and_grads": True,
    "warmup_steps": 2,
    "timed_steps": 10,
    "amp_logged_separately": True,
    "grad_checkpointing_logged_separately": True,
    "record_peak_allocated_vram_bytes": True,
    "record_peak_reserved_vram_bytes": True,
    "record_min_free_vram_bytes": True,
    "record_step_latency_ms_distribution": True,
    "checkpoint_save_load_bit_exact": True,
    "output_tensor_contract_required": True,
    "forbid_single_step_latency_claim": True,
}

REQUIRED_PLAN_KEYS = frozenset(
    {
        "schema",
        "status",
        "approval_ref",
        "approval_sha256",
        "bans",
        "runner",
        "measurement_contract",
        "lifecycle",
        "candidates",
        "weights_policy",
        "isolated_env",
        "source_resolution",
    }
)

REQUIRED_WEIGHTS_POLICY_KEYS = frozenset(
    {
        "canonical_dir",
        "manifest_path",
        "realization_manifest_path",
        "status",
        "field_match_required",
        "sha256_policy",
        "path_rules",
    }
)

REQUIRED_ISOLATED_ENV_KEYS = frozenset(
    {
        "requirements_spec",
        "requirements_lock",
        "venv_path_exact",
        "interpreter_exact_win",
        "interpreter_exact_posix",
        "require_pyvenv_cfg",
        "require_include_system_site_packages_false",
        "create_authorized",
        "realization_evidence_forbidden_in_plan",
    }
)

REQUIRED_PLAN_LOCK_KEYS = frozenset(
    {
        "schema",
        "status",
        "plan",
        "plan_sha256",
        "approval_sha256",
        "approval_audit_sha256",
        "env_requirements_sha256",
        "requirements_lock_sha256",
        "candidate_ids",
    }
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_show(commit: str, rel: str, *, root: Path = _ROOT) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=root)


def git_porcelain(root: Path = _ROOT) -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"bom:{path.as_posix()}")
    if b"\r\n" in raw:
        raise ValueError(f"crlf:{path.as_posix()}")
    return json.loads(raw.decode("utf-8"))


def is_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _get_field(obj: dict[str, Any], key: str, *nested_containers: str) -> Any:
    """Return field by key presence (so exit code 0 is preserved)."""
    if key in obj:
        return obj[key]
    for nest in nested_containers:
        sub = obj.get(nest)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return None


def resolve_under_canonical(
    *, root: Path, local_path: str, canonical_dir_rel: str
) -> tuple[Path | None, list[str]]:
    """Resolve path with absolute/dotdot/symlink containment under canonical_dir_rel."""
    errs: list[str] = []
    if not local_path or local_path.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", local_path):
        return None, ["absolute_local_path_forbidden"]
    p = Path(local_path)
    if ".." in p.parts:
        return None, ["dotdot_forbidden"]
    canon = (root / canonical_dir_rel).resolve()
    try:
        resolved = (root / local_path).resolve(strict=False)
    except OSError:
        return None, ["local_path_unresolvable"]
    try:
        resolved.relative_to(canon)
    except ValueError:
        return None, ["resolved_path_outside_canonical_weights_dir"]
    if resolved.is_symlink() or resolved.is_file():
        try:
            final = resolved.resolve(strict=False)
            final.relative_to(canon)
        except ValueError:
            return None, ["symlink_escape_outside_canonical_dir"]
        except OSError:
            return None, ["local_path_unresolvable"]
    return resolved, errs


def resolve_evidence_path(
    *, root: Path, rel: str, expected_basename: str
) -> tuple[Path | None, list[str]]:
    """Canonical evidence path: under evidence/, exact basename, no traversal."""
    errs: list[str] = []
    rel_n = str(rel).replace("\\", "/")
    if not rel_n or rel_n.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", rel_n):
        return None, ["evidence_absolute_path_forbidden"]
    if ".." in Path(rel_n).parts:
        return None, ["evidence_dotdot_forbidden"]
    if not rel_n.startswith(EVIDENCE_DIR_REL + "/"):
        return None, ["evidence_not_under_evidence_dir"]
    if Path(rel_n).name != expected_basename:
        return None, [f"evidence_basename_mismatch:expected={expected_basename}:got={Path(rel_n).name}"]
    resolved, path_errs = resolve_under_canonical(
        root=root, local_path=rel_n, canonical_dir_rel=EVIDENCE_DIR_REL
    )
    # Remap generic weight errors to evidence vocabulary for clarity.
    mapped: list[str] = []
    for e in path_errs:
        if e == "absolute_local_path_forbidden":
            mapped.append("evidence_absolute_path_forbidden")
        elif e == "dotdot_forbidden":
            mapped.append("evidence_dotdot_forbidden")
        elif e == "resolved_path_outside_canonical_weights_dir":
            mapped.append("evidence_resolved_outside_evidence_dir")
        elif e == "symlink_escape_outside_canonical_dir":
            mapped.append("evidence_symlink_escape")
        else:
            mapped.append(e)
    errs.extend(mapped)
    if resolved is None:
        return None, errs
    if not resolved.is_file():
        return None, errs + [f"evidence_file_missing:{rel_n}"]
    return resolved, errs


def _recompute_evidence_sha(
    root: Path, rel: str, *, expected_basename: str
) -> tuple[str, Path]:
    resolved, errs = resolve_evidence_path(
        root=root, rel=rel, expected_basename=expected_basename
    )
    if resolved is None or errs:
        raise ValueError(";".join(errs) if errs else f"evidence_file_missing:{rel}")
    return sha256_file(resolved), resolved


def parse_pip_freeze_txt(text: str) -> dict[str, str]:
    """Return normalized name->version from pip freeze text (ignore blanks/comments)."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"pip_freeze_line_not_pinned:{line}")
        name, ver = line.split("==", 1)
        name_n = name.strip().lower().replace("_", "-")
        ver_n = ver.strip()
        if not name_n or not ver_n:
            raise ValueError(f"pip_freeze_line_identity:{line}")
        if name_n in out and out[name_n] != ver_n:
            raise ValueError(f"pip_freeze_duplicate_conflict:{name_n}")
        out[name_n] = ver_n
    return out


def validate_pip_freeze_hashes(
    obj: dict[str, Any],
    *,
    index_policy: dict[str, Any],
    freeze_txt: str,
    root: Path | None = None,
    recompute_wheel_files: bool = True,
) -> None:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_pip_freeze_hashes_v1":
        errors.append("pip_freeze_hashes_bad_schema")
    try:
        freeze_map = parse_pip_freeze_txt(freeze_txt)
    except ValueError as exc:
        errors.append(str(exc))
        freeze_map = {}
    if len(freeze_map) < 1:
        errors.append("pip_freeze_txt_empty")
    packages = obj.get("packages")
    hash_map: dict[str, str] = {}
    seen_wheel_resolved: dict[str, str] = {}
    if not isinstance(packages, list) or len(packages) < 1:
        errors.append("pip_freeze_hashes_packages_empty")
    else:
        for i, pkg in enumerate(packages):
            if not isinstance(pkg, dict):
                errors.append(f"pip_freeze_hashes_pkg_not_object:{i}")
                continue
            name = str(pkg.get("name") or "").strip().lower().replace("_", "-")
            ver = str(pkg.get("version") or "").strip()
            if not name or not ver:
                errors.append(f"pip_freeze_hashes_pkg_identity:{i}")
                continue
            if name in hash_map:
                errors.append(f"pip_freeze_hashes_duplicate:{name}")
            hash_map[name] = ver
            if not is_hex_sha256(pkg.get("sha256")):
                errors.append(f"pip_freeze_hashes_pkg_sha:{i}")
                continue
            wheel_rel = pkg.get("wheel_rel")
            if not isinstance(wheel_rel, str) or not wheel_rel.strip():
                errors.append(f"pip_freeze_hashes_wheel_rel_missing:{i}:{name}")
                continue
            wheel_n = wheel_rel.replace("\\", "/")
            if not wheel_n.startswith(WHEELHOUSE_DIR_REL + "/"):
                errors.append(f"pip_freeze_hashes_wheel_not_under_wheelhouse:{name}")
                continue
            if ".." in Path(wheel_n).parts:
                errors.append(f"pip_freeze_hashes_wheel_dotdot:{name}")
                continue
            wheel_name = Path(wheel_n).name.lower()
            name_token = name.replace("-", "_")
            ver_token = ver.lower().replace("+", "_")
            # Wheel filename must encode package identity (PEP 427-ish).
            if not (
                wheel_name.startswith(f"{name}-")
                or wheel_name.startswith(f"{name_token}-")
            ):
                errors.append(f"pip_freeze_hashes_wheel_name_package_mismatch:{name}")
            if ver.lower() not in wheel_name and ver_token not in wheel_name:
                errors.append(f"pip_freeze_hashes_wheel_name_version_mismatch:{name}")
            if recompute_wheel_files:
                if root is None:
                    errors.append("pip_freeze_hashes_root_required_for_wheel_recompute")
                else:
                    resolved, path_errs = resolve_under_canonical(
                        root=root, local_path=wheel_n, canonical_dir_rel=WHEELHOUSE_DIR_REL
                    )
                    if path_errs or resolved is None or not resolved.is_file():
                        errors.append(f"pip_freeze_hashes_wheel_missing:{name}:{path_errs}")
                    else:
                        rkey = str(resolved.resolve())
                        if rkey in seen_wheel_resolved:
                            errors.append(
                                "pip_freeze_hashes_duplicate_resolved_wheel:"
                                f"{name}:also={seen_wheel_resolved[rkey]}"
                            )
                        else:
                            seen_wheel_resolved[rkey] = name
                        actual = sha256_file(resolved)
                        if actual != pkg.get("sha256"):
                            errors.append(
                                f"pip_freeze_hashes_wheel_sha_mismatch:{name}:actual={actual}"
                            )
    if freeze_map and hash_map and freeze_map != hash_map:
        only_freeze = sorted(set(freeze_map) - set(hash_map))
        only_hash = sorted(set(hash_map) - set(freeze_map))
        ver_mismatch = sorted(
            k for k in set(freeze_map) & set(hash_map) if freeze_map[k] != hash_map[k]
        )
        if only_freeze or only_hash or ver_mismatch:
            errors.append(
                "pip_freeze_set_mismatch:"
                f"only_freeze={only_freeze}:only_hash={only_hash}:ver_mismatch={ver_mismatch}"
            )
    torch_idx = (index_policy.get("torch_index") or {})
    if obj.get("torch_index_url") != torch_idx.get("extra_index_url"):
        errors.append("pip_freeze_hashes_torch_index_mismatch")
    if obj.get("cuda_build") != torch_idx.get("cuda_build"):
        errors.append("pip_freeze_hashes_cuda_build_mismatch")
    if errors:
        raise ValueError(";".join(errors))


def validate_python_provenance(
    obj: dict[str, Any], *, python_exact: str
) -> None:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_python_provenance_v1":
        errors.append("python_provenance_bad_schema")
    if obj.get("python_exact") != python_exact:
        errors.append("python_provenance_python_exact_mismatch")
    for key in ("executable", "platform", "implementation", "sys_prefix"):
        val = obj.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"python_provenance_missing:{key}")
    exe = str(obj.get("executable") or "").replace("\\", "/")
    if not any(exe.endswith(suf) or suf in exe for suf in F0_VENV_EXEC_SUFFIXES):
        errors.append("python_provenance_executable_not_f0_venv")
    prefix = str(obj.get("sys_prefix") or "").replace("\\", "/")
    if "_venvs/f0_v5" not in prefix:
        errors.append("python_provenance_sys_prefix_not_f0_venv")
    if obj.get("include_system_site_packages") is not False:
        errors.append("python_provenance_include_system_site_packages_not_false")
    if obj.get("pyvenv_cfg_present") is not True:
        errors.append("python_provenance_pyvenv_cfg_required")
    if errors:
        raise ValueError(";".join(errors))


def validate_cuda_provenance(
    obj: dict[str, Any], *, index_policy: dict[str, Any]
) -> None:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_cuda_provenance_v1":
        errors.append("cuda_provenance_bad_schema")
    torch_idx = index_policy.get("torch_index") or {}
    if obj.get("cuda_build") != torch_idx.get("cuda_build"):
        errors.append("cuda_provenance_cuda_build_mismatch")
    if obj.get("torch_index_url") != torch_idx.get("extra_index_url"):
        errors.append("cuda_provenance_torch_index_mismatch")
    for key in ("torch_version", "cuda_runtime_version", "gpu_name"):
        val = obj.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"cuda_provenance_missing:{key}")
    if obj.get("cuda_available") is not True:
        errors.append("cuda_provenance_cuda_available_required")
    if not isinstance(obj.get("gpu_index"), int) or obj.get("gpu_index") < 0:
        errors.append("cuda_provenance_gpu_index_required")
    if errors:
        raise ValueError(";".join(errors))


def bind_evidence_map(
    *,
    root: Path,
    evidence: dict[str, Any],
    basenames: dict[str, str],
    prefix: str,
) -> tuple[dict[str, str], list[str]]:
    """Resolve evidence paths, enforce exact basename + unique resolved files, recompute SHAs."""
    errors: list[str] = []
    actual_shas: dict[str, str] = {}
    resolved_by_key: dict[str, Path] = {}
    seen_resolved: dict[str, str] = {}
    for key, fname in basenames.items():
        rel = evidence.get(key)
        if not rel:
            errors.append(f"{prefix}_missing_evidence:{key}")
            continue
        try:
            digest, resolved = _recompute_evidence_sha(
                root, str(rel).replace("\\", "/"), expected_basename=fname
            )
        except ValueError as exc:
            errors.append(f"{prefix}_{exc}")
            continue
        resolved_key = str(resolved.resolve())
        if resolved_key in seen_resolved:
            errors.append(
                f"{prefix}_duplicate_resolved_evidence:{key}:also={seen_resolved[resolved_key]}"
            )
        else:
            seen_resolved[resolved_key] = key
        declared = evidence.get(f"{key}_sha256")
        if declared != digest:
            errors.append(f"{prefix}_evidence_sha_mismatch:{key}")
        actual_shas[key] = digest
        resolved_by_key[key] = resolved
    return actual_shas, errors


def find_forbidden_accuracy_keys(obj: Any, *, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_l = str(k).lower()
            here = f"{path}.{k}"
            if path.startswith("$.bans") and key_l in FORBIDDEN_ACCURACY_KEYS:
                continue
            if key_l in FORBIDDEN_ACCURACY_KEYS:
                hits.append(here)
            hits.extend(find_forbidden_accuracy_keys(v, path=here))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(find_forbidden_accuracy_keys(v, path=f"{path}[{i}]"))
    return hits


def executable_tree_sha(*, root: Path = _ROOT) -> str:
    """Hash tracked executable surface relevant to F0 runners/validators/feasibility."""
    rels = sorted(
        [
            "scripts/ms_sta_v5_f0_lib.py",
            "scripts/run_ms_sta_v5_f0.py",
            "scripts/run_ms_sta_v5_f0_feasibility.py",
            "scripts/run_ms_sta_v5_f0_r1_feasibility.py",
            "scripts/create_ms_sta_v5_f0_execution_dag.py",
            "scripts/create_ms_sta_v5_f0_r1_execution_dag.py",
            "scripts/remediate_ms_sta_v5_f0_r1.py",
            "scripts/realize_ms_sta_v5_f0.py",
            "scripts/f0_feasibility/__init__.py",
            "scripts/f0_feasibility/contracts.py",
            "scripts/f0_feasibility/campaign_paths.py",
            "scripts/f0_feasibility/preprocess.py",
            "scripts/f0_feasibility/gpu_runner.py",
            "scripts/f0_feasibility/oneshot.py",
            "scripts/validate_ms_sta_v5_f0_approval.py",
            "scripts/validate_ms_sta_v5_f0_protocol.py",
            "scripts/validate_ms_sta_v5_f0_weights_manifest.py",
            "scripts/promote_ms_sta_v5_f0_protocol.py",
        ]
    )
    h = hashlib.sha256()
    for rel in rels:
        data = (root / rel).read_bytes()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_bytes(data).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _parse_pyvenv_cfg(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def require_isolated_interpreter(python_exe: str, *, root: Path = _ROOT) -> list[str]:
    errs: list[str] = []
    expected = {
        (root / "_venvs/f0_v5/Scripts/python.exe").resolve(),
        (root / "_venvs/f0_v5/bin/python").resolve(),
        (root / "_venvs/f0_v5/bin/python3").resolve(),
    }
    try:
        got = Path(python_exe).resolve()
    except OSError:
        return [f"interpreter_unresolvable:{python_exe}"]
    if got not in expected:
        errs.append(f"interpreter_not_exact_f0_venv:{got.as_posix()}")
    # Reject substring spoofs even if somehow equalized
    low = got.as_posix().lower()
    if "_venvs/f0_v5_evil" in low or "/train_cuda/" in low or "/demo/" in low:
        errs.append("interpreter_forbidden_token")
    venv_root = (root / "_venvs/f0_v5").resolve()
    cfg = venv_root / "pyvenv.cfg"
    if not cfg.is_file():
        errs.append("pyvenv_cfg_missing")
    else:
        parsed = _parse_pyvenv_cfg(cfg)
        if parsed.get("include-system-site-packages", "true").lower() != "false":
            errs.append("include_system_site_packages_not_false")
    # When checking the running interpreter, sys.prefix must match venv root.
    try:
        if Path(sys.executable).resolve() == got:
            if Path(sys.prefix).resolve() != venv_root:
                errs.append(f"sys_prefix_mismatch:{Path(sys.prefix).resolve().as_posix()}")
    except OSError:
        errs.append("sys_prefix_check_failed")
    return errs


def detect_forbidden_interpreter(python_exe: str) -> list[str]:
    low = python_exe.replace("\\", "/").lower()
    errs: list[str] = []
    for token in ("/train_cuda/", "/demo/"):
        if token in low:
            errs.append(f"python_in_forbidden_env:{token.strip('/')}")
    return errs


# --- Approval / audit (unchanged sealed identity) ---


def validate_approval(
    *,
    root: Path = _ROOT,
    approval_path: Path | None = None,
    require_sealed_identity: bool = True,
) -> dict[str, Any]:
    approval_path = approval_path or (root / APPROVAL_REL)
    if not approval_path.is_file():
        raise ValueError(f"missing_approval:{approval_path}")
    raw = approval_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("approval_has_bom")
    if b"\r\n" in raw:
        raise ValueError("approval_has_crlf")
    obj = json.loads(raw.decode("utf-8"))
    digest = sha256_bytes(raw)
    errors: list[str] = []
    if require_sealed_identity:
        try:
            rel = approval_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = approval_path.as_posix()
        if rel != APPROVAL_REL:
            errors.append(f"noncanonical_approval_path:{rel}")
        if digest != EXPECTED_APPROVAL_SHA256:
            errors.append(
                f"approval_sha256_mismatch:got={digest}:expected={EXPECTED_APPROVAL_SHA256}"
            )
        sealed = sha256_bytes(git_show(EXPECTED_APPROVAL_COMMIT, APPROVAL_REL, root=root))
        if sealed != EXPECTED_APPROVAL_SHA256 or digest != sealed:
            errors.append("approval_bytes_drift_from_seal_commit")
    if obj.get("schema") != "ms_sta_v5_f0_approval_v1":
        errors.append("bad_schema")
    if obj.get("decision") != "APPROVED" or obj.get("one_shot") is not True:
        errors.append("decision_or_one_shot")
    if obj.get("scope") != "protocol_and_backbone_feasibility_only":
        errors.append("bad_scope")
    if obj.get("candidate_ids") != EXPECTED_CANDIDATES:
        errors.append("candidate_ids_mismatch")
    for key in (
        "allow_accuracy",
        "allow_d0",
        "allow_training",
        "allow_hpo",
        "allow_test",
        "allow_auto_chain",
    ):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    src = obj.get("r0_source_commit")
    seal = obj.get("r0_seal_commit")
    base = obj.get("approval_base_commit")
    doc_sha = sha256_bytes(git_show(str(src), str(obj["r0_document"]), root=root))
    lock_sha = sha256_bytes(git_show(str(seal), str(obj["r0_lock"]), root=root))
    audit_sha = sha256_bytes(git_show(str(base), str(obj["r0_audit"]), root=root))
    if doc_sha != obj.get("r0_document_sha256"):
        errors.append("r0_document_sha256_mismatch")
    if lock_sha != obj.get("r0_lock_sha256"):
        errors.append("r0_lock_sha256_mismatch")
    if audit_sha != obj.get("r0_audit_sha256"):
        errors.append("r0_audit_sha256_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {
        "ok": True,
        "approval_sha256": digest,
        "approval_commit_expected": EXPECTED_APPROVAL_COMMIT,
        "head": head,
        "r0_document_sha256": doc_sha,
        "r0_lock_sha256": lock_sha,
        "r0_audit_sha256": audit_sha,
        "candidate_ids": list(obj["candidate_ids"]),
        "scope": obj["scope"],
        "one_shot": True,
    }


def validate_approval_audit(*, root: Path = _ROOT) -> dict[str, Any]:
    path = root / APPROVAL_AUDIT_REL
    lock_path = root / APPROVAL_AUDIT_LOCK_REL
    if not path.is_file() or not lock_path.is_file():
        raise ValueError("missing_approval_audit_or_lock")
    obj = load_json(path)
    lock = load_json(lock_path)
    digest = sha256_file(path)
    errors: list[str] = []
    if digest != EXPECTED_APPROVAL_AUDIT_SHA256:
        errors.append("approval_audit_sha256_mismatch")
    if lock.get("audit_sha256") != EXPECTED_APPROVAL_AUDIT_SHA256:
        errors.append("approval_audit_lock_sha_mismatch")
    sealed = sha256_bytes(
        git_show(EXPECTED_APPROVAL_AUDIT_COMMIT, APPROVAL_AUDIT_REL, root=root)
    )
    if sealed != EXPECTED_APPROVAL_AUDIT_SHA256:
        errors.append("approval_audit_seal_commit_bytes_unexpected")
    if obj.get("audit_verdict") != "PASS":
        errors.append("approval_audit_verdict_not_pass")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "approval_audit_sha256": digest, "audit_verdict": "PASS"}


def validate_protocol_draft(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != "ms_sta_v5_f0_protocol_draft_v1":
        raise ValueError("bad_draft_schema")
    if protocol.get("status") != "DRAFT" or protocol.get("frozen") is not False:
        raise ValueError("draft_must_remain_draft")
    ids = [c.get("id") for c in (protocol.get("candidates") or [])]
    if ids != EXPECTED_CANDIDATES:
        raise ValueError("candidate_ids_mismatch")
    return {"ok": True, "mode": "draft", "candidate_ids": ids}


def validate_protocol_plan(plan: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    missing = sorted(REQUIRED_PLAN_KEYS - set(plan.keys()))
    if missing:
        errors.append("plan_missing_keys:" + ",".join(missing))
    if plan.get("schema") != "ms_sta_v5_f0_protocol_plan_v1":
        errors.append("bad_plan_schema")
    if plan.get("status") != "PLAN":
        errors.append("plan_status_required")
    if plan.get("approval_sha256") != EXPECTED_APPROVAL_SHA256:
        errors.append("plan_approval_sha_mismatch")

    runner = plan.get("runner")
    if not isinstance(runner, dict) or not runner:
        errors.append("runner_empty_or_missing")
    elif runner != EXPECTED_RUNNER:
        errors.append("runner_not_exact_expected")
    if any(k in (runner or {}) for k in ("gpu_requires_protocol_frozen", "download_requires_protocol_frozen")):
        errors.append("legacy_protocol_frozen_flags_forbidden")

    meas = plan.get("measurement_contract")
    if not isinstance(meas, dict) or not meas:
        errors.append("measurement_contract_empty_or_missing")
    elif meas != EXPECTED_MEASUREMENT_CONTRACT:
        errors.append("measurement_contract_not_exact_expected")

    wp = plan.get("weights_policy") or {}
    if not isinstance(wp, dict) or not wp:
        errors.append("weights_policy_empty_or_missing")
    else:
        miss_wp = sorted(REQUIRED_WEIGHTS_POLICY_KEYS - set(wp.keys()))
        if miss_wp:
            errors.append("weights_policy_missing_keys:" + ",".join(miss_wp))
        if wp.get("sha256_policy") != "sha256_realized_and_recomputed":
            errors.append("weights_sha256_policy")
        if "sha256" in (wp.get("field_match_required") or []):
            errors.append("field_match_must_not_list_sha256_pre_download")

    iso = plan.get("isolated_env") or {}
    if not isinstance(iso, dict) or not iso:
        errors.append("isolated_env_empty_or_missing")
    else:
        miss_iso = sorted(REQUIRED_ISOLATED_ENV_KEYS - set(iso.keys()))
        if miss_iso:
            errors.append("isolated_env_missing_keys:" + ",".join(miss_iso))

    bans = plan.get("bans") or {}
    for k in ("top1", "top5", "macro_f1", "accuracy_any", "d0", "auto_chain"):
        if bans.get(k) is not True:
            errors.append(f"ban_missing:{k}")
    errors.extend(find_forbidden_accuracy_keys(plan))

    sr = plan.get("source_resolution") or {}
    if sr.get("status") != "RESOLVED_FOR_PLAN":
        errors.append("source_resolution_not_resolved")
    if sr.get("realization_authorized") is not False:
        errors.append("source_resolution_must_not_authorize_realization")
    if not is_hex_sha256(sr.get("requirements_lock_sha256")):
        errors.append("source_resolution_requirements_lock_sha")
    pkgs = sr.get("packages") or {}
    for name in ("torch", "torchvision", "pytorchvideo", "transformers"):
        ver = (pkgs.get(name) or {}).get("version")
        if not ver or PLACEHOLDER_RE.search(str(ver)):
            errors.append(f"source_resolution_package_unpinned:{name}")

    cands = plan.get("candidates") or []
    ids = [c.get("id") for c in cands]
    if ids != EXPECTED_CANDIDATES:
        errors.append("candidate_ids_mismatch")
    for c in cands:
        cid = c.get("id")
        ckpt = c.get("pretrained_checkpoint") or {}
        impl = c.get("implementation") or {}
        inp = c.get("input") or {}
        if ckpt.get("provenance_status") != "SOURCE_GAP":
            errors.append(f"{cid}:checkpoint_file_sha_still_source_gap_required")
        if ckpt.get("sha256") is not None or ckpt.get("expected_sha256") is not None:
            errors.append(f"{cid}:plan_must_keep_file_sha_null")
        if not ckpt.get("url"):
            errors.append(f"{cid}:url_required_after_source_resolution")
        pin = impl.get("package_version_pin")
        if not pin or PLACEHOLDER_RE.search(str(pin)) or pin == "SOURCE_GAP":
            errors.append(f"{cid}:package_version_unresolved")
        ref = impl.get("repository_ref_pin")
        if not ref or PLACEHOLDER_RE.search(str(ref)):
            errors.append(f"{cid}:repository_ref_unresolved")
        if c.get("family") == "videomae":
            if inp.get("layout") != "NTCHW":
                errors.append(f"{cid}:videomae_layout")
            if ckpt.get("license") != "CC-BY-NC-4.0" or not ckpt.get("revision"):
                errors.append(f"{cid}:videomae_license_or_revision")
        if c.get("family") == "video_swin":
            if impl.get("api") != "torchvision.models.video.swin3d_t":
                errors.append(f"{cid}:video_swin_api")

    life = plan.get("lifecycle") or {}
    if life.get("forbids_single_frozen_artifact_for_plan_and_evidence") is not True:
        errors.append("lifecycle_must_forbid_single_frozen_artifact")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "mode": "plan", "candidate_ids": ids}


def validate_protocol_plan_lock(
    lock: dict[str, Any],
    *,
    root: Path,
    plan_sha256: str | None = None,
    env_sha256: str | None = None,
    require_expected_identity: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    missing = sorted(REQUIRED_PLAN_LOCK_KEYS - set(lock.keys()))
    if missing:
        errors.append("plan_lock_missing_keys:" + ",".join(missing))
    if lock.get("schema") != "ms_sta_v5_f0_protocol_plan_lock_v1":
        errors.append("bad_plan_lock_schema")
    if lock.get("status") != "PLAN_LOCKED":
        errors.append("plan_lock_status")
    actual_plan_sha = sha256_file(root / PROTOCOL_PLAN_REL)
    actual_lock_sha = sha256_file(root / PROTOCOL_PLAN_LOCK_REL)
    actual_env_sha = sha256_file(root / ENV_SPEC_REL)
    actual_req_sha = sha256_file(root / REQUIREMENTS_LOCK_REL)
    if plan_sha256 is None:
        plan_sha256 = actual_plan_sha
    if env_sha256 is None:
        env_sha256 = actual_env_sha
    if lock.get("plan_sha256") != plan_sha256 or plan_sha256 != actual_plan_sha:
        errors.append("plan_lock_plan_sha_mismatch")
    if lock.get("env_requirements_sha256") != env_sha256 or env_sha256 != actual_env_sha:
        errors.append("plan_lock_env_sha_mismatch")
    if lock.get("requirements_lock_sha256") != actual_req_sha:
        errors.append("plan_lock_requirements_sha_mismatch")
    if lock.get("approval_sha256") != EXPECTED_APPROVAL_SHA256:
        errors.append("plan_lock_approval_sha_mismatch")
    if lock.get("approval_audit_sha256") != EXPECTED_APPROVAL_AUDIT_SHA256:
        errors.append("plan_lock_audit_sha_mismatch")
    if lock.get("candidate_ids") != EXPECTED_CANDIDATES:
        errors.append("plan_lock_candidate_ids_mismatch")
    if require_expected_identity:
        if actual_plan_sha != EXPECTED_PROTOCOL_PLAN_SHA256:
            errors.append(
                f"plan_sha_not_immutable_expected:got={actual_plan_sha}:expected={EXPECTED_PROTOCOL_PLAN_SHA256}"
            )
        if actual_lock_sha != EXPECTED_PROTOCOL_PLAN_LOCK_SHA256:
            errors.append(
                f"plan_lock_sha_not_immutable_expected:got={actual_lock_sha}:expected={EXPECTED_PROTOCOL_PLAN_LOCK_SHA256}"
            )
        if actual_req_sha != EXPECTED_REQUIREMENTS_LOCK_SHA256:
            errors.append("requirements_lock_sha_not_immutable_expected")
    if errors:
        raise ValueError(";".join(errors))
    return {
        "ok": True,
        "plan_sha256": actual_plan_sha,
        "plan_lock_sha256": actual_lock_sha,
    }


def validate_protocol_plan_audit(
    audit: dict[str, Any],
    *,
    root: Path,
    require_expected_identity: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if audit.get("schema") != "ms_sta_v5_f0_protocol_plan_audit_v1":
        errors.append("bad_plan_audit_schema")
    if audit.get("audit_verdict") != "PASS":
        errors.append("plan_audit_verdict")
    if audit.get("plan_sha256") != EXPECTED_PROTOCOL_PLAN_SHA256:
        errors.append("plan_audit_plan_sha_mismatch")
    if audit.get("plan_lock_sha256") != EXPECTED_PROTOCOL_PLAN_LOCK_SHA256:
        errors.append("plan_audit_lock_sha_mismatch")
    if not audit.get("source_commit"):
        errors.append("plan_audit_missing_source_commit")
    if audit.get("realization_authorized") is not False:
        errors.append("plan_audit_must_not_authorize_realization")
    if audit.get("gpu_authorized") is not False:
        errors.append("plan_audit_must_not_authorize_gpu")
    digest = sha256_file(root / PROTOCOL_PLAN_AUDIT_REL)
    if require_expected_identity:
        if EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256 is None:
            errors.append("plan_audit_expected_sha_not_sealed_yet")
        elif digest != EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256:
            errors.append("plan_audit_sha_mismatch_vs_expected")
        if EXPECTED_PROTOCOL_PLAN_AUDIT_COMMIT:
            sealed = sha256_bytes(
                git_show(EXPECTED_PROTOCOL_PLAN_AUDIT_COMMIT, PROTOCOL_PLAN_AUDIT_REL, root=root)
            )
            if sealed != digest:
                errors.append("plan_audit_bytes_drift_from_seal_commit")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "plan_audit_sha256": digest}


def validate_env_spec(spec: dict[str, Any], *, mode: str = "plan") -> dict[str, Any]:
    """Validate immutable env SPEC only. Realized evidence uses validate_env_realization()."""
    errors: list[str] = []
    if mode != "plan":
        raise ValueError("env_spec_mode_must_be_plan_use_validate_env_realization")
    if spec.get("schema") != "ms_sta_v5_f0_isolated_env_requirements_v1":
        errors.append("bad_env_schema")
    if spec.get("install_authorized") is not False:
        errors.append("env_install_must_be_false_until_authorized")
    binding = spec.get("protocol_binding") or {}
    if not binding.get("protocol_plan") or not binding.get("requirements_lock"):
        errors.append("env_missing_plan_or_requirements_binding")
    venv = spec.get("venv") or {}
    for k in ("python_exact", "cuda_build", "requirements_sha256", "pip_freeze_sha256"):
        if k in venv or k in spec:
            if venv.get(k) not in (None, "") or spec.get(k) not in (None, ""):
                if venv.get(k) or spec.get(k):
                    errors.append(f"plan_env_must_not_embed:{k}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "mode": "plan"}


def validate_env_realization(
    obj: dict[str, Any],
    *,
    root: Path,
    plan_sha256: str,
    env_spec_sha256: str,
    recompute_files: bool = True,
) -> dict[str, Any]:
    """Validate F0_ENV_REALIZATION.json with evidence-file SHA recomputation."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_env_realization_v1":
        errors.append("bad_env_realization_schema")
    if obj.get("status") != "REALIZED":
        errors.append("env_realization_status")
    if obj.get("plan_sha256") != plan_sha256:
        errors.append("env_realization_plan_sha_mismatch")
    if obj.get("env_spec_sha256") != env_spec_sha256:
        errors.append("env_realization_env_spec_sha_mismatch")
    if obj.get("index_policy_sha256") != EXPECTED_INDEX_POLICY_SHA256:
        errors.append("env_realization_index_policy_sha_mismatch")
    if obj.get("snapshot_contract_sha256") != EXPECTED_SNAPSHOT_CONTRACT_SHA256:
        errors.append("env_realization_snapshot_contract_sha_mismatch")
    if obj.get("requirements_lock_sha256") != EXPECTED_REQUIREMENTS_LOCK_SHA256:
        errors.append("env_realization_requirements_lock_sha_mismatch")

    index_policy_path = root / INDEX_POLICY_REL
    if not index_policy_path.is_file():
        errors.append("index_policy_missing")
        index_policy: dict[str, Any] = {}
    else:
        if sha256_file(index_policy_path) != EXPECTED_INDEX_POLICY_SHA256:
            errors.append("index_policy_bytes_sha_mismatch")
        index_policy = load_json(index_policy_path)

    evidence = obj.get("evidence") or {}
    actual_shas: dict[str, str] = {}
    if recompute_files:
        actual_shas, ev_errs = bind_evidence_map(
            root=root,
            evidence=evidence,
            basenames=ENV_EVIDENCE_BASENAMES,
            prefix="env_realization",
        )
        errors.extend(ev_errs)
    else:
        for key in ENV_EVIDENCE_BASENAMES:
            if not evidence.get(key):
                errors.append(f"env_realization_missing_evidence:{key}")

    for key in (
        "pip_check_exit_code",
        "import_smoke_exit_code",
        "cuda_smoke_exit_code",
    ):
        val = _get_field(obj, key)
        if val != 0:
            errors.append(f"env_realization_bad_{key}")

    for top_key, ev_key in ENV_TOPLEVEL_SHA_BINDINGS.items():
        val = _get_field(obj, top_key)
        if not is_hex_sha256(val):
            errors.append(f"env_realization_bad_{top_key}")
        elif recompute_files and ev_key in actual_shas and val != actual_shas[ev_key]:
            errors.append(f"env_realization_toplevel_sha_unbound:{top_key}")

    python_exact = _get_field(obj, "python_exact")
    cuda_build = _get_field(obj, "cuda_build")
    if not isinstance(python_exact, str) or not python_exact.strip():
        errors.append("env_realization_bad_python_exact")
    if not isinstance(cuda_build, str) or not cuda_build.strip():
        errors.append("env_realization_bad_cuda_build")
    elif index_policy:
        expected_cuda = (index_policy.get("torch_index") or {}).get("cuda_build")
        if cuda_build != expected_cuda:
            errors.append("env_realization_cuda_build_vs_index_policy")

    if obj.get("pip_check_ok") is True and not is_hex_sha256(obj.get("pip_check_output_sha256")):
        errors.append("pip_check_ok_without_output_hash")
    if obj.get("import_smoke_ok") is True and not is_hex_sha256(obj.get("import_smoke_output_sha256")):
        errors.append("import_smoke_ok_without_output_hash")

    # Parse structured provenance / freeze hashes (not mere file presence).
    if recompute_files and "pip_freeze_hashes_json" in actual_shas and "pip_freeze_txt" in actual_shas:
        try:
            freeze_txt = (
                root / str(evidence["pip_freeze_txt"]).replace("\\", "/")
            ).read_text(encoding="utf-8")
            hashes_obj = load_json(
                root / str(evidence["pip_freeze_hashes_json"]).replace("\\", "/")
            )
            validate_pip_freeze_hashes(
                hashes_obj,
                index_policy=index_policy,
                freeze_txt=freeze_txt,
                root=root,
                recompute_wheel_files=True,
            )
        except ValueError as exc:
            errors.append(str(exc))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"pip_freeze_hashes_unreadable:{exc}")
    if recompute_files and "python_provenance_json" in actual_shas and isinstance(python_exact, str):
        try:
            prov = load_json(
                root / str(evidence["python_provenance_json"]).replace("\\", "/")
            )
            validate_python_provenance(prov, python_exact=python_exact)
        except ValueError as exc:
            errors.append(str(exc))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"python_provenance_unreadable:{exc}")
    if recompute_files and "cuda_provenance_json" in actual_shas:
        try:
            cuda_prov = load_json(
                root / str(evidence["cuda_provenance_json"]).replace("\\", "/")
            )
            validate_cuda_provenance(cuda_prov, index_policy=index_policy)
        except ValueError as exc:
            errors.append(str(exc))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cuda_provenance_unreadable:{exc}")

    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "evidence_shas": actual_shas}


def validate_realization_approval(obj: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Immutable download/env-only authorization bound to scaffold PASS + plan chain."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_realization_approval_v1":
        errors.append("bad_realization_approval_schema")
    if obj.get("status") != "APPROVED" or obj.get("decision") != "APPROVED":
        errors.append("realization_approval_not_approved")
    if obj.get("scope") != "download_and_isolated_env_realization_only":
        errors.append("realization_approval_bad_scope")
    if obj.get("one_shot") is not True:
        errors.append("realization_approval_one_shot_required")
    operator = obj.get("operator")
    if not isinstance(operator, str) or not operator.strip():
        errors.append("realization_approval_operator_required")
    approved_at = obj.get("approved_at")
    if not isinstance(approved_at, str) or not re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", str(approved_at)
    ):
        errors.append("realization_approval_timestamp_required")
    if obj.get("authorization_base_commit") != EXPECTED_REALIZATION_APPROVAL_BASE_COMMIT:
        errors.append("realization_approval_base_commit_mismatch")
    if obj.get("scaffold_final_audit") != SCAFFOLD_FINAL_AUDIT_REL:
        errors.append("realization_approval_scaffold_audit_path")
    if not (root / SCAFFOLD_FINAL_AUDIT_REL).is_file():
        errors.append("scaffold_final_audit_missing")
    elif sha256_file(root / SCAFFOLD_FINAL_AUDIT_REL) != EXPECTED_SCAFFOLD_FINAL_AUDIT_SHA256:
        errors.append("scaffold_final_audit_sha_mismatch")
    if obj.get("scaffold_final_audit_sha256") != EXPECTED_SCAFFOLD_FINAL_AUDIT_SHA256:
        errors.append("realization_approval_scaffold_sha_field_mismatch")
    binds = {
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "plan_lock_sha256": EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        "plan_audit_sha256": EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256,
        "protocol_approval_sha256": EXPECTED_APPROVAL_SHA256,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "requirements_lock_sha256": EXPECTED_REQUIREMENTS_LOCK_SHA256,
    }
    for key, expected in binds.items():
        if obj.get(key) != expected:
            errors.append(f"realization_approval_{key}_mismatch")
    if list(obj.get("candidate_ids") or []) != list(EXPECTED_CANDIDATES):
        errors.append("realization_approval_candidate_ids_mismatch")
    for key in (
        "allow_weight_download",
        "allow_isolated_env_install",
        "allow_pip_import_cuda_smoke",
    ):
        if obj.get(key) is not True:
            errors.append(f"realization_approval_{key}_required")
    for key in (
        "allow_gpu_feasibility",
        "allow_execution_approval",
        "allow_execution_seal",
        "allow_d0",
        "allow_training",
        "allow_hpo",
        "allow_test",
        "allow_accuracy",
        "allow_auto_chain",
    ):
        if obj.get(key) is not False:
            errors.append(f"realization_approval_{key}_must_be_false")
    if obj.get("stop_after") != "F0_REALIZATION_MANIFEST_AUDIT_PASS":
        errors.append("realization_approval_stop_after")
    # Live plan/lock/audit bytes must still match sealed digests.
    if sha256_file(root / PROTOCOL_PLAN_REL) != EXPECTED_PROTOCOL_PLAN_SHA256:
        errors.append("plan_bytes_sha_mismatch")
    if sha256_file(root / PROTOCOL_PLAN_LOCK_REL) != EXPECTED_PROTOCOL_PLAN_LOCK_SHA256:
        errors.append("plan_lock_bytes_sha_mismatch")
    if sha256_file(root / PROTOCOL_PLAN_AUDIT_REL) != EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256:
        errors.append("plan_audit_bytes_sha_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "scope": obj["scope"]}


def validate_realization_approval_audit(
    obj: dict[str, Any],
    *,
    root: Path,
    realization_approval_sha256: str,
    realization_approval_commit: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_realization_approval_audit_v1":
        errors.append("bad_realization_approval_audit_schema")
    if obj.get("audit_verdict") != "PASS":
        errors.append("realization_approval_audit_not_pass")
    if obj.get("realization_approval") != REALIZATION_APPROVAL_REL:
        errors.append("realization_approval_audit_path")
    if obj.get("realization_approval_sha256") != realization_approval_sha256:
        errors.append("realization_approval_audit_sha_mismatch")
    if obj.get("realization_approval_commit") != realization_approval_commit:
        errors.append("realization_approval_audit_commit_mismatch")
    if obj.get("authorization_base_commit") != EXPECTED_REALIZATION_APPROVAL_BASE_COMMIT:
        errors.append("realization_approval_audit_base_commit_mismatch")
    if obj.get("scaffold_final_audit_sha256") != EXPECTED_SCAFFOLD_FINAL_AUDIT_SHA256:
        errors.append("realization_approval_audit_scaffold_sha_mismatch")
    if obj.get("scope_confirmed") != "download_and_isolated_env_realization_only":
        errors.append("realization_approval_audit_scope")
    allows = obj.get("allows_under_this_audit") or {}
    for key in ("weight_download", "isolated_env_install", "pip_import_cuda_smoke"):
        if allows.get(key) is not True:
            errors.append(f"realization_approval_audit_allow_{key}")
    forbids = obj.get("forbids_under_this_audit") or {}
    for key in (
        "gpu_feasibility",
        "execution_approval_or_seal",
        "d0",
        "training",
        "hpo",
        "test",
        "accuracy",
        "auto_chain",
    ):
        if forbids.get(key) is not True:
            errors.append(f"realization_approval_audit_forbid_{key}")
    if not (root / REALIZATION_APPROVAL_REL).is_file():
        errors.append("realization_approval_missing")
    elif sha256_file(root / REALIZATION_APPROVAL_REL) != realization_approval_sha256:
        errors.append("realization_approval_bytes_sha_mismatch")
    else:
        validate_realization_approval(load_json(root / REALIZATION_APPROVAL_REL), root=root)
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "audit_verdict": "PASS"}


def validate_weights_manifest(
    manifest: dict[str, Any],
    *,
    root: Path,
    plan: dict[str, Any],
    plan_sha256: str,
    recompute_files: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema") != "ms_sta_v5_f0_weights_manifest_v1":
        errors.append("bad_weights_schema")
    if manifest.get("approval_sha256") != EXPECTED_APPROVAL_SHA256:
        errors.append("weights_approval_sha_mismatch")
    if manifest.get("plan_sha256") != plan_sha256:
        errors.append("weights_plan_sha_mismatch")
    if manifest.get("bind_snapshot_contract") is not True:
        errors.append("snapshot_contract_binding_required")
    if manifest.get("snapshot_contract_sha256") != EXPECTED_SNAPSHOT_CONTRACT_SHA256:
        errors.append("weights_snapshot_contract_sha_mismatch")
    if manifest.get("index_policy_sha256") != EXPECTED_INDEX_POLICY_SHA256:
        errors.append("weights_index_policy_sha_mismatch")

    entries = manifest.get("entries") or []
    ids = [e.get("candidate_id") for e in entries]
    if sorted(set(ids)) != sorted(EXPECTED_CANDIDATES):
        errors.append("weights_candidate_ids_mismatch")

    snap_path = root / SNAPSHOT_CONTRACT_REL
    if not snap_path.is_file():
        errors.append("snapshot_contract_missing")
        snap: dict[str, Any] = {}
    else:
        if sha256_file(snap_path) != EXPECTED_SNAPSHOT_CONTRACT_SHA256:
            errors.append("snapshot_contract_bytes_sha_mismatch")
        snap = load_json(snap_path)
        if snap.get("plan_sha256") != plan_sha256:
            errors.append("snapshot_contract_plan_sha_mismatch")

    by_cand: dict[str, list] = {}
    for e in entries:
        by_cand.setdefault(e.get("candidate_id"), []).append(e)

    by_plan = {c["id"]: c for c in plan.get("candidates") or []}
    canon_rel = ((plan.get("weights_policy") or {}).get("canonical_dir") or WEIGHTS_DIR_REL).rstrip(
        "/"
    )
    seen_paths: set[str] = set()
    seen_resolved: dict[str, str] = {}

    for cid, csnap in (snap.get("candidates") or {}).items():
        need = csnap.get("files") or []
        have = by_cand.get(cid) or []
        if len(have) != len(need):
            errors.append(f"{cid}:snapshot_file_count_mismatch")
        pc = by_plan.get(cid) or {}
        plan_ckpt = pc.get("pretrained_checkpoint") or {}
        used_entry_idxs: set[int] = set()
        for fspec in need:
            local_name = fspec.get("local_name")
            role = fspec.get("role")
            matches = []
            for idx, h in enumerate(have):
                if idx in used_entry_idxs:
                    continue
                loc = str(h.get("local_path", "")).replace("\\", "/")
                if h.get("role") != role:
                    continue
                if not loc.endswith("/" + str(local_name)) and Path(loc).name != local_name:
                    continue
                matches.append((idx, h))
            if not matches:
                errors.append(f"{cid}:snapshot_file_missing:{role}:{local_name}")
                continue
            idx, e = matches[0]
            used_entry_idxs.add(idx)

            # Per-file source binding (not plan landing URL for every role).
            if fspec.get("url") is not None and e.get("url") != fspec.get("url"):
                errors.append(f"{cid}:{role}:url_mismatch_vs_snapshot")
            if fspec.get("hf_path") is not None:
                if e.get("hf_path") != fspec.get("hf_path"):
                    errors.append(f"{cid}:{role}:hf_path_mismatch_vs_snapshot")
                rev = csnap.get("revision") or plan_ckpt.get("revision")
                if e.get("revision") != rev:
                    errors.append(f"{cid}:{role}:revision_mismatch_vs_snapshot")
                expected_repo = csnap.get("repo_id")
                if not expected_repo:
                    errors.append(f"{cid}:{role}:snapshot_repo_id_missing")
                elif e.get("repo_id") != expected_repo:
                    errors.append(f"{cid}:{role}:repo_id_mismatch_vs_snapshot")
                if e.get("revision") is None:
                    errors.append(f"{cid}:{role}:revision_required_for_hf")
                if e.get("hf_path") is None:
                    errors.append(f"{cid}:{role}:hf_path_required")
            if e.get("license") != plan_ckpt.get("license"):
                # license still from plan candidate
                if plan_ckpt.get("license") is not None:
                    errors.append(f"{cid}:{role}:license_mismatch_vs_plan")
            if e.get("model_identifier") != pc.get("model_identifier"):
                errors.append(f"{cid}:{role}:model_identifier_mismatch_vs_plan")
            if not is_hex_sha256(e.get("sha256")):
                errors.append(f"{cid}:{role}:weights_sha_not_hex")

            rel = (e.get("local_path") or "").replace("\\", "/")
            if rel in seen_paths:
                errors.append(f"{cid}:{role}:duplicate_local_path")
            seen_paths.add(rel)
            resolved, path_errs = resolve_under_canonical(
                root=root, local_path=rel, canonical_dir_rel=canon_rel
            )
            errors.extend(f"{cid}:{role}:{err}" for err in path_errs)
            if resolved is None:
                continue
            rkey = str(resolved.resolve())
            if rkey in seen_resolved:
                errors.append(f"{cid}:{role}:duplicate_resolved_path:{seen_resolved[rkey]}")
            else:
                seen_resolved[rkey] = f"{cid}:{role}:{local_name}"
            # Must live under weights/<candidate_id>/
            try:
                resolved.relative_to((root / canon_rel / cid).resolve())
            except ValueError:
                errors.append(f"{cid}:{role}:not_under_candidate_dir")
            if not resolved.is_file():
                errors.append(f"{cid}:{role}:weights_file_missing")
                continue
            if Path(resolved.name) != Path(str(local_name)):
                errors.append(f"{cid}:{role}:local_name_mismatch")
            if recompute_files:
                actual = sha256_file(resolved)
                if actual != e.get("sha256"):
                    errors.append(f"{cid}:{role}:weights_sha_mismatch_actual={actual}")
                if e.get("sha256_verified") is True and actual != e.get("sha256"):
                    errors.append(f"{cid}:{role}:sha256_verified_flag_lie")

    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "entries": len(entries)}


def validate_realization_manifest(
    manifest: dict[str, Any],
    *,
    root: Path,
    plan_sha256: str,
    plan_lock_sha256: str,
    weights_manifest_sha256: str,
    env_realization_sha256: str,
    recompute_files: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema") != "ms_sta_v5_f0_realization_manifest_v1":
        errors.append("bad_realization_schema")
    if manifest.get("status") != "REALIZED":
        errors.append("realization_status")
    if manifest.get("plan_sha256") != plan_sha256:
        errors.append("realization_plan_sha_mismatch")
    if manifest.get("plan_lock_sha256") != plan_lock_sha256:
        errors.append("realization_plan_lock_sha_mismatch")
    if manifest.get("weights_manifest_sha256") != weights_manifest_sha256:
        errors.append("realization_weights_sha_mismatch")
    if manifest.get("env_realization_sha256") != env_realization_sha256:
        errors.append("realization_env_realization_sha_mismatch")
    if manifest.get("index_policy_sha256") != EXPECTED_INDEX_POLICY_SHA256:
        errors.append("realization_index_policy_sha_mismatch")
    if manifest.get("snapshot_contract_sha256") != EXPECTED_SNAPSHOT_CONTRACT_SHA256:
        errors.append("realization_snapshot_contract_sha_mismatch")

    evidence = manifest.get("evidence_files") or {}
    actual_shas: dict[str, str] = {}
    if recompute_files:
        actual_shas, ev_errs = bind_evidence_map(
            root=root,
            evidence=evidence,
            basenames=REALIZATION_EVIDENCE_BASENAMES,
            prefix="realization",
        )
        errors.extend(ev_errs)

    for key in (
        "pip_check_exit_code",
        "import_smoke_exit_code",
        "cuda_smoke_exit_code",
    ):
        if _get_field(manifest, key) != 0:
            errors.append(f"realization_bad_{key}")
    for top_key, ev_key in {
        "pip_check_output_sha256": "pip_check_log",
        "import_smoke_output_sha256": "import_smoke_log",
        "cuda_smoke_output_sha256": "cuda_smoke_log",
    }.items():
        val = _get_field(manifest, top_key)
        if not is_hex_sha256(val):
            errors.append(f"realization_bad_{top_key}")
        elif recompute_files and ev_key in actual_shas and val != actual_shas[ev_key]:
            errors.append(f"realization_toplevel_sha_unbound:{top_key}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_realization_manifest_audit(
    obj: dict[str, Any],
    *,
    root: Path,
    plan_sha256: str,
    plan_lock_sha256: str,
    weights_manifest_sha256: str,
    env_realization_sha256: str,
    realization_manifest_sha256: str,
    realization_manifest_rel: str | None = None,
) -> dict[str, Any]:
    """Require PASS audit bound to live realization/env/weights/plan digests."""
    errors: list[str] = []
    realization_rel = realization_manifest_rel or REALIZATION_REL
    if obj.get("schema") != "ms_sta_v5_f0_realization_manifest_audit_v1":
        errors.append("bad_realization_manifest_audit_schema")
    if obj.get("audit_verdict") != "PASS":
        errors.append("realization_manifest_audit_not_pass")
    if obj.get("realization_manifest") != realization_rel:
        errors.append("realization_manifest_audit_path")
    binds = {
        "plan_sha256": plan_sha256,
        "plan_lock_sha256": plan_lock_sha256,
        "weights_manifest_sha256": weights_manifest_sha256,
        "env_realization_sha256": env_realization_sha256,
        "realization_manifest_sha256": realization_manifest_sha256,
    }
    for key, expected in binds.items():
        if obj.get(key) != expected:
            errors.append(f"realization_manifest_audit_{key}_mismatch")
    if not (root / realization_rel).is_file():
        errors.append("realization_manifest_missing_for_audit")
    elif sha256_file(root / realization_rel) != realization_manifest_sha256:
        errors.append("realization_manifest_audit_bytes_mismatch")
    if obj.get("gpu_feasibility_authorized") is not False:
        errors.append("realization_manifest_audit_must_not_authorize_gpu")
    if obj.get("execution_approval_authorized") is not False:
        errors.append("realization_manifest_audit_must_not_authorize_execution")
    if obj.get("auto_chain") is not False:
        errors.append("realization_manifest_audit_auto_chain_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "audit_verdict": "PASS"}


def validate_videomae_small_conversion_evidence(
    obj: dict[str, Any],
    *,
    root: Path,
    weights_manifest_sha256: str,
    recompute_tensors: bool = True,
    weights_manifest_rel: str | None = None,
) -> dict[str, Any]:
    """Recompute bit-exact conversion evidence for VideoMAE-small contract local name."""
    errors: list[str] = []
    weights_rel = weights_manifest_rel or WEIGHTS_REL
    if obj.get("schema") != "ms_sta_v5_f0_videomae_small_conversion_evidence_v1":
        errors.append("bad_videomae_conversion_evidence_schema")
    if obj.get("status") != "VERIFIED":
        errors.append("videomae_conversion_status")
    if obj.get("candidate_id") != "videomae_vit_small":
        errors.append("videomae_conversion_candidate")
    if obj.get("repo_id") != "MCG-NJU/videomae-small-finetuned-kinetics":
        errors.append("videomae_conversion_repo_id")
    if obj.get("revision") != "8fc0b88efda5274a4c2e74d3157c02f1ec0764a7":
        errors.append("videomae_conversion_revision")
    if obj.get("source_hf_path") != "pytorch_model.bin":
        errors.append("videomae_conversion_source_path")
    if obj.get("converted_local_name") != "model.safetensors":
        errors.append("videomae_conversion_local_name")
    if obj.get("conversion_method") != "torch_state_dict_to_safetensors":
        errors.append("videomae_conversion_method")
    if obj.get("weights_manifest_sha256") != weights_manifest_sha256:
        errors.append("videomae_conversion_weights_manifest_sha_mismatch")

    source_rel = f"{WEIGHTS_DIR_REL}/videomae_vit_small/pytorch_model.bin"
    converted_rel = f"{WEIGHTS_DIR_REL}/videomae_vit_small/model.safetensors"
    if obj.get("source_local_path") != source_rel:
        errors.append("videomae_conversion_source_local_path")
    if obj.get("converted_local_path") != converted_rel:
        errors.append("videomae_conversion_converted_local_path")

    source_path = root / source_rel
    converted_path = root / converted_rel
    if not source_path.is_file():
        errors.append("videomae_conversion_source_missing")
    if not converted_path.is_file():
        errors.append("videomae_conversion_converted_missing")
    if source_path.is_file() and converted_path.is_file():
        source_sha = sha256_file(source_path)
        converted_sha = sha256_file(converted_path)
        if obj.get("source_sha256") != source_sha:
            errors.append("videomae_conversion_source_sha_mismatch")
        if obj.get("converted_sha256") != converted_sha:
            errors.append("videomae_conversion_converted_sha_mismatch")
        # Bind to weights manifest entry.
        try:
            weights = load_json(root / weights_rel)
            if sha256_file(root / weights_rel) != weights_manifest_sha256:
                errors.append("videomae_conversion_weights_bytes_mismatch")
            entry = None
            for e in weights.get("entries") or []:
                if (
                    e.get("candidate_id") == "videomae_vit_small"
                    and e.get("role") == "weights"
                    and str(e.get("local_path", "")).endswith("model.safetensors")
                ):
                    entry = e
                    break
            if entry is None:
                errors.append("videomae_conversion_weights_entry_missing")
            else:
                if entry.get("sha256") != converted_sha:
                    errors.append("videomae_conversion_weights_entry_sha_mismatch")
                conv = entry.get("conversion") or {}
                if conv.get("source_sha256") != source_sha:
                    errors.append("videomae_conversion_weights_source_sha_mismatch")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"videomae_conversion_weights_unreadable:{exc}")

        if recompute_tensors:
            try:
                import torch
                from safetensors.torch import load_file
            except ImportError as exc:  # pragma: no cover
                errors.append(f"videomae_conversion_torch_safetensors_required:{exc}")
            else:
                src = torch.load(source_path, map_location="cpu")
                if isinstance(src, dict) and "state_dict" in src:
                    src = src["state_dict"]
                if not isinstance(src, dict):
                    errors.append("videomae_conversion_source_not_state_dict")
                else:
                    dst = load_file(str(converted_path))
                    src_keys = sorted(src.keys())
                    dst_keys = sorted(dst.keys())
                    if obj.get("source_key_count") != len(src_keys):
                        errors.append("videomae_conversion_source_key_count_mismatch")
                    if obj.get("converted_key_count") != len(dst_keys):
                        errors.append("videomae_conversion_converted_key_count_mismatch")
                    if obj.get("keysets_equal") is not True:
                        errors.append("videomae_conversion_keysets_equal_flag")
                    if src_keys != dst_keys:
                        errors.append("videomae_conversion_keysets_not_equal")
                    bit_exact = True
                    dtype_shape_equal = True
                    for k in src_keys:
                        st = src[k]
                        dt = dst[k]
                        if tuple(st.shape) != tuple(dt.shape) or st.dtype != dt.dtype:
                            dtype_shape_equal = False
                            bit_exact = False
                            break
                        if not torch.equal(st.contiguous(), dt.contiguous()):
                            bit_exact = False
                            break
                    if obj.get("dtype_shape_equal") is not True or not dtype_shape_equal:
                        errors.append("videomae_conversion_dtype_shape_not_equal")
                    if obj.get("tensors_bit_exact") is not True or not bit_exact:
                        errors.append("videomae_conversion_tensors_not_bit_exact")
                    # Transformers 4.37.x crashes when safetensors metadata is None.
                    try:
                        from safetensors import safe_open
                    except ImportError as exc:  # pragma: no cover
                        errors.append(f"videomae_conversion_safetensors_required:{exc}")
                    else:
                        with safe_open(str(converted_path), framework="pt") as handle:
                            meta = handle.metadata()
                        if not isinstance(meta, dict):
                            errors.append("videomae_conversion_safetensors_metadata_missing")
                        elif meta.get("format") != "pt":
                            errors.append("videomae_conversion_safetensors_metadata_format")
    # R1+ evidence binds explicit metadata fields; historical F0 may omit them.
    if "safetensors_metadata" in obj or "safetensors_metadata_verified" in obj:
        if obj.get("safetensors_metadata") != {"format": "pt"}:
            errors.append("videomae_conversion_safetensors_metadata_field")
        if obj.get("safetensors_metadata_verified") is not True:
            errors.append("videomae_conversion_safetensors_metadata_verified")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "tensors_bit_exact": True}


def validate_execution_approval(
    obj: dict[str, Any],
    *,
    root: Path,
    plan_sha256: str,
    env_realization_sha256: str,
    realization_manifest_sha256: str,
    weights_manifest_sha256: str,
    executable_tree_sha256: str,
    realization_manifest_audit_sha256: str,
    videomae_small_conversion_evidence_sha256: str,
    realization_manifest_audit_rel: str | None = None,
    conversion_evidence_rel: str | None = None,
    realization_manifest_rel: str | None = None,
    weights_manifest_rel: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    audit_rel = realization_manifest_audit_rel or REALIZATION_MANIFEST_AUDIT_REL
    conv_rel = conversion_evidence_rel or VIDEOMAE_SMALL_CONVERSION_EVIDENCE_REL
    realization_rel = realization_manifest_rel or REALIZATION_REL
    weights_rel = weights_manifest_rel or WEIGHTS_REL
    if obj.get("schema") != "ms_sta_v5_f0_execution_approval_v1":
        errors.append("bad_execution_approval_schema")
    if obj.get("status") != "APPROVED":
        errors.append("execution_approval_status")
    if obj.get("decision") != "APPROVED":
        errors.append("execution_approval_decision")
    if obj.get("one_shot") is not True:
        errors.append("execution_approval_one_shot")
    if obj.get("allow_gpu_feasibility") is not True:
        errors.append("execution_approval_gpu_not_allowed")
    if obj.get("scope") != "backbone_feasibility_gpu_only":
        errors.append("execution_approval_bad_scope")
    operator = obj.get("operator")
    if not isinstance(operator, str) or not operator.strip():
        errors.append("execution_approval_operator_required")
    approved_at = obj.get("approved_at")
    if not isinstance(approved_at, str) or not re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", str(approved_at)
    ):
        errors.append("execution_approval_timestamp_required")
    for key in ("allow_accuracy", "allow_d0", "allow_training", "allow_hpo", "allow_test"):
        if obj.get(key) is not False:
            errors.append(f"execution_approval_{key}_must_be_false")
    binds = {
        "plan_sha256": plan_sha256,
        "env_realization_sha256": env_realization_sha256,
        "realization_manifest_sha256": realization_manifest_sha256,
        "weights_manifest_sha256": weights_manifest_sha256,
        "executable_tree_sha256": executable_tree_sha256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "requirements_lock_sha256": EXPECTED_REQUIREMENTS_LOCK_SHA256,
        "realization_manifest_audit_sha256": realization_manifest_audit_sha256,
        "videomae_small_conversion_evidence_sha256": videomae_small_conversion_evidence_sha256,
    }
    for k, expected in binds.items():
        if obj.get(k) != expected:
            errors.append(f"execution_approval_bad_{k}")
    if sha256_file(root / SNAPSHOT_CONTRACT_REL) != EXPECTED_SNAPSHOT_CONTRACT_SHA256:
        errors.append("execution_approval_snapshot_bytes_drift")
    if sha256_file(root / INDEX_POLICY_REL) != EXPECTED_INDEX_POLICY_SHA256:
        errors.append("execution_approval_index_policy_bytes_drift")
    if not (root / audit_rel).is_file():
        errors.append("execution_approval_realization_audit_missing")
    elif sha256_file(root / audit_rel) != realization_manifest_audit_sha256:
        errors.append("execution_approval_realization_audit_bytes_mismatch")
    else:
        try:
            validate_realization_manifest_audit(
                load_json(root / audit_rel),
                root=root,
                plan_sha256=plan_sha256,
                plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
                weights_manifest_sha256=weights_manifest_sha256,
                env_realization_sha256=env_realization_sha256,
                realization_manifest_sha256=realization_manifest_sha256,
                realization_manifest_rel=realization_rel,
            )
        except ValueError as exc:
            errors.append(f"execution_approval_realization_audit_invalid:{exc}")
    if not (root / conv_rel).is_file():
        errors.append("execution_approval_conversion_evidence_missing")
    elif sha256_file(root / conv_rel) != videomae_small_conversion_evidence_sha256:
        errors.append("execution_approval_conversion_evidence_bytes_mismatch")
    else:
        try:
            validate_videomae_small_conversion_evidence(
                load_json(root / conv_rel),
                root=root,
                weights_manifest_sha256=weights_manifest_sha256,
                recompute_tensors=True,
                weights_manifest_rel=weights_rel,
            )
        except ValueError as exc:
            errors.append(f"execution_approval_conversion_evidence_invalid:{exc}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "bindings": binds}


def validate_execution_lock(
    lock: dict[str, Any],
    *,
    plan_sha256: str,
    weights_manifest_sha256: str,
    env_realization_sha256: str,
    realization_manifest_sha256: str,
    executable_tree_sha256: str,
    execution_approval_sha256: str | None = None,
    index_policy_sha256: str | None = None,
    snapshot_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Lock binds approval (DAG forward only). Must NOT bind audit/seal SHA."""
    errors: list[str] = []
    if lock.get("schema") != "ms_sta_v5_f0_execution_lock_v1":
        errors.append("bad_execution_lock_schema")
    if lock.get("status") != "EXECUTION_LOCKED":
        errors.append("execution_lock_status")
    if "execution_audit_sha256" in lock:
        errors.append("execution_lock_must_not_bind_audit_sha")
    if "execution_seal_sha256" in lock:
        errors.append("execution_lock_must_not_bind_seal_sha")
    mapping = {
        "plan_sha256": plan_sha256,
        "weights_manifest_sha256": weights_manifest_sha256,
        "env_realization_sha256": env_realization_sha256,
        "realization_manifest_sha256": realization_manifest_sha256,
        "executable_tree_sha256": executable_tree_sha256,
        "index_policy_sha256": index_policy_sha256 or EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": snapshot_contract_sha256 or EXPECTED_SNAPSHOT_CONTRACT_SHA256,
    }
    for k, expected in mapping.items():
        if lock.get(k) != expected or not is_hex_sha256(str(lock.get(k) or "")):
            errors.append(f"execution_lock_bad_{k}")
    if execution_approval_sha256 is None or not is_hex_sha256(execution_approval_sha256):
        errors.append("execution_approval_sha_required")
    elif lock.get("execution_approval_sha256") != execution_approval_sha256:
        errors.append("execution_lock_approval_sha_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_audit(
    obj: dict[str, Any],
    *,
    execution_approval_sha256: str,
    execution_lock_sha256: str,
    executable_tree_sha256: str,
) -> dict[str, Any]:
    """Audit binds approval+lock. Must NOT authorize GPU and must NOT be bound by lock."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_execution_audit_v1":
        errors.append("bad_execution_audit_schema")
    if obj.get("audit_verdict") != "PASS":
        errors.append("execution_audit_verdict")
    if obj.get("execution_approval_sha256") != execution_approval_sha256:
        errors.append("execution_audit_approval_sha_mismatch")
    if obj.get("execution_lock_sha256") != execution_lock_sha256:
        errors.append("execution_audit_lock_sha_mismatch")
    if obj.get("executable_tree_sha256") != executable_tree_sha256:
        errors.append("execution_audit_executable_tree_sha_mismatch")
    # GPU authorization lives only on the seal (DAG tip).
    if obj.get("gpu_authorized") is True:
        errors.append("execution_audit_must_not_authorize_gpu")
    operator = obj.get("auditor")
    if not isinstance(operator, str) or not operator.strip():
        errors.append("execution_audit_auditor_required")
    audited_at = obj.get("audited_at")
    if not isinstance(audited_at, str) or not re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", str(audited_at)
    ):
        errors.append("execution_audit_timestamp_required")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_seal(
    obj: dict[str, Any],
    *,
    execution_approval_sha256: str,
    execution_lock_sha256: str,
    execution_audit_sha256: str,
    executable_tree_sha256: str,
) -> dict[str, Any]:
    """Final DAG tip: binds approval+lock+audit; sole GPU authorization artifact."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_execution_seal_v1":
        errors.append("bad_execution_seal_schema")
    if obj.get("status") != "SEALED":
        errors.append("execution_seal_status")
    if obj.get("gpu_authorized") is not True:
        errors.append("execution_seal_gpu_authorized_required")
    if obj.get("one_shot") is not True:
        errors.append("execution_seal_one_shot")
    if obj.get("execution_approval_sha256") != execution_approval_sha256:
        errors.append("execution_seal_approval_sha_mismatch")
    if obj.get("execution_lock_sha256") != execution_lock_sha256:
        errors.append("execution_seal_lock_sha_mismatch")
    if obj.get("execution_audit_sha256") != execution_audit_sha256:
        errors.append("execution_seal_audit_sha_mismatch")
    if obj.get("executable_tree_sha256") != executable_tree_sha256:
        errors.append("execution_seal_executable_tree_sha_mismatch")
    operator = obj.get("operator")
    if not isinstance(operator, str) or not operator.strip():
        errors.append("execution_seal_operator_required")
    sealed_at = obj.get("sealed_at")
    if not isinstance(sealed_at, str) or not re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", str(sealed_at)
    ):
        errors.append("execution_seal_timestamp_required")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "gpu_authorized": True}


def gpu_run_authorization_gate(
    pf: dict[str, Any], *, device: str
) -> tuple[bool, list[str]]:
    """Fail-closed gate used by feasibility runner before any CUDA work."""
    reasons: list[str] = []
    if not pf.get("execution_ok"):
        reasons.append("execution_ok_false")
    if pf.get("side_effect_allowed") is not True:
        reasons.append("side_effect_not_allowed")
    if pf.get("gpu_authorized") is not True:
        reasons.append("gpu_not_authorized")
    refuse = pf.get("refuse_reasons") or []
    if refuse:
        reasons.append("refuse_reasons_nonempty")
    dev = str(device or "").strip().lower()
    if not (dev == "cuda" or dev.startswith("cuda:")):
        reasons.append("device_not_cuda")
    return (len(reasons) == 0), reasons


def preflight(
    *,
    root: Path = _ROOT,
    action: str,
    python_exe: str,
    require_clean_worktree: bool = True,
    require_plan_audit: bool | None = None,
    campaign: str = "f0",
) -> dict[str, Any]:
    from f0_feasibility.campaign_paths import campaign_paths

    paths = campaign_paths(campaign)
    if (root / PROTOCOL_FROZEN_REL).exists():
        raise ValueError("legacy_F0_PROTOCOL_json_forbidden_use_plan_lock_lifecycle")

    approval = validate_approval(root=root, require_sealed_identity=True)
    audit = validate_approval_audit(root=root)
    draft = load_json(root / PROTOCOL_DRAFT_REL)
    validate_protocol_draft(draft)

    plan = load_json(root / PROTOCOL_PLAN_REL)
    validate_protocol_plan(plan)
    env = load_json(root / ENV_SPEC_REL)
    validate_env_spec(env, mode="plan")
    lock = load_json(root / PROTOCOL_PLAN_LOCK_REL)
    lock_out = validate_protocol_plan_lock(lock, root=root, require_expected_identity=True)

    if not (root / INDEX_POLICY_REL).is_file():
        raise ValueError("index_policy_missing")
    if sha256_file(root / INDEX_POLICY_REL) != EXPECTED_INDEX_POLICY_SHA256:
        raise ValueError("index_policy_sha_mismatch")
    if not (root / SNAPSHOT_CONTRACT_REL).is_file():
        raise ValueError("snapshot_contract_missing")
    if sha256_file(root / SNAPSHOT_CONTRACT_REL) != EXPECTED_SNAPSHOT_CONTRACT_SHA256:
        raise ValueError("snapshot_contract_sha_mismatch")

    if require_plan_audit is None:
        require_plan_audit = EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256 is not None
    plan_audit_out = None
    if require_plan_audit:
        if not (root / PROTOCOL_PLAN_AUDIT_REL).is_file():
            raise ValueError("missing_protocol_plan_audit")
        plan_audit_out = validate_protocol_plan_audit(
            load_json(root / PROTOCOL_PLAN_AUDIT_REL), root=root, require_expected_identity=True
        )

    porcelain = git_porcelain(root)
    dirty = bool(porcelain.strip())
    reasons: list[str] = []
    side_effect = action not in {"dry-plan", None}
    bootstrap_actions = {"download-weights", "download", "realize-env", "realize"}
    is_bootstrap = action in bootstrap_actions
    isolated_errs: list[str] = []

    realization_approval_ok = False
    realization_approval_sha = None
    realization_approval_audit_ok = False
    if (root / REALIZATION_APPROVAL_REL).is_file():
        try:
            validate_realization_approval(
                load_json(root / REALIZATION_APPROVAL_REL), root=root
            )
            realization_approval_ok = True
            realization_approval_sha = sha256_file(root / REALIZATION_APPROVAL_REL)
        except ValueError as exc:
            reasons.append(f"realization_approval_invalid:{exc}")
    if (root / REALIZATION_APPROVAL_AUDIT_REL).is_file() and realization_approval_sha:
        try:
            audit_obj = load_json(root / REALIZATION_APPROVAL_AUDIT_REL)
            # Commit is recorded in the audit artifact itself after independent audit.
            commit = str(audit_obj.get("realization_approval_commit") or "")
            validate_realization_approval_audit(
                audit_obj,
                root=root,
                realization_approval_sha256=realization_approval_sha,
                realization_approval_commit=commit,
            )
            realization_approval_audit_ok = True
        except ValueError as exc:
            reasons.append(f"realization_approval_audit_invalid:{exc}")

    realization_authorized = bool(
        realization_approval_ok and realization_approval_audit_ok
    )

    if side_effect and require_clean_worktree and dirty and not (
        is_bootstrap and realization_authorized
    ):
        reasons.append("dirty_worktree")
    if side_effect and not (is_bootstrap and realization_authorized):
        isolated_errs = require_isolated_interpreter(python_exe, root=root)
        reasons.extend(isolated_errs)
    elif not side_effect:
        reasons.extend(detect_forbidden_interpreter(python_exe))
    elif is_bootstrap and realization_authorized:
        # Bootstrap may use host Python 3.11 to create the isolated venv / download.
        reasons.extend(detect_forbidden_interpreter(python_exe))

    plan_sha = lock_out["plan_sha256"]
    plan_lock_sha = lock_out["plan_lock_sha256"]
    weights_ok = False
    weights_error = "weights_manifest_absent"
    weights_sha = None
    weights_rel = paths.weights_manifest_rel
    realization_rel = paths.realization_manifest_rel
    realization_audit_rel = paths.realization_manifest_audit_rel
    conversion_rel = paths.conversion_evidence_rel
    execution_approval_rel = paths.execution_approval_rel
    execution_lock_rel = paths.execution_lock_rel
    execution_audit_rel = paths.execution_audit_rel
    execution_seal_rel = paths.execution_seal_rel
    if (root / weights_rel).is_file():
        try:
            validate_weights_manifest(
                load_json(root / weights_rel),
                root=root,
                plan=plan,
                plan_sha256=plan_sha,
            )
            weights_ok = True
            weights_error = None
            weights_sha = sha256_file(root / weights_rel)
        except ValueError as exc:
            weights_error = str(exc)

    env_realization_ok = False
    env_realization_sha = None
    if (root / ENV_REALIZATION_REL).is_file():
        try:
            env_realization_sha_calc = sha256_file(root / ENV_REALIZATION_REL)
            validate_env_realization(
                load_json(root / ENV_REALIZATION_REL),
                root=root,
                plan_sha256=plan_sha,
                env_spec_sha256=sha256_file(root / ENV_SPEC_REL),
            )
            env_realization_ok = True
            env_realization_sha = env_realization_sha_calc
        except ValueError as exc:
            reasons.append(f"env_realization_invalid:{exc}")

    realization_ok = False
    realization_sha = None
    if (root / realization_rel).is_file():
        try:
            if weights_sha is None:
                raise ValueError("realization_requires_weights_manifest")
            if env_realization_sha is None:
                raise ValueError("realization_requires_env_realization")
            validate_realization_manifest(
                load_json(root / realization_rel),
                root=root,
                plan_sha256=plan_sha,
                plan_lock_sha256=plan_lock_sha,
                weights_manifest_sha256=weights_sha,
                env_realization_sha256=env_realization_sha,
            )
            realization_ok = True
            realization_sha = sha256_file(root / realization_rel)
        except ValueError as exc:
            reasons.append(f"realization_invalid:{exc}")

    realization_audit_ok = False
    realization_audit_sha = None
    if (root / realization_audit_rel).is_file():
        try:
            if not (realization_ok and realization_sha and weights_sha and env_realization_sha):
                raise ValueError("realization_audit_requires_manifest_env_weights")
            validate_realization_manifest_audit(
                load_json(root / realization_audit_rel),
                root=root,
                plan_sha256=plan_sha,
                plan_lock_sha256=plan_lock_sha,
                weights_manifest_sha256=weights_sha,
                env_realization_sha256=env_realization_sha,
                realization_manifest_sha256=realization_sha,
                realization_manifest_rel=realization_rel,
            )
            realization_audit_ok = True
            realization_audit_sha = sha256_file(root / realization_audit_rel)
        except ValueError as exc:
            reasons.append(f"realization_manifest_audit_invalid:{exc}")
    elif realization_ok:
        reasons.append("realization_manifest_audit_absent")

    conversion_ok = False
    conversion_sha = None
    if (root / conversion_rel).is_file():
        try:
            if weights_sha is None:
                raise ValueError("conversion_evidence_requires_weights_manifest")
            validate_videomae_small_conversion_evidence(
                load_json(root / conversion_rel),
                root=root,
                weights_manifest_sha256=weights_sha,
                recompute_tensors=True,
                weights_manifest_rel=weights_rel,
            )
            conversion_ok = True
            conversion_sha = sha256_file(root / conversion_rel)
        except ValueError as exc:
            reasons.append(f"videomae_conversion_evidence_invalid:{exc}")
    elif realization_ok:
        reasons.append("videomae_conversion_evidence_absent")

    exec_tree_sha = executable_tree_sha(root=root)
    execution_ok = False
    if (root / execution_lock_rel).is_file():
        try:
            if not (
                weights_ok
                and realization_ok
                and env_realization_ok
                and realization_audit_ok
                and conversion_ok
                and weights_sha
                and realization_sha
                and env_realization_sha
                and realization_audit_sha
                and conversion_sha
            ):
                raise ValueError(
                    "execution_requires_weights_realization_audit_and_conversion_evidence"
                )
            if not (root / execution_approval_rel).is_file():
                raise ValueError("execution_approval_absent")
            if not (root / execution_audit_rel).is_file():
                raise ValueError("execution_audit_absent")
            if not (root / execution_seal_rel).is_file():
                raise ValueError("execution_seal_absent")
            approval_obj = load_json(root / execution_approval_rel)
            approval_sha = sha256_file(root / execution_approval_rel)
            validate_execution_approval(
                approval_obj,
                root=root,
                plan_sha256=plan_sha,
                env_realization_sha256=env_realization_sha,
                realization_manifest_sha256=realization_sha,
                weights_manifest_sha256=weights_sha,
                executable_tree_sha256=exec_tree_sha,
                realization_manifest_audit_sha256=realization_audit_sha,
                videomae_small_conversion_evidence_sha256=conversion_sha,
                realization_manifest_audit_rel=realization_audit_rel,
                conversion_evidence_rel=conversion_rel,
                realization_manifest_rel=realization_rel,
                weights_manifest_rel=weights_rel,
            )
            # DAG: approval → lock → audit → seal (no reverse lock↔audit bind).
            lock_obj = load_json(root / execution_lock_rel)
            lock_sha = sha256_file(root / execution_lock_rel)
            validate_execution_lock(
                lock_obj,
                plan_sha256=plan_sha,
                weights_manifest_sha256=weights_sha,
                env_realization_sha256=env_realization_sha,
                realization_manifest_sha256=realization_sha,
                executable_tree_sha256=exec_tree_sha,
                execution_approval_sha256=approval_sha,
            )
            audit_obj = load_json(root / execution_audit_rel)
            audit_sha = sha256_file(root / execution_audit_rel)
            validate_execution_audit(
                audit_obj,
                execution_approval_sha256=approval_sha,
                execution_lock_sha256=lock_sha,
                executable_tree_sha256=exec_tree_sha,
            )
            seal_obj = load_json(root / execution_seal_rel)
            validate_execution_seal(
                seal_obj,
                execution_approval_sha256=approval_sha,
                execution_lock_sha256=lock_sha,
                execution_audit_sha256=audit_sha,
                executable_tree_sha256=exec_tree_sha,
            )
            execution_ok = True
        except ValueError as exc:
            reasons.append(f"execution_lock_invalid:{exc}")

    gpu_action = action in {"gpu", "run-gpu", "feasibility"}
    if side_effect:
        if action in {"download-weights", "download", "realize-env", "realize"}:
            if not realization_authorized:
                reasons.append("download_forbidden_until_authorized_download_phase")
                reasons.append(
                    "side_effects_forbidden_until_realization_and_execution_authorization"
                )
            # Authorized bootstrap: do not append the generic side-effect ban.
        elif action in {"d0", "train", "hpo", "test", "accuracy"}:
            reasons.append(f"action_forbidden:{action}")
        elif gpu_action:
            if not execution_ok:
                reasons.append("execution_seal_chain_absent_or_invalid")
            if not weights_ok:
                reasons.append(f"weights_not_verified:{weights_error}")
        else:
            reasons.append("side_effects_forbidden_until_realization_and_execution_authorization")

    # GPU may run only when seal chain is valid and refuse_reasons would be empty.
    gpu_authorized = bool(gpu_action and execution_ok and weights_ok and not reasons)
    side_effect_allowed = bool(
        gpu_authorized or (is_bootstrap and realization_authorized and not reasons)
    )

    return {
        "ok": True,
        "action": action,
        "approval": approval,
        "approval_audit": audit,
        "protocol_mode": "plan_locked",
        "protocol_rel": PROTOCOL_PLAN_REL,
        "protocol": plan,
        "protocol_frozen": False,
        "plan_sha256": plan_sha,
        "plan_lock_sha256": plan_lock_sha,
        "plan_lock_status": lock.get("status"),
        "plan_audit": plan_audit_out,
        "weights_ok": weights_ok,
        "weights_error": weights_error,
        "env_realization_ok": env_realization_ok,
        "realization_ok": realization_ok,
        "realization_audit_ok": realization_audit_ok,
        "conversion_evidence_ok": conversion_ok,
        "execution_ok": execution_ok,
        "executable_tree_sha256": exec_tree_sha,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "worktree_dirty": dirty,
        "refuse_reasons": reasons,
        "side_effect_allowed": side_effect_allowed,
        "realization_authorized": realization_authorized,
        "realization_approval_ok": realization_approval_ok,
        "realization_approval_audit_ok": realization_approval_audit_ok,
        "gpu_authorized": gpu_authorized,
    }


def validate_protocol_frozen(protocol: dict[str, Any]) -> dict[str, Any]:
    raise ValueError(
        "legacy_frozen_protocol_removed:use_plan_lock_then_execution_lock_lifecycle"
    )
