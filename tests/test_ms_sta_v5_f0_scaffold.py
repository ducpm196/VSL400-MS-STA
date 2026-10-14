"""MS-STA v5 F0 remediation-6 — auth gate, seal DAG, freeze set, CUDA contract."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_f0_lib import (  # noqa: E402
    EXPECTED_APPROVAL_SHA256,
    EXPECTED_CANDIDATES,
    EXPECTED_INDEX_POLICY_SHA256,
    EXPECTED_PROTOCOL_PLAN_AUDIT_COMMIT,
    EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256,
    EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
    EXPECTED_PROTOCOL_PLAN_SHA256,
    EXPECTED_REQUIREMENTS_LOCK_SHA256,
    EXPECTED_SNAPSHOT_CONTRACT_SHA256,
    INDEX_POLICY_REL,
    SNAPSHOT_CONTRACT_REL,
    gpu_run_authorization_gate,
    load_json,
    resolve_evidence_path,
    resolve_under_canonical,
    require_isolated_interpreter,
    sha256_file,
    validate_env_realization,
    validate_env_spec,
    validate_execution_approval,
    validate_execution_audit,
    validate_execution_lock,
    validate_execution_seal,
    validate_pip_freeze_hashes,
    validate_protocol_plan,
    validate_protocol_plan_audit,
    validate_protocol_plan_lock,
    validate_realization_manifest,
    validate_weights_manifest,
)

PLAN = _ROOT / "Docs/results/ms_sta_v5_f0/F0_PROTOCOL_PLAN.json"
PLAN_LOCK = _ROOT / "Docs/results/ms_sta_v5_f0/F0_PROTOCOL_PLAN_LOCK.json"
PLAN_AUDIT = _ROOT / "Docs/results/ms_sta_v5_f0/F0_PROTOCOL_PLAN_AUDIT.json"
ENV_SPEC = _ROOT / "Docs/results/ms_sta_v5_f0/F0_ISOLATED_ENV_REQUIREMENTS.json"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=_ROOT, capture_output=True, text=True, check=False)


def _write(path: Path, blob: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def _seed_policy_files(tmp_path: Path) -> None:
    for rel in (INDEX_POLICY_REL, SNAPSHOT_CONTRACT_REL):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / rel, dst)


def _valid_hashes_blob(
    *,
    packages: list[dict] | None = None,
    wheel_sha: str | None = None,
) -> bytes:
    digest = wheel_sha or ("a" * 64)
    pkgs = packages or [
        {
            "name": "torch",
            "version": "2.2.2+cu121",
            "sha256": digest,
            "index_url": "https://download.pytorch.org/whl/cu121",
            "wheel_rel": "Docs/results/ms_sta_v5_f0/evidence/wheelhouse/torch-2.2.2+cu121.whl",
        }
    ]
    obj = {
        "schema": "ms_sta_v5_f0_pip_freeze_hashes_v1",
        "torch_index_url": "https://download.pytorch.org/whl/cu121",
        "cuda_build": "cu121",
        "packages": pkgs,
    }
    return (json.dumps(obj, indent=2) + "\n").encode("utf-8")


def _valid_python_prov() -> bytes:
    obj = {
        "schema": "ms_sta_v5_f0_python_provenance_v1",
        "python_exact": "3.11.0",
        "executable": "D:/proj/_venvs/f0_v5/Scripts/python.exe",
        "sys_prefix": "D:/proj/_venvs/f0_v5",
        "platform": "win_amd64",
        "implementation": "CPython",
        "include_system_site_packages": False,
        "pyvenv_cfg_present": True,
    }
    return (json.dumps(obj, indent=2) + "\n").encode("utf-8")


def _valid_cuda_prov() -> bytes:
    obj = {
        "schema": "ms_sta_v5_f0_cuda_provenance_v1",
        "cuda_build": "cu121",
        "torch_index_url": "https://download.pytorch.org/whl/cu121",
        "torch_version": "2.2.2+cu121",
        "cuda_runtime_version": "12.1",
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_index": 0,
    }
    return (json.dumps(obj, indent=2) + "\n").encode("utf-8")


def _build_valid_env_realization(tmp_path: Path) -> dict:
    _seed_policy_files(tmp_path)
    evidence = tmp_path / "Docs/results/ms_sta_v5_f0/evidence"
    wheel = evidence / "wheelhouse" / "torch-2.2.2+cu121.whl"
    wheel_sha = _write(wheel, b"FAKEWHEEL\n")
    freeze_txt = b"torch==2.2.2+cu121\n"
    files = {
        "pip_check.log": b"ok\n",
        "import_smoke.log": b"import-ok\n",
        "cuda_smoke.log": b"cuda-ok\n",
        "pip_freeze.txt": freeze_txt,
        "pip_freeze.hashes.json": _valid_hashes_blob(wheel_sha=wheel_sha),
        "python_provenance.json": _valid_python_prov(),
        "cuda_provenance.json": _valid_cuda_prov(),
    }
    sha = {name: _write(evidence / name, blob) for name, blob in files.items()}
    return {
        "schema": "ms_sta_v5_f0_env_realization_v1",
        "status": "REALIZED",
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "env_spec_sha256": sha256_file(ENV_SPEC),
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "pip_check_exit_code": 0,
        "import_smoke_exit_code": 0,
        "cuda_smoke_exit_code": 0,
        "pip_check_output_sha256": sha["pip_check.log"],
        "import_smoke_output_sha256": sha["import_smoke.log"],
        "cuda_smoke_output_sha256": sha["cuda_smoke.log"],
        "pip_freeze_sha256": sha["pip_freeze.txt"],
        "requirements_lock_sha256": EXPECTED_REQUIREMENTS_LOCK_SHA256,
        "python_exact": "3.11.0",
        "cuda_build": "cu121",
        "evidence": {
            "pip_check_log": "Docs/results/ms_sta_v5_f0/evidence/pip_check.log",
            "pip_check_log_sha256": sha["pip_check.log"],
            "import_smoke_log": "Docs/results/ms_sta_v5_f0/evidence/import_smoke.log",
            "import_smoke_log_sha256": sha["import_smoke.log"],
            "cuda_smoke_log": "Docs/results/ms_sta_v5_f0/evidence/cuda_smoke.log",
            "cuda_smoke_log_sha256": sha["cuda_smoke.log"],
            "pip_freeze_txt": "Docs/results/ms_sta_v5_f0/evidence/pip_freeze.txt",
            "pip_freeze_txt_sha256": sha["pip_freeze.txt"],
            "pip_freeze_hashes_json": "Docs/results/ms_sta_v5_f0/evidence/pip_freeze.hashes.json",
            "pip_freeze_hashes_json_sha256": sha["pip_freeze.hashes.json"],
            "python_provenance_json": "Docs/results/ms_sta_v5_f0/evidence/python_provenance.json",
            "python_provenance_json_sha256": sha["python_provenance.json"],
            "cuda_provenance_json": "Docs/results/ms_sta_v5_f0/evidence/cuda_provenance.json",
            "cuda_provenance_json_sha256": sha["cuda_provenance.json"],
        },
    }


def test_plan_audit_commit_pinned() -> None:
    assert EXPECTED_PROTOCOL_PLAN_AUDIT_COMMIT == "19d60bc025f03fb5129d32b1d3c0f1cb7b182c88"
    assert sha256_file(PLAN_AUDIT) == EXPECTED_PROTOCOL_PLAN_AUDIT_SHA256
    validate_protocol_plan_audit(load_json(PLAN_AUDIT), root=_ROOT, require_expected_identity=True)


def test_plan_lock_still_immutable() -> None:
    validate_protocol_plan(load_json(PLAN))
    validate_protocol_plan_lock(load_json(PLAN_LOCK), root=_ROOT, require_expected_identity=True)


def test_zero_exit_code_accepted_in_env_realization(tmp_path: Path) -> None:
    obj = _build_valid_env_realization(tmp_path)
    out = validate_env_realization(
        obj,
        root=tmp_path,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        env_spec_sha256=sha256_file(ENV_SPEC),
    )
    assert out["ok"] is True


def test_evidence_dotdot_rejected(tmp_path: Path) -> None:
    obj = _build_valid_env_realization(tmp_path)
    obj["evidence"]["pip_check_log"] = (
        "Docs/results/ms_sta_v5_f0/evidence/../evidence/pip_check.log"
    )
    with pytest.raises(ValueError, match="evidence_dotdot_forbidden"):
        validate_env_realization(
            obj,
            root=tmp_path,
            plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
            env_spec_sha256=sha256_file(ENV_SPEC),
        )


def test_incomplete_one_package_freeze_rejected(tmp_path: Path) -> None:
    obj = _build_valid_env_realization(tmp_path)
    # freeze has two packages; hashes only one
    freeze = b"torch==2.2.2+cu121\nnumpy==1.26.4\n"
    freeze_sha = _write(tmp_path / "Docs/results/ms_sta_v5_f0/evidence/pip_freeze.txt", freeze)
    obj["evidence"]["pip_freeze_txt_sha256"] = freeze_sha
    obj["pip_freeze_sha256"] = freeze_sha
    wheel_sha = sha256_file(
        tmp_path / "Docs/results/ms_sta_v5_f0/evidence/wheelhouse/torch-2.2.2+cu121.whl"
    )
    hashes = _valid_hashes_blob(wheel_sha=wheel_sha)  # only torch
    hsha = _write(tmp_path / "Docs/results/ms_sta_v5_f0/evidence/pip_freeze.hashes.json", hashes)
    obj["evidence"]["pip_freeze_hashes_json_sha256"] = hsha
    with pytest.raises(ValueError, match="pip_freeze_set_mismatch"):
        validate_env_realization(
            obj,
            root=tmp_path,
            plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
            env_spec_sha256=sha256_file(ENV_SPEC),
        )


def test_pip_freeze_hashes_direct_set_mismatch() -> None:
    policy = load_json(_ROOT / INDEX_POLICY_REL)
    with pytest.raises(ValueError, match="pip_freeze_set_mismatch"):
        validate_pip_freeze_hashes(
            json.loads(_valid_hashes_blob().decode()),
            index_policy=policy,
            freeze_txt="torch==2.2.2+cu121\nnumpy==1.26.4\n",
            root=_ROOT,
            recompute_wheel_files=False,
        )


def test_wheel_sha_must_match_file(tmp_path: Path) -> None:
    obj = _build_valid_env_realization(tmp_path)
    # Corrupt declared sha while leaving wheel bytes unchanged.
    hashes_path = tmp_path / "Docs/results/ms_sta_v5_f0/evidence/pip_freeze.hashes.json"
    data = json.loads(hashes_path.read_text(encoding="utf-8"))
    data["packages"][0]["sha256"] = "b" * 64
    blob = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    hsha = _write(hashes_path, blob)
    obj["evidence"]["pip_freeze_hashes_json_sha256"] = hsha
    with pytest.raises(ValueError, match="pip_freeze_hashes_wheel_sha_mismatch"):
        validate_env_realization(
            obj,
            root=tmp_path,
            plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
            env_spec_sha256=sha256_file(ENV_SPEC),
        )


def test_toplevel_sha_must_match_evidence(tmp_path: Path) -> None:
    obj = _build_valid_env_realization(tmp_path)
    obj["pip_check_output_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="toplevel_sha_unbound"):
        validate_env_realization(
            obj,
            root=tmp_path,
            plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
            env_spec_sha256=sha256_file(ENV_SPEC),
        )


def test_fake_realization_without_evidence_files_rejected(tmp_path: Path) -> None:
    manifest = {
        "schema": "ms_sta_v5_f0_realization_manifest_v1",
        "status": "REALIZED",
        "plan_sha256": "a" * 64,
        "plan_lock_sha256": "b" * 64,
        "weights_manifest_sha256": "c" * 64,
        "env_realization_sha256": "d" * 64,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "pip_check_exit_code": 0,
        "import_smoke_exit_code": 0,
        "cuda_smoke_exit_code": 0,
        "pip_check_output_sha256": "e" * 64,
        "import_smoke_output_sha256": "f" * 64,
        "cuda_smoke_output_sha256": "1" * 64,
        "evidence_files": {},
    }
    with pytest.raises(ValueError, match="realization_missing_evidence"):
        validate_realization_manifest(
            manifest,
            root=tmp_path,
            plan_sha256="a" * 64,
            plan_lock_sha256="b" * 64,
            weights_manifest_sha256="c" * 64,
            env_realization_sha256="d" * 64,
        )


def test_env_spec_rejects_realized_mode() -> None:
    with pytest.raises(ValueError, match="env_spec_mode_must_be_plan"):
        validate_env_spec(load_json(ENV_SPEC), mode="realized")


def test_dotdot_still_rejected() -> None:
    _, errs = resolve_under_canonical(
        root=_ROOT,
        local_path="Docs/results/ms_sta_v5_f0/weights/../outside.bin",
        canonical_dir_rel="Docs/results/ms_sta_v5_f0/weights",
    )
    assert "dotdot_forbidden" in errs
    _, eerrs = resolve_evidence_path(
        root=_ROOT,
        rel="Docs/results/ms_sta_v5_f0/evidence/../outside.log",
        expected_basename="outside.log",
    )
    assert "evidence_dotdot_forbidden" in eerrs


def test_spoofed_interpreter_rejected() -> None:
    errs = require_isolated_interpreter(
        r"C:/malicious/_venvs/f0_v5_evil/python.exe", root=_ROOT
    )
    assert any("interpreter_not_exact_f0_venv" in e for e in errs)


def test_snapshot_and_index_policy_sealed() -> None:
    assert sha256_file(_ROOT / SNAPSHOT_CONTRACT_REL) == EXPECTED_SNAPSHOT_CONTRACT_SHA256
    assert sha256_file(_ROOT / INDEX_POLICY_REL) == EXPECTED_INDEX_POLICY_SHA256


def test_snapshot_contract_optional_bypass_rejected(tmp_path: Path) -> None:
    plan = load_json(PLAN)
    _seed_policy_files(tmp_path)
    entries = []
    for cid in EXPECTED_CANDIDATES:
        f = tmp_path / "Docs/results/ms_sta_v5_f0/weights" / cid / "x.bin"
        digest = _write(f, b"x")
        entries.append(
            {
                "candidate_id": cid,
                "role": "weights",
                "local_path": f"Docs/results/ms_sta_v5_f0/weights/{cid}/x.bin",
                "sha256": digest,
                "url": "https://example.invalid",
                "license": "NOPE",
                "revision": None,
                "model_identifier": "wrong",
            }
        )
    manifest = {
        "schema": "ms_sta_v5_f0_weights_manifest_v1",
        "approval_sha256": EXPECTED_APPROVAL_SHA256,
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "entries": entries,
    }
    with pytest.raises(ValueError, match="snapshot_contract_binding_required"):
        validate_weights_manifest(
            manifest, root=tmp_path, plan=plan, plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256
        )


def test_hf_repo_id_required(tmp_path: Path) -> None:
    plan = load_json(PLAN)
    _seed_policy_files(tmp_path)
    snap = load_json(tmp_path / SNAPSHOT_CONTRACT_REL)
    entries = []
    for cid, csnap in snap["candidates"].items():
        for fspec in csnap["files"]:
            local = fspec["local_name"]
            path = tmp_path / "Docs/results/ms_sta_v5_f0/weights" / cid / local
            digest = _write(path, b"x")
            pc = next(c for c in plan["candidates"] if c["id"] == cid)
            entry = {
                "candidate_id": cid,
                "role": fspec["role"],
                "local_path": f"Docs/results/ms_sta_v5_f0/weights/{cid}/{local}",
                "sha256": digest,
                "license": pc["pretrained_checkpoint"]["license"],
                "model_identifier": pc["model_identifier"],
            }
            if "url" in fspec:
                entry["url"] = fspec["url"]
                entry["revision"] = None
            if "hf_path" in fspec:
                entry["hf_path"] = fspec["hf_path"]
                entry["revision"] = csnap["revision"]
                # Intentionally omit repo_id (bypass proof).
            entries.append(entry)
    manifest = {
        "schema": "ms_sta_v5_f0_weights_manifest_v1",
        "approval_sha256": EXPECTED_APPROVAL_SHA256,
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "bind_snapshot_contract": True,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "entries": entries,
    }
    with pytest.raises(ValueError, match="repo_id_mismatch_vs_snapshot"):
        validate_weights_manifest(
            manifest, root=tmp_path, plan=plan, plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256
        )


def test_execution_lock_must_not_bind_audit() -> None:
    with pytest.raises(ValueError, match="execution_lock_must_not_bind_audit_sha"):
        validate_execution_lock(
            {
                "schema": "ms_sta_v5_f0_execution_lock_v1",
                "status": "EXECUTION_LOCKED",
                "plan_sha256": "a" * 64,
                "weights_manifest_sha256": "b" * 64,
                "env_realization_sha256": "c" * 64,
                "realization_manifest_sha256": "d" * 64,
                "executable_tree_sha256": "e" * 64,
                "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
                "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
                "execution_approval_sha256": "f" * 64,
                "execution_audit_sha256": "1" * 64,
            },
            plan_sha256="a" * 64,
            weights_manifest_sha256="b" * 64,
            env_realization_sha256="c" * 64,
            realization_manifest_sha256="d" * 64,
            executable_tree_sha256="e" * 64,
            execution_approval_sha256="f" * 64,
        )


def test_execution_audit_must_not_authorize_gpu() -> None:
    with pytest.raises(ValueError, match="execution_audit_must_not_authorize_gpu"):
        validate_execution_audit(
            {
                "schema": "ms_sta_v5_f0_execution_audit_v1",
                "audit_verdict": "PASS",
                "gpu_authorized": True,
                "auditor": "x",
                "audited_at": "2026-08-13T00:00:00Z",
                "execution_approval_sha256": "a" * 64,
                "execution_lock_sha256": "b" * 64,
                "executable_tree_sha256": "c" * 64,
            },
            execution_approval_sha256="a" * 64,
            execution_lock_sha256="b" * 64,
            executable_tree_sha256="c" * 64,
        )


def test_execution_seal_requires_gpu_flag() -> None:
    with pytest.raises(ValueError, match="execution_seal_gpu_authorized_required"):
        validate_execution_seal(
            {
                "schema": "ms_sta_v5_f0_execution_seal_v1",
                "status": "SEALED",
                "gpu_authorized": False,
                "one_shot": True,
                "operator": "x",
                "sealed_at": "2026-08-13T00:00:00Z",
                "execution_approval_sha256": "a" * 64,
                "execution_lock_sha256": "b" * 64,
                "execution_audit_sha256": "c" * 64,
                "executable_tree_sha256": "d" * 64,
            },
            execution_approval_sha256="a" * 64,
            execution_lock_sha256="b" * 64,
            execution_audit_sha256="c" * 64,
            executable_tree_sha256="d" * 64,
        )


def test_execution_approval_arbitrary_sha_rejected() -> None:
    with pytest.raises(ValueError) as exc:
        validate_execution_approval(
            {
                "schema": "ms_sta_v5_f0_execution_approval_v1",
                "status": "APPROVED",
                "decision": "APPROVED",
                "one_shot": True,
                "allow_gpu_feasibility": True,
                "scope": "backbone_feasibility_gpu_only",
                "operator": "attacker",
                "approved_at": "2026-08-13T00:00:00Z",
                "allow_accuracy": False,
                "allow_d0": False,
                "allow_training": False,
                "allow_hpo": False,
                "allow_test": False,
                "plan_sha256": "a" * 64,
                "env_realization_sha256": "b" * 64,
                "realization_manifest_sha256": "c" * 64,
                "weights_manifest_sha256": "d" * 64,
                "executable_tree_sha256": "e" * 64,
                "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
                "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
                "requirements_lock_sha256": EXPECTED_REQUIREMENTS_LOCK_SHA256,
                "realization_manifest_audit_sha256": "1" * 64,
                "videomae_small_conversion_evidence_sha256": "2" * 64,
            },
            root=_ROOT,
            plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
            env_realization_sha256="b" * 64,
            realization_manifest_sha256="c" * 64,
            weights_manifest_sha256="d" * 64,
            executable_tree_sha256="e" * 64,
            realization_manifest_audit_sha256="1" * 64,
            videomae_small_conversion_evidence_sha256="2" * 64,
        )
    assert "execution_approval_bad_plan_sha256" in str(exc.value)


def test_realization_manifest_audit_missing_or_mutated_rejected() -> None:
    from ms_sta_v5_f0_lib import (
        REALIZATION_MANIFEST_AUDIT_REL,
        REALIZATION_REL,
        WEIGHTS_REL,
        ENV_REALIZATION_REL,
        load_json,
        sha256_file,
        validate_realization_manifest_audit,
    )

    audit = load_json(_ROOT / REALIZATION_MANIFEST_AUDIT_REL)
    out = validate_realization_manifest_audit(
        audit,
        root=_ROOT,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        weights_manifest_sha256=sha256_file(_ROOT / WEIGHTS_REL),
        env_realization_sha256=sha256_file(_ROOT / ENV_REALIZATION_REL),
        realization_manifest_sha256=sha256_file(_ROOT / REALIZATION_REL),
    )
    assert out["ok"] is True
    bad = dict(audit)
    bad["realization_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="realization_manifest_sha256_mismatch"):
        validate_realization_manifest_audit(
            bad,
            root=_ROOT,
            plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
            plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
            weights_manifest_sha256=sha256_file(_ROOT / WEIGHTS_REL),
            env_realization_sha256=sha256_file(_ROOT / ENV_REALIZATION_REL),
            realization_manifest_sha256=sha256_file(_ROOT / REALIZATION_REL),
        )


def test_videomae_conversion_evidence_historical_f0_immutable() -> None:
    """Original F0 conversion evidence bytes stay immutable even after R1 remediation."""
    from ms_sta_v5_f0_lib import VIDEOMAE_SMALL_CONVERSION_EVIDENCE_REL, load_json, sha256_file

    path = _ROOT / VIDEOMAE_SMALL_CONVERSION_EVIDENCE_REL
    obj = load_json(path)
    assert obj["status"] == "VERIFIED"
    assert obj["converted_sha256"] == (
        "aeae93f62f724e206527b88c36cdfc8d80613a2aac57839f9ce21314ca33428b"
    )
    assert "safetensors_metadata" not in obj
    # Digest must remain the FAIL-era artifact; do not recompute against live weights.
    assert sha256_file(path) == (
        "e2bb8562006d25800ffc7704aa46447acfb314f6c21068a42e7fa7848328715d"
    )


@pytest.mark.requires_artifacts
def test_videomae_conversion_evidence_valid_and_rejects_tamper() -> None:
    from f0_feasibility.campaign_paths import F0_R1_PATHS
    from ms_sta_v5_f0_lib import (
        load_json,
        sha256_file,
        validate_videomae_small_conversion_evidence,
    )

    r1_ev = _ROOT / F0_R1_PATHS.conversion_evidence_rel
    r1_w = _ROOT / F0_R1_PATHS.weights_manifest_rel
    if not r1_ev.is_file() or not r1_w.is_file():
        pytest.skip("F0-R1 conversion evidence not materialized yet")

    obj = load_json(r1_ev)
    wsha = sha256_file(r1_w)
    out = validate_videomae_small_conversion_evidence(
        obj,
        root=_ROOT,
        weights_manifest_sha256=wsha,
        recompute_tensors=True,
        weights_manifest_rel=F0_R1_PATHS.weights_manifest_rel,
    )
    assert out["tensors_bit_exact"] is True
    assert obj.get("safetensors_metadata") == {"format": "pt"}
    bad = dict(obj)
    bad["tensors_bit_exact"] = False
    with pytest.raises(ValueError, match="tensors_not_bit_exact"):
        validate_videomae_small_conversion_evidence(
            bad,
            root=_ROOT,
            weights_manifest_sha256=wsha,
            recompute_tensors=True,
            weights_manifest_rel=F0_R1_PATHS.weights_manifest_rel,
        )
    bad2 = dict(obj)
    bad2["converted_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="converted_sha_mismatch"):
        validate_videomae_small_conversion_evidence(
            bad2,
            root=_ROOT,
            weights_manifest_sha256=wsha,
            recompute_tensors=False,
            weights_manifest_rel=F0_R1_PATHS.weights_manifest_rel,
        )


def test_slowfast_module_boundary_matcher_nested_false_positive() -> None:
    from f0_feasibility.contracts import (
        SLOWFAST_TRAINABLE_BLOCKS,
        any_module_prefix,
        matches_module_prefix,
    )

    nested = "blocks.3.multipathway_blocks.0.res_blocks.0.branch1.conv.weight"
    assert matches_module_prefix(nested, "blocks.3") is True
    assert matches_module_prefix(nested, "blocks.0") is False
    assert "blocks.0" in nested  # documents the old substring false-positive
    assert any_module_prefix([nested], "blocks.0") is False
    assert any_module_prefix([nested], "blocks.3") is True

    sample = [
        "blocks.3.multipathway_blocks.0.res_blocks.0.branch1.conv.weight",
        "blocks.4.multipathway_blocks.0.res_blocks.0.branch1.conv.weight",
        "blocks.5.proj.weight",
    ]
    for prefix in SLOWFAST_TRAINABLE_BLOCKS["frozen"]:
        assert any_module_prefix(sample, prefix) is False
    for prefix in SLOWFAST_TRAINABLE_BLOCKS["trainable"]:
        assert any_module_prefix(sample, prefix) is True


def test_f0_final_audit_fail_immutable() -> None:
    from ms_sta_v5_f0_lib import load_json

    audit = load_json(_ROOT / "Docs/results/ms_sta_v5_f0/F0_FINAL_AUDIT.json")
    assert audit["audit_verdict"] == "FAIL"
    assert audit["report_ok"] is False
    assert audit["summary"]["fail_count"] == 2
    assert audit["summary"]["pass_count"] == 2


def test_gpu_gate_rejects_cpu_and_refuse_reasons() -> None:
    pf = {
        "execution_ok": True,
        "side_effect_allowed": True,
        "gpu_authorized": True,
        "refuse_reasons": [],
    }
    ok, reasons = gpu_run_authorization_gate(pf, device="cpu")
    assert ok is False
    assert "device_not_cuda" in reasons

    pf2 = {
        "execution_ok": True,
        "side_effect_allowed": True,
        "gpu_authorized": True,
        "refuse_reasons": ["dirty_worktree"],
    }
    ok2, reasons2 = gpu_run_authorization_gate(pf2, device="cuda")
    assert ok2 is False
    assert "refuse_reasons_nonempty" in reasons2

    pf3 = {
        "execution_ok": True,
        "side_effect_allowed": False,
        "gpu_authorized": False,
        "refuse_reasons": ["x"],
    }
    ok3, reasons3 = gpu_run_authorization_gate(pf3, device="cuda")
    assert ok3 is False
    assert "side_effect_not_allowed" in reasons3
    assert "gpu_not_authorized" in reasons3


def test_feasibility_implementation_surface_and_gpu_refuse() -> None:
    proc = _run([sys.executable, str(_ROOT / "scripts/run_ms_sta_v5_f0_feasibility.py")])
    assert proc.returncode == 0, proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["status"] == "IMPLEMENTATION_PRESENT_AWAITING_EXECUTION_SEAL"
    assert "require_cuda_device" in payload["implementation_surface"]["entrypoints"]
    assert "set_grad_checkpointing" in payload["implementation_surface"]["entrypoints"]
    assert payload["implementation_surface"]["cpu_forbidden"] is True
    # SlowFast modes must not request grad ckpt true.
    sf_modes = payload["measurement_by_candidate"]["slowfast_r50_partial_unfreeze"]["modes"]
    assert all(m["grad_checkpointing_requested"] is False for m in sf_modes)
    gpu = _run(
        [
            sys.executable,
            str(_ROOT / "scripts/run_ms_sta_v5_f0_feasibility.py"),
            "--run-gpu",
            "--device",
            "cpu",
        ]
    )
    assert gpu.returncode == 2
    refuse = json.loads(gpu.stdout)
    assert refuse["refused"] is True
    assert "device_not_cuda" in refuse["gate_reasons"]
    assert refuse["gpu_authorized"] is False


def test_dry_plan_realization_gate_wording_and_gpu_refuse() -> None:
    dry = _run([sys.executable, str(_ROOT / "scripts/run_ms_sta_v5_f0.py"), "--dry-plan"])
    assert dry.returncode == 0, dry.stdout
    payload = json.loads(dry.stdout)
    assert payload["schema"] == "ms_sta_v5_f0_dry_plan_v5"
    assert "F0_FINAL_AUDIT_recorded" in payload["next_required"]
    assert "STOP_no_d0_auto_chain" in payload["next_required"]
    assert "separate_d0_authorization_only_after_review" in payload["next_required"]
    assert payload["plan_sha256"] == EXPECTED_PROTOCOL_PLAN_SHA256
    gpu = _run([sys.executable, str(_ROOT / "scripts/run_ms_sta_v5_f0.py"), "--action", "gpu"])
    assert gpu.returncode == 2


def test_realization_approval_validates() -> None:
    from ms_sta_v5_f0_lib import (
        REALIZATION_APPROVAL_REL,
        load_json,
        validate_realization_approval,
    )

    out = validate_realization_approval(
        load_json(_ROOT / REALIZATION_APPROVAL_REL), root=_ROOT
    )
    assert out["ok"] is True


def test_realization_approval_rejects_gpu_flag() -> None:
    from ms_sta_v5_f0_lib import (
        REALIZATION_APPROVAL_REL,
        load_json,
        validate_realization_approval,
    )

    obj = load_json(_ROOT / REALIZATION_APPROVAL_REL)
    obj["allow_gpu_feasibility"] = True
    with pytest.raises(ValueError, match="allow_gpu_feasibility_must_be_false"):
        validate_realization_approval(obj, root=_ROOT)


def test_contracts_no_mojibake() -> None:
    text = (_ROOT / "scripts/f0_feasibility/contracts.py").read_text(encoding="utf-8")
    assert "â€”" not in text
    assert "—" not in text  # avoid em-dash encoding issues on Windows


def test_grad_ckpt_unsupported_not_reported_applied() -> None:
    from f0_feasibility.gpu_runner import set_grad_checkpointing

    class Dummy:
        pass

    meta = set_grad_checkpointing(Dummy(), requested=True)
    assert meta["requested"] is True
    assert meta["supported"] is False
    assert meta["applied"] is False


def test_videomae_tuple_mismatch_accepted() -> None:
    from f0_feasibility.contracts import CLASSIFIER_KEY_ALLOWLIST
    from f0_feasibility.gpu_runner import assert_videomae_loading_info

    allowed = CLASSIFIER_KEY_ALLOWLIST["videomae_vit_small"]
    out = assert_videomae_loading_info(
        {
            "error_msgs": [],
            "mismatched_keys": [
                ("classifier.weight", (400, 384), (100, 384)),
                ("classifier.bias", (400,), (100,)),
            ],
            "missing_keys": [],
            "unexpected_keys": [],
        },
        allowed=allowed,
    )
    assert out["mismatched_keys"] == sorted(allowed)


def test_videomae_loading_info_rejects_non_classifier() -> None:
    from f0_feasibility.contracts import CLASSIFIER_KEY_ALLOWLIST
    from f0_feasibility.gpu_runner import assert_videomae_loading_info

    with pytest.raises(RuntimeError, match="videomae_loading_info_non_classifier"):
        assert_videomae_loading_info(
            {
                "error_msgs": [],
                "mismatched_keys": [
                    ("classifier.weight", (400, 384), (100, 384)),
                    ("videomae.encoder.layer.0.weight", (1,), (1,)),
                ],
                "missing_keys": [],
                "unexpected_keys": [],
            },
            allowed=CLASSIFIER_KEY_ALLOWLIST["videomae_vit_small"],
        )


def test_videomae_error_msgs_rejected() -> None:
    from f0_feasibility.contracts import CLASSIFIER_KEY_ALLOWLIST
    from f0_feasibility.gpu_runner import assert_videomae_loading_info

    with pytest.raises(RuntimeError, match="videomae_loading_error_msgs"):
        assert_videomae_loading_info(
            {
                "error_msgs": ["boom"],
                "mismatched_keys": [
                    ("classifier.weight", (400, 384), (100, 384)),
                    ("classifier.bias", (400,), (100,)),
                ],
                "missing_keys": [],
                "unexpected_keys": [],
            },
            allowed=CLASSIFIER_KEY_ALLOWLIST["videomae_vit_small"],
        )


def test_per_candidate_failure_keeps_four_results(tmp_path: Path) -> None:
    from f0_feasibility.gpu_runner import run_all_candidates_offline

    # Force cuda check to pass by monkeypatching require_cuda_device and run_candidate.
    import f0_feasibility.gpu_runner as gr

    calls = {"n": 0}

    def _fake_cuda(device: str) -> str:
        return "cuda"

    def _fake_run(cid, **kwargs):
        calls["n"] += 1
        if cid == "video_swin_tiny":
            raise RuntimeError("forced_swin_failure")
        return {"candidate_id": cid, "ok": True, "modes": []}

    monkey = pytest.MonkeyPatch()
    monkey.setattr(gr, "require_cuda_device", _fake_cuda)
    monkey.setattr(gr, "run_candidate_feasibility", _fake_run)
    monkey.setattr(gr, "candidate_snapshot_dir", lambda root, cid: tmp_path / cid)
    monkey.setattr(gr, "list_required_local_files", lambda snap, cid: [])
    monkey.setattr(gr, "assert_local_snapshot_complete", lambda d, n: None)
    try:
        for cid in (
            "slowfast_r50_partial_unfreeze",
            "videomae_vit_small",
            "videomae_vit_base",
            "video_swin_tiny",
        ):
            (tmp_path / cid).mkdir()
        report = run_all_candidates_offline(
            weights_root=tmp_path,
            snapshot_contract={"candidates": {}},
            device="cuda",
        )
    finally:
        monkey.undo()
    assert len(report["candidates"]) == 4
    assert [c["candidate_id"] for c in report["candidates"]] == [
        "slowfast_r50_partial_unfreeze",
        "videomae_vit_small",
        "videomae_vit_base",
        "video_swin_tiny",
    ]
    assert report["ok"] is False
    swin = report["candidates"][3]
    assert swin["ok"] is False
    assert "forced_swin_failure" in swin["error"]


def _mode_payload(cid: str) -> list[dict]:
    from f0_feasibility.contracts import CANDIDATE_MEASUREMENT, measurement_modes_for
    from f0_feasibility.gpu_runner import MEASUREMENT

    timed = int(MEASUREMENT["timed_steps"])
    all_ms = [1.0] * timed
    shape = [1, int(CANDIDATE_MEASUREMENT[cid]["num_classes_dummy"])]
    modes = []
    for m in measurement_modes_for(cid):
        requested = bool(m["grad_checkpointing_requested"])
        dtype = "float16" if m["use_amp"] else "float32"
        modes.append(
            {
                "precision": "amp_fp16" if m["use_amp"] else "fp32",
                "grad_checkpointing": {
                    "requested": requested,
                    "supported": True,
                    "applied": requested,
                },
                "checkpoint_invocation_count": 4 if (requested and cid == "video_swin_tiny") else 0,
                "last_loss": 0.125,
                "observed_output": {"shape": shape, "dtype": dtype},
                "step_latency_ms_distribution": {
                    "n": timed,
                    "p50_ms": 1.0,
                    "p90_ms": 1.0,
                    "mean_ms": 1.0,
                    "all_ms": all_ms,
                },
                "peak_allocated_vram_bytes": 1000,
                "peak_reserved_vram_bytes": 2000,
                "min_free_vram_bytes": 3000,
            }
        )
    return modes


def _load_meta_for(cid: str) -> dict:
    from f0_feasibility.contracts import CLASSIFIER_KEY_ALLOWLIST, SLOWFAST_TRAINABLE_BLOCKS

    if cid.startswith("videomae"):
        allowed = sorted(CLASSIFIER_KEY_ALLOWLIST[cid])
        return {
            "loader": "transformers_local_files_only",
            "loading_info": {
                "mismatched_keys": allowed,
                "missing_keys": [],
                "unexpected_keys": [],
                "allowed_keys": allowed,
                "error_msgs": [],
            },
            "head": {"head_attr": "classifier", "new_out_features": 100},
        }
    if cid == "video_swin_tiny":
        return {
            "missing_keys": [],
            "unexpected_keys": [],
            "allowed_keys": sorted(CLASSIFIER_KEY_ALLOWLIST[cid]),
            "head": {"head_attr": "head", "in_features": 768, "new_out_features": 100},
            "grad_checkpointing_api": "f0_features_checkpoint_adapter",
        }
    return {
        "missing_keys": [],
        "unexpected_keys": [],
        "allowed_keys": [],
        "head": {
            "head_attr": "blocks[-1].proj",
            "in_features": 2304,
            "new_out_features": 100,
        },
        "trainable_blocks": {
            "frozen": list(SLOWFAST_TRAINABLE_BLOCKS["frozen"]),
            "trainable": list(SLOWFAST_TRAINABLE_BLOCKS["trainable"]),
            "trainable_param_count": 3,
        },
    }


def _full_ok_candidate(cid: str) -> dict:
    from f0_feasibility.contracts import CANDIDATE_MEASUREMENT, LAYOUT_CONTRACT

    if cid == "slowfast_r50_partial_unfreeze":
        identities = ["blocks.3.a", "blocks.4.b", "blocks.5.c"]
    elif cid.startswith("videomae"):
        identities = ["classifier.weight", "classifier.bias"]
    else:
        identities = ["head.weight", "head.bias"]
    return {
        "candidate_id": cid,
        "ok": True,
        "device": "cuda",
        "offline_only": True,
        "checkpoint_save_load_bit_exact": True,
        "trainable_param_count": len(identities),
        "trainable_param_identities_sample": identities,
        "load_meta": _load_meta_for(cid),
        "output_tensor_contract": {
            "shape": [1, int(CANDIDATE_MEASUREMENT[cid]["num_classes_dummy"])],
            "dtype": "float16|float32",
        },
        "modes": _mode_payload(cid),
        "layout": LAYOUT_CONTRACT[cid]["layout"],
    }


def _full_report(*, ok: bool = True) -> dict:
    from f0_feasibility.contracts import CANDIDATE_IDS
    from f0_feasibility.gpu_runner import MEASUREMENT

    cands = [_full_ok_candidate(cid) for cid in CANDIDATE_IDS]
    if not ok:
        cands[-1] = {
            "candidate_id": CANDIDATE_IDS[-1],
            "ok": False,
            "error": "forced",
            "error_type": "RuntimeError",
            "device": "cuda",
        }
    return {
        "schema": "ms_sta_v5_f0_feasibility_cuda_report_v1",
        "ok": all(c["ok"] for c in cands),
        "oneshot": True,
        "device": "cuda",
        "network_fetch": False,
        "cpu_forbidden": True,
        "oneshot_required": True,
        "candidate_count": 4,
        "measurement_contract": dict(MEASUREMENT),
        "candidates": cands,
    }


def test_inconsistent_metrics_report_rejected() -> None:
    """Negative proof: previously accepted inconsistent metrics must fail."""
    from f0_feasibility.oneshot import validate_feasibility_report

    report = _full_report()
    # Corrupt exactly as rem-9 negative proof described.
    report["measurement_contract"] = {
        **report["measurement_contract"],
        "timed_steps": 99,  # wrong value, keys still present
    }
    c0 = report["candidates"][0]
    c0["layout"] = "x"
    c0["output_tensor_contract"]["dtype"] = "garbage"
    m0 = c0["modes"][0]
    m0["step_latency_ms_distribution"]["p50_ms"] = -999
    m0["step_latency_ms_distribution"]["mean_ms"] = 999.0  # not matching all_ms
    m0["peak_allocated_vram_bytes"] = 0
    m0["peak_reserved_vram_bytes"] = 0
    m0["min_free_vram_bytes"] = 0
    errs = validate_feasibility_report(report)
    assert errs
    joined = ";".join(errs)
    assert "report_measurement_contract_not_exact" in joined
    assert "layout_mismatch" in joined
    assert "output_dtype_invalid" in joined
    assert "latency_p50_ms_non_positive" in joined or "latency_p50_ms_inconsistent" in joined
    assert "latency_mean_ms_inconsistent" in joined
    assert "peak_allocated_vram_bytes_invalid" in joined
    # Proof flag for auditors.
    inconsistent_metrics_accepted = len(errs) == 0
    assert inconsistent_metrics_accepted is False


def test_metrics_free_report_rejected() -> None:
    from f0_feasibility.oneshot import validate_feasibility_report

    report = {
        "schema": "ms_sta_v5_f0_feasibility_cuda_report_v1",
        "ok": True,
        "oneshot": True,
        "device": "cuda",
        "network_fetch": False,
        "cpu_forbidden": True,
        "measurement_contract": {"timed_steps": 10},
        "candidates": [
            {
                "candidate_id": cid,
                "ok": True,
                "modes": [
                    {
                        "precision": "fp32",
                        "grad_checkpointing": {
                            "requested": False,
                            "supported": True,
                            "applied": False,
                        },
                    }
                ],
            }
            for cid in (
                "slowfast_r50_partial_unfreeze",
                "videomae_vit_small",
                "videomae_vit_base",
                "video_swin_tiny",
            )
        ],
    }
    # Fix mode counts minimally but keep metrics absent.
    from f0_feasibility.contracts import measurement_modes_for

    for c in report["candidates"]:
        c["modes"] = [
            {
                "precision": "amp_fp16" if m["use_amp"] else "fp32",
                "grad_checkpointing": {
                    "requested": m["grad_checkpointing_requested"],
                    "supported": True,
                    "applied": m["grad_checkpointing_requested"],
                },
            }
            for m in measurement_modes_for(c["candidate_id"])
        ]
    errs = validate_feasibility_report(report)
    assert errs
    assert any("last_loss" in e or "latency" in e or "vram" in e or "load_meta" in e for e in errs)


def test_oneshot_claim_and_second_claim_refused(tmp_path: Path) -> None:
    from f0_feasibility.oneshot import claim_execution_attempt, refuse_if_seal_consumed

    claim_execution_attempt(
        root=tmp_path,
        execution_seal_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
    )
    assert "execution_attempt_already_claimed" in refuse_if_seal_consumed(root=tmp_path)
    with pytest.raises(RuntimeError, match="execution_attempt_already_claimed"):
        claim_execution_attempt(
            root=tmp_path,
            execution_seal_sha256="a" * 64,
            executable_tree_sha256="b" * 64,
        )


def test_oneshot_publish_rejects_metrics_free(tmp_path: Path) -> None:
    from f0_feasibility.oneshot import claim_execution_attempt, publish_oneshot_artifacts

    claim_execution_attempt(
        root=tmp_path,
        execution_seal_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
    )
    bad = {
        "schema": "ms_sta_v5_f0_feasibility_cuda_report_v1",
        "ok": True,
        "oneshot": True,
        "device": "cuda",
        "network_fetch": False,
        "cpu_forbidden": True,
        "candidates": [],
    }
    with pytest.raises(ValueError, match="report_invalid"):
        publish_oneshot_artifacts(
            root=tmp_path,
            report=bad,
            execution_seal_sha256="a" * 64,
            executable_tree_sha256="b" * 64,
        )


def test_oneshot_heal_after_partial_promote(tmp_path: Path) -> None:
    from f0_feasibility.contracts import FEASIBILITY_REPORT_REL, configure_feasibility_artifacts
    from f0_feasibility.oneshot import (
        claim_execution_attempt,
        publish_oneshot_artifacts,
        verify_oneshot_artifacts,
    )

    configure_feasibility_artifacts(campaign="f0")
    claim_execution_attempt(
        root=tmp_path,
        execution_seal_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
    )
    report = _full_report()
    publish_oneshot_artifacts(
        root=tmp_path,
        report=report,
        execution_seal_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
    )
    # Simulate crash after READY by deleting one final artifact; staging remains.
    (tmp_path / FEASIBILITY_REPORT_REL).unlink()
    out = verify_oneshot_artifacts(
        root=tmp_path,
        execution_seal_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
        allow_heal=True,
    )
    assert out["ok"] is True
    assert out["healed"] is True
    assert out["inference"] is False
    assert (tmp_path / FEASIBILITY_REPORT_REL).is_file()


def test_oneshot_resume_requires_seal_and_tree(tmp_path: Path) -> None:
    from f0_feasibility.oneshot import (
        claim_execution_attempt,
        publish_oneshot_artifacts,
        verify_oneshot_artifacts,
    )

    claim_execution_attempt(
        root=tmp_path,
        execution_seal_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
    )
    publish_oneshot_artifacts(
        root=tmp_path,
        report=_full_report(),
        execution_seal_sha256="a" * 64,
        executable_tree_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="resume_requires_execution_seal_sha256"):
        verify_oneshot_artifacts(
            root=tmp_path, execution_seal_sha256="", executable_tree_sha256="b" * 64
        )
    with pytest.raises(ValueError, match="attempt_executable_tree_sha_mismatch|ready_executable"):
        verify_oneshot_artifacts(
            root=tmp_path, execution_seal_sha256="a" * 64, executable_tree_sha256="c" * 64
        )
    out = verify_oneshot_artifacts(
        root=tmp_path, execution_seal_sha256="a" * 64, executable_tree_sha256="b" * 64
    )
    assert out["ok"] is True
    assert out["execution_attempt_sha256"]


def test_duplicate_wheel_path_rejected(tmp_path: Path) -> None:
    obj = _build_valid_env_realization(tmp_path)
    wheel = tmp_path / "Docs/results/ms_sta_v5_f0/evidence/wheelhouse/torch-2.2.2+cu121.whl"
    wheel_sha = sha256_file(wheel)
    # Add second package pointing at same wheel file.
    freeze = b"torch==2.2.2+cu121\nnumpy==1.26.4\n"
    freeze_sha = _write(tmp_path / "Docs/results/ms_sta_v5_f0/evidence/pip_freeze.txt", freeze)
    obj["evidence"]["pip_freeze_txt_sha256"] = freeze_sha
    obj["pip_freeze_sha256"] = freeze_sha
    packages = [
        {
            "name": "torch",
            "version": "2.2.2+cu121",
            "sha256": wheel_sha,
            "index_url": "https://download.pytorch.org/whl/cu121",
            "wheel_rel": "Docs/results/ms_sta_v5_f0/evidence/wheelhouse/torch-2.2.2+cu121.whl",
        },
        {
            "name": "numpy",
            "version": "1.26.4",
            "sha256": wheel_sha,
            "index_url": "https://pypi.org/simple",
            "wheel_rel": "Docs/results/ms_sta_v5_f0/evidence/wheelhouse/torch-2.2.2+cu121.whl",
        },
    ]
    blob = _valid_hashes_blob(packages=packages, wheel_sha=wheel_sha)
    hsha = _write(tmp_path / "Docs/results/ms_sta_v5_f0/evidence/pip_freeze.hashes.json", blob)
    obj["evidence"]["pip_freeze_hashes_json_sha256"] = hsha
    with pytest.raises(ValueError) as exc:
        validate_env_realization(
            obj,
            root=tmp_path,
            plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
            env_spec_sha256=sha256_file(ENV_SPEC),
        )
    msg = str(exc.value)
    assert (
        "pip_freeze_hashes_duplicate_resolved_wheel" in msg
        or "pip_freeze_hashes_wheel_name_package_mismatch" in msg
    )


def test_measurement_modes_slowfast_no_grad_ckpt_true() -> None:
    from f0_feasibility.contracts import (
        assert_candidate_measurement_matches_plan,
        measurement_modes_for,
    )

    modes = measurement_modes_for("slowfast_r50_partial_unfreeze")
    assert modes
    assert all(m["grad_checkpointing_requested"] is False for m in modes)
    vmodes = measurement_modes_for("videomae_vit_small")
    assert any(m["grad_checkpointing_requested"] is True for m in vmodes)
    assert any(m["grad_checkpointing_requested"] is False for m in vmodes)
    assert_candidate_measurement_matches_plan(load_json(PLAN))


def test_swin_grad_ckpt_api_installed(monkeypatch) -> None:
    import torch.utils.checkpoint as torch_checkpoint
    from f0_feasibility.gpu_runner import install_swin_grad_checkpointing_api, set_grad_checkpointing

    calls = {"n": 0}

    def _fake_checkpoint(fn, *args, **kwargs):
        calls["n"] += 1
        return fn(*args)

    monkeypatch.setattr(torch_checkpoint, "checkpoint", _fake_checkpoint)

    class Features(list):
        def forward(self, x):
            return x

    class FakeSwin:
        def __init__(self):
            self.features = Features([lambda z: z, lambda z: z])

    model = FakeSwin()
    install_swin_grad_checkpointing_api(model)
    meta = set_grad_checkpointing(model, requested=True)
    assert meta["supported"] is True
    assert meta["applied"] is True
    assert model._f0_grad_ckpt is True
    out = model.features.forward(0)
    assert out == 0
    assert calls["n"] == 2
    assert model._f0_ckpt_invocations == 2


def test_validate_rejects_swin_applied_without_invocation() -> None:
    from f0_feasibility.oneshot import validate_feasibility_report

    report = _full_report()
    for c in report["candidates"]:
        if c["candidate_id"] != "video_swin_tiny":
            continue
        for m in c["modes"]:
            if m["grad_checkpointing"]["applied"]:
                m["checkpoint_invocation_count"] = 0
    errs = validate_feasibility_report(report)
    assert any("checkpoint_invocation_count_required" in e for e in errs)


def test_load_meta_bad_keys_and_head_rejected() -> None:
    from f0_feasibility.oneshot import validate_feasibility_report

    # 1) VideoMAE arbitrary unexpected key
    r1 = _full_report()
    for c in r1["candidates"]:
        if c["candidate_id"] == "videomae_vit_small":
            c["load_meta"]["loading_info"]["unexpected_keys"] = ["encoder.layer.0.weight"]
            break
    e1 = validate_feasibility_report(r1)
    assert any("videomae_vit_small:load_meta_keys_outside_allowlist" in x for x in e1)
    assert (len(e1) == 0) is False  # VIDEOMAE_BAD_UNEXPECTED_ACCEPTED=False

    # 2) Swin arbitrary unexpected key
    r2 = _full_report()
    for c in r2["candidates"]:
        if c["candidate_id"] == "video_swin_tiny":
            c["load_meta"]["unexpected_keys"] = ["features.0.weight"]
            break
    e2 = validate_feasibility_report(r2)
    assert any("video_swin_tiny:load_meta_keys_outside_allowlist" in x for x in e2)
    assert (len(e2) == 0) is False  # SWIN_BAD_UNEXPECTED_ACCEPTED=False

    # 3) SlowFast any missing/unexpected rejected
    r3 = _full_report()
    for c in r3["candidates"]:
        if c["candidate_id"] == "slowfast_r50_partial_unfreeze":
            c["load_meta"]["missing_keys"] = ["blocks.0.conv.weight"]
            break
    e3 = validate_feasibility_report(r3)
    assert any("slowfast_r50_partial_unfreeze:load_meta_keys_must_be_empty" in x for x in e3)
    assert (len(e3) == 0) is False  # SLOWFAST_BAD_LOAD_KEYS_ACCEPTED=False

    # 4) wrong head_attr for every candidate
    for cid in (
        "slowfast_r50_partial_unfreeze",
        "videomae_vit_small",
        "videomae_vit_base",
        "video_swin_tiny",
    ):
        r = _full_report()
        for c in r["candidates"]:
            if c["candidate_id"] == cid:
                c["load_meta"]["head"]["head_attr"] = "wrong_head"
                break
        errs = validate_feasibility_report(r)
        assert any(f"{cid}:load_meta_head_attr_invalid" in x for x in errs)


def test_observed_bfloat16_rejected() -> None:
    from f0_feasibility.oneshot import validate_feasibility_report

    report = _full_report()
    report["candidates"][0]["modes"][0]["observed_output"]["dtype"] = "bfloat16"
    errs = validate_feasibility_report(report)
    assert any("observed_output_dtype_invalid" in e for e in errs)
    undeclared_bfloat16_accepted = len(errs) == 0
    assert undeclared_bfloat16_accepted is False
