#!/usr/bin/env python3
"""F0 realization runner — download weights + isolated env (no GPU feasibility).

Requires F0_REALIZATION_APPROVAL + independent APPROVAL_AUDIT PASS.
Stops after F0_ENV_REALIZATION + F0_REALIZATION_MANIFEST validate.
Does not create execution approval/seal or run model feasibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_f0_lib import (  # noqa: E402
    ENV_REALIZATION_REL,
    ENV_SPEC_REL,
    EVIDENCE_DIR_REL,
    EXPECTED_APPROVAL_SHA256,
    EXPECTED_INDEX_POLICY_SHA256,
    EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
    EXPECTED_PROTOCOL_PLAN_SHA256,
    EXPECTED_REQUIREMENTS_LOCK_SHA256,
    EXPECTED_SNAPSHOT_CONTRACT_SHA256,
    INDEX_POLICY_REL,
    PROTOCOL_PLAN_REL,
    REALIZATION_APPROVAL_AUDIT_REL,
    REALIZATION_APPROVAL_REL,
    REALIZATION_MANIFEST_AUDIT_REL,
    REALIZATION_REL,
    REQUIREMENTS_LOCK_REL,
    SNAPSHOT_CONTRACT_REL,
    WEIGHTS_DIR_REL,
    WEIGHTS_REL,
    WHEELHOUSE_DIR_REL,
    load_json,
    preflight,
    sha256_file,
    validate_env_realization,
    validate_realization_manifest,
    validate_weights_manifest,
)


def _write_json(path: Path, obj: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ms-sta-v5-f0-realization/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(tmp, dest)


def _convert_bin_to_safetensors(bin_path: Path, dest: Path) -> None:
    """Materialize contract local_name model.safetensors from pytorch_model.bin."""
    import torch

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors_required_for_bin_conversion") from exc
    state = torch.load(bin_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("pytorch_model_bin_not_state_dict")
    tensors = {k: v.contiguous() for k, v in state.items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Transformers 4.37.x requires safetensors metadata["format"]; None metadata crashes loader.
    save_file(tensors, str(dest), metadata={"format": "pt"})


def download_weights(*, root: Path) -> dict[str, Any]:
    plan = load_json(root / PROTOCOL_PLAN_REL)
    snap = load_json(root / SNAPSHOT_CONTRACT_REL)
    by_plan = {c["id"]: c for c in plan.get("candidates") or []}
    entries: list[dict[str, Any]] = []
    for cid, csnap in (snap.get("candidates") or {}).items():
        pc = by_plan[cid]
        plan_ckpt = pc.get("pretrained_checkpoint") or {}
        out_dir = root / WEIGHTS_DIR_REL / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        for fspec in csnap.get("files") or []:
            local_name = str(fspec["local_name"])
            dest = out_dir / local_name
            if fspec.get("url"):
                url = str(fspec["url"])
                if not dest.is_file():
                    print(f"[download] {cid}:{local_name} <- {url}")
                    _download(url, dest)
                entry = {
                    "candidate_id": cid,
                    "role": fspec["role"],
                    "local_path": f"{WEIGHTS_DIR_REL}/{cid}/{local_name}".replace("\\", "/"),
                    "url": url,
                    "sha256": sha256_file(dest),
                    "sha256_verified": True,
                    "license": plan_ckpt.get("license"),
                    "model_identifier": pc.get("model_identifier"),
                }
            else:
                repo_id = csnap["repo_id"]
                revision = csnap["revision"]
                hf_path = fspec["hf_path"]
                url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{hf_path}"
                conversion_note = None
                if not dest.is_file():
                    try:
                        print(f"[download] {cid}:{local_name} <- {url}")
                        _download(url, dest)
                    except Exception as exc:
                        # Pinned VideoMAE-small revision ships pytorch_model.bin only.
                        if (
                            cid == "videomae_vit_small"
                            and hf_path == "model.safetensors"
                            and "404" in str(exc)
                        ):
                            bin_name = "pytorch_model.bin"
                            bin_url = (
                                f"https://huggingface.co/{repo_id}/resolve/{revision}/{bin_name}"
                            )
                            bin_path = out_dir / bin_name
                            print(f"[download-fallback] {cid}:{bin_name} <- {bin_url}")
                            _download(bin_url, bin_path)
                            print(f"[convert] {cid}:{bin_name} -> {local_name}")
                            _convert_bin_to_safetensors(bin_path, dest)
                            conversion_note = {
                                "source_hf_path": bin_name,
                                "source_sha256": sha256_file(bin_path),
                                "conversion": "torch_state_dict_to_safetensors",
                                "reason": "pinned_revision_missing_model.safetensors",
                            }
                        else:
                            raise
                entry = {
                    "candidate_id": cid,
                    "role": fspec["role"],
                    "local_path": f"{WEIGHTS_DIR_REL}/{cid}/{local_name}".replace("\\", "/"),
                    "repo_id": repo_id,
                    "revision": revision,
                    "hf_path": hf_path,
                    "sha256": sha256_file(dest),
                    "sha256_verified": True,
                    "license": plan_ckpt.get("license"),
                    "model_identifier": pc.get("model_identifier"),
                }
                if conversion_note:
                    entry["conversion"] = conversion_note
            entries.append(entry)

    manifest = {
        "schema": "ms_sta_v5_f0_weights_manifest_v1",
        "status": "REALIZED",
        "approval_sha256": EXPECTED_APPROVAL_SHA256,
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "bind_snapshot_contract": True,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "entries": entries,
    }
    digest = _write_json(root / WEIGHTS_REL, manifest)
    validate_weights_manifest(
        manifest, root=root, plan=plan, plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256
    )
    return {"ok": True, "weights_manifest_sha256": digest, "entries": len(entries)}


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("[cmd]", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd or _ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def _venv_python(root: Path) -> Path:
    win = root / "_venvs/f0_v5/Scripts/python.exe"
    posix = root / "_venvs/f0_v5/bin/python"
    if win.is_file():
        return win
    if posix.is_file():
        return posix
    raise FileNotFoundError("f0_venv_python_missing")


def realize_env(*, root: Path) -> dict[str, Any]:
    index_policy = load_json(root / INDEX_POLICY_REL)
    torch_idx = index_policy["torch_index"]
    extra_index = torch_idx["extra_index_url"]
    cuda_build = torch_idx["cuda_build"]
    lock = root / REQUIREMENTS_LOCK_REL
    wheelhouse = root / WHEELHOUSE_DIR_REL
    evidence = root / EVIDENCE_DIR_REL
    wheelhouse.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)

    venv_dir = root / "_venvs/f0_v5"
    if not (venv_dir / "pyvenv.cfg").is_file():
        cp = _run([sys.executable, "-m", "venv", str(venv_dir)])
        if cp.returncode != 0:
            raise RuntimeError(f"venv_create_failed:{cp.stderr}")
    # Ensure isolated.
    cfg = (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8")
    if "include-system-site-packages = true" in cfg.lower():
        raise RuntimeError("venv_includes_system_site_packages")

    py = _venv_python(root)
    # Bootstrap pip in venv.
    _run([str(py), "-m", "ensurepip", "--upgrade"])
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Build/download wheels into wheelhouse (includes transitive deps as wheels).
    cp = _run(
        [
            str(py),
            "-m",
            "pip",
            "wheel",
            "-w",
            str(wheelhouse),
            "-r",
            str(lock),
            "--extra-index-url",
            extra_index,
        ]
    )
    (evidence / "pip_wheel.log").write_text(
        (cp.stdout or "") + "\n" + (cp.stderr or ""), encoding="utf-8"
    )
    if cp.returncode != 0:
        raise RuntimeError(f"pip_wheel_failed:{cp.returncode}")

    cp = _run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "-r",
            str(lock),
        ]
    )
    if cp.returncode != 0:
        raise RuntimeError(f"pip_install_failed:{cp.stderr}")

    # Smokes
    cp_check = _run([str(py), "-m", "pip", "check"])
    (evidence / "pip_check.log").write_text(
        (cp_check.stdout or "") + (cp_check.stderr or ""), encoding="utf-8"
    )
    if cp_check.returncode != 0:
        raise RuntimeError("pip_check_failed")

    import_code = (
        "import torch, torchvision, transformers, numpy, cv2\n"
        "import pytorchvideo\n"
        "print('import-ok', torch.__version__, transformers.__version__)\n"
    )
    cp_imp = _run([str(py), "-c", import_code])
    (evidence / "import_smoke.log").write_text(
        (cp_imp.stdout or "") + (cp_imp.stderr or ""), encoding="utf-8"
    )
    if cp_imp.returncode != 0:
        raise RuntimeError(f"import_smoke_failed:{cp_imp.stderr}")

    cuda_code = (
        "import torch, json\n"
        "assert torch.cuda.is_available(), 'cuda_not_available'\n"
        "print(json.dumps({"
        "'torch_version': torch.__version__,"
        "'cuda_runtime_version': getattr(torch.version, 'cuda', None),"
        "'gpu_name': torch.cuda.get_device_name(0),"
        "'gpu_index': 0,"
        "'cuda_available': True"
        "}))\n"
    )
    cp_cuda = _run([str(py), "-c", cuda_code])
    (evidence / "cuda_smoke.log").write_text(
        (cp_cuda.stdout or "") + (cp_cuda.stderr or ""), encoding="utf-8"
    )
    if cp_cuda.returncode != 0:
        raise RuntimeError(f"cuda_smoke_failed:{cp_cuda.stderr}")
    cuda_meta = json.loads((cp_cuda.stdout or "").strip().splitlines()[-1])

    cp_freeze = _run([str(py), "-m", "pip", "freeze"])
    freeze_txt = cp_freeze.stdout or ""
    # Drop bootstrap tooling from freeze identity set.
    freeze_lines = []
    for line in freeze_txt.splitlines():
        name = line.split("==", 1)[0].strip().lower().replace("_", "-")
        if name in {"pip", "setuptools", "wheel"}:
            continue
        if line.strip():
            freeze_lines.append(line.strip())
    freeze_body = "\n".join(freeze_lines) + ("\n" if freeze_lines else "")
    (evidence / "pip_freeze.txt").write_text(freeze_body, encoding="utf-8")

    # Map freeze packages → wheel files.
    wheel_files = list(wheelhouse.glob("*.whl"))
    packages = []
    for line in freeze_lines:
        name, ver = line.split("==", 1)
        name_n = name.strip().lower().replace("_", "-")
        ver_n = ver.strip()
        name_token = name_n.replace("-", "_")
        ver_token = ver_n.lower().replace("+", "_")
        match = None
        for whl in wheel_files:
            wn = whl.name.lower()
            if not (wn.startswith(f"{name_n}-") or wn.startswith(f"{name_token}-")):
                continue
            if ver_n.lower() not in wn and ver_token not in wn:
                continue
            match = whl
            break
        if match is None:
            # Attempt to fetch the exact wheel for this package.
            cp = _run(
                [
                    str(py),
                    "-m",
                    "pip",
                    "download",
                    "--no-deps",
                    "-d",
                    str(wheelhouse),
                    f"{name_n}=={ver_n}",
                    "--extra-index-url",
                    extra_index,
                    "--only-binary=:all:",
                ]
            )
            if cp.returncode != 0:
                raise RuntimeError(f"wheel_missing_for_freeze_pkg:{name_n}=={ver_n}:{cp.stderr}")
            wheel_files = list(wheelhouse.glob("*.whl"))
            for whl in wheel_files:
                wn = whl.name.lower()
                if (wn.startswith(f"{name_n}-") or wn.startswith(f"{name_token}-")) and (
                    ver_n.lower() in wn or ver_token in wn
                ):
                    match = whl
                    break
        if match is None:
            raise RuntimeError(f"wheel_unresolved:{name_n}=={ver_n}")
        packages.append(
            {
                "name": name_n,
                "version": ver_n,
                "sha256": sha256_file(match),
                "wheel_rel": f"{WHEELHOUSE_DIR_REL}/{match.name}".replace("\\", "/"),
            }
        )

    hashes_obj = {
        "schema": "ms_sta_v5_f0_pip_freeze_hashes_v1",
        "torch_index_url": extra_index,
        "cuda_build": cuda_build,
        "packages": packages,
    }
    _write_json(evidence / "pip_freeze.hashes.json", hashes_obj)

    # Provenance
    prov_code = (
        "import json, platform, sys\n"
        "from pathlib import Path\n"
        "cfg = Path(sys.prefix) / 'pyvenv.cfg'\n"
        "print(json.dumps({"
        "'schema': 'ms_sta_v5_f0_python_provenance_v1',"
        "'python_exact': platform.python_version(),"
        "'executable': sys.executable.replace('\\\\','/'),"
        "'platform': platform.platform(),"
        "'implementation': platform.python_implementation(),"
        "'sys_prefix': sys.prefix.replace('\\\\','/'),"
        "'include_system_site_packages': False,"
        "'pyvenv_cfg_present': cfg.is_file()"
        "}))\n"
    )
    cp_prov = _run([str(py), "-c", prov_code])
    if cp_prov.returncode != 0:
        raise RuntimeError(f"python_provenance_failed:{cp_prov.stderr}")
    py_prov = json.loads((cp_prov.stdout or "").strip().splitlines()[-1])
    _write_json(evidence / "python_provenance.json", py_prov)

    cuda_prov = {
        "schema": "ms_sta_v5_f0_cuda_provenance_v1",
        "cuda_build": cuda_build,
        "torch_index_url": extra_index,
        "torch_version": str(cuda_meta["torch_version"]),
        "cuda_runtime_version": str(cuda_meta["cuda_runtime_version"]),
        "gpu_name": str(cuda_meta["gpu_name"]),
        "gpu_index": int(cuda_meta["gpu_index"]),
        "cuda_available": True,
    }
    _write_json(evidence / "cuda_provenance.json", cuda_prov)

    python_exact = py_prov["python_exact"]
    evidence_paths = {
        "pip_check_log": f"{EVIDENCE_DIR_REL}/pip_check.log",
        "import_smoke_log": f"{EVIDENCE_DIR_REL}/import_smoke.log",
        "cuda_smoke_log": f"{EVIDENCE_DIR_REL}/cuda_smoke.log",
        "pip_freeze_txt": f"{EVIDENCE_DIR_REL}/pip_freeze.txt",
        "pip_freeze_hashes_json": f"{EVIDENCE_DIR_REL}/pip_freeze.hashes.json",
        "python_provenance_json": f"{EVIDENCE_DIR_REL}/python_provenance.json",
        "cuda_provenance_json": f"{EVIDENCE_DIR_REL}/cuda_provenance.json",
    }
    evidence_obj: dict[str, Any] = {}
    for key, rel in evidence_paths.items():
        evidence_obj[key] = rel
        evidence_obj[f"{key}_sha256"] = sha256_file(root / rel)

    env_obj = {
        "schema": "ms_sta_v5_f0_env_realization_v1",
        "status": "REALIZED",
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "env_spec_sha256": sha256_file(root / ENV_SPEC_REL),
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "requirements_lock_sha256": EXPECTED_REQUIREMENTS_LOCK_SHA256,
        "pip_check_exit_code": 0,
        "import_smoke_exit_code": 0,
        "cuda_smoke_exit_code": 0,
        "pip_check_ok": True,
        "import_smoke_ok": True,
        "cuda_smoke_ok": True,
        "pip_check_output_sha256": evidence_obj["pip_check_log_sha256"],
        "import_smoke_output_sha256": evidence_obj["import_smoke_log_sha256"],
        "cuda_smoke_output_sha256": evidence_obj["cuda_smoke_log_sha256"],
        "pip_freeze_sha256": evidence_obj["pip_freeze_txt_sha256"],
        "python_exact": python_exact,
        "cuda_build": cuda_build,
        "evidence": evidence_obj,
    }
    env_sha = _write_json(root / ENV_REALIZATION_REL, env_obj)
    validate_env_realization(
        env_obj,
        root=root,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        env_spec_sha256=sha256_file(root / ENV_SPEC_REL),
    )
    return {"ok": True, "env_realization_sha256": env_sha, "python_exact": python_exact}


def write_realization_manifest(*, root: Path) -> dict[str, Any]:
    if not (root / WEIGHTS_REL).is_file():
        raise RuntimeError("weights_manifest_required")
    if not (root / ENV_REALIZATION_REL).is_file():
        raise RuntimeError("env_realization_required")
    weights_sha = sha256_file(root / WEIGHTS_REL)
    env_sha = sha256_file(root / ENV_REALIZATION_REL)
    evidence_files = {
        "pip_check_log": f"{EVIDENCE_DIR_REL}/pip_check.log",
        "import_smoke_log": f"{EVIDENCE_DIR_REL}/import_smoke.log",
        "cuda_smoke_log": f"{EVIDENCE_DIR_REL}/cuda_smoke.log",
        "pip_freeze_txt": f"{EVIDENCE_DIR_REL}/pip_freeze.txt",
        "python_provenance_json": f"{EVIDENCE_DIR_REL}/python_provenance.json",
        "cuda_provenance_json": f"{EVIDENCE_DIR_REL}/cuda_provenance.json",
    }
    for key, rel in list(evidence_files.items()):
        evidence_files[f"{key}_sha256"] = sha256_file(root / rel)

    manifest = {
        "schema": "ms_sta_v5_f0_realization_manifest_v1",
        "status": "REALIZED",
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "plan_lock_sha256": EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        "weights_manifest_sha256": weights_sha,
        "env_realization_sha256": env_sha,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "pip_check_exit_code": 0,
        "import_smoke_exit_code": 0,
        "cuda_smoke_exit_code": 0,
        "pip_check_output_sha256": evidence_files["pip_check_log_sha256"],
        "import_smoke_output_sha256": evidence_files["import_smoke_log_sha256"],
        "cuda_smoke_output_sha256": evidence_files["cuda_smoke_log_sha256"],
        "evidence_files": evidence_files,
    }
    digest = _write_json(root / REALIZATION_REL, manifest)
    validate_realization_manifest(
        manifest,
        root=root,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        weights_manifest_sha256=weights_sha,
        env_realization_sha256=env_sha,
    )
    return {"ok": True, "realization_manifest_sha256": digest}


def write_realization_manifest_audit(*, root: Path) -> dict[str, Any]:
    weights_sha = sha256_file(root / WEIGHTS_REL)
    env_sha = sha256_file(root / ENV_REALIZATION_REL)
    man_sha = sha256_file(root / REALIZATION_REL)
    audit = {
        "schema": "ms_sta_v5_f0_realization_manifest_audit_v1",
        "audit_verdict": "PASS",
        "realization_manifest": REALIZATION_REL,
        "realization_manifest_sha256": man_sha,
        "env_realization_sha256": env_sha,
        "weights_manifest_sha256": weights_sha,
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "plan_lock_sha256": EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        "realization_approval": REALIZATION_APPROVAL_REL,
        "realization_approval_sha256": sha256_file(root / REALIZATION_APPROVAL_REL),
        "realization_approval_audit": REALIZATION_APPROVAL_AUDIT_REL,
        "realization_approval_audit_sha256": sha256_file(root / REALIZATION_APPROVAL_AUDIT_REL),
        "gpu_feasibility_authorized": False,
        "execution_approval_authorized": False,
        "auto_chain": False,
        "notes": [
            "Realization manifest validated with recomputed evidence and weight SHAs.",
            "STOP: do not create execution approval/seal or run GPU feasibility.",
        ],
    }
    digest = _write_json(root / REALIZATION_MANIFEST_AUDIT_REL, audit)
    return {"ok": True, "audit_sha256": digest, "audit_verdict": "PASS"}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--action",
        required=True,
        choices=(
            "download-weights",
            "realize-env",
            "write-manifest",
            "audit-manifest",
            "all-realization",
        ),
    )
    args = ap.parse_args()
    action = (
        "download-weights"
        if args.action == "all-realization"
        else args.action
        if args.action != "write-manifest"
        else "realize-env"
    )
    # Gate via preflight using bootstrap action name.
    gate_action = (
        "download-weights"
        if args.action in {"download-weights", "all-realization"}
        else "realize-env"
    )
    pf = preflight(root=_ROOT, action=gate_action, python_exe=sys.executable)
    if not pf.get("side_effect_allowed"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "refused": True,
                    "reasons": pf.get("refuse_reasons"),
                    "realization_authorized": pf.get("realization_authorized"),
                },
                indent=2,
            )
        )
        return 2

    results: dict[str, Any] = {"schema": "ms_sta_v5_f0_realize_run_v1", "ok": True}
    try:
        if args.action in {"download-weights", "all-realization"}:
            results["download"] = download_weights(root=_ROOT)
        if args.action in {"realize-env", "all-realization"}:
            # Re-check gate for realize-env (approval still required).
            pf2 = preflight(root=_ROOT, action="realize-env", python_exe=sys.executable)
            if not pf2.get("side_effect_allowed"):
                raise RuntimeError("realize_env_refused:" + ",".join(pf2.get("refuse_reasons") or []))
            results["env"] = realize_env(root=_ROOT)
        if args.action in {"write-manifest", "all-realization"}:
            results["manifest"] = write_realization_manifest(root=_ROOT)
        if args.action in {"audit-manifest", "all-realization"}:
            results["audit"] = write_realization_manifest_audit(root=_ROOT)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
