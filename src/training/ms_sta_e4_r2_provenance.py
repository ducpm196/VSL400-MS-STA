"""Canonical provenance helpers for MS-STA E4-R2 training/orchestration.

Fail-closed: dirty worktree, HEAD/upstream drift, tree drift, missing SHAs,
or reuse of existing trial outputs must abort before training.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from src.training.retrain_v4 import executable_tree_sha256 as _exec_tree_sha256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_branch(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True
    ).strip()


def git_dirty(root: Path) -> bool:
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    return bool(porcelain)


def git_upstream_rev(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "@{u}"], cwd=root, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("upstream_not_configured_push_code_lock_first") from exc


def git_is_ancestor(*, root: Path, ancestor: str, tip: str = "HEAD") -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, tip],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def require_clean_code_lock(*, root: Path, expect_branch: str | None = None) -> dict[str, Any]:
    branch = git_branch(root)
    if expect_branch and branch != expect_branch:
        raise RuntimeError(f"branch_mismatch:{branch}!={expect_branch}")
    if git_dirty(root):
        raise RuntimeError("worktree_dirty_refuse_canonical_train")
    head = git_rev(root)
    upstream = git_upstream_rev(root)
    if head != upstream:
        raise RuntimeError(f"head_not_equal_upstream:{head}!={upstream}")
    tree = _exec_tree_sha256(root)
    return {
        "git_revision": head,
        "git_branch": branch,
        "git_dirty": False,
        "git_upstream_revision": upstream,
        "executable_tree_sha256": tree,
    }


def verify_sealed_code_lock(*, root: Path, sealed: dict[str, Any], live: dict[str, Any], policy_sha: str) -> None:
    """Docs-only tip commits may advance HEAD; executable_tree_sha256 is authoritative."""
    impl = str(sealed.get("implementation_git_revision") or sealed.get("git_revision") or "")
    if not impl:
        raise RuntimeError("code_lock_missing_implementation_revision")
    if not git_is_ancestor(root=root, ancestor=impl, tip=live["git_revision"]):
        raise RuntimeError(f"code_lock_impl_not_ancestor:{impl}")
    if sealed.get("executable_tree_sha256") != live["executable_tree_sha256"]:
        raise RuntimeError("code_lock_executable_tree_mismatch")
    if sealed.get("train_policy_sha256") != policy_sha:
        raise RuntimeError("code_lock_train_policy_sha_mismatch")


def feature_root_identity(root: Path, rel: str) -> dict[str, Any]:
    """Bind feature root by dirname token + optional runtime manifest SHA."""
    path = root / rel
    if not path.is_dir():
        raise RuntimeError(f"feature_root_missing:{rel}")
    name = path.name
    token = name.rsplit("_", 1)[-1] if "_" in name else name
    runtime_manifest = path / "runtime" / "FEATURE_ROOT_MANIFEST.json"
    out: dict[str, Any] = {
        "root_rel": rel.replace("\\", "/"),
        "dirname_token": token,
        "exists": True,
    }
    if runtime_manifest.is_file():
        out["runtime_manifest_rel"] = str(runtime_manifest.relative_to(root)).replace("\\", "/")
        out["runtime_manifest_sha256"] = sha256_file(runtime_manifest)
    return out


def cuda_provenance() -> dict[str, Any]:
    avail = bool(torch.cuda.is_available())
    out: dict[str, Any] = {
        "cuda_available": avail,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if avail:
        out["cuda_device_name"] = torch.cuda.get_device_name(0)
        out["cuda_device_count"] = int(torch.cuda.device_count())
        out["cudnn_version"] = str(getattr(torch.backends.cudnn, "version", lambda: None)())
    return out


def build_trial_bindings(
    *,
    root: Path,
    policy_path: Path,
    policy: dict[str, Any],
    variant: str,
    seed: int,
    lr: float,
    dropout: float,
    weight_decay: float,
    distill_weight: float,
    phase: str,
    code_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if code_lock is None:
        code_lock = require_clean_code_lock(
            root=root, expect_branch=str(policy.get("branch") or "research/ms-sta-v6-e4-r2")
        )
    policy_sha = sha256_file(policy_path)
    expected_policy_sha = policy.get("policy_sha256_self")
    if expected_policy_sha and expected_policy_sha != policy_sha:
        raise RuntimeError(f"policy_sha_mismatch:{policy_sha}!={expected_policy_sha}")

    split_rel = policy["split_manifest_rel"]
    split_sha = sha256_file(root / split_rel)
    if split_sha != policy["split_manifest_sha256"]:
        raise RuntimeError("split_manifest_sha_mismatch")

    t_rel = (policy.get("distill") or {}).get("teacher_checkpoint_rel") or policy.get(
        "teacher_checkpoint_rel"
    )
    t_sha_expected = (policy.get("distill") or {}).get("teacher_checkpoint_sha256") or policy.get(
        "teacher_checkpoint_sha256"
    )
    if not t_rel or not t_sha_expected:
        raise RuntimeError("teacher_checkpoint_binding_missing")
    t_path = root / t_rel
    t_sha = sha256_file(t_path)
    if t_sha != t_sha_expected:
        raise RuntimeError("teacher_checkpoint_sha_mismatch")

    feat = policy["feature_roots"]
    rgb_id = feature_root_identity(root, feat["rgb_root_rel"])
    skel_id = feature_root_identity(root, feat["skeleton_root_rel"])
    # Prefer locked tokens from policy when present.
    for key, ident in (("rgb_dirname_token", rgb_id), ("skeleton_dirname_token", skel_id)):
        expected = (policy.get("feature_root_tokens") or {}).get(key)
        if expected and ident["dirname_token"] != expected:
            raise RuntimeError(f"feature_root_token_mismatch:{key}")

    cuda = cuda_provenance()
    req_gpu = ((policy.get("environment_lock") or {}).get("required_gpu_name")) or (
        (policy.get("scientific_environment_lock") or {}).get("required_gpu_name")
    )
    if req_gpu and cuda.get("cuda_device_name") != req_gpu:
        raise RuntimeError(f"cuda_device_mismatch:{cuda.get('cuda_device_name')}!={req_gpu}")

    return {
        "schema": "ms_sta_v6_e4_r2_trial_bindings_v1",
        "canonical": True,
        "campaign": "E4-R2",
        "phase": phase,
        "variant": variant,
        "seed": int(seed),
        "config": {
            "lr": float(lr),
            "dropout": float(dropout),
            "weight_decay": float(weight_decay),
            "distill_weight": float(distill_weight),
        },
        "policy_rel": str(policy_path.relative_to(root)).replace("\\", "/"),
        "policy_sha256": policy_sha,
        "scientific_policy_rel": policy.get("scientific_policy_rel"),
        "teacher_checkpoint_rel": str(Path(t_rel).as_posix()),
        "teacher_checkpoint_sha256": t_sha,
        "split_manifest_rel": split_rel,
        "split_manifest_sha256": split_sha,
        "feature_roots": {"rgb": rgb_id, "skeleton": skel_id},
        "cuda": cuda,
        **code_lock,
    }


def assert_no_drift(bindings: dict[str, Any], *, root: Path) -> None:
    """Detect HEAD / executable-tree drift mid-campaign."""
    if git_dirty(root):
        raise RuntimeError("worktree_became_dirty_mid_phase")
    head = git_rev(root)
    if head != bindings["git_revision"]:
        raise RuntimeError(f"head_drift:{head}!={bindings['git_revision']}")
    tree = _exec_tree_sha256(root)
    if tree != bindings["executable_tree_sha256"]:
        raise RuntimeError("executable_tree_drift")
    try:
        upstream = git_upstream_rev(root)
    except RuntimeError:
        raise
    if upstream != bindings.get("git_upstream_revision"):
        raise RuntimeError("upstream_drift")


def refuse_existing_outputs(out_dir: Path) -> None:
    blockers = ["SUMMARY.json", "best.pt", "VAL_PREDS.json", "PROVENANCE.json"]
    existing = [n for n in blockers if (out_dir / n).is_file()]
    if existing:
        raise RuntimeError(f"refuse_existing_outputs:{out_dir}:{existing}")


def write_provenance(out_dir: Path, bindings: dict[str, Any]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "PROVENANCE.json"
    raw = (json.dumps(bindings, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
