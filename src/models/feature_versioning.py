"""Helpers for versioned feature dirs, provenance, run contracts, eligible IDs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

RUN_CONTRACT_NAME = "_run_contract.json"
ELIGIBLE_MANIFEST_REL = "data/processed/vsl400_eligible_sample_manifest.json"
ELIGIBLE_SCHEMAS = ("vsl400_eligible_sample_manifest_v1", "option_b_eligible_ids_v1")
REQUIRED_EXCLUDED_ID = "019442"


def git_rev(root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def git_dirty(root: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root), stderr=subprocess.DEVNULL
        ).decode()
        return bool(out.strip())
    except Exception:
        return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_id_list(ids: list[str]) -> str:
    return sha256_text("\n".join(ids))


def count_npy(directory: Path) -> int:
    return sum(1 for p in directory.glob("*.npy") if p.is_file())


def assert_fresh_or_empty_dir(
    directory: Path,
    *,
    resume: bool,
    label: str,
) -> None:
    """Refuse non-empty versioned dirs unless resume=True."""
    if resume:
        if not directory.is_dir():
            raise SystemExit(f"[FAIL] --resume requires existing directory: {directory}")
        return
    if not directory.exists():
        return
    if not directory.is_dir():
        raise SystemExit(f"[FAIL] {label} path exists but is not a directory: {directory}")
    npy = count_npy(directory)
    if npy > 0:
        raise SystemExit(
            f"[FAIL] {label}: directory already has {npy} .npy files: {directory}. "
            "Use a new tag or explicit --resume (requires matching _run_contract.json)."
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def rel_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_run_contract(
    *,
    kind: str,
    sample_ids: list[str],
    metadata_sha256: str | None,
    splits: list[str] | str,
    config: dict[str, Any],
    root: Path,
    sample_manifest_sha256: str | None = None,
) -> dict:
    return {
        "schema": "option_b_run_contract_v1",
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_ids": list(sample_ids),
        "sample_id_sha256": sha256_id_list(sample_ids),
        "n_sample_ids": len(sample_ids),
        "metadata_sha256": metadata_sha256,
        "sample_manifest_sha256": sample_manifest_sha256,
        "splits": splits,
        "config": config,
        "git_revision": git_rev(root),
        "git_dirty": git_dirty(root),
    }


def write_run_contract(directory: Path, contract: dict) -> Path:
    """Write _run_contract.json atomically BEFORE any .npy outputs."""
    path = directory / RUN_CONTRACT_NAME
    write_json_atomic(path, contract)
    return path


def require_resume_contract(directory: Path, expected: dict) -> dict:
    """Fail-closed resume: contract must exist and critical fields must match exactly."""
    path = directory / RUN_CONTRACT_NAME
    if not path.is_file():
        raise SystemExit(
            f"[FAIL] --resume requires {RUN_CONTRACT_NAME} in {directory}. "
            "Interrupted runs without a pre-written contract cannot be resumed safely."
        )
    prev = load_json(path)
    keys = (
        "kind",
        "sample_id_sha256",
        "n_sample_ids",
        "metadata_sha256",
        "sample_manifest_sha256",
        "splits",
        "config",
        "git_revision",
        "git_dirty",
    )
    for key in keys:
        if prev.get(key) != expected.get(key):
            raise SystemExit(
                f"[FAIL] --resume contract mismatch on '{key}': "
                f"disk={prev.get(key)!r} planned={expected.get(key)!r}"
            )
    if prev.get("sample_ids") != expected.get("sample_ids"):
        raise SystemExit("[FAIL] --resume contract sample_ids list mismatch")
    return prev


def canonical_eligible_manifest_path(root: Path) -> Path:
    return root / ELIGIBLE_MANIFEST_REL


def require_canonical_sample_manifest_sha(
    claimed: str | None,
    *,
    root: Path,
    registry: dict | None = None,
) -> str:
    """Bind claimed SHA to on-disk eligible manifest (and optional registry field)."""
    if not claimed:
        raise ValueError("sample_manifest_sha256 required (eligible-ID lock)")
    path = canonical_eligible_manifest_path(root)
    if not path.is_file():
        raise ValueError(f"canonical eligible manifest missing: {path}")
    disk = sha256_file(path)
    if claimed != disk:
        raise ValueError(
            f"sample_manifest_sha256 mismatch vs {ELIGIBLE_MANIFEST_REL}: "
            f"claimed={claimed} disk={disk}"
        )
    if registry is not None:
        reg_sha = registry.get("eligible_sample_manifest_sha256")
        if reg_sha and reg_sha != claimed:
            raise ValueError(
                "sample_manifest_sha256 != registry.eligible_sample_manifest_sha256 "
                f"({claimed} != {reg_sha})"
            )
    return disk


def load_eligible_sample_manifest(
    path: Path,
    *,
    root: Path | None = None,
    metadata_csv: Path | None = None,
    require_metadata_match: bool = True,
) -> dict:
    """Load + scientifically validate eligible-ID manifest."""
    man = load_json(path)
    if man.get("schema") not in ELIGIBLE_SCHEMAS:
        raise ValueError(f"unsupported eligible manifest schema: {man.get('schema')}")
    ids = [str(x) for x in man.get("eligible_ids") or []]
    if not ids:
        raise ValueError("eligible_ids empty")
    if len(ids) != len(set(ids)):
        raise ValueError("eligible_ids contain duplicates")
    if REQUIRED_EXCLUDED_ID in set(ids):
        raise ValueError(f"{REQUIRED_EXCLUDED_ID} must not appear in eligible_ids")

    excl_rows = man.get("excluded") or []
    if not isinstance(excl_rows, list):
        raise ValueError("excluded must be a list")
    excl_ids: list[str] = []
    for row in excl_rows:
        if not isinstance(row, dict) or not row.get("video_id"):
            raise ValueError("each excluded entry requires video_id")
        if not (row.get("reason") or row.get("evidence")):
            raise ValueError(f"excluded {row.get('video_id')} needs reason/evidence")
        excl_ids.append(str(row["video_id"]))
    excl_set = set(excl_ids)
    if len(excl_ids) != len(excl_set):
        raise ValueError("excluded video_ids contain duplicates")
    overlap = set(ids) & excl_set
    if overlap:
        raise ValueError(f"eligible ∩ excluded non-empty: {sorted(overlap)[:5]}")
    if REQUIRED_EXCLUDED_ID not in excl_set:
        raise ValueError(f"excluded must include {REQUIRED_EXCLUDED_ID}")

    declared_n = man.get("eligible_n")
    if declared_n is not None and int(declared_n) != len(ids):
        raise ValueError(f"eligible_n={declared_n} != len(eligible_ids)={len(ids)}")
    declared_excl_n = man.get("excluded_n")
    if declared_excl_n is not None and int(declared_excl_n) != len(excl_set):
        raise ValueError(f"excluded_n={declared_excl_n} != excluded unique={len(excl_set)}")

    digest = sha256_id_list(ids)
    if man.get("sample_id_sha256") and man["sample_id_sha256"] != digest:
        raise ValueError("sample_id_sha256 does not match eligible_ids list")

    repo = root
    if repo is None:
        try:
            repo = path.resolve().parents[2]
        except Exception:
            repo = path.parent

    meta_path = metadata_csv
    if meta_path is None:
        rel_meta = str(man.get("metadata") or "metadata.csv").replace("\\", "/")
        # Windows absolute or POSIX absolute → refuse in tracked artifact
        if Path(rel_meta).is_absolute() or (len(rel_meta) >= 3 and rel_meta[1] == ":"):
            raise ValueError(
                "eligible manifest.metadata must be repo-relative (e.g. metadata.csv), "
                f"got absolute: {rel_meta}"
            )
        meta_path = resolve_under_root(repo, Path(rel_meta))

    if require_metadata_match:
        if not meta_path.is_file():
            raise ValueError(f"metadata.csv missing for eligible validation: {meta_path}")
        meta_sha = sha256_file(meta_path)
        if man.get("metadata_sha256") and man["metadata_sha256"] != meta_sha:
            raise ValueError(
                f"metadata_sha256 mismatch: manifest={man.get('metadata_sha256')} disk={meta_sha}"
            )
        df = pd.read_csv(meta_path)
        vsl = df[df["dataset"] == "VSL400"].copy()
        all_ids = set(vsl["video_id"].astype(str))
        union = set(ids) | excl_set
        if union != all_ids:
            missing = sorted(all_ids - union)[:5]
            extra = sorted(union - all_ids)[:5]
            raise ValueError(
                "eligible∪excluded != VSL400 metadata "
                f"(missing_sample={missing} extra={extra})"
            )
        by_split: dict[str, int] = {}
        elig_df = vsl[vsl["video_id"].astype(str).isin(set(ids))]
        for s, n in elig_df.groupby(elig_df["split"].astype(str))["video_id"].count().items():
            by_split[str(s)] = int(n)
        declared_by = man.get("eligible_by_split") or {}
        if dict(declared_by) != by_split:
            raise ValueError(
                f"eligible_by_split mismatch: manifest={declared_by} computed={by_split}"
            )

    return {
        "path": path,
        "sha256": sha256_file(path),
        "eligible_ids": ids,
        "eligible_set": set(ids),
        "excluded": excl_rows,
        "excluded_ids": excl_set,
        "raw": man,
        "test_n": int((man.get("eligible_by_split") or {}).get("test") or 0),
    }


def filter_video_ids(ids: list[str], eligible: set[str] | None) -> list[str]:
    if eligible is None:
        return ids
    return [i for i in ids if i in eligible]
