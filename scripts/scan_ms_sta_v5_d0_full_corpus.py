#!/usr/bin/env python3
"""Full-corpus quality scanner — tensor-authoritative; dual-root carry-forward safe.

Manifest runtimes for skeleton and RGB must be independently bound when tensor roots differ.
Never derive both manifests from one runtime_dir under carry-forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
# Repo root must precede scripts so `import src...` works when cwd is outside the repo
# and when PYTHONPATH is unset (scan-recovery execute path).
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CORPUS_N,
    D0_FULL_CORPUS_APPROVAL_V8_REL,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    OUTPUT_VERSION,
    OUTPUT_VERSION_V15,
    RGB_MANIFEST_REQUIRED_FIELDS,
    RGB_REQUIRED_KEYS,
    SKEL_MANIFEST_REQUIRED_FIELDS,
    SKEL_REQUIRED_KEYS,
    SPLIT_COUNTS,
    assert_meta_tensor_agreement,
    atomic_replace_json,
    canonical_sample_paths,
    compute_quality_stats_from_rgb_tensors,
    default_quality_thresholds,
    evaluate_quality_thresholds,
    load_eligible_identity,
    load_json,
    load_manifest_index,
    manifest_dir,
    require_manifest_fields,
    runtime_dir,
    sha256_file,
    validate_array_contracts,
    validate_npz_keys,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCAN_REPORT_SCHEMA_V7 = "ms_sta_v5_d0_full_corpus_scan_report_v7"
SCAN_REPORT_SCHEMA_V15_RECOVERY = "ms_sta_v5_d0_full_corpus_scan_report_v15_recovery_v1"


class SourceShaVerifier:
    """One RawVideoResolver per scan; at most one resolve+hash per sample_id."""

    def __init__(self, *, root: Path, meta_by: dict[str, dict[str, str]]) -> None:
        from src.data_prep.raw_video_resolve import RawVideoResolver

        self.root = Path(root).resolve()
        self.meta_by = meta_by
        self.resolver = RawVideoResolver(self.root)
        self._observed_by_sample: dict[str, str] = {}
        self.raw_source_hash_count = 0
        self.resolve_count = 0
        self.resolver_instances = 1

    def verify_sample(self, *, sample_id: str, skel_declared: str, rgb_declared: str) -> None:
        _assert_hex64(skel_declared, field="source_sha256", sample_id=sample_id)
        _assert_hex64(rgb_declared, field="source_sha256", sample_id=sample_id)
        if skel_declared != rgb_declared:
            raise RuntimeError("cross_manifest_source_sha_mismatch")
        declared = skel_declared
        cached = self._observed_by_sample.get(sample_id)
        if cached is not None:
            if cached != declared:
                raise RuntimeError(f"source_sha256_mismatch:{sample_id}")
            return
        row = self.meta_by.get(sample_id)
        if row is None:
            raise RuntimeError(f"metadata_row_missing:{sample_id}")
        try:
            self.resolve_count += 1
            path = self.resolver.resolve(row)
            if path is None or not Path(path).is_file():
                raise RuntimeError(f"source_video_unresolvable:{sample_id}")
            self.raw_source_hash_count += 1
            observed = sha256_file(Path(path))
            if observed != declared:
                raise RuntimeError(f"source_sha256_mismatch:{sample_id}")
            self._observed_by_sample[sample_id] = observed
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"source_sha256_resolve_error:{sample_id}:{exc}") from exc

    def stats(self) -> dict[str, Any]:
        return {
            "resolver_instances": self.resolver_instances,
            "resolve_count": self.resolve_count,
            "raw_source_hash_count": self.raw_source_hash_count,
            "unique_samples_cached": len(self._observed_by_sample),
        }


def _load_maps(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]]]:
    import csv

    split_by: dict[str, str] = {}
    signer_by: dict[str, str] = {}
    meta_by: dict[str, dict[str, str]] = {}
    with (root / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["video_id"]
            meta_by[sid] = row
            split_by[sid] = row.get("split") or "unknown"
            signer_by[sid] = row.get("signer_id") or "unknown"
    return split_by, signer_by, meta_by


def _assert_hex64(val: Any, *, field: str, sample_id: str) -> None:
    if val is None or not isinstance(val, str) or not _HEX64.match(val):
        raise RuntimeError(f"manifest_invalid_{field}:{sample_id}")


def _rel_posix(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _eligible_set_sha256(eligible_ids: list[str] | set[str]) -> str:
    payload = "\n".join(sorted(str(x) for x in eligible_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_bind(*, runtime: Path, phase: str, root: Path) -> dict[str, Any]:
    idx_path = manifest_dir(runtime, phase) / "INDEX.json"
    bind: dict[str, Any] = {
        "runtime_rel": _rel_posix(runtime, root),
        "phase": phase,
        "index_rel": _rel_posix(idx_path, root) if idx_path.is_file() else None,
        "index_sha256": sha256_file(idx_path) if idx_path.is_file() else None,
    }
    return bind


def _resolve_scan_roots(
    *,
    binary_root: Path,
    runtime_dir_arg: Path,
    skeleton_source_root: Path | None,
    skeleton_tensor_root: Path | None,
    skeleton_manifest_runtime: Path | None,
    rgb_tensor_root: Path | None,
    rgb_manifest_runtime: Path | None,
) -> tuple[Path, Path, Path, Path, bool]:
    """Return skel_tensor, rgb_tensor, skel_manifest_rt, rgb_manifest_rt, dual_root."""
    rgb_tensor = Path(rgb_tensor_root or binary_root).resolve()
    skel_tensor = Path(
        skeleton_tensor_root or skeleton_source_root or binary_root
    ).resolve()
    rgb_man_rt = Path(rgb_manifest_runtime or runtime_dir_arg).resolve()
    if skeleton_manifest_runtime is not None:
        skel_man_rt = Path(skeleton_manifest_runtime).resolve()
    elif skel_tensor != rgb_tensor:
        # Carry-forward / dual-root: bind skeleton manifests to the skeleton tensor root runtime.
        skel_man_rt = runtime_dir(skel_tensor).resolve()
    else:
        skel_man_rt = Path(runtime_dir_arg).resolve()

    dual_root = skel_tensor != rgb_tensor
    if dual_root and skel_man_rt == rgb_man_rt:
        raise RuntimeError("scan_manifest_runtimes_not_independently_bound")
    return skel_tensor, rgb_tensor, skel_man_rt, rgb_man_rt, dual_root


def scan_full_corpus(
    *,
    binary_root: Path,
    runtime_dir: Path,
    thresholds: dict[str, float] | None = None,
    root: Path = _ROOT,
    skeleton_source_root: Path | None = None,
    skeleton_tensor_root: Path | None = None,
    skeleton_manifest_runtime: Path | None = None,
    rgb_tensor_root: Path | None = None,
    rgb_manifest_runtime: Path | None = None,
    output_version: str | None = None,
    write_artifacts: bool = True,
    output_runtime_dir: Path | None = None,
) -> dict[str, Any]:
    # Resolve once so Windows short/long path forms cannot break relative_to / containment.
    root = Path(root).resolve()
    thr_in = dict(thresholds or default_quality_thresholds())
    thr_frozen = dict(thr_in)
    identity = load_eligible_identity(root=root)
    expected = identity["eligible_id_set"]
    split_by, signer_by, meta_by = _load_maps(root)
    source_verifier = SourceShaVerifier(root=root, meta_by=meta_by)

    skel_root, rgb_root, skel_man_rt, rgb_man_rt, dual_root = _resolve_scan_roots(
        binary_root=binary_root,
        runtime_dir_arg=runtime_dir,
        skeleton_source_root=skeleton_source_root,
        skeleton_tensor_root=skeleton_tensor_root,
        skeleton_manifest_runtime=skeleton_manifest_runtime,
        rgb_tensor_root=rgb_tensor_root,
        rgb_manifest_runtime=rgb_manifest_runtime,
    )

    out_ver = str(output_version or (OUTPUT_VERSION_V15 if dual_root else OUTPUT_VERSION))
    if dual_root and out_ver != OUTPUT_VERSION_V15:
        raise RuntimeError(f"scan_dual_root_requires_output_version_v15_got:{out_ver}")
    if out_ver not in (OUTPUT_VERSION, OUTPUT_VERSION_V15):
        raise RuntimeError(f"scan_output_version_unsupported:{out_ver}")
    if dual_root:
        schema = SCAN_REPORT_SCHEMA_V15_RECOVERY
    elif out_ver == OUTPUT_VERSION_V15:
        schema = SCAN_REPORT_SCHEMA_V15_RECOVERY
    else:
        schema = SCAN_REPORT_SCHEMA_V7

    skel_dir = skel_root / "skeleton_v3"
    rgb_dir = rgb_root / "rgb_token_v3"
    skel_ids = {p.stem for p in skel_dir.glob("*.npz")} if skel_dir.is_dir() else set()
    rgb_ids = {p.stem for p in rgb_dir.glob("*.npz")} if rgb_dir.is_dir() else set()

    global_failures: list[dict[str, Any]] = []
    sample_failures: list[dict[str, Any]] = []

    def _global(err: dict[str, Any]) -> None:
        global_failures.append(err)

    if skel_ids != expected:
        _global(
            {
                "error": "skeleton_id_set_mismatch",
                "missing": sorted(expected - skel_ids)[:20],
                "extra": sorted(skel_ids - expected)[:20],
                "n_missing": len(expected - skel_ids),
                "n_extra": len(skel_ids - expected),
            }
        )
    if rgb_ids != expected:
        _global(
            {
                "error": "rgb_id_set_mismatch",
                "missing": sorted(expected - rgb_ids)[:20],
                "extra": sorted(rgb_ids - expected)[:20],
                "n_missing": len(expected - rgb_ids),
                "n_extra": len(rgb_ids - expected),
            }
        )
    paired = skel_ids & rgb_ids & expected
    if len(paired) != CORPUS_N and not global_failures:
        _global({"error": "paired_count_mismatch", "n_paired": len(paired)})

    split_counts_obs = {"train": 0, "val": 0, "test": 0}
    for sid in paired:
        sp = split_by.get(sid)
        if sp in split_counts_obs:
            split_counts_obs[sp] += 1
    if split_counts_obs != SPLIT_COUNTS:
        _global({"error": "split_counts_mismatch", "observed": split_counts_obs})

    try:
        skel_man = load_manifest_index(skel_man_rt, "skeleton_v3")
        rgb_man = load_manifest_index(rgb_man_rt, "rgb_token_v3")
    except RuntimeError as exc:
        _global({"error": "manifest_index_error", "detail": str(exc)})
        skel_man, rgb_man = {}, {}

    if set(skel_man) != expected and paired == expected:
        _global(
            {
                "error": "skel_manifest_id_set_mismatch",
                "n_manifest": len(skel_man),
                "n_expected": len(expected),
            }
        )
    if set(rgb_man) != expected and paired == expected:
        _global(
            {
                "error": "rgb_manifest_id_set_mismatch",
                "n_manifest": len(rgb_man),
                "n_expected": len(expected),
            }
        )

    for sid in sorted(paired):
        try:
            skel_path = skel_dir / f"{sid}.npz"
            rgb_path = rgb_dir / f"{sid}.npz"
            skel_sidecar = skel_dir / f"{sid}.recovery.json"
            rgb_sidecar = rgb_dir / f"{sid}.meta.json"
            skel_info = validate_npz_keys(skel_path, SKEL_REQUIRED_KEYS)
            rgb_info = validate_npz_keys(rgb_path, RGB_REQUIRED_KEYS)
            validate_array_contracts(skel_path, kind="skel")
            validate_array_contracts(rgb_path, kind="rgb")
            import numpy as np

            rgb = np.load(rgb_path, allow_pickle=False)
            n_win = int(rgb["missing_face"].shape[0])
            if int(rgb["token_face"].shape[0]) != n_win:
                raise RuntimeError("rgb_T_axis_mismatch")
            if not rgb_sidecar.is_file():
                raise RuntimeError("missing_rgb_meta")
            meta = load_json(rgb_sidecar)
            assert_meta_tensor_agreement(rgb_path=rgb_path, meta=meta, sample_id=sid)
            if meta.get("sample_id") not in (None, sid):
                raise RuntimeError("meta_sample_id_mismatch")
            sm = skel_man.get(sid)
            rm = rgb_man.get(sid)
            if sm is None or rm is None:
                raise RuntimeError("missing_manifest_row")
            require_manifest_fields(sm, SKEL_MANIFEST_REQUIRED_FIELDS)
            require_manifest_fields(rm, RGB_MANIFEST_REQUIRED_FIELDS)
            if sm.get("split") != split_by.get(sid) or sm.get("signer_id") != signer_by.get(sid):
                raise RuntimeError("skel_manifest_identity_mismatch")
            if rm.get("split") != split_by.get(sid) or rm.get("signer_id") != signer_by.get(sid):
                raise RuntimeError("rgb_manifest_identity_mismatch")
            if sha256_file(skel_path) != sm.get("output_sha256"):
                raise RuntimeError("skel_manifest_sha_mismatch")
            if sha256_file(rgb_path) != rm.get("output_sha256"):
                raise RuntimeError("rgb_manifest_sha_mismatch")
            if not skel_sidecar.is_file():
                raise RuntimeError("missing_skel_sidecar")
            if sha256_file(skel_sidecar) != sm.get("sidecar_sha256"):
                raise RuntimeError("skel_sidecar_sha_mismatch")
            if sha256_file(rgb_sidecar) != rm.get("sidecar_sha256"):
                raise RuntimeError("rgb_sidecar_sha_mismatch")
            if sha256_file(skel_path) != rm.get("skel_sha256"):
                raise RuntimeError("rgb_skel_sha_mismatch")
            canon_skel_out, canon_skel_side = canonical_sample_paths(
                root=root, binary=skel_root, phase="skeleton_v3", sample_id=sid
            )
            canon_rgb_out, canon_rgb_side = canonical_sample_paths(
                root=root, binary=rgb_root, phase="rgb_token_v3", sample_id=sid
            )
            if sm.get("output_rel") != canon_skel_out or sm.get("sidecar_rel") != canon_skel_side:
                raise RuntimeError("skel_manifest_path_mismatch")
            if rm.get("output_rel") != canon_rgb_out or rm.get("sidecar_rel") != canon_rgb_side:
                raise RuntimeError("rgb_manifest_path_mismatch")
            # Path containment under bound tensor roots.
            skel_out_abs = (root / str(sm["output_rel"])).resolve()
            rgb_out_abs = (root / str(rm["output_rel"])).resolve()
            try:
                skel_out_abs.relative_to(skel_root)
                rgb_out_abs.relative_to(rgb_root)
            except ValueError as exc:
                raise RuntimeError("manifest_path_outside_bound_tensor_root") from exc
            if skel_out_abs != skel_path.resolve() or rgb_out_abs != rgb_path.resolve():
                raise RuntimeError("manifest_path_not_pointing_at_tensor")
            source_verifier.verify_sample(
                sample_id=sid,
                skel_declared=str(sm["source_sha256"]),
                rgb_declared=str(rm["source_sha256"]),
            )
            _ = skel_info, rgb_info
        except Exception as exc:  # noqa: BLE001
            sample_failures.append(
                {"sample_id": sid, "error": str(exc), "error_type": type(exc).__name__}
            )

    stats = {
        "denominator": "window_region_triples_from_rgb_tensor_masks",
        "source_of_truth": "rgb_token_v3_npz_missing_and_partial_arrays",
        "missing_roi_rate_by_region_from_tensor_validity": {
            "face": 1.0,
            "left_hand": 1.0,
            "right_hand": 1.0,
        },
        "missing_roi_rate_by_split_from_tensor_validity": {},
        "missing_roi_rate_by_signer_from_tensor_validity": {},
        "partial_missing_rate_by_region": {"face": 1.0, "left_hand": 1.0, "right_hand": 1.0},
    }
    if paired:
        try:
            stats = compute_quality_stats_from_rgb_tensors(
                rgb_dir=rgb_dir,
                sample_ids=sorted(paired),
                split_by_id=split_by,
                signer_by_id=signer_by,
            )
        except Exception as exc:  # noqa: BLE001
            _global({"error": "tensor_quality_stats_error", "detail": str(exc)})

    eval_thr = evaluate_quality_thresholds(stats, thr_frozen)
    if eval_thr["thresholds_used"] != thr_frozen:
        _global({"error": "scanner_mutated_thresholds"})
    if not eval_thr["ok"]:
        _global({"error": "quality_threshold_breach", "checks": eval_thr["checks"]})

    unique_failed_ids = sorted({str(f["sample_id"]) for f in sample_failures})
    global_failure_count = len(global_failures)
    sample_failure_count = len(sample_failures)
    unique_failed_sample_count = len(unique_failed_ids)
    # Scientific fail count for completion: unique samples + global contract failures (no double-count).
    failed_count = unique_failed_sample_count + global_failure_count
    ok = failed_count == 0 and len(paired) == CORPUS_N

    write_rt = Path(output_runtime_dir or runtime_dir)
    write_rt.mkdir(parents=True, exist_ok=True)
    stats_path = write_rt / "QUALITY_STATS.json"
    if write_artifacts:
        atomic_replace_json(stats_path, stats)
        stats_sha = sha256_file(stats_path)
    else:
        # Ephemeral sha for dry callers
        stats_sha = hashlib.sha256(
            (json.dumps(stats, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest()

    failures_head = (global_failures + sample_failures)[:200]
    report = {
        "schema": schema,
        "output_version": out_ver,
        "ok": ok,
        "n_expected": CORPUS_N,
        "n_skel": len(skel_ids),
        "n_rgb": len(rgb_ids),
        "n_paired": len(paired),
        "failed_count": failed_count,
        "global_failure_count": global_failure_count,
        "sample_failure_count": sample_failure_count,
        "unique_failed_sample_count": unique_failed_sample_count,
        "failures": failures_head,
        "split_counts_observed": split_counts_obs,
        "quality_eval": eval_thr,
        "quality_stats_sha256": stats_sha,
        "thresholds_frozen": thr_frozen,
        "tensor_authoritative": True,
        "dual_root": dual_root,
        "skeleton_source_root_rel": (
            _rel_posix(skel_root, root) if skel_root != rgb_root else None
        ),
        "binary_root_rel": _rel_posix(rgb_root, root),
        "binds": {
            "eligible_id_set_sha256": _eligible_set_sha256(expected),
            "eligible_n": len(expected),
            "skeleton_tensor_root_rel": _rel_posix(skel_root, root),
            "rgb_tensor_root_rel": _rel_posix(rgb_root, root),
            "skeleton_manifest": _manifest_bind(
                runtime=skel_man_rt, phase="skeleton_v3", root=root
            ),
            "rgb_manifest": _manifest_bind(runtime=rgb_man_rt, phase="rgb_token_v3", root=root),
        },
        "source_verify_stats": source_verifier.stats(),
    }
    if write_artifacts:
        atomic_replace_json(write_rt / "SCAN_REPORT.json", report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary-root", required=True, help="RGB tensor binary root")
    ap.add_argument("--runtime-dir", required=True, help="RGB manifest runtime (and write target)")
    ap.add_argument(
        "--skeleton-source-root",
        default=None,
        help="Optional read-only skeleton tensor root (carry-forward dual-root)",
    )
    ap.add_argument("--skeleton-tensor-root", default=None)
    ap.add_argument("--skeleton-manifest-runtime", default=None)
    ap.add_argument("--rgb-tensor-root", default=None)
    ap.add_argument("--rgb-manifest-runtime", default=None)
    ap.add_argument("--output-version", default=None, choices=[OUTPUT_VERSION, OUTPUT_VERSION_V15])
    ap.add_argument(
        "--output-runtime-dir",
        default=None,
        help="Optional write target for SCAN_REPORT/QUALITY_STATS (scan-recovery)",
    )
    ap.add_argument("--no-write", action="store_true", help="Do not write SCAN_REPORT/QUALITY_STATS")
    args = ap.parse_args()
    thr = default_quality_thresholds()
    if (_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL).is_file():
        approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL)
        thr = dict(approval.get("quality_thresholds") or thr)
    elif (_ROOT / D0_FULL_CORPUS_APPROVAL_V8_REL).is_file():
        approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_V8_REL)
        thr = dict(approval.get("quality_thresholds") or thr)
    report = scan_full_corpus(
        binary_root=Path(args.binary_root),
        runtime_dir=Path(args.runtime_dir),
        thresholds=thr,
        root=_ROOT,
        skeleton_source_root=Path(args.skeleton_source_root) if args.skeleton_source_root else None,
        skeleton_tensor_root=Path(args.skeleton_tensor_root) if args.skeleton_tensor_root else None,
        skeleton_manifest_runtime=(
            Path(args.skeleton_manifest_runtime) if args.skeleton_manifest_runtime else None
        ),
        rgb_tensor_root=Path(args.rgb_tensor_root) if args.rgb_tensor_root else None,
        rgb_manifest_runtime=Path(args.rgb_manifest_runtime) if args.rgb_manifest_runtime else None,
        output_version=args.output_version,
        write_artifacts=not args.no_write,
        output_runtime_dir=Path(args.output_runtime_dir) if args.output_runtime_dir else None,
    )
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "failed_count": report["failed_count"],
                "global_failure_count": report["global_failure_count"],
                "sample_failure_count": report["sample_failure_count"],
                "unique_failed_sample_count": report["unique_failed_sample_count"],
                "n_paired": report["n_paired"],
                "output_version": report["output_version"],
                "schema": report["schema"],
                "skeleton_source_root_rel": report.get("skeleton_source_root_rel"),
            },
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
