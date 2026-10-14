"""Promote a versioned feature directory into ms_sta_registry (validated + atomic)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import (
    RUN_CONTRACT_NAME,
    count_npy,
    load_json,
    rel_posix,
    require_canonical_sample_manifest_sha,
    resolve_under_root,
    sha256_file,
)

REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
LEGACY = {
    "skel": "data/features/local_skel",
    "temporal": "data/features/isolated_sf_temporal",
}


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 2


def _resolve_manifest_out_dir(man: dict) -> Path:
    raw = man.get("output_dir") or man.get("out_dir")
    if not raw:
        raise ValueError("manifest missing output_dir/out_dir")
    out_dir = Path(str(raw))
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    return out_dir


def _require_clean_or_pilot(man: dict, *, allow_dirty_pilot: bool) -> str:
    if "git_dirty" not in man:
        raise ValueError("manifest missing git_dirty")
    if not man.get("git_revision"):
        raise ValueError("manifest.git_revision required")
    if man.get("git_dirty") is True and not allow_dirty_pilot:
        raise ValueError(
            "canonical promote refuses git_dirty=true; "
            "code-lock commit first or pass --allow-dirty-pilot"
        )
    return "pilot" if allow_dirty_pilot and man.get("git_dirty") else "canonical"


def _bind_run_contract(target: Path, man: dict) -> dict:
    contract_path = target / RUN_CONTRACT_NAME
    if not contract_path.is_file():
        raise ValueError(f"missing {RUN_CONTRACT_NAME} under {target}")
    contract = load_json(contract_path)
    for key in ("sample_id_sha256", "sample_manifest_sha256", "metadata_sha256"):
        if man.get(key) and contract.get(key) and man.get(key) != contract.get(key):
            raise ValueError(f"{RUN_CONTRACT_NAME}.{key} != end-manifest.{key}")
    return {
        "run_contract": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
    }


def _validate_skel(
    target: Path,
    quality_report: Path,
    manifest: Path,
    *,
    min_coverage: float,
    allow_dirty_pilot: bool,
    registry: dict,
) -> dict:
    if not quality_report.is_file():
        raise ValueError(f"missing --quality-report: {quality_report}")
    if not manifest.is_file():
        raise ValueError(f"missing --manifest (D1 end manifest): {manifest}")

    report = load_json(quality_report)
    man = load_json(manifest)

    if report.get("pass") is not True:
        raise ValueError("quality report pass!=true")
    if report.get("full_scan") is not True:
        raise ValueError("quality report full_scan!=true (refuse subsample claims)")
    skel_in_report = Path(str(report.get("skel_dir") or ""))
    if skel_in_report.resolve() != target.resolve():
        raise ValueError(
            f"quality report skel_dir={skel_in_report} does not match --dir={target}"
        )
    out_dir = _resolve_manifest_out_dir(man)
    if out_dir.resolve() != target.resolve():
        raise ValueError(f"D1 manifest output_dir={out_dir} does not match --dir={target}")

    coverage = float(report.get("coverage") or 0.0)
    if coverage < min_coverage:
        raise ValueError(f"coverage {coverage:.4f} < min {min_coverage}")
    present = int(report.get("present") or 0)
    npy = count_npy(target)
    if present != npy:
        raise ValueError(f"present={present} != npy_count={npy} in directory")
    if npy <= 0:
        raise ValueError("directory has zero .npy files")
    for key in ("bad_shape", "nan_files", "inf_files", "zero_frame_files", "outlier_scale_files"):
        if int(report.get(key) or 0) != 0:
            raise ValueError(f"quality report {key}={report.get(key)} (must be 0)")

    required = (
        "sample_id_sha256",
        "metadata_sha256",
        "sample_manifest_sha256",
        "git_revision",
        "postprocess",
        "shape_contract",
    )
    missing = [k for k in required if not man.get(k)]
    if missing:
        raise ValueError(f"D1 manifest missing provenance keys: {missing}")
    post = man.get("postprocess") or {}
    if not isinstance(post, dict) or not post.get("shoulder_fill"):
        raise ValueError("D1 manifest.postprocess.shoulder_fill required")
    mode = _require_clean_or_pilot(man, allow_dirty_pilot=allow_dirty_pilot)
    eligible_sha = require_canonical_sample_manifest_sha(
        man.get("sample_manifest_sha256"), root=_ROOT, registry=registry
    )
    contract_ev = _bind_run_contract(target, man)

    return {
        "quality_report": str(quality_report),
        "quality_report_sha256": sha256_file(quality_report),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "coverage": coverage,
        "present": present,
        "npy_count": npy,
        "full_scan": True,
        "pass": True,
        "sample_id_sha256": man.get("sample_id_sha256"),
        "sample_manifest_sha256": eligible_sha,
        "metadata_sha256": man.get("metadata_sha256"),
        "git_revision": man.get("git_revision"),
        "git_dirty": man.get("git_dirty"),
        "postprocess": post,
        "mode": mode,
        **contract_ev,
    }


def _validate_temporal(
    target: Path,
    manifest: Path,
    quality_report: Path,
    cuda_provenance: Path,
    *,
    allow_dirty_pilot: bool,
    registry: dict,
) -> dict:
    if not manifest.is_file():
        raise ValueError(f"missing --manifest: {manifest}")
    if not quality_report.is_file():
        raise ValueError(f"missing --quality-report: {quality_report}")
    if not cuda_provenance.is_file():
        raise ValueError(f"missing --cuda-provenance: {cuda_provenance}")

    man = load_json(manifest)
    report = load_json(quality_report)
    cuda = load_json(cuda_provenance)

    out_dir = _resolve_manifest_out_dir(man)
    if out_dir.resolve() != target.resolve():
        raise ValueError(f"manifest out_dir={out_dir} does not match --dir={target}")
    if int(man.get("failed") or 0) != 0:
        raise ValueError(f"manifest failed={man.get('failed')} (must be 0 for promote)")
    saved = int(man.get("saved") or 0) + int(man.get("skipped") or 0)
    npy = count_npy(target)
    if npy <= 0:
        raise ValueError("directory has zero .npy files")
    if saved > 0 and npy < saved:
        raise ValueError(f"npy_count={npy} < manifest saved+skipped={saved}")
    n_ids = man.get("n_sample_ids")
    if n_ids is not None and int(n_ids) != npy:
        raise ValueError(f"n_sample_ids={n_ids} != npy_count={npy}")

    if report.get("pass") is not True:
        raise ValueError("temporal quality report pass!=true")
    if report.get("full_scan") is not True:
        raise ValueError("temporal quality report full_scan!=true (refuse subsample claims)")
    report_dir = Path(str(report.get("temporal_dir") or ""))
    if report_dir.resolve() != target.resolve():
        raise ValueError(
            f"quality report temporal_dir={report_dir} does not match --dir={target}"
        )
    present = int(report.get("present") or 0)
    if present != npy:
        raise ValueError(f"quality present={present} != npy_count={npy}")
    for key in ("bad_shape", "short_l", "nan_files", "inf_files", "all_zero_files", "dtype_mismatch"):
        if int(report.get(key) or 0) != 0:
            raise ValueError(f"quality report {key}={report.get(key)} (must be 0)")
    if report.get("dtype_unified") is not True:
        raise ValueError("quality report dtype_unified!=true")

    if cuda.get("pass") is not True:
        raise ValueError("cuda provenance pass!=true")
    cuda_rev = cuda.get("git_revision")
    if cuda_rev and man.get("git_revision") and cuda_rev != man.get("git_revision"):
        raise ValueError(
            f"cuda provenance git_revision={cuda_rev} != manifest.git_revision={man.get('git_revision')}"
        )
    cuda_out = str(cuda.get("temporal_output_dir") or "").replace("\\", "/")
    target_rel = rel_posix(_ROOT, target)
    if cuda_out and cuda_out != target_rel:
        raise ValueError(
            f"cuda provenance temporal_output_dir={cuda_out} != promote dir={target_rel}"
        )
    backbone_sha = (cuda.get("backbone_checkpoint_identity") or {}).get("sha256")
    if not backbone_sha:
        raise ValueError("cuda provenance missing backbone_checkpoint_identity.sha256")
    pip_freeze_sha = cuda.get("pip_freeze_sha256")
    if not pip_freeze_sha:
        raise ValueError("cuda provenance missing pip_freeze_sha256")

    required_keys = (
        "sample_id_sha256",
        "metadata_sha256",
        "min_temporal_l",
        "shape_contract",
        "git_revision",
        "backbone",
        "sample_manifest_sha256",
    )
    missing = [k for k in required_keys if not man.get(k)]
    if missing:
        raise ValueError(f"manifest missing provenance keys: {missing}")
    mode = _require_clean_or_pilot(man, allow_dirty_pilot=allow_dirty_pilot)
    backbone = man.get("backbone") or {}
    if not isinstance(backbone, dict) or not backbone.get("hub"):
        raise ValueError("manifest.backbone.hub required")
    eligible_sha = require_canonical_sample_manifest_sha(
        man.get("sample_manifest_sha256"), root=_ROOT, registry=registry
    )
    contract_ev = _bind_run_contract(target, man)

    return {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "quality_report": str(quality_report),
        "quality_report_sha256": sha256_file(quality_report),
        "cuda_provenance": str(cuda_provenance),
        "cuda_provenance_sha256": sha256_file(cuda_provenance),
        "pip_freeze_sha256": pip_freeze_sha,
        "backbone_checkpoint_sha256": backbone_sha,
        "saved": man.get("saved"),
        "skipped": man.get("skipped"),
        "failed": 0,
        "npy_count": npy,
        "full_scan": True,
        "pass": True,
        "L_min": report.get("L_min"),
        "L_max": report.get("L_max"),
        "L_mean": report.get("L_mean"),
        "sample_id_sha256": man.get("sample_id_sha256"),
        "sample_manifest_sha256": eligible_sha,
        "metadata_sha256": man.get("metadata_sha256"),
        "git_revision": man.get("git_revision"),
        "git_dirty": man.get("git_dirty"),
        "backbone": backbone,
        "mode": mode,
        **contract_ev,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validated promote of versioned feature dirs")
    ap.add_argument("--kind", required=True, choices=["skel", "temporal"])
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--quality-report", type=Path, default=None)
    ap.add_argument(
        "--cuda-provenance",
        type=Path,
        default=None,
        help="Required for temporal: CUDA canonical PROVENANCE.json",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="End-of-run provenance manifest (required for skel and temporal)",
    )
    ap.add_argument("--min-coverage", type=float, default=0.95)
    ap.add_argument("--force-legacy", action="store_true")
    ap.add_argument(
        "--allow-dirty-pilot",
        action="store_true",
        help="Pilot only: allow promote when git_dirty=true (skel or temporal)",
    )
    args = ap.parse_args()

    target = resolve_under_root(_ROOT, args.dir)
    if not target.is_dir():
        return _fail(f"not a directory: {target}")
    try:
        rel = rel_posix(_ROOT, target)
    except ValueError:
        return _fail(f"--dir must be inside repo root: {target}")

    if rel == LEGACY[args.kind] and not args.force_legacy:
        return _fail(f"Refusing legacy {LEGACY[args.kind]}. Use versioned dir.")

    reg = load_json(REGISTRY) if REGISTRY.is_file() else {}

    try:
        if args.kind == "skel":
            if args.quality_report is None:
                return _fail("--quality-report is required for skel promotion")
            if args.manifest is None:
                return _fail("--manifest (D1 end manifest) is required for skel promotion")
            qpath = resolve_under_root(_ROOT, args.quality_report)
            mpath = resolve_under_root(_ROOT, args.manifest)
            evidence = _validate_skel(
                target,
                qpath,
                mpath,
                min_coverage=args.min_coverage,
                allow_dirty_pilot=bool(args.allow_dirty_pilot),
                registry=reg,
            )
        else:
            if args.manifest is None:
                return _fail("--manifest is required for temporal promotion")
            if args.quality_report is None:
                return _fail("--quality-report is required for temporal promotion")
            if args.cuda_provenance is None:
                return _fail("--cuda-provenance is required for temporal promotion")
            mpath = resolve_under_root(_ROOT, args.manifest)
            qpath = resolve_under_root(_ROOT, args.quality_report)
            cpath = resolve_under_root(_ROOT, args.cuda_provenance)
            evidence = _validate_temporal(
                target,
                mpath,
                qpath,
                cpath,
                allow_dirty_pilot=bool(args.allow_dirty_pilot),
                registry=reg,
            )
    except ValueError as exc:
        return _fail(str(exc))

    feats = dict(reg.get("features") or {})
    now = datetime.now(timezone.utc).isoformat()
    if args.kind == "skel":
        reg["active_skel_dir"] = rel
        feats["active_skel_dir"] = rel
        feats["skel_promoted_at"] = now
        feats["skel_promotion_evidence"] = evidence
        if evidence.get("pass") is True and evidence.get("coverage", 0) >= args.min_coverage:
            reg["d1_pass"] = True
    else:
        reg["active_temporal_dir"] = rel
        feats["active_temporal_dir"] = rel
        feats["temporal_promoted_at"] = now
        feats["temporal_promotion_evidence"] = evidence
        feats["note"] = (
            "Promote requires quality-report+D1 manifest (skel) or "
            "quality-report+temporal manifest+cuda provenance; resume via _run_contract.json."
        )
        guards = dict(reg.get("pre_run_guards") or {})
        guards["temporal_promote_requires_full_scan_and_cuda_provenance"] = True
        reg["pre_run_guards"] = guards
        reasons = list(reg.get("reasons") or [])
        promo_reason = (
            f"Temporal promoted: {rel} (full_scan pass, n={evidence.get('npy_count')}, "
            f"L_raw=[{evidence.get('L_min')},{evidence.get('L_max')}], "
            f"backbone={str(evidence.get('backbone_checkpoint_sha256') or '')[:12]}…)"
        )
        reasons = [r for r in reasons if "Temporal not promoted" not in str(r)]
        if promo_reason not in reasons:
            reasons.insert(0, promo_reason)
        reg["reasons"] = reasons
    # Keep eligible SHA pinned after successful promote
    if evidence.get("sample_manifest_sha256"):
        reg["eligible_sample_manifest_sha256"] = evidence["sample_manifest_sha256"]
        reg["eligible_sample_manifest"] = "data/processed/vsl400_eligible_sample_manifest.json"
    reg["features"] = feats
    reg["updated_at"] = now

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(REGISTRY)
    print(json.dumps({"ok": True, "kind": args.kind, "active": rel, "evidence": evidence}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
