"""Regenerate temporal preflight from live HEAD + CUDA canonical evidence (post-build)."""

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
    ELIGIBLE_MANIFEST_REL,
    count_npy,
    git_dirty,
    git_rev,
    load_json,
    sha256_file,
)

CANONICAL_HEAD_PREFIX = "8741189c657a"
CUDA_PROVENANCE = _ROOT / "logs" / "preflight" / "cuda_train_canonical" / "PROVENANCE.json"
CUDA_SUMMARY = _ROOT / "logs" / "preflight" / "cuda_train_canonical" / "SUMMARY.json"
TEMPORAL_DIR_REL = "data/features/isolated_sf_temporal_v2_8741189c657a"
START_JSON = _ROOT / "logs" / "extract_temporal" / "START.json"
REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
OUT_DEFAULT = _ROOT / "logs" / "extract_skel" / "temporal_preflight.json"


def _load_json_flexible(path: Path) -> dict:
    """Load JSON; tolerate UTF-8 BOM from PowerShell-written artifacts."""
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write evidence-backed temporal preflight JSON")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--temporal-dir",
        type=Path,
        default=_ROOT / TEMPORAL_DIR_REL,
    )
    ap.add_argument(
        "--cuda-provenance",
        type=Path,
        default=CUDA_PROVENANCE,
    )
    ap.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help="Optional temporal full-scan report to bind if already present",
    )
    args = ap.parse_args()

    head = git_rev(_ROOT)
    dirty = git_dirty(_ROOT)
    reg = load_json(REGISTRY) if REGISTRY.is_file() else {}
    elig_path = _ROOT / ELIGIBLE_MANIFEST_REL
    elig_sha = sha256_file(elig_path) if elig_path.is_file() else None

    temporal_dir = args.temporal_dir
    if not temporal_dir.is_absolute():
        temporal_dir = _ROOT / temporal_dir
    man_path = temporal_dir / "_manifest.json"
    contract_path = temporal_dir / "_run_contract.json"
    npy_n = count_npy(temporal_dir) if temporal_dir.is_dir() else 0
    man = load_json(man_path) if man_path.is_file() else None
    contract = load_json(contract_path) if contract_path.is_file() else None
    start = _load_json_flexible(START_JSON) if START_JSON.is_file() else None

    cuda_prov = load_json(args.cuda_provenance) if args.cuda_provenance.is_file() else None
    cuda_sum = load_json(CUDA_SUMMARY) if CUDA_SUMMARY.is_file() else None
    train_cuda_py = _ROOT / "_venvs" / "train_cuda" / "Scripts" / "python.exe"
    legacy_env_py = _ROOT / "env" / "Scripts" / "python.exe"

    qpath = args.quality_report
    quality = None
    if qpath is not None:
        if not qpath.is_absolute():
            qpath = _ROOT / qpath
        quality = load_json(qpath) if qpath.is_file() else None

    gates = {
        "head_is_8741189c": bool(head and str(head).startswith(CANONICAL_HEAD_PREFIX)),
        "temporal_corpus_locked_at_8741189c": bool(
            man and str(man.get("git_revision") or "").startswith(CANONICAL_HEAD_PREFIX)
        ),
        "worktree_clean": dirty is False,
        "ms_sta_nogo": reg.get("status") == "NO-GO" and not reg.get("defense_ready"),
        "d1_skel_promoted": bool(reg.get("d1_pass") and reg.get("active_skel_dir")),
        "eligible_sha_match_registry": (
            elig_sha is not None
            and elig_sha == reg.get("eligible_sample_manifest_sha256")
        ),
        "cuda_provenance_pass": bool(cuda_prov and cuda_prov.get("pass") is True),
        "cuda_summary_pass": bool(cuda_sum and cuda_sum.get("pass") is True),
        "cuda_train_venv_present": train_cuda_py.is_file(),
        "legacy_env_not_required": True,
        "temporal_dir_exists": temporal_dir.is_dir(),
        "temporal_npy_24752": npy_n == 24752,
        "temporal_manifest_present": man is not None,
        "temporal_manifest_failed_zero": bool(
            man and man.get("failed") is not None and int(man.get("failed")) == 0
        ),
        "temporal_manifest_saved_24752": bool(
            man and man.get("saved") is not None and int(man.get("saved")) == 24752
        ),
        "temporal_git_revision_matches_head": bool(
            man and man.get("git_revision") == head
        ),
        "temporal_git_dirty_false": bool(man and man.get("git_dirty") is False),
        "run_contract_present": contract is not None,
        "start_json_binds_cuda_provenance": bool(
            start and str(start.get("provenance") or "").replace("\\", "/").endswith(
                "logs/preflight/cuda_train_canonical/PROVENANCE.json"
            )
        ),
        "active_temporal_still_null": reg.get("active_temporal_dir") is None,
        "quality_full_scan_pass": bool(
            quality and quality.get("pass") is True and quality.get("full_scan") is True
        ),
    }

    # Build already complete — this preflight is promote-readiness, not build-start.
    # After promotion commit, HEAD may advance; corpus lock remains manifest.git_revision.
    # worktree_clean is informational (tooling edits may dirty porcelain before commit);
    # canonical promote binds manifest.git_dirty + quality/cuda evidence instead.
    promote_blockers = [
        k
        for k, ok in gates.items()
        if k
        in (
            "temporal_corpus_locked_at_8741189c",
            "cuda_provenance_pass",
            "temporal_npy_24752",
            "temporal_manifest_failed_zero",
            "temporal_manifest_saved_24752",
            "temporal_git_dirty_false",
            "run_contract_present",
            "quality_full_scan_pass",
        )
        and not ok
    ]
    # quality may be absent until scan runs; mark separately
    if quality is None:
        if "quality_full_scan_pass" not in promote_blockers:
            promote_blockers.append("quality_full_scan_pass")

    already_promoted = reg.get("active_temporal_dir") == TEMPORAL_DIR_REL
    phase = "promoted" if already_promoted else "post_build_promote_readiness"

    backbone = None
    if cuda_prov:
        backbone = (cuda_prov.get("backbone_checkpoint_identity") or {}).get("sha256")

    report = {
        "preflight_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "git_rev": head,
        "git_dirty": dirty,
        "gates": gates,
        "promote_blockers": promote_blockers,
        "ready_to_promote_canonical_temporal": (not already_promoted) and len(promote_blockers) == 0,
        "already_promoted": already_promoted,
        "eligible_n": 24752,
        "eligible_sha": elig_sha,
        "active_skel_dir": reg.get("active_skel_dir"),
        "active_temporal_dir": reg.get("active_temporal_dir"),
        "temporal_out": TEMPORAL_DIR_REL,
        "temporal_npy_count": npy_n,
        "temporal_manifest": str(man_path) if man_path.is_file() else None,
        "temporal_manifest_sha256": sha256_file(man_path) if man_path.is_file() else None,
        "run_contract": str(contract_path) if contract_path.is_file() else None,
        "run_contract_sha256": sha256_file(contract_path) if contract_path.is_file() else None,
        "cuda_provenance": str(args.cuda_provenance) if args.cuda_provenance.is_file() else None,
        "cuda_provenance_sha256": (
            sha256_file(args.cuda_provenance) if args.cuda_provenance.is_file() else None
        ),
        "pip_freeze_sha256": (cuda_prov or {}).get("pip_freeze_sha256"),
        "backbone_checkpoint_sha256": backbone,
        "cuda_python": str(train_cuda_py) if train_cuda_py.is_file() else None,
        "rejected_python": str(legacy_env_py) if legacy_env_py.is_file() else "env/ (rejected)",
        "start_json": str(START_JSON) if START_JSON.is_file() else None,
        "quality_report": str(qpath) if quality is not None and qpath is not None else None,
        "quality_report_sha256": sha256_file(qpath) if quality is not None and qpath and qpath.is_file() else None,
        "notes": [
            "Regenerated for CUDA canonical (_venvs/train_cuda) + temporal corpus lock 8741189c…",
            "Do not use any legacy recommended_cmd pointing at env/.",
            "This file intentionally omits recommended_cmd; build already completed.",
            "Corpus git_revision is locked in temporal _manifest.json; HEAD may advance after promote commit.",
            "Promote only after quality_full_scan_pass with tensor checks.",
            f"Raw L range from build manifest: L_min={None if not man else man.get('L_min')}, "
            f"L_max={None if not man else man.get('L_max')}, L_mean={None if not man else man.get('L_mean')} "
            "(pad to L=8 at train/eval; not eight real windows).",
        ],
        "stale_replaced": {
            "previous_git_rev_in_old_preflight": "66fc5f01ee415fa1dbd68846c38b7254898aae90",
            "previous_temporal_out": "data/features/isolated_sf_temporal_v2_66fc5f01ee415fa1",
            "previous_recommended_python": "env/Scripts/python.exe",
            "action": "ignored_do_not_reuse",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "ok": True,
            "out": str(args.out),
            "ready_to_promote_canonical_temporal": report["ready_to_promote_canonical_temporal"],
            "promote_blockers": promote_blockers,
            "git_rev": head,
            "temporal_npy_count": npy_n,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
