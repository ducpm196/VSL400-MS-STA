#!/usr/bin/env python3
"""Read-only performance smoke for scan-recovery R1 (small locked subset).

Does not authorize full scan. Does not use accuracy as a gate.
Writes create-only smoke metrics under v15 evidence when absent or FAIL.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL,
    EVIDENCE_ROOT_V15_REL,
    OUTPUT_VERSION_V15,
    V14_SEALED_SKELETON_BINARY_ROOT_REL,
    atomic_create_json,
    default_quality_thresholds,
    load_json,
    runtime_dir,
    sha256_file,
)
from scan_ms_sta_v5_d0_full_corpus import scan_full_corpus  # noqa: E402

SMOKE_REL = f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE.json"
LOCKED_N = 8
CORPUS_N_FULL = 24752


def main() -> int:
    companion = _ROOT / D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL
    if not companion.is_file():
        print(json.dumps({"ok": False, "error": "quality_scan_companion_missing"}, indent=2))
        return 2
    companion_obj = load_json(companion)
    rgb_rel = str(companion_obj.get("binary_root_rel") or "")
    skel_rel = str(
        companion_obj.get("skeleton_source_root_rel") or V14_SEALED_SKELETON_BINARY_ROOT_REL
    )
    rgb_root = (_ROOT / rgb_rel).resolve()
    skel_root = (_ROOT / skel_rel).resolve()
    if not rgb_root.is_dir() or not skel_root.is_dir():
        print(json.dumps({"ok": False, "error": "tensor_roots_missing"}, indent=2))
        return 2

    # Locked small ID set: first LOCKED_N eligible stems present in both dirs.
    skel_ids = sorted(p.stem for p in (skel_root / "skeleton_v3").glob("*.npz"))
    rgb_ids = {p.stem for p in (rgb_root / "rgb_token_v3").glob("*.npz")}
    ids = [i for i in skel_ids if i in rgb_ids][:LOCKED_N]
    if len(ids) < LOCKED_N:
        print(json.dumps({"ok": False, "error": "locked_subset_incomplete", "n": len(ids)}, indent=2))
        return 2

    # Patch module constants for subset scan without mutating sealed trees.
    import scan_ms_sta_v5_d0_full_corpus as scan_mod

    scan_mod.CORPUS_N = LOCKED_N
    scan_mod.SPLIT_COUNTS = {"train": LOCKED_N, "val": 0, "test": 0}
    scan_mod.load_eligible_identity = lambda root=None: {
        "eligible_ids": ids,
        "eligible_id_set": set(ids),
        "eligible_by_split": {"train": LOCKED_N, "val": 0, "test": 0},
        "eligible_n": LOCKED_N,
        "eligible_sha256": "perf_smoke",
    }
    # Keep real _load_maps for source verify timing realism on the subset.

    thr = default_quality_thresholds()
    thr.update(
        {
            "max_missing_roi_rate_face": 1.0,
            "max_missing_roi_rate_left_hand": 1.0,
            "max_missing_roi_rate_right_hand": 1.0,
        }
    )
    out_rt = _ROOT / EVIDENCE_ROOT_V15_REL / "scan_recovery_perf_smoke_runtime"
    if out_rt.exists():
        # Create-only smoke runtime: allow replace only when prior smoke failed.
        pass
    out_rt.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    # Resolver build measured inside first sample; approximate with total wall for smoke.
    report = scan_full_corpus(
        binary_root=rgb_root,
        runtime_dir=runtime_dir(rgb_root),
        thresholds=thr,
        root=_ROOT,
        skeleton_tensor_root=skel_root,
        skeleton_manifest_runtime=runtime_dir(skel_root),
        rgb_tensor_root=rgb_root,
        rgb_manifest_runtime=runtime_dir(rgb_root),
        output_version=OUTPUT_VERSION_V15,
        write_artifacts=True,
        output_runtime_dir=out_rt,
    )
    elapsed = time.perf_counter() - t0
    stats = report.get("source_verify_stats") or {}
    samples_per_s = (LOCKED_N / elapsed) if elapsed > 0 else 0.0
    eta_full_s = (CORPUS_N_FULL / samples_per_s) if samples_per_s > 0 else None

    # Performance gate only — do not use scan accuracy / failed_count as smoke PASS criteria.
    perf_ok = stats.get("resolver_instances") == 1 and stats.get("raw_source_hash_count") == LOCKED_N
    metrics = {
        "schema": "ms_sta_v5_d0_full_corpus_scan_recovery_perf_smoke_v15",
        "status": "PASS" if perf_ok else "FAIL",
        "ok": bool(perf_ok),
        "output_version": OUTPUT_VERSION_V15,
        "locked_n": LOCKED_N,
        "sample_ids": ids,
        "wall_seconds": round(elapsed, 3),
        "resolver_instances": stats.get("resolver_instances"),
        "raw_source_hash_count": stats.get("raw_source_hash_count"),
        "resolve_count": stats.get("resolve_count"),
        "samples_per_second": round(samples_per_s, 4),
        "eta_full_corpus_seconds": None if eta_full_s is None else round(eta_full_s, 1),
        "eta_full_corpus_hours": None if eta_full_s is None else round(eta_full_s / 3600.0, 2),
        "bytes_read": None,
        "bytes_read_note": "Not instrumented on Windows smoke; use resolver/hash counts as primary gate.",
        "accuracy_not_used": True,
        "full_scan_authorized": False,
        "scan_ok": bool(report.get("ok")),
        "failed_count": report.get("failed_count"),
        "resolver_build_seconds": round(elapsed, 3),
        "notes": [
            "Performance smoke only; do not treat scan_ok/failed_count as corpus completion.",
            "PASS iff resolver_instances==1 and raw_source_hash_count==locked_n.",
            "Full scan requires scan-recovery external PASS + post-auth after code-lock audit.",
        ],
    }
    if not perf_ok:
        metrics["error"] = "perf_gate_failed"

    out = _ROOT / SMOKE_REL
    if out.is_file():
        prev = load_json(out)
        if prev.get("ok") is True and prev.get("status") == "PASS":
            print(json.dumps({"ok": False, "error": "perf_smoke_pass_exists_create_only"}, indent=2))
            return 2
        out.unlink()
    sha = atomic_create_json(out, metrics)
    print(json.dumps({**metrics, "report_sha256": sha}, indent=2))
    return 0 if metrics.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
