#!/usr/bin/env python3
"""Read-only diagnostic: RGB sidecar region_validity vs tensor missing/partial masks.

Does not rewrite NPZ/sidecar/manifest. Writes create-only evidence JSON under v15.
Regenerate with --replace after contract remediation (never mutates RGB bytes).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN_V15,
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL,
    D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION_V15,
    assert_meta_tensor_agreement,
    atomic_create_json,
    executable_tree_sha256_at_revision,
    git_rev,
    load_json,
    sha256_file,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rgb-root-rel",
        default=None,
        help="Override RGB binary root (default: QUALITY_SCAN companion binary_root_rel)",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Unlink prior diagnostic JSON before create-only rewrite (RGB bytes untouched)",
    )
    args = ap.parse_args()

    quality_companion_path = (
        _ROOT / D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL
    )
    if not quality_companion_path.is_file():
        print(json.dumps({"ok": False, "error": "quality_scan_companion_missing"}, indent=2))
        return 2
    quality_companion = load_json(quality_companion_path)
    quality_companion_sha = sha256_file(quality_companion_path)

    if args.rgb_root_rel:
        rgb_rel = str(args.rgb_root_rel).replace("\\", "/")
    else:
        rgb_rel = str(quality_companion.get("binary_root_rel") or "").replace("\\", "/")
    if not rgb_rel:
        print(json.dumps({"ok": False, "error": "rgb_root_missing"}, indent=2))
        return 2
    rgb_root = (_ROOT / rgb_rel).resolve()
    meta_dir = rgb_root / "rgb_token_v3"
    if not meta_dir.is_dir():
        print(json.dumps({"ok": False, "error": "rgb_token_dir_missing", "rel": rgb_rel}, indent=2))
        return 2

    skeleton_source_root_rel = str(
        quality_companion.get("skeleton_source_root_rel") or ""
    ).replace("\\", "/")
    rgb_index_rel = f"{rgb_rel}/runtime/manifests/rgb_token_v3/INDEX.json"
    rgb_index_path = _ROOT / rgb_index_rel
    if not rgb_index_path.is_file():
        print(json.dumps({"ok": False, "error": "rgb_index_missing", "rel": rgb_index_rel}, indent=2))
        return 2
    rgb_index_sha = sha256_file(rgb_index_path)

    code_lock = git_rev(_ROOT)
    executable_tree = executable_tree_sha256_at_revision(
        FULL_CORPUS_EXECUTABLE_PATHS_V15, code_lock, root=_ROOT
    )
    approval_path = _ROOT / D0_FULL_CORPUS_APPROVAL_V15_REL
    approval_code_lock = None
    approval_tree = None
    if approval_path.is_file():
        approval = load_json(approval_path)
        approval_code_lock = approval.get("code_lock_git_revision")
        approval_tree = approval.get("executable_tree_sha256")

    metas = sorted(meta_dir.glob("*.meta.json"))
    sidecars = len(metas)
    npzs = len(list(meta_dir.glob("*.npz")))
    face_all_missing_samples = 0
    hand_all_missing_samples = 0
    zero_slices = 0
    tensor_vs_region_validity_mismatch = 0
    mismatch_examples: list[str] = []

    for meta_path in metas:
        sid = meta_path.name.replace(".meta.json", "")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        npz_path = meta_dir / f"{sid}.npz"
        windows = meta.get("windows") or []
        sample_face_all = False
        sample_hand_all = False
        for w in windows:
            if not isinstance(w, dict):
                continue
            rv_root = w.get("region_validity") or {}
            if not isinstance(rv_root, dict):
                continue
            for region in ("face", "left_hand", "right_hand"):
                rv = rv_root.get(region) or {}
                if isinstance(rv, dict) and type(rv.get("n_slices")) is int and rv.get("n_slices") <= 0:
                    zero_slices += 1
            face = rv_root.get("face") or {}
            if isinstance(face, dict) and face.get("all_missing") is True:
                sample_face_all = True
            for hand in ("left_hand", "right_hand"):
                hv = rv_root.get(hand) or {}
                if isinstance(hv, dict) and hv.get("all_missing") is True:
                    sample_hand_all = True
        if sample_face_all:
            face_all_missing_samples += 1
        if sample_hand_all:
            hand_all_missing_samples += 1

        if not npz_path.is_file():
            tensor_vs_region_validity_mismatch += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(f"npz_missing:{sid}")
            continue
        try:
            assert_meta_tensor_agreement(rgb_path=npz_path, meta=meta, sample_id=sid)
        except Exception as exc:  # noqa: BLE001
            tensor_vs_region_validity_mismatch += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(f"{sid}:{exc}")

    report = {
        "schema": "ms_sta_v5_d0_full_corpus_v15_rgb_region_validity_diagnostic_v2",
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "output_version": OUTPUT_VERSION_V15,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rgb_root_rel": rgb_rel,
        "skeleton_source_root_rel": skeleton_source_root_rel,
        "quality_scan_used_local_skel_manifest_under_carry_forward_sha256": quality_companion_sha,
        "rgb_index_rel": rgb_index_rel,
        "rgb_index_sha256": rgb_index_sha,
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": executable_tree,
        "approval_code_lock_git_revision": approval_code_lock,
        "approval_executable_tree_sha256": approval_tree,
        "npz_count": npzs,
        "sidecar_count": sidecars,
        "zero_slices": zero_slices,
        "face_all_missing_samples": face_all_missing_samples,
        "hand_all_missing_samples": hand_all_missing_samples,
        "tensor_vs_region_validity_mismatch": tensor_vs_region_validity_mismatch,
        "mismatch_examples": mismatch_examples,
        "ok": (
            sidecars == 24752
            and npzs == 24752
            and zero_slices == 0
            and face_all_missing_samples == 1320
            and hand_all_missing_samples == 0
            and tensor_vs_region_validity_mismatch == 0
            and bool(rgb_index_sha)
            and bool(quality_companion_sha)
            and bool(code_lock)
            and bool(executable_tree)
        ),
        "notes": [
            "Read-only; does not rewrite RGB NPZ/sidecar/manifest.",
            "Canonical agreement via assert_meta_tensor_agreement(region_validity).",
            "Provenance binds quality-scan companion, RGB INDEX, code-lock/executable-tree.",
        ],
    }
    out = _ROOT / D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC_REL
    if args.replace and out.is_file():
        out.unlink()
    sha = atomic_create_json(out, report)
    print(json.dumps({**report, "report_sha256": sha}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
