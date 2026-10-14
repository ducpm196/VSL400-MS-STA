"""Build RGB temporal cache [L,2304] for Option B (required for B1/B4 claims).

Versioned output only for clean corpus jobs:
  --output-dir data/features/isolated_sf_temporal_v2_<tag>
Do NOT write into legacy isolated_sf_temporal without --allow-legacy-temporal.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_prep.raw_video_resolve import RawVideoResolver
from src.load_env import load_project_env
from src.models.feature_versioning import (
    assert_fresh_or_empty_dir,
    build_run_contract,
    git_dirty,
    git_rev,
    load_eligible_sample_manifest,
    require_resume_contract,
    sha256_file,
    sha256_id_list,
    write_run_contract,
)
from src.models.ms_sta_features import FIXED_RGB_L, MIN_TEMPORAL_L, RGB_TEMPORAL_REL
from src.models.rgb_backbone import WINDOW_SIZE, SlowFastFeatureExtractor, _window_tensors
from src.training.train_vsl400_wordlevel import _read_video_frames_sparse

LEGACY_TEMPORAL = _ROOT / RGB_TEMPORAL_REL


def _git_rev() -> str | None:
    return git_rev(_ROOT)


def _sha256_path(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_file(path)


@torch.no_grad()
def extract_windows_2304(
    extractor: SlowFastFeatureExtractor, video_path: Path, max_windows: int = FIXED_RGB_L
) -> np.ndarray:
    frames = _read_video_frames_sparse(video_path, num_frames=max(WINDOW_SIZE, 64))
    n = frames.shape[0]
    step = max(WINDOW_SIZE // 2, 8)
    starts = list(range(0, max(n - WINDOW_SIZE + 1, 1), step))
    if not starts or starts[-1] + WINDOW_SIZE < n:
        starts.append(max(0, n - WINDOW_SIZE))
    if len(starts) < MIN_TEMPORAL_L and n >= WINDOW_SIZE:
        starts = sorted({0, max(0, n - WINDOW_SIZE)})
        if len(starts) < MIN_TEMPORAL_L and n > WINDOW_SIZE:
            mid = max(0, (n - WINDOW_SIZE) // 2)
            starts = sorted({0, mid, max(0, n - WINDOW_SIZE)})
    if len(starts) < MIN_TEMPORAL_L:
        raise ValueError(f"insufficient windows L={len(starts)} < {MIN_TEMPORAL_L} (n_frames={n})")
    if len(starts) > max_windows:
        picks = np.linspace(0, len(starts) - 1, max_windows).astype(int)
        starts = [starts[int(i)] for i in picks]
    feats = []
    device = extractor.device
    for start in starts:
        slow, fast = _window_tensors(frames, start, WINDOW_SIZE)
        feat = extractor.backbone([slow.to(device), fast.to(device)])
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        feats.append(feat.flatten(1).float().squeeze(0).cpu().numpy())
    arr = np.stack(feats, axis=0).astype(np.float32)
    if arr.shape[0] < MIN_TEMPORAL_L:
        raise ValueError(f"raw temporal L={arr.shape[0]} < {MIN_TEMPORAL_L}")
    return arr


def main() -> int:
    load_project_env()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Build versioned RGB temporal cache")
    ap.add_argument("--splits", default="val", help="Comma splits")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=FIXED_RGB_L)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Required versioned dir, e.g. data/features/isolated_sf_temporal_v2_<tag>",
    )
    ap.add_argument(
        "--allow-legacy-temporal",
        action="store_true",
        help="Allow writing to legacy isolated_sf_temporal (discouraged)",
    )
    ap.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated alias; prefer --resume for non-empty dirs",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume non-empty versioned dir; requires matching _run_contract.json (fail-closed)",
    )
    ap.add_argument(
        "--sample-manifest",
        type=Path,
        required=True,
        help="Eligible-ID manifest (excludes 019442). Required for temporal corpus builds.",
    )
    args = ap.parse_args()

    out_dir = args.output_dir if args.output_dir.is_absolute() else _ROOT / args.output_dir
    if out_dir.resolve() == LEGACY_TEMPORAL.resolve() and not args.allow_legacy_temporal:
        print(
            f"[FAIL] Refusing legacy {LEGACY_TEMPORAL}. "
            "Use --output-dir data/features/isolated_sf_temporal_v2_<tag>"
        )
        return 2

    resume = bool(args.resume or args.skip_existing)
    try:
        assert_fresh_or_empty_dir(out_dir, resume=resume, label="temporal-build")
    except SystemExit as exc:
        print(str(exc))
        return 2

    man_path = (
        args.sample_manifest if args.sample_manifest.is_absolute() else _ROOT / args.sample_manifest
    )
    try:
        pack = load_eligible_sample_manifest(man_path, root=_ROOT)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    eligible = pack["eligible_set"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extractor = SlowFastFeatureExtractor(device=device)
    resolver = RawVideoResolver(_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {x.strip() for x in args.splits.split(",") if x.strip()}
    meta_path = _ROOT / "metadata.csv"
    df = pd.read_csv(meta_path)
    df = df[(df["dataset"] == "VSL400") & (df["split"].isin(splits))].reset_index(drop=True)
    before = len(df)
    df = df[df["video_id"].astype(str).isin(eligible)].reset_index(drop=True)
    print(f"[*] Eligible filter: {before} → {len(df)} (excluded from manifest applied)")
    sample_ids = df["video_id"].astype(str).tolist()
    if args.limit > 0:
        df = df.iloc[: args.limit]
        sample_ids = df["video_id"].astype(str).tolist()
    ids_hash = sha256_id_list(sample_ids)
    meta_sha = sha256_file(meta_path)
    contract = build_run_contract(
        kind="rgb_temporal",
        sample_ids=sample_ids,
        metadata_sha256=meta_sha,
        splits=sorted(splits),
        config={
            "max_windows": int(args.max_windows),
            "fixed_rgb_l": FIXED_RGB_L,
            "min_temporal_l": MIN_TEMPORAL_L,
            "limit": int(args.limit or 0),
        },
        root=_ROOT,
        sample_manifest_sha256=pack["sha256"],
    )
    try:
        if resume:
            require_resume_contract(out_dir, contract)
        else:
            write_run_contract(out_dir, contract)
            print(f"[OK] wrote _run_contract.json sample_id_sha256={ids_hash}")
    except SystemExit as exc:
        print(str(exc))
        return 2

    saved = skipped = failed = 0
    lens: list[int] = []
    by_split: dict[str, dict] = {s: {"n": 0, "saved": 0, "failed": 0} for s in splits}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        split = str(row["split"])
        by_split.setdefault(split, {"n": 0, "saved": 0, "failed": 0})
        by_split[split]["n"] += 1
        out = out_dir / f"{vid}.npy"
        if resume and out.is_file():
            skipped += 1
            continue
        path = resolver.resolve(row)
        if path is None or not path.is_file():
            failed += 1
            by_split[split]["failed"] += 1
            continue
        try:
            arr = extract_windows_2304(extractor, path, max_windows=args.max_windows)
            if arr.ndim != 2 or arr.shape[1] != 2304:
                raise ValueError(f"bad shape {arr.shape}")
            np.save(out, arr)
            lens.append(int(arr.shape[0]))
            saved += 1
            by_split[split]["saved"] += 1
            if saved % 25 == 0:
                print(f"[->] saved={saved} L_mean={np.mean(lens):.2f}", flush=True)
        except Exception as exc:
            failed += 1
            by_split[split]["failed"] += 1
            print(f"[WARN] {vid}: {exc}", flush=True)

    backbone_note = {
        "class": "SlowFastFeatureExtractor",
        "hub": "pytorchvideo.models.hub.slowfast_r50",
        "window_size": WINDOW_SIZE,
        "feat_dim": 2304,
        "frozen": True,
    }
    # Optional local weight file if present
    for cand in (
        _ROOT / "data" / "processed" / "slowfast_r50.pth",
        _ROOT / "checkpoints" / "slowfast_r50.pth",
    ):
        digest = _sha256_path(cand)
        if digest:
            backbone_note["weights_path"] = str(cand.relative_to(_ROOT))
            backbone_note["weights_sha256"] = digest
            break

    man_rel = out_dir.relative_to(_ROOT) if out_dir.is_relative_to(_ROOT) else out_dir
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(man_rel).replace("\\", "/"),
        "splits": sorted(splits),
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
        "by_split": by_split,
        "L_min": min(lens) if lens else None,
        "L_max": max(lens) if lens else None,
        "L_mean": float(np.mean(lens)) if lens else None,
        "shape_contract": "(L, 2304)",
        "train_eval_fixed_L": FIXED_RGB_L,
        "min_temporal_l": MIN_TEMPORAL_L,
        "sample_id_sha256": ids_hash,
        "n_sample_ids": len(sample_ids),
        "sample_manifest_sha256": pack["sha256"],
        "excluded_from_manifest": pack["excluded"],
        "metadata_sha256": _sha256_path(meta_path),
        "git_revision": _git_rev(),
        "git_dirty": git_dirty(_ROOT),
        "run_contract": "_run_contract.json",
        "backbone": backbone_note,
        "resume": resume,
        "active_selection": (
            "Promote with: python scripts/promote_feature_dir.py --kind temporal "
            f"--dir {man_rel} --manifest {Path(str(man_rel)) / '_manifest.json'}"
        ),
        "note": (
            f"Raw L<{MIN_TEMPORAL_L} rejected. Eligible IDs exclude proven-bad clips "
            f"(e.g. 019442). Resume uses _run_contract.json, not this end manifest. "
            "Canonical promote requires git_dirty=false unless --allow-dirty-pilot."
        ),
    }
    man_name = f"_manifest_{'_'.join(sorted(splits))}.json"
    (out_dir / man_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if saved or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
