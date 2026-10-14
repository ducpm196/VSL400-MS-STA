"""Extract 75-joint skeleton features + Savitzky-Golay smooth → local_skel/*.npy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_prep.raw_video_resolve import SKIP_VIDEO_DATASETS, RawVideoResolver
from src.data_prep.skel_postprocess import POSTPROCESS_PROVENANCE, postprocess_sequence
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

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

FAIL_FAST_DATASETS = set()
DEFAULT_STATUS = _PROJECT_ROOT / "logs" / "extract_skel" / "status.json"
# Holistic is heavy; stride>1 keeps temporal structure while speeding extract.
DEFAULT_FRAME_STRIDE = 2


class SignLanguageDataEngine:
    def __init__(self, *, model_complexity: int = 0, frame_stride: int = DEFAULT_FRAME_STRIDE) -> None:
        # Lazy MediaPipe import — keeps module importable in CI/train envs without mediapipe.
        import mediapipe as mp

        self.frame_stride = max(1, int(frame_stride))
        self.model_complexity = int(model_complexity)
        self.mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            refine_face_landmarks=True,
        )
        self.pose_indices = [0, 2, 5, 9, 11, 12, 13, 14, 15, 16, 23, 24]  # 12 unique (no dup hip)
        self.face_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        ]

    def extract_frame_landmarks(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (joints[75,3], raw_pose[33,3], missing_mask[75] bool)."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_holistic.process(image_rgb)

        pose_arr = np.zeros((33, 3), dtype=np.float32)
        face_arr = np.zeros((478, 3), dtype=np.float32)
        lh_arr = np.zeros((21, 3), dtype=np.float32)
        rh_arr = np.zeros((21, 3), dtype=np.float32)
        pose_ok = face_ok = lh_ok = rh_ok = False

        if results.pose_landmarks:
            pose_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32
            )
            pose_ok = True
        if results.face_landmarks:
            face_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark], dtype=np.float32
            )
            face_ok = True
        if results.left_hand_landmarks:
            lh_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32
            )
            lh_ok = True
        if results.right_hand_landmarks:
            rh_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32
            )
            rh_ok = True

        selected_pose = pose_arr[self.pose_indices]
        selected_face = face_arr[self.face_indices]
        frame_features = np.concatenate([selected_pose, selected_face, lh_arr, rh_arr], axis=0)
        # Missing mask before any normalization (undetected groups → all joints in group missing)
        missing = np.zeros((75,), dtype=bool)
        if not pose_ok:
            missing[0:12] = True
        if not face_ok:
            missing[12:33] = True
        if not lh_ok:
            missing[33:54] = True
        if not rh_ok:
            missing[54:75] = True
        return frame_features, pose_arr, missing

    def process_video_to_tensor(self, video_path: Path) -> np.ndarray | None:
        cap = cv2.VideoCapture(str(video_path))
        video_data: list[np.ndarray] = []
        raw_pose_history: list[np.ndarray] = []
        missing_hist: list[np.ndarray] = []
        frame_i = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_i % self.frame_stride == 0:
                feat, pose, missing = self.extract_frame_landmarks(frame)
                video_data.append(feat)
                raw_pose_history.append(pose)
                missing_hist.append(missing)
            frame_i += 1

        cap.release()

        if len(video_data) < 5:
            return None

        return postprocess_sequence(
            np.array(video_data, dtype=np.float32),
            np.array(raw_pose_history, dtype=np.float32),
            np.array(missing_hist, dtype=bool),
        )

    # Back-compat alias for callers/tests that still reference the class method.
    postprocess_sequence = staticmethod(postprocess_sequence)

    def run_pipeline(
        self,
        metadata_csv_path: Path,
        processed_dir: Path,
        output_feature_dir: Path,
        *,
        datasets: set[str] | None = None,
        splits: set[str] | None = None,
        limit: int = 0,
        skip_existing: bool = True,
        status_path: Path = DEFAULT_STATUS,
        progress_every: int = 25,
        manifest_path: Path | None = None,
        eligible_ids: set[str] | None = None,
        sample_manifest_sha256: str | None = None,
        write_contract: bool = True,
        resume: bool = False,
    ) -> dict:
        df = pd.read_csv(metadata_csv_path)
        if datasets:
            df = df[df["dataset"].isin(datasets)].reset_index(drop=True)
        if splits:
            df = df[df["split"].isin(splits)].reset_index(drop=True)
        # Prefer val → test → train so Gate D1/E* can unblock early.
        if "split" in df.columns:
            order = {"val": 0, "test": 1, "train": 2}
            df = df.assign(_ord=df["split"].map(lambda s: order.get(str(s), 9))).sort_values(
                ["_ord", "video_id"]
            ).drop(columns=["_ord"]).reset_index(drop=True)
        if eligible_ids is not None:
            before = len(df)
            df = df[df["video_id"].astype(str).isin(eligible_ids)].reset_index(drop=True)
            print(f"[*] Eligible filter: {before} → {len(df)} rows", flush=True)
        if limit and limit > 0:
            df = df.iloc[: int(limit)].reset_index(drop=True)
        output_feature_dir.mkdir(parents=True, exist_ok=True)
        split_tag = "all" if not splits else "_".join(sorted(splits))
        ds_tag = "all" if not datasets else "_".join(sorted(datasets))
        if status_path == DEFAULT_STATUS:
            status_path = _PROJECT_ROOT / "logs" / "extract_skel" / f"status_{ds_tag}_{split_tag}.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path is None:
            manifest_path = output_feature_dir / f"_manifest_{ds_tag}_{split_tag}.json"

        planned_ids = df["video_id"].astype(str).tolist()
        meta_sha = sha256_file(Path(metadata_csv_path)) if Path(metadata_csv_path).is_file() else None
        contract = build_run_contract(
            kind="skel_d1",
            sample_ids=planned_ids,
            metadata_sha256=meta_sha,
            splits=sorted(splits) if splits else "ALL",
            config={
                "frame_stride": self.frame_stride,
                "model_complexity": self.model_complexity,
                "datasets": sorted(datasets) if datasets else "ALL",
                "limit": int(limit or 0),
            },
            root=_PROJECT_ROOT,
            sample_manifest_sha256=sample_manifest_sha256,
        )
        if resume:
            require_resume_contract(output_feature_dir, contract)
        elif write_contract:
            # Atomic contract BEFORE first .npy
            write_run_contract(output_feature_dir, contract)
            print(f"[OK] wrote _run_contract.json sample_id_sha256={contract['sample_id_sha256']}", flush=True)

        resolver = RawVideoResolver(_PROJECT_ROOT)
        # Keep processed_dir override consistent with legacy API.
        resolver.processed_dir = processed_dir

        total = len(df)
        saved = 0
        skipped = 0
        missing: list[str] = []
        soft_miss = 0
        shapes: list[tuple[int, int, int]] = []
        t0 = time.time()

        def write_status(**extra) -> None:
            body = {
                "phase": "running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total": total,
                "saved": saved,
                "skipped_existing": skipped,
                "missing_fail_fast": len(missing),
                "missing_soft": soft_miss,
                "elapsed_s": round(time.time() - t0, 1),
                "datasets": sorted(datasets) if datasets else "ALL",
                "splits": sorted(splits) if splits else "ALL",
                **extra,
            }
            status_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[*] Skeleton extract: rows={total} datasets={sorted(datasets) if datasets else 'ALL'} "
            f"splits={sorted(splits) if splits else 'ALL'} skip_existing={skip_existing}",
            flush=True,
        )
        write_status()

        for i, row in df.iterrows():
            ds = str(row["dataset"])
            vid = str(row["video_id"])
            out = output_feature_dir / f"{vid}.npy"
            if skip_existing and out.is_file():
                skipped += 1
                continue

            if ds in SKIP_VIDEO_DATASETS:
                soft_miss += 1
                continue

            video_path = resolver.resolve(row)
            if video_path is None:
                label = f"{ds}/{row['signer_id']}/{vid}.mp4"
                if ds in FAIL_FAST_DATASETS:
                    missing.append(label)
                else:
                    soft_miss += 1
                continue

            tensor_data = self.process_video_to_tensor(video_path)
            if tensor_data is None:
                label = f"{video_path} (< 5 frames)"
                if ds in FAIL_FAST_DATASETS:
                    missing.append(label)
                else:
                    soft_miss += 1
                continue

            np.save(out, tensor_data)
            shapes.append(tuple(int(x) for x in tensor_data.shape))
            saved += 1
            if saved % progress_every == 0 or (i + 1) == total:
                rate = saved / max(time.time() - t0, 1e-6)
                print(
                    f"[->] saved={saved} skipped={skipped} soft_miss={soft_miss} "
                    f"rate={rate:.2f}/s last={ds}/{vid}",
                    flush=True,
                )
                write_status(last=f"{ds}/{vid}", rate_per_s=round(rate, 3))

        if missing:
            preview = "\n  ".join(missing[:15])
            suffix = f"\n  ... và {len(missing) - 15} lỗi khác" if len(missing) > 15 else ""
            write_status(phase="failed", errors=len(missing))
            raise RuntimeError(
                f"Trích xuất skeleton thất bại ({len(missing)} clip Continuous_News):\n  {preview}{suffix}"
            )

        result = {
            "saved": saved,
            "skipped_existing": skipped,
            "missing_soft": soft_miss,
            "total": total,
            "output": str(output_feature_dir),
            "gate": "D1",
        }
        planned_ids = df["video_id"].astype(str).tolist()
        meta_path = Path(metadata_csv_path)
        # Cache manifest (Option B Gate D1) with provenance for promotion.
        manifest = {
            "gate": "D1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_feature_dir),
            "datasets": sorted(datasets) if datasets else "ALL",
            "splits": sorted(splits) if splits else "ALL",
            "rows_planned": total,
            "sample_ids": planned_ids,
            "sample_id_sha256": sha256_id_list(planned_ids),
            "sample_manifest_sha256": sample_manifest_sha256,
            "metadata_sha256": sha256_file(meta_path) if meta_path.is_file() else None,
            "git_revision": git_rev(_PROJECT_ROOT),
            "git_dirty": git_dirty(_PROJECT_ROOT),
            "run_contract": "_run_contract.json",
            "saved_this_run": saved,
            "skipped_existing": skipped,
            "missing_soft": soft_miss,
            "frame_stride": self.frame_stride,
            "model_complexity": self.model_complexity,
            "postprocess": dict(POSTPROCESS_PROVENANCE),
            "shape_contract": "(T, 75, 3)",
            "shapes_saved_n": len(shapes),
            "t_min": min((s[0] for s in shapes), default=None),
            "t_max": max((s[0] for s in shapes), default=None),
            "note": (
                "git_dirty=true means this SHA does not uniquely identify running code; "
                "prefer a clean code-lock commit before canonical corpus runs. "
                "Resume uses _run_contract.json, not this end-of-run manifest."
            ),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_status(phase="completed", **result, manifest=str(manifest_path))
        print(
            f"[OK] Skeleton extract done: saved={saved} skipped={skipped} "
            f"soft_miss={soft_miss} → {output_feature_dir}",
            flush=True,
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract local_skel/*.npy from videos")
    parser.add_argument(
        "--datasets",
        default="VSL400",
        help="Comma-separated dataset filter (default: VSL400 for Option B). Use ALL for every row.",
    )
    parser.add_argument(
        "--splits",
        default="",
        help="Optional comma-separated splits (val,test,train). Empty = all splits.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows after filters")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--model-complexity", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Feature output directory. Required for --clean-d1 (versioned path).",
    )
    parser.add_argument(
        "--clean-d1",
        action="store_true",
        help="Clean D1 mode: require --output-dir versioned (not legacy local_skel); fresh/empty dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume non-empty versioned dir; requires matching _run_contract.json (fail-closed).",
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help="Eligible-ID manifest (excludes proven-bad clips like 019442). Required for --clean-d1.",
    )
    parser.add_argument(
        "--allow-legacy-skel",
        action="store_true",
        help="Allow writing to data/features/local_skel (discouraged; mixes old artifacts).",
    )
    args = parser.parse_args()

    ds_raw = args.datasets.strip()
    datasets: set[str] | None
    if ds_raw.upper() == "ALL":
        datasets = None
    else:
        datasets = {x.strip() for x in ds_raw.split(",") if x.strip()}

    splits_raw = args.splits.strip()
    splits: set[str] | None = None
    if splits_raw:
        splits = {x.strip() for x in splits_raw.split(",") if x.strip()}

    eligible_ids: set[str] | None = None
    sample_manifest_sha: str | None = None
    if args.sample_manifest is not None:
        man_path = (
            args.sample_manifest
            if args.sample_manifest.is_absolute()
            else _PROJECT_ROOT / args.sample_manifest
        )
        pack = load_eligible_sample_manifest(man_path, root=_PROJECT_ROOT)
        eligible_ids = pack["eligible_set"]
        sample_manifest_sha = pack["sha256"]
        print(
            f"[OK] sample-manifest eligible_n={len(eligible_ids)} excluded_n={len(pack['excluded'])} "
            f"sha={sample_manifest_sha[:12]}…",
            flush=True,
        )
    elif args.clean_d1:
        raise SystemExit(
            "[FAIL] --clean-d1 requires --sample-manifest "
            "(e.g. data/processed/vsl400_eligible_sample_manifest.json)"
        )

    legacy = (_PROJECT_ROOT / "data" / "features" / "local_skel").resolve()
    if args.clean_d1:
        if args.output_dir is None:
            raise SystemExit(
                "[FAIL] --clean-d1 requires --output-dir "
                "(e.g. data/features/local_skel_v2_<tag>). Do not write into local_skel."
            )
        out_dir = args.output_dir if args.output_dir.is_absolute() else _PROJECT_ROOT / args.output_dir
        if out_dir.resolve() == legacy and not args.allow_legacy_skel:
            raise SystemExit(
                "[FAIL] --clean-d1 refuses legacy data/features/local_skel. "
                "Use a new versioned directory."
            )
        assert_fresh_or_empty_dir(out_dir, resume=bool(args.resume), label="clean-d1")
        skip_existing = bool(args.resume)
    else:
        if args.output_dir is None:
            out_dir = legacy
            if not args.allow_legacy_skel:
                print(
                    "[WARN] Writing to legacy local_skel with skip_existing="
                    f"{not args.no_skip_existing}. For clean D1 use --clean-d1 "
                    "--output-dir data/features/local_skel_v2_<tag> "
                    "--sample-manifest data/processed/vsl400_eligible_sample_manifest.json"
                )
        else:
            out_dir = args.output_dir if args.output_dir.is_absolute() else _PROJECT_ROOT / args.output_dir
            if not args.resume:
                assert_fresh_or_empty_dir(out_dir, resume=False, label="output-dir")
        skip_existing = not args.no_skip_existing if not args.resume else True

    engine = SignLanguageDataEngine(
        model_complexity=args.model_complexity,
        frame_stride=args.frame_stride,
    )
    engine.run_pipeline(
        metadata_csv_path=_PROJECT_ROOT / "metadata.csv",
        processed_dir=_PROJECT_ROOT / "data" / "processed",
        output_feature_dir=out_dir,
        datasets=datasets,
        splits=splits,
        limit=max(0, int(args.limit)),
        skip_existing=skip_existing,
        progress_every=max(1, args.progress_every),
        eligible_ids=eligible_ids,
        sample_manifest_sha256=sample_manifest_sha,
        write_contract=True,
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
