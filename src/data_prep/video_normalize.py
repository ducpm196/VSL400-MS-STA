"""Normalize isolated-sign videos to 25 FPS / 720p (skip VOYA npz)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_prep.raw_video_resolve import SKIP_VIDEO_DATASETS, RawVideoResolver

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SKIP_NORMALIZE_DATASETS = set(SKIP_VIDEO_DATASETS)


def normalize_video_stream(input_path: Path, output_path: Path) -> bool:
    """Normalize to 25 FPS, 720p H.264, no audio."""
    if not input_path.exists():
        print(f"[X] Không tìm thấy file gốc: {input_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-r",
        "25",
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-an",
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[X] Lỗi FFmpeg xử lý file {input_path}: {exc}")
        return False


def batch_process_pipeline(
    metadata_csv_path: Path,
    *,
    datasets: set[str] | None = None,
    skip_existing: bool = True,
) -> None:
    df = pd.read_csv(metadata_csv_path)
    if datasets:
        df = df[df["dataset"].isin(datasets)].reset_index(drop=True)
    print(f"[*] Normalize: rows={len(df)} datasets={sorted(datasets) if datasets else 'ALL'}")

    resolver = RawVideoResolver(_PROJECT_ROOT)
    success_count = 0
    skipped = 0
    failures: list[str] = []

    for _, row in df.iterrows():
        ds = str(row["dataset"])
        if ds in SKIP_NORMALIZE_DATASETS:
            skipped += 1
            continue

        out = resolver.processed_output_path(row)
        if skip_existing and out.is_file():
            skipped += 1
            continue

        src = resolver.resolve(row)
        if src is None:
            failures.append(f"Thiếu video nguồn: {ds}/{row['signer_id']}/{row['video_id']}.mp4")
            continue

        if normalize_video_stream(src, out):
            success_count += 1
            if success_count % 10 == 0:
                print(f"[->] Đã chuẩn hóa thành công: {success_count} video.")
        else:
            failures.append(f"FFmpeg thất bại: {src}")

    if failures:
        preview = "\n  ".join(failures[:15])
        suffix = f"\n  ... và {len(failures) - 15} lỗi khác" if len(failures) > 15 else ""
        raise RuntimeError(f"Chuẩn hóa video thất bại ({len(failures)} lỗi):\n  {preview}{suffix}")

    print(f"[OK] Normalize done: success={success_count} skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize videos for Stage-1 RGB path")
    parser.add_argument(
        "--datasets",
        default="VSL-Cropped",
        help="Comma-separated datasets (default: VSL-Cropped). VOYA skipped.",
    )
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    ds_raw = args.datasets.strip()
    datasets = None if ds_raw.upper() == "ALL" else {x.strip() for x in ds_raw.split(",") if x.strip()}
    batch_process_pipeline(
        _PROJECT_ROOT / "metadata.csv",
        datasets=datasets,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
