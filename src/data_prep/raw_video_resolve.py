"""Resolve on-disk video paths for metadata rows (VSL400 / VSL-Cropped)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

# VOYA_VSL Merged/*.npz — not FFmpeg/MediaPipe video inputs.
SKIP_VIDEO_DATASETS = frozenset({"VOYA_VSL"})
VSL400_VIEWS = ("front", "left", "right")


class MetadataRow(Protocol):
    """Minimal row contract: mapping-like access used by resolve paths."""

    def __getitem__(self, key: str) -> Any: ...

    def get(self, key: str, default: Any = None) -> Any: ...


def build_vsl400_index(vsl_root: Path) -> dict[tuple[str, str], Path]:
    """Map (video_id, view) -> mp4 under VSL400/{front,left,right}_view."""
    index: dict[tuple[str, str], Path] = {}
    vsl400 = vsl_root / "raw" / "raw" / "VSL400"
    if not vsl400.is_dir():
        return index
    for view in VSL400_VIEWS:
        tag = f"{view}_view"
        for mp4 in vsl400.rglob(f"{tag}/*.mp4"):
            if tag not in mp4.parts:
                continue
            index[(mp4.stem, view)] = mp4
    return index


def build_vsl_cropped_index(vsl_root: Path) -> dict[str, Path]:
    """Legacy online_sourced + front fallback map video_id -> mp4."""
    index: dict[str, Path] = {}
    if not vsl_root.is_dir():
        return index

    vsl400 = vsl_root / "raw" / "raw" / "VSL400"
    if vsl400.is_dir():
        for mp4 in vsl400.rglob("*.mp4"):
            if "front_view" in mp4.parts:
                index[mp4.stem] = mp4

    online = vsl_root / "raw" / "raw" / "online_sourced"
    if online.is_dir():
        for mp4 in online.rglob("*.mp4"):
            index.setdefault(mp4.stem, mp4)

    return index


class RawVideoResolver:
    """Cached resolver for batch normalize / skeleton extract / word-level train."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.raw_dir = project_root / "data" / "raw"
        self.processed_dir = project_root / "data" / "processed"
        self._vsl_index: dict[str, Path] | None = None
        self._vsl400_index: dict[tuple[str, str], Path] | None = None

    def vsl_index(self) -> dict[str, Path]:
        if self._vsl_index is None:
            self._vsl_index = build_vsl_cropped_index(self.raw_dir / "VSL-Cropped")
        return self._vsl_index

    def vsl400_index(self) -> dict[tuple[str, str], Path]:
        if self._vsl400_index is None:
            self._vsl400_index = build_vsl400_index(self.raw_dir / "VSL-Cropped")
        return self._vsl400_index

    def resolve(self, row: Mapping[str, Any] | MetadataRow, view: str | None = None) -> Path | None:
        dataset = str(row["dataset"])
        video_id = str(row["video_id"])
        signer_id = str(row.get("signer_id", "") or "")
        video_name = f"{video_id}.mp4"
        view_name = (view or str(row.get("view", "") or "front")).strip().lower()
        if view_name.endswith("_view"):
            view_name = view_name.replace("_view", "")

        processed = self.processed_dir / dataset / signer_id / video_name
        if view_name == "front" and processed.is_file():
            return processed

        if dataset in SKIP_VIDEO_DATASETS:
            return None

        if dataset == "VSL400":
            # Exact view only — never fall back left/right → front (would duplicate front in fusion).
            return self.vsl400_index().get((video_id, view_name))

        if dataset == "VSL-Cropped":
            return self.vsl_index().get(video_id)

        flat = self.raw_dir / dataset / signer_id / video_name
        if flat.is_file():
            return flat
        return None

    def processed_output_path(self, row: Mapping[str, Any] | MetadataRow) -> Path:
        return (
            self.processed_dir
            / str(row["dataset"])
            / str(row["signer_id"])
            / f"{row['video_id']}.mp4"
        )
