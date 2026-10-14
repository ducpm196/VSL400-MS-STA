"""Shared sealed RGB/Skeleton input evidence for the defense demo pages.

External caches (rgb_token_v3 + skeleton_v3) are identical for Unified Attention
and Optimized inference. Stabilized skeleton rendering is display-only and never
mutates model tensors.

Streamlit render helpers live in `input_evidence_ui` so hermetic CI can import
this module without streamlit installed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.deployment.skeleton_visualization import skeleton_summary

_ROOT = Path(__file__).resolve().parents[2]
_E4_MANIFEST = _ROOT / "Docs/results/ms_sta_v6_defense_go/E4_INPUT_DEMO_MANIFEST.json"
_FORBIDDEN_TEST_SIGNERS = frozenset(
    {f"signer_{i}" for i in (24, 25, 26, 27, 28)}
)
_RGB_REGIONS = ("global", "left_hand", "right_hand", "face")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_active_defense_registry(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    pointer_path = root / "data/processed/ms_sta_registry_active.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8-sig"))
    registry_path = root / str(pointer["registry_path"])
    raw = registry_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != str(pointer.get("registry_sha256") or ""):
        raise ValueError("active_registry_sha_mismatch")
    registry = json.loads(raw.decode("utf-8-sig"))
    schema = registry.get("schema")
    if schema == "ms_sta_registry_v6_defense_go_v1":
        if (
            registry.get("defense_ready") is not True
            or registry.get("active_variant") != "b4_v6_ctr"
            or registry.get("e4_ready") is not False
        ):
            raise ValueError("active_registry_is_not_v6_defense_go")
        return registry
    if schema == "ms_sta_registry_v5_defense_go_v1":
        if (
            registry.get("defense_ready") is not True
            or registry.get("active_variant") != "b4_star_fixed"
        ):
            raise ValueError("active_registry_is_not_v5_defense_go")
        return registry
    raise ValueError("active_registry_schema_unsupported")


def sealed_feature_paths(
    video_id: str, *, root: Path | None = None, registry: dict | None = None
) -> tuple[Path, Path]:
    """Resolve sealed RGB-token v3 and Skeleton v3 caches for a sample."""
    root = root or _ROOT
    registry = registry or load_active_defense_registry(root)
    roots = registry["feature_roots"]
    rgb = root / str(roots["rgb_root_rel"]) / "rgb_token_v3" / f"{video_id}.npz"
    skel = root / str(roots["skeleton_root_rel"]) / "skeleton_v3" / f"{video_id}.npz"
    return rgb, skel


# Backward-compatible alias (historical name referred to feature lineage, not campaign).
v5_feature_paths = sealed_feature_paths


def rgb_token_summary(npz_path: Path) -> dict[str, Any]:
    """Summarize sealed RGB token v3 without mutating tensors."""
    with np.load(npz_path, allow_pickle=False) as data:
        regions = [r for r in _RGB_REGIONS if f"token_{r}" in data.files]
        if "token_global" not in data.files:
            raise ValueError("rgb_token_missing_global")
        tok = np.asarray(data["token_global"])
        windows = int(tok.shape[0])
        feature_dim = int(tok.shape[-1]) if tok.ndim == 2 else None
        missing = {
            region: int(np.asarray(data[f"missing_{region}"], dtype=bool).sum())
            for region in regions
            if f"missing_{region}" in data.files
        }
        timestamps = None
        if "original_frame_index" in data.files:
            timestamps = [int(x) for x in np.asarray(data["original_frame_index"]).tolist()[:8]]
    return {
        "windows": windows,
        "regions": regions,
        "feature_dim": feature_dim,
        "missing": missing,
        "frame_index_prefix": timestamps,
        "cache_sha256": sha256_file(npz_path),
    }


rgb_feature_summary = rgb_token_summary


def skeleton_tensor_summary(npz_path: Path) -> dict[str, Any]:
    """Summarize Skeleton v3 tensors and masks from the sealed cache only."""
    base = skeleton_summary(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = sorted(data.files)
        missing_prefill = None
        if "missing_mask_prefill" in data.files:
            missing_prefill = int(np.asarray(data["missing_mask_prefill"], dtype=bool).sum())
        validity = np.asarray(data["validity"], dtype=bool)
        valid_count = int(validity.sum())
        total = int(validity.size)
    return {
        **base,
        "arrays": arrays,
        "missing_mask_prefill_true": missing_prefill,
        "valid_joint_count": valid_count,
        "total_joint_slots": total,
        "cache_sha256": sha256_file(npz_path),
        "viz_only_note": (
            "Stabilized animation is VIZ-ONLY display; B4/E4 consume sealed coordinates "
            "unchanged."
        ),
    }


def load_e4_input_demo_manifest(
    path: Path | None = None, *, root: Path | None = None
) -> dict[str, Any]:
    root = root or _ROOT
    path = path or _E4_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"missing_e4_input_demo_manifest:{path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_e4_input_demo_manifest(
    manifest: dict[str, Any], *, root: Path | None = None
) -> list[str]:
    """Fail-closed validator. Returns error strings (empty => OK)."""
    root = root or _ROOT
    errors: list[str] = []
    if manifest.get("schema") != "ms_sta_v6_e4_input_demo_manifest_v1":
        errors.append("bad_schema")
    if manifest.get("e4_ready") is not False:
        errors.append("e4_ready_must_be_false")
    if manifest.get("e4_status") != "E4_ACCURACY_NO_GO":
        errors.append("e4_status_mismatch")
    widths = manifest.get("internal_width") or {}
    if widths.get("b4_v6_ctr") != 256 or widths.get("e4_w192") != 192:
        errors.append("internal_width_mismatch")
    for claim in ("grayscale", "joint_drop", "temporal_stride", "token_merge"):
        if claim not in (manifest.get("forbidden_claims") or []):
            errors.append(f"missing_forbidden_claim:{claim}")

    meta_rel = str(manifest.get("metadata_rel") or "")
    split_rel = str(manifest.get("split_manifest_rel") or "")
    meta_path = root / meta_rel
    split_path = root / split_rel
    if not meta_path.is_file():
        errors.append("metadata_missing")
    elif sha256_file(meta_path) != manifest.get("metadata_sha256"):
        errors.append("metadata_sha_mismatch")
    if not split_path.is_file():
        errors.append("split_manifest_missing")
    elif sha256_file(split_path) != manifest.get("split_manifest_sha256"):
        errors.append("split_manifest_sha_mismatch")

    split: dict[str, Any] = {}
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8-sig"))
    val_ids = {str(x) for x in (split.get("val_ids") or [])}
    test_ids = {str(x) for x in (split.get("test_ids") or [])}

    samples = manifest.get("samples") or []
    if not samples:
        errors.append("no_samples")
    ids: list[str] = []
    for sample in samples:
        sid = str(sample.get("sample_id") or "")
        ids.append(sid)
        if sample.get("split") != "validation":
            errors.append(f"sample_not_validation:{sid}")
        if sid and val_ids and sid not in val_ids:
            errors.append(f"sample_not_in_val_ids:{sid}")
        if sid and sid in test_ids:
            errors.append(f"sample_in_test_ids:{sid}")
        signer = str(sample.get("signer_id") or "")
        if signer in _FORBIDDEN_TEST_SIGNERS:
            errors.append(f"forbidden_test_signer:{sid}:{signer}")
        for key, sha_key in (
            ("video_path_rel", "video_sha256"),
            ("rgb_cache_rel", "rgb_cache_sha256"),
            ("skeleton_cache_rel", "skeleton_cache_sha256"),
        ):
            rel = str(sample.get(key) or "")
            path = root / rel
            if not path.is_file():
                errors.append(f"missing_path:{sid}:{rel}")
                continue
            if sha256_file(path) != sample.get(sha_key):
                errors.append(f"sha_mismatch:{sid}:{key}")
        schema = sample.get("feature_schema") or {}
        if schema.get("rgb_feature_dim") != 2304:
            errors.append(f"rgb_dim:{sid}")
        if schema.get("skeleton_joints") != 75:
            errors.append(f"skel_joints:{sid}")
        if schema.get("rgb_token") != "rgb_token_v3" or schema.get("skeleton") != "skeleton_v3":
            errors.append(f"schema_name:{sid}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate_sample_ids")
    return errors


def resolve_manifest_sample(
    sample_id: str, *, root: Path | None = None, manifest: dict | None = None
) -> dict[str, Any]:
    root = root or _ROOT
    manifest = manifest or load_e4_input_demo_manifest(root=root)
    errors = validate_e4_input_demo_manifest(manifest, root=root)
    if errors:
        raise ValueError("manifest_invalid:" + ",".join(errors[:8]))
    for sample in manifest.get("samples") or []:
        if str(sample.get("sample_id")) == str(sample_id):
            return sample
    raise KeyError(f"sample_not_in_manifest:{sample_id}")


def external_cache_identity(sample: dict[str, Any]) -> dict[str, str]:
    """Hashes that must match between B4 and E4 for the same sample_id."""
    return {
        "sample_id": str(sample["sample_id"]),
        "rgb_cache_sha256": str(sample["rgb_cache_sha256"]),
        "skeleton_cache_sha256": str(sample["skeleton_cache_sha256"]),
        "video_sha256": str(sample["video_sha256"]),
    }


def ffprobe_video_codec(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    name = (proc.stdout or "").strip().splitlines()
    return name[0].strip().lower() if name else None


def make_browser_playable_bytes(src: Path, *, root: Path | None = None) -> tuple[bytes, str]:
    """Return (mp4_bytes, note). VSL400 raw is often mpeg4 — HTML5 needs H.264."""
    root = root or _ROOT
    codec = ffprobe_video_codec(src) or ""
    if codec in {"h264", "avc1", "avc"}:
        return src.read_bytes(), f"codec={codec}"
    preview_dir = root / "logs" / "demo_preview_h264"
    preview_dir.mkdir(parents=True, exist_ok=True)
    mtime_ns = src.stat().st_mtime_ns
    out = preview_dir / f"{src.stem}_{mtime_ns}.mp4"
    if not (out.is_file() and out.stat().st_size > 1024):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-an",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not out.is_file():
            err = (proc.stderr or proc.stdout or "ffmpeg failed")[-500:]
            raise RuntimeError(f"H.264 preview failed: {err}")
    return out.read_bytes(), f"source={codec or 'unknown'} → H.264 preview (HTML5)"
