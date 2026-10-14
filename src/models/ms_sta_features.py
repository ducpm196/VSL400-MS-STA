"""Shared Option B feature paths and RGB temporal length contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

# Repo-relative defaults (legacy / Gate3 pooled). Active dirs may be overridden by registry.
RGB_POOLED_REL = Path("data/features/isolated_sf2304")  # B0 Gate3 / pooled only
RGB_TEMPORAL_REL = Path("data/features/isolated_sf_temporal")  # legacy default — prefer versioned
SKEL_REL = Path("data/features/local_skel")  # legacy default — prefer versioned after D1
REGISTRY_REL = Path("data/processed/ms_sta_registry.json")

FIXED_RGB_L = 8
SLOWFAST_FEAT_DIM = 2304
MIN_TEMPORAL_L = 2


def load_ms_sta_registry(root: Path) -> dict:
    path = root / REGISTRY_REL
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel_or_abs(root: Path, value: str | Path | None, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return root / default
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def resolve_skel_dir(root: Path, registry: dict | None = None) -> Path:
    """Active skeleton cache. Prefer registry.active_skel_dir after D1 promote."""
    reg = registry if registry is not None else load_ms_sta_registry(root)
    feats = reg.get("features") if isinstance(reg.get("features"), dict) else {}
    active = reg.get("active_skel_dir") or feats.get("active_skel_dir")
    return _rel_or_abs(root, active, SKEL_REL)


def resolve_temporal_dir(root: Path, registry: dict | None = None) -> Path:
    """Active RGB temporal cache. Prefer registry.active_temporal_dir after promote."""
    reg = registry if registry is not None else load_ms_sta_registry(root)
    feats = reg.get("features") if isinstance(reg.get("features"), dict) else {}
    active = reg.get("active_temporal_dir") or feats.get("active_temporal_dir")
    return _rel_or_abs(root, active, RGB_TEMPORAL_REL)


def resolve_pooled_dir(root: Path, registry: dict | None = None) -> Path:
    reg = registry if registry is not None else load_ms_sta_registry(root)
    feats = reg.get("features") if isinstance(reg.get("features"), dict) else {}
    active = feats.get("pooled_dir")
    return _rel_or_abs(root, active, RGB_POOLED_REL)


def resolve_rgb_path(root: Path, video_id: str, *, temporal: bool) -> Path:
    base = resolve_temporal_dir(root) if temporal else resolve_pooled_dir(root)
    return base / f"{video_id}.npy"


def load_rgb_array(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != SLOWFAST_FEAT_DIM:
            raise ValueError(f"bad pooled rgb {path}: {arr.shape}")
        return arr[None, :]  # (1, 2304)
    if arr.ndim == 2 and arr.shape[-1] == SLOWFAST_FEAT_DIM:
        return arr
    raise ValueError(f"bad rgb rank/shape {path}: {arr.shape}")


def fix_rgb_length(
    arr: np.ndarray, target_l: int = FIXED_RGB_L
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic sample/pad to (target_l, 2304) with validity mask."""
    if arr.ndim != 2 or arr.shape[-1] != SLOWFAST_FEAT_DIM:
        raise ValueError(f"expect (L,2304), got {arr.shape}")
    l = int(arr.shape[0])
    if l < MIN_TEMPORAL_L:
        raise ValueError(
            f"raw temporal L={l} < {MIN_TEMPORAL_L}; refuse last-token pad that fakes L>={MIN_TEMPORAL_L}"
        )
    if l == target_l:
        return arr.astype(np.float32, copy=False), np.ones(target_l, dtype=bool)
    if l > target_l:
        ix = np.linspace(0, l - 1, target_l).astype(int)
        return arr[ix].astype(np.float32, copy=False), np.ones(target_l, dtype=bool)
    out = np.zeros((target_l, SLOWFAST_FEAT_DIM), dtype=np.float32)
    out[:l] = arr
    valid = np.zeros(target_l, dtype=bool)
    valid[:l] = True
    return out, valid


def to_rgb_tensor(
    arr: np.ndarray, *, fixed_l: int | None = FIXED_RGB_L
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if fixed_l is None:
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)), None
    fixed, valid = fix_rgb_length(arr, fixed_l)
    return torch.from_numpy(fixed), torch.from_numpy(valid)


def valid_mask_to_key_padding_mask(valid: torch.Tensor | None) -> torch.Tensor | None:
    if valid is None:
        return None
    return ~valid.bool()
