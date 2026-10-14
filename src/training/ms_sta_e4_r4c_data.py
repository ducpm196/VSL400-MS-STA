# -*- coding: utf-8 -*-
"""E4-R4C compact-feature dataset (R1 RGB + S_A Skeleton).

Hard-refuses test-role IDs for HPO/confirm loaders. Extraction may cover eligible,
but training roles must stay train/validation-only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.ms_sta_v5_data import FORBIDDEN_ROLES, load_split_role_ids


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class VSL400E4R4CDataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        *,
        feature_root: Path,
        labels: dict[str, int],
        metadata_by_id: dict[str, dict[str, Any]],
        forbid_ids: set[str] | None = None,
        require_all: bool = True,
        rgb_subdir: str = "rgb_R1",
        skel_subdir: str = "skel_S_A",
    ) -> None:
        self.root = Path(feature_root)
        self.rgb_dir = self.root / rgb_subdir
        self.skel_dir = self.root / skel_subdir
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"rgb_dir_missing:{self.rgb_dir}")
        if not self.skel_dir.is_dir():
            raise FileNotFoundError(f"skel_dir_missing:{self.skel_dir}")
        self.labels = labels
        self.metadata_by_id = metadata_by_id
        forbid = set(forbid_ids or ())
        cleaned: list[str] = []
        dropped: list[str] = []
        for sid in sample_ids:
            sid = str(sid)
            if sid in forbid:
                raise ValueError(f"forbidden_sample_id:{sid}")
            if sid not in labels:
                dropped.append(f"nolabel:{sid}")
                continue
            if not (self.rgb_dir / f"{sid}.npz").is_file():
                dropped.append(f"norgb:{sid}")
                continue
            if not (self.skel_dir / f"{sid}.npz").is_file():
                dropped.append(f"noskel:{sid}")
                continue
            cleaned.append(sid)
        if require_all and dropped:
            raise FileNotFoundError(f"missing_features:{dropped[:8]}")
        if not cleaned:
            raise RuntimeError("empty_dataset")
        self.sample_ids = cleaned

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.sample_ids[idx]
        with np.load(self.rgb_dir / f"{sid}.npz", allow_pickle=False) as z:
            rgb = np.asarray(z["rgb_tensor"], dtype=np.float32)
            tvalid = np.asarray(z["temporal_valid"], dtype=bool)
        with np.load(self.skel_dir / f"{sid}.npz", allow_pickle=False) as z:
            coords = np.asarray(z["coordinates"], dtype=np.float32)
            vel = np.asarray(z["velocity"], dtype=np.float32)
            validity = np.asarray(z["validity"], dtype=np.float32)
            spad = np.asarray(z["padding_mask"], dtype=bool) if "padding_mask" in z.files else None

        # Fixed shared T = RGB T_max (R1=32). Never expand past packed RGB length.
        # Crop/pad skeleton into that window; never duplicate frames.
        t = int(rgb.shape[0])
        t_skel = int(coords.shape[0])
        c2 = np.zeros((t, coords.shape[1], 3), dtype=np.float32)
        v2 = np.zeros_like(c2)
        val2 = np.zeros((t, coords.shape[1]), dtype=np.float32)
        pad2 = np.ones((t,), dtype=bool)
        if t_skel > t:
            start = (t_skel - t) // 2
            sl = slice(start, start + t)
            c2[:] = coords[sl]
            v2[:] = vel[sl]
            val2[:] = validity[sl]
            if spad is None:
                pad2[:] = False
            else:
                pad2[:] = spad[sl]
        else:
            n = t_skel
            c2[:n] = coords[:n]
            v2[:n] = vel[:n]
            val2[:n] = validity[:n]
            if spad is None:
                pad2[:n] = False
            else:
                pad2[:n] = spad[:n]
        coords, vel, validity, spad = c2, v2, val2, pad2
        # Preserve all-pad skeleton as zeros + all-True mask (no fabricated joints).
        # Model RGB/base fallback handles TemporalTransformer safely.

        meta = self.metadata_by_id.get(sid, {})
        return {
            "sample_id": sid,
            "label": int(self.labels[sid]),
            "signer": str(meta.get("signer_id") or ""),
            "rgb_tensor": torch.from_numpy(rgb),
            "temporal_valid": torch.from_numpy(tvalid),
            "coordinates": torch.from_numpy(coords),
            "velocity": torch.from_numpy(vel),
            "validity": torch.from_numpy(validity),
            "skel_pad_mask": torch.from_numpy(spad),
        }


def collate_e4_r4c(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "signer": [b["signer"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "rgb_tensor": torch.stack([b["rgb_tensor"] for b in batch], dim=0),
        "temporal_valid": torch.stack([b["temporal_valid"] for b in batch], dim=0),
        "coordinates": torch.stack([b["coordinates"] for b in batch], dim=0),
        "velocity": torch.stack([b["velocity"] for b in batch], dim=0),
        "validity": torch.stack([b["validity"] for b in batch], dim=0),
        "skel_pad_mask": torch.stack([b["skel_pad_mask"] for b in batch], dim=0),
    }


def role_ids_forbid_test(split: dict[str, Any], role: str) -> list[str]:
    if role in FORBIDDEN_ROLES or role == "test":
        raise ValueError(f"test_role_forbidden:{role}")
    return load_split_role_ids(split, role)
