#!/usr/bin/env python3
"""Stage-A Dataset for MS-STA v5 — Skeleton v3 + RGB token v3 NPZ dual-root.

Hard-refuses test-role sample IDs. No silent fallbacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_prep.joint_layout_v3 import N_JOINTS
from src.models.ms_sta_v5 import CHIRALITY_VOCAB, PART_VOCAB, RGB_REGIONS

RoleName = Literal["hpo_train", "tune", "confirm_train", "val"]
FORBIDDEN_ROLES = frozenset({"test"})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split_role_ids(split: dict[str, Any], role: str) -> list[str]:
    if role in FORBIDDEN_ROLES or role == "test":
        raise ValueError(f"test_role_forbidden:{role}")
    key = f"{role}_ids"
    ids = split.get(key)
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"split_role_missing:{key}")
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    out = [str(x) for x in ids]
    leak = sorted(set(out) & test_ids)
    if leak:
        raise ValueError(f"train_val_intersects_test:{leak[:5]}")
    return out


class VSL400MSSTAv5Dataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        *,
        rgb_root: Path,
        skel_root: Path,
        labels: dict[str, int],
        metadata_by_id: dict[str, dict[str, Any]],
        max_rgb_len: int = 8,
        max_skel_len: int = 96,
        forbid_ids: set[str] | None = None,
        require_all: bool = True,
    ) -> None:
        self.rgb_dir = Path(rgb_root) / "rgb_token_v3"
        self.skel_dir = Path(skel_root) / "skeleton_v3"
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"rgb_token_dir_missing:{self.rgb_dir}")
        if not self.skel_dir.is_dir():
            raise FileNotFoundError(f"skel_dir_missing:{self.skel_dir}")
        self.labels = labels
        self.metadata_by_id = metadata_by_id
        self.max_rgb_len = int(max_rgb_len)
        self.max_skel_len = int(max_skel_len)
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
        if not cleaned:
            raise ValueError("dataset_empty_after_feature_filter")
        if require_all and dropped:
            raise ValueError(f"dataset_coverage_gap:{len(dropped)}:{dropped[:5]}")
        self.sample_ids = cleaned

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.sample_ids[idx]
        rgb = np.load(self.rgb_dir / f"{sid}.npz", allow_pickle=False)
        skel = np.load(self.skel_dir / f"{sid}.npz", allow_pickle=False)
        L = int(rgb["token_global"].shape[0])
        if L < 2:
            raise ValueError(f"rgb_temporal_L_lt_2:{sid}:{L}")
        if L > self.max_rgb_len:
            # Deterministic center crop of windows.
            start = (L - self.max_rgb_len) // 2
            sl = slice(start, start + self.max_rgb_len)
        else:
            sl = slice(0, L)
        T = int(skel["coordinates"].shape[0])
        if T > self.max_skel_len:
            start = (T - self.max_skel_len) // 2
            st = slice(start, start + self.max_skel_len)
        else:
            st = slice(0, T)

        meta_row = self.metadata_by_id.get(sid) or {}
        item: dict[str, Any] = {
            "sample_id": sid,
            "label": int(self.labels[sid]),
            "signer": str(meta_row.get("signer_id") or meta_row.get("signer") or ""),
        }
        for r in RGB_REGIONS:
            item[f"token_{r}"] = torch.from_numpy(np.asarray(rgb[f"token_{r}"][sl], dtype=np.float32))
            miss_key = f"missing_{r}"
            if miss_key in rgb.files:
                item[miss_key] = torch.from_numpy(np.asarray(rgb[miss_key][sl], dtype=bool))
            else:
                item[miss_key] = torch.zeros((item[f"token_{r}"].shape[0],), dtype=torch.bool)
        item["coordinates"] = torch.from_numpy(np.asarray(skel["coordinates"][st], dtype=np.float32))
        item["velocity"] = torch.from_numpy(np.asarray(skel["velocity"][st], dtype=np.float32))
        item["validity"] = torch.from_numpy(np.asarray(skel["validity"][st], dtype=np.float32))
        item["rgb_len"] = int(item["token_global"].shape[0])
        item["skel_len"] = int(item["coordinates"].shape[0])
        # Joint identity from NPZ strings if present.
        if "joint_part" in skel.files:
            parts = [PART_VOCAB.get(str(x), 0) for x in skel["joint_part"].tolist()]
        else:
            parts = [0] * N_JOINTS
        if "joint_chirality" in skel.files:
            chirs = [CHIRALITY_VOCAB.get(str(x), 0) for x in skel["joint_chirality"].tolist()]
        else:
            chirs = [0] * N_JOINTS
        item["joint_part_ids"] = torch.tensor(parts, dtype=torch.long)
        item["joint_chirality_ids"] = torch.tensor(chirs, dtype=torch.long)
        return item


def collate_v5(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty_batch")
    max_l = max(int(b["rgb_len"]) for b in batch)
    max_t = max(int(b["skel_len"]) for b in batch)
    bsz = len(batch)
    out: dict[str, Any] = {
        "sample_id": [b["sample_id"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "signer": [b["signer"] for b in batch],
        "rgb_pad_mask": torch.ones(bsz, max_l, dtype=torch.bool),
        "skel_pad_mask": torch.ones(bsz, max_t, dtype=torch.bool),
        "joint_part_ids": batch[0]["joint_part_ids"],
        "joint_chirality_ids": batch[0]["joint_chirality_ids"],
    }
    for r in RGB_REGIONS:
        out[f"token_{r}"] = torch.zeros(bsz, max_l, 2304, dtype=torch.float32)
        out[f"missing_{r}"] = torch.ones(bsz, max_l, dtype=torch.bool)
    out["coordinates"] = torch.zeros(bsz, max_t, N_JOINTS, 3, dtype=torch.float32)
    out["velocity"] = torch.zeros(bsz, max_t, N_JOINTS, 3, dtype=torch.float32)
    out["validity"] = torch.zeros(bsz, max_t, N_JOINTS, dtype=torch.float32)

    for i, b in enumerate(batch):
        l = int(b["rgb_len"])
        t = int(b["skel_len"])
        out["rgb_pad_mask"][i, :l] = False
        out["skel_pad_mask"][i, :t] = False
        for r in RGB_REGIONS:
            out[f"token_{r}"][i, :l] = b[f"token_{r}"]
            out[f"missing_{r}"][i, :l] = b[f"missing_{r}"]
        out["coordinates"][i, :t] = b["coordinates"]
        out["velocity"][i, :t] = b["velocity"]
        out["validity"][i, :t] = b["validity"]
    return out
