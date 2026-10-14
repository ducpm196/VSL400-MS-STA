"""SlowFast-R50 RGB feature extractor for offline global_rgb caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn

RGB_DIM = 256
SLOWFAST_FEAT_DIM = 2304
WINDOW_SIZE = 32
WINDOW_STRIDE = 8
INPUT_SIZE = 224
KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)
MANIFEST_NAME = "_manifest.json"


def align_time_axis(features: np.ndarray, target_len: int) -> np.ndarray:
    if features.shape[0] == target_len:
        return features
    if target_len < 1:
        raise ValueError("target_len phải >= 1")
    indices = np.linspace(0, features.shape[0] - 1, target_len).astype(int)
    return features[indices]


class SlowFastFeatureExtractor(nn.Module):
    """Frozen SlowFast-R50 → Linear(2304→256) per temporal window.

    Backbone weights stay frozen (Kinetics pretrained). Only the linear proj /
    classification head are trained for VSL400 word-level recognition — this is
    not end-to-end SlowFast fine-tuning / domain adaptation of the full network.
    """

    def __init__(self, device: torch.device | str = "cuda") -> None:
        super().__init__()
        from pytorchvideo.models.hub import slowfast_r50

        self.device = torch.device(device)
        backbone = slowfast_r50(pretrained=True).eval()
        self.backbone = nn.Sequential(*list(backbone.blocks.children())[:-1]).eval()
        self.proj = nn.Linear(SLOWFAST_FEAT_DIM, RGB_DIM)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.to(self.device)

    @torch.no_grad()
    def forward_window(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        """slow/fast: (1, 3, T_slow/T_fast, 224, 224) → (256,)"""
        feat = self.backbone([slow, fast])
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        vec = feat.flatten(1)  # (1, 2304)
        return self.proj(vec).squeeze(0)

    def load_proj(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.proj.load_state_dict(state)

    def save_proj(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.proj.state_dict(), path)


def _read_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(resized.astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        raise ValueError(f"Video rỗng: {video_path}")
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def _normalize(frames: np.ndarray) -> np.ndarray:
    mean = np.array(KINETICS_MEAN, dtype=np.float32)
    std = np.array(KINETICS_STD, dtype=np.float32)
    return (frames - mean) / std


def _window_tensors(frames: np.ndarray, start: int, win: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build SlowFast slow/fast tensors for one window."""
    clip = frames[start : start + win]
    if clip.shape[0] < win:
        pad = np.repeat(clip[-1:], win - clip.shape[0], axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    clip = _normalize(clip)
    # (T, H, W, C) → (C, T, H, W)
    fast = torch.from_numpy(clip.transpose(3, 0, 1, 2)).unsqueeze(0)
    slow_idx = np.linspace(0, win - 1, 8).astype(int)
    slow_clip = clip[slow_idx]
    slow = torch.from_numpy(slow_clip.transpose(3, 0, 1, 2)).unsqueeze(0)
    return slow, fast


def extract_slowfast_rgb_features(
    video_path: Path,
    extractor: SlowFastFeatureExtractor,
    *,
    target_len: int | None = None,
    window_size: int = WINDOW_SIZE,
    window_stride: int = WINDOW_STRIDE,
    use_amp: bool = True,
) -> np.ndarray:
    """Extract (T_win, 256) then optionally align to skeleton T."""
    frames = _read_video_frames(video_path)
    n = frames.shape[0]
    if n < 1:
        raise ValueError(f"Video rỗng: {video_path}")

    starts = list(range(0, max(n - window_size + 1, 1), window_stride))
    if not starts or starts[-1] + window_size < n:
        starts.append(max(0, n - window_size))

    feats: list[np.ndarray] = []
    device = extractor.device
    for start in starts:
        slow, fast = _window_tensors(frames, start, window_size)
        slow = slow.to(device, non_blocking=True)
        fast = fast.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            vec = extractor.forward_window(slow, fast)
        feats.append(vec.float().cpu().numpy())

    out = np.stack(feats, axis=0).astype(np.float32)
    if target_len is not None:
        out = align_time_axis(out, int(target_len))
    return out


def write_rgb_manifest(cache_dir: Path, *, backend: str = "slowfast_r50") -> None:
    manifest = {
        "backend": backend,
        "rgb_dim": RGB_DIM,
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_STRIDE,
        "input_size": INPUT_SIZE,
        "slowfast_feat_dim": SLOWFAST_FEAT_DIM,
    }
    (cache_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_rgb_manifest(cache_dir: Path) -> dict | None:
    path = cache_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


BackendName = Literal["projection", "slowfast_r50"]
