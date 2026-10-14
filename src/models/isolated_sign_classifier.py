"""Isolated-sign SlowFast classifier (VSL400 word-level; backbone frozen, head trained)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rgb_backbone import (
    SLOWFAST_FEAT_DIM,
    WINDOW_SIZE,
    SlowFastFeatureExtractor,
    _read_video_frames,
    _window_tensors,
)

DEFAULT_CKPT = Path("data/processed/vsl400_classifier.pth")
DEFAULT_LABELS = Path("data/processed/vsl400_labels.json")
LEGACY_CKPT = Path("data/processed/isolated_sign_classifier.pth")
LEGACY_LABELS = Path("data/processed/isolated_sign_labels.json")


class IsolatedSignHead(nn.Module):
    """2304 → bottleneck(256) → logits. Bottleneck doubles as SlowFast RGB proj."""

    def __init__(self, num_classes: int, bottleneck: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(SLOWFAST_FEAT_DIM, bottleneck)
        self.cls = nn.Sequential(
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(bottleneck, num_classes),
        )

    def forward(self, feat_2304: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(feat_2304)
        return self.cls(z), z


class IsolatedSignClassifier:
    """Inference wrapper: video → top-k gloss / Vietnamese labels."""

    def __init__(
        self,
        ckpt_path: Path | None = None,
        labels_path: Path | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        ckpt_path = Path(ckpt_path) if ckpt_path else (
            DEFAULT_CKPT if DEFAULT_CKPT.is_file() else LEGACY_CKPT
        )
        labels_path = Path(labels_path) if labels_path else (
            DEFAULT_LABELS if DEFAULT_LABELS.is_file() else LEGACY_LABELS
        )
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        self.id_to_gloss: list[str] = labels["id_to_gloss"]
        self.id_to_vietnamese: list[str] = labels["id_to_vietnamese"]
        self.gloss_to_id: dict[str, int] = labels["gloss_to_id"]
        self.num_classes = len(self.id_to_gloss)

        payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.head = IsolatedSignHead(self.num_classes).to(self.device)
        self.head.load_state_dict(payload["head"])
        self.head.eval()

        self.extractor = SlowFastFeatureExtractor(device=self.device)
        # Keep Kinetics backbone; use classification-trained proj for 256-d if present
        if "slowfast_proj" in payload:
            self.extractor.proj.load_state_dict(payload["slowfast_proj"])
        self.extractor.eval()

    @torch.no_grad()
    def embed_video(self, video_path: Path, max_windows: int = 6) -> torch.Tensor:
        frames = _read_video_frames(video_path)
        n = frames.shape[0]
        starts = list(range(0, max(n - WINDOW_SIZE + 1, 1), max(WINDOW_SIZE // 2, 8)))
        if not starts or starts[-1] + WINDOW_SIZE < n:
            starts.append(max(0, n - WINDOW_SIZE))
        if len(starts) > max_windows:
            picks = np.linspace(0, len(starts) - 1, max_windows).astype(int)
            starts = [starts[int(i)] for i in picks]
        feats = []
        for start in starts:
            slow, fast = _window_tensors(frames, start, WINDOW_SIZE)
            slow = slow.to(self.device)
            fast = fast.to(self.device)
            feat = self.extractor.backbone([slow, fast])
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            feats.append(feat.flatten(1).float().squeeze(0))
        return torch.stack(feats, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict(
        self,
        video_path: Path,
        *,
        top_k: int = 5,
    ) -> list[dict]:
        feat = self.embed_video(video_path)
        logits, _ = self.head(feat.unsqueeze(0))
        prob = F.softmax(logits, dim=-1).squeeze(0)
        k = min(top_k, self.num_classes)
        vals, idxs = torch.topk(prob, k=k)
        out = []
        for score, i in zip(vals.tolist(), idxs.tolist()):
            out.append(
                {
                    "gloss": self.id_to_gloss[int(i)],
                    "vietnamese": self.id_to_vietnamese[int(i)],
                    "probability": float(score),
                    "class_id": int(i),
                }
            )
        return out

    @torch.no_grad()
    def predict_windows(
        self,
        video_path: Path,
        *,
        stride: int = 16,
        top_k: int = 1,
        conf_min: float = 0.15,
    ) -> list[dict]:
        """Sliding-window gloss sequence for continuous clips."""
        frames = _read_video_frames(video_path)
        n = frames.shape[0]
        starts = list(range(0, max(n - WINDOW_SIZE + 1, 1), stride))
        if not starts:
            starts = [0]
        seq: list[dict] = []
        prev = None
        for start in starts:
            slow, fast = _window_tensors(frames, start, WINDOW_SIZE)
            slow = slow.to(self.device)
            fast = fast.to(self.device)
            feat = self.extractor.backbone([slow, fast])
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            vec = feat.flatten(1).float()
            logits, _ = self.head(vec)
            prob = F.softmax(logits, dim=-1).squeeze(0)
            score, idx = torch.max(prob, dim=0)
            if float(score) < conf_min:
                continue
            gloss = self.id_to_gloss[int(idx)]
            if gloss == prev:
                # keep higher confidence
                if seq and float(score) > seq[-1]["probability"]:
                    seq[-1]["probability"] = float(score)
                continue
            prev = gloss
            seq.append(
                {
                    "gloss": gloss,
                    "vietnamese": self.id_to_vietnamese[int(idx)],
                    "probability": float(score),
                    "window_start": int(start),
                }
            )
        if top_k == 1:
            return seq
        return seq
