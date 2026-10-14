import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.backbones import build_stgcn_placeholder
from src.models.utils import enforce_reproducibility_seed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class IsolatedSignDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, data_dir: Path, mode: str = "rgb") -> None:
        self.data_dir = Path(data_dir)
        self.mode = mode
        train = metadata_df[metadata_df["split"] == "train"].reset_index(drop=True)
        if mode == "skel":
            skel_dir = self.data_dir / "local_skel"
            mask = train["video_id"].map(lambda vid: (skel_dir / f"{vid}.npy").is_file())
            self.df = train[mask].reset_index(drop=True)
        else:
            self.df = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        video_id = row["video_id"]

        if self.mode == "skel":
            feat_path = self.data_dir / "local_skel" / f"{video_id}.npy"
            if not feat_path.exists():
                raise FileNotFoundError(f"Thiếu skeleton cache: {feat_path}")
            tensor_data = np.load(feat_path).astype(np.float32)
            if tensor_data.ndim != 3 or tensor_data.shape[1:] != (75, 3):
                raise ValueError(
                    f"Skeleton shape không hợp lệ cho {video_id}: {tensor_data.shape} (kỳ vọng T, 75, 3)"
                )
            if tensor_data.shape[0] != 64:
                tensor_data = self._resample_timeline(tensor_data, target_frames=64)
            return torch.tensor(tensor_data), torch.tensor(int(idx % 100))

        raise RuntimeError("Stage-1 chỉ hỗ trợ mode='skel' — không dùng RGB mock.")

    @staticmethod
    def _resample_timeline(data: np.ndarray, target_frames: int = 64) -> np.ndarray:
        indices = np.linspace(0, data.shape[0] - 1, target_frames).astype(int)
        return data[indices]


def configure_backbone_weights(model: nn.Module) -> nn.Module:
    """Freeze early layers; train head only (ResNet layer4/fc or last Linear)."""
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "layer4"):
        for param in model.layer4.parameters():
            param.requires_grad = True
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    # Sequential/MLP mock: no layer4/fc — unfreeze the last Linear head
    if not any(p.requires_grad for p in model.parameters()):
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if linears:
            for param in linears[-1].parameters():
                param.requires_grad = True
    return model


def run_adaptive_training() -> None:
    enforce_reproducibility_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Tiến trình chạy thích ứng miền sử dụng thiết bị tính toán: {device}")

    metadata_df = pd.read_csv(_PROJECT_ROOT / "metadata.csv")
    skel_dataset = IsolatedSignDataset(metadata_df, _PROJECT_ROOT / "data" / "features", mode="skel")
    if len(skel_dataset) == 0:
        raise RuntimeError("Không có mẫu train cho Stage-1 — chạy extract_and_smooth.py trước.")
    print(f"[*] Stage-1 train samples with local_skel: {len(skel_dataset)}")

    train_loader = DataLoader(skel_dataset, batch_size=16, shuffle=True, drop_last=True)

    mock_backbone = build_stgcn_placeholder(num_classes=100).to(device)
    mock_backbone = configure_backbone_weights(mock_backbone)
    trainable = [p for p in mock_backbone.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("Stage-1: không có tham số trainable sau configure_backbone_weights.")

    optimizer = optim.AdamW(trainable, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("[*] Bắt đầu pha huấn luyện thích ứng miền (Stage-1)...")
    mock_backbone.train()
    for epoch in range(1, 3):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1).to(device)
            inputs = torch.mean(inputs, dim=1)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = mock_backbone(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"    -> Epoch [{epoch}/2] Complete. Loss trung bình: {total_loss / len(train_loader):.4f}")

    weight_path = _PROJECT_ROOT / "data" / "processed" / "backbone_stage1.pth"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mock_backbone.state_dict(), weight_path)
    print(f"[OK] Đã xuất trọng số thích ứng miền tại: {weight_path}")


if __name__ == "__main__":
    run_adaptive_training()
