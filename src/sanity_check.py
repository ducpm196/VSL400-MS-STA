import os
import shutil
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.load_env import load_project_env

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED_DIRS = [
    "data/raw/VSL-Cropped",
    "data/processed",
    "data/features/isolated_sf2304",
    "src/data_prep",
    "src/models",
    "src/training",
    "src/deployment",
    "tests",
]

REQUIRED_SCRIPTS = [
    "src/data_prep/build_metadata.py",
    "src/models/isolated_sign_classifier.py",
    "src/training/train_vsl400_wordlevel.py",
    "src/deployment/backend_api.py",
    "src/deployment/app.py",
    "src/deployment/gloss_llm.py",
    "scripts/evaluate_vsl400.py",
    "tests/test_math.py",
]


def run_sanity_check():
    load_project_env()
    errors: list[str] = []

    print("=" * 50)
    print("HỆ THỐNG KIỂM TRA KHÔNG GIAN LÀM VIỆC — VSL400")
    print("=" * 50)

    print(f"[*] Phiên bản PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"[*] Trạng thái kích hoạt CUDA GPU: {cuda_available}")
    if cuda_available:
        print(f"    -> Đang sử dụng thiết bị: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "    -> [CẢNH BÁO]: Chưa nhận diện được GPU. "
            "Hãy kiểm tra lại Driver CUDA nếu muốn huấn luyện nhanh."
        )

    try:
        import mediapipe as mp

        mp.solutions.holistic
        print("[*] Thư viện MediaPipe Holistic: SẴN SÀNG (env_extract)")
    except Exception as e:
        print(
            f"[*] MediaPipe: chưa cài ({e}) — OK cho train/demo; "
            "D1 extract cần env_extract + requirements-extract.txt"
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"[*] Công cụ hệ thống FFmpeg: ĐÃ TÌM THẤY tại ({ffmpeg_path})")
    else:
        errors.append("FFmpeg not found")
        print(
            "    -> [LỖI CHÍ MẠNG]: Hệ thống chưa tìm thấy biến môi trường FFmpeg. "
            "Vui lòng cài đặt lại theo Bước 1."
        )

    hf_ok = bool(os.getenv("HF_TOKEN"))
    kg_ok = bool(os.getenv("KAGGLE_API_TOKEN"))
    print(f"[*] HF_TOKEN trong .env: {'ĐÃ THIẾT LẬP' if hf_ok else 'CHƯA CÓ (optional)'}")
    print(f"[*] KAGGLE_API_TOKEN trong .env: {'ĐÃ THIẾT LẬP' if kg_ok else 'CHƯA CÓ (optional)'}")

    print("[*] Kiểm tra cấu trúc workspace (VSL400 primary):")
    for rel in REQUIRED_DIRS:
        path = _PROJECT_ROOT / rel
        if path.exists():
            print(f"    [OK] {rel}/")
        else:
            errors.append(f"Missing dir: {rel}")
            print(f"    [LỖI] Thiếu thư mục: {rel}/")

    for rel in REQUIRED_SCRIPTS:
        path = _PROJECT_ROOT / rel
        if path.exists():
            print(f"    [OK] {rel}")
        else:
            errors.append(f"Missing file: {rel}")
            print(f"    [LỖI] Thiếu file: {rel}")

    metadata_path = _PROJECT_ROOT / "metadata.csv"
    if metadata_path.exists():
        print(f"[*] metadata.csv: CÓ ({metadata_path.stat().st_size // 1024} KB)")
        try:
            import pandas as pd

            ds = set(pd.read_csv(metadata_path, dtype=str, usecols=["dataset"])["dataset"])
            if "Continuous_News" in ds:
                errors.append("metadata still contains Continuous_News")
                print("    -> [LỖI]: metadata.csv còn Continuous_News — rebuild hoặc lọc lại.")
            elif "VSL400" not in ds:
                errors.append("metadata missing VSL400")
                print("    -> [LỖI]: metadata.csv thiếu VSL400.")
            else:
                print("    -> datasets:", ", ".join(sorted(ds)))
        except Exception as exc:
            errors.append(f"metadata read: {exc}")
            print(f"    -> [LỖI] đọc metadata: {exc}")
    else:
        errors.append("metadata.csv missing")
        print("    -> [CẢNH BÁO]: Chưa có metadata.csv — chạy build_metadata.py sau khi tải dữ liệu.")

    ckpt = _PROJECT_ROOT / "data" / "processed" / "vsl400_classifier.pth"
    labels = _PROJECT_ROOT / "data" / "processed" / "vsl400_labels.json"
    if ckpt.is_file() and labels.is_file():
        print(f"[*] VSL400 classifier: CÓ ({ckpt.name})")
    else:
        print("    -> [CẢNH BÁO]: Chưa có vsl400_classifier.pth — chạy train_vsl400_wordlevel.py")

    print("=" * 50)
    if errors:
        print(f"KẾT LUẬN: Phát hiện {len(errors)} vấn đề cần xử lý.")
        sys.exit(1)
    print("KẾT LUẬN: Workspace OK — sẵn sàng train/eval/demo VSL400 word-level.")
    print("=" * 50)


if __name__ == "__main__":
    run_sanity_check()
