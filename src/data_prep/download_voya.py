import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from huggingface_hub import snapshot_download

from src.load_env import ensure_credentials

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def fetch_voya_dataset():
    ensure_credentials()

    base_dir = Path(__file__).resolve().parents[2]
    target_dir = base_dir / "data" / "raw" / "VOYA_VSL"
    os.makedirs(target_dir, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[!] CẢNH BÁO: Chưa thiết lập HF_TOKEN trong .env — tốc độ tải có thể bị giới hạn.")

    print("=" * 60)
    print("[*] ĐANG TIẾN HÀNH TẢI DATASET VOYA_VSL TỪ HUGGING FACE...")
    print("=" * 60)

    try:
        snapshot_download(
            repo_id="Kateht/VOYA_VSL",
            repo_type="dataset",
            local_dir=str(target_dir),
            ignore_patterns=[".git*", "README.md"],
            token=hf_token,
        )
        print(f"[OK] Đã tải thành công VOYA_VSL về thư mục: {target_dir}")
    except Exception as e:
        print(f"[X] Lỗi trong quá trình tải dữ liệu: {e}")
    print("=" * 60)


if __name__ == "__main__":
    fetch_voya_dataset()
