"""Initialize workspace directory tree for VSL400 MS-STA packaging."""

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "data/raw/VSL-Cropped",
    "data/processed",
    "data/features/isolated_sf2304",
    "data/features/local_skel",
    "logs",
    "src/data_prep",
    "src/models",
    "src/training",
    "src/deployment",
    "tests",
]

REQUIRED_FILES = [
    "requirements.txt",
    "metadata.csv",
    ".env.example",
    "src/sanity_check.py",
    "src/load_env.py",
    "src/data_prep/build_metadata.py",
    "src/data_prep/extract_and_smooth.py",
    "src/models/ms_sta_wordlevel.py",
    "src/models/isolated_sign_classifier.py",
    "src/training/train_vsl400_wordlevel.py",
    "src/deployment/backend_api.py",
    "src/deployment/app.py",
    "tests/test_math.py",
]


def main() -> None:
    print("[*] Khởi tạo cấu trúc workspace (VSL400 MS-STA)...")
    for rel in DIRS:
        path = PROJECT_ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        print(f"    [OK] {rel}/")

    missing = [f for f in REQUIRED_FILES if not (PROJECT_ROOT / f).exists()]
    if missing:
        print("\n[!] Thiếu file bắt buộc:")
        for f in missing:
            print(f"    - {f}")
        raise SystemExit(1)
    print("\n[OK] Toàn bộ file mã nguồn bắt buộc đã có mặt.")


if __name__ == "__main__":
    main()
