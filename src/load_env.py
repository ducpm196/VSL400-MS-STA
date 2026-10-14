import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    """Load tokens from project .env into process environment."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)


def sync_kaggle_credentials() -> bool:
    """Write KAGGLE_API_TOKEN to ~/.kaggle/access_token for Kaggle CLI."""
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        return False

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    access_token_path = kaggle_dir / "access_token"
    access_token_path.write_text(token, encoding="ascii")
    return True


def ensure_credentials() -> None:
    """Load .env and sync external credential stores."""
    load_project_env()
    sync_kaggle_credentials()
