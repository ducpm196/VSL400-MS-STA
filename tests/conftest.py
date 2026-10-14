"""Shared pytest helpers — hermetic CI without local checkpoints/features."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]

# The public repository omits VSL400 metadata, sealed evidence and thesis files
# (VSL400 Data Usage Agreement). Tests listed in private_assets_tests.txt need them
# and are skipped only when they are absent.
_PRIVATE_ASSETS = ("metadata.csv", "data/processed", "Docs/results", "Docs/thesis")
_HAS_PRIVATE_ASSETS = all((_ROOT / rel).exists() for rel in _PRIVATE_ASSETS)
_PRIVATE_TESTS_FILE = Path(__file__).with_name("private_assets_tests.txt")

# Fails at import time without the evidence tree, so it cannot be skipped per test.
collect_ignore = [] if _HAS_PRIVATE_ASSETS else ["test_e1_trainer_enforcement.py"]


def _private_test_ids() -> set[str]:
    lines = _PRIVATE_TESTS_FILE.read_text(encoding="utf-8").splitlines()
    return {f"tests/{line.strip()}" for line in lines if line.strip() and not line.startswith("#")}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_artifacts: needs local checkpoints/features/E2 rows (skipped on clean checkout)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _HAS_PRIVATE_ASSETS:
        return
    private_ids = _private_test_ids()
    skip = pytest.mark.skip(reason="needs VSL400 metadata, sealed evidence or thesis files (not in the public repository)")
    for item in items:
        if item.nodeid.replace("\\", "/") in private_ids:
            item.add_marker(skip)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def ms_sta_tmp_checkpoint(tmp_path: Path):
    """Build a tiny valid B4 checkpoint payload for load-path unit tests."""
    from src.models.ms_sta_wordlevel import build_ms_sta

    model = build_ms_sta("b4", num_classes=400)
    payload = {
        "model": model.state_dict(),
        "variant": "b4",
        "num_classes": 400,
        "e1_selected": True,
        "evaluation_locked": True,
        "defense_ready": False,
        "e2_pass": True,
    }
    path = tmp_path / "b4_seed42.pth"
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, payload


def artifact_or_skip(rel: str, reason: str | None = None) -> Path:
    path = _ROOT / rel
    if not path.is_file():
        pytest.skip(reason or f"artifact missing: {rel}")
    return path
