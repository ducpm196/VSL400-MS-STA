"""Adversarial R5 meta-contract remediation tests for region_validity agreement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    assert_meta_tensor_agreement,
)


def test_executable_paths_include_r5() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_scan_recovery_r5.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15
    assert (
        "scripts/diagnose_ms_sta_v5_d0_v15_rgb_region_validity.py"
        in FULL_CORPUS_EXECUTABLE_PATHS_V15
    )


def _rv(*, all_missing: bool, partial_missing: bool, n_valid: int, n_slices: int) -> dict:
    return {
        "n_slices": n_slices,
        "n_valid": n_valid,
        "valid_fraction": (float(n_valid) / float(n_slices)) if n_slices else 0.0,
        "all_missing": all_missing,
        "partial_missing": partial_missing,
        "slow": {
            "n_slices": n_slices,
            "n_valid": n_valid,
            "valid_fraction": (float(n_valid) / float(n_slices)) if n_slices else 0.0,
            "all_missing": all_missing,
            "partial_missing": partial_missing,
        },
        "fast": {
            "n_slices": 0,
            "n_valid": 0,
            "valid_fraction": 0.0,
            "all_missing": True,
            "partial_missing": False,
        },
    }


def _plant(tmp_path: Path, *, windows: list[dict], miss: dict[str, list[bool]], part: dict[str, list[bool]]) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rgb = tmp_path / "sample.npz"
    payload = {}
    for r in ("face", "left_hand", "right_hand"):
        payload[f"missing_{r}"] = np.asarray(miss[r], dtype=bool)
        payload[f"partial_{r}"] = np.asarray(part[r], dtype=bool)
    np.savez(rgb, **payload)
    meta = {"sample_id": "000001", "windows": windows}
    return rgb, meta


def test_absent_legacy_mask_with_valid_region_validity_passes(tmp_path: Path) -> None:
    """ABSENT_LEGACY_MASK_WITH_VALID_REGION_VALIDITY_PASSES must be true."""
    windows = [
        {
            "region_validity": {
                "face": _rv(all_missing=True, partial_missing=False, n_valid=0, n_slices=40),
                "left_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "right_hand": _rv(all_missing=False, partial_missing=True, n_valid=20, n_slices=40),
            }
            # intentionally no missing_region_mask
        }
    ]
    miss = {
        "face": [True],
        "left_hand": [False],
        "right_hand": [False],
    }
    part = {
        "face": [False],
        "left_hand": [False],
        "right_hand": [True],
    }
    rgb, meta = _plant(tmp_path, windows=windows, miss=miss, part=part)
    assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")
    print("ABSENT_LEGACY_MASK_WITH_VALID_REGION_VALIDITY_PASSES=true")


def test_region_validity_missing_mismatch_accepted_is_false(tmp_path: Path) -> None:
    """REGION_VALIDITY_MISSING_MISMATCH_ACCEPTED must be false."""
    windows = [
        {
            "region_validity": {
                "face": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "left_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "right_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
            }
        }
    ]
    miss = {r: [True] for r in ("face", "left_hand", "right_hand")}  # tensor says missing
    part = {r: [False] for r in ("face", "left_hand", "right_hand")}
    rgb, meta = _plant(tmp_path, windows=windows, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="meta_tensor_missing_mismatch"):
        assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")
    print("REGION_VALIDITY_MISSING_MISMATCH_ACCEPTED=false")


def test_region_validity_partial_mismatch_accepted_is_false(tmp_path: Path) -> None:
    """REGION_VALIDITY_PARTIAL_MISMATCH_ACCEPTED must be false."""
    windows = [
        {
            "region_validity": {
                "face": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "left_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "right_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
            }
        }
    ]
    miss = {r: [False] for r in ("face", "left_hand", "right_hand")}
    part = {r: [True] for r in ("face", "left_hand", "right_hand")}  # tensor says partial
    rgb, meta = _plant(tmp_path, windows=windows, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="meta_tensor_partial_mismatch"):
        assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")
    print("REGION_VALIDITY_PARTIAL_MISMATCH_ACCEPTED=false")


def test_malformed_region_validity_accepted_is_false(tmp_path: Path) -> None:
    """MALFORMED_REGION_VALIDITY_ACCEPTED must be false."""
    miss = {r: [False] for r in ("face", "left_hand", "right_hand")}
    part = {r: [False] for r in ("face", "left_hand", "right_hand")}

    # Missing region_validity entirely.
    rgb, meta = _plant(tmp_path, windows=[{}], miss=miss, part=part)
    with pytest.raises(RuntimeError, match="meta_region_validity_missing"):
        assert_meta_tensor_agreement(rgb_path=rgb, meta=meta, sample_id="000001")

    # Contradictory counts vs flags.
    bad = [
        {
            "region_validity": {
                "face": _rv(all_missing=True, partial_missing=False, n_valid=5, n_slices=40),
                "left_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "right_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
            }
        }
    ]
    rgb2, meta2 = _plant(tmp_path / "b", windows=bad, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="all_missing_vs_counts|counts_inconsistent"):
        assert_meta_tensor_agreement(rgb_path=rgb2, meta=meta2, sample_id="000001")

    # Legacy mask present but disagrees.
    legacy = [
        {
            "missing_region_mask": {"face": True, "left_hand": False, "right_hand": False},
            "region_validity": {
                "face": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "left_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
                "right_hand": _rv(all_missing=False, partial_missing=False, n_valid=40, n_slices=40),
            },
        }
    ]
    rgb3, meta3 = _plant(tmp_path / "c", windows=legacy, miss=miss, part=part)
    with pytest.raises(RuntimeError, match="legacy_missing_region_mask_mismatch"):
        assert_meta_tensor_agreement(rgb_path=rgb3, meta=meta3, sample_id="000001")
    print("MALFORMED_REGION_VALIDITY_ACCEPTED=false")
