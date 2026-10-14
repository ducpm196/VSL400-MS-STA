"""Import boundary: extract_and_smooth must load without MediaPipe installed."""

from __future__ import annotations

import importlib
import sys


def test_extract_and_smooth_imports_without_mediapipe() -> None:
    sys.modules.pop("mediapipe", None)
    sys.modules.pop("src.data_prep.extract_and_smooth", None)
    mod = importlib.import_module("src.data_prep.extract_and_smooth")
    assert hasattr(mod, "SignLanguageDataEngine")
    assert "mediapipe" not in sys.modules


def test_skel_postprocess_importable() -> None:
    mod = importlib.import_module("src.data_prep.skel_postprocess")
    assert callable(mod.postprocess_sequence)
