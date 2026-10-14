"""Minimal Gemini client helpers for optional gloss→text bridge (not continuous SLT)."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

_GEMINI_CLIENT: Any | None = None


def gemini_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def _gemini_client():
    global _GEMINI_CLIENT
    from google import genai

    key = gemini_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=key)
    return _GEMINI_CLIENT


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text and str(text).strip():
        return str(text).strip()
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        if content is not None:
            parts = getattr(content, "parts", None) or []
            joined = "".join(getattr(p, "text", "") or "" for p in parts)
            if joined.strip():
                return joined.strip()
    raise ValueError("Empty Gemini response")
