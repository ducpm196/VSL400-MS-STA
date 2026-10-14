"""Compose Vietnamese text from predicted VSL gloss sequences via Gemini.

Experimental heuristic only — not a certified continuous translator. Always
requires human confirmation; fail-closed when Gemini errors.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.deployment.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    _extract_response_text,
    _gemini_client,
    gemini_api_key,
)


def compose_vietnamese_from_glosses(
    glosses: list[str],
    *,
    vietnamese_words: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """LLM post-process: gloss tokens → one natural Vietnamese sentence.

    Does not invent news content beyond the gloss inventory; asks model to only
    use provided signs. Always requires human confirmation before end-user delivery.
    """
    if not gemini_api_key():
        return {
            "ok": False,
            "error": "GEMINI_API_KEY missing",
            "suggestion_vietnamese_text": None,
        }
    glosses = [g.strip() for g in glosses if g and str(g).strip()]
    if not glosses:
        return {"ok": False, "error": "empty gloss sequence", "suggestion_vietnamese_text": None}

    words = vietnamese_words or glosses
    model_name = (model or os.getenv("GEMINI_ALIGN_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    gloss_line = " → ".join(glosses)
    word_line = " → ".join(words)
    prompt = (
        "Bạn là trợ lý biên dịch ngôn ngữ ký hiệu Việt Nam (VSL) sang tiếng Việt.\n"
        "Cho chuỗi gloss (ký hiệu) đã nhận dạng từ video:\n"
        f"GLOSS: {gloss_line}\n"
        f"TỪ VIỆT GỢI Ý: {word_line}\n\n"
        "Viết ĐÚNG MỘT câu tiếng Việt tự nhiên, ngắn gọn, chỉ dựa trên các ký hiệu trên.\n"
        "Không thêm địa danh, số liệu, hay sự kiện tin tức không có trong gloss.\n"
        "Nếu chuỗi quá ít/không rõ, trả về câu ngắn nhất có thể từ các từ đã cho.\n"
        "Chỉ trả về câu tiếng Việt, không giải thích."
    )
    try:
        from google.genai import types  # type: ignore[import-untyped]

        response = _gemini_client().models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=128,
            ),
        )
        text = _extract_response_text(response)
        text = re.sub(r"^['\"\s]+|['\"\s]+$", "", text.strip())
        # single line
        text = re.split(r"[\n\r]+", text)[0].strip()
        return {
            "ok": True,
            "suggestion_vietnamese_text": text,
            "gloss_sequence": glosses,
            "model": model_name,
            "requires_human_confirmation": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "suggestion_vietnamese_text": None,
            "gloss_sequence": glosses,
        }
