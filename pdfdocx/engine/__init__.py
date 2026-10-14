"""Public PDF to DOCX contract used by the MCP server and by tests."""

from .contract import (  # noqa: F401
    compare_docx_pdf,
    convert_pdf_to_docx,
    engine_contract,
    inspect_pdf,
    verify_docx,
)
from .runtime import runtime_status  # noqa: F401

__all__ = [
    "compare_docx_pdf",
    "convert_pdf_to_docx",
    "engine_contract",
    "inspect_pdf",
    "runtime_status",
    "verify_docx",
]
