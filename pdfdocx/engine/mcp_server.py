#!/usr/bin/env python3
"""MCP server for the verified PDF→DOCX layout engine (requires ``pip install mcp``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_KIT = Path(__file__).resolve().parent.parent
if str(_KIT) not in sys.path:
    sys.path.insert(0, str(_KIT))

from engine.contract import (  # noqa: E402
    compare_docx_pdf as run_compare,
    convert_pdf_to_docx as run_convert,
    engine_contract as run_contract,
    inspect_pdf as run_inspect,
    verify_docx as run_verify,
)
from engine.runtime import runtime_status as run_runtime  # noqa: E402


def _stdio_fallback() -> int:
    print(
        json.dumps(
            {
                "error": "mcp package not installed",
                "hint": "python pdfdocx/install.py",
            }
        ),
        file=sys.stderr,
    )
    return 1


def build_server():
    """Register the pdfdocx tools. Tests call this without opening stdio."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("pdfdocx")

    @mcp.tool()
    def engine_contract() -> dict:
        """Return guarantees, limits, fonts, calibrated margins, and verify profiles.

        Read this before converting. The layout engine is the verified thesis
        converter; this tool does not redefine it.
        """
        return run_contract()

    @mcp.tool()
    def runtime_status() -> dict:
        """Check this machine: pinned packages, Windows fonts, engine files, and Word/WPS registration.

        Does not convert a PDF and does not launch Word. ok is false when a pin,
        the engine file, or a required font is missing. compare_ready is separate
        because visual compare needs Word or WPS and conversion does not.
        """
        return run_runtime()

    @mcp.tool()
    def inspect_pdf(pdf_path: str) -> dict:
        """Measure page size and whether the text block matches the calibrated DATN margins.

        geometry_matches_calibration false means convert_pdf_to_docx will refuse
        unless allow_uncalibrated is true.
        """
        return run_inspect(pdf_path)

    @mcp.tool()
    def convert_pdf_to_docx(pdf_path: str, docx_path: str = "", allow_uncalibrated: bool = False) -> dict:
        """Convert a PDF to DOCX with the verified one-page-per-page layout engine.

        Leave docx_path empty to write <pdf stem>.docx next to the PDF.
        Refuses when the text block does not match the calibrated margins, and
        writes nothing, unless allow_uncalibrated is true.
        """
        return run_convert(pdf_path, docx_path, allow_uncalibrated)

    @mcp.tool()
    def verify_docx(docx_path: str, profile: str = "layout") -> dict:
        """Verify a DOCX. Default profile=layout. profile=datn is the 17 thesis checks."""
        return run_verify(docx_path, profile)

    @mcp.tool()
    def compare_docx_pdf(pdf_path: str, docx_path: str, pages: str = "1", work_dir: str = "") -> dict:
        """Render the DOCX through Word/WPS and compare page text with the PDF.

        pages is a comma-separated 1-based list. Writes side-by-side PNGs under work_dir.
        """
        return run_compare(pdf_path, docx_path, pages, work_dir)

    return mcp


def main() -> int:
    try:
        mcp = build_server()
    except ImportError:
        return _stdio_fallback()
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
