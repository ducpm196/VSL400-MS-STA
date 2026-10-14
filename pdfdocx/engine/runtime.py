# -*- coding: utf-8 -*-
"""Machine check for the pdfdocx kit: pins, fonts, engine files, Word/WPS."""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path

from .contract import compare_backend, converter_module, repo_root

_FONT_FILES = {
    "Times New Roman": "times.ttf",
    "Cambria Math": "cambria.ttc",
    "Courier New": "cour.ttf",
}


def _requirements() -> list[tuple[str, str, str]]:
    """Return (distribution, version, marker) from pdfdocx/requirements.txt."""
    path = Path(__file__).resolve().parents[1] / "requirements.txt"
    rows: list[tuple[str, str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        spec, marker = (line.split(";", 1) + [""])[:2]
        name, version = spec.split("==", 1)
        rows.append((name.strip(), version.strip(), marker.strip()))
    return rows


def _marker_applies(marker: str) -> bool:
    if not marker:
        return True
    if marker == 'sys_platform == "win32"':
        return sys.platform == "win32"
    return True


def _fonts() -> dict:
    if sys.platform != "win32":
        return {"checked": False, "ready": None, "reason": "Font files are checked on Windows only"}
    folder = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    found = {}
    for face, filename in _FONT_FILES.items():
        path = folder / filename
        found[face] = {"found": path.is_file(), "path": str(path)}
    return {"checked": True, "ready": all(item["found"] for item in found.values()), "faces": found}


def runtime_status() -> dict:
    """Report whether this interpreter can run the verified converter."""
    packages = {}
    package_failures = []
    for name, required, marker in _requirements():
        if not _marker_applies(marker):
            packages[name] = {"required": required, "skipped": True, "marker": marker}
            continue
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = ""
        match = installed == required
        packages[name] = {"required": required, "installed": installed, "match": match}
        if not match:
            package_failures.append(name)
    engine_error = ""
    try:
        converter_module()
        engine_ready = True
    except Exception as exc:
        engine_ready = False
        engine_error = f"{type(exc).__name__}: {exc}"
    fonts = _fonts()
    backend = compare_backend()
    root = repo_root()
    engine_path = root / "scripts" / "convert_datn_pdf_to_docx.py"
    compare_path = root / "scripts" / "compare_datn_docx_pdf.py"
    files_ready = engine_path.is_file() and compare_path.is_file()
    fonts_ready = fonts["ready"] is not False
    ok = not package_failures and engine_ready and files_ready and fonts_ready
    return {
        "ok": ok,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "prefix": sys.prefix,
        "packages_ready": not package_failures,
        "packages": packages,
        "package_failures": package_failures,
        "fonts_ready": fonts["ready"],
        "fonts": fonts,
        "engine_ready": engine_ready,
        "engine_error": engine_error,
        "files_ready": files_ready,
        "engine": "scripts/convert_datn_pdf_to_docx.py",
        "compare": "scripts/compare_datn_docx_pdf.py",
        "compare_ready": backend["ready"],
        "compare_backend": backend,
        "install": "python pdfdocx/install.py",
    }
