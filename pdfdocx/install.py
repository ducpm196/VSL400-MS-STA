#!/usr/bin/env python3
"""Create an isolated pdfdocx environment and point Cursor MCP at it.

Run from any checkout: ``python pdfdocx/install.py``.
The environment is ``_venvs/pdfdocx`` and does not use system site-packages.
Each machine rewrites ``.cursor/mcp.json`` so the command is that machine's
interpreter. The layout engine itself is not modified.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "_venvs" / "pdfdocx"
REQUIREMENTS = ROOT / "pdfdocx" / "requirements.txt"
MCP = ROOT / ".cursor" / "mcp.json"
READ_ONLY = ("engine_contract", "inspect_pdf", "verify_docx", "runtime_status")


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _uses_system_site(venv: Path) -> bool:
    cfg = venv / "pyvenv.cfg"
    if not cfg.is_file():
        return False
    return "include-system-site-packages = true" in cfg.read_text(encoding="utf-8").lower()


def _ensure_venv() -> Path:
    if sys.version_info < (3, 11):
        raise SystemExit(f"Python 3.11 or newer is required, found {sys.version.split()[0]}")
    python = venv_python()
    if VENV.exists() and _uses_system_site(VENV):
        shutil.rmtree(VENV)
    if not python.is_file():
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    return python


def _patch_mcp(python: Path) -> None:
    data = json.loads(MCP.read_text(encoding="utf-8")) if MCP.is_file() else {"mcpServers": {}}
    servers = data.setdefault("mcpServers", {})
    block = servers.get("pdfdocx", {})
    block["command"] = str(python)
    block["args"] = ["${workspaceFolder}/pdfdocx/run_mcp.py"]
    env = block.get("env") if isinstance(block.get("env"), dict) else {}
    env["PDFDOCX_REPO_ROOT"] = "${workspaceFolder}"
    block["env"] = env
    approved = [name for name in block.get("autoApprove", []) if isinstance(name, str)]
    for name in READ_ONLY:
        if name not in approved:
            approved.append(name)
    block["autoApprove"] = approved
    servers["pdfdocx"] = block
    MCP.parent.mkdir(parents=True, exist_ok=True)
    MCP.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    python = _ensure_venv()
    _patch_mcp(python)
    print(json.dumps({"ok": True, "python": str(python), "mcp": str(MCP)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
