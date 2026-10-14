#!/usr/bin/env python3
"""Cross-platform pdfdocx MCP launcher. Sets the kit on sys.path — no cwd required."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _kit_root() -> Path:
    return Path(__file__).resolve().parent


def _repo_root(kit: Path) -> Path:
    override = os.environ.get("PDFDOCX_REPO_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return kit.parent.resolve()


def _venv_python(repo: Path) -> Path | None:
    if sys.platform == "win32":
        candidate = repo / "_venvs" / "pdfdocx" / "Scripts" / "python.exe"
    else:
        candidate = repo / "_venvs" / "pdfdocx" / "bin" / "python"
    return candidate if candidate.is_file() else None


def main() -> int:
    kit = _kit_root()
    repo = _repo_root(kit)
    os.environ.setdefault("PDFDOCX_REPO_ROOT", str(repo))
    target = _venv_python(repo)
    if target is not None and os.environ.get("PDFDOCX_NO_REEXEC") != "1":
        if Path(sys.executable).resolve() != target.resolve():
            os.environ["PDFDOCX_NO_REEXEC"] = "1"
            os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])
    os.chdir(kit)
    kit_str = str(kit)
    if kit_str not in sys.path:
        sys.path.insert(0, kit_str)
    from engine.mcp_server import main as mcp_main

    return mcp_main()


if __name__ == "__main__":
    raise SystemExit(main())
