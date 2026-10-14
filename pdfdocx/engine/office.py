# -*- coding: utf-8 -*-
"""Choose the registered Word or WPS COM server without launching it."""

from __future__ import annotations

import sys

# Word.Application is the system default and may point at either product.
# The versioned and WPS ids cover a machine where that default is missing.
_PROGIDS = (
    "Word.Application",
    "Word.Application.16",
    "kwps.Application",
    "KWPS.Application",
)


def _server_command(progid: str) -> str | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, progid + r"\CLSID") as key:
            clsid = winreg.QueryValue(key, None)
    except OSError:
        return None
    for leaf in ("LocalServer32", "LocalServer"):
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf"CLSID\{clsid}\{leaf}") as key:
                value = winreg.QueryValue(key, None)
        except OSError:
            continue
        if value and value.strip():
            return value
    return None


def _product(command: str) -> str:
    low = command.lower()
    if "winword.exe" in low:
        return "microsoft_word"
    if "wps.exe" in low or "kingsoft" in low:
        return "wps"
    return "unknown"


def discover() -> dict:
    """List registered Word and WPS servers. Does not start either application."""
    if sys.platform != "win32":
        return {
            "ready": False,
            "reason": "Word/WPS COM is available only on Windows",
            "product": "",
            "progid": "",
            "server": "",
            "apps": [],
            "launch_order": [],
        }
    apps: list[dict] = []
    seen: set[str] = set()
    for progid in _PROGIDS:
        key = progid.casefold()
        if key in seen:
            continue
        seen.add(key)
        command = _server_command(progid)
        if not command:
            continue
        apps.append({"product": _product(command), "progid": progid, "server": command})
    if not apps:
        return {
            "ready": False,
            "reason": "Neither Word.Application nor kwps.Application is registered",
            "product": "",
            "progid": "",
            "server": "",
            "apps": [],
            "launch_order": [],
        }
    default = next((app for app in apps if app["progid"].casefold() == "word.application"), None)
    if default is None:
        default = next((app for app in apps if app["product"] != "unknown"), apps[0])
    same = [app["progid"] for app in apps if app["product"] == default["product"]]
    rest = [app["progid"] for app in apps if app["progid"] not in same]
    launch_order = []
    for progid in [default["progid"], *same, *rest]:
        if progid not in launch_order:
            launch_order.append(progid)
    return {
        "ready": True,
        "reason": "",
        "product": default["product"],
        "progid": default["progid"],
        "server": default["server"],
        "apps": apps,
        "launch_order": launch_order,
    }
