# -*- coding: utf-8 -*-
"""Verify durable checkpoint registry entries (SHA-256 restore check)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_registry/CHECKPOINT_REGISTRY.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    ok = True
    for e in reg["entries"]:
        p = ROOT / e["durable_rel"]
        if not p.is_file():
            print("MISSING", e["artifact_id"], p)
            ok = False
            continue
        got = sha256_file(p)
        size = p.stat().st_size
        match = got == e["sha256"] and size == int(e["size_bytes"])
        print(
            "OK" if match else "FAIL",
            e["artifact_id"],
            f"size={size}",
            f"sha_match={got == e['sha256']}",
        )
        ok = ok and match
    print("E4_B4SC_REGISTRY_VERIFY_OK" if ok else "E4_B4SC_REGISTRY_VERIFY_FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
