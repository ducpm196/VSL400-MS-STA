"""Download / verify MS-STA release artifacts (checkpoints, E2 rows).

Until a GitHub Release or Zenodo URL is published, --verify-only checks local files
against Docs/results/ms_sta_defense_go/ARTIFACT_MANIFEST.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "Docs" / "results" / "ms_sta_defense_go" / "ARTIFACT_MANIFEST.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _safe_under_root(rel: str) -> Path:
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError(f"manifest path escapes project root: {rel}")
    path = (_ROOT / rel).resolve()
    try:
        path.relative_to(_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {rel}") from exc
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if not args.verify_only and not args.download:
        args.verify_only = True
    if not _MANIFEST.is_file():
        print(f"missing manifest: {_MANIFEST}", file=sys.stderr)
        return 2
    man = json.loads(_MANIFEST.read_text(encoding="utf-8-sig"))
    release_url = man.get("release_url")
    errors: list[str] = []
    for item in man.get("items") or []:
        rel = item["path"]
        try:
            path = _safe_under_root(rel)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        expect = item.get("sha256")
        if args.download:
            if not release_url:
                errors.append(f"no release_url configured for {rel}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{release_url.rstrip('/')}/{Path(rel).name}"
            part = path.with_suffix(path.suffix + ".part")
            print(f"GET {url}")
            try:
                urllib.request.urlretrieve(url, part)  # noqa: S310 — operator-controlled URL
                got = _sha(part)
                if expect and got != expect:
                    part.unlink(missing_ok=True)
                    errors.append(f"sha mismatch after download {rel}: got={got[:12]} expect={str(expect)[:12]}")
                    continue
                os.replace(part, path)
            except Exception as exc:  # noqa: BLE001
                part.unlink(missing_ok=True)
                errors.append(f"download failed {rel}: {type(exc).__name__}")
                continue
        if not path.is_file():
            errors.append(f"missing {rel}")
            continue
        got = _sha(path)
        if expect and got != expect:
            errors.append(f"sha mismatch {rel}: got={got[:12]} expect={str(expect)[:12]}")
        else:
            print(f"OK {rel} ({got[:12]}…)")
    if errors:
        for e in errors:
            print(f"FAIL {e}", file=sys.stderr)
        return 1
    print("all present artifacts verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
