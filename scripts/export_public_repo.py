# -*- coding: utf-8 -*-
"""Build the public, code-only copy of this repository.

The VSL400 Data Usage Agreement (Zenodo record 21366957) forbids publishing the
dataset or any part of it, including its metadata, frames and videos, in a public
code repository. This script exports a git revision, removes everything that
belongs to the dataset or depends on it, swaps in the public README files and
then checks the result:

* no excluded path and no data/media file type is left;
* no credential pattern appears in any text file;
* every relative Markdown link points to a file that exists in the export.

Usage:
    python scripts/export_public_repo.py --out ../VSL_Translation_Project_public [--ref HEAD]
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "scripts" / "public_repo"

# Dataset metadata, sealed evidence (per-clip predictions, features, frames, weights)
# and thesis files that contain participant frames.
EXCLUDE = (
    "metadata.csv",
    "data",
    "Docs/results",
    "Docs/thesis",
    "Docs/references/au.177075292.21810740_v1.pdf",
    "scripts/public_repo",
)
OVERLAY_MAP = {
    "README.md": "README.md",
    "Docs_README.md": "Docs/README.md",
}
FORBIDDEN_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".pyth", ".ckpt", ".safetensors", ".onnx",
    ".mp4", ".avi", ".mov", ".webm", ".jsonl", ".csv", ".pdf", ".pptx", ".docx", ".zip",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
IMAGE_ALLOWED_DIR = "Docs/assets/figures/"
SECRET_PATTERNS = [
    re.compile(p)
    for p in (
        r"AIza[0-9A-Za-z_\-]{35}",
        r"\bhf_[A-Za-z0-9]{30,}",
        r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}",
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    )
]
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")


def export_tree(ref: str, out: Path) -> None:
    data = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter="data")


def remove_excluded(out: Path) -> None:
    for rel in EXCLUDE:
        target = out / rel
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def apply_overlay(out: Path) -> None:
    for src, dst in OVERLAY_MAP.items():
        shutil.copyfile(OVERLAY / src, out / dst)


def unlink_private_targets(out: Path) -> int:
    """Turn Markdown links that point into excluded paths into plain text (the path stays visible)."""
    excluded = [(out / rel).resolve() for rel in EXCLUDE]
    changed = 0
    for path in out.rglob("*.md"):
        text = path.read_text(encoding="utf-8")

        def repl(match: re.Match) -> str:
            label, link = match.group(1), match.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", link, re.I):
                return match.group(0)
            target = (path.parent / link.split("#", 1)[0]).resolve()
            if any(target == e or e in target.parents for e in excluded):
                return label
            return match.group(0)

        new = re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", repl, text)
        if new != text:
            path.write_text(new, encoding="utf-8", newline="")
            changed += 1
    return changed


def check(out: Path) -> list[str]:
    problems: list[str] = []
    files = [p for p in out.rglob("*") if p.is_file()]
    for path in files:
        rel = path.relative_to(out).as_posix()
        if any(rel == e or rel.startswith(e + "/") for e in EXCLUDE):
            problems.append(f"excluded_path_present:{rel}")
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"data_or_media_file:{rel}")
        if suffix in IMAGE_SUFFIXES and not rel.startswith(IMAGE_ALLOWED_DIR):
            problems.append(f"image_outside_allowed_dir:{rel}")
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for rx in SECRET_PATTERNS:
            if rx.search(text):
                problems.append(f"credential_pattern:{rel}")
        if suffix == ".md":
            for match in LINK.finditer(text):
                link = match.group(1)
                if re.match(r"^[a-z][a-z0-9+.-]*:", link, re.I):
                    continue
                if not (path.parent / link).exists():
                    problems.append(f"broken_link:{rel}->{link}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        print(f"refuse: {out} exists and is not empty", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    export_tree(args.ref, out)
    remove_excluded(out)
    apply_overlay(out)
    unlinked = unlink_private_targets(out)
    problems = check(out)
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"exported {args.ref} -> {out} ({n_files} files); private links turned into text in {unlinked} Markdown files")
    for problem in problems:
        print("PROBLEM", problem)
    print("OK" if not problems else f"FAILED ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
