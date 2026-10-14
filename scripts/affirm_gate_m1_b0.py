"""Gate M1 — reaffirm B0 Gate3 locked checkpoint + smoke/canonical eval hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "26d866c33676f921198c9ffc61710ff1c4671f6debe782c81c828124977cfe11"
GATE3 = _ROOT / "Docs" / "results" / "gate3" / "CHECKSUMS.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Option B Gate M1 — affirm B0")
    ap.add_argument("--ckpt", type=Path, default=_ROOT / "data" / "processed" / "vsl400_classifier.pth")
    ap.add_argument("--labels", type=Path, default=_ROOT / "data" / "processed" / "vsl400_labels.json")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "logs" / "gate_m1" / "affirm.json",
    )
    ap.add_argument(
        "--run-smoke-eval",
        action="store_true",
        help="Also run evaluate_vsl400.py --limit 200 --sample stratified (diagnostic)",
    )
    args = ap.parse_args()

    errors: list[str] = []
    notes: list[str] = []
    if not args.ckpt.is_file():
        errors.append(f"missing ckpt {args.ckpt}")
    if not args.labels.is_file():
        errors.append(f"missing labels {args.labels}")

    sha = _sha256(args.ckpt) if args.ckpt.is_file() else None
    if sha and sha != EXPECTED_SHA:
        errors.append(f"ckpt sha mismatch got={sha} expected={EXPECTED_SHA}")
    elif sha:
        notes.append("ckpt sha matches Gate3 locked checkpoint")

    gate3 = {}
    if GATE3.is_file():
        gate3 = json.loads(GATE3.read_text(encoding="utf-8"))
        notes.append(
            f"Gate3 front Top-1={gate3.get('front_top1')} Top-5={gate3.get('front_top5')} n={gate3.get('n')}"
        )
        if gate3.get("checkpoint_sha256") != EXPECTED_SHA:
            errors.append("Docs/results/gate3/CHECKSUMS.json checkpoint_sha256 drift")

    labels = {}
    if args.labels.is_file():
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        if int(labels.get("num_classes", -1)) != 400:
            errors.append(f"labels num_classes={labels.get('num_classes')} expected 400")

    smoke = None
    if args.run_smoke_eval and not errors:
        import subprocess

        out = _ROOT / "logs" / "gate_m1" / "smoke_eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(_ROOT / "scripts" / "evaluate_vsl400.py"),
            "--limit",
            "200",
            "--sample",
            "stratified",
            "--cache-mode",
            "read_only",
            "--out",
            str(out),
        ]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        smoke = {"returncode": proc.returncode, "out": str(out)}
        if proc.returncode != 0:
            errors.append(f"smoke eval failed rc={proc.returncode}: {proc.stderr[-500:]}")
        elif out.is_file():
            summary = json.loads(out.read_text(encoding="utf-8")).get("summary") or {}
            smoke["summary"] = {
                "n": summary.get("n"),
                "top1_acc": summary.get("top1_acc"),
                "top5_acc": summary.get("top5_acc"),
            }
            notes.append(f"smoke stratified n=200 top1={summary.get('top1_acc')} top5={summary.get('top5_acc')}")

    payload = {
        "gate": "M1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pass": len(errors) == 0,
        "ckpt_sha256": sha,
        "expected_sha256": EXPECTED_SHA,
        "gate3_checksums": gate3,
        "labels_num_classes": labels.get("num_classes"),
        "errors": errors,
        "notes": notes,
        "smoke": smoke,
        "b0_canonical": {
            "front_top1": gate3.get("front_top1"),
            "front_top5": gate3.get("front_top5"),
            "n": gate3.get("n"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for n in notes:
        print(f"[NOTE] {n}")
    for e in errors:
        print(f"[FAIL] {e}")
    print(f"[M1] {'PASS' if payload['pass'] else 'FAIL'} -> {args.out}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
