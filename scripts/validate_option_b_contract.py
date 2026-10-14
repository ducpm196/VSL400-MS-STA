"""Gate D0 — audit Option B data contracts on metadata.csv (+ optional VOYA NPZ probe).

Exit 0: primary VSL400 closed-set OK; appendix corpora flagged (not used as closed-set KPI).
Exit 1: VSL400 contract broken (eval classes missing from train, wrong gloss count, etc.).

Usage (repo root):
  .\\env\\Scripts\\python.exe scripts\\validate_option_b_contract.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def _audit_vsl400(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    v = df[df["dataset"] == "VSL400"]
    if v.empty:
        return ["VSL400 rows missing from metadata.csv"]
    n_gloss = int(v["text_gloss"].nunique())
    if n_gloss != 400:
        errors.append(f"VSL400 gloss count={n_gloss} (expected 400)")
    for split_name in ("train", "val", "test"):
        sub = v[v["split"] == split_name]
        if sub.empty:
            errors.append(f"VSL400 split '{split_name}' empty")
            continue
        glosses = set(sub["text_gloss"].astype(str))
        if split_name != "train":
            train_g = set(v[v["split"] == "train"]["text_gloss"].astype(str))
            missing = glosses - train_g
            if missing:
                errors.append(
                    f"VSL400 {split_name}: {len(missing)} glosses absent from train (closed-set broken)"
                )
    return errors


def _flag_appendix(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    vc = df[df["dataset"] == "VSL-Cropped"]
    if not vc.empty:
        tr = set(vc[vc["split"] == "train"]["text_gloss"].astype(str))
        te = set(vc[vc["split"] == "test"]["text_gloss"].astype(str))
        notes.append(
            f"APPENDIX VSL-Cropped online: n={len(vc)} train_gloss={len(tr)} "
            f"test_gloss={len(te)} intersect={len(tr & te)} "
            f"(intersect=0 => not valid closed-set KPI)"
        )
    voya = df[df["dataset"] == "VOYA_VSL"]
    if not voya.empty:
        notes.append(
            f"APPENDIX VOYA_VSL: metadata_rows={len(voya)} "
            f"(expect 1 row/file today; sequences live inside NPZ - do not train CE as-is)"
        )
    return notes


def _probe_voya(merged: Path) -> list[str]:
    notes: list[str] = []
    if not merged.is_dir():
        notes.append("VOYA Merged/ absent — skip NPZ probe")
        return notes
    npzs = sorted(merged.glob("*.npz"))
    notes.append(f"VOYA NPZ files={len(npzs)}")
    if not npzs:
        return notes
    with np.load(npzs[0]) as z:
        keys = list(z.keys())
        notes.append(f"sample={npzs[0].name} keys={keys}")
        if "sequences" in z.files:
            notes.append(f"sequences.shape={z['sequences'].shape}")
        if "labels" in z.files:
            notes.append(f"labels.shape={z['labels'].shape}")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description="Option B Gate D0 data-contract audit")
    ap.add_argument("--metadata", type=Path, default=_ROOT / "metadata.csv")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--probe-voya", action="store_true")
    args = ap.parse_args()

    if not args.metadata.is_file():
        print(f"[FAIL] missing {args.metadata}")
        return 1

    df = pd.read_csv(args.metadata)
    errors = _audit_vsl400(df)
    notes = _flag_appendix(df)
    if args.probe_voya:
        notes.extend(_probe_voya(_ROOT / "data" / "raw" / "VOYA_VSL" / "Merged"))

    # Ontology intersections (informational)
    v400 = set(df.loc[df["dataset"] == "VSL400", "text_gloss"].astype(str))
    voya = set(df.loc[df["dataset"] == "VOYA_VSL", "text_gloss"].astype(str))
    vc = df[df["dataset"] == "VSL-Cropped"]
    vc_tr = set(vc.loc[vc["split"] == "train", "text_gloss"].astype(str)) if len(vc) else set()
    vc_te = set(vc.loc[vc["split"] == "test", "text_gloss"].astype(str)) if len(vc) else set()
    notes.append(f"ontology VOYA intersect VSL400={len(voya & v400)}")
    notes.append(f"ontology online-train intersect VSL400={len(vc_tr & v400)}")
    notes.append(f"ontology online-test intersect VSL400={len(vc_te & v400)}")

    payload = {
        "gate": "D0",
        "primary_ok": len(errors) == 0,
        "errors": errors,
        "notes": notes,
        "counts": df["dataset"].value_counts().to_dict() if "dataset" in df.columns else {},
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for n in notes:
        print(f"[NOTE] {n}")
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print("[D0] FAIL - VSL400 closed-set contract broken")
        return 1
    print("[D0] PASS - VSL400 closed-set OK; appendix corpora flagged only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
