"""Build frozen E1 train/val/test ID manifest from eligible ∩ metadata."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_ids(ids: list[str]) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(i.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    elig_path = _ROOT / "data" / "processed" / "vsl400_eligible_sample_manifest.json"
    labels_path = _ROOT / "data" / "processed" / "vsl400_labels.json"
    meta_path = _ROOT / "metadata.csv"
    out = _ROOT / "data" / "processed" / "vsl400_e1_split_manifest.json"

    elig = json.loads(elig_path.read_text(encoding="utf-8"))
    meta = pd.read_csv(meta_path)
    v = meta[meta["dataset"] == "VSL400"].copy()
    v["video_id"] = v["video_id"].astype(str)
    elig_set = set(map(str, elig["eligible_ids"]))
    rows = v[v["video_id"].isin(elig_set)]
    by: dict[str, list[str]] = {}
    for split, g in rows.groupby(rows["split"].astype(str)):
        by[str(split)] = sorted(g["video_id"].astype(str).tolist())

    assert set(by) == {"train", "val", "test"}, by.keys()
    assert len(by["train"]) == 15992
    assert len(by["val"]) == 2595
    assert len(by["test"]) == 6165
    assert set(by["train"] + by["val"] + by["test"]) == elig_set

    man = {
        "schema": "vsl400_e1_split_manifest_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VSL400",
        "eligible_sample_manifest": "data/processed/vsl400_eligible_sample_manifest.json",
        "eligible_sample_manifest_sha256": _sha256_file(elig_path),
        "metadata": "metadata.csv",
        "metadata_sha256": _sha256_file(meta_path),
        "labels": "data/processed/vsl400_labels.json",
        "labels_sha256": _sha256_file(labels_path),
        "train_ids": by["train"],
        "val_ids": by["val"],
        "test_ids": by["test"],
        "n_train": len(by["train"]),
        "n_val": len(by["val"]),
        "n_test": len(by["test"]),
        "train_id_sha256": _sha_ids(by["train"]),
        "val_id_sha256": _sha_ids(by["val"]),
        "test_id_sha256": _sha_ids(by["test"]),
        "note": (
            "E1 trains on train_ids and selects on val_ids only. "
            "test_ids are lineage-only; E1 must not compute test KPI."
        ),
    }
    text = json.dumps(man, ensure_ascii=False, indent=2) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"ok": True, "out": str(out), "sha256": _sha256_file(out), **{k: man[k] for k in ("n_train", "n_val", "n_test")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
