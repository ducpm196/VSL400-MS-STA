"""Build Path-B retrain-v4 signer-role split manifest (no training, no test KPI).

Deterministic development-pool shuffle (protocol-derived seed). Scientific
artifact bytes are bit-exact across rebuilds (no wall-clock timestamp).
Test signers 24–28 remain fixed and retrospective. v4 val is disjoint from
v3 tune∪val and drawn only from the v3 hpo_train pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import sha256_file, sha256_ids
from src.training.retrain_v4_split import (
    ARTIFACT_CREATED_AT,
    SPLIT_ALGORITHM,
    SPLIT_SEED,
    assign_path_b_roles,
    split_algorithm_metadata,
)

DEFAULT_OUT = _ROOT / "data" / "processed" / "vsl400_retrain_v4_split_manifest.json"


def _norm_signer(value: object) -> str:
    s = str(value).strip()
    if s.startswith("signer_"):
        return s
    if s.isdigit():
        return f"signer_{int(s)}"
    if s.lstrip("0").isdigit() or s == "0":
        return f"signer_{int(s)}"
    raise ValueError(f"unrecognized signer_id={value!r}")


def build_manifest(
    *,
    eligible_path: Path,
    metadata_path: Path,
    labels_path: Path,
    seed: int = SPLIT_SEED,
) -> dict:
    elig = json.loads(eligible_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    meta = pd.read_csv(metadata_path)
    v = meta[meta["dataset"] == "VSL400"].copy()
    v["video_id"] = v["video_id"].astype(str)
    v["signer_norm"] = v["signer_id"].map(_norm_signer)
    elig_set = set(map(str, elig["eligible_ids"]))
    rows = v[v["video_id"].isin(elig_set)].copy()
    if rows.empty:
        raise SystemExit("[FAIL] no eligible VSL400 rows")

    eligible_signers = sorted(rows["signer_norm"].unique().tolist())
    roles = assign_path_b_roles(eligible_signers, seed=seed)

    ids_by_role: dict[str, list[str]] = {}
    for role in ("hpo_train", "tune", "confirm_train", "val", "test"):
        subset = rows[rows["signer_norm"].isin(roles[role])]
        ids = sorted(subset["video_id"].astype(str).tolist())
        if not ids:
            raise SystemExit(f"[FAIL] role {role} has zero eligible IDs")
        ids_by_role[role] = ids

    union = set(ids_by_role["confirm_train"]) | set(ids_by_role["val"]) | set(ids_by_role["test"])
    if union != elig_set:
        raise SystemExit(
            f"[FAIL] confirm_train∪val∪test != eligible "
            f"(|union|={len(union)} |elig|={len(elig_set)})"
        )
    if set(ids_by_role["hpo_train"]) & set(ids_by_role["tune"]):
        raise SystemExit("[FAIL] hpo_train IDs overlap tune IDs")
    if set(ids_by_role["hpo_train"]) | set(ids_by_role["tune"]) != set(
        ids_by_role["confirm_train"]
    ):
        raise SystemExit("[FAIL] hpo_train∪tune IDs != confirm_train IDs")

    gloss_to_id = labels["gloss_to_id"]
    n_classes = int(labels["num_classes"])
    train_rows = rows[rows["video_id"].isin(ids_by_role["confirm_train"])]
    covered = set()
    for g in train_rows["text_gloss"].astype(str):
        if g in gloss_to_id:
            covered.add(int(gloss_to_id[g]))
    if len(covered) != n_classes:
        raise SystemExit(
            f"[FAIL] confirm_train class coverage {len(covered)} != {n_classes}"
        )

    algo = split_algorithm_metadata()
    algo["seed"] = int(seed)

    man = {
        "schema": "ms_sta_retrain_v4_split_manifest_v1",
        "path": "B",
        "created_at": ARTIFACT_CREATED_AT,
        "dataset": "VSL400",
        "test_claim_label": "retrospective_signer_disjoint_confirmation",
        "split_algorithm": algo,
        "parent_v3_split_not_reused_for_independent_confirmation": True,
        "eligible_sample_manifest": "data/processed/vsl400_eligible_sample_manifest.json",
        "eligible_sample_manifest_sha256": sha256_file(eligible_path),
        "metadata": "metadata.csv",
        "metadata_sha256": sha256_file(metadata_path),
        "labels": "data/processed/vsl400_labels.json",
        "labels_sha256": sha256_file(labels_path),
        "missing_signers": list(roles["missing"]),
        "sealed_holdout_signers": [],
        "signers": {
            "hpo_train": list(roles["hpo_train"]),
            "tune": list(roles["tune"]),
            "confirm_train": list(roles["confirm_train"]),
            "val": list(roles["val"]),
            "test": list(roles["test"]),
        },
        "hpo_train_ids": ids_by_role["hpo_train"],
        "tune_ids": ids_by_role["tune"],
        "confirm_train_ids": ids_by_role["confirm_train"],
        "val_ids": ids_by_role["val"],
        "test_ids": ids_by_role["test"],
        "train_ids": ids_by_role["confirm_train"],
        "n_hpo_train": len(ids_by_role["hpo_train"]),
        "n_tune": len(ids_by_role["tune"]),
        "n_confirm_train": len(ids_by_role["confirm_train"]),
        "n_val": len(ids_by_role["val"]),
        "n_test": len(ids_by_role["test"]),
        "hpo_train_id_sha256": sha256_ids(ids_by_role["hpo_train"]),
        "tune_id_sha256": sha256_ids(ids_by_role["tune"]),
        "confirm_train_id_sha256": sha256_ids(ids_by_role["confirm_train"]),
        "val_id_sha256": sha256_ids(ids_by_role["val"]),
        "test_id_sha256": sha256_ids(ids_by_role["test"]),
        "note": (
            f"Path B v4: algorithm={SPLIT_ALGORITHM} seed={seed}. "
            "HPO uses hpo_train/tune; confirmation uses confirm_train/val; "
            "test_ids are lineage-only until evaluation_locked and must be labeled "
            "retrospective_signer_disjoint_confirmation. "
            "v4 val ⊆ v3 hpo_train and disjoint from v3 tune∪val."
        ),
    }
    return man


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build retrain-v4 Path B split manifest")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument(
        "--allow-overwrite-draft",
        action="store_true",
        help="Required to overwrite an existing manifest (draft rebuilds only).",
    )
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else _ROOT / args.out

    if out.exists() and not args.allow_overwrite_draft:
        print(
            f"[FAIL] refuse overwrite existing manifest without "
            f"--allow-overwrite-draft: {out}"
        )
        return 2

    man = build_manifest(
        eligible_path=_ROOT / "data/processed/vsl400_eligible_sample_manifest.json",
        metadata_path=_ROOT / "metadata.csv",
        labels_path=_ROOT / "data/processed/vsl400_labels.json",
        seed=int(args.seed),
    )
    text = json.dumps(man, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(out, text)
    digest = sha256_file(out)
    summary = {
        "ok": True,
        "out": str(out.relative_to(_ROOT)).replace("\\", "/"),
        "sha256": digest,
        "algorithm": SPLIT_ALGORITHM,
        "seed": int(args.seed),
        "created_at": ARTIFACT_CREATED_AT,
        "n_hpo_train": man["n_hpo_train"],
        "n_tune": man["n_tune"],
        "n_confirm_train": man["n_confirm_train"],
        "n_val": man["n_val"],
        "n_test": man["n_test"],
        "signers": man["signers"],
        "path": "B",
        "test_claim_label": man["test_claim_label"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
