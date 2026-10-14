"""Build Path-B retrain-v3 signer-role split manifest (no training, no test KPI)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import PATH_B_SIGNERS, sha256_file, sha256_ids

DEFAULT_OUT = _ROOT / "data" / "processed" / "vsl400_retrain_v3_split_manifest.json"


def _norm_signer(value: object) -> str:
    s = str(value).strip()
    if s.startswith("signer_"):
        return s
    if s.isdigit():
        return f"signer_{int(s)}"
    # metadata may store 001-style
    if s.lstrip("0").isdigit() or s == "0":
        return f"signer_{int(s)}"
    raise ValueError(f"unrecognized signer_id={value!r}")


def build_manifest(
    *,
    eligible_path: Path,
    metadata_path: Path,
    labels_path: Path,
    split_signers_path: Path,
) -> dict:
    elig = json.loads(eligible_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    split_signers = json.loads(split_signers_path.read_text(encoding="utf-8"))
    meta = pd.read_csv(metadata_path)
    v = meta[meta["dataset"] == "VSL400"].copy()
    v["video_id"] = v["video_id"].astype(str)
    v["signer_norm"] = v["signer_id"].map(_norm_signer)
    elig_set = set(map(str, elig["eligible_ids"]))
    rows = v[v["video_id"].isin(elig_set)].copy()
    if rows.empty:
        raise SystemExit("[FAIL] no eligible VSL400 rows")

    # Consistency with locked 26-signer split lists.
    expected_train = set(split_signers["train"])
    expected_val = set(split_signers["val"])
    expected_test = set(split_signers["test"])
    if expected_train | expected_val | expected_test != set(rows["signer_norm"].unique()):
        raise SystemExit("[FAIL] eligible signer set != locked split_signers union")

    roles = {
        "hpo_train": list(PATH_B_SIGNERS["hpo_train"]),
        "tune": list(PATH_B_SIGNERS["tune"]),
        "confirm_train": list(PATH_B_SIGNERS["confirm_train"]),
        "val": list(PATH_B_SIGNERS["val"]),
        "test": list(PATH_B_SIGNERS["test"]),
    }
    missing = list(PATH_B_SIGNERS["missing"])

    # Structural checks
    if set(roles["confirm_train"]) != expected_train:
        raise SystemExit("[FAIL] confirm_train signers != locked train signers")
    if set(roles["val"]) != expected_val:
        raise SystemExit("[FAIL] val signers != locked val signers")
    if set(roles["test"]) != expected_test:
        raise SystemExit("[FAIL] test signers != locked test signers")
    if set(roles["hpo_train"]) | set(roles["tune"]) != set(roles["confirm_train"]):
        raise SystemExit("[FAIL] hpo_train ∪ tune must equal confirm_train")
    if set(roles["hpo_train"]) & set(roles["tune"]):
        raise SystemExit("[FAIL] hpo_train and tune must be disjoint")

    all_role_signers = (
        set(roles["hpo_train"])
        | set(roles["tune"])
        | set(roles["val"])
        | set(roles["test"])
    )
    for s in missing:
        if s in all_role_signers:
            raise SystemExit(f"[FAIL] missing signer {s} appears in a role list")
        if (rows["signer_norm"] == s).any():
            raise SystemExit(f"[FAIL] missing signer {s} has eligible samples")

    # Disjoint development vs retrospective test
    dev = set(roles["confirm_train"]) | set(roles["val"])
    if dev & set(roles["test"]):
        raise SystemExit("[FAIL] test signers overlap development pool")
    if set(roles["tune"]) & set(roles["val"]):
        raise SystemExit("[FAIL] tune overlaps confirmation val")

    ids_by_role: dict[str, list[str]] = {}
    for role, signers in roles.items():
        subset = rows[rows["signer_norm"].isin(signers)]
        ids = sorted(subset["video_id"].astype(str).tolist())
        if not ids:
            raise SystemExit(f"[FAIL] role {role} has zero eligible IDs")
        ids_by_role[role] = ids

    # Unique IDs within each role; union of confirm_train+val+test == eligible
    union = set(ids_by_role["confirm_train"]) | set(ids_by_role["val"]) | set(ids_by_role["test"])
    if union != elig_set:
        raise SystemExit(
            f"[FAIL] confirm_train∪val∪test != eligible "
            f"(|union|={len(union)} |elig|={len(elig_set)})"
        )
    if set(ids_by_role["hpo_train"]) & set(ids_by_role["tune"]):
        raise SystemExit("[FAIL] hpo_train IDs overlap tune IDs")
    if set(ids_by_role["hpo_train"]) | set(ids_by_role["tune"]) != set(ids_by_role["confirm_train"]):
        raise SystemExit("[FAIL] hpo_train∪tune IDs != confirm_train IDs")

    # Class coverage on confirm_train (structural; no metric retry)
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

    man = {
        "schema": "ms_sta_retrain_v3_split_manifest_v1",
        "path": "B",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VSL400",
        "test_claim_label": "retrospective_signer_disjoint_confirmation",
        "eligible_sample_manifest": "data/processed/vsl400_eligible_sample_manifest.json",
        "eligible_sample_manifest_sha256": sha256_file(eligible_path),
        "metadata": "metadata.csv",
        "metadata_sha256": sha256_file(metadata_path),
        "labels": "data/processed/vsl400_labels.json",
        "labels_sha256": sha256_file(labels_path),
        "split_signers": "data/processed/vsl400_split_signers.json",
        "split_signers_sha256": sha256_file(split_signers_path),
        "missing_signers": missing,
        "sealed_holdout_signers": [],
        "signers": roles,
        "hpo_train_ids": ids_by_role["hpo_train"],
        "tune_ids": ids_by_role["tune"],
        "confirm_train_ids": ids_by_role["confirm_train"],
        "val_ids": ids_by_role["val"],
        "test_ids": ids_by_role["test"],
        # Compatibility aliases used by trainer role loader
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
            "Path B: HPO uses hpo_train/tune; confirmation uses confirm_train/val; "
            "test_ids are lineage-only until evaluation_locked and must be labeled "
            "retrospective_signer_disjoint_confirmation. HPO checkpoints are disposable."
        ),
    }
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description="Build retrain-v3 Path B split manifest")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else _ROOT / args.out

    man = build_manifest(
        eligible_path=_ROOT / "data/processed/vsl400_eligible_sample_manifest.json",
        metadata_path=_ROOT / "metadata.csv",
        labels_path=_ROOT / "data/processed/vsl400_labels.json",
        split_signers_path=_ROOT / "data/processed/vsl400_split_signers.json",
    )
    text = json.dumps(man, ensure_ascii=False, indent=2) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    digest = sha256_file(out)
    summary = {
        "ok": True,
        "out": str(out.relative_to(_ROOT)).replace("\\", "/"),
        "sha256": digest,
        "n_hpo_train": man["n_hpo_train"],
        "n_tune": man["n_tune"],
        "n_confirm_train": man["n_confirm_train"],
        "n_val": man["n_val"],
        "n_test": man["n_test"],
        "path": "B",
        "test_claim_label": man["test_claim_label"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
