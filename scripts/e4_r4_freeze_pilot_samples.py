# -*- coding: utf-8 -*-
"""Freeze E4-R4 representation pilot sample IDs (signer-stratified; no test).

Fail-closed: wrong SHAs, missing metadata, duplicates, test/tune overlap → SystemExit(2).
No assert (safe under python -O).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/processed/vsl400_retrain_v4_split_manifest.json"
ELIGIBLE = ROOT / "data/processed/vsl400_eligible_sample_manifest.json"
META = ROOT / "metadata.csv"
POLICY = ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4_POLICY.json"
OUT_DIR = ROOT / "Docs/results/ms_sta_v6_e4_r4"
OUT = OUT_DIR / "E4_R4_PILOT_SAMPLE_FREEZE.json"

N_TRAIN_PER_SIGNER = 8
N_VAL_PER_SIGNER = 4
N_QA_CONTACT = 24
SEED = 42

EXPECTED_SPLIT_SHA = "e7cce5c9d4416a563418f27bba5fc8e89205fece2e5c733c7779782f0b503b09"
EXPECTED_ELIGIBLE_SHA = "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die(msg: str) -> None:
    print(f"E4_R4_PILOT_FREEZE_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if not SPLIT.is_file():
        die("missing_split_manifest")
    if not ELIGIBLE.is_file():
        die("missing_eligible_manifest")
    if not META.is_file():
        die("missing_metadata_csv")
    if not POLICY.is_file():
        die("missing_policy")

    split_sha = sha256_file(SPLIT)
    elig_sha = sha256_file(ELIGIBLE)
    if split_sha != EXPECTED_SPLIT_SHA:
        die(f"split_sha_mismatch:{split_sha}")
    if elig_sha != EXPECTED_ELIGIBLE_SHA:
        die(f"eligible_sha_mismatch:{elig_sha}")

    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("split_manifest_sha256") != split_sha:
        die("policy_split_sha_drift")
    if policy.get("eligible_manifest_sha256") != elig_sha:
        die("policy_eligible_sha_drift")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    test_ids = set(map(str, split["test_ids"]))
    tune_ids = set(map(str, split["tune_ids"]))
    hpo = set(map(str, split["hpo_train_ids"]))
    val = set(map(str, split["val_ids"]))

    if hpo & test_ids:
        die("hpo_overlaps_test_in_manifest")
    if val & test_ids:
        die("val_overlaps_test_in_manifest")
    if hpo & tune_ids:
        die("hpo_overlaps_tune_in_manifest")

    id_to_signer: dict[str, str] = {}
    with META.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        try:
            id_key = next(k for k in fields if k.lower() in {"video_id", "sample_id", "id", "clip_id"})
            sig_key = next(k for k in fields if "signer" in k.lower())
        except StopIteration as e:
            die(f"metadata_columns_missing:{e}")
        for row in reader:
            sid = str(row[id_key]).strip()
            if not sid:
                die("empty_video_id_in_metadata")
            raw_sig = str(row[sig_key]).strip()
            if not raw_sig:
                die(f"missing_signer_for:{sid}")
            sig = raw_sig.replace("signer_", "").replace("Signer_", "")
            id_to_signer[sid] = sig
            if sid.isdigit():
                id_to_signer[str(int(sid))] = sig
                id_to_signer[f"{int(sid):06d}"] = sig

    def stratified(pool: set[str], per_signer: int, salt: str) -> list[str]:
        by: dict[str, list[str]] = defaultdict(list)
        missing_meta = []
        for sid in sorted(pool):
            if sid in test_ids:
                die(f"test_id_in_pool:{sid}")
            if sid in tune_ids and salt == "train":
                die(f"tune_id_in_train_pool:{sid}")
            sig = id_to_signer.get(sid)
            if sig is None:
                missing_meta.append(sid)
                continue
            by[sig].append(sid)
        if missing_meta:
            die(f"missing_metadata_for_{len(missing_meta)}_ids_example:{missing_meta[:5]}")
        chosen: list[str] = []
        for sig in sorted(by.keys(), key=lambda s: int(s) if s.isdigit() else s):
            ids = sorted(by[sig], key=lambda x: hashlib.sha256(f"{SEED}:{salt}:{x}".encode()).hexdigest())
            chosen.extend(ids[:per_signer])
        if len(chosen) != len(set(chosen)):
            die("duplicate_ids_after_stratify")
        return chosen

    pilot_train = stratified(hpo, N_TRAIN_PER_SIGNER, "train")
    pilot_val = stratified(val, N_VAL_PER_SIGNER, "val")
    qa_pool = set(pilot_train) | set(pilot_val)
    qa = sorted(qa_pool, key=lambda x: hashlib.sha256(f"{SEED}:qa:{x}".encode()).hexdigest())[
        :N_QA_CONTACT
    ]

    for label, ids in (("train", pilot_train), ("val", pilot_val), ("qa", qa)):
        s = set(ids)
        if s & test_ids:
            die(f"{label}_overlaps_test")
        if label == "train" and (s & tune_ids):
            die("train_overlaps_tune")
        if len(ids) != len(s):
            die(f"{label}_has_duplicates")

    if set(pilot_train) & set(pilot_val):
        die("train_val_overlap")

    payload = {
        "schema": "ms_sta_v6_e4_r4_pilot_sample_freeze_v2",
        "campaign": "E4-R4",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "n_train_per_signer": N_TRAIN_PER_SIGNER,
        "n_val_per_signer": N_VAL_PER_SIGNER,
        "n_qa_contact": N_QA_CONTACT,
        "roles": {
            "pilot_train": "from hpo_train_ids only",
            "pilot_val": "from val_ids only — representation QA only; forbid architecture selection",
            "forbid_test_ids": True,
            "forbid_tune_as_train": True,
        },
        "split_manifest_sha256": split_sha,
        "eligible_sample_manifest_sha256": elig_sha,
        "metadata_sha256": sha256_file(META),
        "policy_sha256": sha256_file(POLICY),
        "n_pilot_train": len(pilot_train),
        "n_pilot_val": len(pilot_val),
        "n_qa": len(qa),
        "pilot_train_ids": pilot_train,
        "pilot_val_ids": pilot_val,
        "qa_contact_ids": qa,
        "pilot_train_ids_sha256": hashlib.sha256("\n".join(pilot_train).encode()).hexdigest(),
        "pilot_val_ids_sha256": hashlib.sha256("\n".join(pilot_val).encode()).hexdigest(),
        "qa_contact_ids_sha256": hashlib.sha256("\n".join(qa).encode()).hexdigest(),
        "note": "Frozen before representation metrics. Alignment/mask/motion FAIL rejects candidate.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print("Wrote", OUT)
    print("train", len(pilot_train), "val", len(pilot_val), "qa", len(qa))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
