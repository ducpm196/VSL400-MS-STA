# -*- coding: utf-8 -*-
"""Independent train/val/test split + leakage audit from locked manifests."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs" / "results" / "project_predefense_audit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    split_rel = "data/processed/vsl400_retrain_v4_split_manifest.json"
    split_path = ROOT / split_rel
    labels_path = ROOT / "data/processed/vsl400_labels.json"
    e4 = json.loads((ROOT / "Docs/results/ms_sta_v6_e4_final/E4_FINAL_CONTROLLED_NO_GO.json").read_text(encoding="utf-8"))
    reg = json.loads((ROOT / "data/processed/ms_sta_registry_v6_defense_go.json").read_text(encoding="utf-8"))

    gaps: list[str] = []
    findings: list[dict[str, Any]] = []

    if not split_path.exists():
        gaps.append(split_rel)
        atomic_write_json(
            OUT / "DATA_SPLIT_INDEPENDENT_AUDIT.json",
            {"status": "SOURCE_GAP", "missing": gaps, "created_at": NOW},
        )
        return 1

    split = json.loads(split_path.read_text(encoding="utf-8"))
    computed_split_sha = sha256_file(split_path)

    # verify embedded id list SHAs if present
    sha_checks = {}
    for key, id_key in [
        ("val_id_sha256", "val_ids"),
        ("test_id_sha256", "test_ids"),
        ("train_id_sha256", "train_ids"),
        ("hpo_train_id_sha256", "hpo_train_ids"),
        ("tune_id_sha256", "tune_ids"),
        ("confirm_train_id_sha256", "confirm_train_ids"),
    ]:
        if key in split and id_key in split:
            payload = json.dumps(split[id_key], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            # manifests often hash newline-joined or json — try common forms
            forms = {
                "json_compact": hashlib.sha256(payload).hexdigest(),
                "json_indent2": hashlib.sha256(json.dumps(split[id_key], indent=2, ensure_ascii=False).encode()).hexdigest(),
                "newline_join": hashlib.sha256(("\n".join(split[id_key]) + "\n").encode()).hexdigest(),
                "newline_join_no_trailing": hashlib.sha256("\n".join(split[id_key]).encode()).hexdigest(),
            }
            sha_checks[key] = {
                "expected": split[key],
                "matches_any_form": split[key] in forms.values(),
                "matched_form": next((n for n, h in forms.items() if h == split[key]), None),
            }

    sets = {
        "hpo_train": set(split.get("hpo_train_ids") or []),
        "tune": set(split.get("tune_ids") or []),
        "confirm_train": set(split.get("confirm_train_ids") or []),
        "train": set(split.get("train_ids") or []),
        "val": set(split.get("val_ids") or []),
        "test": set(split.get("test_ids") or []),
    }

    raw_counts = {
        "schema": "predefense_data_split_raw_counts_v1",
        "created_at": NOW,
        "split_manifest": split_rel,
        "split_manifest_sha256": computed_split_sha,
        "counts": {k: len(v) for k, v in sets.items()},
        "declared_counts": {
            "n_hpo_train": split.get("n_hpo_train"),
            "n_tune": split.get("n_tune"),
            "n_confirm_train": split.get("n_confirm_train"),
            "n_val": split.get("n_val"),
            "n_test": split.get("n_test"),
        },
        "signers_field_n": len(split.get("signers") or {}),
        "missing_signers": split.get("missing_signers"),
        "sealed_holdout_signers": split.get("sealed_holdout_signers"),
        "fixed_test_signers": (split.get("split_algorithm") or {}).get("fixed_test_signers"),
        "test_claim_label": split.get("test_claim_label"),
    }

    # count consistency
    for name, key in [
        ("hpo_train", "n_hpo_train"),
        ("tune", "n_tune"),
        ("confirm_train", "n_confirm_train"),
        ("val", "n_val"),
        ("test", "n_test"),
    ]:
        if split.get(key) is not None and split[key] != len(sets[name]):
            findings.append(
                {
                    "id": "LEAK-COUNT-MISMATCH",
                    "severity": "P0",
                    "claim_type": "FACT",
                    "detail": f"{key} declared {split[key]} but list len {len(sets[name])}",
                }
            )

    # duplicate IDs within each split
    for name, ids in [
        ("hpo_train", split.get("hpo_train_ids") or []),
        ("tune", split.get("tune_ids") or []),
        ("confirm_train", split.get("confirm_train_ids") or []),
        ("train", split.get("train_ids") or []),
        ("val", split.get("val_ids") or []),
        ("test", split.get("test_ids") or []),
    ]:
        c = Counter(ids)
        dups = [i for i, n in c.items() if n > 1]
        if dups:
            findings.append(
                {
                    "id": f"LEAK-DUP-IDS-{name}",
                    "severity": "P0",
                    "claim_type": "FACT",
                    "n_dup_ids": len(dups),
                    "examples": dups[:10],
                }
            )

    # pairwise overlaps (critical)
    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
        ("hpo_train", "val"),
        ("hpo_train", "test"),
        ("tune", "val"),
        ("tune", "test"),
        ("confirm_train", "val"),
        ("confirm_train", "test"),
        ("hpo_train", "tune"),  # may be intentional disjoint pools
    ]
    overlaps = {}
    for a, b in pairs:
        inter = sorted(sets[a] & sets[b])
        overlaps[f"{a}_x_{b}"] = {"n": len(inter), "examples": inter[:10]}
        if a in {"train", "confirm_train", "hpo_train", "tune"} and b in {"val", "test"} and inter:
            findings.append(
                {
                    "id": f"LEAK-SAMPLE-OVERLAP-{a}-{b}",
                    "severity": "P0",
                    "claim_type": "FACT",
                    "n": len(inter),
                    "examples": inter[:20],
                }
            )
        if {a, b} == {"val", "test"} and inter:
            findings.append(
                {
                    "id": "LEAK-VAL-TEST-OVERLAP",
                    "severity": "P0",
                    "claim_type": "FACT",
                    "n": len(inter),
                    "examples": inter[:20],
                }
            )

    # signer/path via metadata.csv (split.metadata) + eligible id list provenance
    import csv

    eligible_rel = split.get("eligible_sample_manifest")
    eligible_path = ROOT / eligible_rel if eligible_rel else None
    signer_by_id: dict[str, str] = {}
    path_by_id: dict[str, str] = {}
    label_by_id: dict[str, Any] = {}
    content_hash_by_id: dict[str, str] = {}

    if eligible_path and eligible_path.exists():
        elig_sha = sha256_file(eligible_path)
        raw_counts["eligible_manifest_sha256_computed"] = elig_sha
        raw_counts["eligible_manifest_sha256_expected"] = split.get("eligible_sample_manifest_sha256")
        raw_counts["eligible_sha_match"] = elig_sha == split.get("eligible_sample_manifest_sha256")
    else:
        gaps.append(str(eligible_rel))

    meta_rel = split.get("metadata") or "metadata.csv"
    meta_path = ROOT / meta_rel
    if not meta_path.exists():
        gaps.append(meta_rel)
        findings.append(
            {
                "id": "LEAK-METADATA-MISSING",
                "severity": "P1",
                "claim_type": "SOURCE_GAP",
                "detail": "Cannot verify signer/path leakage without metadata.csv",
            }
        )
    else:
        raw_counts["metadata_sha256_computed"] = sha256_file(meta_path)
        raw_counts["metadata_sha256_expected"] = split.get("metadata_sha256")
        raw_counts["metadata_sha_match"] = raw_counts["metadata_sha256_computed"] == split.get("metadata_sha256")
        with meta_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = str(row.get("video_id") or row.get("sample_id") or row.get("id") or "").strip()
                if not sid:
                    continue
                # prefer signer_id; do not silently accept empty
                signer = str(row.get("signer_id") or "").strip()
                if not signer:
                    signer = str(row.get("signer") or "").strip()
                signer_by_id[sid] = signer
                path_by_id[sid] = str(
                    row.get("path")
                    or row.get("video_path")
                    or row.get("rel_path")
                    or row.get("filepath")
                    or row.get("file")
                    or ""
                ).strip()
                # metadata may only have gloss; label index comes from labels.json if needed
                if "label" in row and row["label"] != "":
                    try:
                        label_by_id[sid] = int(row["label"])
                    except ValueError:
                        label_by_id[sid] = row["label"]
                for hk in ("content_sha256", "sha256", "video_sha256"):
                    if row.get(hk):
                        content_hash_by_id[sid] = str(row[hk])
                        break
        raw_counts["metadata_rows_mapped"] = len(signer_by_id)
        raw_counts["metadata_has_path_column"] = any(bool(p) for p in list(path_by_id.values())[:100])
        # empty signer_id rate on scientific splits
        empty = sum(1 for sid in (sets["train"] | sets["val"] | sets["test"]) if not signer_by_id.get(sid))
        if empty:
            findings.append(
                {
                    "id": "LEAK-EMPTY-SIGNER-ON-SPLIT-IDS",
                    "severity": "P1",
                    "claim_type": "FACT",
                    "n_empty": empty,
                }
            )
        # cross-check declared pool signers in split['signers']
        declared = split.get("signers") or {}
        if isinstance(declared, dict) and declared.get("test"):
            observed_test = sorted({signer_by_id[i] for i in sets["test"] if signer_by_id.get(i)})
            expected_test = sorted(declared.get("test") or (split.get("split_algorithm") or {}).get("fixed_test_signers") or [])
            raw_counts["declared_vs_metadata_test_signers"] = {
                "expected": expected_test,
                "observed": observed_test,
                "match": observed_test == expected_test,
            }

    signer_sets = {}
    if signer_by_id:
        for name, ids in sets.items():
            signer_sets[name] = {signer_by_id.get(i, "") for i in ids if signer_by_id.get(i)}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test"), ("confirm_train", "test"), ("hpo_train", "test")]:
            inter = sorted((signer_sets.get(a) or set()) & (signer_sets.get(b) or set()) - {""})
            overlaps[f"signer_{a}_x_{b}"] = {"n": len(inter), "signers": inter}
            # test is signer-disjoint by design vs train/val
            if b == "test" and a in {"train", "confirm_train", "hpo_train", "val"} and inter:
                # val may share? Protocol: fixed test signers — train/val should not include them
                findings.append(
                    {
                        "id": f"LEAK-SIGNER-OVERLAP-{a}-{b}",
                        "severity": "P0",
                        "claim_type": "FACT",
                        "signers": inter,
                    }
                )

        # verify fixed test signers
        fixed = set((split.get("split_algorithm") or {}).get("fixed_test_signers") or [])
        test_signers = signer_sets.get("test") or set()
        if fixed and test_signers != fixed:
            findings.append(
                {
                    "id": "LEAK-TEST-SIGNER-SET-MISMATCH",
                    "severity": "P1",
                    "claim_type": "FACT",
                    "expected": sorted(fixed),
                    "observed": sorted(test_signers),
                }
            )

    # duplicate paths across splits
    if path_by_id:
        path_to_ids = defaultdict(list)
        for sid, pth in path_by_id.items():
            if pth:
                path_to_ids[pth].append(sid)
        multi = {p: ids for p, ids in path_to_ids.items() if len(ids) > 1}
        cross_split_path = []
        for pth, ids in multi.items():
            membership = {name: sorted(set(ids) & sets[name]) for name in ("train", "val", "test", "hpo_train", "tune")}
            splits_hit = [n for n, v in membership.items() if v]
            if len([n for n in splits_hit if n in {"train", "val", "test", "hpo_train", "tune"}]) >= 2:
                # same path in disjoint scientific splits
                if ("val" in splits_hit and "test" in splits_hit) or (
                    "test" in splits_hit and any(x in splits_hit for x in ("train", "hpo_train", "confirm_train"))
                ):
                    cross_split_path.append({"path": pth, "membership": membership})
        overlaps["duplicate_paths_total"] = len(multi)
        overlaps["cross_split_paths_critical"] = {"n": len(cross_split_path), "examples": cross_split_path[:10]}
        if cross_split_path:
            findings.append(
                {
                    "id": "LEAK-PATH-CROSS-SPLIT",
                    "severity": "P0",
                    "claim_type": "FACT",
                    "n": len(cross_split_path),
                    "examples": cross_split_path[:10],
                }
            )

    if content_hash_by_id:
        hash_to_ids = defaultdict(list)
        for sid, h in content_hash_by_id.items():
            hash_to_ids[h].append(sid)
        # content hash shared across train/test
        cross = []
        for h, ids in hash_to_ids.items():
            if len(ids) < 2:
                continue
            if (set(ids) & sets["test"]) and (set(ids) & (sets["train"] | sets["confirm_train"] | sets["hpo_train"] | sets["val"])):
                cross.append({"hash": h, "ids": ids[:10]})
        overlaps["content_hash_cross_test"] = {"n": len(cross), "examples": cross[:10]}
        if cross:
            findings.append(
                {
                    "id": "LEAK-CONTENT-HASH-CROSS-TEST",
                    "severity": "P0",
                    "claim_type": "FACT",
                    "n": len(cross),
                }
            )
    else:
        findings.append(
            {
                "id": "LEAK-CONTENT-HASH-ABSENT",
                "severity": "P2",
                "claim_type": "SOURCE_GAP",
                "detail": "No content_sha256 on eligible samples; path/id/signer checks still apply",
            }
        )

    # labels consistency
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        raw_counts["labels_sha256_computed"] = sha256_file(labels_path)
        raw_counts["labels_sha256_expected"] = split.get("labels_sha256")
        raw_counts["labels_sha_match"] = raw_counts["labels_sha256_computed"] == split.get("labels_sha256")
        raw_counts["n_labels"] = len(labels) if isinstance(labels, (dict, list)) else None
    else:
        gaps.append("data/processed/vsl400_labels.json")

    # E4 did not open locked test
    e4_test_opened = bool(e4.get("test_opened"))
    if e4_test_opened:
        findings.append(
            {
                "id": "LEAK-E4-TEST-OPENED",
                "severity": "P0",
                "claim_type": "FACT",
                "detail": "E4_FINAL claims test_opened=true",
            }
        )
    if reg.get("e4_ready") is not False:
        findings.append(
            {
                "id": "LEAK-E4-READY-NOT-FALSE",
                "severity": "P0",
                "claim_type": "FACT",
                "e4_ready": reg.get("e4_ready"),
            }
        )

    # HPO vs test: ensure test ids not in hpo/tune
    # already covered by overlaps

    p0 = [f for f in findings if f.get("severity") == "P0"]
    p1 = [f for f in findings if f.get("severity") == "P1" and f.get("claim_type") != "SOURCE_GAP"]
    status = "PASS" if not p0 and not p1 and not (gaps and any(f.get("id") == "LEAK-ELIGIBLE-MANIFEST-MISSING" for f in findings) and not signer_by_id) else (
        "SOURCE_GAP" if gaps and not p0 else ("FAIL" if p0 or p1 else "PASS")
    )
    # refine: if no P0 and signer checks ran OK, PASS even with P2 content-hash gap
    if not p0 and not [f for f in findings if f.get("severity") == "P1" and f.get("claim_type") == "FACT"]:
        if signer_by_id or "LEAK-ELIGIBLE-MANIFEST-MISSING" not in [f["id"] for f in findings]:
            status = "PASS" if not p0 else status
        elif not signer_by_id:
            status = "SOURCE_GAP"

    # Better status logic:
    blocking = [f for f in findings if f.get("severity") in {"P0", "P1"} and f.get("claim_type") == "FACT"]
    source_gaps = [f for f in findings if f.get("claim_type") == "SOURCE_GAP" and f.get("severity") in {"P0", "P1"}]
    if blocking:
        status = "FAIL"
    elif source_gaps:
        status = "SOURCE_GAP"
    else:
        status = "PASS"

    audit = {
        "schema": "predefense_data_split_independent_audit_v1",
        "created_at": NOW,
        "status": status,
        "split_manifest": split_rel,
        "split_manifest_sha256": computed_split_sha,
        "id_list_sha_checks": sha_checks,
        "overlaps": overlaps,
        "signer_sets": {k: sorted(v) for k, v in signer_sets.items()} if signer_sets else {},
        "e4_test_opened": e4_test_opened,
        "e4_ready": reg.get("e4_ready"),
        "registry_active_variant": reg.get("active_variant"),
        "gaps": gaps,
        "notes": [
            "Did not read locked test labels beyond IDs required for leakage checks",
            "Augmentation/cache overwrite across splits not re-executed live (feature roots are sealed paths in registry)",
        ],
    }
    atomic_write_json(OUT / "DATA_SPLIT_RAW_COUNTS.json", raw_counts)
    atomic_write_json(OUT / "DATA_SPLIT_INDEPENDENT_AUDIT.json", audit)
    atomic_write_json(
        OUT / "LEAKAGE_FINDINGS.json",
        {
            "schema": "predefense_leakage_findings_v1",
            "created_at": NOW,
            "status": status,
            "findings": findings,
            "blocking_count": len(blocking),
            "source_gap_count": len(source_gaps),
        },
    )
    print(json.dumps({"status": status, "findings": len(findings), "gaps": gaps}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
