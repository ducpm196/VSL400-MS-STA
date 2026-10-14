#!/usr/bin/env python3
"""Independent Stage-A audit for b4_v6_ctr anomaly (validation only).

Recomputes metrics from VAL_PREDS; verifies split/feature/checkpoint SHAs;
scans train–val exact duplicates by source_sha256 / output_sha256.
Does not trust SUMMARY.json as sole evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.eval.ms_sta_v5_m0_strong_baseline import (  # noqa: E402
    hierarchical_bootstrap_ci,
    load_json,
    PredRow,
    sha256_file,
)


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def load_pred_rows(path: Path) -> dict[str, PredRow]:
    obj = load_json(path)
    out: dict[str, PredRow] = {}
    for r in obj["rows"]:
        sid = str(r["sample_id"])
        out[sid] = PredRow(
            sample_id=sid,
            signer=str(r.get("signer") or ""),
            label=int(r["label"]),
            correct=bool(r["correct"]),
            top5_hit=bool(r["top5_hit"]),
        )
    return out


def recompute_from_preds(preds: dict[str, PredRow]) -> dict[str, float]:
    n = max(len(preds), 1)
    top1 = sum(1 for p in preds.values() if p.correct) / n
    top5 = sum(1 for p in preds.values() if p.top5_hit) / n
    # Macro-F1 needs per-class; approximate from correct/label only if full preds lack scores.
    # Prefer exact agreement with stored fields; F1 left as SOURCE_GAP unless labels allow.
    by_label: dict[int, list[bool]] = defaultdict(list)
    for p in preds.values():
        by_label[int(p.label)].append(bool(p.correct))
    f1s = []
    for hits in by_label.values():
        # Without FP/FN from full confusion, use per-class recall as proxy only when all predicted.
        # Here VAL_PREDS stores correct/top5 only — report recall_macro as diagnostic, not Macro-F1.
        f1s.append(sum(hits) / max(len(hits), 1))
    recall_macro = sum(f1s) / max(len(f1s), 1)
    return {"top1": top1, "top5": top5, "recall_macro_proxy": recall_macro, "n": float(n)}


def load_sample_manifest(
    root: Path, modality: str, needed_ids: set[str]
) -> tuple[dict[str, dict[str, str]], int]:
    # Prefer compact INDEX.json (full corpus); fall back to per-sample files.
    sub = "skeleton_v3" if modality == "skel" else "rgb_token_v3"
    index_path = root / "runtime" / "manifests" / sub / "INDEX.json"
    out: dict[str, dict[str, str]] = {}
    if index_path.is_file():
        index = load_json(index_path)
        records = index.get("records") or {}
        missing = 0
        for sid in needed_ids:
            obj = records.get(sid)
            if obj is None:
                missing += 1
                continue
            out[sid] = {
                "source_sha256": str(obj.get("source_sha256") or ""),
                "output_sha256": str(obj.get("output_sha256") or ""),
                "signer_id": str(obj.get("signer_id") or ""),
                "split": str(obj.get("split") or ""),
            }
        return out, missing

    sample_dir = root / "runtime" / "manifests" / sub / "samples"
    if not sample_dir.is_dir():
        return out, len(needed_ids)
    missing = 0
    for sid in needed_ids:
        p = sample_dir / f"{sid}.json"
        if not p.is_file():
            missing += 1
            continue
        obj = load_json(p)
        out[sid] = {
            "source_sha256": str(obj.get("source_sha256") or ""),
            "output_sha256": str(obj.get("output_sha256") or ""),
            "signer_id": str(obj.get("signer_id") or ""),
            "split": str(obj.get("split") or ""),
        }
    return out, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="Docs/results/ms_sta_v6_ctr_matched_policy/CTR_MATCHED_POLICY.json")
    ap.add_argument(
        "--out",
        default="",
        help="Optional output JSON path (default: <audit_out_rel>/INDEPENDENT_AUDIT.json)",
    )
    args = ap.parse_args()
    policy = load_json(_ROOT / args.policy)
    out_dir = _ROOT / policy["audit_out_rel"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "INDEPENDENT_AUDIT.json"
    if not out_path.is_absolute():
        out_path = _ROOT / out_path

    split_path = _ROOT / policy["split_manifest_rel"]
    split_sha = sha256_file(split_path)
    split = load_json(split_path)
    labels_path = _ROOT / "data/processed/vsl400_labels.json"
    labels = load_json(labels_path)
    labels_sha = sha256_file(labels_path)

    skel_root = _ROOT / policy["feature_roots"]["skeleton_root_rel"]
    rgb_root = _ROOT / policy["feature_roots"]["rgb_root_rel"]
    role_ids = {
        "confirm_train": [str(x) for x in split["confirm_train_ids"]],
        "val": [str(x) for x in split["val_ids"]],
        "tune": [str(x) for x in split["tune_ids"]],
        "hpo_train": [str(x) for x in split["hpo_train_ids"]],
        "test": [str(x) for x in split["test_ids"]],
    }
    needed_dup = set(role_ids["confirm_train"]) | set(role_ids["val"])
    # Coverage universe for feature-manifest completeness (includes sealed test holdout IDs).
    needed_coverage = needed_dup | set(role_ids["test"]) | set(role_ids["tune"]) | set(role_ids["hpo_train"])
    skel_man, skel_missing = load_sample_manifest(skel_root, "skel", needed_coverage)
    rgb_man, rgb_missing = load_sample_manifest(rgb_root, "rgb", needed_coverage)

    # Role disjointness (critical)
    overlaps = {}
    pairs = [("confirm_train", "val"), ("hpo_train", "tune"), ("confirm_train", "test"), ("val", "test")]
    for a, b in pairs:
        sa, sb = set(role_ids[a]), set(role_ids[b])
        overlaps[f"{a}_x_{b}"] = sorted(sa & sb)[:20]
        overlaps[f"{a}_x_{b}_n"] = len(sa & sb)

    # Signer disjointness train vs val if claimed
    meta_csv = _ROOT / "metadata.csv"
    import csv

    signer_by_id: dict[str, str] = {}
    gloss_by_id: dict[str, str] = {}
    with meta_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row["video_id"]).zfill(6) if row["video_id"].isdigit() else str(row["video_id"])
            # Prefer zero-pad 6 if numeric
            if row["video_id"].isdigit():
                sid = f"{int(row['video_id']):06d}"
            signer_by_id[sid] = str(row["signer_id"])
            gloss_by_id[sid] = str(row.get("text_gloss") or "")

    train_signers = {signer_by_id.get(i, "?") for i in role_ids["confirm_train"]}
    val_signers = {signer_by_id.get(i, "?") for i in role_ids["val"]}
    sealed = set(str(x) for x in (split.get("sealed_holdout_signers") or []))

    # Duplicate scan: source_sha / output_sha shared across confirm_train and val
    def dup_report(man: dict[str, dict[str, str]], key: str) -> dict[str, Any]:
        train_set = set(role_ids["confirm_train"])
        val_set = set(role_ids["val"])
        by_hash: dict[str, list[str]] = defaultdict(list)
        for sid, row in man.items():
            h = row.get(key) or ""
            if not h:
                continue
            by_hash[h].append(sid)
        exact_cross = []
        for h, ids in by_hash.items():
            t = [i for i in ids if i in train_set]
            v = [i for i in ids if i in val_set]
            if t and v:
                exact_cross.append({"hash": h, "train_ids": t[:10], "val_ids": v[:10], "n_train": len(t), "n_val": len(v)})
        return {"key": key, "n_cross_hash_groups": len(exact_cross), "examples": exact_cross[:25]}

    skel_src_dup = dup_report(skel_man, "source_sha256")
    skel_out_dup = dup_report(skel_man, "output_sha256")
    rgb_src_dup = dup_report(rgb_man, "source_sha256")
    rgb_out_dup = dup_report(rgb_man, "output_sha256")

    # Checkpoint audit + VAL_PREDS recompute
    lock = load_json(_ROOT / "Docs/results/ms_sta_v6/V6_TEACHER_LOCK.json")
    confirm_root = _ROOT / policy["candidate_confirm_root_rel"]
    seeds = [int(s) for s in policy["seeds_confirm"]]
    variants = ["b4_v6_ctr", "b3_mlp_strong", "rgb_strong"]
    recompute: dict[str, Any] = {}
    ckpt_checks: dict[str, Any] = {}
    for variant in variants:
        recompute[variant] = {}
        for seed in seeds:
            preds_path = confirm_root / f"{variant}_seed{seed}" / "VAL_PREDS.json"
            summary_path = confirm_root / f"{variant}_seed{seed}" / "SUMMARY.json"
            ckpt_path = confirm_root / f"{variant}_seed{seed}" / "best.pt"
            entry: dict[str, Any] = {"preds_exists": preds_path.is_file(), "summary_exists": summary_path.is_file()}
            if preds_path.is_file():
                preds = load_pred_rows(preds_path)
                metrics = recompute_from_preds(preds)
                entry["recomputed"] = metrics
                if summary_path.is_file():
                    summary = load_json(summary_path)
                    entry["summary_top1"] = float(summary.get("best_val_top1") or 0.0)
                    entry["summary_top5"] = float(summary.get("best_val_top5") or 0.0)
                    entry["top1_abs_diff"] = abs(metrics["top1"] - entry["summary_top1"])
                    entry["top5_abs_diff"] = abs(metrics["top5"] - entry["summary_top5"])
                # signer distribution
                entry["n_signers"] = len({p.signer for p in preds.values()})
                entry["signer_counts"] = dict(Counter(p.signer for p in preds.values()))
            if ckpt_path.is_file():
                observed = sha256_file(ckpt_path)
                entry["checkpoint_sha256"] = observed
                expected = ((lock.get("checkpoints") or {}).get(str(seed)) or {}).get("sha256") if variant == "b4_v6_ctr" else None
                if expected:
                    entry["checkpoint_sha_match_lock"] = observed == expected
            recompute[variant][str(seed)] = entry
            ckpt_checks[f"{variant}_seed{seed}"] = entry.get("checkpoint_sha_match_lock")

    # Paired Δ from VAL_PREDS only (candidate vs old mlp — reported as mismatched baseline)
    mismatched_gate = None
    if all(recompute["b4_v6_ctr"][str(s)].get("preds_exists") and recompute["b3_mlp_strong"][str(s)].get("preds_exists") for s in seeds):
        panels_t = {s: load_pred_rows(confirm_root / f"b4_v6_ctr_seed{s}" / "VAL_PREDS.json") for s in seeds}
        panels_b = {s: load_pred_rows(confirm_root / f"b3_mlp_strong_seed{s}" / "VAL_PREDS.json") for s in seeds}
        ids = sorted(panels_t[seeds[0]].keys())
        d1, signers = [], []
        for sid in ids:
            vals = [float(panels_t[s][sid].correct) - float(panels_b[s][sid].correct) for s in seeds]
            d1.append(sum(vals) / len(vals))
            signers.append(panels_t[seeds[0]][sid].signer)
        mean_pp, lo, hi = hierarchical_bootstrap_ci(d1, signers, n_boot=10000, seed=42)
        t_top1 = [recompute["b4_v6_ctr"][str(s)]["recomputed"]["top1"] for s in seeds]
        b_top1 = [recompute["b3_mlp_strong"][str(s)]["recomputed"]["top1"] for s in seeds]
        mismatched_gate = {
            "baseline": "b3_mlp_strong",
            "note": "ENCODER_MISMATCHED — JointGraphEncoder vs CTR; not architecture-sufficient",
            "candidate_top1_recomputed": t_top1,
            "baseline_top1_recomputed": b_top1,
            "mean_delta_top1_pp": 100.0 * (sum(t_top1[i] - b_top1[i] for i in range(3)) / 3.0),
            "hierarchical_ci95_top1_pp": [lo, hi],
            "mean_hierarchical_boot_pp": mean_pp,
        }

    report = {
        "schema": "ms_sta_v6_independent_audit_v1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": {
            "rel": policy["split_manifest_rel"],
            "sha256_observed": split_sha,
            "sha256_expected": policy["split_manifest_sha256"],
            "sha_match": split_sha == policy["split_manifest_sha256"],
            "n_roles": {k: len(v) for k, v in role_ids.items()},
            "role_id_overlaps": overlaps,
            "sealed_holdout_signers": sorted(sealed),
            "sealed_holdout_status": (
                "NOT_APPLICABLE_NO_SEALED_HOLDOUT" if not sealed else "PRESENT"
            ),
            "confirm_train_signers_n": len(train_signers),
            "val_signers_n": len(val_signers),
            "train_val_signer_intersection": sorted(train_signers & val_signers),
            "train_val_signer_disjoint": len(train_signers & val_signers) == 0,
            "val_subset_of_sealed_holdout": (
                "NOT_APPLICABLE_NO_SEALED_HOLDOUT"
                if not sealed
                else bool(val_signers) and val_signers.issubset(sealed)
            ),
            "feature_coverage_universe_n": len(needed_coverage),
            "feature_dup_scan_universe_n": len(needed_dup),
        },
        "labels": {
            "rel": "data/processed/vsl400_labels.json",
            "sha256": labels_sha,
            "n_classes": len(labels.get("gloss_to_id") or {}),
            "split_labels_sha256": split.get("labels_sha256"),
        },
        "features": {
            "skeleton_root_rel": policy["feature_roots"]["skeleton_root_rel"],
            "rgb_root_rel": policy["feature_roots"]["rgb_root_rel"],
            "skel_manifest_n": len(skel_man),
            "rgb_manifest_n": len(rgb_man),
            "skel_manifest_missing_n": skel_missing,
            "rgb_manifest_missing_n": rgb_missing,
            "duplicates": {
                "skel_source_sha_train_val": skel_src_dup,
                "skel_output_sha_train_val": skel_out_dup,
                "rgb_source_sha_train_val": rgb_src_dup,
                "rgb_output_sha_train_val": rgb_out_dup,
            },
        },
        "checkpoints_and_preds": recompute,
        "mismatched_baseline_recompute": mismatched_gate,
        "language": {
            "canonical_teacher": False,
            "correct_phrase": "b4_v6_ctr is a Stage-A candidate with unusually high validation; not a canonical teacher or defense model.",
            "retain_defense": "b4_star_fixed",
        },
        "verdicts": {},
    }

    # Verdicts
    leak_flags = []
    if overlaps["confirm_train_x_val_n"]:
        leak_flags.append("id_overlap_confirm_train_val")
    if skel_src_dup["n_cross_hash_groups"] or rgb_src_dup["n_cross_hash_groups"]:
        leak_flags.append("source_sha_cross_train_val")
    if skel_out_dup["n_cross_hash_groups"] or rgb_out_dup["n_cross_hash_groups"]:
        leak_flags.append("output_sha_cross_train_val")
    if train_signers & val_signers:
        leak_flags.append("signer_overlap_confirm_train_val")

    summary_mismatch = []
    for seed in seeds:
        e = recompute["b4_v6_ctr"][str(seed)]
        if e.get("top1_abs_diff") is not None and e["top1_abs_diff"] > 1e-9:
            summary_mismatch.append({"seed": seed, "top1_abs_diff": e["top1_abs_diff"]})

    report["verdicts"] = {
        "split_sha_ok": report["split"]["sha_match"],
        "no_train_val_id_overlap": overlaps["confirm_train_x_val_n"] == 0,
        "no_source_or_output_sha_leak": len(leak_flags) == 0 or leak_flags == ["signer_overlap_confirm_train_val"],
        "leak_flags": leak_flags,
        "val_preds_agree_summary_top1": len(summary_mismatch) == 0,
        "summary_mismatch_details": summary_mismatch,
        "ckpt_lock_match_b4_v6_ctr": all(
            recompute["b4_v6_ctr"][str(s)].get("checkpoint_sha_match_lock") is True for s in seeds
        ),
        "anomaly_absolute_top1_still_present": bool(
            mismatched_gate and max(mismatched_gate["candidate_top1_recomputed"]) > 0.75
        ),
        "architecture_claim_allowed": False,
        "architecture_claim_blocked_reason": "Companion provenance only; CTR-matched architecture gate is recorded separately.",
        "train_val_signer_disjoint": len(train_signers & val_signers) == 0,
        "sealed_holdout_status": report["split"]["sealed_holdout_status"],
    }

    report["audit_edition"] = {
        "coverage_universe": "confirm_train ∪ val ∪ tune ∪ hpo_train ∪ test",
        "dup_scan_universe": "confirm_train × val",
        "feature_coverage_universe_n": len(needed_coverage),
        "feature_dup_scan_universe_n": len(needed_dup),
    }

    sha = atomic_write(out_path, report)
    print(json.dumps({"event": "audit_written", "path": str(out_path.relative_to(_ROOT)).replace('\\', '/'), "sha256": sha, "leak_flags": leak_flags}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
