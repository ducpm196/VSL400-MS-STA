# -*- coding: utf-8 -*-
"""Independent metric recalculation from raw B4 prediction dumps.

Does NOT import historical summarizers. Writes:
  Docs/results/project_predefense_audit/METRIC_INDEPENDENT_RECALC_V2.json
  Docs/results/project_predefense_audit/METRIC_RECALC_RAW.log
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs" / "results" / "project_predefense_audit"
TOL = 1e-9  # locked absolute tolerance for rate comparisons


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


def macro_f1(y_true: list[int], y_pred: list[int], num_classes: int = 400) -> float:
    """Unweighted mean of per-class F1; classes with support 0 contribute 0."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1
    scores: list[float] = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        scores.append((2 * tp[c] / denom) if denom else 0.0)
    return sum(scores) / num_classes


def mcnemar_discordant(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    a_only = b_only = 0
    for ac, bc in zip(a_correct, b_correct):
        if ac and not bc:
            a_only += 1
        elif bc and not ac:
            b_only += 1
    n = a_only + b_only
    # exact two-sided binomial under p=0.5 (no SciPy). For large n use erf normal approx.
    if n == 0:
        p: float | None = 1.0
        method = "exact_binomial_two_sided_mcnemar_independent"
    elif n <= 1000:
        from math import comb

        k = min(a_only, b_only)
        tail = sum(comb(n, i) for i in range(0, k + 1))
        p = min(1.0, 2.0 * tail / float(2**n))
        method = "exact_binomial_two_sided_mcnemar_independent"
    else:
        # continuity-corrected normal approx for Bin(n,0.5)
        from math import erfc, sqrt

        z = (abs(a_only - b_only) - 1.0) / sqrt(float(n))
        p = max(0.0, min(1.0, erfc(z / sqrt(2.0))))  # two-sided via erfc
        method = "normal_approx_two_sided_mcnemar_independent_n_gt_1000"
    return {
        "primary_only": a_only,
        "baseline_only": b_only,
        "n_discordant": n,
        "exact_two_sided_p": p if n <= 1000 else None,
        "approx_two_sided_p": p if n > 1000 else None,
        "method": method,
    }


def recalc_test_aligned(log: list[str]) -> dict[str, Any]:
    pred_path = ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_PREDS_ALIGNED.jsonl"
    report_path = ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_REPORT.json"
    final_path = ROOT / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json"
    split_path = ROOT / "data/processed/vsl400_retrain_v4_split_manifest.json"
    reg = json.loads((ROOT / "data/processed/ms_sta_v6_defense_go.json").read_text(encoding="utf-8")) if False else None
    # registry path
    reg = json.loads((ROOT / "data/processed/ms_sta_registry_v6_defense_go.json").read_text(encoding="utf-8"))

    if not pred_path.exists():
        return {"status": "SOURCE_GAP", "reason": "TEST_PREDS_ALIGNED.jsonl missing", "path": str(pred_path)}

    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    log.append(f"loaded_aligned_rows={len(rows)}")
    sample_ids = [r["sample_id"] for r in rows]
    dup = len(sample_ids) - len(set(sample_ids))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = set(split["test_ids"])
    coverage_missing = sorted(test_ids - set(sample_ids))
    coverage_extra = sorted(set(sample_ids) - test_ids)

    # signer field: aligned dump uses signer_id
    signer_keys = set()
    for r in rows:
        if "signer_id" in r:
            signer_keys.add("signer_id")
        if "signer" in r:
            signer_keys.add("signer")

    variants = ["b4_v6_ctr", "b3_mlp_ctr", "b3_ctr_concat", "rgb_strong"]
    metrics: dict[str, Any] = {}
    for v in variants:
        y_true: list[int] = []
        y_pred: list[int] = []
        top5_hits = 0
        per_signer: dict[str, list[bool]] = defaultdict(list)
        ckpts: set[str] = set()
        for r in rows:
            m = r["models"][v]
            yt = int(r["label"])
            yp = int(m["pred"])
            y_true.append(yt)
            y_pred.append(yp)
            top5_hits += int(bool(m.get("top5_hit")))
            sid = r.get("signer_id") or r.get("signer") or "UNKNOWN"
            per_signer[str(sid)].append(yt == yp)
            if m.get("checkpoint_sha256"):
                ckpts.add(m["checkpoint_sha256"])
        n = len(y_true)
        top1 = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n
        top5 = top5_hits / n
        mf1 = macro_f1(y_true, y_pred, 400)
        signer_top1 = {s: (sum(xs) / len(xs)) for s, xs in sorted(per_signer.items())}
        metrics[v] = {
            "n": n,
            "top1": top1,
            "top5": top5,
            "macro_f1": mf1,
            "per_signer_top1": signer_top1,
            "checkpoint_sha256_set": sorted(ckpts),
        }

    # paired B4 vs MLP
    b4_corr = [r["models"]["b4_v6_ctr"]["pred"] == r["label"] for r in rows]
    mlp_corr = [r["models"]["b3_mlp_ctr"]["pred"] == r["label"] for r in rows]
    delta_top1 = metrics["b4_v6_ctr"]["top1"] - metrics["b3_mlp_ctr"]["top1"]
    mcn = mcnemar_discordant(b4_corr, mlp_corr)

    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}

    def near(a: float, b: float) -> bool:
        return abs(float(a) - float(b)) <= TOL

    comparisons = []
    for v in variants:
        if v in report.get("variants", {}):
            rv = report["variants"][v]
            comparisons.append(
                {
                    "source": "TEST_REPORT.json",
                    "variant": v,
                    "top1_match": near(metrics[v]["top1"], rv["top1"]),
                    "top5_match": near(metrics[v]["top5"], rv["top5"]),
                    "macro_f1_match": near(metrics[v]["macro_f1"], rv["macro_f1"]),
                    "recalc": {k: metrics[v][k] for k in ("top1", "top5", "macro_f1", "n")},
                    "reported": {k: rv[k] for k in ("top1", "top5", "macro_f1", "n")},
                }
            )
    if final.get("test"):
        ft = final["test"]
        comparisons.append(
            {
                "source": "FINAL_RESULTS.json.test",
                "b4_top1_match": near(metrics["b4_v6_ctr"]["top1"], ft["b4_top1"]),
                "b4_top5_match": near(metrics["b4_v6_ctr"]["top5"], ft["b4_top5"]),
                "b4_macro_f1_match": near(metrics["b4_v6_ctr"]["macro_f1"], ft["b4_macro_f1"]),
                "delta_top1_pp_match": near(delta_top1 * 100.0, ft["delta_top1_pp"]),
            }
        )

    ckpt_ok = reg["checkpoint_sha256"] in metrics["b4_v6_ctr"]["checkpoint_sha256_set"]
    all_match = all(
        c.get("top1_match", True)
        and c.get("top5_match", True)
        and c.get("macro_f1_match", True)
        and c.get("b4_top1_match", True)
        and c.get("b4_top5_match", True)
        and c.get("b4_macro_f1_match", True)
        and c.get("delta_top1_pp_match", True)
        for c in comparisons
    )
    status = "PASS" if (
        dup == 0
        and not coverage_missing
        and not coverage_extra
        and len(rows) == 6165
        and ckpt_ok
        and all_match
        and "signer_id" in signer_keys
    ) else "FAIL"

    return {
        "split": "test_oneshot_aligned",
        "status": status,
        "source_path": str(pred_path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(pred_path),
        "n_rows": len(rows),
        "duplicate_sample_ids": dup,
        "coverage_missing_vs_split": len(coverage_missing),
        "coverage_extra_vs_split": len(coverage_extra),
        "signer_field_keys_observed": sorted(signer_keys),
        "metrics": metrics,
        "paired_b4_vs_b3_mlp_ctr": {
            "delta_top1": delta_top1,
            "delta_top1_pp": delta_top1 * 100.0,
            "mcnemar": mcn,
        },
        "checkpoint_sha_matches_registry": ckpt_ok,
        "comparisons": comparisons,
        "tolerance": TOL,
    }


def recalc_val_export(log: list[str]) -> dict[str, Any]:
    man_path = ROOT / "Docs/results/ms_sta_v6_e4_r2/B4_CONFIRM_VAL_PREDS_EXPORT_MANIFEST.json"
    if not man_path.exists():
        return {"status": "SOURCE_GAP", "reason": "VAL_PREDS export manifest missing"}
    man = json.loads(man_path.read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))
    val_ids = set(split["val_ids"])
    seeds_out = []
    for fmeta in man["files"]:
        path = ROOT / fmeta["path_rel"]
        computed = sha256_file(path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        rows = blob["rows"]
        sample_ids = [r["sample_id"] for r in rows]
        dup = len(sample_ids) - len(set(sample_ids))
        # field: 'signer' in this schema (historical key) — flag as note
        signer_key = "signer_id" if rows and "signer_id" in rows[0] else ("signer" if rows and "signer" in rows[0] else None)
        y_true = [int(r["label"]) for r in rows]
        y_pred = [int(r["pred"]) for r in rows]
        top1 = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(rows)
        top5 = sum(int(bool(r.get("top5_hit"))) for r in rows) / len(rows)
        mf1 = macro_f1(y_true, y_pred, 400)
        per_signer: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            sid = str(r.get("signer_id") or r.get("signer") or "UNKNOWN")
            per_signer[sid].append(int(r["label"]) == int(r["pred"]))
        seeds_out.append(
            {
                "seed": fmeta["seed"],
                "path": fmeta["path_rel"],
                "sha_match": computed == fmeta["sha256"],
                "computed_sha256": computed,
                "n": len(rows),
                "duplicate_sample_ids": dup,
                "coverage_missing": len(val_ids - set(sample_ids)),
                "coverage_extra": len(set(sample_ids) - val_ids),
                "signer_field": signer_key,
                "recalc_top1": top1,
                "recalc_top5": top5,
                "recalc_macro_f1": mf1,
                "reported_top1": blob.get("top1"),
                "reported_top5": blob.get("top5"),
                "reported_macro_f1": blob.get("macro_f1"),
                "top1_match": abs(top1 - float(blob["top1"])) <= TOL,
                "top5_match": abs(top5 - float(blob["top5"])) <= TOL,
                "macro_f1_match": abs(mf1 - float(blob["macro_f1"])) <= TOL,
                "checkpoint_sha256": blob.get("checkpoint_sha256"),
                "per_signer_top1": {s: sum(xs) / len(xs) for s, xs in sorted(per_signer.items())},
            }
        )
        log.append(f"val_seed={fmeta['seed']} n={len(rows)} top1={top1:.10f} sha_match={computed == fmeta['sha256']}")

    ok = all(
        s["sha_match"]
        and s["duplicate_sample_ids"] == 0
        and s["coverage_missing"] == 0
        and s["coverage_extra"] == 0
        and s["top1_match"]
        and s["top5_match"]
        and s["macro_f1_match"]
        for s in seeds_out
    )
    return {
        "split": "confirm_val_export",
        "status": "PASS" if ok else "FAIL",
        "manifest_sha256": sha256_file(man_path),
        "n_val_required": man.get("n_val_required"),
        "seeds": seeds_out,
        "note_signer_field": "VAL_PREDS schema uses key 'signer' (not signer_id); values still populated",
        "tolerance": TOL,
    }


def main() -> int:
    NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log: list[str] = [f"started={NOW}", f"python={sys.executable}"]
    test = recalc_test_aligned(log)
    val = recalc_val_export(log)
    overall = "PASS" if test.get("status") == "PASS" and val.get("status") == "PASS" else (
        "SOURCE_GAP" if "SOURCE_GAP" in (test.get("status"), val.get("status")) else "FAIL"
    )
    out = {
        "schema": "predefense_metric_independent_recalc_v2",
        "created_at": NOW,
        "method": "raw_dump_recompute_no_summarizer_import",
        "overall_status": overall,
        "test": test,
        "validation": val,
        "claims_covered": [
            "B4 test Top-1/Top-5/Macro-F1 vs FINAL_RESULTS/TEST_REPORT",
            "B4 confirm val Top-1/Top-5/Macro-F1 vs VAL_PREDS export",
            "paired delta B4 vs b3_mlp_ctr on test",
            "checkpoint SHA provenance on prediction rows",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "METRIC_INDEPENDENT_RECALC_V2.json", out)
    (OUT / "METRIC_RECALC_RAW.log").write_text("\n".join(log) + "\n" + json.dumps({"overall": overall}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": overall, "test": test.get("status"), "val": val.get("status")}, indent=2))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
