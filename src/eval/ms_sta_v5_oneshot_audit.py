#!/usr/bin/env python3
"""Read-only audit of MS-STA v5 one-shot TEST_PREDS (no re-inference)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.ms_sta_v5_m0_strong_baseline import hierarchical_bootstrap_ci
from src.training.retrain_v3_stats import mcnemar

VARIANTS = ("rgb_fixed", "b3_star_fixed", "b3_mlp_roi_fixed", "b4_star_fixed")
PRIMARY = "b4_star_fixed"
BASELINE = "b3_mlp_roi_fixed"
EXPECTED_N = 6165
EXPECTED_SIGNERS = ("signer_24", "signer_25", "signer_26", "signer_27", "signer_28")
ONESHOT_REL = "Docs/results/ms_sta_v5_oneshot"


class OneshotAuditError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _macro_f1(labels: list[int], preds: list[int], num_classes: int = 400) -> float:
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for y, p in zip(labels, preds):
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    f1s: list[float] = []
    for c in range(num_classes):
        denom = int(2 * tp[c] + fp[c] + fn[c])
        if denom == 0:
            continue
        f1s.append(float(2 * tp[c] / denom))
    return float(sum(f1s) / max(len(f1s), 1))


def load_preds_strict(path: Path) -> dict[str, dict[str, Any]]:
    obj = load_json(path)
    rows = obj.get("rows")
    if not isinstance(rows, list):
        raise OneshotAuditError(f"rows_missing:{path}")
    out: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        sid = str(r.get("sample_id") or "")
        if not sid:
            raise OneshotAuditError(f"empty_sample_id:{path}:{i}")
        if sid in out:
            raise OneshotAuditError(f"duplicate_sample_id:{path}:{sid}")
        top5 = r.get("top5")
        if not isinstance(top5, list) or len(top5) != 5:
            raise OneshotAuditError(f"top5_invalid:{path}:{sid}")
        label = int(r["label"])
        pred = int(r["pred"])
        correct = bool(r["correct"])
        top5_hit = bool(r["top5_hit"])
        if correct != (pred == label):
            raise OneshotAuditError(f"correct_flag_mismatch:{path}:{sid}")
        if top5_hit != (label in top5):
            raise OneshotAuditError(f"top5_flag_mismatch:{path}:{sid}")
        signer = str(r.get("signer") or "")
        if not signer:
            raise OneshotAuditError(f"signer_empty:{path}:{sid}")
        out[sid] = {
            "sample_id": sid,
            "signer": signer,
            "label": label,
            "pred": pred,
            "correct": correct,
            "top5_hit": top5_hit,
            "top5": [int(x) for x in top5],
        }
    if len(out) != EXPECTED_N:
        raise OneshotAuditError(f"n_mismatch:{path}:{len(out)}")
    return out


def audit_oneshot(root: Path) -> dict[str, Any]:
    oneshot = root / ONESHOT_REL
    report_path = oneshot / "TEST_REPORT.json"
    freeze_path = oneshot / "PRETEST_FREEZE.json"
    auth_path = oneshot / "ONE_SHOT_TEST_AUTHORIZATION.json"
    if not report_path.is_file():
        raise OneshotAuditError("TEST_REPORT_missing")

    report = load_json(report_path)
    freeze = load_json(freeze_path)
    auth = load_json(auth_path)
    policy = load_json(root / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json")
    split_path = root / policy["split_manifest_rel"]
    split = load_json(split_path)
    expected_ids = [str(x) for x in split["test_ids"]]
    if len(expected_ids) != EXPECTED_N:
        raise OneshotAuditError(f"split_test_n:{len(expected_ids)}")
    if len(expected_ids) != len(set(expected_ids)):
        raise OneshotAuditError("split_test_ids_not_unique")

    panels: dict[str, dict[str, dict[str, Any]]] = {}
    pred_sha: dict[str, str] = {}
    for v in VARIANTS:
        rel = f"TEST_PREDS_{v}_seed42.json"
        path = oneshot / rel
        panels[v] = load_preds_strict(path)
        pred_sha[v] = sha256_file(path)
        if set(panels[v].keys()) != set(expected_ids):
            raise OneshotAuditError(f"id_set_mismatch:{v}")

    # Cross-panel alignment.
    for sid in expected_ids:
        signers = {panels[v][sid]["signer"] for v in VARIANTS}
        labels = {panels[v][sid]["label"] for v in VARIANTS}
        if len(signers) != 1:
            raise OneshotAuditError(f"signer_misaligned:{sid}:{signers}")
        if len(labels) != 1:
            raise OneshotAuditError(f"label_misaligned:{sid}:{labels}")

    signers_present = sorted({panels[PRIMARY][sid]["signer"] for sid in expected_ids})
    if tuple(signers_present) != EXPECTED_SIGNERS:
        raise OneshotAuditError(f"signers_mismatch:{signers_present}")

    # Checkpoint SHA bind.
    ckpt_sha: dict[str, str] = {}
    for v in VARIANTS:
        meta = freeze["checkpoints_seed42"][v]
        path = root / meta["checkpoint_rel"]
        digest = sha256_file(path)
        if digest != meta["checkpoint_sha256"]:
            raise OneshotAuditError(f"ckpt_sha_mismatch:{v}")
        ckpt_sha[v] = digest

    metrics: dict[str, dict[str, Any]] = {}
    for v in VARIANTS:
        rows = [panels[v][sid] for sid in expected_ids]
        labels = [r["label"] for r in rows]
        preds = [r["pred"] for r in rows]
        top1 = sum(1 for r in rows if r["correct"]) / EXPECTED_N
        top5 = sum(1 for r in rows if r["top5_hit"]) / EXPECTED_N
        f1 = _macro_f1(labels, preds)
        per_signer: dict[str, dict[str, float]] = {}
        by_s: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_s[r["signer"]].append(r)
        for s, xs in sorted(by_s.items()):
            n = len(xs)
            per_signer[s] = {
                "n": float(n),
                "top1": sum(1 for r in xs if r["correct"]) / n,
                "top5": sum(1 for r in xs if r["top5_hit"]) / n,
            }
        metrics[v] = {
            "top1": top1,
            "top5": top5,
            "macro_f1": f1,
            "n": EXPECTED_N,
            "per_signer": per_signer,
        }
        # Match locked report within float noise.
        locked = report["variants"][v]
        for key in ("top1", "top5", "macro_f1"):
            if abs(float(locked[key]) - float(metrics[v][key])) > 1e-12:
                raise OneshotAuditError(
                    f"report_metric_mismatch:{v}:{key}:{locked[key]}!={metrics[v][key]}"
                )

    # Paired B4 vs MLP.
    d1: list[float] = []
    d5: list[float] = []
    signer_list: list[str] = []
    b4_corr = []
    mlp_corr = []
    for sid in expected_ids:
        p = panels[PRIMARY][sid]
        b = panels[BASELINE][sid]
        d1.append(float(p["correct"]) - float(b["correct"]))
        d5.append(float(p["top5_hit"]) - float(b["top5_hit"]))
        signer_list.append(p["signer"])
        b4_corr.append(bool(p["correct"]))
        mlp_corr.append(bool(b["correct"]))

    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(d1, signer_list, n_boot=10000, seed=42)
    mean5_pp, ci5_lo, ci5_hi = hierarchical_bootstrap_ci(d5, signer_list, n_boot=10000, seed=42)

    # Per-signer absolute + deltas.
    per_signer_compare: dict[str, Any] = {}
    for s in EXPECTED_SIGNERS:
        b4s = metrics[PRIMARY]["per_signer"][s]
        mlps = metrics[BASELINE]["per_signer"][s]
        per_signer_compare[s] = {
            "b4_top1": b4s["top1"],
            "mlp_top1": mlps["top1"],
            "delta_top1_pp": 100.0 * (b4s["top1"] - mlps["top1"]),
            "b4_top5": b4s["top5"],
            "mlp_top5": mlps["top5"],
            "delta_top5_pp": 100.0 * (b4s["top5"] - mlps["top5"]),
            "n": b4s["n"],
        }

    # Explicit Top-5 regression callout (signer 27).
    s27_top5_delta_pp = per_signer_compare["signer_27"]["delta_top5_pp"]
    top1_all_positive = all(per_signer_compare[s]["delta_top1_pp"] > 0 for s in EXPECTED_SIGNERS)
    top5_all_positive = all(per_signer_compare[s]["delta_top5_pp"] > 0 for s in EXPECTED_SIGNERS)

    mc = mcnemar(np.asarray(mlp_corr, dtype=bool), np.asarray(b4_corr, dtype=bool))
    # mcnemar API: baseline_correct=MLP, other_correct=B4 → other_only = B4-only
    discordant = {
        "b4_only_correct": mc["other_only_correct"],
        "mlp_only_correct": mc["baseline_only_correct"],
        "n_discordant": mc["n_discordant"],
        "mcnemar_exact_p": mc["p_value_exact"],
        "method": mc["method"],
    }

    locked_delta = report["b4_minus_mlp"]
    if abs(locked_delta["hierarchical_mean_delta_top1_pp"] - mean_pp) > 1e-9:
        raise OneshotAuditError("hier_top1_mismatch_vs_report")
    if abs(locked_delta["hierarchical_ci95_top1_pp"][0] - ci_lo) > 1e-9:
        raise OneshotAuditError("hier_ci_lo_mismatch_vs_report")
    if abs(locked_delta["hierarchical_ci95_top1_pp"][1] - ci_hi) > 1e-9:
        raise OneshotAuditError("hier_ci_hi_mismatch_vs_report")

    return {
        "schema": "ms_sta_v5_oneshot_test_audit_v1",
        "status": "AUDIT_PASS",
        "no_reinference": True,
        "retrospective_signer_disjoint_confirmation": True,
        "pristine_blind_claim": False,
        "expected_n": EXPECTED_N,
        "unique_ids": True,
        "aligned_panels": True,
        "b4_e_loaded": False,
        "variants": list(VARIANTS),
        "metrics_recomputed": metrics,
        "b4_minus_mlp": {
            "delta_top1": metrics[PRIMARY]["top1"] - metrics[BASELINE]["top1"],
            "delta_top5": metrics[PRIMARY]["top5"] - metrics[BASELINE]["top5"],
            "delta_macro_f1": metrics[PRIMARY]["macro_f1"] - metrics[BASELINE]["macro_f1"],
            "hierarchical_mean_delta_top1_pp": mean_pp,
            "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
            "hierarchical_mean_delta_top5_pp": mean5_pp,
            "hierarchical_ci95_top5_pp": [ci5_lo, ci5_hi],
            "per_signer": per_signer_compare,
            "top1_improved_all_signers": top1_all_positive,
            "top5_improved_all_signers": top5_all_positive,
            "signer_27_top5_delta_pp": s27_top5_delta_pp,
            "discordant_top1": discordant,
        },
        "claim_discipline": {
            "allowed": (
                "On local Front-view protocol, B4* exceeds RGB-only, feature-concat B3*, "
                "and strong baseline B3+MLP. Gains are consistent across three validation "
                "seeds and confirmed retrospectively on the signer-disjoint test set "
                "(6165 videos)."
            ),
            "forbidden": [
                "pristine blind test",
                "every metric improved on every signer",
                "B4-E is the proposed or final model",
            ],
            "top5_not_uniform": (
                f"signer_27 Top-5 delta vs MLP = {s27_top5_delta_pp:.2f} pp "
                "(do not claim Top-5 rises on all signers)."
            ),
        },
        "bindings": {
            "test_report_sha256": sha256_file(report_path),
            "pretest_freeze_sha256": sha256_file(freeze_path),
            "oneshot_auth_sha256": sha256_file(auth_path),
            "split_manifest_sha256": sha256_file(split_path),
            "prediction_sha256": pred_sha,
            "checkpoint_sha256": ckpt_sha,
            "validation_lock_v2_sha256": sha256_file(
                root / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json"
            ),
            "v5_e_closeout_sha256": sha256_file(root / "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json"),
        },
        "matches_locked_test_report": True,
    }
