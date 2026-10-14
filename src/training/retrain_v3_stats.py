"""Statistical helpers for retrain-v3 final retrospective evaluation (pure numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np


REQUIRED_TEST_VARIANTS = ("b0", "b3", "b4", "b4e")
NONINFERIORITY_MARGIN_PP = -1.0
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp == fp == fn == 0:
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s) * 100.0) if f1s else 0.0


def topk_accuracy(y_true: np.ndarray, topk: np.ndarray) -> float:
    hits = [y_true[i] in set(topk[i].tolist()) for i in range(len(y_true))]
    return float(np.mean(hits) * 100.0) if len(hits) else 0.0


def topk_hit_mask(y_true: np.ndarray, topk: np.ndarray) -> np.ndarray:
    return np.array(
        [bool(y_true[i] in set(topk[i].tolist())) for i in range(len(y_true))],
        dtype=bool,
    )


def mcnemar(baseline_correct: np.ndarray, other_correct: np.ndarray) -> dict[str, Any]:
    b_only = int(np.sum(baseline_correct & ~other_correct))
    o_only = int(np.sum(~baseline_correct & other_correct))
    # Exact binomial two-sided p-value under null P(other-only)=0.5 among discordants.
    n = b_only + o_only
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest

        p = float(binomtest(k=o_only, n=n, p=0.5, alternative="two-sided").pvalue)
    return {
        "baseline_only_correct": b_only,
        "other_only_correct": o_only,
        "n_discordant": n,
        "n_common_pairs": int(len(baseline_correct)),
        "p_value_exact": float(p),
        "method": "exact_binomial_two_sided_mcnemar",
    }


def holm_correction(p_values: list[tuple[str, float]]) -> dict[str, Any]:
    """Holm step-down on named p-values; returns adjusted decisions at alpha=0.05."""
    alpha = 0.05
    ordered = sorted(p_values, key=lambda kv: kv[1])
    m = len(ordered)
    decisions: dict[str, Any] = {}
    reject_rest = False
    for i, (name, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        reject = (not reject_rest) and (p <= thresh)
        if not reject:
            reject_rest = True
        decisions[name] = {
            "p_raw": float(p),
            "holm_threshold": float(thresh),
            "reject_null_at_0.05": bool(reject),
        }
    return {"alpha": alpha, "tests": decisions}


def signer_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signers: list[str],
) -> dict[str, Any]:
    by: dict[str, dict[str, float | int]] = {}
    for s in sorted(set(signers)):
        idx = [i for i, ss in enumerate(signers) if ss == s]
        if not idx:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        by[s] = {
            "n": int(len(idx)),
            "top1": float((yp == yt).mean() * 100.0),
        }
    worst = min(by.items(), key=lambda kv: float(kv[1]["top1"])) if by else None
    return {
        "per_signer": by,
        "worst_signer": {"signer": worst[0], **worst[1]} if worst else None,
    }


def _signer_groups(signers: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    signer_ids = sorted(set(signers))
    groups = {
        s: np.array([i for i, ss in enumerate(signers) if ss == s], dtype=int) for s in signer_ids
    }
    return signer_ids, groups


def signer_stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    signers: list[str],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap CI of Top-1 by resampling whole signers (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    signer_ids, groups = _signer_groups(signers)
    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(signer_ids, size=len(signer_ids), replace=True)
        idx = np.concatenate([groups[s] for s in chosen])
        stats.append(float((y_pred[idx] == y_true[idx]).mean() * 100.0))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return {
        "top1_point": float((y_pred == y_true).mean() * 100.0),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "method": "signer_cluster_bootstrap",
    }


def paired_signer_bootstrap_delta_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    signers: list[str],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """Paired delta CI for (B − A) in percentage points via signer-cluster bootstrap.

    ``correct_*`` are boolean hit arrays aligned to the same sample order.
    """
    if len(correct_a) != len(correct_b) or len(correct_a) != len(signers):
        raise ValueError("paired bootstrap requires aligned correct masks and signers")
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    point = float((b.mean() - a.mean()) * 100.0)
    rng = np.random.default_rng(seed)
    signer_ids, groups = _signer_groups(signers)
    stats: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(signer_ids, size=len(signer_ids), replace=True)
        idx = np.concatenate([groups[s] for s in chosen])
        stats.append(float((b[idx].mean() - a[idx].mean()) * 100.0))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return {
        "label": f"{label_b}_minus_{label_a}",
        "delta_pp": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_common_pairs": int(len(a)),
        "method": "paired_signer_cluster_bootstrap",
    }


def noninferiority_from_paired_ci(
    paired_delta: dict[str, Any],
    *,
    margin_pp: float = NONINFERIORITY_MARGIN_PP,
) -> dict[str, Any]:
    """Non-inferiority PASS iff lower 95% CI of (candidate − reference) >= margin.

    Default margin is −1.0 pp (candidate may be at most 1 pp worse).
    """
    lo = float(paired_delta["ci_low"])
    delta = float(paired_delta["delta_pp"])
    return {
        "delta_pp": delta,
        "ci_low": lo,
        "ci_high": float(paired_delta["ci_high"]),
        "margin_pp": float(margin_pp),
        "noninferior": bool(lo >= float(margin_pp)),
        "decision_rule": "ci_low >= margin_pp",
        "n_boot": paired_delta.get("n_boot"),
        "seed": paired_delta.get("seed"),
        "method": paired_delta.get("method"),
    }


def require_exact_variants(present: set[str], required: tuple[str, ...] = REQUIRED_TEST_VARIANTS) -> None:
    missing = [v for v in required if v not in present]
    extra = sorted(present - set(required))
    if missing or extra:
        raise ValueError(f"variant set mismatch missing={missing} extra={extra}")


def require_sha_present_and_equal(actual: object, expected: object, *, label: str) -> None:
    if actual is None or expected is None or actual == "" or expected == "":
        raise ValueError(f"{label} SHA missing (None/empty not allowed)")
    if str(actual) != str(expected):
        raise ValueError(f"{label} SHA mismatch")
