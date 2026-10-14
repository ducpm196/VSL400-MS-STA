# -*- coding: utf-8 -*-
"""Coverage set algebra for S5/R1 prediction dumps (precedence-safe)."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def coverage_sets(
    expected_ids: Sequence[str] | Iterable[str],
    observed_ids: Sequence[str] | Iterable[str],
) -> dict[str, list[str] | int | bool]:
    """Compute missing/extra/duplicate vs an expected ID universe.

    Uses explicit parentheses so union is applied before set difference:
    extra = (observed ∪ …) − expected — never `A | B - C` without parens.
    """
    expected = [str(x) for x in expected_ids]
    observed = [str(x) for x in observed_ids]
    exp_set = set(expected)
    obs_set = set(observed)
    counts = Counter(observed)
    duplicates = sorted([sid for sid, n in counts.items() if n > 1])
    missing = sorted(exp_set - obs_set)
    # Correct form (must keep parentheses):
    extra = sorted(obs_set - exp_set)
    return {
        "n_expected": len(expected),
        "n_observed": len(observed),
        "n_unique_observed": len(obs_set),
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "duplicate_count": int(sum(n - 1 for n in counts.values() if n > 1)),
        "exact_coverage": not missing and not extra and not duplicates and len(observed) == len(expected),
    }


def paired_arm_coverage(
    expected_ids: Sequence[str],
    ids_a: Sequence[str],
    ids_b: Sequence[str],
) -> dict[str, list[str] | int | bool]:
    """Coverage when two arms must each cover expected_ids exactly."""
    a = coverage_sets(expected_ids, ids_a)
    b = coverage_sets(expected_ids, ids_b)
    # Wrong legacy form (documented for regression): set(a)|set(b)-set(exp)
    # Correct union-then-diff:
    union_extra = sorted((set(ids_a) | set(ids_b)) - set(expected_ids))
    union_missing = sorted((set(expected_ids) - set(ids_a)) | (set(expected_ids) - set(ids_b)))
    return {
        "arm_a": a,
        "arm_b": b,
        "union_missing": union_missing,
        "union_extra": union_extra,
        "exact_coverage": bool(a["exact_coverage"] and b["exact_coverage"]),
    }


# Documented buggy expression (do not use in production):
BUGGY_EXTRA_EXPR = "sorted(set(ids_r1) | set(ids_s5) - set(val_ids))"
CORRECT_EXTRA_EXPR = "sorted((set(ids_r1) | set(ids_s5)) - set(val_ids))"
CORRECT_MISSING_EXPR = (
    "sorted((set(val_ids) - set(ids_r1)) | (set(val_ids) - set(ids_s5)))"
)
