# -*- coding: utf-8 -*-
"""Regression tests for S5 coverage aggregation (set operator precedence)."""
from __future__ import annotations

from src.optimization.e4_syskd.coverage_sets import (
    BUGGY_EXTRA_EXPR,
    CORRECT_EXTRA_EXPR,
    coverage_sets,
    paired_arm_coverage,
)


def test_missing_only():
    r = coverage_sets(["a", "b", "c"], ["a", "b"])
    assert r["missing"] == ["c"]
    assert r["extra"] == []
    assert r["exact_coverage"] is False


def test_extra_only():
    r = coverage_sets(["a", "b"], ["a", "b", "x"])
    assert r["missing"] == []
    assert r["extra"] == ["x"]
    assert r["exact_coverage"] is False


def test_missing_and_extra():
    r = coverage_sets(["a", "b", "c"], ["a", "x"])
    assert r["missing"] == ["b", "c"]
    assert r["extra"] == ["x"]
    assert r["exact_coverage"] is False


def test_duplicate():
    r = coverage_sets(["a", "b"], ["a", "b", "a"])
    assert r["duplicates"] == ["a"]
    assert r["duplicate_count"] == 1
    assert r["exact_coverage"] is False


def test_exact_coverage():
    r = coverage_sets(["a", "b"], ["b", "a"])
    assert r["missing"] == []
    assert r["extra"] == []
    assert r["duplicates"] == []
    assert r["exact_coverage"] is True


def test_set_operator_precedence_bug():
    val = {"a", "b", "c"}
    r1 = {"a", "b", "c"}
    s5 = {"a", "b", "c"}
    # Buggy: `-` binds tighter than `|` → A | (B - C) == A when B⊆C
    buggy = sorted(set(r1) | set(s5) - set(val))
    correct = sorted((set(r1) | set(s5)) - set(val))
    assert buggy == sorted(r1)
    assert correct == []
    assert BUGGY_EXTRA_EXPR.startswith("sorted(set(ids_r1)")
    assert CORRECT_EXTRA_EXPR.startswith("sorted((set(ids_r1)")


def test_paired_arm_exact():
    p = paired_arm_coverage(["1", "2"], ["1", "2"], ["2", "1"])
    assert p["exact_coverage"] is True
    assert p["union_extra"] == []
    assert p["union_missing"] == []
