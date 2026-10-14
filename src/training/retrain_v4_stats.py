"""Statistical helpers for retrain-v4 accuracy gates (pure numpy; no training)."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


def _require_unit_interval(arr: np.ndarray, *, label: str) -> None:
    if arr.size == 0:
        raise ValueError(f"{label} must be non-empty")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{label} values must lie in [0, 1]")


def seed_average_correctness(
    sample_ids: list[str],
    seed_correct: dict[int | str, np.ndarray | list[bool]],
    *,
    seed_order: list[int] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Mean correctness per sample across confirmation seeds.

    ``seed_correct[seed]`` must be aligned to the same ``sample_ids`` order.
    Prefer ``align_rows_by_sample_id`` when joining heterogeneous seed exports.
    Does **not** pool ``len(seeds) * n`` rows as independent observations.
    """
    if not sample_ids:
        raise ValueError("sample_ids must be non-empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    seeds = list(seed_order) if seed_order is not None else sorted(seed_correct.keys(), key=str)
    if not seeds:
        raise ValueError("seed_correct is empty")
    mats: list[np.ndarray] = []
    n = len(sample_ids)
    for s in seeds:
        if s not in seed_correct:
            raise ValueError(f"missing seed column {s!r}")
        col = np.asarray(seed_correct[s], dtype=float).reshape(-1)
        if col.shape[0] != n:
            raise ValueError(f"seed {s!r} length {col.shape[0]} != n_samples {n}")
        _require_unit_interval(col, label=f"seed {s!r} correctness")
        mats.append(col)
    avg = np.mean(np.stack(mats, axis=0), axis=0)
    return list(sample_ids), avg


def align_rows_by_sample_id(
    rows_by_seed: dict[int | str, list[dict[str, Any]]],
    *,
    id_key: str = "sample_id",
    correct_key: str = "correct_top1",
    signer_key: str = "signer_id",
) -> tuple[list[str], list[str], dict[int | str, np.ndarray]]:
    """Join heterogeneous per-seed row exports by ``sample_id`` (never by list order).

    Returns ``(sample_ids, signers, seed_correct)`` ready for seed averaging.
    """
    if not rows_by_seed:
        raise ValueError("rows_by_seed must be non-empty")
    id_sets = []
    for seed, rows in rows_by_seed.items():
        if not rows:
            raise ValueError(f"seed {seed!r} has no rows")
        ids = [str(r[id_key]) for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"seed {seed!r} has duplicate sample_id")
        id_sets.append(set(ids))
    common = set.intersection(*id_sets)
    if not common:
        raise ValueError("no common sample_id across seeds")
    sample_ids = sorted(common)

    signer_map: dict[str, str] = {}
    for seed, rows in rows_by_seed.items():
        by_id = {str(r[id_key]): r for r in rows}
        for sid in sample_ids:
            signer = str(by_id[sid][signer_key])
            if not signer:
                raise ValueError(f"empty signer_id for sample_id={sid!r} seed={seed!r}")
            prev = signer_map.get(sid)
            if prev is not None and prev != signer:
                raise ValueError(
                    f"signer mismatch for sample_id={sid!r}: {prev!r} vs {signer!r}"
                )
            signer_map[sid] = signer

    seed_correct: dict[int | str, np.ndarray] = {}
    for seed, rows in rows_by_seed.items():
        by_id = {str(r[id_key]): r for r in rows}
        vals = []
        for sid in sample_ids:
            raw = by_id[sid][correct_key]
            vals.append(float(raw))
        arr = np.asarray(vals, dtype=float)
        _require_unit_interval(arr, label=f"seed {seed!r} correctness")
        seed_correct[seed] = arr
    signers = [signer_map[sid] for sid in sample_ids]
    return sample_ids, signers, seed_correct


def hierarchical_paired_bootstrap_delta_ci(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    signers: list[str],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
    label_a: str = "b3",
    label_b: str = "b4r",
) -> dict[str, Any]:
    """Paired Δ CI (B−A) via hierarchical bootstrap: signer → sample-in-signer.

    ``correct_*`` are aligned per-sample scores in [0, 1] (e.g. seed-averaged
    correctness) or boolean hits. Point estimate uses the full aligned set.
    Forbidden: treating stacked multi-seed rows as i.i.d. samples.
    """
    if int(n_boot) <= 0:
        raise ValueError("n_boot must be > 0")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if len(correct_a) != len(correct_b) or len(correct_a) != len(signers):
        raise ValueError("hierarchical paired bootstrap requires aligned arrays")
    if any(not str(s) for s in signers):
        raise ValueError("signers must be non-empty strings")
    a = np.asarray(correct_a, dtype=float).reshape(-1)
    b = np.asarray(correct_b, dtype=float).reshape(-1)
    _require_unit_interval(a, label="correct_a")
    _require_unit_interval(b, label="correct_b")
    point = float((b.mean() - a.mean()) * 100.0)

    signer_ids = sorted(set(signers))
    if not signer_ids:
        raise ValueError("signers must be non-empty")
    groups = {
        s: np.array([i for i, ss in enumerate(signers) if ss == s], dtype=int)
        for s in signer_ids
    }
    rng = np.random.default_rng(int(seed))
    stats: list[float] = []
    for _ in range(int(n_boot)):
        chosen_signers = rng.choice(signer_ids, size=len(signer_ids), replace=True)
        idx_parts: list[np.ndarray] = []
        for s in chosen_signers:
            g = groups[s]
            drawn = rng.choice(g, size=len(g), replace=True)
            idx_parts.append(drawn)
        idx = np.concatenate(idx_parts)
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
        "n_signers": int(len(signer_ids)),
        "method": "hierarchical_paired_signer_then_sample_bootstrap",
        "unit_of_analysis": "seed_averaged_per_sample_correctness",
        "forbid_pool_seeds_as_iid": True,
        "join_by_sample_id_required": True,
    }
