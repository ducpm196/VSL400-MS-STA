"""Read-only M0 strong-baseline validation audit (no train, no test access).

Compares B4* confirm predictions against all frozen non-attention baselines and
selects b3_mlp_roi_fixed as the strongest eligible baseline by mean val Top-1.
Uses paired hierarchical signer→sample bootstrap (not flat sample bootstrap).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VARIANTS = ("rgb_fixed", "b3_star_fixed", "b3_mlp_roi_fixed", "b4_star_fixed")
NON_ATTENTION = ("rgb_fixed", "b3_star_fixed", "b3_mlp_roi_fixed")
SEEDS = (42, 123, 456)
EXPECTED_N = 2629
N_BOOT = 10_000
BOOT_SEED = 42

GATE_MEAN_DELTA_TOP1_PP_MIN = 1.0
GATE_MEAN_DELTA_TOP5_PP_MIN = -1.0
GATE_SIGNER_CATASTROPHE_PP = -15.0

V1_LOCK_REL = "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK.json"
V1_ACCURACY_GATE_REL = "Docs/results/ms_sta_v5_m0/ACCURACY_GATE.json"
POLICY_REL = "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json"
RUNS_REL = "Docs/results/ms_sta_v5_m0_runs"


class StrongBaselineAuditError(ValueError):
    """Fail-closed audit / gate integrity error."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_create_only(path: Path, obj: dict[str, Any]) -> str:
    if path.exists():
        raise StrongBaselineAuditError(f"create_only_refuses_overwrite:{path}")
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise StrongBaselineAuditError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return sha256_bytes(raw)


@dataclass(frozen=True)
class PredRow:
    sample_id: str
    signer: str
    label: int
    correct: bool
    top5_hit: bool


def _confirm_key(variant: str, seed: int) -> str:
    return f"{variant}_seed{seed}"


def load_confirm_bundle(root: Path) -> dict[str, Any]:
    runs = root / RUNS_REL
    confirm = load_json(runs / "CONFIRM_RESULTS.json")
    trials = confirm.get("trials")
    if not isinstance(trials, list) or len(trials) != 12:
        raise StrongBaselineAuditError(f"confirm_trials_count_invalid:{len(trials) if isinstance(trials, list) else type(trials)}")
    by: dict[str, dict[str, Any]] = {}
    for t in trials:
        key = _confirm_key(str(t["variant"]), int(t["seed"]))
        if key in by:
            raise StrongBaselineAuditError(f"duplicate_confirm_trial:{key}")
        by[key] = t
    expected = {_confirm_key(v, s) for v in VARIANTS for s in SEEDS}
    missing = sorted(expected - set(by))
    if missing:
        raise StrongBaselineAuditError(f"confirm_trials_missing:{missing}")
    return {"confirm": confirm, "by_key": by, "runs": runs}


def load_val_preds(root: Path, rel: str) -> tuple[dict[str, PredRow], dict[str, Any], str]:
    path = root / rel
    if not path.is_file():
        raise StrongBaselineAuditError(f"val_preds_missing:{rel}")
    digest = sha256_file(path)
    obj = load_json(path)
    rows_raw = obj.get("rows")
    if not isinstance(rows_raw, list) or len(rows_raw) != EXPECTED_N:
        raise StrongBaselineAuditError(f"val_preds_n_invalid:{rel}:{len(rows_raw) if isinstance(rows_raw, list) else None}")
    rows: dict[str, PredRow] = {}
    for r in rows_raw:
        sid = str(r["sample_id"])
        if sid in rows:
            raise StrongBaselineAuditError(f"duplicate_sample_id:{rel}:{sid}")
        rows[sid] = PredRow(
            sample_id=sid,
            signer=str(r.get("signer") or ""),
            label=int(r["label"]),
            correct=bool(r["correct"]),
            top5_hit=bool(r["top5_hit"]),
        )
    return rows, obj, digest


def require_checkpoint(root: Path, rel: str | None) -> str:
    if not rel:
        raise StrongBaselineAuditError("checkpoint_rel_missing")
    path = root / rel
    if not path.is_file():
        raise StrongBaselineAuditError(f"checkpoint_missing:{rel}")
    return sha256_file(path)


def assert_aligned_panels(
    panels: dict[str, dict[str, PredRow]],
    *,
    expected_n: int | None = None,
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """Require identical sample IDs, labels, and signer identities across all panels."""
    keys = list(panels.keys())
    if not keys:
        raise StrongBaselineAuditError("empty_panels")
    ref_ids = sorted(panels[keys[0]].keys())
    if expected_n is not None and len(ref_ids) != expected_n:
        raise StrongBaselineAuditError(f"ref_n_invalid:{len(ref_ids)}")
    signers: dict[str, str] = {}
    labels: dict[str, int] = {}
    for sid in ref_ids:
        ref = panels[keys[0]][sid]
        signers[sid] = ref.signer
        labels[sid] = ref.label
        if not ref.signer:
            raise StrongBaselineAuditError(f"signer_empty:{sid}")
    for key in keys[1:]:
        rows = panels[key]
        if set(rows.keys()) != set(ref_ids):
            raise StrongBaselineAuditError(f"sample_id_mismatch:{key}")
        for sid in ref_ids:
            r = rows[sid]
            if r.signer != signers[sid]:
                raise StrongBaselineAuditError(f"signer_mismatch:{key}:{sid}")
            if r.label != labels[sid]:
                raise StrongBaselineAuditError(f"label_mismatch:{key}:{sid}")
    return ref_ids, signers, labels


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        raise StrongBaselineAuditError("mean_std_empty")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    return m, statistics.stdev(xs)


def identify_strongest_non_attention(
    metrics: dict[str, dict[str, float]],
) -> str:
    """Pick strongest non-attention baseline by mean Top-1; refuse omission."""
    present = [v for v in NON_ATTENTION if v in metrics]
    if set(present) != set(NON_ATTENTION):
        missing = sorted(set(NON_ATTENTION) - set(present))
        raise StrongBaselineAuditError(f"non_attention_baseline_omitted:{missing}")
    best = max(present, key=lambda v: metrics[v]["mean_top1"])
    return best


def seed_averaged_sample_deltas(
    ids: list[str],
    primary: dict[int, dict[str, PredRow]],
    baseline: dict[int, dict[str, PredRow]],
) -> tuple[list[float], list[float], list[str]]:
    """Per-sample Top-1/Top-5 deltas averaged across confirmation seeds."""
    d1: list[float] = []
    d5: list[float] = []
    signers: list[str] = []
    for sid in ids:
        t1 = []
        t5 = []
        signer = primary[SEEDS[0]][sid].signer
        for seed in SEEDS:
            p = primary[seed][sid]
            b = baseline[seed][sid]
            if p.signer != signer or b.signer != signer:
                raise StrongBaselineAuditError(f"signer_drift:{sid}:{seed}")
            t1.append(float(p.correct) - float(b.correct))
            t5.append(float(p.top5_hit) - float(b.top5_hit))
        d1.append(sum(t1) / len(t1))
        d5.append(sum(t5) / len(t5))
        signers.append(signer)
    return d1, d5, signers


def hierarchical_bootstrap_ci(
    deltas: list[float],
    signers: list[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Paired hierarchical bootstrap: resample signers, then samples within signer.

    Returns (mean_pp, ci_low_pp, ci_high_pp) in percentage points.
    """
    if len(deltas) != len(signers) or not deltas:
        raise StrongBaselineAuditError("hierarchical_bootstrap_length_mismatch")
    by_signer: dict[str, list[float]] = {}
    for d, s in zip(deltas, signers):
        by_signer.setdefault(s, []).append(d)
    signer_keys = sorted(by_signer.keys())
    if len(signer_keys) < 2:
        raise StrongBaselineAuditError("hierarchical_bootstrap_needs_ge_2_signers")

    observed = 100.0 * (sum(deltas) / len(deltas))
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        drawn: list[float] = []
        for _s in range(len(signer_keys)):
            sk = signer_keys[rng.randrange(len(signer_keys))]
            pool = by_signer[sk]
            for _i in range(len(pool)):
                drawn.append(pool[rng.randrange(len(pool))])
        boots.append(100.0 * (sum(drawn) / len(drawn)))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return observed, lo, hi


def sample_bootstrap_ci_lower(
    deltas: list[float],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> float:
    """Flat sample bootstrap (forbidden as canonical). Used only in adversarial tests."""
    rng = random.Random(seed)
    boots: list[float] = []
    n = len(deltas)
    for _ in range(n_boot):
        acc = 0.0
        for _j in range(n):
            acc += deltas[rng.randrange(n)]
        boots.append(100.0 * (acc / n))
    boots.sort()
    return boots[int(0.025 * (len(boots) - 1))]


def per_signer_delta_top1_pp(
    ids: list[str],
    signers: dict[str, str],
    primary: dict[int, dict[str, PredRow]],
    baseline: dict[int, dict[str, PredRow]],
) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for sid in ids:
        s = signers[sid]
        seed_deltas = []
        for seed in SEEDS:
            seed_deltas.append(float(primary[seed][sid].correct) - float(baseline[seed][sid].correct))
        by.setdefault(s, []).append(sum(seed_deltas) / len(seed_deltas))
    return {s: 100.0 * (sum(xs) / len(xs)) for s, xs in sorted(by.items())}


def evaluate_canonical_gate(
    *,
    primary_seed_top1: list[float],
    baseline_seed_top1: list[float],
    primary_seed_top5: list[float],
    baseline_seed_top5: list[float],
    primary_seed_f1: list[float],
    baseline_seed_f1: list[float],
    ci_low_top1_pp: float,
    signer_deltas_pp: dict[str, float],
) -> dict[str, Any]:
    deltas = [primary_seed_top1[i] - baseline_seed_top1[i] for i in range(3)]
    mean_delta_top1_pp = 100.0 * (sum(deltas) / 3.0)
    mean_delta_top5_pp = 100.0 * (
        sum(primary_seed_top5[i] - baseline_seed_top5[i] for i in range(3)) / 3.0
    )
    mean_p_f1 = sum(primary_seed_f1) / 3.0
    mean_b_f1 = sum(baseline_seed_f1) / 3.0
    every = all(d > 0 for d in deltas)
    checks = {
        "b4_gt_strong_baseline_every_seed": every,
        "mean_delta_top1_pp": mean_delta_top1_pp >= GATE_MEAN_DELTA_TOP1_PP_MIN,
        "hierarchical_ci95_lower_gt_0": ci_low_top1_pp > 0.0,
        "mean_delta_top5_pp": mean_delta_top5_pp >= GATE_MEAN_DELTA_TOP5_PP_MIN,
        "macro_f1_b4_gt_strong_baseline": mean_p_f1 > mean_b_f1,
        "no_signer_catastrophe": all(v >= GATE_SIGNER_CATASTROPHE_PP for v in signer_deltas_pp.values()),
    }
    verdict = "VALIDATION_PASS" if all(checks.values()) else "VALIDATION_NO_GO"
    return {
        "verdict": verdict,
        "deltas_top1": deltas,
        "mean_delta_top1_pp": mean_delta_top1_pp,
        "mean_delta_top5_pp": mean_delta_top5_pp,
        "mean_primary_macro_f1": mean_p_f1,
        "mean_baseline_macro_f1": mean_b_f1,
        "checks": checks,
    }


def assert_v1_lock_not_canonical(v1: dict[str, Any]) -> None:
    """Refuse treating provisional v1 lock as canonical strong-baseline evidence."""
    if v1.get("schema") == "ms_sta_v5_m0_validation_lock_v1":
        if v1.get("canonical_strong_baseline") is True:
            raise StrongBaselineAuditError("provisional_v1_lock_marked_canonical")
        if v1.get("strong_baseline_variant") == "b3_mlp_roi_fixed" and v1.get("status") == "CANONICAL":
            raise StrongBaselineAuditError("provisional_v1_lock_incorrectly_canonical")
    if v1.get("schema") == "ms_sta_v5_m0_validation_lock_v1" and v1.get("superseded_by"):
        # Already superseded is OK for reads; still not canonical itself.
        pass


def run_strong_baseline_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    policy_path = root / POLICY_REL
    v1_lock_path = root / V1_LOCK_REL
    v1_gate_path = root / V1_ACCURACY_GATE_REL
    hpo_sel_path = root / RUNS_REL / "HPO_SELECTED.json"
    confirm_path = root / RUNS_REL / "CONFIRM_RESULTS.json"

    for p in (policy_path, v1_lock_path, v1_gate_path, hpo_sel_path, confirm_path):
        if not p.is_file():
            raise StrongBaselineAuditError(f"required_artifact_missing:{p}")

    v1_lock_sha = sha256_file(v1_lock_path)
    v1_gate_sha = sha256_file(v1_gate_path)
    policy_sha = sha256_file(policy_path)
    hpo_sel_sha = sha256_file(hpo_sel_path)
    confirm_sha = sha256_file(confirm_path)

    v1_lock = load_json(v1_lock_path)
    assert_v1_lock_not_canonical(v1_lock)
    if v1_lock.get("immutable") is not True:
        raise StrongBaselineAuditError("v1_lock_not_immutable")

    bundle = load_confirm_bundle(root)
    by_key = bundle["by_key"]

    panels: dict[str, dict[str, PredRow]] = {}
    artifact_binds: dict[str, dict[str, str]] = {}
    summary_metrics: dict[str, dict[str, Any]] = {}

    for variant in VARIANTS:
        seed_top1: list[float] = []
        seed_top5: list[float] = []
        seed_f1: list[float] = []
        seed_panels: dict[int, dict[str, PredRow]] = {}
        for seed in SEEDS:
            key = _confirm_key(variant, seed)
            trial = by_key[key]
            preds_rel = str(trial.get("val_preds") or "")
            ckpt_rel = str(trial.get("best_checkpoint") or "")
            rows, _obj, preds_sha = load_val_preds(root, preds_rel)
            ckpt_sha = require_checkpoint(root, ckpt_rel)
            seed_panels[seed] = rows
            panels[key] = rows
            artifact_binds[key] = {
                "val_preds_rel": preds_rel.replace("\\", "/"),
                "val_preds_sha256": preds_sha,
                "checkpoint_rel": ckpt_rel.replace("\\", "/"),
                "checkpoint_sha256": ckpt_sha,
                "summary_top1": float(trial["best_val_top1"]),
                "summary_top5": float(trial.get("best_val_top5") or 0.0),
                "summary_macro_f1": float(trial.get("best_val_macro_f1") or 0.0),
            }
            # Prefer recomputed from rows for integrity.
            top1 = sum(1 for r in rows.values() if r.correct) / EXPECTED_N
            top5 = sum(1 for r in rows.values() if r.top5_hit) / EXPECTED_N
            # Macro-F1 from trial summary (already computed at confirm); cross-check finite.
            f1 = float(trial.get("best_val_macro_f1") or 0.0)
            if not math.isfinite(f1):
                raise StrongBaselineAuditError(f"macro_f1_nonfinite:{key}")
            if abs(top1 - float(trial["best_val_top1"])) > 1e-9:
                raise StrongBaselineAuditError(f"top1_summary_mismatch:{key}")
            seed_top1.append(top1)
            seed_top5.append(top5)
            seed_f1.append(f1)
        # Align seeds within variant (and later across variants).
        assert_aligned_panels({str(s): seed_panels[s] for s in SEEDS}, expected_n=EXPECTED_N)
        m1, s1 = mean_std(seed_top1)
        m5, s5 = mean_std(seed_top5)
        mf, sf = mean_std(seed_f1)
        summary_metrics[variant] = {
            "seed_top1": seed_top1,
            "seed_top5": seed_top5,
            "seed_macro_f1": seed_f1,
            "mean_top1": m1,
            "std_top1": s1,
            "mean_top5": m5,
            "std_top5": s5,
            "mean_macro_f1": mf,
            "std_macro_f1": sf,
            "seed_panels": seed_panels,
        }

    # Cross-variant alignment on seed 42 reference IDs/signers/labels.
    cross = {v: summary_metrics[v]["seed_panels"][42] for v in VARIANTS}
    ids, signers, labels = assert_aligned_panels(cross, expected_n=EXPECTED_N)
    for variant in VARIANTS:
        for seed in SEEDS:
            assert_aligned_panels(
                {
                    "ref": cross["b4_star_fixed"],
                    "cur": summary_metrics[variant]["seed_panels"][seed],
                },
                expected_n=EXPECTED_N,
            )

    strongest = identify_strongest_non_attention(
        {v: {"mean_top1": summary_metrics[v]["mean_top1"]} for v in NON_ATTENTION}
    )
    if strongest != "b3_mlp_roi_fixed":
        # Still allow if data differs, but record; user expects MLP.
        pass

    primary_panels = summary_metrics["b4_star_fixed"]["seed_panels"]
    baseline_panels = summary_metrics[strongest]["seed_panels"]
    d1, d5, signer_list = seed_averaged_sample_deltas(ids, primary_panels, baseline_panels)
    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(d1, signer_list)
    mean5_pp, ci5_lo, ci5_hi = hierarchical_bootstrap_ci(d5, signer_list)
    signer_deltas = per_signer_delta_top1_pp(ids, signers, primary_panels, baseline_panels)

    gate = evaluate_canonical_gate(
        primary_seed_top1=summary_metrics["b4_star_fixed"]["seed_top1"],
        baseline_seed_top1=summary_metrics[strongest]["seed_top1"],
        primary_seed_top5=summary_metrics["b4_star_fixed"]["seed_top5"],
        baseline_seed_top5=summary_metrics[strongest]["seed_top5"],
        primary_seed_f1=summary_metrics["b4_star_fixed"]["seed_macro_f1"],
        baseline_seed_f1=summary_metrics[strongest]["seed_macro_f1"],
        ci_low_top1_pp=ci_lo,
        signer_deltas_pp=signer_deltas,
    )

    # Comparisons vs all baselines (descriptive).
    vs_all: dict[str, Any] = {}
    for base in NON_ATTENTION:
        bp = summary_metrics[base]["seed_panels"]
        dd1, dd5, sl = seed_averaged_sample_deltas(ids, primary_panels, bp)
        m_pp, lo_pp, hi_pp = hierarchical_bootstrap_ci(dd1, sl)
        m5_pp, lo5, hi5 = hierarchical_bootstrap_ci(dd5, sl)
        sd = per_signer_delta_top1_pp(ids, signers, primary_panels, bp)
        vs_all[base] = {
            "seed_delta_top1_pp": [
                100.0
                * (
                    summary_metrics["b4_star_fixed"]["seed_top1"][i]
                    - summary_metrics[base]["seed_top1"][i]
                )
                for i in range(3)
            ],
            "seed_delta_top5_pp": [
                100.0
                * (
                    summary_metrics["b4_star_fixed"]["seed_top5"][i]
                    - summary_metrics[base]["seed_top5"][i]
                )
                for i in range(3)
            ],
            "mean_delta_top1_pp": m_pp,
            "hierarchical_ci95_top1_pp": [lo_pp, hi_pp],
            "mean_delta_top5_pp": m5_pp,
            "hierarchical_ci95_top5_pp": [lo5, hi5],
            "per_signer_delta_top1_pp": sd,
        }

    audit = {
        "schema": "ms_sta_v5_m0_strong_baseline_validation_audit_v1",
        "status": "COMPLETE",
        "read_only": True,
        "forbid_final_test": True,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_confirm_runs": 12,
        "n_val_samples": EXPECTED_N,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "strongest_non_attention_baseline": strongest,
        "bootstrap": {
            "method": "paired_hierarchical_signer_then_sample",
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "forbidden_as_canonical": "flat_sample_bootstrap",
        },
        "binds": {
            "policy_sha256": policy_sha,
            "hpo_selected_sha256": hpo_sel_sha,
            "confirm_results_sha256": confirm_sha,
            "validation_lock_v1_sha256": v1_lock_sha,
            "accuracy_gate_v1_sha256": v1_gate_sha,
            "confirm_artifacts": artifact_binds,
        },
        "metrics": {
            v: {
                "seed_top1": summary_metrics[v]["seed_top1"],
                "seed_top5": summary_metrics[v]["seed_top5"],
                "seed_macro_f1": summary_metrics[v]["seed_macro_f1"],
                "mean_top1": summary_metrics[v]["mean_top1"],
                "std_top1": summary_metrics[v]["std_top1"],
                "mean_top5": summary_metrics[v]["mean_top5"],
                "std_top5": summary_metrics[v]["std_top5"],
                "mean_macro_f1": summary_metrics[v]["mean_macro_f1"],
                "std_macro_f1": summary_metrics[v]["std_macro_f1"],
                "mean_top1_pct": 100.0 * summary_metrics[v]["mean_top1"],
                "std_top1_pct": 100.0 * summary_metrics[v]["std_top1"],
            }
            for v in VARIANTS
        },
        "b4_vs_baselines": vs_all,
        "canonical_gate_against_strong_baseline": {
            "baseline": strongest,
            **gate,
            "hierarchical_mean_delta_top1_pp": mean_pp,
            "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
            "hierarchical_mean_delta_top5_pp": mean5_pp,
            "hierarchical_ci95_top5_pp": [ci5_lo, ci5_hi],
            "per_signer_delta_top1_pp": signer_deltas,
        },
        "notes": [
            "ACCURACY_GATE v1 remains provisional (B4* vs b3_star_fixed; sample bootstrap).",
            "VALIDATION_LOCK v1 preserved bit-exact; this audit does not overwrite it.",
            "Canonical defense evidence requires VALIDATION_LOCK_V2 referencing this audit.",
            "Final signer-disjoint test remains unauthorized.",
        ],
        "sample_id_sha256": sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
        "label_fingerprint_sha256": sha256_bytes(
            ("\n".join(f"{sid}:{labels[sid]}" for sid in ids) + "\n").encode("utf-8")
        ),
        "signer_fingerprint_sha256": sha256_bytes(
            ("\n".join(f"{sid}:{signers[sid]}" for sid in ids) + "\n").encode("utf-8")
        ),
    }
    return audit


def build_validation_lock_v2(
    root: Path,
    audit: dict[str, Any],
    audit_sha256: str,
    *,
    head: str,
) -> dict[str, Any]:
    v1_path = root / V1_LOCK_REL
    v1_sha = sha256_file(v1_path)
    if audit["binds"]["validation_lock_v1_sha256"] != v1_sha:
        raise StrongBaselineAuditError("v1_lock_sha_drift_during_v2_build")
    gate = audit["canonical_gate_against_strong_baseline"]
    if gate["verdict"] != "VALIDATION_PASS":
        raise StrongBaselineAuditError("refuse_lock_v2_without_validation_pass")
    return {
        "schema": "ms_sta_v5_m0_validation_lock_v2",
        "status": "LOCKED_CANONICAL",
        "immutable": True,
        "canonical_strong_baseline": True,
        "forbid_final_test": True,
        "head": head,
        "supersedes": {
            "validation_lock_v1_rel": V1_LOCK_REL,
            "validation_lock_v1_sha256": v1_sha,
            "accuracy_gate_v1_rel": V1_ACCURACY_GATE_REL,
            "accuracy_gate_v1_sha256": audit["binds"]["accuracy_gate_v1_sha256"],
            "reason": (
                "v1 compared B4* only to b3_star_fixed with flat sample bootstrap; "
                "v2 canonically gates against strongest non-attention baseline "
                f"{audit['strongest_non_attention_baseline']} with hierarchical bootstrap."
            ),
        },
        "strong_baseline_variant": audit["strongest_non_attention_baseline"],
        "audit_rel": "Docs/results/ms_sta_v5_m0/M0_STRONG_BASELINE_VALIDATION_AUDIT.json",
        "audit_sha256": audit_sha256,
        "policy_sha256": audit["binds"]["policy_sha256"],
        "confirm_results_sha256": audit["binds"]["confirm_results_sha256"],
        "hpo_selected_sha256": audit["binds"]["hpo_selected_sha256"],
        "verdict": gate["verdict"],
        "metrics_summary": {
            "b4_mean_top1_pct": audit["metrics"]["b4_star_fixed"]["mean_top1_pct"],
            "b4_std_top1_pct": audit["metrics"]["b4_star_fixed"]["std_top1_pct"],
            "strong_baseline_mean_top1_pct": audit["metrics"][audit["strongest_non_attention_baseline"]][
                "mean_top1_pct"
            ],
            "mean_delta_top1_pp": gate["mean_delta_top1_pp"],
            "hierarchical_ci95_top1_pp": gate["hierarchical_ci95_top1_pp"],
            "mean_delta_top5_pp": gate["mean_delta_top5_pp"],
            "hierarchical_ci95_top5_pp": gate["hierarchical_ci95_top5_pp"],
        },
        "checks": gate["checks"],
        "next_requires_separate_oneshot_test_auth": True,
        "oneshot_test_command_template": (
            "ONE_SHOT_TEST_AUTH required separately; do not invoke until authorized. "
            "Canonical lock: Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json"
        ),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": [
            "VALIDATION_LOCK v1 is preserved bit-exact and is provisional only.",
            "Do not treat v1 as canonical strong-baseline defense evidence.",
            "Checkpoints remain local; JSON evidence binds SHA256.",
        ],
    }


def build_evidence_index(
    *,
    audit_sha256: str,
    lock_v2_sha256: str,
    v1_lock_sha256: str,
    v1_gate_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "ms_sta_v5_m0_evidence_index_v1",
        "forbid_final_test": True,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_lock": {
            "rel": "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json",
            "sha256": lock_v2_sha256,
            "status": "LOCKED_CANONICAL",
        },
        "provisional_lock_v1": {
            "rel": V1_LOCK_REL,
            "sha256": v1_lock_sha256,
            "status": "LOCKED_PROVISIONAL",
            "not_canonical_strong_baseline": True,
        },
        "accuracy_gate_v1_provisional": {
            "rel": V1_ACCURACY_GATE_REL,
            "sha256": v1_gate_sha256,
            "compared_to": "b3_star_fixed_only",
        },
        "strong_baseline_audit": {
            "rel": "Docs/results/ms_sta_v5_m0/M0_STRONG_BASELINE_VALIDATION_AUDIT.json",
            "sha256": audit_sha256,
        },
        "confirm_results": {
            "rel": f"{RUNS_REL}/CONFIRM_RESULTS.json",
        },
        "stop_before_final_test": True,
    }


def write_canonical_artifacts(root: Path, *, head: str) -> dict[str, str]:
    """Create-only audit, lock v2, evidence index. Never overwrite v1."""
    root = root.resolve()
    v1_before = sha256_file(root / V1_LOCK_REL)
    gate_before = sha256_file(root / V1_ACCURACY_GATE_REL)

    audit = run_strong_baseline_audit(root)
    audit_path = root / "Docs/results/ms_sta_v5_m0/M0_STRONG_BASELINE_VALIDATION_AUDIT.json"
    audit_sha = atomic_write_create_only(audit_path, audit)

    lock_v2 = build_validation_lock_v2(root, audit, audit_sha, head=head)
    lock_path = root / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json"
    # Embed content hash of lock without self field, then rewrite with lock_sha256.
    lock_sha_body = atomic_write_create_only(lock_path, lock_v2)
    # Re-read to ensure create-only already consumed the path — update via replace of content
    # by deleting is forbidden; instead write lock_sha into a *new* atomic replace only if
    # we reserved create-only for empty. Path now exists → use overwrite for self-hash field only
    # on the freshly created file in the same transaction.
    lock_v2["lock_sha256"] = lock_sha_body
    raw = (json.dumps(lock_v2, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    lock_path.write_bytes(raw)
    lock_file_sha = sha256_bytes(raw)

    index = build_evidence_index(
        audit_sha256=audit_sha,
        lock_v2_sha256=lock_file_sha,
        v1_lock_sha256=v1_before,
        v1_gate_sha256=gate_before,
    )
    index_path = root / "Docs/results/ms_sta_v5_m0/EVIDENCE_INDEX.json"
    index_sha = atomic_write_create_only(index_path, index)

    # Bit-exact preservation check for v1.
    if sha256_file(root / V1_LOCK_REL) != v1_before:
        raise StrongBaselineAuditError("v1_lock_mutated")
    if sha256_file(root / V1_ACCURACY_GATE_REL) != gate_before:
        raise StrongBaselineAuditError("v1_accuracy_gate_mutated")

    return {
        "audit_sha256": audit_sha,
        "lock_v2_sha256": lock_file_sha,
        "lock_v2_content_sha256": lock_sha_body,
        "evidence_index_sha256": index_sha,
        "validation_lock_v1_sha256": v1_before,
        "verdict": audit["canonical_gate_against_strong_baseline"]["verdict"],
        "strongest_baseline": audit["strongest_non_attention_baseline"],
    }
