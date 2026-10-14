# -*- coding: utf-8 -*-
"""Independent verifier for E4_B4SC_V1 E1 factorial — recomputes from raw artifacts."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.b4q.invariants import sha256_file  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
LOCK = OUT / "E4_B4SC_V1_E1_SCHEDULE_LOCK.json"
ARMS = ("E0", "E1a", "E1b", "E1c")


def macro_f1(y_true: list[int], y_pred: list[int], n_classes: int = 400) -> float:
    y_true_a = np.asarray(y_true, dtype=np.int64)
    y_pred_a = np.asarray(y_pred, dtype=np.int64)
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred_a == c) & (y_true_a == c)))
        fp = int(np.sum((y_pred_a == c) & (y_true_a != c)))
        fn = int(np.sum((y_pred_a != c) & (y_true_a == c)))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else float(0.5 * (s[mid - 1] + s[mid]))


def p90(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(0.9 * (len(s) - 1)))])


def bootstrap_speedup(base: list[float], cand: list[float], *, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    speeds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        speeds.append(float(np.median(b[idx]) / max(float(np.median(c[idx])), 1e-9)))
    lo, hi = np.quantile(speeds, [0.025, 0.975])
    med_b, med_c = float(np.median(b)), float(np.median(c))
    return {
        "speedup_factor": med_b / max(med_c, 1e-9),
        "speedup_bootstrap_median": float(np.median(speeds)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "latency_reduction": 1.0 - med_c / max(med_b, 1e-9),
    }


def acc_from_preds(rows: list[dict]) -> dict:
    y = [int(r["y_true"]) for r in rows]
    p = [int(r["y_pred"]) for r in rows]
    top1 = float(np.mean([int(a == b) for a, b in zip(y, p)]))
    # top5 if present
    top5_hits = []
    for r in rows:
        t5 = r.get("top5") or []
        if t5:
            top5_hits.append(int(int(r["y_true"]) in set(int(x) for x in t5)))
        else:
            top5_hits.append(int(r.get("top5_hit", int(r["y_pred"] == r["y_true"]))))
    macro = macro_f1(y, p, n_classes=400)
    by_s: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_s[str(r.get("signer_id") or "")].append(int(r["y_pred"] == r["y_true"]))
    return {
        "n": len(rows),
        "top1": top1,
        "top5": float(np.mean(top5_hits)) if top5_hits else float("nan"),
        "macro_f1": macro,
        "signer_top1": {s: float(np.mean(v)) for s, v in by_s.items() if s},
    }


def paired_delta(base: list[dict], cand: list[dict], *, n_boot: int, seed: int) -> dict:
    ba = {r["sample_id"]: int(r["y_pred"] == r["y_true"]) for r in base}
    ca = {r["sample_id"]: int(r["y_pred"] == r["y_true"]) for r in cand}
    ids = sorted(set(ba) & set(ca))
    diffs = np.asarray([100.0 * (ca[i] - ba[i]) for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = [float(diffs[rng.integers(0, len(diffs), size=len(diffs))].mean()) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    b01 = b10 = 0
    for i in ids:
        if ba[i] == 0 and ca[i] == 1:
            b01 += 1
        elif ba[i] == 1 and ca[i] == 0:
            b10 += 1
    return {
        "mean_delta_top1_pp": float(diffs.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "mcnemar": {"b_only": b01, "a_only": b10, "n": len(ids)},
    }


def sample_medians(rows: list[dict], key: str = "e2e_ms") -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("measured") and r.get("success"):
            by[r["sample_id"]].append(float(r[key]))
    return {sid: median(vs) for sid, vs in by.items()}


def run_verify(*, run_id: str = "E1_R1") -> dict[str, Any]:
    mismatches: list[str] = []
    gaps: list[str] = []
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    pol = json.loads((ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json").read_text(encoding="utf-8"))
    parity_path = OUT / "E4_B4SC_V1_E1_INPUT_PARITY_REPORT.json"
    raw_path = OUT / "E4_B4SC_V1_E1_EFFICIENCY_RAW.jsonl"
    if not parity_path.is_file():
        gaps.append("missing_parity_report")
    if not raw_path.is_file():
        gaps.append("missing_efficiency_raw")

    pred_paths = {
        "E0": OUT / "E4_B4SC_V1_E1_E0_PREDICTIONS.jsonl",
        "E1a": OUT / "E4_B4SC_V1_E1_E1A_PREDICTIONS.jsonl",
        "E1b": OUT / "E4_B4SC_V1_E1_E1B_PREDICTIONS.jsonl",
        "E1c": OUT / "E4_B4SC_V1_E1_E1C_PREDICTIONS.jsonl",
    }
    parity_fail_retain = False
    gap_tok = OUT / "E4_B4SC_V1_E1_TOKEN_PARITY_SOURCE_GAP.json"
    if gap_tok.is_file() or (OUT / "E4_B4SC_V1_E1_DECODE_PARITY_SOURCE_GAP.json").is_file():
        parity_fail_retain = True
        if not (OUT / "E4_B4SC_V1_E1_E1B_PREDICTIONS_SOURCE_GAP.json").is_file():
            gaps.append("missing_E1b_pred_source_gap")
        if not (OUT / "E4_B4SC_V1_E1_E1C_PREDICTIONS_SOURCE_GAP.json").is_file():
            gaps.append("missing_E1c_pred_source_gap")

    for a, p in pred_paths.items():
        if a in ("E1b", "E1c") and parity_fail_retain:
            continue
        if not p.is_file():
            gaps.append(f"missing_pred_{a}")

    out: dict[str, Any] = {
        "schema": "ms_sta_v6_e4_b4sc_v1_e1_independent_verify_v1",
        "run_id": run_id,
        "schedule_lock_sha256": sha256_file(LOCK),
        "e4_ready": False,
        "mismatches": mismatches,
        "source_gaps": gaps,
    }
    if gaps:
        out["pass"] = False
        out["eligible_arms"] = []
        out["selected_system_base"] = None
        out["terminal_hint"] = "E1_BLOCKED_SOURCE_GAP"
        return out

    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    raw = load_jsonl(raw_path)
    ids = list(lock["bindings"]["sample_ids"])
    n_meas = int(lock["measurement_schedule"]["measured_rounds_per_sample_per_arm"])
    measured = [r for r in raw if r.get("measured")]
    if any(not r.get("success") for r in measured):
        mismatches.append("measured_row_failure")

    # arm order balance (full latin square still locked even if E1b/E1c not timed)
    order_counts = defaultdict(int)
    for sid in ids:
        order_counts[tuple(lock["measurement_schedule"]["arm_order_by_sample"][sid])] += 1
    if len(order_counts) != 4:
        mismatches.append("arm_order_not_latin_square_balanced")

    # accuracy
    active_arms = ("E0", "E1a") if parity_fail_retain else ARMS
    preds = {a: load_jsonl(pred_paths[a]) for a in active_arms}
    acc = {a: acc_from_preds(rows) for a, rows in preds.items()}
    for a, rows in preds.items():
        if len(rows) != 2629:
            mismatches.append(f"pred_n_{a}:{len(rows)}")
        ids_p = [r["sample_id"] for r in rows]
        if len(ids_p) != len(set(ids_p)):
            mismatches.append(f"duplicate_ids_{a}")

    deltas = {a: paired_delta(preds["E0"], preds[a], n_boot=10000, seed=42) for a in active_arms if a != "E0"}
    gate = pol["accuracy_gate"]
    acc_pass = {}
    for a in [x for x in active_arms if x != "E0"]:
        d = deltas[a]
        worst = 0.0
        s0 = acc["E0"]["signer_top1"]
        s1 = acc[a]["signer_top1"]
        if s0 and s1:
            worst = min(100.0 * (s1.get(s, 0) - s0.get(s, 0)) for s in s0)
        top5_delta_pp = 100.0 * (acc[a]["top5"] - acc["E0"]["top5"])
        macro_delta_pp = 100.0 * (acc[a]["macro_f1"] - acc["E0"]["macro_f1"])
        ok = (
            d["ci95_lo"] >= gate["hierarchical_ci95_lower_top1_pp_min"]
            and top5_delta_pp >= gate["delta_top5_pp_min"]
            and macro_delta_pp >= gate["delta_macro_f1_pp_min"]
            and worst >= gate["per_signer_delta_top1_pp_min"]
        )
        acc_pass[a] = {
            "pass": ok,
            "delta": d,
            "top5_delta_pp": top5_delta_pp,
            "macro_delta_pp": macro_delta_pp,
            "worst_signer_delta_pp": worst,
        }
    acc_pass["E0"] = {"pass": True, "note": "baseline"}
    for a in ("E1b", "E1c"):
        if a not in active_arms:
            acc_pass[a] = {"pass": False, "reason": "token_parity_SOURCE_GAP"}

    # efficiency aggregates
    eff_arms = tuple(sorted({r["arm"] for r in measured}))
    sample_e2e = {a: sample_medians([r for r in measured if r["arm"] == a], "e2e_ms") for a in eff_arms}
    for a in eff_arms:
        if set(sample_e2e[a]) != set(ids):
            mismatches.append(f"efficiency_sample_coverage_{a}")

    campaign_median = {a: median([sample_e2e[a][sid] for sid in ids]) for a in eff_arms}
    campaign_p90 = {
        a: p90([float(r["e2e_ms"]) for r in measured if r["arm"] == a and r.get("success")]) for a in eff_arms
    }
    vram = {
        a: median([float(r["peak_allocated_vram_mb"]) for r in measured if r["arm"] == a and r.get("success")])
        for a in eff_arms
    }

    paired = {}
    for a in eff_arms:
        base = [sample_e2e["E0"][sid] for sid in ids]
        cand = [sample_e2e[a][sid] for sid in ids]
        paired[a] = bootstrap_speedup(base, cand, n_boot=10000, seed=42)

    # factorial effects only if full 2x2 present
    factorial_effects = None
    if set(eff_arms) >= set(ARMS):
        def arm_vec(a: str) -> np.ndarray:
            return np.asarray([sample_e2e[a][sid] for sid in ids], dtype=np.float64)

        factorial_effects = {
            "precision_amp_saves_approx_median": float(
                np.median(0.5 * (arm_vec("E0") + arm_vec("E1b")) - 0.5 * (arm_vec("E1a") + arm_vec("E1c")))
            ),
            "decode_batching_saves_approx_median": float(
                np.median(0.5 * (arm_vec("E0") + arm_vec("E1a")) - 0.5 * (arm_vec("E1b") + arm_vec("E1c")))
            ),
            "interaction_approx_median": float(
                np.median(0.5 * (arm_vec("E0") + arm_vec("E1c")) - 0.5 * (arm_vec("E1a") + arm_vec("E1b")))
            ),
            "note": "Positive value ≈ ms saved by the factor (median of paired sample contrasts).",
        }
    else:
        factorial_effects = {
            "status": "INCOMPLETE_2x2",
            "eff_arms": list(eff_arms),
            "note": "E1b/E1c rejected by token parity; cannot estimate decode/batching main effect or interaction.",
        }

    # eligibility
    eligible = []
    rule = lock["base_selection_rule_frozen_before_metrics"]
    for a in eff_arms:
        if a in ("E1b", "E1c"):
            continue
        if a not in acc_pass or not acc_pass[a]["pass"]:
            continue
        if a == "E0":
            eligible.append(a)
            continue
        p90_reg = campaign_p90[a] > campaign_p90["E0"] * 1.0000001
        vram_reg = vram[a] > vram["E0"] * 1.0000001 and rule.get("efficiency_gate_from_e4_policy", {}).get(
            "forbid_vram_increase_vs_e0", True
        )
        ci_ok = paired[a]["ci95_lo"] > 1.0
        if p90_reg or vram_reg or not ci_ok:
            continue
        eligible.append(a)

    # selection
    selected = None
    if eligible:
        meds = {a: campaign_median[a] for a in eligible}
        best = min(meds.values())
        close = [a for a, m in meds.items() if abs(m - best) / max(best, 1e-9) < 0.01]
        if len(close) > 1:
            order = rule["if_medians_within_1pct_prefer_simpler_order"]
            selected = sorted(close, key=lambda x: order.index(x) if x in order else 99)[0]
        else:
            selected = min(meds, key=meds.get)

    # stage medians
    stages = {}
    for a in eff_arms:
        stages[a] = {
            k: median([float(r[k]) for r in measured if r["arm"] == a and r.get("success")])
            for k in ("decode_ms", "preprocess_ms", "slowfast_roi_ms", "head_ms", "e2e_ms")
        }

    expected_measured = len(ids) * len(eff_arms) * n_meas
    if len(measured) != expected_measured:
        # replace earlier mismatch if present
        mismatches = [m for m in mismatches if not m.startswith("measured_rows:")]
        mismatches.append(f"measured_rows:{len(measured)}!={expected_measured}")

    terminal = "E1_BLOCKED_SOURCE_GAP"
    if parity_fail_retain:
        terminal = "E1_DECODE_PARITY_NO_GO"
    elif len(mismatches) == 0 and parity.get("pass"):
        if selected and selected != "E0":
            terminal = "E1_FACTORIAL_PASS_READY_FOR_E2E3"
        else:
            terminal = "E1_PASS_NO_SYSTEM_OPTIMIZATION_SELECTED"

    out.update(
        {
            "pass": len(mismatches) == 0 and (parity.get("pass", False) or parity_fail_retain),
            "parity_pass": parity.get("pass", False),
            "parity_fail_retain_e0_e1a": parity_fail_retain,
            "recomputed": {
                "accuracy": acc,
                "deltas_vs_E0": deltas,
                "accuracy_gate": acc_pass,
                "campaign_e2e_median_ms": campaign_median,
                "campaign_e2e_p90_ms": campaign_p90,
                "paired_vs_E0": paired,
                "vram_median_mb": vram,
                "stage_medians_ms": stages,
                "factorial_main_effects_ms": factorial_effects,
                "measured_row_count": len(measured),
                "expected_measured_row_count": expected_measured,
            },
            "eligible_arms": eligible,
            "selected_system_base": selected,
            "terminal_state_candidate": terminal,
            "slow_compression_opened": False,
        }
    )

    # write factorial report create-only if missing
    report_path = OUT / "E4_B4SC_V1_E1_FACTORIAL_REPORT.json"
    if not report_path.is_file():
        report_path.write_text(
            json.dumps(
                {
                    "schema": "ms_sta_v6_e4_b4sc_v1_e1_factorial_report_v1",
                    "run_id": run_id,
                    "e4_ready": False,
                    "slow_compression_opened": False,
                    "accuracy_wording": "no observed regression on the locked validation set",
                    "recomputed_from_raw": True,
                    **out["recomputed"],
                    "eligible_arms": eligible,
                    "selected_system_base": selected,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return out


def main() -> int:
    ver = run_verify()
    out = OUT / "E4_B4SC_V1_E1_INDEPENDENT_VERIFY.json"
    if out.is_file():
        print("exists", out)
        return 2
    out.write_text(json.dumps(ver, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("VERIFY_PASS" if ver.get("pass") else "VERIFY_FAIL", ver.get("selected_system_base"), ver.get("terminal_state_candidate"))
    return 0 if ver.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
