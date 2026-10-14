# -*- coding: utf-8 -*-
"""Recompute S5 val accuracy after set-extra precedence fix (raw dumps unchanged)."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "Docs/results/e4_system_kd_v1"
REG = ROOT / "Docs/results/e4_system_kd_v1_registry"
MAIN = Path(r"D:/TTNT/DNTA/VSL_Translation_Project")
BIND = "ed86157ddcbcb0dff8d1d454f9d572a409b9b5f50c4169b87fc16b34938a69eb"
POL = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
CKPT = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_lf(path: Path, obj: dict) -> str:
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def macro_f1(y_true, y_pred, n_classes: int = 400) -> float:
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


def main() -> int:
    split = json.loads((MAIN / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))
    val_ids = list(split["val_ids"])
    r1_path = REG / "E4_SYSKD_V1_S5_R1_VAL_PREDICTIONS.jsonl"
    s5_path = REG / "E4_SYSKD_V1_S5_VAL_PREDICTIONS.jsonl"
    r1_rows = [json.loads(l) for l in r1_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    s5_rows = [json.loads(l) for l in s5_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids_r1 = [r["sample_id"] for r in r1_rows]
    ids_s5 = [r["sample_id"] for r in s5_rows]
    from src.optimization.e4_syskd.coverage_sets import paired_arm_coverage

    cov = paired_arm_coverage(val_ids, ids_r1, ids_s5)
    missing = list(cov["union_missing"])
    extra = list(cov["union_extra"])
    dup = int(cov["arm_a"]["duplicate_count"]) + int(cov["arm_b"]["duplicate_count"])

    abs_max = 0.0
    abs_sum = 0.0
    abs_n = 0
    agree = top5_agree = nan_inf = 0
    s5_by = {r["sample_id"]: r for r in s5_rows}
    for r in r1_rows:
        s = s5_by[r["sample_id"]]
        a = np.asarray(r["logits"], dtype=np.float32)
        b = np.asarray(s["logits"], dtype=np.float32)
        d = np.abs(a - b)
        abs_max = max(abs_max, float(d.max()))
        abs_sum += float(d.sum())
        abs_n += int(d.size)
        agree += int(r["prediction"] == s["prediction"])
        top5_agree += int(set(r["top5"]) == set(s["top5"]))
        nan_inf += int(not np.isfinite(a).all() or not np.isfinite(b).all())

    def acc(rows):
        y = [int(r["y_true"]) for r in rows]
        p = [int(r["y_pred"]) for r in rows]
        top5 = [int(int(r["y_true"]) in set(r["top5"])) for r in rows]
        by_s: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_s[str(r["signer_id"])].append(int(r["y_pred"] == r["y_true"]))
        return {
            "n": len(rows),
            "top1": float(np.mean([int(a == b) for a, b in zip(y, p)])),
            "top5": float(np.mean(top5)),
            "macro_f1": macro_f1(y, p),
            "signer_top1": {s: float(np.mean(v)) for s, v in by_s.items() if s},
        }

    a_r1, a_s5 = acc(r1_rows), acc(s5_rows)
    ba = {r["sample_id"]: int(r["y_pred"] == r["y_true"]) for r in r1_rows}
    ca = {r["sample_id"]: int(r["y_pred"] == r["y_true"]) for r in s5_rows}
    keys = sorted(set(ba) & set(ca))
    diffs = np.asarray([100.0 * (ca[i] - ba[i]) for i in keys], dtype=np.float64)
    rng = np.random.default_rng(42)
    boots = [float(diffs[rng.integers(0, len(diffs), size=len(diffs))].mean()) for _ in range(10000)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    mean_delta = float(diffs.mean())
    delta_top5_pp = 100.0 * (a_s5["top5"] - a_r1["top5"])
    delta_f1_pp = 100.0 * (a_s5["macro_f1"] - a_r1["macro_f1"])
    signer_deltas = {
        s: 100.0 * (a_s5["signer_top1"].get(s, 0) - a_r1["signer_top1"].get(s, 0))
        for s in set(a_r1["signer_top1"]) | set(a_s5["signer_top1"])
    }
    worst_signer = min(signer_deltas.values()) if signer_deltas else 0.0
    mean_abs = abs_sum / max(abs_n, 1)
    exact = abs_max == 0.0 and mean_abs == 0.0
    fidelity_ok = (
        abs_max <= 1e-4
        and mean_abs <= 1e-5
        and (agree / 2629) == 1.0
        and (top5_agree / 2629) == 1.0
        and nan_inf == 0
    )
    cov_ok = len(r1_rows) == 2629 and len(s5_rows) == 2629 and dup == 0 and not missing and not extra
    gate_ok = (
        cov_ok
        and fidelity_ok
        and lo >= -0.3
        and delta_f1_pp >= -0.5
        and delta_top5_pp >= -0.3
        and worst_signer >= -1.0
    )

    report = {
        "schema": "e4_syskd_v1_s5_val_accuracy_report_v1",
        "created_at": utc_now(),
        "n": 2629,
        "coverage": "2629/2629" if cov_ok else f"{len(r1_rows)}/{len(s5_rows)}",
        "duplicate": dup,
        "missing": missing,
        "extra": extra,
        "nan_inf": nan_inf,
        "R1": a_r1,
        "S5": {k: v for k, v in a_s5.items() if k != "signer_top1"},
        "S5_signer_top1": a_s5["signer_top1"],
        "paired_delta_top1_pp": mean_delta,
        "paired_ci95_top1_pp": [float(lo), float(hi)],
        "delta_top5_pp": delta_top5_pp,
        "delta_macro_f1_pp": delta_f1_pp,
        "worst_person_performing_sign_delta_top1_pp": worst_signer,
        "per_person_performing_sign_delta_top1_pp": signer_deltas,
        "prediction_agreement": agree / 2629,
        "top5_set_agreement": top5_agree / 2629,
        "logit_max_abs": abs_max,
        "logit_mean_abs": mean_abs,
        "EXACT_LOGIT_PARITY": exact,
        "pass": gate_ok,
        "coverage_bugfix": "operator_precedence_on_set_extra_recomputed_from_raw_dumps",
        "r1_predictions_sha256": sha256_file(r1_path),
        "s5_predictions_sha256": sha256_file(s5_path),
        "binding_sha256": BIND,
        "policy_sha256": POL,
        "checkpoint_sha256": CKPT,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_VAL_ACCURACY_REPORT.json", report)
    verify = {
        "schema": "e4_syskd_v1_s5_val_independent_verify_v1",
        "created_at": utc_now(),
        "recomputed_n_r1": len(r1_rows),
        "recomputed_n_s5": len(s5_rows),
        "recomputed_top1_r1": a_r1["top1"],
        "recomputed_top1_s5": a_s5["top1"],
        "recomputed_logit_max_abs": abs_max,
        "recomputed_extra_count": len(extra),
        "recomputed_missing_count": len(missing),
        "pass": gate_ok,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_VAL_INDEPENDENT_VERIFY.json", verify)

    parity = json.loads((ART / "E4_SYSKD_V1_S5_PARITY_REPORT.json").read_text(encoding="utf-8"))
    eff = json.loads((ART / "E4_SYSKD_V1_S5_EFFICIENCY_REPORT.json").read_text(encoding="utf-8"))
    if not parity.get("pass"):
        term, prov = "E4_SYSKD_V1_S5_PARITY_NO_GO", None
    elif not gate_ok:
        term, prov = "E4_SYSKD_V1_S5_ACCURACY_NO_GO", None
    elif not eff.get("pass"):
        term, prov = "E4_SYSKD_V1_S5_EFFICIENCY_NO_GO", None
    else:
        term, prov = "E4_SYSKD_V1_S5_VAL_PROMOTION_READY", "S5"
    out = {
        "schema": "e4_syskd_v1_s5_decision_v1",
        "created_at": utc_now(),
        "terminal": term,
        "provisional_candidate": prov,
        "selected": None,
        "e4_ready": False,
        "parity_pass": bool(parity.get("pass")),
        "accuracy_pass": gate_ok,
        "efficiency_pass": bool(eff.get("pass")),
        "b4_system_of_record": True,
        "note": "selected remains null until confirmation/test protocol. Accuracy report recomputed after set-extra precedence fix; raw logits unchanged.",
    }
    write_lf(ART / "E4_SYSKD_V1_S5_TERMINAL.json", out)
    print(
        json.dumps(
            {
                "terminal": term,
                "pass": gate_ok,
                "exact": exact,
                "extra": len(extra),
                "top1": a_r1["top1"],
                "worst_signer": worst_signer,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
