# -*- coding: utf-8 -*-
"""Independent PRE-D4 reconciliation verifier (does not import reconcile report builder)."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import coverage_sets  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"
ONESHOT = CODE_ROOT / "Docs/results/ms_sta_v6_oneshot"
REPORT = ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_REPORT.json"
OUT = ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_VERIFY.json"
CKPT = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"


def macro_f1(y_true: list[int], y_pred: list[int], n_classes: int = 400) -> float:
    yt = np.asarray(y_true, np.int64)
    yp = np.asarray(y_pred, np.int64)
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((yp == c) & (yt == c)))
        fp = int(np.sum((yp == c) & (yt != c)))
        fn = int(np.sum((yp != c) & (yt == c)))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def hierarchical_bootstrap_ci(deltas: list[float], signers: list[str], n_boot: int = 10000, seed: int = 42):
    by = {}
    for d, s in zip(deltas, signers):
        by.setdefault(s, []).append(d)
    keys = sorted(by)
    observed = 100.0 * (sum(deltas) / len(deltas))
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        drawn = []
        for _s in range(len(keys)):
            sk = keys[rng.randrange(len(keys))]
            pool = by[sk]
            for _i in range(len(pool)):
                drawn.append(pool[rng.randrange(len(pool))])
        boots.append(100.0 * (sum(drawn) / len(drawn)))
    boots.sort()
    return observed, boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]


def load_b4():
    out = {}
    for line in (ONESHOT / "TEST_PREDS_ALIGNED.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        b4 = row["models"]["b4_v6_ctr"]
        out[str(row["sample_id"])] = {
            "y_true": int(row["label"]),
            "y_pred": int(b4["pred"]),
            "top5": [int(x) for x in b4["top5_ids"]],
            "signer_id": str(row["signer_id"]),
        }
    return out


def load_online(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["sample_id"])] = {
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "top5": [int(x) for x in row["top5"]],
            "signer_id": str(row["signer_id"]),
            "checkpoint_sha256": str(row.get("checkpoint_sha256") or ""),
        }
    return out


def metrics(ids, table):
    y = [table[i]["y_true"] for i in ids]
    p = [table[i]["y_pred"] for i in ids]
    t5 = [int(table[i]["y_true"] in set(table[i]["top5"])) for i in ids]
    return {
        "n": len(ids),
        "top1": float(np.mean([int(a == b) for a, b in zip(y, p)])),
        "top5": float(np.mean(t5)),
        "macro_f1": macro_f1(y, p),
    }


def main() -> int:
    gen = json.loads(REPORT.read_text(encoding="utf-8"))
    ptr = json.loads((ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_RAW_POINTER.json").read_text(encoding="utf-8"))
    raw = CODE_ROOT / ptr["path"]
    if sha256_file(raw) != ptr["sha256"]:
        raise SystemExit("raw_sha_mismatch")
    binding = json.loads((ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json").read_text(encoding="utf-8"))
    data_root = Path(binding["canonical_data_root"])
    ids = [
        str(x)
        for x in json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))[
            "test_ids"
        ]
    ]
    b4, r1, s5 = load_b4(), load_online(REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_R1.jsonl"), load_online(
        REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_S5.jsonl"
    )
    cov_ok = (
        coverage_sets(ids, list(b4)).get("exact_coverage")
        and coverage_sets(ids, list(r1)).get("exact_coverage")
        and coverage_sets(ids, list(s5)).get("exact_coverage")
    )
    m_b4, m_r1, m_s5 = metrics(ids, b4), metrics(ids, r1), metrics(ids, s5)
    d1, signers = [], []
    for sid in ids:
        pc = int(s5[sid]["y_pred"] == s5[sid]["y_true"])
        bc = int(b4[sid]["y_pred"] == b4[sid]["y_true"])
        d1.append(float(pc - bc))
        signers.append(s5[sid]["signer_id"])
        if r1[sid]["checkpoint_sha256"] != CKPT or s5[sid]["checkpoint_sha256"] != CKPT:
            raise SystemExit("ckpt")
        if b4[sid]["y_true"] != s5[sid]["y_true"] or r1[sid]["y_pred"] != s5[sid]["y_pred"]:
            # target must match; r1/s5 preds must match for parity check
            if b4[sid]["y_true"] != s5[sid]["y_true"]:
                raise SystemExit("target_mismatch")
    # r1==s5 preds
    if any(r1[sid]["y_pred"] != s5[sid]["y_pred"] for sid in ids):
        r1_s5_agree = False
    else:
        r1_s5_agree = True
    mean_pp, lo, hi = hierarchical_bootstrap_ci(d1, signers)
    delta = {
        "top1": 100.0 * (m_s5["top1"] - m_b4["top1"]),
        "top5": 100.0 * (m_s5["top5"] - m_b4["top5"]),
        "macro_f1": 100.0 * (m_s5["macro_f1"] - m_b4["macro_f1"]),
    }
    match = (
        abs(m_b4["top1"] - gen["metrics"]["B4_canonical_cache"]["top1"]) < 1e-12
        and abs(m_s5["top1"] - gen["metrics"]["S5_online"]["top1"]) < 1e-12
        and abs(delta["top1"] - gen["observed_delta_pp_S5_minus_B4"]["top1"]) < 1e-9
        and abs(lo - gen["paired"]["S5_minus_B4"]["hierarchical_ci95_top1_pp"][0]) < 1e-9
        and abs(hi - gen["paired"]["S5_minus_B4"]["hierarchical_ci95_top1_pp"][1]) < 1e-9
        and gen["terminal"] == "E4_SYSKD_V1_PRE_D4_R0_ONLINE_COMPARATOR_SOURCE_GAP"
        and gen["d4_allowed"] is False
        and r1_s5_agree
        and cov_ok
    )
    # raw line count
    n_raw = sum(1 for l in raw.read_text(encoding="utf-8").splitlines() if l.strip())
    out = {
        "schema": "e4_syskd_v1_pre_d4_reconciliation_verify_v1",
        "verifier": "scripts/e4_syskd_v1_pre_d4_reconcile_verify.py",
        "pass": bool(match) and n_raw == 6165,
        "matches_generator_report": bool(match),
        "coverage_exact": bool(cov_ok),
        "r1_s5_prediction_agreement": r1_s5_agree,
        "metrics": {"B4": m_b4, "R1": m_r1, "S5": m_s5},
        "observed_delta_pp_S5_minus_B4": delta,
        "hierarchical_ci95_top1_pp": [lo, hi],
        "terminal": gen["terminal"],
        "case": gen["case"],
        "d4_allowed": False,
        "selected": None,
        "e4_ready": False,
        "demo_closed": True,
        "b4_retained": True,
        "report_sha256": sha256_file(REPORT),
        "raw_sha256": ptr["sha256"],
    }
    raw_out = (json.dumps(out, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    OUT.write_bytes(raw_out)
    print(json.dumps({k: out[k] for k in ("pass", "terminal", "observed_delta_pp_S5_minus_B4", "hierarchical_ci95_top1_pp")}, indent=2))
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
