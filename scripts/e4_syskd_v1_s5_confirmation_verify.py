# -*- coding: utf-8 -*-
"""Independent confirmation verifier — recomputes from sealed dumps (not the generator)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / "Docs/results/e4_system_kd_v1"
MAIN = Path(r"D:/TTNT/DNTA/VSL_Translation_Project")

from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import paired_arm_coverage  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def verify_seed_accuracy(seed: int, val_ids: list[str]) -> dict:
    ptr = json.loads((ART / f"E4_SYSKD_V1_S5_CONFIRMATION_PREDICTIONS_SEED{seed}.json").read_text(encoding="utf-8"))
    r1_path = ROOT / ptr["r1_path"]
    s5_path = ROOT / ptr["s5_path"]
    assert sha256_file(r1_path) == ptr["r1_sha256"]
    assert sha256_file(s5_path) == ptr["s5_sha256"]
    r1 = [json.loads(l) for l in r1_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    s5 = [json.loads(l) for l in s5_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cov = paired_arm_coverage(val_ids, [r["sample_id"] for r in r1], [r["sample_id"] for r in s5])
    s5_by = {r["sample_id"]: r for r in s5}
    abs_max = 0.0
    agree = 0
    for r in r1:
        s = s5_by[r["sample_id"]]
        d = np.abs(np.asarray(r["logits"], np.float32) - np.asarray(s["logits"], np.float32))
        abs_max = max(abs_max, float(d.max()))
        agree += int(r["prediction"] == s["prediction"])
    top1 = float(np.mean([int(r["y_pred"] == r["y_true"]) for r in s5]))
    seed_rep = json.loads((ART / f"E4_SYSKD_V1_S5_CONFIRMATION_ACCURACY_SEED{seed}.json").read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "coverage_exact": bool(cov["exact_coverage"]),
        "recomputed_logit_max_abs": abs_max,
        "recomputed_top1_agreement": agree / max(len(r1), 1),
        "recomputed_s5_top1": top1,
        "matches_seed_report_top1": abs(top1 - float(seed_rep["S5"]["top1"])) < 1e-12,
        "dump_sha_match": True,
        "EXACT_LOGIT_PARITY": abs_max == 0.0,
    }


def verify_efficiency() -> dict:
    raw = ART / "E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_RAW.jsonl"
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_seed: dict[int, list] = defaultdict(list)
    for r in rows:
        if r.get("success"):
            by_seed[int(r["seed"])].append(r)
    out = {}
    for seed, rs in sorted(by_seed.items()):
        out[str(seed)] = {
            "n_ok": len(rs),
            "expected": 360,
            "coverage_ok": len(rs) == 360,
            "arms": sorted({r["arm"] for r in rs}),
        }
    return out


def main() -> int:
    val_ids = json.loads(
        (MAIN / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8")
    )["val_ids"]
    acc_rep = json.loads((ART / "E4_SYSKD_V1_S5_CONFIRMATION_ACCURACY_REPORT.json").read_text(encoding="utf-8"))
    eff_rep = json.loads((ART / "E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_REPORT.json").read_text(encoding="utf-8"))
    errata = json.loads((ART / "E4_SYSKD_V1_S5_AGGREGATION_ERRATA.json").read_text(encoding="utf-8"))
    cg = json.loads((ART / "E4_SYSKD_V1_CODEGRAPH_CAMPAIGN_AUDIT.json").read_text(encoding="utf-8"))
    per = [verify_seed_accuracy(s, val_ids) for s in (42, 123, 456)]
    eff_cov = verify_efficiency()
    t1 = [float(x["S5"]["top1"]) for x in acc_rep["per_seed"]]
    mean_ok = abs(float(np.mean(t1)) - float(acc_rep["mean_top1"])) < 1e-12
    verify = {
        "schema": "e4_syskd_v1_s5_confirmation_accuracy_verify_v1",
        "created_at": utc_now(),
        "implementation": "scripts/e4_syskd_v1_s5_confirmation_verify.py + coverage_sets",
        "not_the_generator": True,
        "per_seed": per,
        "mean_top1_matches_report": mean_ok,
        "errata_pass": bool(errata.get("pass")),
        "codegraph_pass": bool(cg.get("pass")),
        "pass": all(
            p["coverage_exact"] and p["recomputed_top1_agreement"] == 1.0 and p["EXACT_LOGIT_PARITY"] for p in per
        )
        and bool(acc_rep.get("pass"))
        and mean_ok,
    }
    (ART / "E4_SYSKD_V1_S5_CONFIRMATION_ACCURACY_VERIFY.json").write_text(
        json.dumps(verify, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    eff_verify = {
        "schema": "e4_syskd_v1_s5_confirmation_efficiency_verify_v1",
        "created_at": utc_now(),
        "per_seed_row_coverage": eff_cov,
        "report_pass": bool(eff_rep.get("pass")),
        "report_mean_speedup": eff_rep.get("mean_speedup"),
        "pass": all(v["coverage_ok"] for v in eff_cov.values()) and bool(eff_rep.get("pass")),
    }
    (ART / "E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_VERIFY.json").write_text(
        json.dumps(eff_verify, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"acc_verify_pass": verify["pass"], "eff_verify_pass": eff_verify["pass"]}, indent=2))
    return 0 if verify["pass"] and eff_verify["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
