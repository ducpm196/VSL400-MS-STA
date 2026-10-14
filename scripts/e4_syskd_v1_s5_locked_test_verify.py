# -*- coding: utf-8 -*-
"""Independent locked-test metrics verifier (generator ≠ this process)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.e4_syskd_v1_s5_formal import BIND_SHA, POL_SHA, macro_f1, write_lf  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import paired_arm_coverage  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"
POINTER = ART / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS.json"
MANIFEST = ART / "E4_SYSKD_V1_S5_LOCKED_TEST_MANIFEST.json"
AUTH = ART / "E4_SYSKD_V1_TEST_OPENING_AUTHORIZATION.json"
REPORT = ART / "E4_SYSKD_V1_S5_LOCKED_TEST_REPORT.json"
OUT = ART / "E4_SYSKD_V1_S5_LOCKED_TEST_INDEPENDENT_VERIFY.json"
EXPECTED_TIP = "b52c23b0e3598bd78731c4e3e28c7dbb5e5eb8f2"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    ptr = json.loads(POINTER.read_text(encoding="utf-8"))
    gen = json.loads(REPORT.read_text(encoding="utf-8"))
    r1_path = CODE_ROOT / ptr["r1_path"]
    s5_path = CODE_ROOT / ptr["s5_path"]
    if sha256_file(r1_path) != ptr["r1_sha256"] or sha256_file(s5_path) != ptr["s5_sha256"]:
        raise SystemExit("pointer_sha_mismatch")
    if sha256_file(MANIFEST) != auth.get("locked_test_manifest_sha256"):
        raise SystemExit("manifest_auth_mismatch")

    binding = json.loads((ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json").read_text(encoding="utf-8"))
    data_root = Path(binding["canonical_data_root"])
    test_ids = list(
        json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))[
            "test_ids"
        ]
    )

    def dedup(rows: list[dict]) -> list[dict]:
        by = {r["sample_id"]: r for r in rows}
        return [by[sid] for sid in test_ids if sid in by]

    r1_rows, s5_rows = dedup(load_rows(r1_path)), dedup(load_rows(s5_path))
    cov = paired_arm_coverage(test_ids, [r["sample_id"] for r in r1_rows], [r["sample_id"] for r in s5_rows])

    def acc(rows: list[dict]) -> dict:
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
    s5_by = {r["sample_id"]: r for r in s5_rows}
    abs_max = 0.0
    abs_sum = 0.0
    abs_n = 0
    agree = top5_agree = nan_inf = 0
    for r in r1_rows:
        s = s5_by[r["sample_id"]]
        d = np.abs(np.asarray(r["logits"], np.float32) - np.asarray(s["logits"], np.float32))
        abs_max = max(abs_max, float(d.max()))
        abs_sum += float(d.sum())
        abs_n += int(d.size)
        agree += int(r["prediction"] == s["prediction"])
        top5_agree += int(set(r["top5"]) == set(s["top5"]))
        nan_inf += int(
            not np.isfinite(np.asarray(r["logits"])).all() or not np.isfinite(np.asarray(s["logits"])).all()
        )
    mean_abs = abs_sum / max(abs_n, 1)
    fidelity_ok = (
        bool(cov["exact_coverage"])
        and abs_max <= 1e-4
        and mean_abs <= 1e-5
        and agree == len(test_ids)
        and top5_agree == len(test_ids)
        and nan_inf == 0
    )
    match_gen = (
        abs(a_r1["top1"] - float(gen["R1"]["top1"])) < 1e-12
        and abs(a_s5["top1"] - float(gen["S5"]["top1"])) < 1e-12
        and abs(a_r1["macro_f1"] - float(gen["R1"]["macro_f1"])) < 1e-12
        and abs(a_s5["macro_f1"] - float(gen["S5"]["macro_f1"])) < 1e-12
        and float(gen["logit_max_abs"]) == abs_max
        and bool(gen["pass"]) == (fidelity_ok and bool(cov["exact_coverage"]))
    )
    out = {
        "schema": "e4_syskd_v1_s5_locked_test_independent_verify_v1",
        "verifier": "scripts/e4_syskd_v1_s5_locked_test_verify.py",
        "n": len(test_ids),
        "coverage": cov,
        "R1": {k: v for k, v in a_r1.items() if k != "signer_top1"},
        "S5": {k: v for k, v in a_s5.items() if k != "signer_top1"},
        "S5_signer_top1": a_s5["signer_top1"],
        "logit_max_abs": abs_max,
        "logit_mean_abs": mean_abs,
        "prediction_agreement": agree / len(test_ids),
        "nan_inf": nan_inf,
        "fidelity_pass": fidelity_ok,
        "matches_generator_report": match_gen,
        "pass": fidelity_ok and match_gen and bool(cov["exact_coverage"]),
        "policy_sha256": POL_SHA,
        "binding_sha256": BIND_SHA,
        "code_sha": EXPECTED_TIP,
        "selected": None,
        "e4_ready": False,
        "d4_demo_closed": True,
    }
    write_lf(OUT, out, overwrite=True)
    print(json.dumps({k: out[k] for k in ("pass", "matches_generator_report", "fidelity_pass", "logit_max_abs", "R1", "S5")}, indent=2))
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
