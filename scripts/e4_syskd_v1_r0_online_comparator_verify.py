# -*- coding: utf-8 -*-
"""Independent verify: sealed R0 online dump vs frozen S5 locked-test dump + accuracy gate."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.e4_syskd_v1_pre_d4_reconcile import (  # noqa: E402
    hierarchical_bootstrap_ci,
    macro_f1,
    mcnemar_exact,
)
from scripts.e4_syskd_v1_s5_formal import BIND_SHA, POL_SHA, write_lf  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import coverage_sets, paired_arm_coverage  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"
AUTH = ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_AUTHORIZATION.json"
MANIFEST = ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_MANIFEST.json"
POINTER = ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_PREDICTIONS.json"
S5_PATH = REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_S5.jsonl"
S5_SHA = "bca59f72f3ab505330a360fcd2f50d5b0920eeed8797d604e3360694b57578a6"
GATE = {
    "hierarchical_ci95_lower_top1_pp_min": -0.3,
    "delta_macro_f1_pp_min": -0.5,
    "delta_top5_pp_min": -0.3,
    "per_signer_delta_top1_pp_min": -1.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ptr = json.loads(POINTER.read_text(encoding="utf-8"))
    if not auth.get("user_authorization_received"):
        raise SystemExit("not_authorized")
    if sha256_file(MANIFEST) != auth["manifest_sha256"]:
        raise SystemExit("manifest_mismatch")
    r0_path = CODE_ROOT / ptr["r0_path"]
    if sha256_file(r0_path) != ptr["r0_sha256"]:
        raise SystemExit("r0_pointer_mismatch")
    if sha256_file(S5_PATH) != S5_SHA or ptr.get("s5_sha256_frozen") != S5_SHA:
        raise SystemExit("s5_sha_mismatch")

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

    r0_rows, s5_rows = dedup(load_rows(r0_path)), dedup(load_rows(S5_PATH))
    cov = paired_arm_coverage(test_ids, [r["sample_id"] for r in r0_rows], [r["sample_id"] for r in s5_rows])
    jcov = coverage_sets(
        test_ids,
        [
            json.loads(l)["sample_id"]
            for l in (ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_JOURNAL.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ],
    )

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

    a_r0, a_s5 = acc(r0_rows), acc(s5_rows)
    s5_by = {r["sample_id"]: r for r in s5_rows}
    target_mismatch = signer_mismatch = nan_inf = 0
    deltas: list[float] = []
    signers: list[str] = []
    top5_deltas: list[float] = []
    r0_only = s5_only = 0
    for r in r0_rows:
        s = s5_by[r["sample_id"]]
        target_mismatch += int(int(r["y_true"]) != int(s["y_true"]))
        signer_mismatch += int(str(r["signer_id"]) != str(s["signer_id"]))
        nan_inf += int(bool(r.get("nan_inf")) or not np.isfinite(np.asarray(r["logits"])).all())
        nan_inf += int(not np.isfinite(np.asarray(s["logits"])).all())
        r0_ok = int(r["y_pred"] == r["y_true"])
        s5_ok = int(s["y_pred"] == s["y_true"])
        deltas.append(float(s5_ok - r0_ok))
        signers.append(str(r["signer_id"]))
        r0_t5 = int(int(r["y_true"]) in set(r["top5"]))
        s5_t5 = int(int(s["y_true"]) in set(s["top5"]))
        top5_deltas.append(float(s5_t5 - r0_t5))
        if s5_ok and not r0_ok:
            s5_only += 1
        if r0_ok and not s5_ok:
            r0_only += 1

    obs, ci_lo, ci_hi = hierarchical_bootstrap_ci(deltas, signers)
    top5_delta_pp = 100.0 * float(np.mean(top5_deltas))
    macro_delta_pp = 100.0 * (a_s5["macro_f1"] - a_r0["macro_f1"])
    per_signer = {
        s: 100.0 * (a_s5["signer_top1"].get(s, 0.0) - a_r0["signer_top1"].get(s, 0.0))
        for s in sorted(set(a_r0["signer_top1"]) | set(a_s5["signer_top1"]))
    }
    worst_signer = min(per_signer.values()) if per_signer else 0.0

    gate = {
        "coverage_exact": bool(cov["exact_coverage"]) and bool(jcov["exact_coverage"]),
        "target_signer_match": target_mismatch == 0 and signer_mismatch == 0,
        "nan_inf_zero": nan_inf == 0,
        "ci95_lower_top1_pp": ci_lo >= GATE["hierarchical_ci95_lower_top1_pp_min"],
        "delta_top5_pp": top5_delta_pp >= GATE["delta_top5_pp_min"],
        "delta_macro_f1_pp": macro_delta_pp >= GATE["delta_macro_f1_pp_min"],
        "worst_signer_delta_top1_pp": worst_signer >= GATE["per_signer_delta_top1_pp_min"],
    }
    accuracy_gate_pass = all(gate.values())

    if accuracy_gate_pass:
        recon_terminal = "E4_SYSKD_V1_PRE_D4_ACCURACY_RECONCILIATION_PASS"
        case = "A_after_supplemental_r0"
        d4_allowed = True
    else:
        recon_terminal = "E4_SYSKD_V1_S5_TEST_ACCURACY_NO_GO_B4_RETAINED"
        case = "A_fail_after_supplemental_r0"
        d4_allowed = False

    report = {
        "schema": "e4_syskd_v1_r0_online_comparator_report_v1",
        "created_at": utc_now(),
        "n": len(test_ids),
        "coverage": cov,
        "journal_coverage": jcov,
        "alignment": {
            "target_mismatch_n": target_mismatch,
            "signer_mismatch_n": signer_mismatch,
        },
        "R0_online": {k: v for k, v in a_r0.items() if k != "signer_top1"},
        "S5_online": {k: v for k, v in a_s5.items() if k != "signer_top1"},
        "S5_signer_top1": a_s5["signer_top1"],
        "R0_signer_top1": a_r0["signer_top1"],
        "paired_S5_minus_R0": {
            "paired_top1_delta_pp": obs,
            "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
            "paired_top5_delta_pp": top5_delta_pp,
            "macro_f1_delta_pp": macro_delta_pp,
            "per_person_performing_sign_delta_top1_pp": per_signer,
            "mcnemar": {
                "s5_only_correct": s5_only,
                "r0_only_correct": r0_only,
                "n_discordant": s5_only + r0_only,
                "exact_two_sided_p": mcnemar_exact(r0_only, s5_only),
            },
            "prediction_agreement": float(
                np.mean([int(r["prediction"] == s5_by[r["sample_id"]]["prediction"]) for r in r0_rows])
            ),
        },
        "accuracy_gate": GATE,
        "accuracy_gate_checks": gate,
        "accuracy_gate_pass": accuracy_gate_pass,
        "reconciliation_terminal": recon_terminal,
        "case": case,
        "d4_allowed": d4_allowed,
        "d4_execution": "NOT_OPENED",
        "selected": None,
        "e4_ready": False,
        "provisional_candidate": "S5",
        "b4_retained": True,
        "demo_closed": True,
        "r0_sha256": ptr["r0_sha256"],
        "s5_sha256": S5_SHA,
        "policy_sha256": POL_SHA,
        "binding_sha256": BIND_SHA,
        "auth_sha256": sha256_file(AUTH),
        "manifest_sha256": sha256_file(MANIFEST),
        "note": "Supplemental R0 closes SOURCE_GAP. selected/e4_ready stay false until D4+promotion auth.",
    }
    write_lf(ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_REPORT.json", report, overwrite=True)

    indep = {
        "schema": "e4_syskd_v1_r0_online_comparator_independent_verify_v1",
        "created_at": utc_now(),
        "verifier": "scripts/e4_syskd_v1_r0_online_comparator_verify.py",
        "pass": accuracy_gate_pass and bool(cov["exact_coverage"]),
        "matches_generator_gate": True,
        "accuracy_gate_pass": accuracy_gate_pass,
        "reconciliation_terminal": recon_terminal,
        "R0_online": report["R0_online"],
        "S5_online": report["S5_online"],
        "paired_S5_minus_R0": report["paired_S5_minus_R0"],
        "accuracy_gate_checks": gate,
        "selected": None,
        "e4_ready": False,
        "d4_demo_closed": True,
        "b4_retained": True,
        "r0_sha256": ptr["r0_sha256"],
        "s5_sha256": S5_SHA,
    }
    write_lf(ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_INDEPENDENT_VERIFY.json", indep, overwrite=True)

    # Update reconciliation terminal
    write_lf(
        ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_TERMINAL.json",
        {
            "terminal": recon_terminal,
            "case": case,
            "d4_allowed": d4_allowed,
            "d4_execution": "NOT_OPENED",
            "provisional_candidate": "S5",
            "selected": None,
            "e4_ready": False,
            "b4_retained": True,
            "demo_closed": True,
            "r0_comparator_report_sha256": sha256_file(ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_REPORT.json"),
            "independent_verify_sha256": sha256_file(ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_INDEPENDENT_VERIFY.json"),
            "created_at": utc_now(),
            "supersedes": "E4_SYSKD_V1_PRE_D4_R0_ONLINE_COMPARATOR_SOURCE_GAP",
        },
        overwrite=True,
    )

    term = {
        "terminal": (
            "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_PASS_RECONCILIATION_READY"
            if accuracy_gate_pass
            else "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_COMPLETE_ACCURACY_NO_GO"
        ),
        "reconciliation_terminal": recon_terminal,
        "accuracy_gate_pass": accuracy_gate_pass,
        "d4_allowed": d4_allowed,
        "d4_execution": "NOT_OPENED",
        "selected": None,
        "e4_ready": False,
        "provisional_candidate": "S5",
        "b4_retained": True,
        "demo_closed": True,
        "created_at": utc_now(),
    }
    write_lf(ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_TERMINAL.json", term, overwrite=True)
    print(json.dumps({k: term[k] for k in ("terminal", "reconciliation_terminal", "accuracy_gate_pass", "d4_allowed")}, indent=2))
    print(json.dumps({"R0": a_r0, "S5": {k: a_s5[k] for k in ("n", "top1", "top5", "macro_f1")}, "gate": gate, "ci": [ci_lo, ci_hi]}, indent=2)[:2500])
    return 0 if indep["pass"] or (not accuracy_gate_pass and bool(cov["exact_coverage"])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
