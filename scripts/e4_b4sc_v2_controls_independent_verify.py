# -*- coding: utf-8 -*-
"""Independent verifier for E4_B4SC_V2 C0–C2 — recomputes PASS/FAIL from raw reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
AMEND = OUT / "E4_B4SC_V2_CONTROLS_THRESHOLD_AMENDMENT.json"
VERIFY = OUT / "E4_B4SC_V2_CONTROLS_INDEPENDENT_VERIFY.json"


def main() -> int:
    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    th0 = amend["C0_exact_clone"]
    th2 = amend["C2_kd_mechanics"]
    gaps = []
    mismatches = []

    def load(name: str):
        p = OUT / name
        if not p.is_file():
            gaps.append(name)
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    c0 = load("E4_B4SC_V2_C0_REPORT.json")
    c1 = load("E4_B4SC_V2_C1_REPORT.json")
    c2 = load("E4_B4SC_V2_C2_REPORT.json")
    term = load("E4_B4SC_V2_CONTROLS_TERMINAL.json")
    plan = load("E4_B4SC_V2_C1_IDENTITY_PLAN.json")

    out = {
        "schema": "ms_sta_v6_e4_b4sc_v2_controls_independent_verify_v1",
        "e4_ready": False,
        "source_gaps": gaps,
        "mismatches": mismatches,
    }
    if gaps and c0 is None:
        out["pass"] = False
        out["terminal_state"] = "BLOCKED_SOURCE_GAP"
        VERIFY.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print("VERIFY_BLOCKED", gaps)
        return 2

    # recompute C0
    c0_pass = False
    if c0:
        p = c0["probe"]
        v = c0["val_prediction_parity"]
        c0_pass = (
            bool(c0.get("hash_compare_pass"))
            and float(p["logit_max_abs_diff_max"]) <= float(th0["logit_max_abs_diff"])
            and float(p["logit_cosine_mean"]) >= float(th0["logit_cosine_min"])
            and all(float(c) >= float(th0["stage_feature_cosine_min"]) for c in p["stage_cosine_mean"])
            and all(float(x) >= float(th0["roi_token_cosine_min_per_region"]) for x in p["roi_cosine_mean"].values())
            and float(p["fast_max_abs_diff_max"]) <= float(th0["fast_pathway_max_abs_diff"])
            and float(v["match_rate"]) == 1.0
        )
        if bool(c0.get("pass")) != c0_pass:
            mismatches.append("c0_report_pass_disagree")

    c1_pass = None
    if not c0_pass:
        expected = "V2_C0_CODEPATH_NO_GO"
    elif c1 is None:
        gaps.append("missing_c1_after_c0_pass")
        expected = "BLOCKED_SOURCE_GAP"
    else:
        p = c1["probe_vs_C0"]
        v = c1["val_prediction_parity"]
        id_ok = not c1.get("identity_issues")
        if plan:
            for si in range(4):
                st = plan["stages"][str(si)]
                if st["out_idx"] != list(range(st["teacher_out"])):
                    mismatches.append(f"plan_out_not_identity_{si}")
                if st["inner_idx"] != list(range(st["teacher_inner"])):
                    mismatches.append(f"plan_inner_not_identity_{si}")
        c1_pass = (
            id_ok
            and bool(c1.get("hash_compare_vs_C0_pass"))
            and float(p["logit_max_abs_diff_max"]) <= float(th0["logit_max_abs_diff"])
            and float(p["logit_cosine_mean"]) >= float(th0["logit_cosine_min"])
            and all(float(c) >= float(th0["stage_feature_cosine_min"]) for c in p["stage_cosine_mean"])
            and all(float(x) >= float(th0["roi_token_cosine_min_per_region"]) for x in p["roi_cosine_mean"].values())
            and float(p["fast_max_abs_diff_max"]) <= float(th0["fast_pathway_max_abs_diff"])
            and float(v["match_rate"]) == 1.0
        )
        if bool(c1.get("pass")) != c1_pass:
            mismatches.append("c1_report_pass_disagree")
        if not c1_pass:
            expected = "V2_C1_TRANSPLANT_NO_GO"
        elif c2 is None:
            gaps.append("missing_c2_after_c1_pass")
            expected = "BLOCKED_SOURCE_GAP"
        else:
            r = c2["result"]
            c2_pass = (
                bool(r["adapter_in_optimizer"])
                and float(r["grad_adapter_l2_max"]) >= float(th2["adapter_grad_l2_min"])
                and float(r["grad_slow_l2_max"]) >= float(th2["slow_student_grad_l2_min"])
                and float(r["fast_param_max_abs_delta"]) <= float(th2["fast_param_max_abs_delta"])
                and float(r["loss_reduction_rel"]) >= float(th2["kd_stage_loss_reduction_min_rel"])
                and float(r["loss_last"]) < float(r["loss0"])
                and float(r["cosine_increase"]) >= float(th2["stage_cosine_mean_increase_min"])
                and float(r["cosine_after"]) >= float(th2["final_mean_stage_cosine_min_absolute"])
            )
            if bool(c2.get("pass")) != c2_pass:
                mismatches.append("c2_report_pass_disagree")
            expected = "V2_CONTROLS_PASS_S5_READY" if c2_pass else "V2_C2_KD_MECHANICS_NO_GO"

    if term and term.get("terminal_state") != expected and expected != "BLOCKED_SOURCE_GAP":
        mismatches.append(f"terminal_disagree:{term.get('terminal_state')}!={expected}")

    out.update(
        {
            "C0_pass_recomputed": c0_pass,
            "C1_pass_recomputed": c1_pass,
            "terminal_expected": expected,
            "terminal_reported": None if not term else term.get("terminal_state"),
            "pass": len(gaps) == 0 and len(mismatches) == 0,
            "S5_ready": expected == "V2_CONTROLS_PASS_S5_READY",
            "e4_ready": False,
        }
    )
    VERIFY.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("E4_B4SC_V2_CONTROLS_VERIFY", expected, "pass=", out["pass"])
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
