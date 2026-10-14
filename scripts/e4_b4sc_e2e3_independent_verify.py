# -*- coding: utf-8 -*-
"""Independent verifier for E4_B4SC_V1 E2/E3 — recomputes from raw artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e4_b4sc_e1_independent_verify import (  # noqa: E402
    acc_from_preds,
    bootstrap_speedup,
    load_jsonl,
    median,
    p90,
    paired_delta,
    sample_medians,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_b4sc.narrow_slowfast import E2_WIDTHS, E3_WIDTHS  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POL_TRAIN = OUT / "E4_B4SC_V1_E2E3_TRAINING_POLICY.json"
VERIFY_PATH = OUT / "E4_B4SC_V1_E2E3_INDEPENDENT_VERIFY.json"


def main() -> int:
    gaps: list[str] = []
    mismatches: list[str] = []
    tpol = json.loads(POL_TRAIN.read_text(encoding="utf-8"))
    chan = json.loads((OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json").read_text(encoding="utf-8"))
    e2_audit = json.loads((OUT / "E4_B4SC_V1_E2_ARCHITECTURE_AUDIT.json").read_text(encoding="utf-8"))
    e3_audit = json.loads((OUT / "E4_B4SC_V1_E3_ARCHITECTURE_AUDIT.json").read_text(encoding="utf-8"))

    # architecture widths from lock + audit
    e2_w = list(chan["E2"]["slow_widths"])
    e3_w = list(chan["E3"]["slow_widths"])
    if e2_w != list(E2_WIDTHS.slow_widths) or e2_w != [224, 448, 896, 1792]:
        mismatches.append(f"e2_widths:{e2_w}")
    if e3_w != list(E3_WIDTHS.slow_widths) or e3_w != [192, 384, 768, 1536]:
        mismatches.append(f"e3_widths:{e3_w}")
    if e2_audit.get("slow_stage_widths") != e2_w:
        mismatches.append("e2_audit_width_mismatch")
    if e3_audit.get("slow_stage_widths") != e3_w:
        mismatches.append("e3_audit_width_mismatch")

    acc_path = OUT / "E4_B4SC_V1_E2_SCREEN_ACCURACY_REPORT.json"
    pred_e2 = OUT / "E4_B4SC_V1_E2_SCREEN_E2_PREDICTIONS.jsonl"
    pred_e0 = OUT / "E4_B4SC_V1_E1_E0_PREDICTIONS.jsonl"
    eff_raw = OUT / "E4_B4SC_V1_E2_EFFICIENCY_RAW.jsonl"
    eff_rep = OUT / "E4_B4SC_V1_E2_EFFICIENCY_REPORT.json"

    if not acc_path.is_file():
        gaps.append("missing_e2_accuracy_report")
    if not pred_e2.is_file():
        gaps.append("missing_e2_predictions")
    if not pred_e0.is_file():
        gaps.append("missing_e0_predictions")

    out: dict[str, Any] = {
        "schema": "ms_sta_v6_e4_b4sc_v1_e2e3_independent_verify_v1",
        "e4_ready": False,
        "training_policy_sha256": sha256_file(POL_TRAIN),
        "channel_lock_sha256": sha256_file(OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json"),
        "architecture": {
            "E2_slow_widths": e2_w,
            "E3_slow_widths": e3_w,
            "E2_deployed_params": e2_audit.get("params", {}).get("deployed_total_approx"),
            "E3_deployed_params": e3_audit.get("params", {}).get("deployed_total_approx"),
            "channel_lock_file_sha256": sha256_file(OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json"),
        },
        "mismatches": mismatches,
        "source_gaps": gaps,
    }

    if gaps:
        out["pass"] = False
        out["terminal_state"] = "BLOCKED_SOURCE_GAP"
        out["provisional_candidate"] = None
        VERIFY_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("E4_B4SC_E2E3_VERIFY_BLOCKED", gaps, flush=True)
        return 2

    e0_rows = load_jsonl(pred_e0)
    e2_rows = load_jsonl(pred_e2)
    if len(e2_rows) != 2629:
        mismatches.append(f"e2_pred_n:{len(e2_rows)}")
    e0_by = {r["sample_id"]: r for r in e0_rows}
    e0_aligned = [e0_by[r["sample_id"]] for r in e2_rows]
    acc_e0 = acc_from_preds(e0_aligned)
    acc_e2 = acc_from_preds(e2_rows)
    delta = paired_delta(e0_aligned, e2_rows, n_boot=10000, seed=42)
    gate = tpol["accuracy_gate"]
    top5_delta_pp = 100.0 * (acc_e2["top5"] - acc_e0["top5"])
    macro_delta_pp = 100.0 * (acc_e2["macro_f1"] - acc_e0["macro_f1"])
    s0, s2 = acc_e0["signer_top1"], acc_e2["signer_top1"]
    worst_signer = min((100.0 * (s2.get(s, 0) - s0[s]) for s in s0), default=0.0) if s0 else 0.0
    acc_pass = (
        delta["ci95_lo"] >= gate["hierarchical_ci95_lower_top1_pp_min"]
        and top5_delta_pp >= gate["delta_top5_pp_min"]
        and macro_delta_pp >= gate["delta_macro_f1_pp_min"]
        and worst_signer >= gate["per_signer_delta_top1_pp_min"]
    )
    report_acc = json.loads(acc_path.read_text(encoding="utf-8"))
    if bool(report_acc.get("accuracy_gate_pass")) != acc_pass:
        mismatches.append("accuracy_gate_report_disagree")

    out["accuracy"] = {
        "E0": acc_e0,
        "E2": acc_e2,
        "paired_delta": delta,
        "top5_delta_pp": top5_delta_pp,
        "macro_delta_pp": macro_delta_pp,
        "worst_signer_delta_pp": worst_signer,
        "accuracy_gate_pass": acc_pass,
    }

    e3_opened = False
    e3_acc_pass = None
    eff_pass = None
    e3_eff_pass = None
    terminal = None
    candidate = None

    if not acc_pass:
        terminal = "E2_ACCURACY_NO_GO_E3_NOT_OPENED"
        candidate = None
    else:
        if not eff_raw.is_file() or not eff_rep.is_file():
            gaps.append("missing_e2_efficiency")
            terminal = "BLOCKED_SOURCE_GAP"
        else:
            raw = load_jsonl(eff_raw)
            measured = [r for r in raw if r.get("measured") and r.get("success")]
            ids = sorted({r["sample_id"] for r in measured})
            sample_e2e = {
                a: sample_medians([r for r in measured if r["arm"] == a], "e2e_ms")
                for a in ("E0", "E1a", "E2")
            }
            campaign_median = {a: median([sample_e2e[a][sid] for sid in ids]) for a in sample_e2e}
            campaign_p90 = {
                a: p90([float(r["e2e_ms"]) for r in measured if r["arm"] == a]) for a in sample_e2e
            }
            base = [sample_e2e["E0"][sid] for sid in ids]
            cand = [sample_e2e["E2"][sid] for sid in ids]
            paired = bootstrap_speedup(base, cand, n_boot=10000, seed=42)
            egate = tpol["efficiency_gate"]
            vram = {
                a: median([float(r["peak_allocated_vram_mb"]) for r in measured if r["arm"] == a])
                for a in sample_e2e
            }
            eff_pass = (
                paired["speedup_factor"] >= egate["e2e_median_speedup_vs_E0_min"]
                and paired["ci95_lo"] >= egate["e2e_speedup_ci95_lower_min"]
                and campaign_p90["E2"] <= campaign_p90["E0"] * 1.05
                and vram["E2"] <= vram["E0"] + 1e-6
            )
            out["efficiency"] = {
                "campaign_median_e2e_ms": campaign_median,
                "campaign_p90_e2e_ms": campaign_p90,
                "paired_E2_vs_E0": paired,
                "structured_compression_saving_vs_E1a_ms": campaign_median["E1a"] - campaign_median["E2"],
                "total_saving_vs_E0_ms": campaign_median["E0"] - campaign_median["E2"],
                "vram_mb": vram,
                "efficiency_gate_pass": eff_pass,
            }
            # E3 legality
            e3_may_open = acc_pass and (not eff_pass)
            e3_acc_path = OUT / "E4_B4SC_V1_E3_ACCURACY_REPORT.json"
            if e3_acc_path.is_file():
                e3_opened = True
                e3_acc_pass = bool(json.loads(e3_acc_path.read_text(encoding="utf-8")).get("accuracy_gate_pass"))
            if e3_opened and not e3_may_open:
                mismatches.append("e3_opened_illegally")
            if not e3_opened and e3_may_open:
                # may be terminal E2_EFFICIENCY_NO_GO_E3_READY if E3 not yet run
                terminal = "E2_EFFICIENCY_NO_GO_E3_READY"
                candidate = None
            elif not e3_opened and eff_pass:
                terminal = "E2_PROVISIONAL_PASS_E3_NOT_OPENED"
                candidate = "E2"
            elif e3_opened:
                e3_eff = OUT / "E4_B4SC_V1_E3_EFFICIENCY_REPORT.json"
                if e3_eff.is_file():
                    e3_eff_pass = bool(json.loads(e3_eff.read_text(encoding="utf-8")).get("efficiency_gate_pass"))
                if e3_acc_pass and e3_eff_pass:
                    terminal = "E3_PROVISIONAL_PASS"
                    candidate = "E3"
                elif e3_acc_pass and not e3_eff_pass:
                    terminal = "E2_E3_EFFICIENCY_NO_GO"
                    candidate = None
                else:
                    terminal = "E2_E3_EFFICIENCY_NO_GO" if acc_pass else "E2_ACCURACY_NO_GO_E3_NOT_OPENED"
                    candidate = "E2" if acc_pass and not e3_acc_pass else None

    out["mismatches"] = mismatches
    out["source_gaps"] = gaps
    out["pass"] = len(gaps) == 0 and len(mismatches) == 0
    out["terminal_state"] = terminal
    out["provisional_candidate"] = candidate
    out["e3_opened"] = e3_opened
    out["conditional_e3_legal"] = bool(acc_pass and (eff_pass is False))
    VERIFY_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("E4_B4SC_E2E3_VERIFY", terminal, "candidate=", candidate, "pass=", out["pass"], flush=True)
    return 0 if out["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
