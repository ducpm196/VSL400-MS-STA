# -*- coding: utf-8 -*-
"""B4Q-V1 close-out independent verifier.

Recomputes Gate A / efficiency from raw jsonl; token drift from ladder parity rows;
sealed accuracy / alignment / signer deltas from freshly dumped prediction rows.
Online prediction rows were not persisted by the ladder process (SOURCE_GAP).
This script verifies Gate A / efficiency / sealed accuracy independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.b4q_run_gate_a_and_native_ladder import (  # noqa: E402
    eval_val_sealed,
    hierarchical_boot_delta,
)
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_b4q"
POL = ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_create_only(path: Path, obj: dict | list) -> None:
    if path.is_file():
        raise SystemExit(f"create_only_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, list):
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in obj), encoding="utf-8")
    else:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def p90(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return float(s[int(0.9 * (len(s) - 1))])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dump_pred_jsonl(path: Path, rows: list[dict], arm: str) -> None:
    if path.is_file():
        raise SystemExit(f"create_only_exists:{path}")
    with path.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            rec = dict(r)
            rec["arm"] = arm
            rec["idx"] = i
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def metrics_from_rows(rows: list[dict]) -> dict:
    n = len(rows)
    top1 = sum(int(r["top1"]) for r in rows) / max(n, 1)
    top5 = sum(int(r["top5"]) for r in rows) / max(n, 1)
    signer_hits: dict[str, list[int]] = {}
    for r in rows:
        signer_hits.setdefault(str(r.get("signer") or ""), []).append(int(r["top1"]))
    signer_top1 = {s: float(np.mean(v)) for s, v in signer_hits.items() if v}
    return {"n": n, "top1": top1, "top5": top5, "signer_top1": signer_top1}


def alignment(base: list[dict], cand: list[dict]) -> dict:
    assert len(base) == len(cand)
    agree = sum(int(b["pred"] == c["pred"]) for b, c in zip(base, cand))
    y_ok = all(int(b["y"]) == int(c["y"]) for b, c in zip(base, cand))
    return {
        "n": len(base),
        "pred_agreement_rate": agree / max(len(base), 1),
        "label_alignment_ok": y_ok,
    }


def signer_deltas_pp(base_m: dict, cand_m: dict) -> dict:
    out = {}
    for s, t1 in base_m["signer_top1"].items():
        if s in cand_m["signer_top1"]:
            out[s] = 100.0 * (cand_m["signer_top1"][s] - t1)
    return out


def recompute_efficiency(raw_path: Path, report: dict) -> tuple[dict, bool]:
    by: dict[str, list[dict]] = {}
    for r in load_jsonl(raw_path):
        by.setdefault(r["arm"], []).append(r)
    recomp = {}
    ok = True
    for arm, eff in (report.get("efficiency") or {}).items():
        rows = by.get(arm) or []
        totals = [float(r["total_ms"]) for r in rows]
        rgb = [float(r["rgb_representation_ms"]) for r in rows]
        entry = {
            "n": len(rows),
            "total_median_ms": median(totals) if totals else None,
            "total_p90_ms": p90(totals) if totals else None,
            "rgb_median_ms": median(rgb) if rgb else None,
        }
        recomp[arm] = entry
        for k in ("n", "total_median_ms", "total_p90_ms", "rgb_median_ms"):
            a, b = eff[k], entry[k]
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > 1e-6:
                    ok = False
            elif a != b:
                ok = False
    return recomp, ok


def recompute_gate_a(raw_path: Path, profile: dict) -> tuple[dict, bool]:
    rows = load_jsonl(raw_path)
    keys = [
        "video_decode_ms",
        "rgb_representation_ms",
        "skeleton_representation_ms",
        "h2d_ms",
        "model_forward_ms",
        "total_ms",
    ]
    recomputed = {k: median([float(r[k]) for r in rows]) for k in keys}
    stages = profile["stage_medians_ms"]
    ok = all(abs(recomputed[k] - stages[k]) < 1e-9 for k in keys) and len(rows) == int(profile["n_rows"])
    return recomputed, ok


def token_drift_from_report(report: dict) -> dict:
    q2 = []
    cl = []
    for row in report.get("parity", {}).get("rows") or []:
        if "q2_vs_q0" in row:
            q2.append(row["q2_vs_q0"])
        if "q2_channels_last_vs_q0" in row:
            cl.append(row["q2_channels_last_vs_q0"])
    def summarize(xs: list[dict]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "max_of_max_abs": max(float(x["max_abs"]) for x in xs),
            "mean_of_mean_abs": float(np.mean([float(x["mean_abs"]) for x in xs])),
            "all_finite_pass_flags": all(bool(x.get("pass")) for x in xs),
            "role": "SUPPORTING_DIAGNOSTIC_RAW_DRIFT",
            "not_formal_policy_token_magnitude_gate": True,
        }
    return {"q2_vs_q0": summarize(q2), "q2_channels_last_vs_q0": summarize(cl)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["static", "sealed", "all"], default="all")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-name", type=str, default="", help="Create-only verify JSON filename under b4q/")
    ap.add_argument("--pred-prefix", type=str, default="", help="Create-only sealed pred filename prefix")
    args = ap.parse_args()

    pol = json.loads(POL.read_text(encoding="utf-8"))
    report_path = OUT / "B4Q_V1_NATIVE_LADDER_REPORT.json"
    gate_raw = OUT / "B4Q_V1_GATE_A_RAW.jsonl"
    gate_prof = OUT / "B4Q_V1_GATE_A_STAGE_PROFILE.json"
    eff_raw = OUT / "B4Q_V1_NATIVE_EFFICIENCY_RAW.jsonl"
    out_path = OUT / "B4Q_V1_CLOSEOUT_INDEPENDENT_VERIFY.json"
    if args.out_name:
        out_path = OUT / args.out_name
    if out_path.is_file():
        raise SystemExit(f"create_only_exists:{out_path}")

    pred_prefix = args.pred_prefix or "B4Q_V1_SEALED_PRED"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate_a_med, gate_a_ok = recompute_gate_a(gate_raw, json.loads(gate_prof.read_text(encoding="utf-8")))
    eff_recomp, eff_ok = recompute_efficiency(eff_raw, report)
    token = token_drift_from_report(report)

    payload: dict = {
        "schema": "ms_sta_v6_b4q_v1_closeout_independent_verify_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ladder_report_sha256": sha256_file(report_path),
        "gate_a_raw_sha256": sha256_file(gate_raw),
        "efficiency_raw_sha256": sha256_file(eff_raw),
        "gate_a": {"pass": gate_a_ok, "recomputed_stage_medians_ms": gate_a_med},
        "efficiency": {"pass": eff_ok, "recomputed_efficiency_core": eff_recomp},
        "token_parity_diagnostic": token,
        "token_threshold_note": "SUPPORTING_DIAGNOSTIC_NOT_POLICY_FROZEN — see B4Q_V1_CLOSEOUT_ERRATA.json",
        "accuracy_sealed": None,
        "accuracy_online": {
            "status": "SOURCE_GAP_PREDICTION_ROWS_NOT_PERSISTED_BY_LADDER",
            "note": "Original ladder kept online prediction rows in-memory only. Sealed path is independently re-verified; online aggregates remain in B4Q_V1_NATIVE_LADDER_REPORT.json.",
        },
        "verdict_consistent": report.get("verdict") == "B4Q_V1_EFFICIENCY_NO_GO",
        "pass": False,
    }

    if args.phase in ("sealed", "all"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
        ckpt = torch.load(ROOT / pol["b4"]["checkpoint_rel"], map_location=device, weights_only=False)
        b4.load_state_dict(ckpt["model"])

        print("dump sealed Q0...", flush=True)
        q0 = eval_val_sealed(model=b4, device=device, pol=pol, amp_fp16=False)
        print("dump sealed Q1 inference...", flush=True)
        q1i = eval_val_sealed(model=b4, device=device, pol=pol, amp_fp16=False)
        print("dump sealed Q1 AMP...", flush=True)
        q1a = eval_val_sealed(model=b4, device=device, pol=pol, amp_fp16=True)

        p0 = OUT / f"{pred_prefix}_Q0.jsonl"
        p1i = OUT / f"{pred_prefix}_Q1_INFERENCE.jsonl"
        p1a = OUT / f"{pred_prefix}_Q1_AMP.jsonl"
        dump_pred_jsonl(p0, q0["rows"], "Q0_B4_EAGER_FP32")
        dump_pred_jsonl(p1i, q1i["rows"], "Q1_HEAD_INFERENCE_MODE_FP32")
        dump_pred_jsonl(p1a, q1a["rows"], "Q1_HEAD_AMP_FP16")

        # Independent recompute from dumps (re-read)
        r0 = load_jsonl(p0)
        r1i = load_jsonl(p1i)
        r1a = load_jsonl(p1a)
        m0, m1i, m1a = metrics_from_rows(r0), metrics_from_rows(r1i), metrics_from_rows(r1a)
        d1i = hierarchical_boot_delta(r0, r1i, n_boot=args.n_boot, seed=42)
        d1a = hierarchical_boot_delta(r0, r1a, n_boot=args.n_boot, seed=42)
        s1i = signer_deltas_pp(m0, m1i)
        s1a = signer_deltas_pp(m0, m1a)
        g = pol["accuracy_gate"]

        def acc_pass(delta: dict, top5_b: float, top5_c: float, worst: float) -> bool:
            return (
                delta["mean_delta_top1_pp"] >= g["mean_delta_top1_pp_min"]
                and delta["ci95_lo"] >= g["hierarchical_ci95_lower_top1_pp_min"]
                and 100.0 * (top5_c - top5_b) >= g["delta_top5_pp_min"]
                and worst >= g["per_signer_delta_top1_pp_min"]
            )

        worst_i = min(s1i.values()) if s1i else 0.0
        worst_a = min(s1a.values()) if s1a else 0.0
        # Match report aggregates within tight tolerance
        rep_q0 = report["accuracy"]["Q0_B4_EAGER_FP32"]
        match_q0 = abs(m0["top1"] - rep_q0["top1"]) < 1e-12 and abs(m0["top5"] - rep_q0["top5"]) < 1e-12

        payload["accuracy_sealed"] = {
            "pred_artifacts": {
                "Q0": str(p0.relative_to(ROOT)).replace("\\", "/"),
                "Q1_INFERENCE": str(p1i.relative_to(ROOT)).replace("\\", "/"),
                "Q1_AMP": str(p1a.relative_to(ROOT)).replace("\\", "/"),
                "sha256": {
                    "Q0": sha256_file(p0),
                    "Q1_INFERENCE": sha256_file(p1i),
                    "Q1_AMP": sha256_file(p1a),
                },
            },
            "recomputed": {
                "Q0": m0,
                "Q1_INFERENCE": m1i,
                "Q1_AMP": m1a,
            },
            "alignment": {
                "Q1_INFERENCE_vs_Q0": alignment(r0, r1i),
                "Q1_AMP_vs_Q0": alignment(r0, r1a),
            },
            "delta_vs_q0": {"Q1_INFERENCE": d1i, "Q1_AMP": d1a},
            "signer_delta_top1_pp": {"Q1_INFERENCE": s1i, "Q1_AMP": s1a},
            "accuracy_gate_pass": {
                "Q1_INFERENCE": acc_pass(d1i, m0["top5"], m1i["top5"], worst_i),
                "Q1_AMP": acc_pass(d1a, m0["top5"], m1a["top5"], worst_a),
            },
            "matches_ladder_report_q0_aggregates": match_q0,
            "pass": bool(
                match_q0
                and acc_pass(d1i, m0["top5"], m1i["top5"], worst_i)
                and acc_pass(d1a, m0["top5"], m1a["top5"], worst_a)
                and alignment(r0, r1i)["label_alignment_ok"]
                and alignment(r0, r1a)["label_alignment_ok"]
            ),
        }

    sealed_ok = bool((payload.get("accuracy_sealed") or {}).get("pass"))
    payload["pass"] = bool(
        gate_a_ok and eff_ok and payload["verdict_consistent"] and (sealed_ok if args.phase in ("sealed", "all") else True)
    )
    payload["completeness"] = {
        "gate_a": gate_a_ok,
        "efficiency": eff_ok,
        "sealed_accuracy": sealed_ok if payload.get("accuracy_sealed") else False,
        "online_accuracy": False,
        "online_status": payload["accuracy_online"]["status"],
    }
    write_create_only(out_path, payload)
    print("WROTE", out_path, "pass", payload["pass"], flush=True)
    return 0 if payload["pass"] or args.phase == "static" else 1


if __name__ == "__main__":
    raise SystemExit(main())
