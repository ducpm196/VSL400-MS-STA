# -*- coding: utf-8 -*-
"""Independent R4B validator (create-only). Does not re-run SlowFast bench.

Checks: SHA binds, trial counts, stage-sum recomputation, gate recomputation,
promoted list, INVALIDATED run isolation, incremental VRAM scope labels.
Requires separate VRAM accounting verify artifact before full-corpus unlock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4"
LEDGER = OUT / "E4_R4B_RUN_LEDGER.json"
PROTOCOL = OUT / "E4_R4B_STAGE_MATCHED_PROTOCOL.json"
RAW = OUT / "E4_R4B_RAW_RESULTS.jsonl"
REPORT = OUT / "E4_R4B_STAGE_BREAKDOWN_REPORT.json"
CLOSEOUT = OUT / "E4_R4B_CLOSEOUT.json"
BENCH = ROOT / "scripts/bench_ms_sta_v6_e4_r4b_stage_matched.py"
VALIDATOR_OUT = OUT / "E4_R4B_INDEPENDENT_VALIDATOR.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median(xs: list[float]) -> float:
    # Match bench_ms_sta_v6_e4_r4b_stage_matched.median (upper element for even n).
    ys = sorted(xs)
    return float(ys[len(ys) // 2]) if ys else float("nan")


def p90(xs: list[float]) -> float:
    ys = sorted(xs)
    return float(ys[max(0, int(0.9 * (len(ys) - 1)))]) if ys else float("nan")


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def bootstrap_speedup_ci(b4: list[float], e4: list[float], n_boot: int, seed: int) -> dict:
    # Match numpy Generator path used by the bench for CI stability checks on gate outcomes.
    import numpy as np

    rng = np.random.default_rng(seed)
    b = np.asarray(b4, dtype=np.float64)
    c = np.asarray(e4, dtype=np.float64)
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    speeds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mb, mc = float(np.median(b[idx])), float(np.median(c[idx]))
        speeds.append((mb - mc) / max(mb, 1e-9))
    lo, hi = np.quantile(speeds, [0.025, 0.975])
    return {"speedup_median": float(np.median(speeds)), "ci95_lo": float(lo), "ci95_hi": float(hi)}


def write_create_only(path: Path, obj: dict) -> str:
    if path.is_file():
        raise SystemExit(f"refuse_overwrite:{path}")
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram-verify-rel", default="Docs/results/ms_sta_v6_e4_r4/E4_R4B_VRAM_ACCOUNTING_VERIFY.json")
    args = ap.parse_args()

    checks: list[dict] = []
    fail = False

    def add(name: str, ok: bool, detail: dict | None = None) -> None:
        nonlocal fail
        checks.append({"check": name, "pass": bool(ok), "detail": detail or {}})
        if not ok:
            fail = True

    # --- ledger + SHA binds ---
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in ledger["runs"]}
    add("ledger_has_three_runs", set(runs) == {"run1", "run2", "run3"})
    add("run1_invalidated", runs["run1"]["label"] == "INVALIDATED" and runs["run1"]["invalidation_code"] == "INVALID_DATASET_RESOLVER")
    add("run2_invalidated", runs["run2"]["label"] == "INVALIDATED" and runs["run2"]["invalidation_code"] == "INVALID_VRAM_ACCOUNTING")
    add("run3_canonical", runs["run3"]["label"] == "CANONICAL_STAGE_MATCHED_PASS")
    add("run1_not_primary", runs["run1"]["include_in_primary_result_table"] is False)
    add("run2_not_primary", runs["run2"]["include_in_primary_result_table"] is False)
    add("run3_is_primary", runs["run3"]["include_in_primary_result_table"] is True)

    for key, art in runs["run3"]["artifacts"].items():
        p = ROOT / art["rel"]
        ok = p.is_file() and sha256_file(p) == art["sha256"]
        add(f"run3_sha_{key}", ok, {"rel": art["rel"], "expected": art["sha256"], "got": sha256_file(p) if p.is_file() else None})

    for key, art in runs["run2"]["artifacts"].items():
        p = ROOT / art["rel"]
        ok = p.is_file() and sha256_file(p) == art["sha256"]
        add(f"run2_sha_{key}", ok, {"rel": art["rel"]})

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))

    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    add("raw_total_rows_132", len(rows) == 132, {"n": len(rows)})

    ctr = Counter((r.get("cache_mode"), r.get("model_id")) for r in rows)
    expected_models = ["B4_V6_CTR", "R0_S_A_d128", "R1_S_A_d128", "R2_S_A_d128"]
    trial_ok = True
    for mid in expected_models:
        if ctr[("hot", mid)] != 24 or ctr[("cold", mid)] != 6 or ctr[("warmup", mid)] != 3:
            trial_ok = False
    add("trial_counts_hot24_cold6_warmup3", trial_ok, {"counter": {f"{a}|{b}": c for (a, b), c in sorted(ctr.items())}})

    # shared sample IDs across models within hot trials
    hot_by_model = defaultdict(list)
    for r in rows:
        if r.get("cache_mode") == "hot":
            hot_by_model[r["model_id"]].append(r)
    sample_sets = {mid: [x["sample_id"] for x in recs] for mid, recs in hot_by_model.items()}
    same_order = all(sample_sets[mid] == sample_sets["B4_V6_CTR"] for mid in expected_models)
    add("hot_sample_order_matched_across_models", same_order)

    # stage timing recomputation
    stage_tol_ms = 1e-3
    stage_ok = True
    bad_stage = []
    for r in rows:
        if r.get("cache_mode") not in {"hot", "cold"}:
            continue
        parts = [
            float(r["video_decode_ms"]),
            float(r["rgb_representation_ms"]),
            float(r["skeleton_representation_ms"]),
            float(r.get("h2d_ms") or 0.0),
            float(r["model_forward_ms"]),
        ]
        recomputed = sum(parts)
        if abs(recomputed - float(r["total_ms"])) > stage_tol_ms:
            stage_ok = False
            bad_stage.append({"sample_id": r["sample_id"], "model_id": r["model_id"], "recomputed": recomputed, "total": r["total_ms"]})
    add("stage_sum_equals_total", stage_ok, {"n_bad": len(bad_stage), "examples": bad_stage[:5]})

    # incremental VRAM scope labels on canonical raw
    scopes = {r.get("vram_scope") for r in rows if r.get("cache_mode") == "hot"}
    add(
        "canonical_vram_scope_incremental_labels",
        scopes == {
            "incremental_b4_exclusive_extract_through_forward",
            "incremental_e4_exclusive_compact_through_forward",
        },
        {"scopes": sorted(scopes)},
    )

    # code formula presence (static)
    bench_src = BENCH.read_text(encoding="utf-8")
    add("bench_resets_peak_memory_stats", "reset_peak_memory_stats" in bench_src)
    add("bench_sync_before_base_alloc", "torch.cuda.synchronize()" in bench_src and "memory_allocated()" in bench_src)
    add(
        "bench_incremental_formula_peak_minus_base",
        "max_memory_allocated() - b4_base_alloc" in bench_src and "max_memory_allocated() - e4_base_alloc" in bench_src,
    )
    add(
        "bench_does_not_compare_e4_to_b4_absolute_without_incremental",
        "vram_scope" in bench_src and "incremental_" in bench_src,
    )
    # canonical raw lacks allocated/reserved telemetry → must be covered by verify artifact
    has_alloc_fields = any("allocated_before_mb" in r for r in rows)
    add("canonical_raw_lacks_allocated_reserved_fields_expected", has_alloc_fields is False)

    # recompute gate from hot
    gate = protocol["gate"]
    b4_totals = [float(r["total_ms"]) for r in hot_by_model["B4_V6_CTR"]]
    b4_med = median(b4_totals)
    b4_p90 = p90(b4_totals)
    b4_vram = median([float(r["peak_vram_mb"]) for r in hot_by_model["B4_V6_CTR"]])
    b4_rss = median([float(r["rss_delta_mb"]) for r in hot_by_model["B4_V6_CTR"]])

    recomputed_rows = []
    for mid in ["R0_S_A_d128", "R1_S_A_d128", "R2_S_A_d128"]:
        recs = hot_by_model[mid]
        totals = [float(r["total_ms"]) for r in recs]
        med = median(totals)
        speedup = (b4_med - med) / max(b4_med, 1e-9)
        ci = bootstrap_speedup_ci(b4_totals, totals, n_boot=int(gate["n_bootstrap"]), seed=int(gate["bootstrap_seed"]))
        reasons = []
        if speedup < float(gate["primary_total_median_speedup_min"]):
            reasons.append("total_speedup_lt_8pct")
        if ci["ci95_lo"] <= 0:
            reasons.append("ci_lo_le_0")
        if p90(totals) > b4_p90:
            reasons.append("p90_regress")
        vram = median([float(r["peak_vram_mb"]) for r in recs])
        rss = median([float(r["rss_delta_mb"]) for r in recs])
        if (vram - b4_vram) / max(b4_vram, 1e-9) > float(gate["vram_regress_max_frac"]):
            reasons.append("vram_regress")
        if (rss - b4_rss) / max(abs(b4_rss), 1e-9) > float(gate["rss_regress_max_frac"]) and rss > b4_rss:
            reasons.append("rss_regress")
        recomputed_rows.append(
            {
                "candidate_id": mid,
                "total_median_ms": med,
                "speedup_vs_b4": speedup,
                "bootstrap_ci": ci,
                "gate_pass": not reasons,
                "reject_reasons": reasons,
            }
        )

    # match report medians within tolerance
    report_by_id = {r["candidate_id"]: r for r in report["candidate_rows"]}
    med_match = True
    for rr in recomputed_rows:
        pr = report_by_id[rr["candidate_id"]]
        if abs(pr["total_median_ms"] - rr["total_median_ms"]) > 1e-6:
            med_match = False
        if bool(pr["gate_pass"]) != bool(rr["gate_pass"]):
            med_match = False
    add("report_matches_recomputed_gates", med_match, {"recomputed": recomputed_rows})

    passers = [r for r in recomputed_rows if r["gate_pass"]]
    passers = sorted(passers, key=lambda r: (-r["speedup_vs_b4"], r["candidate_id"]))
    promoted = [r["candidate_id"] for r in passers[: int(gate["promote_max"])]]
    add("promoted_exact_R1", promoted == ["R1_S_A_d128"], {"promoted": promoted})
    add("closeout_promoted_exact_R1", closeout.get("promoted_candidate_ids") == ["R1_S_A_d128"])
    add("report_promoted_exact_R1", report.get("promoted") == ["R1_S_A_d128"])
    add("closeout_verdict_pass", closeout.get("verdict") == "E4_R4B_STAGE_MATCHED_PASS")

    # device/protocol bind
    add("protocol_gpu_rtx2060", protocol.get("required_gpu_name") == "NVIDIA GeForce RTX 2060")
    add("report_gpu_rtx2060", (report.get("environment") or {}).get("gpu") == "NVIDIA GeForce RTX 2060")
    add("batch1_fp32", protocol.get("benchmark", {}).get("batch_size") == 1 and protocol.get("benchmark", {}).get("dtype") == "fp32")

    # VRAM verify artifact required
    vram_path = ROOT / args.vram_verify_rel
    if not vram_path.is_file():
        add("vram_accounting_verify_present", False, {"path": args.vram_verify_rel})
    else:
        vram = json.loads(vram_path.read_text(encoding="utf-8"))
        add("vram_accounting_verify_present", True, {"sha256": sha256_file(vram_path)})
        add("vram_verify_status_pass", vram.get("status") == "PASS", {"status": vram.get("status")})
        add("vram_verify_records_allocated_and_reserved", bool(vram.get("records_allocated_and_reserved")))
        add("vram_verify_formula", vram.get("formula") == "peak_during_stage_minus_allocated_immediately_before_stage")
        add("vram_verify_r1_still_pass_vs_b4_incremental", bool(vram.get("r1_incremental_vram_gate_pass")))

    status = "PASS" if not fail else "FAIL"
    payload = {
        "schema": "ms_sta_v6_e4_r4b_independent_validator_v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "n_fail": sum(1 for c in checks if not c["pass"]),
        "promoted_candidate_ids": promoted,
        "forbid_full_corpus_unless_pass": True,
        "e4_ready": False,
    }
    write_create_only(VALIDATOR_OUT, payload)
    print("VALIDATOR", status, "n_fail", payload["n_fail"])
    for c in checks:
        if not c["pass"]:
            print("FAIL", c["check"], c.get("detail"))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
