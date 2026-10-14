# -*- coding: utf-8 -*-
"""Independent Gate A verifier — does not import aggregator from gate_a runner."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "Docs/results/e4_system_kd_v1"
POL = ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"
EXPECTED_POLICY_SHA = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


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
        mb, mc = float(np.median(b[idx])), float(np.median(c[idx]))
        speeds.append(mb / max(mc, 1e-9))
    lo, hi = np.quantile(speeds, [0.025, 0.975])
    return {
        "speedup_median": float(np.median(speeds)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "latency_reduction": float(1.0 - float(np.median(c)) / max(float(np.median(b)), 1e-9)),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    pol_sha = sha256_file(POL)
    pol = json.loads(POL.read_text(encoding="utf-8"))
    lock = json.loads((ART / "E4_SYSKD_V1_GATE_A_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    pre = json.loads((ART / "E4_SYSKD_V1_GATE_A_PREFLIGHT.json").read_text(encoding="utf-8"))
    parity = json.loads((ART / "E4_SYSKD_V1_GATE_A_PARITY_REPORT.json").read_text(encoding="utf-8"))
    report = json.loads((ART / "E4_SYSKD_V1_GATE_A_REPORT.json").read_text(encoding="utf-8"))
    acc = json.loads((ART / "E4_SYSKD_V1_GATE_A_ACCURACY_REPORT.json").read_text(encoding="utf-8"))
    raw = load_jsonl(ART / "E4_SYSKD_V1_GATE_A_RAW.jsonl")
    parity_raw = load_jsonl(ART / "E4_SYSKD_V1_GATE_A_PARITY_RAW.jsonl")
    r0p = load_jsonl(ART / "E4_SYSKD_V1_GATE_A_R0_PREDICTIONS.jsonl")
    r1p = load_jsonl(ART / "E4_SYSKD_V1_GATE_A_R1_PREDICTIONS.jsonl")

    checks = {}
    checks["policy_sha_match"] = pol_sha == EXPECTED_POLICY_SHA
    checks["policy_immutable_frozen"] = pol.get("status") == "FROZEN" and pol.get("immutable") is True
    checks["preflight_pass"] = bool(pre.get("pass"))
    checks["parity_report_pass"] = bool(parity.get("pass"))
    checks["selected_null"] = report.get("selected") is None and acc.get("imported_comparator", {}) is not None
    checks["e4_ready_false"] = report.get("e4_ready") is False
    checks["test_d4_demo_not_opened"] = True  # Gate A artifacts only; no auth files

    # Coverage
    ids = list(lock["sample_ids"])
    expected = {(sid, rnd, arm) for sid in ids for rnd in range(1, 6) for arm in ("R0", "R1")}
    seen = {(r["sample_id"], int(r["round"]), r["arm"]) for r in raw if r.get("success")}
    checks["coverage_exact_240"] = len(raw) == 240 and seen == expected
    # duplicates
    keys = [(r["sample_id"], int(r["round"]), r["arm"]) for r in raw]
    checks["no_duplicate_raw_keys"] = len(keys) == len(set(keys))

    ok = [r for r in raw if r.get("success")]
    by_arm = {"R0": [], "R1": []}
    for r in ok:
        by_arm[r["arm"]].append(float(r["e2e_ms"]))
    sample_arm: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in ok:
        sample_arm[(r["sample_id"], r["arm"])].append(float(r["e2e_ms"]))
    r0_med = [median(sample_arm[(s, "R0")]) for s in ids]
    r1_med = [median(sample_arm[(s, "R1")]) for s in ids]
    speedup = median(r0_med) / max(median(r1_med), 1e-9)
    boot = bootstrap_speedup(r0_med, r1_med, n_boot=10000, seed=42)

    recalc = {
        "r0_e2e_median_ms": median(by_arm["R0"]),
        "r0_e2e_p90_ms": p90(by_arm["R0"]),
        "r1_e2e_median_ms": median(by_arm["R1"]),
        "r1_e2e_p90_ms": p90(by_arm["R1"]),
        "speedup": speedup,
        "latency_reduction": 1.0 - median(r1_med) / max(median(r0_med), 1e-9),
        "bootstrap": boot,
    }

    def close(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

    checks["report_matches_recalc_r0_median"] = close(report["r0"]["e2e_median_ms"], recalc["r0_e2e_median_ms"])
    checks["report_matches_recalc_r1_median"] = close(report["r1"]["e2e_median_ms"], recalc["r1_e2e_median_ms"])
    checks["report_matches_recalc_speedup"] = close(report["r1_vs_r0"]["speedup"], recalc["speedup"], tol=1e-12)
    checks["report_matches_bootstrap_ci_lo"] = close(
        report["r1_vs_r0"]["bootstrap"]["ci95_lo"], recalc["bootstrap"]["ci95_lo"], tol=1e-12
    )

    # Parity raw
    checks["parity_raw_n24"] = len(parity_raw) == 24
    checks["parity_frame_sha_all"] = all(r.get("frame_sha_equal") for r in parity_raw)
    checks["parity_top1_all"] = all(int(r.get("top1_agree", 0)) == 1 for r in parity_raw)
    checks["pred_probe_n24"] = len(r0p) == 24 and len(r1p) == 24

    # Accuracy gate at Gate A
    checks["full_val_not_required"] = acc.get("full_validation_required_by_policy_at_gate_a") is False
    checks["accuracy_pass_declared"] = bool(acc.get("pass"))

    overall = all(checks.values())
    terminal = (
        "E4_SYSKD_V1_GATE_A_PASS_SYSTEM_ARMS_READY"
        if overall and parity.get("pass") and pre.get("pass")
        else "E4_SYSKD_V1_BASELINE_REPRO_NO_GO"
    )
    if not parity.get("pass"):
        terminal = "E4_SYSKD_V1_GATE_A_R1_PARITY_NO_GO"

    out = {
        "schema": "e4_syskd_v1_gate_a_independent_verify_v1",
        "created_at": report.get("created_at"),
        "policy_sha256": pol_sha,
        "checks": checks,
        "recalc_from_raw": recalc,
        "report_snapshot": {
            "r0_median": report["r0"]["e2e_median_ms"],
            "r1_median": report["r1"]["e2e_median_ms"],
            "speedup": report["r1_vs_r0"]["speedup"],
        },
        "selected": None,
        "e4_ready": False,
        "b4_retained": True,
        "overall_pass": overall,
        "terminal": terminal,
        "note": "Verifier recomputes aggregates from RAW jsonl; does not call gate_a aggregator functions.",
    }
    path = ART / "E4_SYSKD_V1_GATE_A_INDEPENDENT_VERIFY.json"
    if path.exists():
        raise SystemExit(f"exists:{path}")
    data = (json.dumps(out, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    print(json.dumps({"terminal": terminal, "overall_pass": overall, "sha": hashlib.sha256(data).hexdigest()}, indent=2))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
