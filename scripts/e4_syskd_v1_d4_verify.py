# -*- coding: utf-8 -*-
"""Independent re-check of D4 RAW vs report (generator ≠ this process)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.optimization.b4q.invariants import sha256_file  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
RAW = ART / "E4_SYSKD_V1_D4_EXECUTION_RAW.jsonl"
REPORT = ART / "E4_SYSKD_V1_D4_ACCEPTANCE_REPORT.json"
TERM = ART / "E4_SYSKD_V1_D4_TERMINAL.json"
OUT = ART / "E4_SYSKD_V1_D4_INDEPENDENT_VERIFY.json"
SCHED = ART / "E4_SYSKD_V1_D4_SCHEDULE_LOCK.json"
EXPECTED_SCHED_SHA = "ff61e9b491f31f0344e8368be4c4f0b8c06d681de80e745ce3b83967015f334b"
EXPECTED_NAMES = [
    "locked_test_terminal_present",
    "post_test_artifact_verify_pass",
    "checkpoint_sha_matches_freeze",
    "policy_sha_matches_freeze",
    "candidate_code_sha_matches_auth",
    "provisional_candidate_is_S5",
    "selected_remains_null_until_promotion",
    "e4_ready_false_until_promotion",
    "b4_retained_as_system_of_record",
    "exact_checkpoint_load_s5_runtime",
    "forward_smoke_demo_batch_fail_closed_paths",
    "missing_rgb_fail_closed",
    "missing_skeleton_fail_closed",
    "cache_miss_fail_closed",
    "wrong_sha_detected",
    "demo_not_cutover",
    "no_threshold_or_candidate_mutation",
]


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    term = json.loads(TERM.read_text(encoding="utf-8"))
    raw = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    names = [r["scenario"] for r in raw]
    n_pass = sum(1 for r in raw if r.get("pass"))
    # Sealed D4 acceptance report is the pre-promotion snapshot.
    report_ok = (
        report.get("n_pass") == 17
        and report.get("d4_pass") is True
        and report.get("e4_ready") is False
        and report.get("selected") is None
        and report.get("demo_closed") is True
    )
    # Terminal may remain sealed pre-promotion OR be updated after authorized S5 promotion.
    promo_auth = ART / "E4_SYSKD_V1_PROMOTION_AUTHORIZATION.json"
    promoted = False
    if promo_auth.is_file():
        pa = json.loads(promo_auth.read_text(encoding="utf-8"))
        promoted = bool(pa.get("user_authorization_received")) and pa.get("transition_applied", {}).get(
            "selected"
        ) == "S5"
    if promoted:
        term_ok = (
            term.get("d4_pass") is True
            and term.get("terminal") == "E4_SYSKD_V1_D4_PASS_PROMOTION_READY"
            and term.get("selected") == "S5"
            and term.get("e4_ready") is True
            and term.get("demo_closed") is True
            and str(term.get("demo_cutover") or "").upper() in {"NOT_OPENED", "NOT_AUTHORIZED", "FALSE"}
        )
    else:
        term_ok = term.get("e4_ready") is False and term.get("selected") is None
    ok = (
        names == EXPECTED_NAMES
        and len(raw) == 17
        and n_pass == 17
        and report_ok
        and term_ok
        and sha256_file(SCHED) == EXPECTED_SCHED_SHA
        and sha256_file(RAW) == report.get("raw_sha256")
    )
    out = {
        "schema": "e4_syskd_v1_d4_independent_verify_v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "verifier": "scripts/e4_syskd_v1_d4_verify.py",
        "pass": ok,
        "n_scenarios": len(raw),
        "n_pass": n_pass,
        "scenario_order_exact": names == EXPECTED_NAMES,
        "raw_sha256": sha256_file(RAW),
        "report_sha256": sha256_file(REPORT),
        "terminal": term.get("terminal"),
        "matches_generator_report": ok and term.get("terminal") == "E4_SYSKD_V1_D4_PASS_PROMOTION_READY",
        "post_promotion_terminal_ok": promoted and term_ok,
        "selected": term.get("selected"),
        "e4_ready": term.get("e4_ready"),
        "demo_closed": True,
        "b4_retained": True,
        "schedule_lock_sha256": EXPECTED_SCHED_SHA,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # refresh terminal independent sha if needed
    term["independent_verify_sha256"] = sha256_file(OUT)
    TERM.write_text(json.dumps(term, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pass", "n_pass", "terminal", "matches_generator_report")}, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
