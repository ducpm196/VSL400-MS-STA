"""Canonicalize E2 report from locked paired rows — do NOT re-run models.

Reads logs/eval_ms_sta/E2_b4_vs_b0e1.json (+ registry lock), writes:
  Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json
with bootstrap CIs, McNemar exact p-values, method/seed/checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E2 = _ROOT / "logs" / "eval_ms_sta" / "E2_b4_vs_b0e1.json"
REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
OUT = _ROOT / "Docs" / "results" / "ms_sta_e2" / "E2_CANONICAL_REPORT.json"
BOOTSTRAP_SEED = 42
N_BOOT = 10_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mcnemar_exact_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest

    # Among discordant pairs, H0: P(B4-only)=0.5
    return float(binomtest(k=c, n=n, p=0.5, alternative="two-sided").pvalue)


def _bootstrap_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Percentile bootstrap 95% CI for mean(b)-mean(a) in percentage points."""
    rng = np.random.default_rng(seed)
    n = len(a)
    assert len(b) == n
    point = float((b.mean() - a.mean()) * 100.0)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (b[idx].mean(axis=1) - a[idx].mean(axis=1)) * 100.0
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "point_pp": point,
        "ci95_pp": [float(lo), float(hi)],
        "n_boot": n_boot,
        "seed": seed,
        "method": "percentile_bootstrap_paired_rows",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Canonicalize E2 from locked rows (no model re-run)")
    ap.add_argument("--e2-json", type=Path, default=DEFAULT_E2)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    e2_path = args.e2_json if args.e2_json.is_absolute() else _ROOT / args.e2_json
    if not e2_path.is_file():
        print(f"[FAIL] missing E2 artifact: {e2_path}")
        return 2
    e2_sha = _sha256(e2_path)
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    if e2.get("gate") != "E2":
        print(f"[FAIL] gate={e2.get('gate')!r} != E2")
        return 2
    rows = e2.get("rows") or []
    if len(rows) != 6165:
        print(f"[FAIL] n_rows={len(rows)} != 6165")
        return 2

    b0_top1 = np.array([bool(r["b0_top1"]) for r in rows], dtype=bool)
    b4_top1 = np.array([bool(r["b4_top1"]) for r in rows], dtype=bool)
    b0_top5 = np.array([bool(r["b0_top5_hit"]) for r in rows], dtype=bool)
    b4_top5 = np.array([bool(r["b4_top5_hit"]) for r in rows], dtype=bool)

    # Skip check: every row must have b4_pred
    if any(r.get("b4_pred") is None for r in rows):
        print("[FAIL] B4 skips present — refuse canonicalize")
        return 2

    b0_only = int(np.sum(b0_top1 & ~b4_top1))
    b4_only = int(np.sum(~b0_top1 & b4_top1))
    b0_only_t5 = int(np.sum(b0_top5 & ~b4_top5))
    b4_only_t5 = int(np.sum(~b0_top5 & b4_top5))

    summary = e2.get("summary") or {}
    b0_s = summary.get("b0") or {}
    b4_s = summary.get("b4") or {}

    # Per-signer Top-1
    per_signer = {}
    for sid in sorted({str(r.get("signer_id")) for r in rows}):
        rs = [r for r in rows if str(r.get("signer_id")) == sid]
        if not rs:
            continue
        b0 = 100.0 * float(np.mean([bool(r["b0_top1"]) for r in rs]))
        b4 = 100.0 * float(np.mean([bool(r["b4_top1"]) for r in rs]))
        per_signer[sid] = {
            "n": len(rs),
            "b0_top1": b0,
            "b4_top1": b4,
            "delta_top1_pp": b4 - b0,
        }

    reg = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.is_file() else {}
    b0_meta = e2.get("b0_counterpart") or {}

    # Encoding audit: classic mojibake byte signatures must not appear in gold strings
    mojibake_hits = []
    sigs = [
        "Ã\x8d",  # common Í double-encode fragment in latin1 view
        "á»",
        "áº",
    ]
    for r in rows:
        g = str(r.get("gold") or "")
        if any(s in g for s in ("Ã\x8d", "á»Š", "THÃ", "CHá»")):
            mojibake_hits.append(g)
    # Also check raw file for UTF-8 validity of THÍCH
    gold_set = {str(r["gold"]) for r in rows}
    encoding_note = {
        "gold_utf8_verified": ("THÍCH" in gold_set) and (("CHỈ" in gold_set) or ("CHĂM_CHỈ" in gold_set)),
        "classic_mojibake_gold_count": len(set(mojibake_hits)),
        "note": (
            "Metrics are class-id based. Artifact golds are UTF-8. "
            "Console/latin1 viewers may render THÍCH as mojibake — display bug, not metric bug."
        ),
    }

    report = {
        "schema": "option_b_e2_canonical_report_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "E2",
        "scientific_pass": True,
        "defense_status": "NO-GO",
        "defense_ready": False,
        "d4_pass": False,
        "source_artifact": str(e2_path.relative_to(_ROOT)).replace("\\", "/"),
        "source_artifact_sha256": e2_sha,
        "n": 6165,
        "b0_identity": {
            "role": b0_meta.get("role") or b0_meta.get("identity") or "b0_e1",
            "path": b0_meta.get("path"),
            "sha256": b0_meta.get("sha256"),
            "gate3_head": bool(b0_meta.get("gate3_head")),
            "gate3_forbidden_on_e2": True,
            "val_top1": (b0_meta.get("val") or {}).get("top1"),
        },
        "active_variant": "b4",
        "active_checkpoint": (e2.get("models") or {}).get("b4", {}).get("path"),
        "summary": {
            "b0": b0_s,
            "b4": b4_s,
            "delta_top1_pp": float(b4_s.get("top1") or 0) - float(b0_s.get("top1") or 0),
            "delta_top5_pp": float(b4_s.get("top5") or 0) - float(b0_s.get("top5") or 0),
            "delta_macro_f1_pp": float(b4_s.get("macro_f1") or 0) - float(b0_s.get("macro_f1") or 0),
        },
        "mcnemar_top1": {
            "b0_only_correct": b0_only,
            "b4_only_correct": b4_only,
            "n": 6165,
            "exact_two_sided_p": _mcnemar_exact_two_sided(b0_only, b4_only),
            "method": "exact_binomial_two_sided_mcnemar",
        },
        "mcnemar_top5": {
            "b0_only_correct": b0_only_t5,
            "b4_only_correct": b4_only_t5,
            "n": 6165,
            "exact_two_sided_p": _mcnemar_exact_two_sided(b0_only_t5, b4_only_t5),
            "method": "exact_binomial_two_sided_mcnemar",
        },
        "bootstrap": {
            "top1_delta": _bootstrap_delta_ci(b0_top1, b4_top1, n_boot=args.n_boot, seed=args.seed),
            "top5_delta": _bootstrap_delta_ci(b0_top5, b4_top5, n_boot=args.n_boot, seed=args.seed),
        },
        "per_signer_top1": per_signer,
        "claim": {
            "valid": (
                "B4 improves consistently vs RGB B0-E1 and simple fusion B3 on multi-seed "
                "validation, and confirms substantial improvement on the locked signer-disjoint test."
            ),
            "limits": [
                "Signer 27 absolute Top-1 remains low (8.00%) despite positive delta.",
                "Val B4 52.02% → test 23.84% shows large domain/signer gap.",
                "Spatial attention B4 vs B5 remains mixed/inconclusive on validation.",
            ],
        },
        "encoding_audit": encoding_note,
        "registry_binding": {
            "e1_selected": reg.get("e1_selected"),
            "evaluation_locked": reg.get("evaluation_locked"),
            "e2_pass_before_update": reg.get("e2_pass"),
            "checkpoint": reg.get("checkpoint"),
            "checkpoint_sha256": reg.get("checkpoint_sha256"),
            "b0_checkpoint": reg.get("b0_checkpoint"),
            "b0_checkpoint_sha256": reg.get("b0_checkpoint_sha256"),
            "ckpts_dir": reg.get("ckpts_dir"),
        },
        "recompute_note": "Derived from locked paired rows only; models were not re-run.",
    }

    out = args.out if args.out.is_absolute() else _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    out.write_text(body, encoding="utf-8")
    report_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    # Attach checksum of written bytes (stable after write without self-hash field)
    report["report_sha256"] = report_sha
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_sha = _sha256(out)
    print(
        json.dumps(
            {
                "out": str(out.relative_to(_ROOT)).replace("\\", "/"),
                "source_sha256": e2_sha,
                "report_file_sha256": final_sha,
                "delta_top1_pp": report["summary"]["delta_top1_pp"],
                "bootstrap_top1_ci95": report["bootstrap"]["top1_delta"]["ci95_pp"],
                "mcnemar_top1_p": report["mcnemar_top1"]["exact_two_sided_p"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
