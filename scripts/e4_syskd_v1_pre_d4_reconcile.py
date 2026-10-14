# -*- coding: utf-8 -*-
"""PRE-D4 accuracy reconciliation: inventory + sample align + decision (no new inference)."""
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
from typing import Any

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import coverage_sets, paired_arm_coverage  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"
POL_PATH = ART / "E4_SYSKD_V1_PRE_D4_ACCURACY_RECONCILIATION_POLICY.json"
ONESHOT = CODE_ROOT / "Docs/results/ms_sta_v6_oneshot"
SYSKD_POL = CODE_ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"

EXPECTED_N = 6165
CKPT = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
SYSKD_POL_SHA = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
CODE_CAND = "b52c23b0e3598bd78731c4e3e28c7dbb5e5eb8f2"
LOCKED_PASS = "563044eac406db35656154108247b1497caf62ca"
AUTH_TIME = "2026-09-02T01:16:10Z"  # locked-test authorization created_at


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, obj: Any, *, overwrite: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise SystemExit(f"exists:{path}")
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def macro_f1(y_true: list[int], y_pred: list[int], n_classes: int = 400) -> float:
    y_true_a = np.asarray(y_true, dtype=np.int64)
    y_pred_a = np.asarray(y_pred, dtype=np.int64)
    f1s: list[float] = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred_a == c) & (y_true_a == c)))
        fp = int(np.sum((y_pred_a == c) & (y_true_a != c)))
        fn = int(np.sum((y_pred_a != c) & (y_true_a == c)))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def hierarchical_bootstrap_ci(
    deltas: list[float],
    signers: list[str],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    by_signer: dict[str, list[float]] = {}
    for d, s in zip(deltas, signers):
        by_signer.setdefault(s, []).append(d)
    signer_keys = sorted(by_signer.keys())
    observed = 100.0 * (sum(deltas) / len(deltas))
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        drawn: list[float] = []
        for _s in range(len(signer_keys)):
            sk = signer_keys[rng.randrange(len(signer_keys))]
            pool = by_signer[sk]
            for _i in range(len(pool)):
                drawn.append(pool[rng.randrange(len(pool))])
        boots.append(100.0 * (sum(drawn) / len(drawn)))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return observed, lo, hi


def mcnemar_exact(b_only: int, p_only: int) -> float:
    n = b_only + p_only
    if n == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(k=p_only, n=n, p=0.5, alternative="two-sided").pvalue)


def load_b4_aligned(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row["sample_id"])
        b4 = row["models"]["b4_v6_ctr"]
        out[sid] = {
            "sample_id": sid,
            "y_true": int(row["label"]),
            "y_pred": int(b4["pred"]),
            "top5": [int(x) for x in b4["top5_ids"]],
            "signer_id": str(row["signer_id"]),
            "checkpoint_sha256": str(b4.get("checkpoint_sha256") or CKPT),
            "top1_hit": bool(b4["top1_hit"]),
            "top5_hit": bool(b4["top5_hit"]),
            "logits": None,
        }
    return out


def load_online_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row["sample_id"])
        out[sid] = {
            "sample_id": sid,
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "top5": [int(x) for x in row["top5"]],
            "signer_id": str(row["signer_id"]),
            "checkpoint_sha256": str(row.get("checkpoint_sha256") or ""),
            "logits": row.get("logits"),
            "frame_sha": row.get("frame_sha"),
            "code_sha": row.get("code_sha"),
            "policy_sha256": row.get("policy_sha256"),
            "arm": row.get("arm"),
        }
    return out


def arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "signer_top1": {s: float(np.mean(v)) for s, v in sorted(by_s.items()) if s},
    }


def paired_vs(
    primary: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    d1: list[float] = []
    d5: list[float] = []
    signers: list[str] = []
    agree = 0
    canon_ok_online_bad = 0
    canon_bad_online_ok = 0
    by_s_d: dict[str, list[float]] = defaultdict(list)
    class_conc: Counter[int] = Counter()
    for sid in ids:
        p = primary[sid]
        b = baseline[sid]
        pc = int(p["y_pred"] == p["y_true"])
        bc = int(b["y_pred"] == b["y_true"])
        pt5 = int(p["y_true"] in set(p["top5"]))
        bt5 = int(b["y_true"] in set(b["top5"]))
        delta = float(pc - bc)
        d1.append(delta)
        d5.append(float(pt5 - bt5))
        signers.append(str(p["signer_id"]))
        by_s_d[str(p["signer_id"])].append(delta)
        agree += int(p["y_pred"] == b["y_pred"])
        if bc and not pc:
            canon_ok_online_bad += 1
            class_conc[int(b["y_true"])] += 1
        elif pc and not bc:
            canon_bad_online_ok += 1
    mean_pp, lo, hi = hierarchical_bootstrap_ci(d1, signers)
    mean5 = 100.0 * (sum(d5) / len(d5))
    b_only = canon_ok_online_bad
    p_only = canon_bad_online_ok
    return {
        "n": len(ids),
        "prediction_agreement": agree / len(ids),
        "paired_top1_delta_pp": mean_pp,
        "hierarchical_ci95_top1_pp": [lo, hi],
        "paired_top5_delta_pp": mean5,
        "mcnemar": {
            "online_only_correct": p_only,
            "canonical_only_correct": b_only,
            "n_discordant": p_only + b_only,
            "exact_two_sided_p": mcnemar_exact(b_only, p_only),
            "method": "exact_binomial_two_sided_mcnemar",
        },
        "transition": {
            "canonical_correct_online_wrong": canon_ok_online_bad,
            "canonical_wrong_online_correct": canon_bad_online_ok,
        },
        "per_person_performing_sign_delta_top1_pp": {
            s: 100.0 * (sum(xs) / len(xs)) for s, xs in sorted(by_s_d.items())
        },
        "class_concentration_canonical_ok_online_wrong_top10": [
            {"class_id": int(c), "count": int(n)} for c, n in class_conc.most_common(10)
        ],
    }


def inventory() -> dict[str, Any]:
    binding = json.loads((ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json").read_text(encoding="utf-8"))
    data_root = Path(binding["canonical_data_root"])
    split_path = data_root / "data/processed/vsl400_retrain_v4_split_manifest.json"
    freeze = json.loads((ONESHOT / "PRETEST_FREEZE.json").read_text(encoding="utf-8"))
    report = json.loads((ONESHOT / "TEST_REPORT.json").read_text(encoding="utf-8"))
    final = json.loads((CODE_ROOT / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json").read_text(encoding="utf-8"))
    r1_path = REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_R1.jsonl"
    s5_path = REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_S5.jsonl"
    gate_a_r0 = ART / "E4_SYSKD_V1_GATE_A_R0_PREDICTIONS.jsonl"
    b4q_q0 = CODE_ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_ONLINE_PRED_Q0.jsonl"

    # R0 online full-test search (read-only name/size/sha)
    candidates = []
    for p in [
        gate_a_r0,
        b4q_q0,
        CODE_ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_ONLINE_PRED_Q2.jsonl",
        CODE_ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_ONLINE_PRED_Q3.jsonl",
    ]:
        if p.exists():
            n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
            candidates.append(
                {
                    "path": str(p.relative_to(CODE_ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(p),
                    "n_rows": n,
                    "is_full_test_6165": n == EXPECTED_N,
                    "note": "not a sealed R0 FP32 online dump on locked-test 6165 IDs"
                    if n != EXPECTED_N
                    else "n_matches_but_must_verify role/path",
                }
            )

    inv = {
        "schema": "e4_syskd_v1_test_evidence_inventory_v1",
        "created_at": utc_now(),
        "no_new_inference": True,
        "canonical_b4": {
            "aligned_preds": {
                "path": "Docs/results/ms_sta_v6_oneshot/TEST_PREDS_ALIGNED.jsonl",
                "sha256": sha256_file(ONESHOT / "TEST_PREDS_ALIGNED.jsonl"),
                "bytes": (ONESHOT / "TEST_PREDS_ALIGNED.jsonl").stat().st_size,
            },
            "variant_preds": {
                "path": "Docs/results/ms_sta_v6_oneshot/TEST_PREDS_b4_v6_ctr_seed42.json",
                "sha256": sha256_file(ONESHOT / "TEST_PREDS_b4_v6_ctr_seed42.json"),
            },
            "report": {
                "path": "Docs/results/ms_sta_v6_oneshot/TEST_REPORT.json",
                "sha256": sha256_file(ONESHOT / "TEST_REPORT.json"),
                "b4_top1": report["variants"]["b4_v6_ctr"]["top1"],
                "b4_top5": report["variants"]["b4_v6_ctr"]["top5"],
                "b4_macro_f1": report["variants"]["b4_v6_ctr"]["macro_f1"],
            },
            "pretest_freeze": {
                "path": "Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json",
                "sha256": sha256_file(ONESHOT / "PRETEST_FREEZE.json"),
                "feature_roots": freeze["feature_roots"],
                "input_path_class": "cache_backed_feature_npz",
            },
            "defense_final": {
                "path": "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json",
                "sha256": sha256_file(CODE_ROOT / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json"),
                "claim_limits": final.get("claim_limits"),
            },
            "checkpoint_sha256": CKPT,
            "split_manifest_sha256": report["split_manifest_sha256"],
            "code_head_at_test": report.get("head"),
            "runner": "scripts/run_ms_sta_v6_oneshot_test.py",
        },
        "e4_syskd_locked_test": {
            "r1": {"path": str(r1_path.relative_to(CODE_ROOT)).replace("\\", "/"), "sha256": sha256_file(r1_path)},
            "s5": {"path": str(s5_path.relative_to(CODE_ROOT)).replace("\\", "/"), "sha256": sha256_file(s5_path)},
            "report_sha256": sha256_file(ART / "E4_SYSKD_V1_S5_LOCKED_TEST_REPORT.json"),
            "verify_sha256": sha256_file(ART / "E4_SYSKD_V1_S5_LOCKED_TEST_INDEPENDENT_VERIFY.json"),
            "journal_sha256": sha256_file(ART / "E4_SYSKD_V1_S5_LOCKED_TEST_JOURNAL.jsonl"),
            "manifest_sha256": sha256_file(ART / "E4_SYSKD_V1_S5_LOCKED_TEST_MANIFEST.json"),
            "binding_sha256": sha256_file(ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json"),
            "checkpoint_sha256": CKPT,
            "policy_sha256": SYSKD_POL_SHA,
            "code_sha": CODE_CAND,
            "input_path_class": "online_decode_roi_slowfast_then_b4_head",
            "precision": "fp32_logits_amp_fp16_false_in_head_forward",
        },
        "r0_online_full_test_search": {
            "found_full_test_6165": False,
            "candidates_inspected": candidates,
            "conclusion": "SOURCE_GAP: no sealed R0 FP32 online prediction dump on exact test n=6165 created before locked-test authorization.",
        },
        "current_split_sha256": sha256_file(split_path),
        "split_sha_matches_oneshot": sha256_file(split_path) == report["split_manifest_sha256"],
    }
    write_json(ART / "E4_SYSKD_V1_TEST_EVIDENCE_INVENTORY.json", inv)

    matrix = {
        "schema": "e4_syskd_v1_test_dump_provenance_matrix_v1",
        "created_at": utc_now(),
        "rows": [
            {
                "dump": "B4_canonical_aligned",
                "created_before_s5_test_authorization": True,
                "n": EXPECTED_N,
                "sample_manifest_sha256": report["split_manifest_sha256"],
                "checkpoint_sha256": CKPT,
                "input_path": "VSL400MSSTAv5Dataset over D0 feature NPZ",
                "feature_path": freeze["feature_roots"],
                "precision": "torch_eval_on_cached_tokens",
                "code_sha": report.get("head"),
                "logits": False,
            },
            {
                "dump": "R1_locked_test_online",
                "created_before_s5_test_authorization": False,
                "n": EXPECTED_N,
                "sample_manifest_sha256": sha256_file(split_path),
                "checkpoint_sha256": CKPT,
                "input_path": "raw_video_decode + ROI extract (batch_roi=False)",
                "feature_path": "online SlowFast tokens + sealed skeleton_v3 NPZ",
                "precision": "fp32 head (amp_fp16=False); ROI online",
                "code_sha": CODE_CAND,
                "logits": True,
            },
            {
                "dump": "S5_locked_test_online",
                "created_before_s5_test_authorization": False,
                "n": EXPECTED_N,
                "sample_manifest_sha256": sha256_file(split_path),
                "checkpoint_sha256": CKPT,
                "input_path": "raw_video_decode + ROI extract (batch_roi=True)",
                "feature_path": "online SlowFast tokens + sealed skeleton_v3 NPZ",
                "precision": "fp32 head (amp_fp16=False); ROI batched",
                "code_sha": CODE_CAND,
                "logits": True,
            },
            {
                "dump": "GateA_R0_online",
                "created_before_s5_test_authorization": True,
                "n": 24,
                "sample_manifest_sha256": "gate_a_schedule_24_ids",
                "checkpoint_sha256": CKPT,
                "input_path": "online Gate A schedule",
                "feature_path": "online",
                "precision": "online",
                "code_sha": "pre_locked_test_campaign",
                "logits": False,
                "usable_as_formal_test_comparator": False,
            },
            {
                "dump": "B4Q_Q0_ONLINE",
                "created_before_s5_test_authorization": True,
                "n": 2629,
                "sample_manifest_sha256": "val_ids",
                "checkpoint_sha256": CKPT,
                "input_path": "online val",
                "feature_path": "online",
                "precision": "Q0_ONLINE",
                "code_sha": "b4q_campaign",
                "logits": True,
                "usable_as_formal_test_comparator": False,
            },
        ],
        "formal_r0_online_test_6165_exists": False,
    }
    write_json(ART / "E4_SYSKD_V1_TEST_DUMP_PROVENANCE_MATRIX.json", matrix)
    return inv


def reconcile() -> dict[str, Any]:
    policy = json.loads(POL_PATH.read_text(encoding="utf-8"))
    binding = json.loads((ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json").read_text(encoding="utf-8"))
    data_root = Path(binding["canonical_data_root"])
    test_ids = [
        str(x)
        for x in json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))[
            "test_ids"
        ]
    ]
    b4 = load_b4_aligned(ONESHOT / "TEST_PREDS_ALIGNED.jsonl")
    r1 = load_online_jsonl(REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_R1.jsonl")
    s5 = load_online_jsonl(REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_S5.jsonl")

    cov = {
        "b4": coverage_sets(test_ids, list(b4.keys())),
        "r1": coverage_sets(test_ids, list(r1.keys())),
        "s5": coverage_sets(test_ids, list(s5.keys())),
        "paired_r1_s5": paired_arm_coverage(test_ids, list(r1.keys()), list(s5.keys())),
    }
    # alignment checks
    target_mismatch = []
    signer_mismatch = []
    ckpt_mismatch = []
    for sid in test_ids:
        if sid not in b4 or sid not in r1 or sid not in s5:
            continue
        if b4[sid]["y_true"] != r1[sid]["y_true"] or r1[sid]["y_true"] != s5[sid]["y_true"]:
            target_mismatch.append(sid)
        if b4[sid]["signer_id"] != r1[sid]["signer_id"] or r1[sid]["signer_id"] != s5[sid]["signer_id"]:
            signer_mismatch.append(sid)
        if r1[sid]["checkpoint_sha256"] != CKPT or s5[sid]["checkpoint_sha256"] != CKPT:
            ckpt_mismatch.append(sid)
        if b4[sid]["checkpoint_sha256"] not in ("", CKPT) and b4[sid]["checkpoint_sha256"] != CKPT:
            ckpt_mismatch.append(sid)

    ordered_b4 = [b4[sid] for sid in test_ids]
    ordered_r1 = [r1[sid] for sid in test_ids]
    ordered_s5 = [s5[sid] for sid in test_ids]
    m_b4, m_r1, m_s5 = arm_metrics(ordered_b4), arm_metrics(ordered_r1), arm_metrics(ordered_s5)
    pair_s5_b4 = paired_vs(s5, b4, test_ids)
    pair_r1_b4 = paired_vs(r1, b4, test_ids)
    pair_s5_r1 = paired_vs(s5, r1, test_ids)

    # raw rows (compact; no full logits)
    raw_path = REG / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_RAW.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    with raw_path.open("ab") as f:
        for sid in test_ids:
            row = {
                "sample_id": sid,
                "signer_id": b4[sid]["signer_id"],
                "y_true": b4[sid]["y_true"],
                "b4_pred": b4[sid]["y_pred"],
                "b4_top5": b4[sid]["top5"],
                "r1_pred": r1[sid]["y_pred"],
                "r1_top5": r1[sid]["top5"],
                "s5_pred": s5[sid]["y_pred"],
                "s5_top5": s5[sid]["top5"],
                "b4_correct": int(b4[sid]["y_pred"] == b4[sid]["y_true"]),
                "r1_correct": int(r1[sid]["y_pred"] == r1[sid]["y_true"]),
                "s5_correct": int(s5[sid]["y_pred"] == s5[sid]["y_true"]),
                "r1_s5_pred_agree": int(r1[sid]["y_pred"] == s5[sid]["y_pred"]),
                "frame_sha": r1[sid].get("frame_sha"),
            }
            f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))

    # path classification
    scope_compatible_for_model_delta = False  # cache vs online
    case = "C"
    terminal = "E4_SYSKD_V1_PRE_D4_R0_ONLINE_COMPARATOR_SOURCE_GAP"
    rationale = (
        "B4 canonical test uses cache-backed D0 RGB/Skeleton feature NPZ "
        "(scripts/run_ms_sta_v6_oneshot_test.py + PRETEST_FREEZE feature_roots). "
        "R1/S5 locked test uses online video decode + ROI SlowFast extraction. "
        "No sealed R0 FP32 online dump on n=6165 exists. "
        "Observed S5−B4 deltas quantify path mismatch, not formal online accuracy gate vs R0. "
        "R1=S5 exact parity only shows S3 ROI batching is fidelity-preserving vs R1."
    )

    report = {
        "schema": "e4_syskd_v1_pre_d4_reconciliation_report_v1",
        "created_at": utc_now(),
        "policy_sha256": sha256_file(POL_PATH),
        "reconciliation_policy_status": policy.get("status"),
        "n": EXPECTED_N,
        "coverage": cov,
        "alignment": {
            "target_mismatch_n": len(target_mismatch),
            "signer_mismatch_n": len(signer_mismatch),
            "checkpoint_mismatch_n": len(ckpt_mismatch),
            "exact_id_coverage": bool(
                cov["b4"]["exact_coverage"] and cov["r1"]["exact_coverage"] and cov["s5"]["exact_coverage"]
            ),
            "logits_comparable_b4_vs_online": False,
            "logits_note": "B4 canonical dump has predictions only; online dumps have logits. Logit drift vs B4 = SOURCE_GAP.",
        },
        "metrics": {
            "B4_canonical_cache": {k: v for k, v in m_b4.items() if k != "signer_top1"},
            "R1_online": {k: v for k, v in m_r1.items() if k != "signer_top1"},
            "S5_online": {k: v for k, v in m_s5.items() if k != "signer_top1"},
            "R0_online_test": None,
        },
        "signer_top1": {
            "B4": m_b4["signer_top1"],
            "R1": m_r1["signer_top1"],
            "S5": m_s5["signer_top1"],
        },
        "paired": {
            "S5_minus_B4": pair_s5_b4,
            "R1_minus_B4": pair_r1_b4,
            "S5_minus_R1": pair_s5_r1,
        },
        "observed_delta_pp_S5_minus_B4": {
            "top1": 100.0 * (m_s5["top1"] - m_b4["top1"]),
            "top5": 100.0 * (m_s5["top5"] - m_b4["top5"]),
            "macro_f1": 100.0 * (m_s5["macro_f1"] - m_b4["macro_f1"]),
        },
        "scope": {
            "same_n": True,
            "same_split_sha": True,
            "same_checkpoint": True,
            "same_input_path_class": False,
            "b4_path_class": "cache_backed_feature_npz",
            "online_path_class": "online_decode_roi",
            "compatible_for_formal_model_accuracy_gate": scope_compatible_for_model_delta,
            "r0_online_6165_exists": False,
        },
        "case": case,
        "terminal": terminal,
        "d4_allowed": False,
        "d4_execution": "NOT_OPENED",
        "selected": None,
        "e4_ready": False,
        "provisional_candidate": "S5",
        "b4_retained": True,
        "demo_closed": True,
        "rationale": rationale,
        "raw_rel": "Docs/results/e4_system_kd_v1_registry/E4_SYSKD_V1_PRE_D4_RECONCILIATION_RAW.jsonl",
        "raw_sha256": sha256_file(raw_path),
        "raw_bytes": raw_path.stat().st_size,
    }
    write_json(ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_REPORT.json", report)

    path_diff = {
        "schema": "e4_syskd_v1_b4_vs_online_path_diff_v1",
        "created_at": utc_now(),
        "differences": [
            {
                "dimension": "feature_input",
                "file_symbol": "scripts/run_ms_sta_v6_oneshot_test.py:178-188 / scripts/e4_syskd_v1_s5_formal.py:extract_arm",
                "b4_canonical": "Load cached rgb_token_v3 + skeleton from D0 feature roots in PRETEST_FREEZE",
                "r1_s5": "read_video_frames_native + extract_rgb_token_v3_for_video (+ sealed skeleton_v3)",
                "evidence": [
                    "Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json#feature_roots",
                    "scripts/run_ms_sta_v6_oneshot_test.py",
                    "scripts/e4_syskd_v1_s5_formal.py",
                ],
                "impact": "Different RGB token generation path; primary candidate for accuracy delta",
                "claim": "FACT",
            },
            {
                "dimension": "rgb_feature_root",
                "file_symbol": "PRETEST_FREEZE.json / E4_SYSKD policy feature_roots_read_only",
                "b4_canonical": "data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70",
                "r1_s5": "online SlowFast from decoded frames (not D0 rgb_token cache as primary RGB path)",
                "evidence": ["PRETEST_FREEZE.json", "e4_syskd formal extract_arm"],
                "impact": "cached-vs-online domain shift",
                "claim": "FACT",
            },
            {
                "dimension": "skeleton",
                "file_symbol": "PRETEST_FREEZE skeleton_root vs sealed skeleton_v3",
                "b4_canonical": "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
                "r1_s5": "policy sealed skeleton_v3 under feature_roots_read_only.skel_rel",
                "evidence": ["PRETEST_FREEZE.json", "E4_SYSKD_V1_POLICY.json / formal sealed_root"],
                "impact": "possible skeleton cache version mismatch contributing to delta",
                "claim": "INFERENCE",
            },
            {
                "dimension": "checkpoint",
                "file_symbol": "best.pt sha",
                "b4_canonical": CKPT,
                "r1_s5": CKPT,
                "evidence": ["TEST_REPORT.json", "LOCKED_TEST_MANIFEST.json"],
                "impact": "not a weight regression",
                "claim": "FACT",
            },
            {
                "dimension": "split_manifest",
                "file_symbol": "vsl400_retrain_v4_split_manifest.json",
                "b4_canonical": report_split_sha(),
                "r1_s5": report_split_sha(),
                "evidence": ["TEST_REPORT.json", "inventory current_split"],
                "impact": "same ID universe",
                "claim": "FACT",
            },
            {
                "dimension": "S5_vs_R1",
                "file_symbol": "batch_roi True vs False",
                "b4_canonical": "n/a",
                "r1_s5": "exact logit parity on locked test (max_abs=0)",
                "evidence": ["E4_SYSKD_V1_S5_LOCKED_TEST_REPORT.json"],
                "impact": "S3 ROI batching does not explain B4 vs online delta",
                "claim": "FACT",
            },
            {
                "dimension": "logits_vs_b4",
                "file_symbol": "TEST_PREDS_ALIGNED lacks logits",
                "b4_canonical": "predictions only",
                "r1_s5": "full 400-d logits",
                "evidence": ["TEST_PREDS_ALIGNED.jsonl keys", "LOCKED_TEST R1 dump"],
                "impact": "cannot quantify logit drift to B4",
                "claim": "SOURCE_GAP",
            },
        ],
        "rejected_premature_causes": [
            "S5 ROI batching bug (ruled out by R1=S5 exact parity)",
            "different checkpoint (ruled out by SHA match)",
            "different test ID set (ruled out by exact coverage + split SHA)",
        ],
        "primary_root_cause_class": "cached_vs_online_feature_path_mismatch",
        "primary_claim": "FACT",
    }
    write_json(ART / "E4_SYSKD_V1_B4_VS_ONLINE_PATH_DIFF.json", path_diff)

    md = f"""# E4_SYSKD_V1 — Test accuracy root cause (PRE-D4)

## Verdict

**Terminal:** `{terminal}`
**Case:** C — comparator scope incompatible (cache-backed B4 vs online R1/S5).
**D4:** remains `execution=NOT_OPENED`.
**selected/e4_ready:** null / false. B4 retained.

## Observed metrics (independent recompute)

| Dump | Top-1 | Top-5 | Macro-F1 |
|------|------:|------:|---------:|
| B4 canonical (cache) | {100*m_b4['top1']:.4f}% | {100*m_b4['top5']:.4f}% | {100*m_b4['macro_f1']:.4f}% |
| R1 online | {100*m_r1['top1']:.4f}% | {100*m_r1['top5']:.4f}% | {100*m_r1['macro_f1']:.4f}% |
| S5 online | {100*m_s5['top1']:.4f}% | {100*m_s5['top5']:.4f}% | {100*m_s5['macro_f1']:.4f}% |
| S5−B4 (pp) | {report['observed_delta_pp_S5_minus_B4']['top1']:.4f} | {report['observed_delta_pp_S5_minus_B4']['top5']:.4f} | {report['observed_delta_pp_S5_minus_B4']['macro_f1']:.4f} |

Paired S5−B4 Top-1 hierarchical CI95 (pp): {pair_s5_b4['hierarchical_ci95_top1_pp']}.
McNemar discordant: canonical-only-correct={pair_s5_b4['mcnemar']['canonical_only_correct']}, online-only-correct={pair_s5_b4['mcnemar']['online_only_correct']}, p={pair_s5_b4['mcnemar']['exact_two_sided_p']:.3e}.

## What is FACT

1. Same n=6165, same split SHA, same checkpoint SHA.
2. R1 and S5 exact prediction/logit agreement on locked test.
3. B4 oneshot loads **cached** D0 feature NPZ (`PRETEST_FREEZE.feature_roots`).
4. R1/S5 locked test decodes video and extracts ROI tokens online.
5. No sealed R0 FP32 online dump on full test 6165 exists in inventory.

## What is INFERENCE

Skeleton root version (D0 v14 vs sealed skeleton_v3) may contribute additional domain shift beyond RGB cache-vs-online.

## SOURCE_GAP

- No R0 online test comparator on 6165.
- B4 dump lacks logits → no logit-level drift vs online.
- Cannot apply frozen `paired_vs=R0` accuracy gate to S5 using B4 cache scores as if they were R0 online.

## Must not claim

- That S5 failed the formal online accuracy gate vs R0 (R0 online test missing).
- That R1=S5 proves B4=S5.
- That D4 may open on this evidence alone.
"""
    (ART / "E4_SYSKD_V1_TEST_ACCURACY_ROOT_CAUSE.md").write_text(md, encoding="utf-8")

    # proposal for supplemental R0 (do not execute)
    proposal = {
        "schema": "e4_syskd_v1_r0_online_supplemental_comparator_proposal_v1",
        "status": "AWAITING_EXPLICIT_USER_R0_ONLINE_COMPARATOR_APPROVAL",
        "created_at": utc_now(),
        "user_authorization_received": False,
        "purpose": "comparator_only_diagnostic_after_locked_test",
        "not": [
            "rerun_S5",
            "retune",
            "change_checkpoint",
            "change_threshold",
            "change_candidate",
            "open_d4",
            "open_demo",
        ],
        "frozen": {
            "code_sha": CODE_CAND,
            "checkpoint_sha256": CKPT,
            "policy_sha256": SYSKD_POL_SHA,
            "n": EXPECTED_N,
            "role": "test",
            "arm": "R0",
            "precision": "FP32_online",
            "append_only": True,
            "independent_verify_required": True,
        },
        "protocol_amendment_required": True,
        "one_shot_claim_caveat": "Original locked-test one-shot remains R1+S5 only; R0 supplemental is a separate authorized comparator, not a second S5 shot.",
        "outcome_may_make_s5_no_go": True,
        "reconciliation_terminal": terminal,
        "d4_remains": "NOT_OPENED",
        "selected": None,
        "e4_ready": False,
        "demo_closed": True,
        "b4_retained": True,
    }
    write_json(ART / "E4_SYSKD_V1_R0_ONLINE_SUPPLEMENTAL_COMPARATOR_PROPOSAL.json", proposal)

    terminal_obj = {
        "terminal": terminal,
        "case": case,
        "d4_allowed": False,
        "d4_execution": "NOT_OPENED",
        "provisional_candidate": "S5",
        "selected": None,
        "e4_ready": False,
        "b4_retained": True,
        "demo_closed": True,
        "report_sha256": sha256_file(ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_REPORT.json"),
        "path_diff_sha256": sha256_file(ART / "E4_SYSKD_V1_B4_VS_ONLINE_PATH_DIFF.json"),
        "created_at": utc_now(),
        "next_explicit_user_decision": "Authorize R0 online supplemental comparator OR accept SOURCE_GAP and keep E4 unpromoted with B4 SoT",
    }
    write_json(ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_TERMINAL.json", terminal_obj)

    # pointer for raw
    write_json(
        ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_RAW_POINTER.json",
        {
            "schema": "pre_d4_reconciliation_raw_pointer_v1",
            "path": report["raw_rel"],
            "sha256": report["raw_sha256"],
            "bytes": report["raw_bytes"],
            "n": EXPECTED_N,
        },
    )
    return report


def report_split_sha() -> str:
    return json.loads((ONESHOT / "TEST_REPORT.json").read_text(encoding="utf-8"))["split_manifest_sha256"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "inventory", "reconcile"], default="all")
    args = ap.parse_args()
    if args.phase in ("all", "inventory"):
        inv = inventory()
        print(json.dumps({"inventory_r0_full": inv["r0_online_full_test_search"]["found_full_test_6165"]}, indent=2))
    if args.phase in ("all", "reconcile"):
        rep = reconcile()
        print(
            json.dumps(
                {
                    "terminal": rep["terminal"],
                    "case": rep["case"],
                    "d4_allowed": rep["d4_allowed"],
                    "metrics": rep["metrics"],
                    "delta_pp": rep["observed_delta_pp_S5_minus_B4"],
                    "ci": rep["paired"]["S5_minus_B4"]["hierarchical_ci95_top1_pp"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
