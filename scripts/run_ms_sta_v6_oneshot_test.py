#!/usr/bin/env python3
"""One-shot retrospective signer-disjoint test for frozen MS-STA v6 seed-42 models.

Requires sealed ONE_SHOT_TEST_AUTHORIZATION. E4 excluded after accuracy NO-GO.
Create-only artifacts under Docs/results/ms_sta_v6_oneshot/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_rev  # noqa: E402
from src.eval.ms_sta_v5_m0_strong_baseline import PredRow, hierarchical_bootstrap_ci  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json  # noqa: E402
from src.training.ms_sta_v5_train import evaluate, load_metadata_by_id  # noqa: E402

AUTH_REL = "Docs/results/ms_sta_v6_oneshot/ONE_SHOT_TEST_AUTHORIZATION.json"
FREEZE_REL = "Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json"
OUT_DIR = _ROOT / "Docs/results/ms_sta_v6_oneshot"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, obj: Any) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _mcnemar_exact_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(k=c, n=n, p=0.5, alternative="two-sided").pvalue)


def _rows_to_pred(rows: list[dict[str, Any]]) -> dict[str, PredRow]:
    out: dict[str, PredRow] = {}
    for r in rows:
        sid = str(r["sample_id"])
        if sid in out:
            raise SystemExit(f"duplicate_sample_id_in_eval_rows:{sid}")
        out[sid] = PredRow(
            sample_id=sid,
            signer=str(r["signer"]),
            label=int(r["label"]),
            correct=bool(r["correct"]),
            top5_hit=bool(r["top5_hit"]),
        )
    return out


def _paired_deltas(
    primary: dict[str, PredRow], baseline: dict[str, PredRow]
) -> tuple[list[str], list[float], list[float], list[str], dict[str, float]]:
    ids = sorted(primary.keys())
    if set(ids) != set(baseline.keys()):
        raise RuntimeError("pred_id_mismatch")
    d1: list[float] = []
    d5: list[float] = []
    signers: list[str] = []
    by_signer: dict[str, list[float]] = {}
    for sid in ids:
        p = primary[sid]
        b = baseline[sid]
        if p.signer != b.signer or p.label != b.label:
            raise RuntimeError(f"alignment_fail:{sid}")
        delta = float(p.correct) - float(b.correct)
        d1.append(delta)
        d5.append(float(p.top5_hit) - float(b.top5_hit))
        signers.append(p.signer)
        by_signer.setdefault(p.signer, []).append(delta)
    signer_pp = {s: 100.0 * (sum(xs) / len(xs)) for s, xs in sorted(by_signer.items())}
    return ids, d1, d5, signers, signer_pp


def _mcnemar(primary: dict[str, PredRow], baseline: dict[str, PredRow]) -> dict[str, Any]:
    b_only = 0
    p_only = 0
    for sid in primary:
        pc = primary[sid].correct
        bc = baseline[sid].correct
        if pc and not bc:
            p_only += 1
        elif bc and not pc:
            b_only += 1
    return {
        "primary_only": p_only,
        "baseline_only": b_only,
        "n_discordant": p_only + b_only,
        "exact_two_sided_p": _mcnemar_exact_two_sided(b_only, p_only),
        "method": "exact_binomial_two_sided_mcnemar",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm-retrospective-test", action="store_true", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    auth = load_json(_ROOT / AUTH_REL)
    freeze = load_json(_ROOT / FREEZE_REL)
    completion_path = OUT_DIR / "TEST_COMPLETION.json"
    report_path = OUT_DIR / "TEST_REPORT.json"
    if completion_path.is_file() or report_path.is_file():
        raise SystemExit("oneshot_already_complete_refuse_rerun")
    if auth.get("status") != "AUTHORIZED" or auth.get("allow_final_test") is not True:
        raise SystemExit("oneshot_auth_invalid")
    if auth.get("confirm_retrospective_test") is not True:
        raise SystemExit("confirm_flag_required_in_auth")
    if not args.confirm_retrospective_test:
        raise SystemExit("cli_confirm_required")
    if sha256_file(_ROOT / FREEZE_REL) != auth["pretest_freeze_sha256"]:
        raise SystemExit("freeze_sha_mismatch")
    if freeze.get("status") != "FROZEN":
        raise SystemExit("freeze_not_frozen")
    if auth.get("e4_included") is not False or freeze.get("e4_included") is not False:
        raise SystemExit("e4_must_be_excluded")
    if sha256_file(_ROOT / auth["e4_no_go_rel"]) != auth["e4_no_go_sha256"]:
        raise SystemExit("e4_no_go_sha_mismatch")

    variants = list(auth["test_variants"])
    forbidden = set(auth.get("forbidden_on_test") or [])
    if any(v in forbidden for v in variants):
        raise SystemExit("forbidden_variant_in_auth")

    split_path = _ROOT / freeze["split_manifest_rel"]
    if sha256_file(split_path) != auth["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")
    split = load_json(split_path)
    test_ids = [str(x) for x in split["test_ids"]]
    if len(test_ids) != int(auth["expected_test_n"]):
        raise SystemExit(f"test_n_mismatch:{len(test_ids)}")

    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}

    train_val = set()
    for key in ("hpo_train_ids", "tune_ids", "confirm_train_ids", "val_ids", "train_ids"):
        train_val.update(str(x) for x in (split.get(key) or []))
    leak = sorted(set(test_ids) & train_val)
    if leak:
        raise SystemExit(f"test_intersects_train_val:{leak[:5]}")

    rgb_root = _ROOT / freeze["feature_roots"]["rgb_root_rel"]
    skel_root = _ROOT / freeze["feature_roots"]["skeleton_root_rel"]
    ds = VSL400MSSTAv5Dataset(
        test_ids,
        rgb_root=rgb_root,
        skel_root=skel_root,
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(),
        require_all=True,
    )
    if len(ds) != len(test_ids):
        raise SystemExit(f"dataset_n_mismatch:{len(ds)}!={len(test_ids)}")
    signers = sorted({str(meta[sid]["signer"]) for sid in test_ids})
    expected = [str(x) for x in freeze["expected_test_signers"]]
    if signers != expected:
        raise SystemExit(f"test_signers_mismatch:{signers}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type != "cuda":
        raise SystemExit("oneshot_test_requires_cuda")

    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_v5)
    seed = int(auth["primary_seed"])
    model_cfg = freeze["model"]
    per_variant: dict[str, Any] = {}
    pred_by_variant: dict[str, dict[str, PredRow]] = {}
    jsonl_path = OUT_DIR / "TEST_PREDS_ALIGNED.jsonl"
    if jsonl_path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{jsonl_path}")

    aligned_by_id: dict[str, dict[str, Any]] = {}

    for variant in variants:
        meta_ck = freeze["checkpoints_seed42"][variant]
        ckpt_path = _ROOT / meta_ck["checkpoint_rel"]
        if sha256_file(ckpt_path) != meta_ck["checkpoint_sha256"]:
            raise SystemExit(f"ckpt_sha_mismatch:{variant}")
        print(json.dumps({"event": "test_eval_start", "variant": variant, "seed": seed}), flush=True)
        model = build_ms_sta_v6(
            variant,
            num_classes=int(model_cfg["num_classes"]),
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            temporal_layers=int(model_cfg["temporal_layers"]),
            dropout=float(meta_ck["dropout"]),
            alpha_max=float(model_cfg["alpha_max"]),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        t0 = time.perf_counter()
        metrics = evaluate(model, loader, device, collect_rows=True)
        elapsed = time.perf_counter() - t0
        rows = metrics.pop("rows")
        pred_by_variant[variant] = _rows_to_pred(rows)
        per_variant[variant] = {
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "macro_f1": metrics["macro_f1"],
            "n": int(metrics["n"]),
            "elapsed_sec": elapsed,
            "checkpoint_rel": meta_ck["checkpoint_rel"],
            "checkpoint_sha256": meta_ck["checkpoint_sha256"],
        }
        atomic_write(
            OUT_DIR / f"TEST_PREDS_{variant}_seed{seed}.json",
            {"variant": variant, "seed": seed, "checkpoint_sha256": meta_ck["checkpoint_sha256"], "rows": rows},
        )
        for r in rows:
            sid = str(r["sample_id"])
            slot = aligned_by_id.setdefault(
                sid,
                {
                    "sample_id": sid,
                    "signer_id": str(r["signer"]),
                    "label": int(r["label"]),
                    "models": {},
                },
            )
            if slot["label"] != int(r["label"]) or slot["signer_id"] != str(r["signer"]):
                raise SystemExit(f"aligned_meta_mismatch:{sid}")
            slot["models"][variant] = {
                "top5_ids": r["top5"],
                "top1_hit": bool(r["correct"]),
                "top5_hit": bool(r["top5_hit"]),
                "pred": int(r["pred"]),
                "checkpoint_sha256": meta_ck["checkpoint_sha256"],
            }
        print(
            json.dumps(
                {
                    "event": "test_eval_done",
                    "variant": variant,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "macro_f1": metrics["macro_f1"],
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for sid in sorted(aligned_by_id):
            fh.write(json.dumps(aligned_by_id[sid], ensure_ascii=False) + "\n")

    _, d1, d5, signers_list, signer_deltas = _paired_deltas(
        pred_by_variant["b4_v6_ctr"], pred_by_variant["b3_mlp_ctr"]
    )
    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(d1, signers_list, n_boot=10000, seed=42)
    mean5_pp, ci5_lo, ci5_hi = hierarchical_bootstrap_ci(d5, signers_list, n_boot=10000, seed=42)
    mcn = _mcnemar(pred_by_variant["b4_v6_ctr"], pred_by_variant["b3_mlp_ctr"])

    b4 = per_variant["b4_v6_ctr"]
    mlp = per_variant["b3_mlp_ctr"]
    gate = freeze["promotion_gate"]
    checks = {
        "b4_top1_gt_b3_mlp": bool(b4["top1"] > mlp["top1"]),
        "paired_ci95_lower_pp_gt_0": bool(ci_lo > 0.0),
        "no_material_top5_regression": bool((b4["top5"] - mlp["top5"]) >= -0.01),
        "no_material_macro_f1_regression": bool((b4["macro_f1"] - mlp["macro_f1"]) >= -0.01),
        "n_exact": bool(b4["n"] == int(auth["expected_test_n"])),
    }
    promote = all(checks.values()) if all(gate.values()) else False
    verdict = "B4_PROMOTION_GATE_PASS" if promote else "B4_PROMOTION_GATE_FAIL_RETAIN_V5"

    report = {
        "schema": "ms_sta_v6_oneshot_test_report_v1",
        "status": "TEST_COMPLETE",
        "retrospective_signer_disjoint": True,
        "pristine_blind_claim": False,
        "proposed_final_model": "b4_v6_ctr",
        "primary_contrast": "b4_v6_ctr_vs_b3_mlp_ctr",
        "seed": seed,
        "head": git_rev(_ROOT),
        "auth_sha256": sha256_file(_ROOT / AUTH_REL),
        "freeze_sha256": sha256_file(_ROOT / FREEZE_REL),
        "split_manifest_sha256": auth["split_manifest_sha256"],
        "test_n": len(test_ids),
        "test_signers": signers,
        "e4_evaluated": False,
        "variants": per_variant,
        "b4_minus_mlp": {
            "delta_top1": b4["top1"] - mlp["top1"],
            "delta_top5": b4["top5"] - mlp["top5"],
            "delta_macro_f1": b4["macro_f1"] - mlp["macro_f1"],
            "hierarchical_mean_delta_top1_pp": mean_pp,
            "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
            "hierarchical_mean_delta_top5_pp": mean5_pp,
            "hierarchical_ci95_top5_pp": [ci5_lo, ci5_hi],
            "per_signer_delta_top1_pp": signer_deltas,
            "mcnemar_top1": mcn,
        },
        "promotion_gate_checks": checks,
        "verdict": verdict,
        "aligned_jsonl_rel": "Docs/results/ms_sta_v6_oneshot/TEST_PREDS_ALIGNED.jsonl",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_sha = atomic_write(OUT_DIR / "TEST_REPORT.json", report)
    atomic_write(
        OUT_DIR / "TEST_COMPLETION.json",
        {
            "schema": "ms_sta_v6_oneshot_test_completion_v1",
            "status": "COMPLETE",
            "verdict": verdict,
            "report_sha256": report_sha,
            "e4_evaluated": False,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(
        json.dumps(
            {
                "event": "ONESHOT_TEST_COMPLETE",
                "verdict": verdict,
                "report_sha256": report_sha,
                "b4_top1": b4["top1"],
                "mlp_top1": mlp["top1"],
                "ci95_pp": [ci_lo, ci_hi],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if promote else 2


if __name__ == "__main__":
    raise SystemExit(main())
