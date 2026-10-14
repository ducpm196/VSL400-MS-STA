"""Retrain-v3 retrospective test evaluation from EVAL_LOCK (scientific stats required)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import load_eligible_sample_manifest, load_json, sha256_file
from src.models.ms_sta_features import resolve_pooled_dir, resolve_skel_dir, resolve_temporal_dir
from src.models.ms_sta_wordlevel import build_ms_sta
from src.training.retrain_v3 import (
    build_demo_set_ms_sta_v3,
    load_policy,
    model_variant_name,
    require_frozen_v3_policy,
)
from src.training.retrain_v3_stats import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    REQUIRED_TEST_VARIANTS,
    holm_correction,
    macro_f1,
    mcnemar,
    noninferiority_from_paired_ci,
    paired_signer_bootstrap_delta_ci,
    require_exact_variants,
    require_sha_present_and_equal,
    signer_metrics,
    signer_stratified_bootstrap_ci,
    topk_accuracy,
    topk_hit_mask,
)
from src.training.train_ms_sta_vsl400 import VSL400MSSTADataset, _collate

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_retrain_v3_policy/RETRAIN_POLICY.json"


@torch.no_grad()
def _predict_ordered(model, loader, device, variant: str, expected_ids: list[str]) -> dict:
    model.eval()
    by_id: dict[str, dict] = {}
    for rgb, skel, y, batch_vids, rgb_kpm in loader:
        rgb = rgb.to(device)
        y = y.to(device)
        skel_t = skel.to(device) if skel is not None else None
        kpm = rgb_kpm.to(device) if rgb_kpm is not None else None
        if variant in ("b0", "b1"):
            logits = model(rgb=rgb, rgb_key_padding_mask=kpm)
        elif variant == "b2":
            logits = model(skel=skel_t)
        else:
            logits = model(rgb=rgb, skel=skel_t, rgb_key_padding_mask=kpm)
        prob = torch.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)
        t5 = prob.topk(5, dim=-1).indices
        for i, vid in enumerate(batch_vids):
            by_id[str(vid)] = {
                "y": int(y[i].item()),
                "pred": int(pred[i].item()),
                "top5": [int(x) for x in t5[i].tolist()],
            }
    missing = [vid for vid in expected_ids if vid not in by_id]
    if missing:
        raise RuntimeError(f"missing predictions for {len(missing)} IDs e.g. {missing[:5]}")
    extra = sorted(set(by_id) - set(expected_ids))
    if extra:
        raise RuntimeError(f"unexpected prediction IDs: {extra[:5]}")
    y_all = np.array([by_id[v]["y"] for v in expected_ids], dtype=np.int64)
    p_all = np.array([by_id[v]["pred"] for v in expected_ids], dtype=np.int64)
    t5_all = np.array([by_id[v]["top5"] for v in expected_ids], dtype=np.int64)
    return {"y": y_all, "pred": p_all, "top5": t5_all, "video_ids": list(expected_ids)}


def _norm_signer(value: object) -> str:
    s = str(value).strip()
    if s.startswith("signer_"):
        return s
    if s.isdigit() or s.lstrip("0").isdigit() or s == "0":
        return f"signer_{int(s)}"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--eval-lock", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confirm-retrospective-test", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    args = ap.parse_args()
    if not args.confirm_retrospective_test:
        print("[FAIL] refuse test without --confirm-retrospective-test")
        return 2
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[FAIL] CUDA unavailable for final test")
        return 2

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    policy_sha = sha256_file(policy_path)
    status_dir = _ROOT / str((policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3")
    lock_path = args.eval_lock or (status_dir / "EVAL_LOCK.json")
    if not Path(lock_path).is_absolute():
        lock_path = _ROOT / lock_path
    if not lock_path.is_file():
        print(f"[FAIL] missing EVAL_LOCK: {lock_path}")
        return 2
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not lock.get("evaluation_locked"):
        print("[FAIL] evaluation_locked != true")
        return 2
    if lock.get("test_claim_label") != policy.get("test_claim_label"):
        print("[FAIL] test_claim_label mismatch")
        return 2
    if int(lock.get("primary_seed") or -1) != 42:
        print("[FAIL] primary_seed must be 42")
        return 2

    ckpts = lock.get("checkpoints") or {}
    try:
        require_exact_variants(set(ckpts), REQUIRED_TEST_VARIANTS)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    split_path = _ROOT / "data/processed/vsl400_retrain_v3_split_manifest.json"
    split = load_json(split_path)
    split_sha = sha256_file(split_path)
    expected_split = (policy.get("lineage") or {}).get("split_manifest_sha256")
    try:
        require_sha_present_and_equal(split_sha, expected_split, label="split_manifest")
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    elig = load_eligible_sample_manifest(
        _ROOT / "data/processed/vsl400_eligible_sample_manifest.json", root=_ROOT
    )
    test_ids = [str(x) for x in split["test_ids"]]
    if set(test_ids) - elig["eligible_set"]:
        print("[FAIL] test_ids not subset of eligible")
        return 2
    if len(test_ids) != int(elig["test_n"]) or len(test_ids) != 6165:
        print(f"[FAIL] test id count {len(test_ids)} != 6165/eligible")
        return 2

    labels = load_json(_ROOT / "data/processed/vsl400_labels.json")
    num_classes = int(labels["num_classes"])
    registry = load_json(_ROOT / "data/processed/ms_sta_registry.json")
    meta = pd.read_csv(_ROOT / "metadata.csv")
    df = meta[meta["dataset"] == "VSL400"].copy()
    df["video_id"] = df["video_id"].astype(str)
    signer_by_id = {
        str(r.video_id): _norm_signer(r.signer_id) for r in df.itertuples(index=False)
    }
    signers = [signer_by_id[v] for v in test_ids]
    device = torch.device(args.device)

    preds_by_variant: dict[str, dict] = {}
    for variant in REQUIRED_TEST_VARIANTS:
        meta_ckpt = ckpts[variant]
        ckpt_path = _ROOT / meta_ckpt["path"]
        actual = sha256_file(ckpt_path)
        if actual != meta_ckpt["sha256"]:
            print(f"[FAIL] checkpoint sha mismatch for {variant}")
            return 2
        if meta_ckpt.get("checkpoint_status") != "final_selected_checkpoint":
            print(f"[FAIL] {variant} is not final_selected_checkpoint")
            return 2
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if int(payload.get("seed") or -1) != 42:
            print(f"[FAIL] {variant} seed != 42")
            return 2
        try:
            require_sha_present_and_equal(
                payload.get("retrain_v3_policy_sha256"), policy_sha, label=f"{variant} policy"
            )
            require_sha_present_and_equal(
                payload.get("manifest_sha256"), split_sha, label=f"{variant} split/manifest"
            )
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 2
        if payload.get("checkpoint_status") != "final_selected_checkpoint":
            print(f"[FAIL] {variant} payload checkpoint_status not final")
            return 2
        if "skel_temporal_stride" not in payload:
            print(f"[FAIL] {variant} missing skel_temporal_stride")
            return 2
        model_var = model_variant_name(variant)
        stride = int(payload["skel_temporal_stride"])
        expected_stride = 4 if variant == "b4e" else 1
        if stride != expected_stride:
            print(f"[FAIL] {variant} skel_temporal_stride={stride} != {expected_stride}")
            return 2
        model = build_ms_sta(
            model_var,
            num_classes=int(payload["num_classes"]),
            disable_spatial=bool(payload.get("disable_spatial", False)),
            skel_temporal_stride=stride,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        require_skel = model_var in ("b2", "b3", "b4", "b5") or variant == "b4e"
        require_temporal = model_var in ("b1", "b3", "b4", "b5") or variant == "b4e"
        if model_var == "b0":
            require_skel = False
            require_temporal = False
        ds = VSL400MSSTADataset(
            df,
            labels["gloss_to_id"],
            require_skel=require_skel,
            require_temporal_rgb=require_temporal,
            rgb_pooled_dir=resolve_pooled_dir(_ROOT, registry),
            rgb_temporal_dir=resolve_temporal_dir(_ROOT, registry),
            skel_dir=resolve_skel_dir(_ROOT, registry),
            video_ids=test_ids,
        )
        if len(ds) != len(test_ids):
            print(f"[FAIL] missing features for test IDs: got {len(ds)} / {len(test_ids)}")
            return 2
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=_collate)
        try:
            preds_by_variant[variant] = _predict_ordered(
                model, loader, device, model_var, test_ids
            )
        except Exception as exc:
            print(f"[FAIL] predict {variant}: {exc}")
            return 2

    # Common ordered labels
    y_ref = preds_by_variant["b0"]["y"]
    for v, pack in preds_by_variant.items():
        if not np.array_equal(pack["y"], y_ref):
            print(f"[FAIL] label order mismatch for {v}")
            return 2
        if pack["video_ids"] != test_ids:
            print(f"[FAIL] video_id order mismatch for {v}")
            return 2

    variant_metrics: dict[str, Any] = {}
    correct: dict[str, np.ndarray] = {}
    top5_hit: dict[str, np.ndarray] = {}
    n_boot = int(args.bootstrap)
    for v, pack in preds_by_variant.items():
        yt, yp, t5 = pack["y"], pack["pred"], pack["top5"]
        correct[v] = yp == yt
        top5_hit[v] = topk_hit_mask(yt, t5)
        sm = signer_metrics(yt, yp, signers)
        variant_metrics[v] = {
            "checkpoint": ckpts[v]["path"],
            "checkpoint_sha256": ckpts[v]["sha256"],
            "n": int(len(yt)),
            "top1": float(correct[v].mean() * 100.0),
            "top5": topk_accuracy(yt, t5),
            "macro_f1": macro_f1(yt, yp, num_classes),
            "per_signer": sm["per_signer"],
            "worst_signer": sm["worst_signer"],
            "bootstrap_ci_top1": signer_stratified_bootstrap_ci(
                yt, yp, signers, n_boot=n_boot, seed=42
            ),
        }

    m_b4_b0 = mcnemar(correct["b0"], correct["b4"])
    m_b4_b3 = mcnemar(correct["b3"], correct["b4"])
    holm = holm_correction(
        [
            ("b4_gt_b0", m_b4_b0["p_value_exact"]),
            ("b4_gt_b3", m_b4_b3["p_value_exact"]),
        ]
    )
    holm["tests"]["b4_gt_b0"]["directional_support"] = (
        m_b4_b0["other_only_correct"] > m_b4_b0["baseline_only_correct"]
    )
    holm["tests"]["b4_gt_b3"]["directional_support"] = (
        m_b4_b3["other_only_correct"] > m_b4_b3["baseline_only_correct"]
    )

    paired_deltas = {
        "top1_b4_minus_b0": paired_signer_bootstrap_delta_ci(
            correct["b0"], correct["b4"], signers, n_boot=n_boot, seed=42, label_a="b0", label_b="b4"
        ),
        "top1_b4_minus_b3": paired_signer_bootstrap_delta_ci(
            correct["b3"], correct["b4"], signers, n_boot=n_boot, seed=42, label_a="b3", label_b="b4"
        ),
        "top1_b4e_minus_b4": paired_signer_bootstrap_delta_ci(
            correct["b4"], correct["b4e"], signers, n_boot=n_boot, seed=42, label_a="b4", label_b="b4e"
        ),
        "top5_b4_minus_b0": paired_signer_bootstrap_delta_ci(
            top5_hit["b0"], top5_hit["b4"], signers, n_boot=n_boot, seed=42, label_a="b0", label_b="b4"
        ),
        "top5_b4_minus_b3": paired_signer_bootstrap_delta_ci(
            top5_hit["b3"], top5_hit["b4"], signers, n_boot=n_boot, seed=42, label_a="b3", label_b="b4"
        ),
        "top5_b4e_minus_b4": paired_signer_bootstrap_delta_ci(
            top5_hit["b4"], top5_hit["b4e"], signers, n_boot=n_boot, seed=42, label_a="b4", label_b="b4e"
        ),
    }
    ni = noninferiority_from_paired_ci(paired_deltas["top1_b4e_minus_b4"], margin_pp=-1.0)
    hypothesis_decisions = {
        "b4_superior_to_b0": {
            "supported": bool(
                holm["tests"]["b4_gt_b0"]["reject_null_at_0.05"]
                and holm["tests"]["b4_gt_b0"]["directional_support"]
            ),
            "holm": holm["tests"]["b4_gt_b0"],
            "mcnemar": m_b4_b0,
            "paired_delta": paired_deltas["top1_b4_minus_b0"],
        },
        "b4_superior_to_b3": {
            "supported": bool(
                holm["tests"]["b4_gt_b3"]["reject_null_at_0.05"]
                and holm["tests"]["b4_gt_b3"]["directional_support"]
            ),
            "holm": holm["tests"]["b4_gt_b3"],
            "mcnemar": m_b4_b3,
            "paired_delta": paired_deltas["top1_b4_minus_b3"],
        },
        "b4e_noninferior_to_b4_margin_m1pp": {
            "supported": bool(ni["noninferior"]),
            "noninferiority": ni,
            "paired_delta": paired_deltas["top1_b4e_minus_b4"],
        },
    }

    # Per-sample rows aligned to the locked 6,165 ID order for all four models.
    rows = []
    for i, vid in enumerate(test_ids):
        rows.append(
            {
                "video_id": vid,
                "signer_id": signers[i],
                "y": int(y_ref[i]),
                "pred_b0": int(preds_by_variant["b0"]["pred"][i]),
                "pred_b3": int(preds_by_variant["b3"]["pred"][i]),
                "pred_b4": int(preds_by_variant["b4"]["pred"][i]),
                "pred_b4e": int(preds_by_variant["b4e"]["pred"][i]),
                "top5_b0": [int(x) for x in preds_by_variant["b0"]["top5"][i].tolist()],
                "top5_b3": [int(x) for x in preds_by_variant["b3"]["top5"][i].tolist()],
                "top5_b4": [int(x) for x in preds_by_variant["b4"]["top5"][i].tolist()],
                "top5_b4e": [int(x) for x in preds_by_variant["b4e"]["top5"][i].tolist()],
                "top5_hit_b0": bool(top5_hit["b0"][i]),
                "top5_hit_b3": bool(top5_hit["b3"][i]),
                "top5_hit_b4": bool(top5_hit["b4"][i]),
                "top5_hit_b4e": bool(top5_hit["b4e"][i]),
            }
        )

    results = {
        "schema": "ms_sta_retrain_v3_test_matrix_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_claim_label": lock.get("test_claim_label"),
        "evaluation_locked": True,
        "eval_lock_sha256": sha256_file(lock_path),
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "eligible_manifest_sha256": elig["sha256"],
        "n_test": len(test_ids),
        "bootstrap_n": n_boot,
        "bootstrap_seed": 42,
        "required_variants": list(REQUIRED_TEST_VARIANTS),
        "variants_present": list(REQUIRED_TEST_VARIANTS),
        "common_pair_n": len(test_ids),
        "common_pair_counts": {
            "n_ids": len(test_ids),
            "mcnemar_b4_vs_b0_n_discordant": m_b4_b0["n_discordant"],
            "mcnemar_b4_vs_b3_n_discordant": m_b4_b3["n_discordant"],
        },
        "variants": variant_metrics,
        "paired_deltas": paired_deltas,
        "mcnemar": {"b4_vs_b0": m_b4_b0, "b4_vs_b3": m_b4_b3},
        "holm": holm,
        "b4e_vs_b4_noninferiority": ni,
        "hypothesis_decisions": hypothesis_decisions,
        "per_sample_rows_path": "logs/ms_sta_retrain_v3/TEST_PER_SAMPLE.jsonl",
        "demo_set_v3_path": "data/processed/demo_set_ms_sta_v3.json",
        "reuse_e2_evaluator": False,
        "reuse_metrics": False,
    }

    out = status_dir / "TEST_MATRIX.json"
    rows_path = status_dir / "TEST_PER_SAMPLE.jsonl"
    status_dir.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    id_to_gloss = {int(v): str(k) for k, v in labels["gloss_to_id"].items()}
    # Prefer reverse map from id_to_gloss if present.
    if "id_to_gloss" in labels:
        id_to_gloss = {int(k): str(v) for k, v in labels["id_to_gloss"].items()}
    active = str(lock.get("active_variant") or "b4")
    demo = build_demo_set_ms_sta_v3(
        rows,
        active_variant=active if active in ("b4", "b4e") else "b4",
        id_to_gloss=id_to_gloss,
        test_matrix_sha256=sha256_file(out),
        policy_sha256=policy_sha,
        split_manifest_sha256=split_sha,
    )
    demo_path = _ROOT / "data/processed/demo_set_ms_sta_v3.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(json.dumps(demo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out),
                "demo_set_v3": str(demo_path),
                "n_test": len(test_ids),
                "variants": list(REQUIRED_TEST_VARIANTS),
                "b4_top1": variant_metrics["b4"]["top1"],
                "b4e_ni": ni,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
