"""Validation-only evaluation of retrain-v3 confirmation finals (no test IDs).

Produces VALIDATION_METRICS.json with Top-1/Top-5, macro-F1, per-signer, worst-signer
for all 21 final checkpoints on the frozen Path-B val split.
"""

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
    load_policy,
    model_variant_name,
    require_complete_phase_rows,
    require_confirm_final_lifecycle,
    require_frozen_v3_policy,
)
from src.training.retrain_v3_stats import macro_f1, signer_metrics, topk_accuracy, topk_hit_mask
from src.training.train_ms_sta_vsl400 import VSL400MSSTADataset, _collate

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_retrain_v3_policy/RETRAIN_POLICY.json"


def _git_rev() -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:
        return None


def _norm_signer(value: object) -> str:
    s = str(value).strip()
    if s.startswith("signer_"):
        return s
    if s.isdigit() or s.lstrip("0").isdigit() or s == "0":
        return f"signer_{int(s)}"
    return s


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
    y_all = np.array([by_id[v]["y"] for v in expected_ids], dtype=np.int64)
    p_all = np.array([by_id[v]["pred"] for v in expected_ids], dtype=np.int64)
    t5_all = np.array([by_id[v]["top5"] for v in expected_ids], dtype=np.int64)
    return {"y": y_all, "pred": p_all, "top5": t5_all, "video_ids": list(expected_ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[FAIL] CUDA unavailable for validation eval")
        return 2

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    policy_sha = sha256_file(policy_path)
    status_dir = _ROOT / str(
        (policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3"
    )
    state_path = status_dir / "RUN_STATE.json"
    if not state_path.is_file():
        print("[FAIL] missing RUN_STATE.json")
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    runs = list(((state.get("phases") or {}).get("confirm") or {}).get("runs") or [])
    try:
        require_complete_phase_rows(runs, expected_n=21, phase="confirm")
        require_confirm_final_lifecycle(runs)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    split_path = _ROOT / "data/processed/vsl400_retrain_v3_split_manifest.json"
    split = load_json(split_path)
    split_sha = sha256_file(split_path)
    expected_split = (policy.get("lineage") or {}).get("split_manifest_sha256")
    if expected_split and expected_split != split_sha:
        print("[FAIL] split sha mismatch")
        return 2
    if split.get("path") != "B":
        print("[FAIL] split path must be B")
        return 2

    val_ids = [str(x) for x in split["val_ids"]]
    test_ids = set(str(x) for x in split.get("test_ids") or [])
    if set(val_ids) & test_ids:
        print("[FAIL] val/test ID overlap")
        return 2

    elig = load_eligible_sample_manifest(
        _ROOT / "data/processed/vsl400_eligible_sample_manifest.json", root=_ROOT
    )
    if set(val_ids) - elig["eligible_set"]:
        print("[FAIL] val_ids not subset of eligible")
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
    signers = [signer_by_id[v] for v in val_ids]
    device = torch.device(args.device)

    rows_out: list[dict[str, Any]] = []
    test_ids_touched = False
    for row in runs:
        variant = str(row["variant"])
        seed = int(row["seed"])
        final_rel = row["final_checkpoint"]
        ckpt_path = _ROOT / final_rel
        actual = sha256_file(ckpt_path)
        if actual != row.get("final_checkpoint_sha256"):
            print(f"[FAIL] final sha mismatch {row.get('run_key')}")
            return 2
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_status") != "final_selected_checkpoint":
            print(f"[FAIL] payload not final for {row.get('run_key')}")
            return 2
        if int(payload.get("seed") or -1) != seed:
            print(f"[FAIL] seed mismatch in payload for {row.get('run_key')}")
            return 2
        model_var = model_variant_name(variant)
        stride = int(payload.get("skel_temporal_stride") or (4 if variant == "b4e" else 1))
        expected_stride = 4 if variant == "b4e" else 1
        if stride != expected_stride:
            print(f"[FAIL] stride mismatch for {row.get('run_key')}")
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
            video_ids=val_ids,
        )
        if len(ds) != len(val_ids):
            print(f"[FAIL] missing val features: {len(ds)}/{len(val_ids)}")
            return 2
        loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=_collate)
        print(f"[eval] {row.get('run_key')} on val n={len(val_ids)}", flush=True)
        try:
            pack = _predict_ordered(model, loader, device, model_var, val_ids)
        except Exception as exc:
            print(f"[FAIL] predict {row.get('run_key')}: {exc}")
            return 2
        if set(pack["video_ids"]) & test_ids:
            test_ids_touched = True
            print("[FAIL] predictions included test IDs")
            return 2
        yt, yp, t5 = pack["y"], pack["pred"], pack["top5"]
        sm = signer_metrics(yt, yp, signers)
        rows_out.append(
            {
                "run_key": row.get("run_key"),
                "variant": variant,
                "seed": seed,
                "config_id": row.get("config_id"),
                "final_checkpoint": final_rel,
                "final_checkpoint_sha256": actual,
                "n": int(len(yt)),
                "top1": float((yp == yt).mean() * 100.0),
                "top5": topk_accuracy(yt, t5),
                "macro_f1": macro_f1(yt, yp, num_classes),
                "per_signer": sm["per_signer"],
                "worst_signer": sm["worst_signer"],
                "confirm_status_top1": row.get("best_val_top1"),
                "confirm_status_top5": row.get("best_val_top5"),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {
        "schema": "ms_sta_retrain_v3_validation_metrics_v1",
        "split_role": "val",
        "val_n": len(val_ids),
        "test_ids_touched": bool(test_ids_touched),
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "training_git_revision": state.get("git_revision"),
        "git_revision": state.get("git_revision"),
        "evaluator_git_revision": _git_rev(),
        "evaluator_git_dirty": _git_dirty(),
        "evaluator_script": "scripts/evaluate_ms_sta_retrain_v3_validation.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows_out,
    }
    out_path = status_dir / "VALIDATION_METRICS.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": out_path.as_posix(), "n_runs": len(rows_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
