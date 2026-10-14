"""B3 vs B4 confirmation diagnosis on development/val IDs only (no test).

Collects feature norms, stream-gate stats, cross-attention entropy, and
per-signer Top-1 for paired B3/B4 finals. Writes DIAGNOSIS_B3_B4.json.
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
from src.training.retrain_v3_stats import signer_metrics
from src.training.train_ms_sta_vsl400 import VSL400MSSTADataset, _collate

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_retrain_v3_policy/RETRAIN_POLICY.json"


def _git_rev() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip() or None
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


def _entropy(weights: torch.Tensor, eps: float = 1e-8) -> float:
    # weights: (B, Tq, Tk) or (B, 2)
    w = weights.clamp_min(eps)
    ent = -(w * w.log()).sum(dim=-1)
    return float(ent.mean().item())


@torch.no_grad()
def _diagnose_variant(
    model,
    loader,
    device,
    variant: str,
    expected_ids: list[str],
    signers: list[str],
) -> dict[str, Any]:
    model.eval()
    by_id: dict[str, dict] = {}
    rgb_norms: list[float] = []
    skel_norms: list[float] = []
    gated_norms: list[float] = []
    gate_rgb: list[float] = []
    gate_fused: list[float] = []
    cross_ents: list[float] = []

    for rgb, skel, y, batch_vids, rgb_kpm in loader:
        rgb = rgb.to(device)
        y = y.to(device)
        skel_t = skel.to(device) if skel is not None else None
        kpm = rgb_kpm.to(device) if rgb_kpm is not None else None
        if variant == "b3":
            logits, attn = model(rgb=rgb, skel=skel_t, rgb_key_padding_mask=kpm, return_attention=True)
            # Recompute pools for norms via encode helpers
            rgb_tok, rgb_pool, _ = model._encode_rgb(rgb, key_padding_mask=kpm)
            skel_tok, skel_pool, _, _ = model._encode_skel(
                skel_t, batch=rgb.shape[0], device=device
            )
            rgb_norms.extend(rgb_pool.norm(dim=-1).tolist())
            skel_norms.extend(skel_pool.norm(dim=-1).tolist())
            del rgb_tok, skel_tok, attn
        else:
            logits, attn = model(rgb=rgb, skel=skel_t, rgb_key_padding_mask=kpm, return_attention=True)
            rgb_tok, rgb_pool, _ = model._encode_rgb(rgb, key_padding_mask=kpm)
            skel_tok, skel_pool, _, _ = model._encode_skel(
                skel_t, batch=rgb.shape[0], device=device
            )
            fused_tok, _ = model.cross(skel_tok, rgb_tok, key_padding_mask=kpm)
            fused_pool = fused_tok.mean(dim=1)
            gated, gate_w = model.gate(rgb_pool, fused_pool)
            rgb_norms.extend(rgb_pool.norm(dim=-1).tolist())
            skel_norms.extend(skel_pool.norm(dim=-1).tolist())
            gated_norms.extend(gated.norm(dim=-1).tolist())
            gate_rgb.extend(gate_w[:, 0].tolist())
            gate_fused.extend(gate_w[:, 1].tolist())
            cross_ents.append(_entropy(attn["cross"]))
            del rgb_tok, skel_tok, fused_tok, fused_pool, gated, gate_w

        pred = logits.argmax(dim=-1)
        for i, vid in enumerate(batch_vids):
            by_id[str(vid)] = {"y": int(y[i].item()), "pred": int(pred[i].item())}

    missing = [vid for vid in expected_ids if vid not in by_id]
    if missing:
        raise RuntimeError(f"missing predictions for {len(missing)} IDs")
    yt = np.array([by_id[v]["y"] for v in expected_ids], dtype=np.int64)
    yp = np.array([by_id[v]["pred"] for v in expected_ids], dtype=np.int64)
    sm = signer_metrics(yt, yp, signers)

    def _stats(xs: list[float]) -> dict[str, float] | None:
        if not xs:
            return None
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    return {
        "n": int(len(yt)),
        "top1": float((yp == yt).mean() * 100.0),
        "per_signer": sm["per_signer"],
        "worst_signer": sm["worst_signer"],
        "feature_norms": {
            "rgb_pool": _stats(rgb_norms),
            "skel_pool": _stats(skel_norms),
            "gated": _stats(gated_norms),
        },
        "stream_gate": {
            "rgb_weight": _stats(gate_rgb),
            "fused_weight": _stats(gate_fused),
        },
        "cross_attention_entropy_mean": float(np.mean(cross_ents)) if cross_ents else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--split-role",
        choices=("val", "train"),
        default="val",
        help="Development diagnosis on train or val IDs (never test).",
    )
    ap.add_argument("--seed", type=int, default=None, help="Optional single seed; default all three.")
    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[FAIL] CUDA unavailable")
        return 2

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    status_dir = _ROOT / str(
        (policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3"
    )
    state = json.loads((status_dir / "RUN_STATE.json").read_text(encoding="utf-8"))
    runs = list(((state.get("phases") or {}).get("confirm") or {}).get("runs") or [])
    require_complete_phase_rows(runs, expected_n=21, phase="confirm")
    require_confirm_final_lifecycle(runs)

    split_path = _ROOT / "data/processed/vsl400_retrain_v3_split_manifest.json"
    split = load_json(split_path)
    if split.get("path") != "B":
        print("[FAIL] path B required")
        return 2
    ids = [str(x) for x in split[f"{args.split_role}_ids"]]
    test_ids = set(str(x) for x in split.get("test_ids") or [])
    if set(ids) & test_ids:
        print("[FAIL] development IDs overlap test")
        return 2

    elig = load_eligible_sample_manifest(
        _ROOT / "data/processed/vsl400_eligible_sample_manifest.json", root=_ROOT
    )
    if set(ids) - elig["eligible_set"]:
        print("[FAIL] IDs not subset of eligible")
        return 2

    labels = load_json(_ROOT / "data/processed/vsl400_labels.json")
    registry = load_json(_ROOT / "data/processed/ms_sta_registry.json")
    meta = pd.read_csv(_ROOT / "metadata.csv")
    df = meta[meta["dataset"] == "VSL400"].copy()
    df["video_id"] = df["video_id"].astype(str)
    signer_by_id = {str(r.video_id): _norm_signer(r.signer_id) for r in df.itertuples(index=False)}
    signers = [signer_by_id[v] for v in ids]
    device = torch.device(args.device)

    seeds = [args.seed] if args.seed is not None else [42, 123, 456]
    by_key = {(str(r["variant"]), int(r["seed"])): r for r in runs}
    pairs: list[dict[str, Any]] = []

    for seed in seeds:
        for variant in ("b3", "b4"):
            row = by_key.get((variant, seed))
            if row is None:
                print(f"[FAIL] missing confirm row {variant} seed={seed}")
                return 2
            ckpt = _ROOT / row["final_checkpoint"]
            if sha256_file(ckpt) != row.get("final_checkpoint_sha256"):
                print(f"[FAIL] sha mismatch {row.get('run_key')}")
                return 2
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            model = build_ms_sta(
                model_variant_name(variant),
                num_classes=int(payload["num_classes"]),
                disable_spatial=bool(payload.get("disable_spatial", False)),
                skel_temporal_stride=int(payload.get("skel_temporal_stride") or 1),
            )
            model.load_state_dict(payload["model"], strict=True)
            model.to(device)
            ds = VSL400MSSTADataset(
                df,
                labels["gloss_to_id"],
                require_skel=True,
                require_temporal_rgb=True,
                rgb_pooled_dir=resolve_pooled_dir(_ROOT, registry),
                rgb_temporal_dir=resolve_temporal_dir(_ROOT, registry),
                skel_dir=resolve_skel_dir(_ROOT, registry),
                video_ids=ids,
            )
            loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=_collate)
            print(f"[diag] {row.get('run_key')} role={args.split_role} n={len(ids)}", flush=True)
            pack = _diagnose_variant(model, loader, device, variant, ids, signers)
            pairs.append(
                {
                    "run_key": row.get("run_key"),
                    "variant": variant,
                    "seed": seed,
                    "split_role": args.split_role,
                    **pack,
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Pairwise deltas per seed
    deltas = []
    for seed in seeds:
        b3 = next(p for p in pairs if p["variant"] == "b3" and p["seed"] == seed)
        b4 = next(p for p in pairs if p["variant"] == "b4" and p["seed"] == seed)
        deltas.append(
            {
                "seed": seed,
                "top1_b4_minus_b3_pp": float(b4["top1"] - b3["top1"]),
                "gate_rgb_mean": (b4.get("stream_gate") or {}).get("rgb_weight", {}).get("mean"),
                "gate_fused_mean": (b4.get("stream_gate") or {}).get("fused_weight", {}).get("mean"),
                "cross_entropy_mean": b4.get("cross_attention_entropy_mean"),
                "worst_signer_b3": b3.get("worst_signer"),
                "worst_signer_b4": b4.get("worst_signer"),
            }
        )

    out = {
        "schema": "ms_sta_retrain_v3_b3_b4_diagnosis_v1",
        "split_role": args.split_role,
        "n_ids": len(ids),
        "test_ids_touched": False,
        "policy_sha256": sha256_file(policy_path),
        "split_manifest_sha256": sha256_file(split_path),
        "training_git_revision": state.get("git_revision"),
        "diagnoser_git_revision": _git_rev(),
        "diagnoser_git_dirty": _git_dirty(),
        "diagnoser_script": "scripts/diagnose_ms_sta_b3_b4_v3.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": pairs,
        "per_seed_deltas": deltas,
        "hypothesis_note": (
            "If B4 gate collapses toward one stream or cross entropy is near-uniform, "
            "prefer residual-on-B3 (variant b4r) in a registered v4 campaign rather than "
            "replacing concat fusion. This diagnosis does not reopen v3 lock/test."
        ),
    }
    out_path = status_dir / "DIAGNOSIS_B3_B4.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": out_path.as_posix(), "deltas": deltas}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
