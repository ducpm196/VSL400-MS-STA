# -*- coding: utf-8 -*-
"""Independent cached-vs-online teacher logit parity for E4-R4C confirm_train.

Thresholds declared BEFORE seeing errors (frozen in this file).
Online recompute MUST use the same producer batch grouping (batch_size=8,
shuffle=False, contiguous confirm_train order) because collate padding is
batch-dependent; regrouping samples changes logits without implying a bad cache.

Create-only: Docs/results/ms_sta_v6_e4_r4/E4_R4C_TEACHER_LOGITS_PARITY.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_split_role_ids

# Declared before run — do not change after seeing errors.
PARITY_MAX_ABS_ERR = 1e-5
PARITY_MEAN_ABS_ERR = 1e-6
HOLDOUT_N = 64
HOLDOUT_SEED = 42
TOPK = 5
PRODUCER_BATCH_SIZE = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(path: Path) -> dict[str, dict]:
    import csv

    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/teacher_logits_confirm_train.npz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_TEACHER_LOGITS_PARITY.json",
    )
    args = ap.parse_args()
    cache = args.cache if args.cache.is_absolute() else ROOT / args.cache
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise SystemExit(f"refuse_overwrite:{out}")
    if not cache.is_file():
        raise SystemExit(f"missing_cache:{cache}")

    train_pol = json.loads(
        (ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json").read_text(encoding="utf-8")
    )
    split = json.loads((ROOT / train_pol["split_manifest_rel"]).read_text(encoding="utf-8"))
    if sha256_file(ROOT / train_pol["split_manifest_rel"]) != train_pol["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")
    ckpt = ROOT / train_pol["teacher_checkpoint_rel"]
    if sha256_file(ckpt) != train_pol["teacher_checkpoint_sha256"]:
        raise SystemExit("teacher_sha_mismatch")

    blob = np.load(cache, allow_pickle=False)
    cached_ids = [str(x) for x in blob["sample_ids"].tolist()]
    cached_logits = np.asarray(blob["logits"], dtype=np.float32)
    meta_json = json.loads(str(blob["meta_json"].item() if hasattr(blob["meta_json"], "item") else blob["meta_json"]))
    if cached_logits.ndim != 2 or cached_logits.shape[1] != 400:
        raise SystemExit(f"bad_shape:{cached_logits.shape}")
    if len(cached_ids) != cached_logits.shape[0]:
        raise SystemExit("id_logit_len_mismatch")
    if not np.isfinite(cached_logits).all():
        raise SystemExit("cached_non_finite")
    if meta_json.get("role") != "confirm_train":
        raise SystemExit("role_mismatch")
    if len(set(cached_ids)) != len(cached_ids):
        raise SystemExit("duplicate_cached_ids")

    confirm_ids = load_split_role_ids(split, "confirm_train")
    if cached_ids != [str(x) for x in confirm_ids]:
        raise SystemExit("cached_ids_order_mismatch_vs_confirm_train")

    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    if set(cached_ids) & test_ids:
        raise SystemExit("cache_contains_test_ids")

    rng = np.random.default_rng(HOLDOUT_SEED)
    n = len(cached_ids)
    hold_n = min(HOLDOUT_N, n)
    hold_idx = sorted(rng.choice(n, size=hold_n, replace=False).tolist())
    hold_ids = [cached_ids[i] for i in hold_idx]

    # Expand to producer batch windows (same as precompute: bs=8, shuffle=False).
    batch_starts = sorted({(i // PRODUCER_BATCH_SIZE) * PRODUCER_BATCH_SIZE for i in hold_idx})
    recompute_idx: list[int] = []
    for start in batch_starts:
        end = min(start + PRODUCER_BATCH_SIZE, n)
        recompute_idx.extend(range(start, end))

    meta = load_meta(ROOT / "metadata.csv")
    labels_obj = json.loads((ROOT / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    labels = {}
    for sid, row in meta.items():
        gloss = str(row.get("text_gloss") or "").strip()
        if gloss in g2i:
            labels[sid] = int(g2i[gloss])

    ds = VSL400MSSTAv5Dataset(
        confirm_ids,
        rgb_root=ROOT / "data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70",
        skel_root=ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required")
    model = build_ms_sta_v6("b4_v6_ctr").to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    online_by_id: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in batch_starts:
            end = min(start + PRODUCER_BATCH_SIZE, n)
            idxs = list(range(start, end))
            loader = DataLoader(
                Subset(ds, idxs),
                batch_size=len(idxs),
                shuffle=False,
                num_workers=0,
                collate_fn=collate_v5,
            )
            batch = next(iter(loader))
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(x)["logits"].float().cpu().numpy().astype(np.float32)
            if not np.isfinite(logits).all():
                raise SystemExit("online_non_finite")
            for i, sid in enumerate(batch["sample_id"]):
                online_by_id[str(sid)] = logits[i]

    abs_errs = []
    argmax_agree = 0
    topk_agree = 0
    id_to_pos = {sid: i for i, sid in enumerate(confirm_ids)}
    for sid in hold_ids:
        c = cached_logits[id_to_pos[sid]]
        o = online_by_id[sid]
        abs_errs.append(np.abs(c - o))
        if int(np.argmax(c)) == int(np.argmax(o)):
            argmax_agree += 1
        if set(np.argsort(-c)[:TOPK].tolist()) == set(np.argsort(-o)[:TOPK].tolist()):
            topk_agree += 1
    stacked = np.stack(abs_errs, axis=0)
    max_abs = float(stacked.max())
    mean_abs = float(stacked.mean())
    pass_parity = (
        max_abs <= PARITY_MAX_ABS_ERR
        and mean_abs <= PARITY_MEAN_ABS_ERR
        and argmax_agree == hold_n
        and topk_agree == hold_n
    )

    report = {
        "schema": "ms_sta_v6_e4_r4c_teacher_logits_parity_v2",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cache_rel": str(cache.relative_to(ROOT)).replace("\\", "/"),
        "cache_sha256": sha256_file(cache),
        "teacher_checkpoint_sha256": train_pol["teacher_checkpoint_sha256"],
        "split_manifest_sha256": train_pol["split_manifest_sha256"],
        "role": "confirm_train",
        "n_cache": n,
        "ordered_sample_ids_sha256": hashlib.sha256("\n".join(cached_ids).encode("utf-8")).hexdigest(),
        "producer_batch_alignment": {
            "batch_size": PRODUCER_BATCH_SIZE,
            "shuffle": False,
            "note": "Online holdout recomputed inside original producer batch windows to match collate padding.",
            "n_batch_windows": len(batch_starts),
            "n_recomputed_samples": len(recompute_idx),
        },
        "prior_fail_artifact": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_TEACHER_LOGITS_PARITY_FAIL_batch_misaligned_checker.json",
        "holdout": {
            "n": hold_n,
            "seed": HOLDOUT_SEED,
            "selection": "numpy_default_rng_choice_sorted_indices_declared_before_errors",
            "sample_ids": hold_ids,
        },
        "thresholds_declared_before_run": {
            "max_abs_err": PARITY_MAX_ABS_ERR,
            "mean_abs_err": PARITY_MEAN_ABS_ERR,
            "require_argmax_agreement": True,
            "require_topk_set_agreement": TOPK,
        },
        "observed": {
            "max_abs_err": max_abs,
            "mean_abs_err": mean_abs,
            "argmax_agree": argmax_agree,
            "topk_agree": topk_agree,
        },
        "gate_c": "PASS" if pass_parity else "FAIL",
        "e4_ready": False,
        "cache_meta": meta_json,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PARITY", report["gate_c"], "max_abs", max_abs, "mean_abs", mean_abs)
    return 0 if pass_parity else 2


if __name__ == "__main__":
    raise SystemExit(main())
