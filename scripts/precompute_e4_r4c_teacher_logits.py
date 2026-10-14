# -*- coding: utf-8 -*-
"""Precompute B4 teacher logits for E4-R4C confirm_train (run ONLY after HPO 8/8).

Creates create-only cache: sample_id → float32[400] logits.
Does not launch confirm. Requires explicit --i-understand-post-hpo-boundary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_split_role_ids


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-understand-post-hpo-boundary", action="store_true", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/teacher_logits_confirm_train.npz")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--require-hpo-selection", action="store_true", default=True)
    args = ap.parse_args()

    sel = ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/HPO_SELECTION.json"
    if args.require_hpo_selection and not sel.is_file():
        raise SystemExit("missing_hpo_selection_stop_before_cache")
    boundary = ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_POST_HPO_BOUNDARY_POLICY.json"
    if not boundary.is_file():
        raise SystemExit("missing_post_hpo_boundary_policy")

    train_pol = json.loads((ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / train_pol["split_manifest_rel"]).read_text(encoding="utf-8"))
    if sha256_file(ROOT / train_pol["split_manifest_rel"]) != train_pol["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")
    ckpt = ROOT / train_pol["teacher_checkpoint_rel"]
    if sha256_file(ckpt) != train_pol["teacher_checkpoint_sha256"]:
        raise SystemExit("teacher_sha_mismatch")

    out = args.out if args.out.is_absolute() else (ROOT / args.out)
    if out.is_file():
        raise SystemExit(f"refuse_overwrite:{out}")

    meta = load_meta(ROOT / "metadata.csv")
    labels_obj = json.loads((ROOT / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    labels = {}
    for sid, row in meta.items():
        gloss = str(row.get("text_gloss") or "").strip()
        if gloss in g2i:
            labels[sid] = int(g2i[gloss])
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ids = load_split_role_ids(split, "confirm_train")
    ds = VSL400MSSTAv5Dataset(
        ids,
        rgb_root=ROOT / "data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70",
        skel_root=ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5",
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_v5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required")
    model = build_ms_sta_v6("b4_v6_ctr").to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()

    sample_ids: list[str] = []
    logits_list: list[np.ndarray] = []
    n_batches = len(loader)
    with torch.inference_mode():
        for bi, batch in enumerate(loader, start=1):
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(x)["logits"].float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise SystemExit("non_finite_teacher_logits")
            for i, sid in enumerate(batch["sample_id"]):
                sample_ids.append(str(sid))
                logits_list.append(np.asarray(logits[i], dtype=np.float32))
            if bi == 1 or bi % 50 == 0 or bi == n_batches:
                print(f"TEACHER_LOGITS_PROGRESS {bi}/{n_batches} n={len(sample_ids)}", flush=True)

    arr = np.stack(logits_list, axis=0)
    lock_path = ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_CODE_LOCK.json"
    if not lock_path.is_file():
        raise SystemExit("missing_hpo_code_lock_for_producer_bind")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "HPO_CODE_LOCK_PASS":
        raise SystemExit("hpo_code_lock_not_pass")
    producer_script = Path(__file__).resolve()
    meta_out = {
        "schema": "ms_sta_v6_e4_r4c_teacher_logits_v1",
        "role": "confirm_train",
        "n": len(sample_ids),
        "dtype": "float32",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "teacher_variant": "b4_v6_ctr",
        "teacher_checkpoint_rel": train_pol["teacher_checkpoint_rel"],
        "teacher_checkpoint_sha256": train_pol["teacher_checkpoint_sha256"],
        "split_manifest_rel": train_pol["split_manifest_rel"],
        "split_manifest_sha256": train_pol["split_manifest_sha256"],
        "ordered_sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "producer_script_rel": str(producer_script.relative_to(ROOT)).replace("\\", "/"),
        "producer_script_sha256": sha256_file(producer_script),
        "producer_hpo_code_lock_rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_CODE_LOCK.json",
        "producer_hpo_code_lock_sha256": sha256_file(lock_path),
        "producer_hpo_code_lock_git_head": lock.get("git_head"),
        "producer_executable_tree_sha256": lock.get("executable_tree_sha256"),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "e4_ready": False,
        "note": "Use only after cached-vs-online parity PASS; does not authorize confirm by itself.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # numpy savez_compressed appends .npz unless the path already ends with .npz
    tmp = out.with_name(out.stem + ".writing.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(
        tmp,
        sample_ids=np.asarray(sample_ids),
        logits=arr,
        meta_json=np.asarray(json.dumps(meta_out)),
    )
    tmp.replace(out)
    print("TEACHER_LOGITS_OK", out, "n", len(sample_ids), "sha", sha256_file(out)[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
