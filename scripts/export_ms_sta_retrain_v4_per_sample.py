"""Export aligned per-sample validation JSONL from retrain-v4 final checkpoints.

Does not train. Requires FROZEN policy and 12 confirm finals. Emits one JSONL per
variant/seed under --out-dir/{variant}/seed{N}.jsonl with exact val IDs.

Fail-closed: verifies RUN_STATE / SELECTED_CONFIGS / status final SHA / payload
identity / policy+split+git locks / active feature dirs before export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import load_json, sha256_file
from src.models.ms_sta_features import resolve_pooled_dir, resolve_skel_dir, resolve_temporal_dir
from src.models.ms_sta_wordlevel import build_ms_sta
from src.training.retrain_v3 import final_stem_from_candidate_stem
from src.training.retrain_v4 import (
    CONFIRM_SEEDS,
    CONFIRM_VARIANTS,
    exact_id_list_sha256,
    lineage_block,
    load_policy,
    require_exact_str,
    require_frozen_v4_policy,
    require_hex64,
    run_key,
    verify_checkpoint_payload_identity,
)
from src.training.train_ms_sta_vsl400 import VSL400MSSTADataset, _collate

REGISTRY_PATH = _ROOT / "data/processed/ms_sta_registry.json"


def _norm_signer(value: object) -> str:
    s = str(value).strip()
    if s.startswith("signer_"):
        return s
    if s.isdigit() or s.lstrip("0").isdigit() or s == "0":
        return f"signer_{int(s)}"
    return s


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_ROOT.resolve()).as_posix()


@torch.no_grad()
def _predict_by_id(model, loader, device, variant: str) -> dict[str, dict]:
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
            yi = int(y[i].item())
            pi = int(pred[i].item())
            top5 = [int(x) for x in t5[i].tolist()]
            by_id[str(vid)] = {
                "y_true": yi,
                "y_pred": pi,
                "correct_top1": int(pi == yi),
                "correct_top5": int(yi in top5),
                "top5": top5,
            }
    return by_id


def _verify_feature_dirs(policy: dict, *, rgb_temporal_dir: Path, skel_dir: Path) -> None:
    runtime = policy.get("runtime_locks") or {}
    active_skel = str(runtime.get("active_skel_dir") or "").replace("\\", "/")
    active_temp = str(runtime.get("active_temporal_dir") or "").replace("\\", "/")
    if active_skel:
        require_exact_str(_rel(skel_dir), active_skel, label="active_skel_dir")
    if active_temp:
        require_exact_str(_rel(rgb_temporal_dir), active_temp, label="active_temporal_dir")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--variants", default="b3,b4r")
    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[FAIL] CUDA unavailable for per-sample export")
        return 2

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v4_policy(policy, root=_ROOT)
    policy_sha = sha256_file(policy_path)
    lineage = lineage_block(policy)
    split_path = _ROOT / str(lineage["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha = require_hex64(
        lineage.get("split_manifest_sha256"), label="lineage.split_manifest_sha256"
    )
    if sha256_file(split_path) != split_sha:
        print("[FAIL] split manifest sha mismatch vs lineage")
        return 2
    val_ids = [str(x) for x in split["val_ids"]]
    expected = sorted(val_ids)
    if len(expected) != int(lineage.get("n_val") or split.get("n_val") or len(expected)):
        print("[FAIL] n_val mismatch")
        return 2

    labels = load_json(_ROOT / "data/processed/vsl400_labels.json")
    gloss_to_id = labels["gloss_to_id"]
    num_classes = int(labels["num_classes"])
    meta = pd.read_csv(_ROOT / "metadata.csv")
    v = meta[meta["dataset"] == "VSL400"].copy()
    v["video_id"] = v["video_id"].astype(str)
    v["signer_norm"] = v["signer_id"].map(_norm_signer)
    signer_by_id = {
        str(r.video_id): str(r.signer_norm) for r in v.itertuples(index=False)
    }

    reg = load_json(REGISTRY_PATH) if REGISTRY_PATH.is_file() else {}
    rgb_pooled_dir = resolve_pooled_dir(_ROOT, reg)
    rgb_temporal_dir = resolve_temporal_dir(_ROOT, reg)
    skel_dir = resolve_skel_dir(_ROOT, reg)
    try:
        _verify_feature_dirs(policy, rgb_temporal_dir=rgb_temporal_dir, skel_dir=skel_dir)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    device = torch.device(args.device)

    art = policy.get("artifact_dirs") or {}
    ckpt_dir = _ROOT / str(art.get("checkpoints") or "data/processed/ms_sta_retrain_v4")
    status_dir = _ROOT / str(art.get("status") or "logs/ms_sta_retrain_v4")
    run_state_path = status_dir / "RUN_STATE.json"
    selected_path = status_dir / "SELECTED_CONFIGS.json"
    if not run_state_path.is_file():
        print("[FAIL] missing RUN_STATE.json")
        return 2
    if not selected_path.is_file():
        print("[FAIL] missing SELECTED_CONFIGS.json")
        return 2

    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_sha = sha256_file(selected_path)
    run_state_sha = sha256_file(run_state_path)
    try:
        require_exact_str(run_state.get("policy_sha256"), policy_sha, label="RUN_STATE.policy_sha256")
        require_exact_str(
            run_state.get("split_manifest_sha256"),
            split_sha,
            label="RUN_STATE.split_manifest_sha256",
        )
        git_rev = str(run_state.get("git_revision") or "").strip()
        if not git_rev:
            raise ValueError("RUN_STATE.git_revision missing")
        recorded_selected = str(run_state.get("selected_configs_sha256") or "").strip()
        if recorded_selected:
            require_exact_str(
                recorded_selected, selected_sha, label="RUN_STATE.selected_configs_sha256"
            )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    configs = selected["selected"]
    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    if set(variants) != {"b3", "b4r"}:
        # Primary gate export is exactly B3/B4R (6 JSONL). Extra variants need a separate path.
        if not set(variants).issubset(set(CONFIRM_VARIANTS)):
            print(f"[FAIL] unsupported variants {variants}")
            return 2
        if set(variants) != {"b3", "b4r"}:
            print("[FAIL] primary per-sample export requires exactly variants=b3,b4r")
            return 2
    out_root = args.out_dir if args.out_dir.is_absolute() else _ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = _ROOT / "metadata.csv"
    labels_path = _ROOT / "data/processed/vsl400_labels.json"
    signer_metadata_sha = sha256_file(meta_path)
    # Bind labels too via combined digest for lock/evaluator.
    import hashlib

    signer_metadata_sha = hashlib.sha256(
        (signer_metadata_sha + ":" + sha256_file(labels_path)).encode("utf-8")
    ).hexdigest()
    id_sha = exact_id_list_sha256(expected)

    print(
        json.dumps(
            {
                "run_state_sha256": run_state_sha,
                "selected_configs_sha256": selected_sha,
                "policy_sha256": policy_sha,
                "split_manifest_sha256": split_sha,
                "git_revision": git_rev,
                "exact_id_sha256": id_sha,
                "n_val": len(expected),
            },
            indent=2,
        )
    )

    jsonl_shas: dict[str, str] = {}
    final_shas: dict[str, str] = {}
    path_map: dict[str, str] = {}

    for variant in variants:
        if variant not in CONFIRM_VARIANTS:
            print(f"[FAIL] unsupported variant {variant}")
            return 2
        require_skel = variant in ("b2", "b3", "b4", "b4r", "b5")
        require_temporal = variant in ("b1", "b3", "b4", "b4r", "b5")
        ds = VSL400MSSTADataset(
            v,
            gloss_to_id,
            require_skel=require_skel,
            require_temporal_rgb=require_temporal,
            rgb_pooled_dir=rgb_pooled_dir,
            rgb_temporal_dir=rgb_temporal_dir,
            skel_dir=skel_dir,
            video_ids=val_ids,
        )
        loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=_collate)
        for seed in CONFIRM_SEEDS:
            cid = configs[variant]
            stem = run_key(
                {
                    "phase": "confirm",
                    "variant": variant,
                    "config_id": cid,
                    "seed": seed,
                    "skel_temporal_stride": 1,
                }
            )
            status_path = status_dir / f"{stem}.json"
            cand_path = ckpt_dir / f"{stem}.pth"
            final_path = ckpt_dir / f"{final_stem_from_candidate_stem(stem)}.pth"
            if not status_path.is_file() or not final_path.is_file():
                print(f"[FAIL] missing status/final for {stem}")
                return 2
            status = json.loads(status_path.read_text(encoding="utf-8"))
            final_sha = sha256_file(final_path)
            try:
                require_exact_str(
                    status.get("checkpoint_status"),
                    "final_selected_checkpoint",
                    label="status.checkpoint_status",
                )
                require_exact_str(
                    status.get("final_checkpoint_sha256"),
                    final_sha,
                    label="status.final_checkpoint_sha256",
                )
                require_exact_str(
                    status.get("retrain_v4_policy_sha256"),
                    policy_sha,
                    label="status.retrain_v4_policy_sha256",
                )
                require_exact_str(
                    status.get("manifest_sha256"),
                    split_sha,
                    label="status.manifest_sha256",
                )
                require_exact_str(
                    status.get("git_revision"),
                    git_rev,
                    label="status.git_revision",
                )
                payload = torch.load(final_path, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict):
                    raise ValueError(f"final payload must be dict: {final_path}")
                verify_checkpoint_payload_identity(
                    payload,
                    phase="confirm",
                    variant=variant,
                    config_id=str(cid),
                    seed=int(seed),
                    expected_policy_sha=policy_sha,
                    expected_split_sha=split_sha,
                    expected_git_rev=git_rev,
                    require_final=True,
                )
                if cand_path.is_file():
                    require_exact_str(
                        status.get("source_candidate_sha256"),
                        sha256_file(cand_path),
                        label="status.source_candidate_sha256",
                    )
            except ValueError as exc:
                print(f"[FAIL] identity for {stem}: {exc}")
                return 2

            residual_init = payload.get("residual_init") or (
                "zero_delta_head" if variant == "b4r" else "none"
            )
            model = build_ms_sta(
                variant,
                num_classes=num_classes,
                residual_init=residual_init if variant == "b4r" else "zero_delta_head",
            ).to(device)
            model.load_state_dict(payload["model"], strict=True)
            preds = _predict_by_id(model, loader, device, variant)
            missing = [i for i in expected if i not in preds]
            if missing:
                print(f"[FAIL] missing preds n={len(missing)} e.g. {missing[:5]}")
                return 2
            out_dir = out_root / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"seed{seed}.jsonl"
            with out_path.open("w", encoding="utf-8", newline="\n") as fh:
                for sid in expected:
                    row = preds[sid]
                    signer = signer_by_id.get(sid, "")
                    if not signer:
                        print(f"[FAIL] missing signer for {sid}")
                        return 2
                    fh.write(
                        json.dumps(
                            {
                                "sample_id": sid,
                                "signer_id": signer,
                                "seed": int(seed),
                                "variant": variant,
                                "y_true": row["y_true"],
                                "y_pred": row["y_pred"],
                                "correct_top1": row["correct_top1"],
                                "correct_top5": row["correct_top5"],
                                "top5": row["top5"],
                                "final_checkpoint_sha256": final_sha,
                                "policy_sha256": policy_sha,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            key = f"{variant}/seed{seed}"
            jsonl_shas[key] = sha256_file(out_path)
            final_shas[key] = final_sha
            path_map[key] = _rel(out_path)
            print(f"[ok] wrote {out_path} n={len(expected)}")

    manifest = {
        "schema": "ms_sta_retrain_v4_per_sample_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": git_rev,
        "n_val": len(expected),
        "exact_id_sha256": id_sha,
        "signer_metadata_sha256": signer_metadata_sha,
        "jsonl_sha256_by_variant_seed": jsonl_shas,
        "final_checkpoint_sha256_by_variant_seed": final_shas,
        "jsonl_paths_by_variant_seed": path_map,
        "variants": sorted(variants),
        "seeds": list(CONFIRM_SEEDS),
        "run_state_sha256": run_state_sha,
        "selected_configs_sha256": selected_sha,
    }
    man_path = out_root / "PER_SAMPLE_MANIFEST.json"
    tmp = man_path.with_suffix(man_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, man_path)
    print(
        json.dumps(
            {
                "ok": True,
                "per_sample_manifest": _rel(man_path),
                "per_sample_manifest_sha256": sha256_file(man_path),
                "n_jsonl": len(jsonl_shas),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
