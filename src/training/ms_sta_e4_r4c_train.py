# -*- coding: utf-8 -*-
"""Train / evaluate MS-STA v6 E4-R4C (R1+S_A) with optional logit KD from B4 teacher.

Teacher forward uses sealed B4 feature roots (read-only). Student uses compact E4-R4C root.
Never opens test role. Never writes B4 checkpoints / registry / FINAL_RESULTS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]

from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.ms_sta_v6_e4_r4 import build_ms_sta_v6_e4_r4  # noqa: E402
from src.training.ms_sta_e4_r4c_data import (  # noqa: E402
    VSL400E4R4CDataset,
    collate_e4_r4c,
    load_json,
    role_ids_forbid_test,
)
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def _kl_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, collect_rows: bool = False) -> dict[str, Any]:
    model.eval()
    correct = 0
    top5 = 0
    n = 0
    rows = []
    all_preds = []
    all_labels = []
    for batch in loader:
        labels = batch["label"].to(device)
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        # Model owns all-pad skeleton RGB/base fallback; do not fabricate joints here.
        tvalid = x["temporal_valid"]
        if tvalid.dtype != torch.bool:
            tvalid = tvalid.bool()
        none_valid = ~tvalid.any(dim=1)
        if bool(none_valid.any()):
            tvalid = tvalid.clone()
            tvalid[none_valid, 0] = True
            x["temporal_valid"] = tvalid
        out = model(x)
        logits = out["logits"]
        if not torch.isfinite(logits).all():
            raise RuntimeError("non_finite_logits_in_eval")
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        top5 += int((logits.topk(5, dim=-1).indices == labels.unsqueeze(-1)).any(dim=-1).sum().item())
        n += int(labels.numel())
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if collect_rows:
            probs = torch.softmax(logits, dim=-1)
            for i, sid in enumerate(batch["sample_id"]):
                lab = int(labels[i].item())
                rows.append(
                    {
                        "sample_id": sid,
                        "signer": batch["signer"][i],
                        "label": lab,
                        "pred": int(pred[i].item()),
                        "correct": bool(pred[i].item() == lab),
                        "top5_hit": bool(lab in logits[i].topk(5).indices.tolist()),
                        "prob_true": float(probs[i, lab].item()),
                    }
                )
    # macro-F1
    from collections import defaultdict

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, y in zip(all_preds, all_labels):
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    f1s = []
    for c in set(all_labels):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return {
        "n": n,
        "top1": correct / max(n, 1),
        "top5": top5 / max(n, 1),
        "macro_f1": float(sum(f1s) / max(len(f1s), 1)),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-policy", type=Path, required=True)
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-role", required=True)
    ap.add_argument("--val-role", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--distill-weight", type=float, default=None)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--config-id", type=str, default="")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument(
        "--teacher-logits-cache",
        type=Path,
        default=None,
        help="Optional confirm_train teacher logit cache (.npz); performance-only when Gate C PASS.",
    )
    args = ap.parse_args()

    policy = load_json(_ROOT / args.train_policy if not args.train_policy.is_absolute() else args.train_policy)
    if policy.get("status") != "FROZEN":
        raise SystemExit("train_policy_not_frozen")
    if policy.get("candidate_id") != "R1_S_A_d128":
        raise SystemExit("wrong_candidate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required")
    gpu = torch.cuda.get_device_name(0)
    if gpu != policy["scientific_environment_lock"]["required_gpu_name"]:
        raise SystemExit(f"gpu_mismatch:{gpu}")

    split = load_json(_ROOT / policy["split_manifest_rel"])
    if sha256_file(_ROOT / policy["split_manifest_rel"]) != policy["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")
    teacher_path = _ROOT / policy["teacher_checkpoint_rel"]
    if sha256_file(teacher_path) != policy["teacher_checkpoint_sha256"]:
        raise SystemExit("teacher_sha_mismatch")

    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {}
    for sid, row in meta.items():
        gloss = str(row.get("text_gloss") or "").strip()
        if gloss in g2i:
            labels[sid] = int(g2i[gloss])

    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    train_ids = role_ids_forbid_test(split, args.train_role)
    val_ids = role_ids_forbid_test(split, args.val_role)
    if set(train_ids) & test_ids or set(val_ids) & test_ids:
        raise SystemExit("test_leakage")

    feature_root = args.feature_root if args.feature_root.is_absolute() else (_ROOT / args.feature_root)
    train_ds = VSL400E4R4CDataset(train_ids, feature_root=feature_root, labels=labels, metadata_by_id=meta, forbid_ids=test_ids)
    val_ds = VSL400E4R4CDataset(val_ids, feature_root=feature_root, labels=labels, metadata_by_id=meta, forbid_ids=test_ids)

    bs = int(args.batch_size or policy.get("hpo", {}).get("batch_size") or policy.get("confirm", {}).get("batch_size") or 8)
    nw = int(args.num_workers)
    pin = bool(args.pin_memory)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        collate_fn=collate_e4_r4c,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        collate_fn=collate_e4_r4c,
    )

    # Teacher B4 roots (read-only) — logit KD only; same IDs
    b4_rgb = _ROOT / "data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70"
    b4_skel = _ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5"
    teacher_train_ds = None
    teacher_index: dict[str, int] = {}
    teacher_logits_by_id: dict[str, torch.Tensor] | None = None
    if args.teacher_logits_cache is not None:
        import numpy as np

        cache_path = args.teacher_logits_cache if args.teacher_logits_cache.is_absolute() else (_ROOT / args.teacher_logits_cache)
        if not cache_path.is_file():
            raise SystemExit(f"missing_teacher_logits_cache:{cache_path}")
        if not str(args.train_role).startswith("confirm"):
            raise SystemExit("teacher_logits_cache_only_for_confirm_roles")
        blob = np.load(cache_path, allow_pickle=False)
        c_ids = [str(x) for x in blob["sample_ids"].tolist()]
        c_logits = blob["logits"]
        if set(c_ids) & test_ids:
            raise SystemExit("teacher_cache_contains_test_ids")
        missing = [sid for sid in train_ids if sid not in set(c_ids)]
        if missing:
            raise SystemExit(f"teacher_cache_missing_train_ids:{missing[:8]}")
        teacher_logits_by_id = {
            sid: torch.from_numpy(np.asarray(c_logits[i], dtype=np.float32)) for i, sid in enumerate(c_ids)
        }
    else:
        teacher_train_ds = VSL400MSSTAv5Dataset(
            train_ids, rgb_root=b4_rgb, skel_root=b4_skel, labels=labels, metadata_by_id=meta, forbid_ids=test_ids
        )
        teacher_index = {sid: i for i, sid in enumerate(teacher_train_ds.sample_ids)}

    torch.manual_seed(int(args.seed))
    model = build_ms_sta_v6_e4_r4(rgb_id="R1", skel_layout="S_A", d_model=int(policy["model"]["d_model"])).to(device)
    teacher = None
    if teacher_logits_by_id is None:
        teacher = build_ms_sta_v6("b4_v6_ctr").to(device)
        ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    distill_cfg = policy.get("distill") or {}
    distill_w = float(args.distill_weight if args.distill_weight is not None else distill_cfg.get("weight", 0.5))
    distill_temp = float(distill_cfg.get("temperature", 2.0))

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    out_dir = args.out_dir if args.out_dir.is_absolute() else (_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.epochs > 0:
        max_epochs = int(args.epochs)
    elif args.train_role.startswith("confirm"):
        max_epochs = int(policy["confirm"]["epochs"])
    else:
        max_epochs = int(policy["hpo"]["max_epochs"])
    early = bool(args.early_stopping or (policy.get("hpo", {}).get("early_stopping") and not args.train_role.startswith("confirm")))
    patience = int(args.patience or policy.get("hpo", {}).get("patience", 2))

    history = []
    best_top1 = -1.0
    best_path = out_dir / "best.pt"
    bad = 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_train_batches = len(train_loader)
        for batch in train_loader:
            labels_t = batch["label"].to(device)
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            # Model owns all-pad skeleton RGB/base fallback; do not fabricate joints here.
            tvalid = x["temporal_valid"]
            if tvalid.dtype != torch.bool:
                tvalid = tvalid.bool()
            none_valid = ~tvalid.any(dim=1)
            if bool(none_valid.any()):
                tvalid = tvalid.clone()
                tvalid[none_valid, 0] = True
                x["temporal_valid"] = tvalid
            out = model(x)
            if not torch.isfinite(out["logits"]).all():
                raise RuntimeError("non_finite_logits_in_train")
            loss = F.cross_entropy(out["logits"], labels_t)
            if distill_w > 0:
                if teacher_logits_by_id is not None:
                    t_logits = torch.stack(
                        [teacher_logits_by_id[str(sid)] for sid in batch["sample_id"]], dim=0
                    ).to(device)
                else:
                    assert teacher is not None and teacher_train_ds is not None
                    # Build teacher batch aligned by sample_id
                    t_items = []
                    for sid in batch["sample_id"]:
                        t_items.append(teacher_train_ds[teacher_index[sid]])
                    t_batch = collate_v5(t_items)
                    t_x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t_batch.items()}
                    with torch.no_grad():
                        t_out = teacher(t_x)
                    t_logits = t_out["logits"]
                loss = loss + distill_w * _kl_logits(out["logits"], t_logits, distill_temp)
            if not torch.isfinite(loss):
                raise RuntimeError("non_finite_loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError("non_finite_grad")
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            if n_batches == 1 or n_batches % 200 == 0:
                print(f"  train_batch {n_batches}/{n_train_batches} loss={loss.item():.4f}", flush=True)

        metrics = evaluate(model, val_loader, device, collect_rows=False)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_top1": metrics["top1"],
            "val_top5": metrics["top5"],
            "val_macro_f1": metrics["macro_f1"],
            "elapsed_sec": time.time() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{max_epochs} loss={row['train_loss']:.4f} "
            f"top1={metrics['top1']*100:.2f} top5={metrics['top5']*100:.2f}",
            flush=True,
        )
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": int(args.seed),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "config_id": args.config_id,
                    "candidate_id": "R1_S_A_d128",
                    "val_top1": best_top1,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            bad += 1
            if early and bad >= patience:
                print("early_stop", epoch, flush=True)
                break

    # Final eval with rows
    ckpt_best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_best["model"])
    final = evaluate(model, val_loader, device, collect_rows=True)
    pred_path = out_dir / "val_predictions.json"
    pred_path.write_text(
        json.dumps(
            {
                "schema": "ms_sta_v6_e4_r4c_val_preds_v1",
                "candidate_id": "R1_S_A_d128",
                "seed": int(args.seed),
                "config_id": args.config_id,
                "n": final["n"],
                "top1": final["top1"],
                "top5": final["top5"],
                "macro_f1": final["macro_f1"],
                "rows": final["rows"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "history.json").write_text(json.dumps({"history": history}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": "ms_sta_v6_e4_r4c_train_summary_v1",
        "candidate_id": "R1_S_A_d128",
        "seed": int(args.seed),
        "config_id": args.config_id,
        "best_val_top1": best_top1,
        "final_top1": final["top1"],
        "final_top5": final["top5"],
        "final_macro_f1": final["macro_f1"],
        "feature_root": str(feature_root.relative_to(_ROOT)).replace("\\", "/"),
        "e4_ready": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TRAIN_OK", out_dir, "best_top1", round(best_top1 * 100, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
