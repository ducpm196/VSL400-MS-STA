#!/usr/bin/env python3
"""Train one MS-STA v5 Stage-A trial (HPO or confirm). Never opens test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_v5 import build_ms_sta_v5  # noqa: E402
from src.models.ms_sta_v5_e2 import build_ms_sta_v5_e2  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.ms_sta_v6_e3 import build_ms_sta_v6_e3  # noqa: E402
from src.models.ms_sta_v6_e4 import build_ms_sta_v6_e4  # noqa: E402
from src.models.ms_sta_v6_e4_r2 import build_ms_sta_v6_e4_r2  # noqa: E402
from src.models.ms_sta_v6_e4_r3 import build_ms_sta_v6_e4_r3  # noqa: E402
from src.training.ms_sta_e4_r2_provenance import (  # noqa: E402
    assert_no_drift,
    build_trial_bindings,
    refuse_existing_outputs,
    sha256_file,
    verify_sealed_code_lock,
    write_provenance,
)
from src.training.ms_sta_v5_data import (  # noqa: E402
    VSL400MSSTAv5Dataset,
    collate_v5,
    load_json,
    load_split_role_ids,
)

_E2_VARIANTS = frozenset({"e2_s2", "e2_lite_s2", "e2_lite_s4", "e1_late_s4"})
_V6_VARIANTS = frozenset(
    {
        "rgb_strong",
        "b3_strong",
        "b3_mlp_strong",
        "b3_mlp_ctr",
        "b3_ctr_concat",
        "b4_v6",
        "b4_v6_ctr",
        "b4_v6_bi",
        "b4_v6_ctr_bi",
    }
)
_E3_VARIANTS = frozenset({"e3_lite_s2", "e3_lite_s4", "e3_balanced_s2"})
_E4_VARIANTS = frozenset({"e4_w192", "e4_w192_t1", "e4_w160_t1"})
_E4R2_VARIANTS = frozenset({"e4r2_A", "e4r2_B", "e4r2_C", "e4r2_D", "e4r2_E"})
_E4R3_VARIANTS = frozenset({"e4r3_A", "e4r3_B", "e4r3_C"})


def _e4r2_arm(variant: str) -> str:
    if variant not in _E4R2_VARIANTS:
        raise ValueError(f"not_e4r2_variant:{variant}")
    return variant.split("_", 1)[1]


def _e4r3_arm(variant: str) -> str:
    if variant not in _E4R3_VARIANTS:
        raise ValueError(f"not_e4r3_variant:{variant}")
    return variant.split("_", 1)[1]


def _kl_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = float(temperature)
    log_p = F.log_softmax(student_logits / t, dim=-1)
    q = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata_by_id(meta_csv: Path) -> dict[str, dict[str, Any]]:
    """Bind VSL400 rows by video_id (zero-padded sample id) + text_gloss."""
    out: dict[str, dict[str, Any]] = {}
    with meta_csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("dataset") or "") != "VSL400":
                continue
            sid = str(row.get("video_id") or row.get("sample_id") or "").strip()
            if not sid:
                continue
            gloss = str(row.get("text_gloss") or row.get("gloss") or "").strip()
            signer = str(row.get("signer_id") or row.get("signer") or "").strip()
            out[sid] = {
                "signer": signer,
                "gloss": gloss,
            }
    if not out:
        raise RuntimeError(f"metadata_empty_or_unmapped:{meta_csv}")
    return out


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    *,
    collect_rows: bool = False,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    top5 = 0
    rows: list[dict[str, Any]] = []
    # Per-class counts for macro-F1 (400-way).
    tp = torch.zeros(400, dtype=torch.long)
    fp = torch.zeros(400, dtype=torch.long)
    fn = torch.zeros(400, dtype=torch.long)
    for batch in loader:
        labels = batch["label"].to(device)
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(x)["logits"]
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        top5_hits = (logits.topk(5, dim=-1).indices == labels.unsqueeze(-1)).any(dim=-1)
        top5 += int(top5_hits.sum().item())
        pred_i = pred.detach().cpu()
        lab_i = labels.detach().cpu()
        for i in range(lab_i.numel()):
            y = int(lab_i[i].item())
            p = int(pred_i[i].item())
            if p == y:
                tp[y] += 1
            else:
                fp[p] += 1
                fn[y] += 1
        if collect_rows:
            top5_idx = logits.topk(5, dim=-1).indices.cpu().tolist()
            for i, sid in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sid,
                        "signer": batch["signer"][i],
                        "label": int(labels[i].item()),
                        "pred": int(pred[i].item()),
                        "correct": bool(pred[i].item() == labels[i].item()),
                        "top5_hit": bool(top5_hits[i].item()),
                        "top5": top5_idx[i],
                    }
                )
    f1s: list[float] = []
    for c in range(400):
        denom = (2 * tp[c] + fp[c] + fn[c]).item()
        if denom == 0:
            continue
        f1s.append(float((2 * tp[c]).item() / denom))
    macro_f1 = sum(f1s) / max(len(f1s), 1)
    out: dict[str, Any] = {
        "top1": correct / max(total, 1),
        "top5": top5 / max(total, 1),
        "macro_f1": macro_f1,
        "n": float(total),
    }
    if collect_rows:
        out["rows"] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--phase", choices=["hpo", "confirm"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--distill-weight", type=float, default=None)
    args = ap.parse_args()

    policy = load_json(Path(args.policy))
    if policy.get("status") != "FROZEN" or policy.get("train_allowed") is not True:
        raise SystemExit("policy_not_frozen_or_train_disallowed")
    if policy.get("forbid_final_test") is not True:
        raise SystemExit("policy_must_forbid_final_test")
    if args.variant == "e1_late_s4":
        raise SystemExit("e1_late_s4_is_inference_only_control")

    out_dir = Path(args.out_dir)
    trial_bindings: dict[str, Any] | None = None
    require_prov = (
        bool(policy.get("require_canonical_provenance"))
        or args.variant in _E4R2_VARIANTS
        or args.variant in _E4R3_VARIANTS
    )
    if require_prov:
        refuse_existing_outputs(out_dir)
        policy_path = Path(args.policy)
        # Optional sealed code-lock (written after research push).
        code_lock_rel = policy.get("code_lock_rel")
        sealed = None
        if code_lock_rel:
            sealed_path = _ROOT / code_lock_rel
            if not sealed_path.is_file():
                raise SystemExit(f"code_lock_missing:{code_lock_rel}")
            sealed = load_json(sealed_path)
        distill_w_early = (
            float(args.distill_weight)
            if args.distill_weight is not None
            else float((policy.get("distill") or {}).get("weight", 0.0))
        )
        trial_bindings = build_trial_bindings(
            root=_ROOT,
            policy_path=policy_path,
            policy=policy,
            variant=args.variant,
            seed=int(args.seed),
            lr=float(args.lr),
            dropout=float(args.dropout),
            weight_decay=float(args.weight_decay),
            distill_weight=distill_w_early,
            phase=args.phase,
        )
        if sealed is not None:
            verify_sealed_code_lock(
                root=_ROOT,
                sealed=sealed,
                live=trial_bindings,
                policy_sha=trial_bindings["policy_sha256"],
            )
            trial_bindings["code_lock_rel"] = str(code_lock_rel).replace("\\", "/")
            trial_bindings["code_lock_sha256"] = sha256_file(sealed_path)
            trial_bindings["implementation_git_revision"] = sealed.get(
                "implementation_git_revision"
            ) or sealed.get("git_revision")
        write_provenance(out_dir, trial_bindings)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))

    split = load_json(_ROOT / policy["split_manifest_rel"])
    observed = __import__("hashlib").sha256((_ROOT / policy["split_manifest_rel"]).read_bytes()).hexdigest()
    if observed != policy["split_manifest_sha256"]:
        raise SystemExit("split_manifest_sha_mismatch")

    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    gloss_to_id = labels_obj["gloss_to_id"]
    # Map sample_id -> class via metadata gloss.
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels: dict[str, int] = {}
    for sid, row in meta.items():
        g = row.get("gloss")
        if g in gloss_to_id:
            labels[sid] = int(gloss_to_id[g])

    if args.phase == "hpo":
        train_role = policy["hpo"]["train_role"]
        val_role = policy["hpo"]["val_role"]
        max_epochs = int(policy["hpo"]["max_epochs"])
        early = bool(policy["hpo"]["early_stopping"])
        patience = int(policy["hpo"]["patience"])
        batch_size = int(policy["hpo"]["batch_size"])
    else:
        train_role = policy["confirm"]["train_role"]
        val_role = policy["confirm"]["val_role"]
        max_epochs = int(policy["confirm"]["epochs"])
        early = False
        patience = 10**9
        batch_size = int(policy["confirm"]["batch_size"])

    train_ids = load_split_role_ids(split, train_role)
    val_ids = load_split_role_ids(split, val_role)
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    if set(train_ids) & test_ids or set(val_ids) & test_ids:
        raise SystemExit("test_leakage_detected")

    rgb_root = _ROOT / policy["feature_roots"]["rgb_root_rel"]
    skel_root = _ROOT / policy["feature_roots"]["skeleton_root_rel"]
    train_ds = VSL400MSSTAv5Dataset(
        train_ids, rgb_root=rgb_root, skel_root=skel_root, labels=labels, metadata_by_id=meta, forbid_ids=test_ids
    )
    val_ds = VSL400MSSTAv5Dataset(
        val_ids, rgb_root=rgb_root, skel_root=skel_root, labels=labels, metadata_by_id=meta, forbid_ids=test_ids
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_v5
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_v5
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    num_classes = int(labels_obj.get("num_classes") or 400)
    if args.variant in _E2_VARIANTS:
        model = build_ms_sta_v5_e2(
            args.variant,  # type: ignore[arg-type]
            num_classes=num_classes,
            dropout=float(args.dropout),
            alpha_max=float(policy["model"]["alpha_max"]),
        ).to(device)
    elif args.variant in _E3_VARIANTS:
        model = build_ms_sta_v6_e3(
            args.variant,
            num_classes=num_classes,
            dropout=float(args.dropout),
            alpha_max=float(policy["model"]["alpha_max"]),
        ).to(device)
    elif args.variant in _E4_VARIANTS:
        model = build_ms_sta_v6_e4(
            args.variant,
            num_classes=num_classes,
            dropout=float(args.dropout),
            alpha_max=float(policy["model"]["alpha_max"]),
        ).to(device)
    elif args.variant in _E4R2_VARIANTS:
        model = build_ms_sta_v6_e4_r2(
            _e4r2_arm(args.variant),
            num_classes=num_classes,
            d_model=int(policy["model"].get("d_model", 256)),
            nhead=int(policy["model"].get("nhead", 4)),
            temporal_layers=int(policy["model"].get("temporal_layers", 2)),
            dropout=float(args.dropout),
            alpha_max=float(policy["model"]["alpha_max"]),
            rgb_rank=int(policy["model"].get("rgb_rank", 384)),
            skel_keep_t=int(policy["model"].get("skel_keep_t", 16)),
            ctr_layers=int(policy["model"].get("ctr_layers", 2)),
        ).to(device)
    elif args.variant in _E4R3_VARIANTS:
        model = build_ms_sta_v6_e4_r3(
            _e4r3_arm(args.variant),
            num_classes=num_classes,
            d_model=int(policy["model"].get("d_model", 256)),
            nhead=int(policy["model"].get("nhead", 4)),
            temporal_layers=int(policy["model"].get("temporal_layers", 2)),
            dropout=float(args.dropout),
            alpha_max=float(policy["model"]["alpha_max"]),
            rgb_rank=int(policy["model"].get("rgb_rank", 384)),
            ctr_layers=int(policy["model"].get("ctr_layers", 2)),
        ).to(device)
    elif args.variant in _V6_VARIANTS:
        model = build_ms_sta_v6(
            args.variant,
            num_classes=num_classes,
            d_model=int(policy["model"]["d_model"]),
            nhead=int(policy["model"]["nhead"]),
            temporal_layers=int(policy["model"]["temporal_layers"]),
            dropout=float(args.dropout),
            alpha_max=float(policy["model"]["alpha_max"]),
            gate_entropy_reg=float(policy["model"].get("gate_entropy_reg", 0.0)),
        ).to(device)
    else:
        model = build_ms_sta_v5(
            args.variant,
            num_classes=num_classes,
            d_model=int(policy["model"]["d_model"]),
            nhead=int(policy["model"]["nhead"]),
            temporal_layers=int(policy["model"]["temporal_layers"]),
            dropout=float(args.dropout),
            alpha_max=float(policy["model"]["alpha_max"]),
        ).to(device)
    aux_w = (
        float(policy["confirm"].get("aux_ce_weight_b4", 0.2))
        if args.variant in {"b4_star_fixed", "b4_star_e_stride4"} or args.variant.startswith("b4_v6")
        else float(policy.get("distill", {}).get("aux_ce_weight", 0.0))
    )
    distill_cfg = policy.get("distill") or {}
    distill_w = (
        float(args.distill_weight)
        if args.distill_weight is not None
        else float(distill_cfg.get("weight", 0.0))
    )
    distill_temp = float(distill_cfg.get("temperature", 2.0))
    feature_w = float(distill_cfg.get("feature_weight", 0.0))
    feature_u_w = float(distill_cfg.get("feature_u_weight", feature_w))
    teacher = None
    feat_adapters: torch.nn.ModuleDict | None = None
    if distill_w > 0 or feature_w > 0 or feature_u_w > 0:
        if (
            args.variant not in _E2_VARIANTS
            and args.variant not in _E3_VARIANTS
            and args.variant not in _E4_VARIANTS
            and args.variant not in _E4R2_VARIANTS
            and args.variant not in _E4R3_VARIANTS
        ):
            raise SystemExit("distill_only_supported_for_e2_or_e3_or_e4_or_e4r2_or_e4r3_variants")
        t_rel = distill_cfg.get("teacher_checkpoint_rel") or policy.get("teacher_checkpoint_rel")
        t_abs = distill_cfg.get("teacher_checkpoint_abs") or policy.get("teacher_checkpoint_abs")
        t_sha = distill_cfg.get("teacher_checkpoint_sha256") or policy.get("teacher_checkpoint_sha256")
        if (not t_rel and not t_abs) or not t_sha:
            raise SystemExit("teacher_checkpoint_missing_in_policy")
        t_path = Path(t_abs) if t_abs else (_ROOT / t_rel)
        observed = __import__("hashlib").sha256(t_path.read_bytes()).hexdigest()
        if observed != t_sha:
            raise SystemExit("teacher_checkpoint_sha_mismatch")
        t_variant = str(distill_cfg.get("teacher_variant") or "b4_star_fixed")
        if t_variant in _V6_VARIANTS:
            teacher = build_ms_sta_v6(
                t_variant,
                num_classes=num_classes,
                d_model=int(policy["model"]["d_model"]),
                nhead=int(policy["model"]["nhead"]),
                temporal_layers=int(policy["model"]["temporal_layers"]),
                dropout=float(distill_cfg.get("teacher_dropout", 0.1)),
                alpha_max=float(policy["model"]["alpha_max"]),
                gate_entropy_reg=0.0,
            ).to(device)
        else:
            teacher = build_ms_sta_v5(
                t_variant,
                num_classes=num_classes,
                d_model=int(policy["model"]["d_model"]),
                nhead=int(policy["model"]["nhead"]),
                temporal_layers=int(policy["model"]["temporal_layers"]),
                dropout=float(distill_cfg.get("teacher_dropout", 0.1)),
                alpha_max=float(policy["model"]["alpha_max"]),
            ).to(device)
        ckpt = torch.load(t_path, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Train-only adapters when dims differ; never written into student checkpoint.
        s_dim = int(getattr(model, "d_model", policy["model"]["d_model"]))
        t_dim = int(getattr(teacher, "d_model", policy["model"]["d_model"]))
        if s_dim != t_dim and (feature_w > 0 or feature_u_w > 0):
            feat_adapters = torch.nn.ModuleDict(
                {
                    "h_base": torch.nn.Linear(s_dim, t_dim, bias=False),
                    "u": torch.nn.Linear(s_dim, t_dim, bias=False),
                }
            ).to(device)

    train_params = list(model.parameters())
    if feat_adapters is not None:
        train_params += list(feat_adapters.parameters())
    opt = torch.optim.AdamW(train_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: list[dict[str, Any]] = []
    best_top1 = -1.0
    best_path = out_dir / "best.pt"
    bad = 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        if trial_bindings is not None:
            assert_no_drift(trial_bindings, root=_ROOT)
        model.train()
        if feat_adapters is not None:
            feat_adapters.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            labels_t = batch["label"].to(device)
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(x)
            loss = F.cross_entropy(out["logits"], labels_t)
            if aux_w > 0 and "logits_base" in out:
                loss = loss + aux_w * F.cross_entropy(out["logits_base"], labels_t)
            if "aux_reg" in out:
                loss = loss + out["aux_reg"]
            if teacher is not None and (distill_w > 0 or feature_w > 0 or feature_u_w > 0):
                with torch.no_grad():
                    t_out = teacher(x)
                if distill_w > 0:
                    loss = loss + distill_w * _kl_logits(out["logits"], t_out["logits"], distill_temp)
                if feature_w > 0:
                    sh = out["h_base"]
                    th = t_out["h_base"].detach()
                    if feat_adapters is not None:
                        sh = feat_adapters["h_base"](sh)
                    if sh.shape != th.shape:
                        raise RuntimeError("feature_distill_h_base_dim_mismatch")
                    loss = loss + feature_w * F.mse_loss(sh, th)
                if feature_u_w > 0 and "u" in out and "u" in t_out:
                    su = out["u"]
                    tu = t_out["u"].detach()
                    if feat_adapters is not None:
                        su = feat_adapters["u"](su)
                    if su.shape != tu.shape:
                        raise RuntimeError("feature_distill_u_dim_mismatch")
                    loss = loss + feature_u_w * F.mse_loss(su, tu)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_params = train_params
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        val_metrics = evaluate(model, val_loader, device, collect_rows=False)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_n": val_metrics["n"],
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_top5 = float(val_metrics["top5"])
            best_macro_f1 = float(val_metrics["macro_f1"])
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": args.variant,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_top1": best_top1,
                    "val_top5": best_top5,
                    "val_macro_f1": best_macro_f1,
                    "lr": args.lr,
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "distill_weight": distill_w,
                    "phase": args.phase,
                    "bindings": trial_bindings,
                    "canonical": bool(trial_bindings is not None),
                },
                best_path,
            )
        else:
            bad += 1
            if early and bad >= patience:
                break

    # Reload best checkpoint and export per-sample validation rows (confirm/HPO audit).
    best_top5 = -1.0
    best_macro_f1 = -1.0
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_top5 = float(ckpt.get("val_top5", -1.0))
        best_macro_f1 = float(ckpt.get("val_macro_f1", -1.0))
    final_eval = evaluate(model, val_loader, device, collect_rows=True)
    preds_path = out_dir / "VAL_PREDS.json"
    preds_path.write_text(
        json.dumps(
            {
                "schema": "ms_sta_v5_m0_val_preds_v1",
                "variant": args.variant,
                "phase": args.phase,
                "seed": args.seed,
                "n": int(final_eval["n"]),
                "top1": final_eval["top1"],
                "top5": final_eval["top5"],
                "macro_f1": final_eval["macro_f1"],
                "rows": final_eval["rows"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": "ms_sta_v5_m0_trial_summary_v1",
        "variant": args.variant,
        "phase": args.phase,
        "seed": args.seed,
        "lr": args.lr,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "distill_weight": distill_w,
        "best_val_top1": best_top1,
        "best_val_top5": best_top5 if best_top5 >= 0 else final_eval["top5"],
        "best_val_macro_f1": best_macro_f1 if best_macro_f1 >= 0 else final_eval["macro_f1"],
        "epochs_run": len(history),
        "device": str(device),
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "elapsed_sec": time.time() - started,
        "best_checkpoint": str(best_path.relative_to(_ROOT)).replace("\\", "/") if best_path.is_file() else None,
        "val_preds": str(preds_path.relative_to(_ROOT)).replace("\\", "/"),
        "history": history,
        "canonical": bool(trial_bindings is not None),
        "bindings": trial_bindings,
        "git_revision": None if trial_bindings is None else trial_bindings.get("git_revision"),
        "git_dirty": None if trial_bindings is None else trial_bindings.get("git_dirty"),
        "executable_tree_sha256": None if trial_bindings is None else trial_bindings.get("executable_tree_sha256"),
        "policy_sha256": None if trial_bindings is None else trial_bindings.get("policy_sha256"),
        "teacher_checkpoint_sha256": None if trial_bindings is None else trial_bindings.get("teacher_checkpoint_sha256"),
        "split_manifest_sha256": None if trial_bindings is None else trial_bindings.get("split_manifest_sha256"),
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
