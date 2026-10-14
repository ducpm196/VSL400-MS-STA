"""Gate E1 — train MS-STA variants on VSL400 (validation-only selection).

RGB temporal contract: `isolated_sf_temporal` (L,2304) fixed to FIXED_RGB_L.
B0 may fall back to pooled `isolated_sf2304` only when variant=b0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.load_env import load_project_env
from src.models.isolated_sign_classifier import DEFAULT_CKPT
from src.models.feature_versioning import (
    git_dirty,
    git_rev,
    load_eligible_sample_manifest,
    load_json,
    sha256_file,
)
from src.models.ms_sta_features import (
    FIXED_RGB_L,
    MIN_TEMPORAL_L,
    load_rgb_array,
    resolve_pooled_dir,
    resolve_skel_dir,
    resolve_temporal_dir,
    to_rgb_tensor,
    valid_mask_to_key_padding_mask,
)
from src.models.ms_sta_wordlevel import Variant, build_ms_sta
from src.models.utils import enforce_reproducibility_seed
from src.training.e1_policy_enforce import (
    apply_initialization,
    check_collapse_guard,
    enforce_cuda_device,
    enforce_data_and_runtime_locks,
    enforce_hyperparams,
    refuse_free_ablation_flags,
    refuse_output_collision,
    require_frozen_policy,
    resolve_variant_control,
    resolve_warmup,
    set_head_only_trainable,
)
from src.training.retrain_v3 import (
    final_stem_from_candidate_stem,
    model_variant_name,
    promote_candidate_checkpoint,
    refuse_gate3_init,
    require_frozen_v3_policy,
)
from src.training.retrain_v4 import (
    build_optimizer_and_scheduler,
    checkpoint_hyperparameters,
    lineage_block,
    refuse_train_until_frozen,
    require_frozen_v4_policy,
    residual_init_for_variant,
    should_early_stop,
)

OUT_DIR = _ROOT / "data" / "processed" / "ms_sta_e1_v2"
STATUS_DIR = _ROOT / "logs" / "gate_e1_v2"
E1_POLICY_PATH = _ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY.json"
REGISTRY_PATH = _ROOT / "data" / "processed" / "ms_sta_registry.json"
DEFAULT_SPLIT_MANIFEST = _ROOT / "data" / "processed" / "vsl400_e1_split_manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class VSL400MSSTADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        gloss_to_id: dict[str, int],
        *,
        require_skel: bool,
        require_temporal_rgb: bool,
        rgb_pooled_dir: Path,
        rgb_temporal_dir: Path,
        skel_dir: Path,
        target_skel_t: int = 64,
        fixed_rgb_l: int = FIXED_RGB_L,
        video_ids: list[str] | None = None,
    ) -> None:
        self.gloss_to_id = gloss_to_id
        self.require_skel = require_skel
        self.require_temporal_rgb = require_temporal_rgb
        self.rgb_pooled_dir = rgb_pooled_dir
        self.rgb_temporal_dir = rgb_temporal_dir
        self.skel_dir = skel_dir
        self.target_skel_t = target_skel_t
        self.fixed_rgb_l = fixed_rgb_l
        allowed = set(video_ids) if video_ids is not None else None
        rows = []
        for _, r in df.iterrows():
            vid = str(r["video_id"])
            if allowed is not None and vid not in allowed:
                continue
            gloss = str(r["text_gloss"])
            if gloss not in gloss_to_id:
                continue
            temporal_p = self.rgb_temporal_dir / f"{vid}.npy"
            pooled_p = self.rgb_pooled_dir / f"{vid}.npy"
            skel_p = self.skel_dir / f"{vid}.npy"
            if require_temporal_rgb:
                if not temporal_p.is_file():
                    continue
            else:
                if not pooled_p.is_file():
                    continue
            if require_skel and not skel_p.is_file():
                continue
            rows.append((vid, gloss_to_id[gloss], skel_p.is_file(), temporal_p.is_file()))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        vid, y, has_skel, _has_temporal = self.rows[idx]
        if self.require_temporal_rgb:
            rgb = load_rgb_array(self.rgb_temporal_dir / f"{vid}.npy")
            if rgb.shape[0] < MIN_TEMPORAL_L:
                raise RuntimeError(f"{vid} temporal L={rgb.shape[0]} < {MIN_TEMPORAL_L}")
            rgb_t, valid = to_rgb_tensor(rgb, fixed_l=self.fixed_rgb_l)
            if rgb_t.shape[0] != self.fixed_rgb_l:
                raise RuntimeError(f"{vid} rgb L={rgb_t.shape[0]} != {self.fixed_rgb_l}")
        else:
            rgb = load_rgb_array(self.rgb_pooled_dir / f"{vid}.npy")
            rgb_t, valid = to_rgb_tensor(rgb, fixed_l=None)
        skel = None
        if has_skel:
            skel = np.load(self.skel_dir / f"{vid}.npy").astype(np.float32)
            if skel.shape[0] != self.target_skel_t:
                ix = np.linspace(0, skel.shape[0] - 1, self.target_skel_t).astype(int)
                skel = skel[ix]
        return {
            "video_id": vid,
            "rgb": rgb_t,
            "rgb_valid": valid,
            "skel": None if skel is None else torch.from_numpy(skel),
            "y": torch.tensor(y, dtype=torch.long),
        }


def _collate(batch):
    rgb = torch.stack([b["rgb"] for b in batch], dim=0)
    ys = torch.stack([b["y"] for b in batch], dim=0)
    vids = [b["video_id"] for b in batch]
    if all(b["rgb_valid"] is not None for b in batch):
        rgb_valid = torch.stack([b["rgb_valid"] for b in batch], dim=0)
        rgb_kpm = valid_mask_to_key_padding_mask(rgb_valid)
    else:
        rgb_kpm = None
    if all(b["skel"] is not None for b in batch):
        skel = torch.stack([b["skel"] for b in batch], dim=0)
    else:
        skel = None
    return rgb, skel, ys, vids, rgb_kpm


def _sample_weighted_mean(weighted_sum: float, n: int) -> float:
    """epoch_loss = Σ(batch_loss × batch_size) / Σ(batch_size)."""
    return float(weighted_sum) / float(max(int(n), 1))


def _finite_or_null(value: float | None, *, field: str, missing: dict[str, str]) -> float | None:
    """Never emit NaN/Inf; record a reason when a field cannot be reported."""
    if value is None:
        missing[field] = "not_collected"
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        missing[field] = "non_numeric"
        return None
    if not math.isfinite(v):
        missing[field] = "non_finite"
        return None
    return v


@torch.no_grad()
def _eval_top(model, loader, device, variant: str) -> dict:
    model.eval()
    top1 = top5 = n = 0
    weighted_loss = 0.0
    crit = torch.nn.CrossEntropyLoss()
    for rgb, skel, y, _, rgb_kpm in loader:
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
        bs = int(y.numel())
        # Sample-weighted: mean CE × batch_size, then divide by Σ batch_size.
        weighted_loss += float(crit(logits, y).item()) * bs
        pred = logits.argmax(dim=-1)
        top1 += int((pred == y).sum().item())
        topk = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
        top5 += int((topk == y.unsqueeze(1)).any(dim=1).sum().item())
        n += bs
    return {
        "n": n,
        "top1": 100.0 * top1 / max(n, 1),
        "top5": 100.0 * top5 / max(n, 1),
        "loss": _sample_weighted_mean(weighted_loss, n),
        "loss_aggregation": "sample_weighted_mean_ce",
    }


def _grad_norm(params) -> float | None:
    total = 0.0
    found = False
    for p in params:
        if p is None or p.grad is None:
            continue
        total += float(p.grad.detach().norm().item() ** 2)
        found = True
    if not found:
        return None
    return float(total**0.5)


def _root_sum_square(values: list[float | None]) -> float | None:
    parts = [float(v) for v in values if v is not None]
    if not parts:
        return None
    return float(sum(v * v for v in parts) ** 0.5)


def _extend_per_sample(dst: list[float], tensor) -> None:
    """Append per-sample scalars from a 1-D or scalar tensor (never batch means)."""
    if tensor is None:
        return
    t = tensor.detach().reshape(-1)
    for i in range(int(t.numel())):
        dst.append(float(t[i].item()))


def _b4r_forward_diagnostics(
    model,
    loader,
    device,
    *,
    max_batches: int = 8,
    diagnostic_split: str = "val",
    loader_shuffle: bool = False,
) -> dict:
    """Deterministic eval-time residual/gate diagnostics (no grad).

    Gate mean/std and logit-norm stats are aggregated over **samples**, not batch means.
    """
    model.eval()
    base_ns: list[float] = []
    delta_ns: list[float] = []
    ratios: list[float] = []
    gate_rgb: list[float] = []
    gate_fused: list[float] = []
    gate_ent: list[float] = []
    scale_vals: list[float] = []
    n_samples = 0
    batches_seen = 0
    with torch.no_grad():
        for bi, (rgb, skel, y, _, rgb_kpm) in enumerate(loader):
            if bi >= int(max_batches):
                break
            batches_seen += 1
            rgb = rgb.to(device)
            skel_t = skel.to(device) if skel is not None else None
            kpm = rgb_kpm.to(device) if rgb_kpm is not None else None
            _logits, attn = model(
                rgb=rgb, skel=skel_t, rgb_key_padding_mask=kpm, return_attention=True
            )
            bs = int(y.shape[0]) if hasattr(y, "shape") else int(rgb.shape[0])
            n_samples += bs
            base_ps = attn.get("base_logits_norm_per_sample")
            if base_ps is not None:
                _extend_per_sample(base_ns, base_ps)
            elif "base_logits_norm" in attn:
                _extend_per_sample(base_ns, attn["base_logits_norm"])
            delta_ps = attn.get("delta_logits_norm_per_sample")
            if delta_ps is not None:
                _extend_per_sample(delta_ns, delta_ps)
            elif "delta_logits_norm" in attn:
                _extend_per_sample(delta_ns, attn["delta_logits_norm"])
            ratio_ps = attn.get("delta_over_base_ratio_per_sample")
            if ratio_ps is not None:
                _extend_per_sample(ratios, ratio_ps)
            elif "delta_over_base_ratio" in attn:
                _extend_per_sample(ratios, attn["delta_over_base_ratio"])
            if "residual_scale" in attn:
                scale_vals.append(float(attn["residual_scale"].reshape(()).item()))
            gate_w = attn.get("stream_gate")
            if gate_w is not None:
                w = gate_w.detach()
                # Per-sample gate weights (not batch means).
                for i in range(int(w.shape[0])):
                    gate_rgb.append(float(w[i, 0].item()))
                    gate_fused.append(float(w[i, 1].item()))
                    wi = w[i].clamp_min(1e-8)
                    gate_ent.append(float((-(wi * wi.log()).sum()).item()))

    def _mean(xs: list[float]) -> float | None:
        return float(sum(xs) / len(xs)) if xs else None

    def _std(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return 0.0 if xs else None
        m = sum(xs) / len(xs)
        return float((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)

    missing: dict[str, str] = {}
    out = {
        "diagnostic_split": diagnostic_split,
        "diagnostic_max_batches": int(max_batches),
        "diagnostic_n_samples": int(n_samples),
        "diagnostic_batches_seen": int(batches_seen),
        "diagnostic_loader_shuffle": bool(loader_shuffle),
        "residual_scale": _finite_or_null(_mean(scale_vals), field="residual_scale", missing=missing),
        "base_logits_norm": _finite_or_null(_mean(base_ns), field="base_logits_norm", missing=missing),
        "delta_logits_norm": _finite_or_null(
            _mean(delta_ns), field="delta_logits_norm", missing=missing
        ),
        "delta_over_base_ratio": _finite_or_null(
            _mean(ratios), field="delta_over_base_ratio", missing=missing
        ),
        "stream_gate_rgb_mean": _finite_or_null(
            _mean(gate_rgb), field="stream_gate_rgb_mean", missing=missing
        ),
        "stream_gate_fused_mean": _finite_or_null(
            _mean(gate_fused), field="stream_gate_fused_mean", missing=missing
        ),
        "stream_gate_rgb_std": _finite_or_null(
            _std(gate_rgb), field="stream_gate_rgb_std", missing=missing
        ),
        "stream_gate_fused_std": _finite_or_null(
            _std(gate_fused), field="stream_gate_fused_std", missing=missing
        ),
        "stream_gate_entropy": _finite_or_null(
            _mean(gate_ent), field="stream_gate_entropy", missing=missing
        ),
        "aggregation": "per_sample_mean_std",
    }
    if missing:
        out["missing_fields"] = missing
    return out


def main() -> None:
    load_project_env()
    ap = argparse.ArgumentParser(description="Option B Gate E1 — MS-STA val-first training")
    ap.add_argument("--variant", default="b1", choices=["b0", "b1", "b2", "b3", "b4", "b4r", "b5"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
        help="Frozen E1 split manifest (train/val IDs). Required unless --allow-unfrozen-pilot.",
    )
    ap.add_argument(
        "--sample-manifest",
        type=Path,
        default=_ROOT / "data" / "processed" / "vsl400_eligible_sample_manifest.json",
        help="Eligible-ID manifest (excludes 019442). Intersected with train/val sets.",
    )
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run all epochs (v4 confirmation). Checkpoint still tracks best val Top-1.",
    )
    ap.add_argument(
        "--residual-init",
        default=None,
        choices=["zero_delta_head", "zero_scale", "none"],
        help="B4R residual init mode (required for variant=b4r under retrain-v4).",
    )
    ap.add_argument(
        "--warm-b0",
        action="store_true",
        help="Pilot only. Canonical E1 derives init from policy and refuses this flag.",
    )
    ap.add_argument(
        "--disable-spatial",
        action="store_true",
        help="Pilot only. Canonical E1 derives spatial flag from policy.",
    )
    ap.add_argument(
        "--disable-temporal",
        action="store_true",
        help="Pilot only. Canonical E1 derives temporal flag from policy.",
    )
    ap.add_argument(
        "--skel-temporal-stride",
        type=int,
        default=1,
        help="Pilot only. Average-pool Skeleton tokens before temporal attention; canonical E1 requires 1.",
    )
    ap.add_argument(
        "--pilot-init-mode",
        choices=[
            "gate3_rgb_proj_and_classifier",
            "gate3_rgb_proj_and_compatible_classifier",
            "gate3_rgb_proj_only",
            "random",
        ],
        help="Pilot only. Explicit initialization for isolated research extensions.",
    )
    ap.add_argument(
        "--allow-unfrozen-pilot",
        action="store_true",
        help="Pilot only: allow missing/unmatched E1 policy (never defense-ready).",
    )
    ap.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Pilot only with --allow-unfrozen-pilot: overwrite existing ckpt/status.",
    )
    ap.add_argument(
        "--e1-policy",
        type=Path,
        default=E1_POLICY_PATH,
        help="Frozen E1 policy JSON (Docs/results/ms_sta_e1_policy/E1_POLICY.json)",
    )
    ap.add_argument(
        "--retrain-v3-policy",
        type=Path,
        default=None,
        help="Frozen retrain-v3 policy (Path B). Enables fresh-init canonical mode.",
    )
    ap.add_argument(
        "--retrain-v4-policy",
        type=Path,
        default=None,
        help="Frozen retrain-v4 policy (Path B). Enables B4R residual + v4 contracts.",
    )
    ap.add_argument("--config-id", default=None, help="HPO/confirm config id (v3/v4)")
    ap.add_argument(
        "--phase",
        default=None,
        choices=["hpo_screen", "hpo_full", "hpo", "confirm", "lock", "test", "d4"],
        help="Retrain phase label for artifacts",
    )
    ap.add_argument(
        "--train-role",
        default=None,
        choices=["hpo_train", "confirm_train"],
        help="Split role for training IDs (v3)",
    )
    ap.add_argument(
        "--val-role",
        default=None,
        choices=["tune", "val"],
        help="Split role for selection IDs (v3); test role forbidden here",
    )
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument(
        "--experiment-variant",
        default=None,
        help="Logical variant id for artifacts (e.g. b4e); model still uses --variant",
    )
    ap.add_argument(
        "--artifact-stem",
        default=None,
        help="Override checkpoint/status filename stem (v3)",
    )
    args = ap.parse_args()
    allow_pilot = bool(args.allow_unfrozen_pilot)
    v3_mode = args.retrain_v3_policy is not None
    v4_mode = args.retrain_v4_policy is not None
    if v3_mode and v4_mode:
        raise SystemExit("[FAIL] refuse both --retrain-v3-policy and --retrain-v4-policy")
    if args.skel_temporal_stride < 1:
        raise SystemExit("[FAIL] --skel-temporal-stride must be >= 1")
    if not allow_pilot and args.pilot_init_mode:
        raise SystemExit("[FAIL] --pilot-init-mode requires --allow-unfrozen-pilot")
    if (v3_mode or v4_mode) and allow_pilot:
        raise SystemExit("[FAIL] retrain-v3/v4 refuses --allow-unfrozen-pilot")
    early_stopping = not bool(args.disable_early_stopping)
    if v4_mode and str(args.phase) == "confirm" and early_stopping:
        raise SystemExit("[FAIL] retrain-v4 confirmation requires --disable-early-stopping")
    if v4_mode and str(args.phase) == "hpo" and not early_stopping:
        raise SystemExit("[FAIL] retrain-v4 HPO requires early stopping enabled")
    residual_init = str(args.residual_init or "none")
    if args.variant == "b4r" and residual_init == "none" and not v4_mode:
        residual_init = "zero_delta_head"
    if args.variant == "b4r" and residual_init not in ("zero_delta_head", "zero_scale"):
        raise SystemExit("[FAIL] variant=b4r requires --residual-init zero_delta_head|zero_scale")
    if args.variant != "b4r" and args.residual_init not in (None, "none"):
        raise SystemExit("[FAIL] --residual-init is only valid for variant=b4r")
    if v3_mode and args.e1_policy != E1_POLICY_PATH and Path(args.e1_policy).name.lower().startswith("e1"):
        # Allow default e1 path to remain unset effectively when v3 policy provided.
        pass
    v4_policy = None
    v4_policy_sha = None
    skel_temporal_stride = int(args.skel_temporal_stride)
    if v4_mode:
        v4_path = (
            args.retrain_v4_policy
            if Path(args.retrain_v4_policy).is_absolute()
            else _ROOT / args.retrain_v4_policy
        )
        if not v4_path.is_file():
            raise SystemExit(f"[FAIL] missing retrain-v4 policy: {v4_path}")
        v4_policy = load_json(v4_path)
        v4_policy_sha = sha256_file(v4_path)
        try:
            require_frozen_v4_policy(v4_policy, root=_ROOT)
            refuse_free_ablation_flags(
                warm_b0=bool(args.warm_b0),
                disable_spatial=bool(args.disable_spatial),
                disable_temporal=bool(args.disable_temporal),
                allow_pilot=False,
            )
        except ValueError as exc:
            raise SystemExit(f"[FAIL] {exc}") from exc
        if not args.config_id or not args.phase or not args.train_role or not args.val_role:
            raise SystemExit(
                "[FAIL] retrain-v4 requires --config-id --phase --train-role --val-role"
            )
        if args.phase not in ("hpo", "confirm"):
            raise SystemExit(f"[FAIL] retrain-v4 trainer phase must be hpo|confirm, got {args.phase}")
        if args.val_role == "test" or args.train_role == "test":
            raise SystemExit("[FAIL] retrain-v4 trainer refuses test role access")
        space = v4_policy.get("hpo_search_space") or {}
        if args.weight_decay is None:
            raise SystemExit("[FAIL] retrain-v4 requires --weight-decay from frozen config")
        space_lrs = [float(x) for x in (space.get("learning_rate") or [])]
        space_wds = [float(x) for x in (space.get("weight_decay") or [])]
        if not any(abs(float(args.weight_decay) - w) < 1e-15 for w in space_wds):
            raise SystemExit(f"[FAIL] weight_decay={args.weight_decay} outside frozen search space")
        if not any(abs(float(args.lr) - x) < 1e-15 for x in space_lrs):
            raise SystemExit(f"[FAIL] lr={args.lr} outside frozen search space")
        if int(args.batch_size) != int(space.get("batch_size", 32)):
            raise SystemExit("[FAIL] batch_size must match frozen search space")
        exp_clip = float(space.get("gradient_clipping", 1.0))
        grad_clip = float(args.grad_clip if args.grad_clip is not None else exp_clip)
        if abs(grad_clip - exp_clip) > 1e-12:
            raise SystemExit("[FAIL] grad_clip must match frozen search space")
        args.grad_clip = grad_clip
        code_lock = v4_policy.get("code_lock") or {}
        if code_lock.get("require_clean_worktree", False) and git_dirty(_ROOT):
            raise SystemExit("[FAIL] frozen retrain-v4 policy requires a clean worktree")
        exp_var = str(args.experiment_variant or args.variant)
        vctrl_all = v4_policy.get("variant_controls") or {}
        if exp_var not in vctrl_all:
            raise SystemExit(f"[FAIL] unknown experiment variant {exp_var}")
        expected_residual = residual_init_for_variant(str(args.variant), v4_policy)
        if str(args.variant) == "b4r":
            if residual_init != expected_residual and args.residual_init is None:
                residual_init = expected_residual
            if residual_init != expected_residual:
                raise SystemExit(
                    f"[FAIL] residual_init={residual_init!r} != policy {expected_residual!r}"
                )
        skel_temporal_stride = int(vctrl_all[exp_var].get("skel_temporal_stride", 1))
        if args.skel_temporal_stride not in (1, skel_temporal_stride):
            raise SystemExit(
                "[FAIL] retrain-v4 refuses free --skel-temporal-stride; use experiment variant controls"
            )
    elif v3_mode and args.e1_policy != E1_POLICY_PATH and Path(args.e1_policy).name.lower().startswith("e1"):
        pass

    policy = None
    policy_sha = None
    v3_policy = None
    v3_policy_sha = None
    if v4_mode:
        policy = v4_policy
        policy_sha = v4_policy_sha
    elif v3_mode:
        v3_path = (
            args.retrain_v3_policy
            if Path(args.retrain_v3_policy).is_absolute()
            else _ROOT / args.retrain_v3_policy
        )
        if not v3_path.is_file():
            raise SystemExit(f"[FAIL] missing retrain-v3 policy: {v3_path}")
        v3_policy = load_json(v3_path)
        v3_policy_sha = sha256_file(v3_path)
        policy = v3_policy
        policy_sha = v3_policy_sha
        try:
            require_frozen_v3_policy(v3_policy)
            refuse_free_ablation_flags(
                warm_b0=bool(args.warm_b0),
                disable_spatial=bool(args.disable_spatial),
                disable_temporal=bool(args.disable_temporal),
                allow_pilot=False,
            )
        except ValueError as exc:
            raise SystemExit(f"[FAIL] {exc}") from exc
        if not args.config_id or not args.phase or not args.train_role or not args.val_role:
            raise SystemExit(
                "[FAIL] retrain-v3 requires --config-id --phase --train-role --val-role"
            )
        if args.val_role == "test" or args.train_role == "test":
            raise SystemExit("[FAIL] retrain-v3 trainer refuses test role access")
        space = v3_policy.get("hpo_search_space") or {}
        if args.weight_decay is None:
            raise SystemExit("[FAIL] retrain-v3 requires --weight-decay from frozen config")
        space_lrs = [float(x) for x in (space.get("learning_rate") or [])]
        space_wds = [float(x) for x in (space.get("weight_decay") or [])]
        if not any(abs(float(args.weight_decay) - w) < 1e-15 for w in space_wds):
            raise SystemExit(f"[FAIL] weight_decay={args.weight_decay} outside frozen search space")
        if not any(abs(float(args.lr) - x) < 1e-15 for x in space_lrs):
            raise SystemExit(f"[FAIL] lr={args.lr} outside frozen search space")
        if int(args.batch_size) != int(space.get("batch_size", 32)):
            raise SystemExit("[FAIL] batch_size must match frozen search space")
        exp_clip = float(space.get("gradient_clipping", 1.0))
        grad_clip = float(args.grad_clip if args.grad_clip is not None else exp_clip)
        if abs(grad_clip - exp_clip) > 1e-12:
            raise SystemExit("[FAIL] grad_clip must match frozen search space")
        args.grad_clip = grad_clip
        code_lock = v3_policy.get("code_lock") or {}
        if code_lock.get("require_clean_worktree", False) and git_dirty(_ROOT):
            raise SystemExit("[FAIL] frozen retrain-v3 policy requires a clean worktree")
        # Stride: from experiment variant controls when provided, else CLI must be 1
        # unless policy variant override (b4e) is selected via --experiment-variant.
        exp_var = str(args.experiment_variant or args.variant)
        vctrl_all = v3_policy.get("variant_controls") or {}
        if exp_var not in vctrl_all:
            raise SystemExit(f"[FAIL] unknown experiment variant {exp_var}")
        skel_temporal_stride = int(vctrl_all[exp_var].get("skel_temporal_stride", 1))
        if args.skel_temporal_stride != 1 and args.skel_temporal_stride != skel_temporal_stride:
            raise SystemExit(
                "[FAIL] retrain-v3 refuses free --skel-temporal-stride; use experiment variant controls"
            )
        if args.skel_temporal_stride == skel_temporal_stride:
            pass
        elif args.skel_temporal_stride == 1:
            pass
        else:
            raise SystemExit("[FAIL] skel stride mismatch vs policy")
    elif args.e1_policy is not None and Path(args.e1_policy).is_file():
        policy_path = args.e1_policy if Path(args.e1_policy).is_absolute() else _ROOT / args.e1_policy
        policy = load_json(policy_path)
        policy_sha = sha256_file(policy_path)
        try:
            require_frozen_policy(policy, allow_pilot=allow_pilot)
            refuse_free_ablation_flags(
                warm_b0=bool(args.warm_b0),
                disable_spatial=bool(args.disable_spatial),
                disable_temporal=bool(args.disable_temporal),
                allow_pilot=allow_pilot,
            )
            enforce_hyperparams(
                policy,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
                allow_pilot=allow_pilot,
            )
        except ValueError as exc:
            raise SystemExit(f"[FAIL] {exc}") from exc
        if allow_pilot:
            skel_temporal_stride = int(args.skel_temporal_stride)
        else:
            if args.skel_temporal_stride != 1:
                raise SystemExit(
                    "[FAIL] canonical E1 refuses free --skel-temporal-stride; derive it from policy"
                )
            skel_temporal_stride = int(
                ((policy or {}).get("model_overrides") or {}).get("skel_temporal_stride", 1)
            )
            if skel_temporal_stride < 1:
                raise SystemExit("[FAIL] policy model_overrides.skel_temporal_stride must be >= 1")
            code_lock = (policy or {}).get("code_lock") or {}
            if code_lock.get("require_clean_worktree", False) and git_dirty(_ROOT):
                raise SystemExit("[FAIL] frozen experiment policy requires a clean worktree")
            lineage = (policy or {}).get("lineage") or {}
            base_policy_rel = lineage.get("base_e1_policy")
            base_policy_sha = lineage.get("base_e1_policy_sha256")
            if base_policy_rel:
                base_policy_path = _ROOT / str(base_policy_rel)
                if not base_policy_path.is_file():
                    raise SystemExit(f"[FAIL] base E1 policy missing: {base_policy_path}")
                actual_base_sha = sha256_file(base_policy_path)
                if base_policy_sha and actual_base_sha != base_policy_sha:
                    raise SystemExit(
                        f"[FAIL] base E1 policy sha mismatch: {actual_base_sha} != {base_policy_sha}"
                    )
    elif not allow_pilot:
        raise SystemExit(f"[FAIL] missing frozen E1 policy: {args.e1_policy}")
    else:
        skel_temporal_stride = int(args.skel_temporal_stride)

    enforce_reproducibility_seed(args.seed)
    try:
        device_name = enforce_cuda_device(
            policy or {},
            requested_device=args.device,
            cuda_available=torch.cuda.is_available(),
            allow_pilot=allow_pilot,
        )
    except ValueError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
    device = torch.device(device_name)
    experiment_variant = str(args.experiment_variant or args.variant)
    model_var = model_variant_name(experiment_variant)
    if model_var != args.variant and (v3_mode or v4_mode):
        raise SystemExit(
            f"[FAIL] --variant={args.variant} must match model for experiment {experiment_variant} ({model_var})"
        )
    variant: Variant = args.variant  # type: ignore[assignment]

    if v4_mode:
        vctrl = (v4_policy or {}).get("variant_controls", {}).get(experiment_variant) or {}
        disable_spatial = bool(vctrl.get("disable_spatial", variant == "b5"))
        disable_temporal = bool(vctrl.get("disable_temporal", False))
        require_skel = bool(
            vctrl.get("require_skel", variant in ("b2", "b3", "b4", "b4r", "b5"))
        )
        require_temporal = bool(
            vctrl.get(
                "require_temporal_rgb",
                variant in ("b1", "b3", "b4", "b4r", "b5"),
            )
        )
        init_mode = str(vctrl.get("init") or "random")
        try:
            refuse_gate3_init(init_mode)
        except ValueError as exc:
            raise SystemExit(f"[FAIL] {exc}") from exc
        skel_temporal_stride = int(vctrl.get("skel_temporal_stride", skel_temporal_stride))
        weight_decay = float(args.weight_decay)
        label_smoothing = float(
            ((v4_policy or {}).get("hpo_search_space") or {}).get("label_smoothing", 0.05)
        )
        use_lr_schedule = True
        residual_init = residual_init_for_variant(str(variant), v4_policy) if variant == "b4r" else "none"
    elif v3_mode:
        vctrl = (v3_policy or {}).get("variant_controls", {}).get(experiment_variant) or {}
        disable_spatial = bool(vctrl.get("disable_spatial", variant == "b5"))
        disable_temporal = bool(vctrl.get("disable_temporal", False))
        require_skel = bool(vctrl.get("require_skel", variant in ("b2", "b3", "b4", "b5")))
        require_temporal = bool(
            vctrl.get("require_temporal_rgb", variant in ("b1", "b3", "b4", "b5"))
        )
        init_mode = str(vctrl.get("init") or "random")
        try:
            refuse_gate3_init(init_mode)
        except ValueError as exc:
            raise SystemExit(f"[FAIL] {exc}") from exc
        skel_temporal_stride = int(vctrl.get("skel_temporal_stride", skel_temporal_stride))
        weight_decay = float(args.weight_decay)
        label_smoothing = float(((v3_policy or {}).get("hpo_search_space") or {}).get("label_smoothing", 0.05))
        use_lr_schedule = True
    elif policy is not None and not allow_pilot:
        try:
            vctrl = resolve_variant_control(policy, variant)
        except ValueError as exc:
            raise SystemExit(f"[FAIL] {exc}") from exc
        disable_spatial = bool(vctrl["disable_spatial"])
        disable_temporal = bool(vctrl["disable_temporal"])
        require_skel = bool(vctrl["require_skel"])
        require_temporal = bool(vctrl["require_temporal_rgb"])
        init_mode = str(vctrl["init"])
        weight_decay = 0.05
        label_smoothing = 0.05
        use_lr_schedule = False
    else:
        disable_spatial = bool(args.disable_spatial or variant == "b5")
        disable_temporal = bool(args.disable_temporal)
        require_skel = variant in ("b2", "b3", "b4", "b5")
        require_temporal = variant in ("b1", "b3", "b4", "b5")
        if args.pilot_init_mode:
            init_mode = str(args.pilot_init_mode)
        elif args.warm_b0 or variant == "b0":
            init_mode = "gate3_rgb_proj_and_classifier" if variant == "b0" else "gate3_rgb_proj_only"
        elif variant == "b2":
            init_mode = "random"
        else:
            init_mode = "gate3_rgb_proj_only"
        weight_decay = float(args.weight_decay) if args.weight_decay is not None else 0.05
        label_smoothing = 0.05
        use_lr_schedule = False
    if skel_temporal_stride > 1 and not require_skel:
        raise SystemExit(
            f"[FAIL] Skeleton temporal stride {skel_temporal_stride} requires a Skeleton variant"
        )

    labels_path = _ROOT / "data" / "processed" / "vsl400_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    gloss_to_id = labels["gloss_to_id"]
    num_classes = int(labels["num_classes"])

    train_ids = val_ids = None
    manifest_hash = None
    split_manifest_path: Path | None = None
    role_key_train = {
        "hpo_train": "hpo_train_ids",
        "confirm_train": "confirm_train_ids",
    }
    role_key_val = {"tune": "tune_ids", "val": "val_ids"}
    if v4_mode:
        man_rel = lineage_block(v4_policy or {}).get("split_manifest")
        split_manifest_path = _ROOT / str(
            man_rel or "data/processed/vsl400_retrain_v4_split_manifest.json"
        )
        default_e1 = DEFAULT_SPLIT_MANIFEST.resolve()
        explicit = args.manifest.resolve() if args.manifest is not None else None
        if explicit is not None and explicit != default_e1:
            split_manifest_path = explicit
        if not split_manifest_path.is_file():
            raise SystemExit(f"[FAIL] --manifest not found: {split_manifest_path}")
        man = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if man.get("path") != "B":
            raise SystemExit("[FAIL] retrain-v4 split manifest path must be B")
        if "test" in (args.train_role, args.val_role):
            raise SystemExit("[FAIL] trainer refuses test role")
        tk = role_key_train[str(args.train_role)]
        vk = role_key_val[str(args.val_role)]
        if tk not in man or vk not in man:
            raise SystemExit(f"[FAIL] split manifest missing roles {tk}/{vk}")
        train_ids = [str(x) for x in man[tk]]
        val_ids = [str(x) for x in man[vk]]
        test_ids = set(str(x) for x in man.get("test_ids") or [])
        if test_ids & set(train_ids) or test_ids & set(val_ids):
            raise SystemExit("[FAIL] test IDs leaked into train/val roles")
        manifest_hash = _sha256_file(split_manifest_path)
        expected = lineage_block(v4_policy or {}).get("split_manifest_sha256")
        if expected and expected != manifest_hash:
            raise SystemExit(
                f"[FAIL] v4 split manifest sha mismatch: {manifest_hash} != {expected}"
            )
    elif v3_mode:
        man_rel = ((v3_policy or {}).get("lineage") or {}).get("split_manifest")
        split_manifest_path = _ROOT / str(
            man_rel or "data/processed/vsl400_retrain_v3_split_manifest.json"
        )
        # Do not let the E1 argparse default (--manifest=vsl400_e1_split_manifest) override Path B.
        default_e1 = DEFAULT_SPLIT_MANIFEST.resolve()
        explicit = args.manifest.resolve() if args.manifest is not None else None
        if explicit is not None and explicit != default_e1:
            split_manifest_path = explicit
        if not split_manifest_path.is_file():
            raise SystemExit(f"[FAIL] --manifest not found: {split_manifest_path}")
        man = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if man.get("path") != "B":
            raise SystemExit("[FAIL] retrain-v3 split manifest path must be B")
        if "test" in (args.train_role, args.val_role):
            raise SystemExit("[FAIL] trainer refuses test role")
        tk = role_key_train[str(args.train_role)]
        vk = role_key_val[str(args.val_role)]
        if tk not in man or vk not in man:
            raise SystemExit(f"[FAIL] split manifest missing roles {tk}/{vk}")
        train_ids = [str(x) for x in man[tk]]
        val_ids = [str(x) for x in man[vk]]
        # Fail-closed: never silently include test IDs in train/val roles.
        test_ids = set(str(x) for x in man.get("test_ids") or [])
        if test_ids & set(train_ids) or test_ids & set(val_ids):
            raise SystemExit("[FAIL] test IDs leaked into train/val roles")
        manifest_hash = _sha256_file(split_manifest_path)
        expected = ((v3_policy or {}).get("lineage") or {}).get("split_manifest_sha256")
        if expected and expected != manifest_hash:
            raise SystemExit(
                f"[FAIL] v3 split manifest sha mismatch: {manifest_hash} != {expected}"
            )
    elif args.manifest is not None:
        split_manifest_path = (
            args.manifest if Path(args.manifest).is_absolute() else _ROOT / args.manifest
        )
        if not split_manifest_path.is_file():
            raise SystemExit(f"[FAIL] --manifest not found: {split_manifest_path}")
        man = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        train_ids = [str(x) for x in man["train_ids"]]
        val_ids = [str(x) for x in man["val_ids"]]
        manifest_hash = _sha256_file(split_manifest_path)
        if policy is not None and not allow_pilot:
            expected = (policy.get("data_locks") or {}).get("e1_split_manifest_sha256")
            if expected and expected != manifest_hash:
                raise SystemExit(
                    f"[FAIL] split manifest sha mismatch vs E1 policy: "
                    f"{manifest_hash} != {expected}"
                )
    else:
        if not allow_pilot:
            raise SystemExit("[FAIL] --manifest required for frozen E1 (or pass --allow-unfrozen-pilot)")
        print(
            "[WARN] No --manifest: sample sets are NOT frozen (not fair ablation). "
            "Pass a frozen manifest before defense-ready E1."
        )

    eligible = None
    sample_manifest_sha = None
    if args.sample_manifest is not None:
        sample_man_path = (
            args.sample_manifest
            if Path(args.sample_manifest).is_absolute()
            else _ROOT / args.sample_manifest
        )
        if not sample_man_path.is_file():
            raise SystemExit(
                f"[FAIL] --sample-manifest not found: {sample_man_path}. "
                "Build with: python scripts/build_vsl400_eligible_manifest.py"
            )
        pack = load_eligible_sample_manifest(sample_man_path, root=_ROOT)
        eligible = pack["eligible_set"]
        sample_manifest_sha = pack["sha256"]
        if policy is not None and not allow_pilot:
            expected_elig = (policy.get("data_locks") or {}).get("eligible_sample_manifest_sha256")
            if not expected_elig and v3_mode:
                expected_elig = ((policy.get("lineage") or {}).get("eligible_sample_manifest_sha256"))
            if expected_elig and expected_elig != sample_manifest_sha:
                raise SystemExit(
                    f"[FAIL] eligible manifest sha mismatch vs policy: "
                    f"{sample_manifest_sha} != {expected_elig}"
                )
        print(
            f"[OK] sample-manifest eligible_n={len(eligible)} excluded_n={len(pack['excluded'])} "
            f"sha={sample_manifest_sha[:12]}… (test KPI n={pack['test_n']})",
            flush=True,
        )
        if train_ids is not None:
            train_ids = [i for i in train_ids if i in eligible]
            val_ids = [i for i in val_ids if i in eligible]

    df = pd.read_csv(_ROOT / "metadata.csv")
    v = df[df["dataset"] == "VSL400"].copy()
    if eligible is not None:
        v = v[v["video_id"].astype(str).isin(eligible)].copy()
    # Path B HPO tune signers live in metadata split=train; filter by role IDs only in v3/v4.
    if v3_mode or v4_mode:
        train_df = v
        val_df = v
    else:
        train_df = v[v["split"] == "train"]
        val_df = v[v["split"] == "val"]

    reg = load_json(REGISTRY_PATH) if REGISTRY_PATH.is_file() else {}
    rgb_pooled_dir = resolve_pooled_dir(_ROOT, reg)
    rgb_temporal_dir = resolve_temporal_dir(_ROOT, reg)
    skel_dir = resolve_skel_dir(_ROOT, reg)
    print(
        f"[paths] pooled={rgb_pooled_dir} temporal={rgb_temporal_dir} skel={skel_dir} device={device}",
        flush=True,
    )

    try:
        # For v3, skip E1-specific e1_split hash by not passing a conflicting data_locks key;
        # lineage/split checks already enforced above.
        runtime_prov = enforce_data_and_runtime_locks(
            policy or {},
            root=_ROOT,
            registry=reg,
            labels_path=labels_path,
            split_manifest_sha=None if (v3_mode or v4_mode) else manifest_hash,
            eligible_manifest_sha=sample_manifest_sha,
            pooled_dir=rgb_pooled_dir,
            temporal_dir=rgb_temporal_dir,
            skel_dir=skel_dir,
            allow_pilot=allow_pilot,
        )
    except ValueError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc

    art = (policy or {}).get("artifact_dirs") or {}
    if v4_mode:
        default_ckpt = "data/processed/ms_sta_retrain_v4"
        default_status = "logs/ms_sta_retrain_v4"
    elif v3_mode:
        default_ckpt = "data/processed/ms_sta_retrain_v3"
        default_status = "logs/ms_sta_retrain_v3"
    else:
        default_ckpt = "data/processed/ms_sta_e1_v2"
        default_status = "logs/gate_e1_v2"
    out_dir = _ROOT / str(art.get("checkpoints") or default_ckpt)
    status_dir = _ROOT / str(art.get("status") or default_status)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact_stem:
        stem = str(args.artifact_stem)
    elif v4_mode or v3_mode:
        stem = (
            f"{args.phase}__{experiment_variant}__{args.config_id}__seed{args.seed}"
            f"__stride{skel_temporal_stride}"
        )
    else:
        stem = f"{variant}_seed{args.seed}"
    best_path = out_dir / f"{stem}.pth"
    status_path = status_dir / f"{stem}.json"
    try:
        refuse_output_collision(
            checkpoint_path=best_path,
            status_path=status_path,
            allow_overwrite=bool(args.force_overwrite),
            allow_pilot=allow_pilot,
        )
    except ValueError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc

    warmup_plan = {"epochs": 0, "modules": []}
    if policy is not None and not allow_pilot and not v3_mode and not v4_mode:
        warmup_plan = resolve_warmup(policy, variant)
    if v3_mode or v4_mode:
        # Optimizer LR warm-up replaces historical head-only Gate3 warm-up.
        warmup_plan = {"epochs": 0, "modules": [], "optimizer_lr_warmup_epochs": 1}

    train_ds = VSL400MSSTADataset(
        train_df,
        gloss_to_id,
        require_skel=require_skel,
        require_temporal_rgb=require_temporal,
        rgb_pooled_dir=rgb_pooled_dir,
        rgb_temporal_dir=rgb_temporal_dir,
        skel_dir=skel_dir,
        video_ids=train_ids,
    )
    val_ds = VSL400MSSTADataset(
        val_df,
        gloss_to_id,
        require_skel=require_skel,
        require_temporal_rgb=require_temporal,
        rgb_pooled_dir=rgb_pooled_dir,
        rgb_temporal_dir=rgb_temporal_dir,
        skel_dir=skel_dir,
        video_ids=val_ids,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Empty dataset variant={variant} train={len(train_ds)} val={len(val_ds)}. "
            f"Need {'temporal RGB + ' if require_temporal else ''}{'skel' if require_skel else 'RGB'}."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    model = build_ms_sta(
        variant,
        num_classes=num_classes,
        disable_spatial=disable_spatial,
        disable_temporal=disable_temporal,
        skel_temporal_stride=skel_temporal_stride,
        residual_init=residual_init if variant == "b4r" else "zero_delta_head",
    ).to(device)
    gate3_path = _ROOT / DEFAULT_CKPT
    if policy is not None:
        gate3_rel = (policy.get("initialization") or {}).get("gate3_checkpoint")
        if gate3_rel:
            gate3_path = _ROOT / str(gate3_rel)
    try:
        applied_init = apply_initialization(model, init_mode=init_mode, gate3_ckpt_path=gate3_path)
    except ValueError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
    print(
        f"[init] variant={variant} mode={applied_init} "
        f"disable_spatial={disable_spatial} disable_temporal={disable_temporal} "
        f"warmup_epochs={warmup_plan['epochs']} warmup_modules={warmup_plan['modules']}",
        flush=True,
    )

    if warmup_plan["epochs"] > 0:
        set_head_only_trainable(model, modules=warmup_plan["modules"], enabled=True)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = None
    sched_per_step = False
    if use_lr_schedule:
        opt, scheduler, sched_meta = build_optimizer_and_scheduler(
            (p for p in model.parameters() if p.requires_grad),
            lr=float(args.lr),
            weight_decay=float(weight_decay),
            epochs=int(args.epochs),
            steps_per_epoch=max(1, len(train_loader)),
            torch_module=torch,
        )
        sched_per_step = True
        print(
            f"[sched] linear_warmup_steps={sched_meta['warmup_steps']} start_factor=0.01 "
            f"then cosine_steps={sched_meta['cosine_steps']} residual_init={residual_init}",
            flush=True,
        )
    else:
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=weight_decay,
        )

    best_val1 = -1.0
    best_val5 = -1.0
    bad = 0
    t0 = time.time()
    history = []
    train_id_list = [r[0] for r in train_ds.rows]
    val_id_list = [r[0] for r in val_ds.rows]
    # Pin code-lock at trainer process start; status must not silently follow later HEAD moves.
    head_rev = git_rev(_ROOT)
    head_dirty = git_dirty(_ROOT)
    skel_ev = (reg.get("features") or {}).get("skel_promotion_evidence") or {}
    temporal_ev = (reg.get("features") or {}).get("temporal_promotion_evidence") or {}
    b0_role = (
        "b0_v4"
        if (variant == "b0" and v4_mode)
        else ("b0_v3" if (variant == "b0" and v3_mode) else ("b0_e1" if variant == "b0" else None))
    )
    collapse_failed = False
    checkpoint_status = "candidate"
    effective_manifest = None
    if split_manifest_path is not None:
        try:
            effective_manifest = str(split_manifest_path.resolve().relative_to(_ROOT.resolve()))
        except Exception:
            effective_manifest = str(split_manifest_path)

    def _write_status(*, extra: dict | None = None) -> None:
        # Refuse to emit status if HEAD drifted under this trainer process.
        live_rev = git_rev(_ROOT)
        live_dirty = git_dirty(_ROOT)
        if live_rev and head_rev and live_rev != head_rev:
            raise SystemExit(
                f"[FAIL] git_revision changed during trainer process: {live_rev} != {head_rev}"
            )
        if (v3_mode or v4_mode) and live_dirty:
            raise SystemExit("[FAIL] worktree became dirty during trainer process")
        payload = {
            "gate": "retrain_v4" if v4_mode else ("retrain_v3" if v3_mode else "E1"),
            "e1_version": None if (v3_mode or v4_mode) else "v2",
            "retrain_v3": bool(v3_mode),
            "retrain_v4": bool(v4_mode),
            "phase": args.phase if (v3_mode or v4_mode) else None,
            "config_id": args.config_id if (v3_mode or v4_mode) else None,
            "experiment_variant": experiment_variant if (v3_mode or v4_mode) else variant,
            "variant": variant,
            "residual_init": residual_init,
            "seed": args.seed,
            "train_role": args.train_role if (v3_mode or v4_mode) else "train",
            "val_role": args.val_role if (v3_mode or v4_mode) else "val",
            "train_n": len(train_ds),
            "val_n": len(val_ds),
            "best_val_top1": best_val1 if best_val1 >= 0 else None,
            "best_val_top5": best_val5 if best_val5 >= 0 else None,
            "best_ckpt": str(best_path) if best_path.is_file() else None,
            "checkpoint_status": checkpoint_status,
            "selection": "val_top1",
            "test_eval": "deferred",
            "init_mode": applied_init,
            "warmup": warmup_plan,
            "disable_spatial": disable_spatial,
            "disable_temporal": disable_temporal,
            "skel_temporal_stride": skel_temporal_stride,
            "early_stopping": bool(early_stopping),
            "device": str(device),
            "e1_policy_sha256": None if (v3_mode or v4_mode) else policy_sha,
            "retrain_v3_policy_sha256": policy_sha if v3_mode else None,
            "retrain_v4_policy_sha256": policy_sha if v4_mode else None,
            # Effective split path actually loaded (not the E1 argparse default).
            "manifest": effective_manifest,
            "manifest_sha256": manifest_hash,
            "history": history,
            "elapsed_s": round(time.time() - t0, 1),
            "git_revision": head_rev,
            "git_dirty": head_dirty,
            "test_claim_label": (
                ((v4_policy or {}).get("test_protocol") or {}).get("label")
                if v4_mode
                else (((v3_policy or {}).get("test_claim_label")) if v3_mode else None)
            ),
        }
        if extra:
            payload.update(extra)
        status_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for epoch in range(1, args.epochs + 1):
        if warmup_plan["epochs"] > 0 and epoch == warmup_plan["epochs"] + 1:
            set_head_only_trainable(model, modules=warmup_plan["modules"], enabled=False)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
            print(
                f"[{variant}] unfreeze-all after warmup epoch {warmup_plan['epochs']}",
                flush=True,
            )

        model.train()
        weighted_loss = 0.0
        correct = 0
        correct5 = 0
        seen = 0
        steps = 0
        grad_delta_acc = 0.0
        grad_cross_acc = 0.0
        grad_spatial_acc = 0.0
        grad_rgb_temp_acc = 0.0
        grad_skel_temp_acc = 0.0
        grad_gate_acc = 0.0
        grad_delta_steps = 0
        grad_cross_steps = 0
        grad_spatial_steps = 0
        grad_rgb_temp_steps = 0
        grad_skel_temp_steps = 0
        grad_gate_steps = 0
        collect_b4r_diag = bool(v4_mode and str(variant) == "b4r")
        for rgb, skel, y, _, rgb_kpm in train_loader:
            rgb = rgb.to(device)
            y = y.to(device)
            skel_t = skel.to(device) if skel is not None else None
            kpm = rgb_kpm.to(device) if rgb_kpm is not None else None
            opt.zero_grad(set_to_none=True)
            if variant in ("b0", "b1"):
                logits = model(rgb=rgb, rgb_key_padding_mask=kpm)
            elif variant == "b2":
                logits = model(skel=skel_t)
            else:
                logits = model(rgb=rgb, skel=skel_t, rgb_key_padding_mask=kpm)
            loss = crit(logits, y)
            loss.backward()
            if collect_b4r_diag:
                g_delta = _grad_norm(model.cls.parameters())
                g_cross = _grad_norm(model.cross.parameters())
                g_spatial = _grad_norm(model.spatial.parameters())
                g_rgb_t = _grad_norm(model.rgb_temporal.parameters())
                g_skel_t = _grad_norm(model.skel_temporal.parameters())
                g_gate = _grad_norm(model.gate.parameters())
                if g_delta is not None:
                    grad_delta_acc += g_delta
                    grad_delta_steps += 1
                if g_cross is not None:
                    grad_cross_acc += g_cross
                    grad_cross_steps += 1
                if g_spatial is not None:
                    grad_spatial_acc += g_spatial
                    grad_spatial_steps += 1
                if g_rgb_t is not None:
                    grad_rgb_temp_acc += g_rgb_t
                    grad_rgb_temp_steps += 1
                if g_skel_t is not None:
                    grad_skel_temp_acc += g_skel_t
                    grad_skel_temp_steps += 1
                if g_gate is not None:
                    grad_gate_acc += g_gate
                    grad_gate_steps += 1
            if (v3_mode or v4_mode) and args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            if scheduler is not None and sched_per_step:
                scheduler.step()
            bs = int(y.numel())
            weighted_loss += float(loss.item()) * bs
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            topk = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            correct5 += int((topk == y.unsqueeze(1)).any(dim=1).sum().item())
            seen += bs
            steps += 1
        train_loss = _sample_weighted_mean(weighted_loss, seen)
        train_top1 = 100.0 * correct / max(seen, 1)
        train_top5 = 100.0 * correct5 / max(seen, 1)
        val_m = _eval_top(model, val_loader, device, variant)
        hist = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_aggregation": "sample_weighted_mean_ce",
            "train_top1": train_top1,
            "train_top5": train_top5,
            "lr": float(opt.param_groups[0]["lr"]),
            "head_only_warmup": bool(warmup_plan["epochs"] and epoch <= warmup_plan["epochs"]),
            "val": val_m,
            "val_loss": val_m.get("loss"),
            "val_loss_aggregation": val_m.get("loss_aggregation"),
            "val_top1": val_m.get("top1"),
            "val_top5": val_m.get("top5"),
            "checkpoint_status": "candidate",
        }
        if collect_b4r_diag:
            missing_grad: dict[str, str] = {}

            def _step_mean(acc: float, n_steps: int, field: str) -> float | None:
                if n_steps <= 0:
                    missing_grad[field] = "no_grad_steps"
                    return None
                return _finite_or_null(acc / n_steps, field=field, missing=missing_grad)

            g_delta = _step_mean(
                grad_delta_acc, grad_delta_steps, "delta_head_grad_norm_preclip_step_mean"
            )
            g_cross = _step_mean(
                grad_cross_acc, grad_cross_steps, "cross_attention_grad_norm_preclip_step_mean"
            )
            g_spatial = _step_mean(
                grad_spatial_acc,
                grad_spatial_steps,
                "spatial_attention_grad_norm_preclip_step_mean",
            )
            g_rgb_t = _step_mean(
                grad_rgb_temp_acc,
                grad_rgb_temp_steps,
                "rgb_temporal_grad_norm_preclip_step_mean",
            )
            g_skel_t = _step_mean(
                grad_skel_temp_acc,
                grad_skel_temp_steps,
                "skel_temporal_grad_norm_preclip_step_mean",
            )
            g_gate = _step_mean(
                grad_gate_acc, grad_gate_steps, "stream_gate_grad_norm_preclip_step_mean"
            )
            # Optional aggregate: RSS of the four attention groups (not a substitute).
            attn_rss = _root_sum_square([g_cross, g_spatial, g_rgb_t, g_skel_t])
            attn_rss = _finite_or_null(
                attn_rss, field="attention_branch_grad_norm", missing=missing_grad
            )
            fwd = _b4r_forward_diagnostics(
                model,
                val_loader,
                device,
                max_batches=8,
                diagnostic_split="val",
                loader_shuffle=False,
            )
            hist["b4r_diagnostics"] = {
                **fwd,
                "delta_head_grad_norm_preclip_step_mean": g_delta,
                "cross_attention_grad_norm_preclip_step_mean": g_cross,
                "spatial_attention_grad_norm_preclip_step_mean": g_spatial,
                "rgb_temporal_grad_norm_preclip_step_mean": g_rgb_t,
                "skel_temporal_grad_norm_preclip_step_mean": g_skel_t,
                "stream_gate_grad_norm_preclip_step_mean": g_gate,
                "attention_branch_grad_norm": attn_rss,
                "attention_branch_grad_norm_definition": (
                    "root_sum_square_of_cross_spatial_rgb_temporal_skel_temporal"
                ),
                "residual_init": residual_init,
            }
            if missing_grad:
                hist["b4r_diagnostics"].setdefault("missing_fields", {}).update(missing_grad)
        history.append(hist)
        print(
            f"[{variant}] epoch={epoch} loss={train_loss:.4f} train@1={train_top1:.2f} "
            f"train@5={train_top5:.2f} "
            f"val@1={val_m['top1']:.2f} val@5={val_m['top5']:.2f} n={val_m['n']}",
            flush=True,
        )
        if collect_b4r_diag:
            d = hist["b4r_diagnostics"]
            print(
                f"[{variant}] diagnostics residual_scale={d.get('residual_scale')} "
                f"delta/base={d.get('delta_over_base_ratio')} "
                f"gate_ent={d.get('stream_gate_entropy')} "
                f"grad_delta={d.get('delta_head_grad_norm_preclip_step_mean')} "
                f"grad_cross={d.get('cross_attention_grad_norm_preclip_step_mean')} "
                f"grad_spatial={d.get('spatial_attention_grad_norm_preclip_step_mean')} "
                f"grad_rgb_t={d.get('rgb_temporal_grad_norm_preclip_step_mean')} "
                f"grad_skel_t={d.get('skel_temporal_grad_norm_preclip_step_mean')} "
                f"grad_gate={d.get('stream_gate_grad_norm_preclip_step_mean')}",
                flush=True,
            )
        try:
            if policy is not None:
                check_collapse_guard(
                    policy,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_top1=train_top1,
                    allow_pilot=allow_pilot,
                )
        except ValueError as exc:
            collapse_failed = True
            print(f"[FAIL] {exc}", flush=True)
            _write_status(extra={"collapse_guard_failed": True, "error": str(exc)})
            raise SystemExit(f"[FAIL] {exc}") from exc

        if val_m["top1"] > best_val1:
            best_val1 = float(val_m["top1"])
            best_val5 = float(val_m["top5"])
            bad = 0
            checkpoint_status = "candidate"
            payload = {
                "schema_version": 4 if v4_mode else (3 if v3_mode else 2),
                "artifact_kind": (
                    "ms_sta_retrain_v4"
                    if v4_mode
                    else ("ms_sta_retrain_v3" if v3_mode else "ms_sta")
                ),
                "e1_version": None if (v3_mode or v4_mode) else "v2",
                "retrain_v3": bool(v3_mode),
                "retrain_v4": bool(v4_mode),
                "phase": args.phase if (v3_mode or v4_mode) else None,
                "config_id": args.config_id if (v3_mode or v4_mode) else None,
                "experiment_variant": experiment_variant if (v3_mode or v4_mode) else variant,
                "defense_ready": False,
                "e1_selected": False,
                "evaluation_locked": False,
                "checkpoint_status": "candidate",
                "variant": variant,
                "residual_init": residual_init,
                "seed": args.seed,
                "b0_identity_role": b0_role,
                "init_mode": applied_init,
                "warmup": warmup_plan,
                "disable_spatial": disable_spatial,
                "disable_temporal": disable_temporal,
                "skel_temporal_stride": skel_temporal_stride,
                "model": model.state_dict(),
                "num_classes": num_classes,
                "val": val_m,
                "selection": "val_top1",
                "test_eval": "deferred",
                "git_revision": head_rev,
                "git_dirty": head_dirty,
                "e1_policy_sha256": None if (v3_mode or v4_mode) else policy_sha,
                "retrain_v3_policy_sha256": policy_sha if v3_mode else None,
                "retrain_v4_policy_sha256": policy_sha if v4_mode else None,
                "device": str(device),
                "hyperparameters": {
                    **checkpoint_hyperparameters(
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(weight_decay),
                        label_smoothing=float(label_smoothing),
                        grad_clip=float(args.grad_clip) if args.grad_clip is not None else None,
                        patience=None if not early_stopping else int(args.patience),
                        early_stopping=bool(early_stopping),
                        residual_init=str(residual_init),
                        skel_temporal_stride=int(skel_temporal_stride),
                    ),
                    "fixed_rgb_l": FIXED_RGB_L,
                    "min_temporal_l": MIN_TEMPORAL_L,
                    "require_temporal_rgb": require_temporal,
                    "require_skel": require_skel,
                    "b0_rgb_source": "isolated_sf2304" if variant == "b0" else None,
                    "head_only_warmup_epochs": warmup_plan["epochs"],
                    "optimizer_lr_warmup_epochs": 1 if use_lr_schedule else 0,
                    "scheduler": (
                        "linear_warmup_1_epoch_then_cosine" if use_lr_schedule else "none"
                    ),
                },
                "train_role": args.train_role if (v3_mode or v4_mode) else "train",
                "val_role": args.val_role if (v3_mode or v4_mode) else "val",
                "train_ids": train_id_list,
                "val_ids": val_id_list,
                "labels_sha256": runtime_prov.get("labels_sha256") or _sha256_file(labels_path),
                "manifest_sha256": manifest_hash,
                "sample_manifest_sha256": sample_manifest_sha,
                "eligible_n": len(eligible) if eligible is not None else None,
                "active_skel_dir": reg.get("active_skel_dir"),
                "active_temporal_dir": reg.get("active_temporal_dir"),
                "skel_manifest_sha256": runtime_prov.get("skel_manifest_sha256")
                or skel_ev.get("manifest_sha256"),
                "temporal_manifest_sha256": runtime_prov.get("temporal_manifest_sha256")
                or temporal_ev.get("manifest_sha256"),
                "cuda_provenance_sha256": runtime_prov.get("cuda_provenance_sha256"),
                "backbone_checkpoint_sha256": runtime_prov.get("backbone_checkpoint_sha256"),
                "pip_freeze_sha256": runtime_prov.get("pip_freeze_sha256"),
                "gate3_checkpoint_sha256": None
                if (v3_mode or v4_mode)
                else runtime_prov.get("gate3_checkpoint_sha256"),
                "test_claim_label": (
                    ((v4_policy or {}).get("test_protocol") or {}).get("label")
                    if v4_mode
                    else (((v3_policy or {}).get("test_claim_label")) if v3_mode else None)
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(payload, best_path)
        else:
            bad += 1
            if should_early_stop(
                bad_epochs=bad,
                patience=args.patience,
                early_stopping=early_stopping,
            ):
                print(f"[{variant}] early stop patience={args.patience}", flush=True)
                _write_status()
                break

        _write_status()

    if not collapse_failed:
        print(f"[done] locked {best_path} val@1={best_val1:.2f} val@5={best_val5:.2f}", flush=True)
        if (
            (v3_mode or v4_mode)
            and best_path.is_file()
            and best_val1 >= 0
            and str(args.phase) == "confirm"
        ):
            final_path = best_path.with_name(final_stem_from_candidate_stem(best_path.stem) + ".pth")
            promo = promote_candidate_checkpoint(best_path, final_path)
            checkpoint_status = "final_selected_checkpoint"
            _write_status(
                extra={
                    "checkpoint_status": "final_selected_checkpoint",
                    "final_checkpoint": promo["final_path"],
                    "final_checkpoint_sha256": promo["final_sha256"],
                    "source_candidate_sha256": promo["source_candidate_sha256"],
                    "lifecycle": promo,
                    "residual_init": residual_init,
                }
            )
            print(
                f"[promote] final={promo['final_path']} "
                f"src={promo['source_candidate_sha256'][:12]}… "
                f"final_sha={promo['final_sha256'][:12]}…",
                flush=True,
            )
        elif v3_mode or v4_mode:
            _write_status(
                extra={
                    "checkpoint_status": "candidate",
                    "hpo_disposable": str(args.phase) in {"hpo_screen", "hpo_full", "hpo"},
                    "residual_init": residual_init,
                }
            )


if __name__ == "__main__":
    main()
