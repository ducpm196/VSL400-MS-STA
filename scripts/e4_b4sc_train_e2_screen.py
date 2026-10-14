# -*- coding: utf-8 -*-
"""E4_B4SC_V1 E2 narrow SlowFast screen training (seed42, policy-locked).

Trains narrowed Slow pathway + RoiToHeadProjection + StageKDAdapters with frozen
teacher SlowFast, frozen B4 head, and E1a Q2 precision (sequential decode only).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import (  # noqa: E402
    b4_item_from_extract,
    sealed_landmarks,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    INPUT_SIZE,
    REGIONS,
    ROI_OUT,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    FrozenSlowFastBlock4,
    PathwayMaps,
    _boxes_for_native_frame,
    _local_frame_for_feat_t,
    _resize_letterbox,
    extract_rgb_token_v3_for_video,
    landmark_xy_to_resized,
    nearest_skel_landmark,
    read_video_frames_native,
    window_starts,
    window_tensors,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, maybe_autocast  # noqa: E402
from src.optimization.e4_b4sc.distill_losses import (  # noqa: E402
    kl_logit_loss,
    roi_region_loss,
    stage_feature_loss,
    temporal_delta_loss,
)
from src.optimization.e4_b4sc.narrow_slowfast import (  # noqa: E402
    E2_WIDTHS,
    NarrowSlowFastBlock4,
    RoiToHeadProjection,
    StageKDAdapters,
    TEACHER_SLOW_WIDTHS,
)
from src.training.ms_sta_v5_data import collate_v5, load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POL_TRAIN = OUT / "E4_B4SC_V1_E2E3_TRAINING_POLICY.json"
POL_E4 = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"
LOCK_PATH = OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json"
SUBSET_PATH = OUT / "E4_B4SC_V1_E2_SCREEN_TRAIN_SUBSET_IDS.json"
LOG_PATH = OUT / "E4_B4SC_V1_E2_SCREEN_TRAIN_LOG.jsonl"
CKPT_DIR = OUT / "checkpoints/e2_screen_seed42"
BEST_CKPT = CKPT_DIR / "best.pt"
VAL_MONITOR_N = 200


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_E2_SCREEN_FAIL:{msg}")


def ensure_train_cuda() -> None:
    venv_py = ROOT / "_venvs/train_cuda/Scripts/python.exe"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        os.execv(str(venv_py), [str(venv_py), *sys.argv])


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif obj is None:
        path.write_text("", encoding="utf-8")
    else:
        path.write_text(str(obj), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with (ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("dataset") or "") != "VSL400":
                continue
            sid = str(row.get("video_id") or "").strip()
            if sid:
                out[sid] = dict(row)
    return out


def stratified_sample_ids(
    pool_ids: list[str],
    meta: dict[str, dict[str, Any]],
    n: int,
    seed: int,
    *,
    signer_field: str,
) -> list[str]:
    by_sig: dict[str, list[str]] = defaultdict(list)
    for sid in pool_ids:
        row = meta.get(sid)
        if row is None:
            die(f"missing_metadata:{sid}")
        sig = str(row.get(signer_field) or row.get("signer") or "").strip()
        if not sig:
            die(f"missing_signer:{sid}")
        by_sig[sig].append(sid)
    for sig in by_sig:
        by_sig[sig] = sorted(by_sig[sig], key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    total = len(pool_ids)
    if n > total:
        die(f"subset_n_exceeds_pool:{n}>{total}")
    signers = sorted(by_sig.keys())
    qmap: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for sig in signers:
        exact = n * len(by_sig[sig]) / total
        floor = int(np.floor(exact))
        qmap[sig] = floor
        remainders.append((exact - floor, sig))
        assigned += floor
    remainders.sort(key=lambda x: (-x[0], x[1]))
    extra = n - assigned
    for i in range(extra):
        qmap[remainders[i % len(remainders)][1]] += 1
    chosen: list[str] = []
    for sig in signers:
        chosen.extend(by_sig[sig][: qmap[sig]])
    if len(chosen) != n:
        die(f"stratified_count_mismatch:{len(chosen)}!={n}")
    if len(set(chosen)) != n:
        die("stratified_duplicates")
    return chosen


class AmpRoiFp32SlowFastTrainable(nn.Module):
    """E1a Q2 SlowFast AMP + FP32 maps; gradients enabled for student training."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def load_meta(self) -> dict[str, Any]:
        return self.base.load_meta

    def forward_maps(self, slow: torch.Tensor, fast: torch.Tensor) -> PathwayMaps:
        with maybe_autocast(True, torch.float16):
            maps = self.base.forward_maps(slow, fast)
        return PathwayMaps(slow=maps.slow.float(), fast=maps.fast.float())

    def forward_maps_with_stages(
        self, slow: torch.Tensor, fast: torch.Tensor
    ) -> tuple[PathwayMaps, list[torch.Tensor]]:
        with maybe_autocast(True, torch.float16):
            maps, stages = self.base.forward_maps_with_stages(slow, fast)
        return PathwayMaps(slow=maps.slow.float(), fast=maps.fast.float()), [s.float() for s in stages]


def teacher_forward_maps_with_stages(
    teacher: FrozenSlowFastBlock4,
    slow: torch.Tensor,
    fast: torch.Tensor,
) -> tuple[PathwayMaps, list[torch.Tensor]]:
    """Collect teacher pathway maps + 4 Slow stage features (pre-fusion slice)."""
    dev = teacher.device
    x: Any = [slow.to(dev), fast.to(dev)]
    slow_feats: list[torch.Tensor] = []
    for i, block in enumerate(teacher.blocks):
        x = block(x)
        if i >= 1 and isinstance(x, (list, tuple)):
            w = int(TEACHER_SLOW_WIDTHS[i - 1])
            feat = x[0]
            if feat.shape[1] < w:
                raise RuntimeError(f"teacher_slow_feat_narrow:{feat.shape[1]}<{w}")
            slow_feats.append(feat[:, :w])
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise RuntimeError(f"unexpected_teacher_block4_output:{type(x)}")
    if len(slow_feats) != 4:
        raise RuntimeError(f"expected_4_teacher_slow_stages:{len(slow_feats)}")
    return PathwayMaps(slow=x[0][:, : TEACHER_SLOW_WIDTHS[-1]], fast=x[1]), slow_feats


def roi_align_token_diff(feat: torch.Tensor, box: np.ndarray | None) -> tuple[torch.Tensor, bool]:
    if box is None:
        return torch.zeros(feat.shape[1], device=feat.device, dtype=feat.dtype), True
    if feat.ndim != 4:
        raise RuntimeError(f"roi_align_expects_bchw:{tuple(feat.shape)}")
    _, _, h, w = feat.shape
    scale_x = w / float(INPUT_SIZE)
    scale_y = h / float(INPUT_SIZE)
    b = box.copy()
    b[0] *= scale_x
    b[2] *= scale_x
    b[1] *= scale_y
    b[3] *= scale_y
    rois = torch.tensor([[0.0, b[0], b[1], b[2], b[3]]], device=feat.device, dtype=feat.dtype)
    out = roi_align(feat, rois, output_size=(ROI_OUT, ROI_OUT), spatial_scale=1.0, aligned=True)
    return out.mean(dim=(2, 3)).squeeze(0), False


def masked_mean_tokens_diff(
    tokens: list[torch.Tensor],
    missing: list[bool],
) -> tuple[torch.Tensor, bool]:
    valid = [tok for tok, miss in zip(tokens, missing) if not miss]
    if not valid:
        dim = int(tokens[0].shape[0]) if tokens else 0
        return tokens[0].new_zeros(dim) if tokens else torch.zeros(0), True
    return torch.stack(valid, dim=0).mean(dim=0), False


def student_extract_with_teacher_stages(
    *,
    frames_native: np.ndarray,
    frame_idx: np.ndarray,
    fps: float,
    landmarks_by_frame: dict[int, np.ndarray],
    missing_by_frame: dict[int, np.ndarray],
    skel_frame_index: np.ndarray,
    skel_timestamps: np.ndarray,
    student: AmpRoiFp32SlowFastTrainable,
    teacher: FrozenSlowFastBlock4,
    device: torch.device,
    window_size: int = WINDOW_SIZE,
    window_stride: int = WINDOW_STRIDE,
) -> tuple[dict[str, torch.Tensor], list[tuple[list[torch.Tensor], list[torch.Tensor]]]]:
    resized: list[np.ndarray] = []
    metas: list[dict[str, float]] = []
    for fr in frames_native:
        canvas, meta = _resize_letterbox(fr, INPUT_SIZE)
        resized.append(canvas)
        metas.append(meta)
    frames_u8 = np.stack(resized, axis=0)
    t_native = frames_u8.shape[0]
    starts = window_starts(t_native, window_size=window_size, window_stride=window_stride)

    region_seq: dict[str, list[torch.Tensor]] = {r: [] for r in REGIONS}
    stage_pairs: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    for start in starts:
        slow_t, fast_t, _valid_mask, n_valid = window_tensors(frames_u8, start, window_size)
        slow_t = slow_t.to(device)
        fast_t = fast_t.to(device)
        maps, stu_stages = student.forward_maps_with_stages(slow_t, fast_t)
        with torch.no_grad():
            with maybe_autocast(True, torch.float16):
                _, tea_stages = teacher_forward_maps_with_stages(teacher, slow_t, fast_t)
            tea_stages = [t.float() for t in tea_stages]
        stage_pairs.append((stu_stages, tea_stages))

        end = min(t_native, start + window_size)
        center = start + (end - start) // 2
        align = nearest_skel_landmark(
            center_frame=int(frame_idx[center]),
            fps=fps,
            skel_frame_index=skel_frame_index,
            skel_timestamps=skel_timestamps,
        )
        lm_frame = int(align["landmark_frame"])
        lm = landmarks_by_frame.get(lm_frame)
        miss = missing_by_frame.get(lm_frame)
        if lm is None or miss is None:
            die(f"landmark_missing:center={frame_idx[center]}:lm={lm_frame}")

        t_slow = int(maps.slow.shape[2])
        t_fast = int(maps.fast.shape[2])
        per_slow: dict[str, list[torch.Tensor]] = {r: [] for r in REGIONS}
        per_fast: dict[str, list[torch.Tensor]] = {r: [] for r in REGIONS}
        miss_slow: dict[str, list[bool]] = {r: [] for r in REGIONS}
        miss_fast: dict[str, list[bool]] = {r: [] for r in REGIONS}

        for t_idx in range(t_slow):
            local = _local_frame_for_feat_t(
                pathway="slow", t_idx=t_idx, t_feat=t_slow, n_valid=n_valid, window_size=window_size
            )
            native = int(frame_idx[min(start + local, t_native - 1)])
            boxes_t, _, _, _, _ = _boxes_for_native_frame(
                native_frame=native,
                canvas_meta=metas[min(start + local, t_native - 1)],
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                skel_frame_index=skel_frame_index,
                skel_timestamps=skel_timestamps,
                fps=fps,
                enable_pose_wrist_prior=True,
                enable_ambiguity_aware_prior=True,
                ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
            )
            feat_t = maps.slow[:, :, t_idx, :, :]
            for region in REGIONS:
                tok, mflag = roi_align_token_diff(feat_t, boxes_t[region])
                per_slow[region].append(tok)
                miss_slow[region].append(bool(mflag))

        for t_idx in range(t_fast):
            local = _local_frame_for_feat_t(
                pathway="fast", t_idx=t_idx, t_feat=t_fast, n_valid=n_valid, window_size=window_size
            )
            native = int(frame_idx[min(start + local, t_native - 1)])
            boxes_t, _, _, _, _ = _boxes_for_native_frame(
                native_frame=native,
                canvas_meta=metas[min(start + local, t_native - 1)],
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                skel_frame_index=skel_frame_index,
                skel_timestamps=skel_timestamps,
                fps=fps,
                enable_pose_wrist_prior=True,
                enable_ambiguity_aware_prior=True,
                ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
            )
            feat_t = maps.fast[:, :, t_idx, :, :]
            for region in REGIONS:
                tok, mflag = roi_align_token_diff(feat_t, boxes_t[region])
                per_fast[region].append(tok)
                miss_fast[region].append(bool(mflag))

        for region in REGIONS:
            tok_s, _ = masked_mean_tokens_diff(per_slow[region], miss_slow[region])
            tok_f, _ = masked_mean_tokens_diff(per_fast[region], miss_fast[region])
            region_seq[region].append(torch.cat([tok_s, tok_f], dim=0))

    tokens = {r: torch.stack(v, dim=0) for r, v in region_seq.items()}
    return tokens, stage_pairs


def project_tokens(
    roi_proj: RoiToHeadProjection,
    tokens: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return roi_proj(tokens)


def build_b4_batch_from_projected(
    *,
    sid: str,
    label: int,
    signer: str,
    proj_tokens: dict[str, torch.Tensor],
    sealed: dict[str, Any],
    device: torch.device,
    max_rgb_len: int = 8,
    max_skel_len: int = 96,
) -> dict[str, Any]:
    from src.data_prep.joint_layout_v3 import N_JOINTS
    from src.models.ms_sta_v5 import CHIRALITY_VOCAB, PART_VOCAB, RGB_REGIONS

    l = int(proj_tokens["global"].shape[0])
    if l > max_rgb_len:
        start = (l - max_rgb_len) // 2
        sl = slice(start, start + max_rgb_len)
    else:
        sl = slice(0, l)
    rgb_len = sl.stop - sl.start
    t_total = int(sealed["coordinates"].shape[0])
    if t_total > max_skel_len:
        st0 = (t_total - max_skel_len) // 2
        skel_len = max_skel_len
    else:
        st0 = 0
        skel_len = t_total

    batch: dict[str, Any] = {
        "sample_id": [sid],
        "label": torch.tensor([label], dtype=torch.long, device=device),
        "signer": [signer],
        "rgb_pad_mask": torch.zeros((1, rgb_len), dtype=torch.bool, device=device),
        "skel_pad_mask": torch.zeros((1, skel_len), dtype=torch.bool, device=device),
    }
    for r in RGB_REGIONS:
        batch[f"token_{r}"] = proj_tokens[r][sl].unsqueeze(0)
        batch[f"missing_{r}"] = torch.zeros((1, rgb_len), dtype=torch.bool, device=device)
    batch["coordinates"] = (
        torch.from_numpy(np.asarray(sealed["coordinates"][st0 : st0 + skel_len], dtype=np.float32))
        .unsqueeze(0)
        .to(device)
    )
    batch["velocity"] = (
        torch.from_numpy(np.asarray(sealed["velocity"][st0 : st0 + skel_len], dtype=np.float32))
        .unsqueeze(0)
        .to(device)
    )
    batch["validity"] = (
        torch.from_numpy(np.asarray(sealed["validity"][st0 : st0 + skel_len], dtype=np.float32))
        .unsqueeze(0)
        .to(device)
    )
    if "joint_part" in sealed:
        parts = [PART_VOCAB.get(str(x), 0) for x in sealed["joint_part"].tolist()]
    else:
        parts = [0] * N_JOINTS
    if "joint_chirality" in sealed:
        chirs = [CHIRALITY_VOCAB.get(str(x), 0) for x in sealed["joint_chirality"].tolist()]
    else:
        chirs = [0] * N_JOINTS
    batch["joint_part_ids"] = torch.tensor(parts, dtype=torch.long, device=device)
    batch["joint_chirality_ids"] = torch.tensor(chirs, dtype=torch.long, device=device)
    return batch


def head_logits_trainable(b4_head: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    out = b4_head(batch)
    logits = out["logits"].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("non_finite_logits")
    return logits


@torch.no_grad()
def head_logits_eval(b4_head: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    b4_head.eval()
    out = b4_head(batch)
    logits = out["logits"].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("non_finite_logits_eval")
    return logits


def cosine_lr(epoch: int, max_epochs: int, base_lr: float, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(warmup_epochs, 1))
    if max_epochs <= warmup_epochs:
        return base_lr
    progress = (epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def process_sample(
    *,
    sid: str,
    row: dict[str, Any],
    label: int,
    signer: str,
    video_path: Path,
    sealed_path: Path,
    teacher_amp: AmpRoiFp32SlowFast,
    teacher_raw: FrozenSlowFastBlock4,
    student_amp: AmpRoiFp32SlowFastTrainable,
    roi_proj: RoiToHeadProjection,
    stage_adapters: StageKDAdapters,
    b4_head: nn.Module,
    device: torch.device,
    lambdas: dict[str, float],
    kl_temperature: float,
    roi_regions: list[str],
    train: bool,
) -> dict[str, Any]:
    frames, frame_idx, fps = read_video_frames_native(str(video_path))
    landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)

    with torch.no_grad():
        teacher_extract = extract_rgb_token_v3_for_video(
            video_path=str(video_path),
            extractor=teacher_amp,
            landmarks_by_frame=landmarks_by_frame,
            missing_by_frame=missing_by_frame,
            skel_frame_index=sfi,
            skel_timestamps=sts,
            sample_id=sid,
            enable_pose_wrist_prior=True,
            enable_ambiguity_aware_prior=True,
            ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
            repo_root=ROOT,
            frames_native=frames,
            frame_idx=frame_idx,
            fps=fps,
            pathway_window_batch_size=1,
            window_size=WINDOW_SIZE,
            window_stride=WINDOW_STRIDE,
        )
    teacher_tokens_np = teacher_extract["tokens"]
    teacher_tokens = {
        r: torch.from_numpy(np.asarray(teacher_tokens_np[r], dtype=np.float32)).to(device) for r in REGIONS
    }

    if train:
        student_tokens, stage_pairs = student_extract_with_teacher_stages(
            frames_native=frames,
            frame_idx=frame_idx,
            fps=fps,
            landmarks_by_frame=landmarks_by_frame,
            missing_by_frame=missing_by_frame,
            skel_frame_index=sfi,
            skel_timestamps=sts,
            student=student_amp,
            teacher=teacher_raw,
            device=device,
        )
        proj_tokens = project_tokens(roi_proj, student_tokens)
        batch = build_b4_batch_from_projected(
            sid=sid,
            label=label,
            signer=signer,
            proj_tokens=proj_tokens,
            sealed=sealed,
            device=device,
        )

        student_logits = head_logits_trainable(b4_head, batch)
        with torch.no_grad():
            teacher_item = b4_item_from_extract(sid, teacher_extract, sealed, label, signer)
            teacher_batch = collate_v5([teacher_item])
            teacher_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in teacher_batch.items()}
            teacher_logits = head_logits_eval(b4_head, teacher_batch)

        label_t = torch.tensor([label], device=device, dtype=torch.long)
        loss_ce = F.cross_entropy(student_logits, label_t)
        loss_kl = kl_logit_loss(student_logits, teacher_logits, kl_temperature)

        stage_losses = []
        for stu_stages, tea_stages in stage_pairs:
            adapted = stage_adapters(stu_stages)
            stage_losses.append(stage_feature_loss(adapted, tea_stages))
        loss_stage = torch.stack(stage_losses).mean() if stage_losses else student_logits.new_zeros(())

        stu_roi = {r: proj_tokens[r].unsqueeze(0) for r in roi_regions}
        tea_roi = {r: teacher_tokens[r].unsqueeze(0) for r in roi_regions}
        loss_roi, _ = roi_region_loss(stu_roi, tea_roi, roi_regions)

        loss_temporal = temporal_delta_loss(
            proj_tokens["global"].unsqueeze(0),
            teacher_tokens["global"].unsqueeze(0),
        )

        total = (
            loss_ce
            + lambdas["logit"] * loss_kl
            + lambdas["stage"] * loss_stage
            + lambdas["roi"] * loss_roi
            + lambdas["temporal"] * loss_temporal
        )
        pred = int(student_logits.argmax(dim=-1).item())
        return {
            "loss": total,
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_kl": float(loss_kl.detach().cpu()),
            "loss_stage": float(loss_stage.detach().cpu()),
            "loss_roi": float(loss_roi.detach().cpu()),
            "loss_temporal": float(loss_temporal.detach().cpu()),
            "correct": pred == label,
            "nan": not bool(torch.isfinite(total).all()),
        }

    # eval path
    with torch.no_grad():
        student_tokens, _ = student_extract_with_teacher_stages(
            frames_native=frames,
            frame_idx=frame_idx,
            fps=fps,
            landmarks_by_frame=landmarks_by_frame,
            missing_by_frame=missing_by_frame,
            skel_frame_index=sfi,
            skel_timestamps=sts,
            student=student_amp,
            teacher=teacher_raw,
            device=device,
        )
        proj_tokens = project_tokens(roi_proj, student_tokens)
        batch = build_b4_batch_from_projected(
            sid=sid,
            label=label,
            signer=signer,
            proj_tokens=proj_tokens,
            sealed=sealed,
            device=device,
        )
        logits = head_logits_eval(b4_head, batch)
        pred = int(logits.argmax(dim=-1).item())
    return {"correct": pred == label, "loss_ce": float(F.cross_entropy(logits, torch.tensor([label], device=device)).cpu())}


def save_checkpoint(
    path: Path,
    *,
    student: NarrowSlowFastBlock4,
    roi_proj: RoiToHeadProjection,
    stage_adapters: StageKDAdapters,
    epoch: int,
    val_top1: float,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ms_sta_v6_e4_b4sc_v1_e2_screen_checkpoint_v1",
        "arm": "E2",
        "seed": 42,
        "epoch": epoch,
        "val_top1_monitor": val_top1,
        "student": student.state_dict(),
        "roi_to_head_projection": roi_proj.state_dict(),
        "stage_kd_adapters_train_only": stage_adapters.state_dict(),
        "meta": meta,
        "note": "Teacher SlowFast + B4 head frozen; adapters train-only at deploy.",
    }
    torch.save(payload, path)


def main() -> int:
    ensure_train_cuda()

    ap = argparse.ArgumentParser(description="E4_B4SC E2 narrow SlowFast screen training (seed42)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="Optional smoke cap on optimizer steps")
    ap.add_argument("--epochs", type=int, default=0, help="Override epochs only with --allow-epoch-override")
    ap.add_argument(
        "--allow-epoch-override",
        action="store_true",
        help="Permit --epochs for smoke; policy coefficients unchanged",
    )
    args = ap.parse_args()

    if not POL_TRAIN.is_file():
        die("missing_training_policy")
    tpol = json.loads(POL_TRAIN.read_text(encoding="utf-8"))
    if tpol.get("status") != "FROZEN_BEFORE_TRAINING":
        die("policy_not_frozen")
    if not POL_E4.is_file():
        die("missing_e4_policy")
    pol = json.loads(POL_E4.read_text(encoding="utf-8"))
    if pol.get("status") != "FROZEN":
        die("e4_policy_not_frozen")
    if not LOCK_PATH.is_file():
        die("missing_channel_lock")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if sha256_file(POL_TRAIN) != lock.get("training_policy_sha256"):
        die("training_policy_sha_mismatch_vs_lock")
    plan_e2 = lock["E2"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("cuda_required")
    gpu = torch.cuda.get_device_name(0)
    req_gpu = pol["scientific_environment_lock"]["required_gpu_name"]
    if gpu != req_gpu:
        die(f"gpu_mismatch:{gpu}!={req_gpu}")

    screen = tpol["training_protocol"]["screen"]
    max_epochs = int(screen["max_epochs"])
    if int(args.epochs) > 0:
        if not args.allow_epoch_override:
            die("epochs_override_requires_allow_epoch_override")
        max_epochs = int(args.epochs)

    split_path = ROOT / tpol["data"]["split_manifest_rel"]
    if sha256_file(split_path) != tpol["data"]["split_manifest_sha256"]:
        die("split_manifest_sha_mismatch")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    if tpol["data"].get("forbid_test"):
        pass

    meta = load_meta()
    labels_obj = json.loads((ROOT / tpol["data"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]

    def label_for(sid: str) -> int:
        gloss = str(meta[sid].get("text_gloss") or "").strip()
        if gloss not in g2i:
            die(f"missing_label:{sid}")
        return int(g2i[gloss])

    confirm_ids = load_split_role_ids(split, tpol["data"]["train_role"])
    val_ids_all = load_split_role_ids(split, tpol["data"]["val_role"])
    if set(confirm_ids) & test_ids or set(val_ids_all) & test_ids:
        die("test_leakage")

    subset_n = int(screen["train_subset_n"])
    subset_seed = int(screen["train_subset_seed"])
    train_ids = stratified_sample_ids(
        confirm_ids,
        meta,
        subset_n,
        subset_seed,
        signer_field=str(tpol["data"]["signer_field"]),
    )
    val_ids = stratified_sample_ids(
        val_ids_all,
        meta,
        min(VAL_MONITOR_N, len(val_ids_all)),
        subset_seed + 1,
        signer_field=str(tpol["data"]["signer_field"]),
    )

    create_only(SUBSET_PATH, {
        "schema": "ms_sta_v6_e4_b4sc_v1_e2_screen_train_subset_ids_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arm": "E2",
        "seed": int(screen["seed"]),
        "train_role": tpol["data"]["train_role"],
        "train_subset_n": subset_n,
        "train_subset_seed": subset_seed,
        "sample_ids": train_ids,
        "sample_ids_sha256": hashlib.sha256(json.dumps(train_ids, separators=(",", ":")).encode()).hexdigest(),
        "val_monitor_n": len(val_ids),
        "val_monitor_ids": val_ids,
        "val_monitor_note": f"monitor-only; not full val {len(val_ids_all)}",
        "training_policy_sha256": sha256_file(POL_TRAIN),
        "channel_lock_sha256": sha256_file(LOCK_PATH),
    })
    create_only(LOG_PATH, None)
    if BEST_CKPT.is_file():
        die(f"exists:{BEST_CKPT}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    teacher_raw = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    teacher_amp = AmpRoiFp32SlowFast(teacher_raw, channels_last=False)

    student_raw = NarrowSlowFastBlock4(teacher_raw, plan_e2, device=device, requires_grad_slow=True)
    student_amp = AmpRoiFp32SlowFastTrainable(student_raw)
    token_dim = E2_WIDTHS.token_dim
    roi_proj = RoiToHeadProjection(token_dim, out_dim=2304).to(device)
    stage_adapters = StageKDAdapters(tuple(plan_e2["slow_widths"]), TEACHER_SLOW_WIDTHS).to(device)

    b4_head = build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt_path = ROOT / tpol["checkpoints"]["b4_teacher_rel"]
    if sha256_file(b4_ckpt_path) != tpol["checkpoints"]["b4_teacher_sha256"]:
        die("b4_teacher_sha_mismatch")
    b4_ckpt = torch.load(b4_ckpt_path, map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()
    for p in b4_head.parameters():
        p.requires_grad_(False)

    opt_cfg = tpol["optimizer"]
    opt = torch.optim.AdamW(
        list(student_raw.parameters()) + list(roi_proj.parameters()) + list(stage_adapters.parameters()),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg["betas"]),
    )
    lambdas = {
        "logit": float(tpol["distillation"]["lambda_logit"]),
        "stage": float(tpol["distillation"]["lambda_stage"]),
        "roi": float(tpol["distillation"]["lambda_roi"]),
        "temporal": float(tpol["distillation"]["lambda_temporal"]),
    }
    kl_temperature = float(tpol["distillation"]["kl_temperature"])
    roi_regions = list(tpol["distillation"]["roi_regions_separate"])
    grad_clip = float(tpol["training_protocol"]["grad_clip_norm"])
    patience = int(screen["early_stop_patience"])

    skel_root = ROOT / tpol["data"]["skel_rel"] / "skeleton_v3"
    resolver = RawVideoResolver(ROOT)

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_meta = {
        "git_head": git_head,
        "gpu": gpu,
        "pathway_window_batch_size": 1,
        "precision": tpol["training_protocol"]["precision"],
        "lambdas": lambdas,
        "kl_temperature": kl_temperature,
        "max_epochs": max_epochs,
        "train_subset_n": len(train_ids),
        "val_monitor_n": len(val_ids),
    }

    best_val_top1 = -1.0
    bad_epochs = 0
    global_step = 0
    seed = int(screen["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    for epoch in range(max_epochs):
        lr = cosine_lr(epoch, max_epochs, float(opt_cfg["lr"]), int(opt_cfg["warmup_epochs"]))
        for pg in opt.param_groups:
            pg["lr"] = lr

        student_raw.train()
        roi_proj.train()
        stage_adapters.train()

        order = rng.permutation(len(train_ids))
        train_correct = 0
        train_n = 0
        sums = defaultdict(float)
        grad_norm_last = 0.0
        nan_seen = False

        for i_pos, idx in enumerate(order):
            sid = train_ids[int(idx)]
            row = meta[sid]
            label = label_for(sid)
            signer = str(row.get("signer_id") or row.get("signer") or "")
            video_path = Path(resolver.resolve(row))
            sealed_path = skel_root / f"{sid}.npz"
            if not sealed_path.is_file():
                die(f"sealed_missing:{sealed_path}")

            opt.zero_grad(set_to_none=True)
            out = process_sample(
                sid=sid,
                row=row,
                label=label,
                signer=signer,
                video_path=video_path,
                sealed_path=sealed_path,
                teacher_amp=teacher_amp,
                teacher_raw=teacher_raw,
                student_amp=student_amp,
                roi_proj=roi_proj,
                stage_adapters=stage_adapters,
                b4_head=b4_head,
                device=device,
                lambdas=lambdas,
                kl_temperature=kl_temperature,
                roi_regions=roi_regions,
                train=True,
            )
            if out.get("nan"):
                nan_seen = True
                die("nan_loss_detected")
            loss = out["loss"]
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                list(student_raw.parameters()) + list(roi_proj.parameters()) + list(stage_adapters.parameters()),
                grad_clip,
            ))
            grad_norm_last = grad_norm
            opt.step()

            train_correct += int(out["correct"])
            train_n += 1
            for k in ("loss_ce", "loss_kl", "loss_stage", "loss_roi", "loss_temporal"):
                sums[k] += float(out[k])
            sums["loss_total"] += float(loss.detach().cpu())
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"E2_SCREEN step={global_step} epoch={epoch+1}/{max_epochs} "
                    f"loss={float(loss.detach().cpu()):.4f} acc={train_correct/max(train_n,1):.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm:.3f}",
                    flush=True,
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"E2_SCREEN max_steps={args.max_steps} reached", flush=True)
                break

        train_acc = train_correct / max(train_n, 1)
        train_loss = sums["loss_total"] / max(train_n, 1)

        # monitor val (200 samples)
        student_raw.eval()
        roi_proj.eval()
        val_correct = 0
        val_n = 0
        val_ce = 0.0
        for sid in val_ids:
            row = meta[sid]
            label = label_for(sid)
            signer = str(row.get("signer_id") or row.get("signer") or "")
            video_path = Path(resolver.resolve(row))
            sealed_path = skel_root / f"{sid}.npz"
            out = process_sample(
                sid=sid,
                row=row,
                label=label,
                signer=signer,
                video_path=video_path,
                sealed_path=sealed_path,
                teacher_amp=teacher_amp,
                teacher_raw=teacher_raw,
                student_amp=student_amp,
                roi_proj=roi_proj,
                stage_adapters=stage_adapters,
                b4_head=b4_head,
                device=device,
                lambdas=lambdas,
                kl_temperature=kl_temperature,
                roi_regions=roi_regions,
                train=False,
            )
            val_correct += int(out["correct"])
            val_n += 1
            val_ce += float(out["loss_ce"])
        val_acc = val_correct / max(val_n, 1)
        val_ce_mean = val_ce / max(val_n, 1)

        epoch_row = {
            "schema": "ms_sta_v6_e4_b4sc_v1_e2_screen_train_epoch_v1",
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "loss_ce": sums["loss_ce"] / max(train_n, 1),
            "loss_kl": sums["loss_kl"] / max(train_n, 1),
            "loss_stage": sums["loss_stage"] / max(train_n, 1),
            "loss_roi": sums["loss_roi"] / max(train_n, 1),
            "loss_temporal": sums["loss_temporal"] / max(train_n, 1),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_acc_monitor_only": True,
            "val_n": val_n,
            "val_ce": val_ce_mean,
            "grad_norm": grad_norm_last,
            "nan_detected": nan_seen,
            "global_step": global_step,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        append_jsonl(LOG_PATH, epoch_row)
        print(
            f"E2_SCREEN epoch={epoch+1} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc_monitor={val_acc:.4f} (n={val_n})",
            flush=True,
        )

        if val_acc > best_val_top1:
            best_val_top1 = val_acc
            bad_epochs = 0
            save_checkpoint(
                BEST_CKPT,
                student=student_raw,
                roi_proj=roi_proj,
                stage_adapters=stage_adapters,
                epoch=epoch + 1,
                val_top1=val_acc,
                meta=run_meta,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"E2_SCREEN early_stop patience={patience} best_val_top1={best_val_top1:.4f}", flush=True)
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    print(
        "E4_B4SC_E2_SCREEN_OK",
        f"best_val_top1_monitor={best_val_top1:.4f}",
        f"ckpt={BEST_CKPT.relative_to(ROOT)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
