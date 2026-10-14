# -*- coding: utf-8 -*-
"""E4_B4SC_V2 C0→C1→C2 instrument-validity controls (create-only artifacts)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import sealed_landmarks  # noqa: E402
from scripts.e4_b4sc_train_e2_screen import (  # noqa: E402
    AmpRoiFp32SlowFastTrainable,
    build_b4_batch_from_projected,
    head_logits_eval,
    load_meta,
    stratified_sample_ids,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    REGIONS,
    PathwayMaps,
    FrozenSlowFastBlock4,
    extract_rgb_token_v3_for_video,
    read_video_frames_native,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, maybe_autocast  # noqa: E402
from src.optimization.e4_b4sc.distill_losses import stage_feature_loss  # noqa: E402
from src.optimization.e4_b4sc.narrow_slowfast import StageKDAdapters, TEACHER_SLOW_WIDTHS  # noqa: E402
from src.optimization.e4_b4sc_v2.factory import (  # noqa: E402
    assert_identity_plan,
    build_v2_slowfast,
    compare_hash_maps,
    identity_channel_plan,
    module_param_buffer_hashes,
)
from src.training.ms_sta_v5_data import load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POL = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V2_POLICY.json"
AMEND = OUT / "E4_B4SC_V2_CONTROLS_THRESHOLD_AMENDMENT.json"
C2_START = OUT / "E4_B4SC_V2_C2_START_STATE_PROCEDURE_AMENDMENT.json"
STATUS = OUT / "E4_B4SC_V2_CAMPAIGN_STATUS.json"
ART = {
    "probe_ids": OUT / "E4_B4SC_V2_C0_TRAIN_PROBE_IDS.json",
    "val_ids": OUT / "E4_B4SC_V2_C0_VAL_PARITY_IDS.json",
    "c0": OUT / "E4_B4SC_V2_C0_REPORT.json",
    "c1": OUT / "E4_B4SC_V2_C1_REPORT.json",
    "c2": OUT / "E4_B4SC_V2_C2_REPORT.json",
    "c2_micro": OUT / "E4_B4SC_V2_C2_MICROSET_IDS.json",
    "terminal": OUT / "E4_B4SC_V2_CONTROLS_TERMINAL.json",
    "identity_plan": OUT / "E4_B4SC_V2_C1_IDENTITY_PLAN.json",
}
C0_C1_SKIP_KEYS = ("probe_ids", "val_ids", "c0", "c1", "identity_plan", "c2_micro")


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_V2_CONTROLS_FAIL:{msg}")


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(str(obj), encoding="utf-8")


def ensure_train_cuda() -> None:
    venv_py = ROOT / "_venvs/train_cuda/Scripts/python.exe"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        os.execv(str(venv_py), [str(venv_py), *sys.argv])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def wrap_amp(base: nn.Module) -> AmpRoiFp32SlowFastTrainable:
    return AmpRoiFp32SlowFastTrainable(base)


def extract_tokens(extractor: nn.Module, sid: str, row: dict, resolver: RawVideoResolver, skel_root: Path):
    vpath = resolver.resolve(row)
    sealed = skel_root / f"{sid}.npz"
    if vpath is None or not sealed.is_file():
        die(f"resolve:{sid}")
    frames, frame_idx, fps = read_video_frames_native(str(vpath))
    landmarks_by_frame, missing_by_frame, sfi, sts, sealed_obj = sealed_landmarks(sealed)
    extract = extract_rgb_token_v3_for_video(
        video_path=str(vpath),
        extractor=extractor,
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
    )
    return extract, sealed_obj, frames, frame_idx, fps, landmarks_by_frame, missing_by_frame, sfi, sts


def forward_stages_one_window(
    model: nn.Module,
    frames_u8_meta: tuple,
    device: torch.device,
):
    """Unused placeholder — stages collected via extract path maps."""
    raise NotImplementedError


@torch.no_grad()
def probe_pair_metrics(
    *,
    teacher_amp: nn.Module,
    student_amp: nn.Module,
    teacher_raw: FrozenSlowFastBlock4,
    student_raw: nn.Module,
    ids: list[str],
    meta: dict,
    resolver: RawVideoResolver,
    skel_root: Path,
    device: torch.device,
    b4_head: nn.Module,
    g2i: dict,
) -> dict[str, Any]:
    from src.models.rgb_token_v3 import (
        INPUT_SIZE,
        WINDOW_SIZE,
        WINDOW_STRIDE,
        _resize_letterbox,
        window_starts,
        window_tensors,
    )

    logit_abs = []
    logit_cos = []
    stage_cos = [[] for _ in range(4)]
    roi_cos = {r: [] for r in REGIONS}
    fast_abs = []
    preds = []

    for sid in ids:
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        y = int(g2i[gloss])
        signer = str(row.get("signer_id") or "")
        tea_ex, sealed, frames, frame_idx, fps, lbf, mbf, sfi, sts = extract_tokens(
            teacher_amp, sid, row, resolver, skel_root
        )
        stu_ex, _, _, _, _, _, _, _, _ = extract_tokens(student_amp, sid, row, resolver, skel_root)

        # ROI tokens
        for r in REGIONS:
            t = torch.from_numpy(np.asarray(tea_ex["tokens"][r], dtype=np.float32))
            s = torch.from_numpy(np.asarray(stu_ex["tokens"][r], dtype=np.float32))
            L = min(t.shape[0], s.shape[0])
            roi_cos[r].append(cosine(t[:L], s[:L]))

        # logits via teacher tokens (both should match if extract identical)
        def logits_from_extract(ex):
            tokens = {r: torch.from_numpy(np.asarray(ex["tokens"][r], dtype=np.float32)).to(device) for r in REGIONS}
            batch = build_b4_batch_from_projected(
                sid=sid, label=y, signer=signer, proj_tokens=tokens, sealed=sealed, device=device
            )
            return head_logits_eval(b4_head, batch)

        lt = logits_from_extract(tea_ex)
        ls = logits_from_extract(stu_ex)
        logit_abs.append(float((lt - ls).abs().max().item()))
        logit_cos.append(cosine(lt, ls))
        preds.append({"sample_id": sid, "y_true": y, "y_pred_teacher": int(lt.argmax(-1)), "y_pred_student": int(ls.argmax(-1))})

        # one-window stage/fast compare
        resized = []
        for fr in frames:
            canvas, _ = _resize_letterbox(fr, INPUT_SIZE)
            resized.append(canvas)
        frames_u8 = np.stack(resized, axis=0)
        starts = window_starts(frames_u8.shape[0], window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE)
        start = starts[0]
        slow_t, fast_t, _, _ = window_tensors(frames_u8, start, WINDOW_SIZE)
        slow_t = slow_t.to(device)
        fast_t = fast_t.to(device)
        # Structural Fast/stage compare in FP32 only (identical precision for both paths).
        with torch.cuda.amp.autocast(enabled=False):
            x: Any = [slow_t.float(), fast_t.float()]
            tea_stages = []
            for i, block in enumerate(teacher_raw.blocks):
                x = block(x)
                if i >= 1 and isinstance(x, (list, tuple)):
                    tea_stages.append(x[0][:, : TEACHER_SLOW_WIDTHS[i - 1]].float())
            tea_maps = PathwayMaps(slow=x[0][:, :2048].float(), fast=x[1].float())
            smaps, sstages = student_raw.forward_maps_with_stages(slow_t.float(), fast_t.float())
            smaps = PathwayMaps(slow=smaps.slow.float(), fast=smaps.fast.float())
            sstages = [s.float() for s in sstages]
        fast_abs.append(float((tea_maps.fast - smaps.fast).abs().max().item()))
        for si in range(4):
            stage_cos[si].append(cosine(sstages[si], tea_stages[si]))

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n": len(ids),
        "logit_max_abs_diff_max": float(np.max(logit_abs)) if logit_abs else float("nan"),
        "logit_cosine_mean": mean(logit_cos),
        "fast_max_abs_diff_max": float(np.max(fast_abs)) if fast_abs else float("nan"),
        "stage_cosine_mean": [mean(stage_cos[i]) for i in range(4)],
        "roi_cosine_mean": {r: mean(roi_cos[r]) for r in REGIONS},
        "pred_rows": preds,
    }


def val_pred_parity(
    *,
    student_amp: nn.Module,
    ids: list[str],
    e0_by: dict[str, dict],
    meta: dict,
    resolver: RawVideoResolver,
    skel_root: Path,
    device: torch.device,
    b4_head: nn.Module,
    g2i: dict,
) -> dict[str, Any]:
    match = 0
    n = 0
    mismatches = []
    for i, sid in enumerate(ids):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        y = int(g2i[gloss])
        signer = str(row.get("signer_id") or "")
        ex, sealed, *_ = extract_tokens(student_amp, sid, row, resolver, skel_root)
        tokens = {r: torch.from_numpy(np.asarray(ex["tokens"][r], dtype=np.float32)).to(device) for r in REGIONS}
        batch = build_b4_batch_from_projected(
            sid=sid, label=y, signer=signer, proj_tokens=tokens, sealed=sealed, device=device
        )
        with torch.no_grad():
            logits = head_logits_eval(b4_head, batch)
            pred = int(logits.argmax(-1).item())
        e0p = int(e0_by[sid]["y_pred"])
        ok = pred == e0p
        match += int(ok)
        n += 1
        if not ok and len(mismatches) < 20:
            mismatches.append({"sample_id": sid, "y_pred": pred, "e0_pred": e0p, "y_true": y})
        if (i + 1) % 50 == 0:
            print(f"VAL_PARITY {i+1}/{len(ids)}", flush=True)
    return {"n": n, "match_rate": match / max(n, 1), "n_match": match, "mismatches_head": mismatches}


def freeze_bn(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def apply_locked_slow_perturbation(
    student: nn.Module,
    *,
    seed: int,
    gaussian_std: float,
    name_substring: str = ".multipathway_blocks.0.",
) -> dict[str, Any]:
    """Locked Slow-only Gaussian noise for C2 start-state (procedure amendment)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    n_tensors = 0
    n_elems = 0
    l2_sq = 0.0
    with torch.no_grad():
        for name, p in student.named_parameters():
            if name_substring not in name:
                continue
            noise = torch.randn(p.shape, generator=rng, dtype=torch.float32) * float(gaussian_std)
            noise = noise.to(device=p.device, dtype=p.dtype)
            p.add_(noise)
            n_tensors += 1
            n_elems += int(p.numel())
            l2_sq += float(noise.detach().float().pow(2).sum().item())
    return {
        "seed": int(seed),
        "gaussian_std": float(gaussian_std),
        "name_substring": name_substring,
        "n_tensors": n_tensors,
        "n_elems": n_elems,
        "noise_l2": float(l2_sq ** 0.5),
    }


def run_c2(
    *,
    teacher_raw: FrozenSlowFastBlock4,
    student_raw: nn.Module,
    ids: list[str],
    meta: dict,
    resolver: RawVideoResolver,
    skel_root: Path,
    device: torch.device,
    th: dict[str, Any],
    start_state: dict[str, Any],
) -> dict[str, Any]:
    from src.models.rgb_token_v3 import (
        INPUT_SIZE,
        WINDOW_SIZE,
        WINDOW_STRIDE,
        _resize_letterbox,
        window_starts,
        window_tensors,
    )

    student_raw.train()
    for name, p in student_raw.named_parameters():
        if ".multipathway_blocks.0." in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    freeze_bn(student_raw)

    pert = start_state["slow_only_parameter_perturbation"]
    if not bool(pert.get("enabled", False)):
        die("c2_start_state_perturbation_required")
    pert_meta = apply_locked_slow_perturbation(
        student_raw,
        seed=int(pert["seed"]),
        gaussian_std=float(pert["gaussian_std"]),
        name_substring=str(pert.get("apply_to_name_substring", ".multipathway_blocks.0.")),
    )

    adapters = StageKDAdapters(TEACHER_SLOW_WIDTHS, TEACHER_SLOW_WIDTHS).to(device)
    adapters.train()
    opt = torch.optim.AdamW(
        list(p for p in student_raw.parameters() if p.requires_grad) + list(adapters.parameters()),
        lr=1e-3,
    )
    adapter_in_opt = any(id(p) in {id(x) for g in opt.param_groups for x in g["params"]} for p in adapters.parameters())

    # cache stages for first window of each micro id
    cache = []
    for sid in ids:
        row = meta[sid]
        vpath = resolver.resolve(row)
        sealed = skel_root / f"{sid}.npz"
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        resized = []
        for fr in frames:
            canvas, _ = _resize_letterbox(fr, INPUT_SIZE)
            resized.append(canvas)
        frames_u8 = np.stack(resized, axis=0)
        starts = window_starts(frames_u8.shape[0], window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE)
        slow_t, fast_t, _, _ = window_tensors(frames_u8, starts[0], WINDOW_SIZE)
        slow_t = slow_t.to(device)
        fast_t = fast_t.to(device)
        with torch.no_grad():
            x: Any = [slow_t, fast_t]
            tea_stages = []
            for i, block in enumerate(teacher_raw.blocks):
                x = block(x)
                if i >= 1 and isinstance(x, (list, tuple)):
                    tea_stages.append(x[0][:, : TEACHER_SLOW_WIDTHS[i - 1]].float().detach())
        cache.append({"slow": slow_t, "fast": fast_t, "tea_stages": tea_stages})

    def mean_stage_cosine(stu_stages, tea_stages):
        return float(np.mean([cosine(s.detach(), t) for s, t in zip(stu_stages, tea_stages)]))

    def per_stage_cosines(stu_stages, tea_stages):
        return [cosine(s.detach(), t) for s, t in zip(stu_stages, tea_stages)]

    # before = AFTER locked Slow perturbation
    with torch.no_grad():
        cos_before = []
        cos_before_per_stage = []
        for item in cache:
            _, stu_stages = student_raw.forward_maps_with_stages(item["slow"], item["fast"])
            adapted = adapters([s.float() for s in stu_stages])
            cos_before.append(mean_stage_cosine(adapted, item["tea_stages"]))
            cos_before_per_stage.append(per_stage_cosines(adapted, item["tea_stages"]))
        cos_before_mean = float(np.mean(cos_before))
        cos_before_stages_mean = [float(np.mean([row[i] for row in cos_before_per_stage])) for i in range(4)]

    fast_before = {
        n: p.detach().cpu().clone()
        for n, p in student_raw.named_parameters()
        if ".multipathway_blocks.1." in n
    }
    slow_before = {
        n: p.detach().cpu().clone()
        for n, p in student_raw.named_parameters()
        if ".multipathway_blocks.0." in n
    }

    loss0 = None
    loss_last = None
    grad_adapter = 0.0
    grad_slow = 0.0
    history = []
    max_steps = int(th["microset"]["max_steps"])
    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        total = student_raw.blocks[0].parameters().__next__().new_zeros(())
        cos_step = []
        for item in cache:
            maps, stu_stages = student_raw.forward_maps_with_stages(item["slow"], item["fast"])
            stu_stages = [s.float() for s in stu_stages]
            adapted = adapters(stu_stages)
            loss = stage_feature_loss(adapted, item["tea_stages"])
            total = total + loss
            cos_step.append(mean_stage_cosine(adapted, item["tea_stages"]))
        total = total / max(len(cache), 1)
        if step == 0:
            loss0 = float(total.detach().cpu())
        total.backward()
        g_ad = 0.0
        for p in adapters.parameters():
            if p.grad is not None:
                g_ad += float(p.grad.detach().norm().item())
        g_sl = 0.0
        for n, p in student_raw.named_parameters():
            if p.grad is not None and ".multipathway_blocks.0." in n:
                g_sl += float(p.grad.detach().norm().item())
        grad_adapter = max(grad_adapter, g_ad)
        grad_slow = max(grad_slow, g_sl)
        opt.step()
        loss_last = float(total.detach().cpu())
        history.append({"step": step, "loss": loss_last, "cosine_mean": float(np.mean(cos_step))})
        if (step + 1) % 10 == 0:
            print(f"C2 step={step+1} loss={loss_last:.6f} cos={history[-1]['cosine_mean']:.4f}", flush=True)

    with torch.no_grad():
        cos_after = []
        cos_after_per_stage = []
        for item in cache:
            _, stu_stages = student_raw.forward_maps_with_stages(item["slow"], item["fast"])
            adapted = adapters([s.float() for s in stu_stages])
            cos_after.append(mean_stage_cosine(adapted, item["tea_stages"]))
            cos_after_per_stage.append(per_stage_cosines(adapted, item["tea_stages"]))
        cos_after_mean = float(np.mean(cos_after))
        cos_after_stages_mean = [float(np.mean([row[i] for row in cos_after_per_stage])) for i in range(4)]

    fast_delta = 0.0
    for n, p in student_raw.named_parameters():
        if n in fast_before:
            fast_delta = max(fast_delta, float((p.detach().cpu() - fast_before[n]).abs().max().item()))
    slow_delta = 0.0
    for n, p in student_raw.named_parameters():
        if n in slow_before:
            slow_delta = max(slow_delta, float((p.detach().cpu() - slow_before[n]).abs().max().item()))

    rel_red = (loss0 - loss_last) / max(abs(loss0), 1e-12) if loss0 is not None and loss_last is not None else float("nan")
    pass_c2 = (
        adapter_in_opt
        and grad_adapter >= float(th["adapter_grad_l2_min"])
        and grad_slow >= float(th["slow_student_grad_l2_min"])
        and fast_delta <= float(th["fast_param_max_abs_delta"])
        and rel_red >= float(th["kd_stage_loss_reduction_min_rel"])
        and loss_last < loss0
        and (cos_after_mean - cos_before_mean) >= float(th["stage_cosine_mean_increase_min"])
        and cos_after_mean >= float(th["final_mean_stage_cosine_min_absolute"])
    )
    return {
        "adapter_in_optimizer": adapter_in_opt,
        "teacher_eval": True,
        "bn_frozen": True,
        "start_state_amendment_id": "V2_C2_START_STATE_SLOW_PERTURB_V1",
        "slow_perturbation": pert_meta,
        "grad_adapter_l2_max": grad_adapter,
        "grad_slow_l2_max": grad_slow,
        "fast_param_max_abs_delta": fast_delta,
        "slow_param_max_abs_delta": slow_delta,
        "loss0": loss0,
        "loss_last": loss_last,
        "loss_reduction_rel": rel_red,
        "cosine_before": cos_before_mean,
        "cosine_after": cos_after_mean,
        "cosine_increase": cos_after_mean - cos_before_mean,
        "cosine_before_per_stage_mean": cos_before_stages_mean,
        "cosine_after_per_stage_mean": cos_after_stages_mean,
        "history_tail": history[-5:],
        "pass": pass_c2,
    }


def main() -> int:
    ensure_train_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--resume-from",
        choices=["C0", "C2"],
        default="C0",
        help="C2: reuse existing C0/C1 PASS artifacts; recreate only C2+terminal",
    )
    args = ap.parse_args()
    resume_c2 = args.resume_from == "C2"

    if resume_c2:
        for k in C0_C1_SKIP_KEYS:
            if not ART[k].is_file():
                die(f"resume_missing:{ART[k]}")
        c0_prev = json.loads(ART["c0"].read_text(encoding="utf-8"))
        c1_prev = json.loads(ART["c1"].read_text(encoding="utf-8"))
        if not c0_prev.get("pass") or not c1_prev.get("pass"):
            die("resume_requires_c0_c1_pass")
        for k in ("c2", "terminal"):
            if ART[k].exists():
                die(f"exists:{ART[k]} (archive FAIL before resume)")
    else:
        for p in ART.values():
            if p.exists():
                die(f"exists:{p}")

    if not AMEND.is_file() or not POL.is_file() or not C2_START.is_file():
        die("missing_policy_or_amendment")
    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    c2_start_doc = json.loads(C2_START.read_text(encoding="utf-8"))
    pol = json.loads(POL.read_text(encoding="utf-8"))
    th0 = amend["C0_exact_clone"]
    th1 = amend["C1_identity_transplant"]
    th2 = amend["C2_kd_mechanics"]
    c2_start = c2_start_doc["C2_start_state"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("cuda_required")
    gpu = torch.cuda.get_device_name(0)
    if gpu != pol["scientific_environment_lock"]["required_gpu_name"]:
        die(f"gpu:{gpu}")

    meta = load_meta()
    labels = json.loads((ROOT / pol["data"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels["gloss_to_id"]
    split = json.loads((ROOT / pol["data"]["split_manifest_rel"]).read_text(encoding="utf-8"))
    train_ids_all = load_split_role_ids(split, pol["data"]["train_role"])
    val_ids_all = load_split_role_ids(split, pol["data"]["val_role"])
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    resolver = RawVideoResolver(ROOT)
    skel_root = ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3"

    if resume_c2:
        probe_ids = json.loads(ART["probe_ids"].read_text(encoding="utf-8"))["sample_ids"]
        val_ids = json.loads(ART["val_ids"].read_text(encoding="utf-8"))["sample_ids"]
        print("RESUME_FROM_C2: skipping C0/C1 (PASS artifacts reused)", flush=True)
        c0_pass = True
        c1_pass = True
    else:
        probe_ids = stratified_sample_ids(
            train_ids_all, meta, int(th0["train_only_probe"]["n"]), int(th0["train_only_probe"]["seed"]), signer_field="signer_id"
        )
        if set(probe_ids) & test_ids:
            die("probe_test_leak")
        create_only(
            ART["probe_ids"],
            {
                "schema": "ms_sta_v6_e4_b4sc_v2_c0_train_probe_ids_v1",
                "n": len(probe_ids),
                "seed": th0["train_only_probe"]["seed"],
                "sample_ids": probe_ids,
                "sha256": hashlib.sha256(json.dumps(probe_ids, separators=(",", ":")).encode()).hexdigest(),
            },
        )
        val_ids = stratified_sample_ids(
            val_ids_all, meta, int(th0["val_prediction_parity"]["n"]), int(th0["val_prediction_parity"]["seed"]), signer_field="signer_id"
        )
        create_only(
            ART["val_ids"],
            {
                "schema": "ms_sta_v6_e4_b4sc_v2_c0_val_parity_ids_v1",
                "n": len(val_ids),
                "seed": th0["val_prediction_parity"]["seed"],
                "sample_ids": val_ids,
                "sha256": hashlib.sha256(json.dumps(val_ids, separators=(",", ":")).encode()).hexdigest(),
            },
        )

    e0_rows = [
        json.loads(l)
        for l in (ROOT / th0["val_prediction_parity"]["e0_predictions_rel"]).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    e0_by = {r["sample_id"]: r for r in e0_rows}

    teacher_raw = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["checkpoints"]["slowfast_rel"],
        expected_sha256=pol["checkpoints"]["slowfast_sha256"],
        device=device,
        repo_root=ROOT,
    )
    teacher_amp = AmpRoiFp32SlowFast(teacher_raw, channels_last=False)
    teacher_hash = module_param_buffer_hashes(teacher_raw.blocks)

    b4_head = build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt = torch.load(ROOT / pol["checkpoints"]["b4_teacher_rel"], map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    terminal = None

    if not resume_c2:
        # ----- C0 / C1 -----
        print("C0 exact clone", flush=True)
        c0_raw = build_v2_slowfast(teacher_raw, "C0_EXACT", device=device)
        c0_amp = wrap_amp(c0_raw)
        c0_hash = module_param_buffer_hashes(c0_raw.blocks)
        hash_cmp = compare_hash_maps(teacher_hash, c0_hash)
        probe = probe_pair_metrics(
            teacher_amp=teacher_amp,
            student_amp=c0_amp,
            teacher_raw=teacher_raw,
            student_raw=c0_raw,
            ids=probe_ids,
            meta=meta,
            resolver=resolver,
            skel_root=skel_root,
            device=device,
            b4_head=b4_head,
            g2i=g2i,
        )
        print("C0 val prediction parity", flush=True)
        valp = val_pred_parity(
            student_amp=c0_amp,
            ids=val_ids,
            e0_by=e0_by,
            meta=meta,
            resolver=resolver,
            skel_root=skel_root,
            device=device,
            b4_head=b4_head,
            g2i=g2i,
        )
        c0_pass = (
            hash_cmp["pass"]
            and probe["logit_max_abs_diff_max"] <= float(th0["logit_max_abs_diff"])
            and probe["logit_cosine_mean"] >= float(th0["logit_cosine_min"])
            and all(c >= float(th0["stage_feature_cosine_min"]) for c in probe["stage_cosine_mean"])
            and all(v >= float(th0["roi_token_cosine_min_per_region"]) for v in probe["roi_cosine_mean"].values())
            and probe["fast_max_abs_diff_max"] <= float(th0["fast_pathway_max_abs_diff"])
            and valp["match_rate"] == 1.0
        )
        c0_report = {
            "schema": "ms_sta_v6_e4_b4sc_v2_c0_report_v1",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_head": git_head,
            "amendment_sha256": sha256_file(AMEND),
            "hash_compare": {k: hash_cmp[k] for k in hash_cmp if k != "params"},
            "hash_compare_pass": hash_cmp["pass"],
            "probe": {k: v for k, v in probe.items() if k != "pred_rows"},
            "val_prediction_parity": valp,
            "pass": c0_pass,
            "e4_ready": False,
        }
        create_only(ART["c0"], c0_report)
        print("C0", "PASS" if c0_pass else "FAIL", flush=True)
        if not c0_pass:
            terminal = "V2_C0_CODEPATH_NO_GO"
            create_only(
                ART["terminal"],
                {
                    "schema": "ms_sta_v6_e4_b4sc_v2_controls_terminal_v1",
                    "terminal_state": terminal,
                    "C0": False,
                    "C1": None,
                    "C2": None,
                    "e4_ready": False,
                    "S5_ready": False,
                },
            )
            return 2

        # ----- C1 -----
        print("C1 identity transplant", flush=True)
        plan = identity_channel_plan()
        id_issues = assert_identity_plan(plan)
        create_only(ART["identity_plan"], plan)
        c1_raw = build_v2_slowfast(teacher_raw, "C1_IDENTITY", device=device)
        c1_amp = wrap_amp(c1_raw)
        c1_hash = module_param_buffer_hashes(c1_raw.blocks)
        hash_c1 = compare_hash_maps(c0_hash, c1_hash)
        probe1 = probe_pair_metrics(
            teacher_amp=c0_amp,
            student_amp=c1_amp,
            teacher_raw=c0_raw,  # compare stages vs C0 clone
            student_raw=c1_raw,
            ids=probe_ids,
            meta=meta,
            resolver=resolver,
            skel_root=skel_root,
            device=device,
            b4_head=b4_head,
            g2i=g2i,
        )
        # For C1, teacher_raw.forward in probe uses c0_raw which has forward_maps_with_stages — but probe uses teacher_raw.blocks manually for teacher_raw path.
        # Fix: probe_pair_metrics uses teacher_raw.blocks — ExactWidthClone has .blocks too. Good.
        valp1 = val_pred_parity(
            student_amp=c1_amp,
            ids=val_ids,
            e0_by=e0_by,
            meta=meta,
            resolver=resolver,
            skel_root=skel_root,
            device=device,
            b4_head=b4_head,
            g2i=g2i,
        )
        c1_pass = (
            not id_issues
            and hash_c1["pass"]
            and probe1["logit_max_abs_diff_max"] <= float(th0["logit_max_abs_diff"])
            and probe1["logit_cosine_mean"] >= float(th0["logit_cosine_min"])
            and all(c >= float(th0["stage_feature_cosine_min"]) for c in probe1["stage_cosine_mean"])
            and all(v >= float(th0["roi_token_cosine_min_per_region"]) for v in probe1["roi_cosine_mean"].values())
            and probe1["fast_max_abs_diff_max"] <= float(th0["fast_pathway_max_abs_diff"])
            and valp1["match_rate"] == 1.0
        )
        c1_report = {
            "schema": "ms_sta_v6_e4_b4sc_v2_c1_report_v1",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_head": git_head,
            "identity_issues": id_issues,
            "index_hashes": {str(si): plan["stages"][str(si)]["out_idx_sha256"] for si in range(4)},
            "hash_compare_vs_C0_pass": hash_c1["pass"],
            "hash_compare_vs_C0": {k: hash_c1[k] for k in ("pass", "mismatch_params", "mismatch_buffers", "params_keys_equal", "buffers_keys_equal")},
            "probe_vs_C0": {k: v for k, v in probe1.items() if k != "pred_rows"},
            "val_prediction_parity": valp1,
            "pass": c1_pass,
            "e4_ready": False,
        }
        create_only(ART["c1"], c1_report)
        print("C1", "PASS" if c1_pass else "FAIL", flush=True)
        if not c1_pass:
            terminal = "V2_C1_TRANSPLANT_NO_GO"
            create_only(
                ART["terminal"],
                {
                    "schema": "ms_sta_v6_e4_b4sc_v2_controls_terminal_v1",
                    "terminal_state": terminal,
                    "C0": True,
                    "C1": False,
                    "C2": None,
                    "e4_ready": False,
                    "S5_ready": False,
                },
            )
            return 2

    # ----- C2 -----
    print("C2 KD mechanics", flush=True)
    if resume_c2 and ART["c2_micro"].is_file():
        micro_ids = json.loads(ART["c2_micro"].read_text(encoding="utf-8"))["sample_ids"]
        print(f"RESUME_FROM_C2: reuse locked microset n={len(micro_ids)}", flush=True)
    else:
        micro_ids = stratified_sample_ids(
            train_ids_all, meta, int(th2["microset"]["n"]), int(th2["microset"]["seed"]), signer_field="signer_id"
        )
        if set(micro_ids) & set(val_ids_all):
            die("micro_val_leak")
        if set(micro_ids) & test_ids:
            die("micro_test_leak")
        create_only(
            ART["c2_micro"],
            {
                "schema": "ms_sta_v6_e4_b4sc_v2_c2_microset_ids_v1",
                "n": len(micro_ids),
                "seed": th2["microset"]["seed"],
                "sample_ids": micro_ids,
                "sha256": hashlib.sha256(json.dumps(micro_ids, separators=(",", ":")).encode()).hexdigest(),
                "forbid_val_test": True,
            },
        )
    # fresh identity student for C2 training dynamics (train mode; BN frozen inside run_c2)
    c2_student = build_v2_slowfast(
        teacher_raw, "C1_IDENTITY", device=device, requires_grad_slow=True, eval_mode=False
    )
    c2_out = run_c2(
        teacher_raw=teacher_raw,
        student_raw=c2_student,
        ids=micro_ids,
        meta=meta,
        resolver=resolver,
        skel_root=skel_root,
        device=device,
        th=th2,
        start_state=c2_start,
    )
    c2_pass = bool(c2_out["pass"])
    create_only(
        ART["c2"],
        {
            "schema": "ms_sta_v6_e4_b4sc_v2_c2_report_v1",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_head": git_head,
            "thresholds": th2,
            "start_state_amendment_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_C2_START_STATE_PROCEDURE_AMENDMENT.json",
            "start_state_amendment_id": c2_start_doc.get("amendment_id"),
            "result": c2_out,
            "pass": c2_pass,
            "e4_ready": False,
        },
    )
    print("C2", "PASS" if c2_pass else "FAIL", flush=True)

    def _write_status(term: str, c2_ok: bool) -> None:
        st = json.loads(STATUS.read_text(encoding="utf-8"))
        st["status"] = term
        st["controls"] = {
            "C0": "PASS",
            "C1": "PASS",
            "C2": "PASS" if c2_ok else "FAIL",
            "compression_blocked_until_all_pass": not c2_ok,
        }
        st["e4_ready"] = False
        st["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        st["history"].append(
            {
                "status": term,
                "updated_at": st["updated_at"],
                "e4_ready": False,
                "note": "C2 start-state procedure amendment applied" if resume_c2 else None,
            }
        )
        st["next"] = (
            ["open_V2_S5_internal_bottleneck_after_controls"]
            if c2_ok
            else ["fix_C2_instrument_or_stop_no_S5"]
        )
        STATUS.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not c2_pass:
        terminal = "V2_C2_KD_MECHANICS_NO_GO"
        create_only(
            ART["terminal"],
            {
                "schema": "ms_sta_v6_e4_b4sc_v2_controls_terminal_v1",
                "terminal_state": terminal,
                "C0": True,
                "C1": True,
                "C2": False,
                "e4_ready": False,
                "S5_ready": False,
            },
        )
        _write_status(terminal, False)
        return 2

    terminal = "V2_CONTROLS_PASS_S5_READY"
    create_only(
        ART["terminal"],
        {
            "schema": "ms_sta_v6_e4_b4sc_v2_controls_terminal_v1",
            "terminal_state": terminal,
            "C0": True,
            "C1": True,
            "C2": True,
            "e4_ready": False,
            "S5_ready": True,
            "forbid_open_S45_or_full_directly": True,
        },
    )
    _write_status(terminal, True)
    print("E4_B4SC_V2_CONTROLS_OK", terminal, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
