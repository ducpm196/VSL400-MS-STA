# -*- coding: utf-8 -*-
"""E4_B4SC_V1 E2 postmortem — create-only; does not change V1 verdict."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
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
    project_tokens,
    student_extract_with_teacher_stages,
    teacher_forward_maps_with_stages,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    REGIONS,
    FrozenSlowFastBlock4,
    extract_rgb_token_v3_for_video,
    read_video_frames_native,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, maybe_autocast  # noqa: E402
from src.optimization.e4_b4sc.narrow_slowfast import (  # noqa: E402
    E2_WIDTHS,
    FUSION_OUT,
    NarrowSlowFastBlock4,
    RoiToHeadProjection,
    StageKDAdapters,
    TEACHER_SLOW_WIDTHS,
    _get,
)
from src.training.ms_sta_v5_data import load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POST = OUT / "E4_B4SC_V1_E2_POSTMORTEM.json"
POL_TRAIN = OUT / "E4_B4SC_V1_E2E3_TRAINING_POLICY.json"
POL_E4 = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"
LOCK = OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json"
SUBSET = OUT / "E4_B4SC_V1_E2_SCREEN_TRAIN_SUBSET_IDS.json"
TRAIN_LOG = OUT / "E4_B4SC_V1_E2_SCREEN_TRAIN_LOG.jsonl"
ACC_REP = OUT / "E4_B4SC_V1_E2_SCREEN_ACCURACY_REPORT.json"
PRED_E2 = OUT / "E4_B4SC_V1_E2_SCREEN_E2_PREDICTIONS.jsonl"
PRED_E0 = OUT / "E4_B4SC_V1_E1_E0_PREDICTIONS.jsonl"
CLOSEOUT = OUT / "E4_B4SC_V1_E2E3_CLOSEOUT.json"
CKPT = OUT / "checkpoints/e2_screen_seed42/best.pt"

PROBE_N = 64  # feature / BN probes
# full train-subset accuracy is required by postmortem brief


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_E2_POSTMORTEM_FAIL:{msg}")


def create_only(path: Path, obj: dict) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_train_cuda() -> None:
    venv_py = ROOT / "_venvs/train_cuda/Scripts/python.exe"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        os.execv(str(venv_py), [str(venv_py), *sys.argv])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a.float(), b.float()).item())


def audit_channel_consistency(plan: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    notes: list[str] = []
    stages = plan["stages"]
    prev_out: list[int] | None = None
    for si in range(4):
        st = stages[str(si)]
        out_idx = list(st["out_idx"])
        inner_idx = list(st["inner_idx"])
        if len(out_idx) != int(st["student_out"]):
            issues.append(f"stage{si}_out_len")
        if len(set(out_idx)) != len(out_idx):
            issues.append(f"stage{si}_out_dup")
        if out_idx != sorted(out_idx):
            issues.append(f"stage{si}_out_unsorted")
        if any(i < 0 or i >= int(st["teacher_out"]) for i in out_idx):
            issues.append(f"stage{si}_out_oob")
        if len(inner_idx) != int(st["student_inner"]):
            issues.append(f"stage{si}_inner_len")
        if len(set(inner_idx)) != len(inner_idx):
            issues.append(f"stage{si}_inner_dup")
        # residual identity: same out_idx used for branch1 and branch2.conv_c by construction
        notes.append(f"stage{si}_residual_shares_out_idx_by_construction")
        # lateral: fusion kept in full in apply_channel_plan
        notes.append(f"stage{si}_lateral_fusion_channels_kept_full_{FUSION_OUT[si]}")
        if prev_out is not None:
            # next-stage Slow inputs are prev out_idx (teacher-space), not remapped independently
            if list(st.get("documented_prev_out_idx") or prev_out) and True:
                notes.append(f"stage{si}_input_slow_identity_from_stage{si-1}_out_idx")
        prev_out = out_idx
    # width drop fraction
    drops = [
        1.0 - int(stages[str(i)]["student_out"]) / float(TEACHER_SLOW_WIDTHS[i]) for i in range(4)
    ]
    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "notes": notes,
        "width_reduction_fraction_per_stage": drops,
        "uniform_output_width_keep_fraction": 0.875,
        "interpretation": (
            "Index arrays are internally consistent with the locked prune procedure; "
            "this does NOT prove preserved teacher channel semantics after residual/lateral composition."
        ),
    }


def collect_bn_stats(module: nn.Module, plan: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for si in range(4):
        block_i = int(plan["stages"][str(si)]["block_index"])
        stage = module.blocks[block_i].multipathway_blocks[0]
        bn = stage.res_blocks[-1].branch2.norm_c
        key = f"{prefix}_stage{si}_norm_c"
        out[key] = {
            "num_features": int(bn.num_features),
            "running_mean_l2": float(bn.running_mean.detach().float().norm().item()),
            "running_var_mean": float(bn.running_var.detach().float().mean().item()),
            "running_mean_sha16": sha256_tensor(bn.running_mean)[:16],
            "running_var_sha16": sha256_tensor(bn.running_var)[:16],
            "gamma_l1": float(bn.weight.detach().float().abs().sum().item()) if bn.affine else None,
        }
    return out


def sha256_tensor(t: torch.Tensor) -> str:
    import hashlib

    arr = t.detach().float().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def bn_drift(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    drift = {}
    for k in keys:
        if k not in a or k not in b:
            continue
        drift[k] = {
            "mean_l2_a": a[k]["running_mean_l2"],
            "mean_l2_b": b[k]["running_mean_l2"],
            "var_mean_a": a[k]["running_var_mean"],
            "var_mean_b": b[k]["running_var_mean"],
            "mean_sha_equal": a[k]["running_mean_sha16"] == b[k]["running_mean_sha16"],
            "var_sha_equal": a[k]["running_var_sha16"] == b[k]["running_var_sha16"],
        }
    return drift


def entropy_from_logits(logits: np.ndarray) -> float:
    x = logits.astype(np.float64)
    x = x - x.max()
    p = np.exp(x)
    p = p / max(p.sum(), 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def eval_accuracy_on_ids(
    *,
    ids: list[str],
    student_amp: AmpRoiFp32SlowFastTrainable,
    roi_proj: RoiToHeadProjection,
    b4_head: nn.Module,
    meta: dict,
    g2i: dict,
    resolver: RawVideoResolver,
    skel_root: Path,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    correct = 0
    n = 0
    nan = False
    for i, sid in enumerate(ids):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        vpath = resolver.resolve(row)
        sealed = skel_root / f"{sid}.npz"
        if vpath is None or not sealed.is_file() or gloss not in g2i:
            die(f"resolve:{label}:{sid}")
        y = int(g2i[gloss])
        signer = str(row.get("signer_id") or "")
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        landmarks_by_frame, missing_by_frame, sfi, sts, sealed_obj = sealed_landmarks(sealed)
        with torch.no_grad():
            extract = extract_rgb_token_v3_for_video(
                video_path=str(vpath),
                extractor=student_amp,
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
            tokens = {
                r: torch.from_numpy(np.asarray(extract["tokens"][r], dtype=np.float32)).to(device)
                for r in REGIONS
            }
            proj = project_tokens(roi_proj, tokens)
            batch = build_b4_batch_from_projected(
                sid=sid, label=y, signer=signer, proj_tokens=proj, sealed=sealed_obj, device=device
            )
            logits = head_logits_eval(b4_head, batch)
            if not bool(torch.isfinite(logits).all()):
                nan = True
            pred = int(logits.argmax(dim=-1).item())
        correct += int(pred == y)
        n += 1
        if (i + 1) % 100 == 0:
            print(f"POSTMORTEM_ACC[{label}] {i+1}/{len(ids)}", flush=True)
    return {"n": n, "top1": correct / max(n, 1), "nan_inf": nan, "label": label}


def feature_probe(
    *,
    ids: list[str],
    teacher_raw: FrozenSlowFastBlock4,
    student_amp: AmpRoiFp32SlowFastTrainable,
    student_raw: NarrowSlowFastBlock4,
    adapters: StageKDAdapters | None,
    roi_proj: RoiToHeadProjection | None,
    meta: dict,
    resolver: RawVideoResolver,
    skel_root: Path,
    device: torch.device,
    plan: dict[str, Any],
) -> dict[str, Any]:
    stage_cos = [[] for _ in range(4)]
    stage_mse = [[] for _ in range(4)]
    stage_cos_adapted = [[] for _ in range(4)] if adapters is not None else None
    roi_mse = {r: [] for r in ("global", "left_hand", "right_hand", "face")}
    roi_cos = {r: [] for r in ("global", "left_hand", "right_hand", "face")}

    teacher_amp = AmpRoiFp32SlowFast(teacher_raw, channels_last=False)

    for i, sid in enumerate(ids):
        row = meta[sid]
        vpath = resolver.resolve(row)
        sealed = skel_root / f"{sid}.npz"
        if vpath is None or not sealed.is_file():
            die(f"probe_resolve:{sid}")
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        landmarks_by_frame, missing_by_frame, sfi, sts, _sealed = sealed_landmarks(sealed)
        with torch.no_grad():
            # one window path via student_extract_with_teacher_stages for stages
            stu_tokens, stage_pairs = student_extract_with_teacher_stages(
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
            # average over windows
            for stu_stages, tea_stages in stage_pairs:
                for si, (s, t) in enumerate(zip(stu_stages, tea_stages)):
                    out_idx = torch.as_tensor(plan["stages"][str(si)]["out_idx"], device=t.device)
                    t_sel = t.index_select(1, out_idx)
                    if s.shape[1] != t_sel.shape[1]:
                        die(f"stage_shape:{si}:{tuple(s.shape)}vs{tuple(t_sel.shape)}")
                    stage_cos[si].append(cosine(s, t_sel))
                    stage_mse[si].append(mse(s, t_sel))
                if adapters is not None and stage_cos_adapted is not None:
                    adapted_list = adapters(stu_stages)
                    for si, (a, t) in enumerate(zip(adapted_list, tea_stages)):
                        stage_cos_adapted[si].append(cosine(a, t))

            # teacher ROI tokens (2304) vs student projected or raw
            tea_extract = extract_rgb_token_v3_for_video(
                video_path=str(vpath),
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
            )
            for r in roi_mse:
                tea = torch.from_numpy(np.asarray(tea_extract["tokens"][r], dtype=np.float32))
                if roi_proj is not None:
                    stu = project_tokens(roi_proj, {rr: stu_tokens[rr] for rr in REGIONS})[r].detach().cpu()
                else:
                    # compare Fast tail 256-d exactly; Slow via last-stage out_idx on concatenated token
                    # student token = [slow1792|fast256]; teacher = [slow2048|fast256]
                    stu_raw = stu_tokens[r].detach().cpu()
                    out_idx = np.asarray(plan["stages"]["3"]["out_idx"], dtype=np.int64)
                    # teacher slow part
                    t_slow = tea[..., :2048]
                    t_fast = tea[..., 2048:]
                    s_slow = stu_raw[..., :1792]
                    s_fast = stu_raw[..., 1792:]
                    t_slow_sel = t_slow[..., out_idx]
                    tea = torch.cat([t_slow_sel, t_fast], dim=-1)
                    stu = torch.cat([s_slow, s_fast], dim=-1)
                # align lengths
                L = min(int(tea.shape[0]), int(stu.shape[0]))
                tea_a = tea[:L].reshape(-1)
                stu_a = stu[:L].reshape(-1)
                if tea_a.numel() != stu_a.numel():
                    # dim mismatch for projected case should match 2304
                    die(f"roi_dim:{r}:{tea_a.numel()}vs{stu_a.numel()}")
                roi_mse[r].append(float(F.mse_loss(stu_a, tea_a).item()))
                roi_cos[r].append(float(F.cosine_similarity(stu_a.unsqueeze(0), tea_a.unsqueeze(0)).item()))
        if (i + 1) % 16 == 0:
            print(f"POSTMORTEM_FEAT {i+1}/{len(ids)}", flush=True)

    def mean_std(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        a = np.asarray(xs, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}

    out: dict[str, Any] = {
        "n_videos": len(ids),
        "stage_channel_aligned": {
            f"stage{si}": {"cosine": mean_std(stage_cos[si]), "mse": mean_std(stage_mse[si])}
            for si in range(4)
        },
        "roi_regions": {
            r: {"cosine": mean_std(roi_cos[r]), "mse": mean_std(roi_mse[r])} for r in roi_mse
        },
    }
    if stage_cos_adapted is not None:
        out["stage_after_train_only_adapters"] = {
            f"stage{si}": {"cosine": mean_std(stage_cos_adapted[si])} for si in range(4)
        }
    return out


def main() -> int:
    ensure_train_cuda()
    if POST.exists():
        die(f"exists:{POST}")

    gaps: list[str] = []
    for p in (POL_TRAIN, LOCK, SUBSET, TRAIN_LOG, ACC_REP, PRED_E2, PRED_E0, CLOSEOUT):
        if not p.is_file():
            gaps.append(f"missing:{p.name}")
    if gaps:
        die(";".join(gaps))

    tpol = json.loads(POL_TRAIN.read_text(encoding="utf-8"))
    pol = json.loads(POL_E4.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8"))
    acc_rep = json.loads(ACC_REP.read_text(encoding="utf-8"))
    train_epochs = load_jsonl(TRAIN_LOG)
    e2_rows = load_jsonl(PRED_E2)
    e0_rows = load_jsonl(PRED_E0)
    plan = lock["E2"]

    # --- verdict immutability ---
    if closeout.get("terminal_state") != "E2_ACCURACY_NO_GO_E3_NOT_OPENED":
        die("closeout_terminal_mismatch")

    # --- checkpoint retention ---
    ckpt_rel = "Docs/results/ms_sta_v6_e4_b4sc/checkpoints/e2_screen_seed42/best.pt"
    ckpt_abs = str((ROOT / ckpt_rel).resolve())
    retention: dict[str, Any]
    if CKPT.is_file():
        sz = CKPT.stat().st_size
        sha = sha256_file(CKPT)
        expected = closeout["screen_training"]["checkpoint_sha256"]
        retention = {
            "status": "PRESENT_ON_WORKTREE",
            "relative_path": ckpt_rel,
            "absolute_path": ckpt_abs,
            "size_bytes": sz,
            "sha256": sha,
            "sha256_matches_closeout": sha == expected,
            "gitignored": True,
            "durable_storage_registered": False,
            "risk": "ARTIFACT_RETENTION_GAP" if True else None,
            "risk_note": (
                "File is present locally but gitignored (**/best.pt). "
                "No durable remote/registry path is recorded — loss of this worktree "
                "would leave only SHA-256 without recoverable weights."
            ),
            "metadata": {
                "schema_expected": "ms_sta_v6_e4_b4sc_v1_e2_screen_checkpoint_v1",
                "arm": "E2",
                "seed": 42,
            },
        }
        payload = torch.load(CKPT, map_location="cpu", weights_only=False)
        retention["metadata"].update(
            {
                "epoch": payload.get("epoch"),
                "val_top1_monitor": payload.get("val_top1_monitor"),
                "git_head_in_ckpt": (payload.get("meta") or {}).get("git_head"),
                "keys": sorted(payload.keys()),
            }
        )
    else:
        retention = {
            "status": "MISSING",
            "relative_path": ckpt_rel,
            "absolute_path": ckpt_abs,
            "sha256_closeout_only": closeout["screen_training"]["checkpoint_sha256"],
            "gitignored": True,
            "durable_storage_registered": False,
            "risk": "ARTIFACT_RETENTION_GAP",
            "risk_note": "best.pt missing; SHA-256 alone cannot reproduce weights.",
        }
        die("best_pt_missing_required_for_postmortem_measurements")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("cuda_required")
    gpu = torch.cuda.get_device_name(0)
    if gpu != pol["scientific_environment_lock"]["required_gpu_name"]:
        die(f"gpu_mismatch:{gpu}")

    meta = load_meta()
    labels_obj = json.loads((ROOT / tpol["data"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    resolver = RawVideoResolver(ROOT)
    skel_root = ROOT / tpol["data"]["skel_rel"] / "skeleton_v3"
    train_ids = list(subset["sample_ids"])
    val_monitor_ids = list(subset["val_monitor_ids"])
    probe_ids = train_ids[:PROBE_N]

    channel_audit = audit_channel_consistency(plan)

    # models
    teacher_raw = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    transplant = NarrowSlowFastBlock4(teacher_raw, plan, device=device, requires_grad_slow=False)
    transplant_amp = AmpRoiFp32SlowFastTrainable(transplant)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    roi_init = RoiToHeadProjection(E2_WIDTHS.token_dim, out_dim=2304).to(device)
    adapters_init = StageKDAdapters(tuple(plan["slow_widths"]), TEACHER_SLOW_WIDTHS).to(device)

    trained = NarrowSlowFastBlock4(teacher_raw, plan, device=device, requires_grad_slow=False)
    trained_amp = AmpRoiFp32SlowFastTrainable(trained)
    roi_tr = RoiToHeadProjection(E2_WIDTHS.token_dim, out_dim=2304).to(device)
    adapters_tr = StageKDAdapters(tuple(plan["slow_widths"]), TEACHER_SLOW_WIDTHS).to(device)
    payload = torch.load(CKPT, map_location=device, weights_only=False)
    trained.load_state_dict(payload["student"])
    roi_tr.load_state_dict(payload["roi_to_head_projection"])
    adapters_tr.load_state_dict(payload["stage_kd_adapters_train_only"])
    trained.eval()
    roi_tr.eval()
    adapters_tr.eval()
    transplant.eval()
    roi_init.eval()

    b4_head = build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt = torch.load(ROOT / tpol["checkpoints"]["b4_teacher_rel"], map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()

    # BN: transplant (teacher-sliced) vs trained
    bn_tx = collect_bn_stats(transplant, plan, "transplant")
    bn_tr = collect_bn_stats(trained, plan, "trained")
    bn_keys = [f"transplant_stage{si}_norm_c" for si in range(4)]
    # remap keys for drift compare
    bn_tx_k = {f"stage{si}": bn_tx[f"transplant_stage{si}_norm_c"] for si in range(4)}
    bn_tr_k = {f"stage{si}": bn_tr[f"trained_stage{si}_norm_c"] for si in range(4)}
    bn_compare = {}
    for si in range(4):
        k = f"stage{si}"
        bn_compare[k] = {
            "transplant_mean_sha16": bn_tx_k[k]["running_mean_sha16"],
            "trained_mean_sha16": bn_tr_k[k]["running_mean_sha16"],
            "identical_to_transplant": bn_tx_k[k]["running_mean_sha16"] == bn_tr_k[k]["running_mean_sha16"]
            and bn_tx_k[k]["running_var_sha16"] == bn_tr_k[k]["running_var_sha16"],
            "transplant_var_mean": bn_tx_k[k]["running_var_mean"],
            "trained_var_mean": bn_tr_k[k]["running_var_mean"],
            "mean_l2_transplant": bn_tx_k[k]["running_mean_l2"],
            "mean_l2_trained": bn_tr_k[k]["running_mean_l2"],
        }

    print("POSTMORTEM feature probe (transplant, channel-aligned ROI)", flush=True)
    feat_tx = feature_probe(
        ids=probe_ids,
        teacher_raw=teacher_raw,
        student_amp=transplant_amp,
        student_raw=transplant,
        adapters=None,
        roi_proj=None,
        meta=meta,
        resolver=resolver,
        skel_root=skel_root,
        device=device,
        plan=plan,
    )
    print("POSTMORTEM feature probe (trained + adapters + roi_proj)", flush=True)
    feat_tr = feature_probe(
        ids=probe_ids,
        teacher_raw=teacher_raw,
        student_amp=trained_amp,
        student_raw=trained,
        adapters=adapters_tr,
        roi_proj=roi_tr,
        meta=meta,
        resolver=resolver,
        skel_root=skel_root,
        device=device,
        plan=plan,
    )

    print("POSTMORTEM transplant-init accuracy on val_monitor", flush=True)
    acc_tx_mon = eval_accuracy_on_ids(
        ids=val_monitor_ids,
        student_amp=transplant_amp,
        roi_proj=roi_init,
        b4_head=b4_head,
        meta=meta,
        g2i=g2i,
        resolver=resolver,
        skel_root=skel_root,
        device=device,
        label="transplant_init_val_monitor",
    )
    print("POSTMORTEM ablation transplant_slow+trained_proj on val_monitor", flush=True)
    acc_tx_trproj = eval_accuracy_on_ids(
        ids=val_monitor_ids,
        student_amp=transplant_amp,
        roi_proj=roi_tr,
        b4_head=b4_head,
        meta=meta,
        g2i=g2i,
        resolver=resolver,
        skel_root=skel_root,
        device=device,
        label="transplant_slow_trained_proj_val_monitor",
    )
    print("POSTMORTEM trained accuracy on full train_subset_2400", flush=True)
    acc_tr_train = eval_accuracy_on_ids(
        ids=train_ids,
        student_amp=trained_amp,
        roi_proj=roi_tr,
        b4_head=b4_head,
        meta=meta,
        g2i=g2i,
        resolver=resolver,
        skel_root=skel_root,
        device=device,
        label="trained_train_subset_2400",
    )

    # prediction-level diagnostics from locked val artifacts
    e0_by = {r["sample_id"]: r for r in e0_rows}
    e2_by = {r["sample_id"]: r for r in e2_rows}
    agree = 0
    ent_e2 = []
    logits_npz = OUT / "E4_B4SC_V1_E2_SCREEN_E2_LOGITS.npz"
    if logits_npz.is_file():
        z = np.load(logits_npz, allow_pickle=True)
        logits = z["logits"]
        lids = [str(x) for x in z["sample_ids"].tolist()]
        for i, sid in enumerate(lids):
            ent_e2.append(entropy_from_logits(logits[i]))
            if sid in e0_by and sid in e2_by and int(e2_by[sid]["y_pred"]) == int(e0_by[sid]["y_pred"]):
                agree += 1
        teacher_agree = agree / max(len(lids), 1)
    else:
        teacher_agree = float("nan")
        gaps.append("missing_logits_for_entropy")

    conf = Counter()
    for r in e2_rows:
        if int(r["y_pred"]) != int(r["y_true"]):
            conf[(int(r["y_true"]), int(r["y_pred"]))] += 1
    top_conf = [
        {"y_true": int(a), "y_pred": int(b), "n": int(n)}
        for (a, b), n in conf.most_common(15)
    ]
    # concentration: share of errors in top-15 pairs
    n_err = sum(conf.values())
    top15_share = sum(x["n"] for x in top_conf) / max(n_err, 1)

    # diagnosis rule from brief
    train_top1 = float(acc_tr_train["top1"])
    val_top1 = float(acc_rep["e2"]["top1"])
    if train_top1 < 0.70:
        train_val_pattern = "LOW_TRAIN_AND_LOW_VAL_suggests_optimization_or_transplant_defect"
    elif train_top1 - val_top1 >= 0.10:
        train_val_pattern = "HIGH_TRAIN_LOW_VAL_suggests_overfit_or_distribution_problem"
    else:
        train_val_pattern = "TRAIN_VAL_BOTH_BELOW_E0_mixed"

    # feature divergence before training?
    pretrain_stage_cos = [
        feat_tx["stage_channel_aligned"][f"stage{si}"]["cosine"]["mean"] for si in range(4)
    ]
    pretrain_divergent = any((not math.isfinite(c)) or c < 0.85 for c in pretrain_stage_cos)

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    report = {
        "schema": "ms_sta_v6_e4_b4sc_v1_e2_postmortem_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head,
        "v1_verdict_unchanged": True,
        "v1_terminal_state": "E2_ACCURACY_NO_GO_E3_NOT_OPENED",
        "e4_ready": False,
        "e3_opened": False,
        "no_post_hoc_tolerance": True,
        "rejected_claim_precise": (
            "Simultaneous Slow stage output-width narrowing to 87.5% with the locked "
            "channel transplant and KD only on 2400 samples × 3 epochs."
        ),
        "not_claimed": (
            "Structured Slow-pathway compression in general is infeasible."
        ),
        "checkpoint_retention": retention,
        "epoch_curves_from_train_log": train_epochs,
        "loss_components_by_epoch": [
            {
                "epoch": e["epoch"],
                "loss_ce": e["loss_ce"],
                "loss_kl": e["loss_kl"],
                "loss_stage": e["loss_stage"],
                "loss_roi": e["loss_roi"],
                "loss_temporal": e["loss_temporal"],
                "train_acc": e["train_acc"],
                "val_acc_monitor": e["val_acc"],
                "lr": e["lr"],
            }
            for e in train_epochs
        ],
        "accuracy": {
            "E0_full_val_top1": acc_rep["e0"]["top1"],
            "E2_full_val_top1": acc_rep["e2"]["top1"],
            "paired_delta_top1_pp": acc_rep["paired_delta_top1"]["mean_delta_top1_pp"],
            "transplant_init_val_monitor": acc_tx_mon,
            "transplant_init_note": (
                "RoiToHeadProjection re-initialized with seed=42; not byte-identical to "
                "the exact ephemeral init at training start (init weights were not checkpointed)."
            ),
            "ablation_transplant_slow_plus_trained_proj_val_monitor": acc_tx_trproj,
            "trained_train_subset_2400": acc_tr_train,
            "trained_full_val": {
                "n": acc_rep["n"],
                "top1": acc_rep["e2"]["top1"],
                "top5": acc_rep["e2"]["top5"],
                "macro_f1": acc_rep["e2"]["macro_f1"],
            },
            "train_val_pattern": train_val_pattern,
        },
        "feature_alignment": {
            "probe_n": PROBE_N,
            "probe_ids_sha256": __import__("hashlib")
            .sha256(json.dumps(probe_ids, separators=(",", ":")).encode())
            .hexdigest(),
            "pre_training_transplant_channel_aligned": feat_tx,
            "post_training_with_adapters_and_roi_proj": feat_tr,
            "pretrain_feature_divergence_flag": pretrain_divergent,
            "pretrain_stage_cosine_means": pretrain_stage_cos,
        },
        "bn_running_stats": {
            "recalibration_step_in_v1": False,
            "recalibration_before_after": "SOURCE_GAP_v1_had_no_separate_BN_recalibration",
            "transplant_sliced_from_teacher": bn_tx,
            "after_finetune": bn_tr,
            "drift_transplant_to_trained": bn_compare,
            "note": (
                "Transplant copies sliced teacher running stats via replace_bn3d; "
                "V1 never ran a dedicated train-only BN recalibration before fine-tune."
            ),
        },
        "channel_index_consistency": channel_audit,
        "logit_diagnostics_full_val": {
            "n": len(e2_rows),
            "teacher_student_pred_agreement_vs_E0": teacher_agree,
            "e2_logit_entropy_mean": float(np.mean(ent_e2)) if ent_e2 else float("nan"),
            "e2_logit_entropy_std": float(np.std(ent_e2)) if ent_e2 else float("nan"),
            "error_pair_top15": top_conf,
            "error_concentration_top15_share": top15_share,
            "n_errors": n_err,
        },
        "hypotheses": [
            {
                "id": "roi_head_distribution_shift",
                "suspicion": "high",
                "evidence_summary": (
                    f"Student token_dim={E2_WIDTHS.token_dim} vs teacher 2304; "
                    f"transplant+init_proj monitor top1={acc_tx_mon['top1']:.4f}; "
                    f"transplant+trained_proj monitor top1={acc_tx_trproj['top1']:.4f}; "
                    f"trained full-val top1={val_top1:.4f}."
                ),
            },
            {
                "id": "bn_running_stats_after_slice",
                "suspicion": "high",
                "evidence_summary": (
                    "No V1 BN recalibration; running stats drift from transplant→trained "
                    f"(identical_flags={[bn_compare[f'stage{i}']['identical_to_transplant'] for i in range(4)]})."
                ),
            },
            {
                "id": "channel_mapping_breaks_residual_lateral",
                "suspicion": "high",
                "evidence_summary": (
                    f"Index audit pass={channel_audit['pass']} but semantics unproven; "
                    f"pretrain stage cosines={pretrain_stage_cos}."
                ),
            },
            {
                "id": "insufficient_2400x3_recovery",
                "suspicion": "high",
                "evidence_summary": (
                    f"Train-subset top1={train_top1:.4f} after 3 epochs; full val={val_top1:.4f}; "
                    f"pattern={train_val_pattern}. Screen schedule cannot reject architecture alone."
                ),
            },
            {
                "id": "roi_adapter_incomplete_recovery",
                "suspicion": "medium_high",
                "evidence_summary": "See ROI region cosine/mse in feature_alignment.post_training.",
            },
            {
                "id": "true_capacity_shortage",
                "suspicion": "insufficient_evidence",
                "evidence_summary": (
                    "12.5% width cut vs 23.7pp Top-1 drop is disproportionate; "
                    "capacity shortage not established without a cleaner init/interface control."
                ),
            },
        ],
        "v2_architecture_recommendation": {
            "do_not_repeat": "Reduce residual stage output widths [256,512,1024,2048]",
            "preferred": "Keep stage C_out unchanged; compress Slow internal bottleneck C_mid only",
            "preserve": [
                "ROI input width interface to head (2304)",
                "lateral fusion interfaces",
                "Fast pathway",
                "hand/finger joints",
                "sequential decode",
                "E0 scientific baseline",
                "E1a training/system base",
            ],
            "candidate_internal_ladders_unverified": {
                "if_teacher_inner_is_64_128_256_512": {
                    "light": [56, 112, 224, 448],
                    "medium": [48, 96, 192, 384],
                },
                "must_verify_on_model_then_lock_in_new_V2_policy": True,
                "must_not_backfill_into_V1": True,
            },
            "sequential_compression_order": [
                "stage5_only",
                "stage4_and_5",
                "full_slow_if_prior_keep_accuracy",
            ],
            "training_policy_fixes": [
                "2400x3 is capability/smoke only — not architecture NO-GO evidence",
                "formal accuracy gate only after locked full-train schedule",
                "dedicated train-only BN recalibration before fine-tune",
                "lock SVD/channel-map init before metrics",
            ],
        },
        "source_gaps": [
            "v1_no_separate_BN_recalibration_before_after_pair",
            "training_start_roi_proj_weights_not_checkpointed",
            *([g for g in [retention.get("risk")] if g == "ARTIFACT_RETENTION_GAP"]),
        ],
        "explicit": [
            "This postmortem does not change E2_ACCURACY_NO_GO_E3_NOT_OPENED",
            "E3 remains closed",
            "No post-hoc parity tolerance introduced",
            "Shape tests PASS ≠ semantic channel identity",
        ],
    }

    create_only(POST, report)
    print(
        "E4_B4SC_E2_POSTMORTEM_OK",
        f"train2400={train_top1:.4f}",
        f"val={val_top1:.4f}",
        f"tx_mon={acc_tx_mon['top1']:.4f}",
        f"pattern={train_val_pattern}",
        f"retention={retention['status']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
