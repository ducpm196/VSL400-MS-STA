# -*- coding: utf-8 -*-
"""E4_B4SC_V2 S5 transplant postmortem (train-only diagnostics; create-only).

Does NOT reopen S5 training, does NOT change V2 thresholds/verdict.
Produces: Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_TRANSPLANT_POSTMORTEM.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e4_b4sc_train_e2_screen import (  # noqa: E402
    AmpRoiFp32SlowFast,
    build_b4_batch_from_projected,
    build_ms_sta_v6,
    head_logits_eval,
    load_meta,
)
from scripts.e4_b4sc_v2_run_controls import (  # noqa: E402
    REGIONS,
    FrozenSlowFastBlock4,
    RawVideoResolver,
    cosine,
    ensure_train_cuda,
    extract_tokens,
    load_split_role_ids,
    stratified_sample_ids,
)
from src.optimization.e4_b4sc_v2.factory import build_v2_slowfast  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POLICY = OUT / "E4_B4SC_V2_S5_POLICY_LOCK.json"
CHAN = OUT / "E4_B4SC_V2_S5_CHANNEL_LOCK.json"
FID_IDS = OUT / "E4_B4SC_V2_S5_FIDELITY_SET_IDS.json"
FID_REP = OUT / "E4_B4SC_V2_S5_TRANSPLANT_FIDELITY.json"
BN_PROT = OUT / "E4_B4SC_V2_BN_RECALIBRATION_PROTOCOL.json"
ARCH = OUT / "E4_B4SC_V2_S5_ARCHITECTURE_AUDIT.json"
POST = OUT / "E4_B4SC_V2_S5_TRANSPLANT_POSTMORTEM.json"
STATUS = OUT / "E4_B4SC_V2_CAMPAIGN_STATUS.json"
BN_CACHE = OUT / "E4_B4SC_V2_S5_POSTMORTEM_BN_CACHE.pt"
SKEL = ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3"
WIDTHS = [256, 512, 1024, 2048]


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_V2_S5_POSTMORTEM_FAIL:{msg}")


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha_bn_stats(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, buf in sorted(module.named_buffers()):
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            h.update(name.encode())
            h.update(buf.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def kl_mean(t_logits: torch.Tensor, s_logits: torch.Tensor) -> float:
    p = F.softmax(t_logits, dim=-1)
    log_q = F.log_softmax(s_logits, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def set_stage5_slow_bn_train_stats_only(model: nn.Module) -> list[str]:
    """Authorized BN scope: Slow stage5 BN only; update running stats; no grads."""
    touched: list[str] = []
    for n, m in model.named_modules():
        is_bn = isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
        # names are like blocks.4.multipathway_blocks.0.res_blocks.*.branch2.norm_*
        if is_bn and n.startswith("blocks.4.multipathway_blocks.0."):
            m.train()
            for p in m.parameters():
                p.requires_grad_(False)
            touched.append(n)
        else:
            m.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return touched


def load_first_windows(frames: list, device: torch.device):
    from src.models.rgb_token_v3 import INPUT_SIZE, WINDOW_SIZE, WINDOW_STRIDE, _resize_letterbox, window_starts, window_tensors

    resized = []
    for fr in frames:
        canvas, _ = _resize_letterbox(fr, INPUT_SIZE)
        resized.append(canvas)
    frames_u8 = np.stack(resized, axis=0)
    starts = window_starts(frames_u8.shape[0], window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE)
    slow_t, fast_t, _, _ = window_tensors(frames_u8, starts[0], WINDOW_SIZE)
    slow2 = fast2 = None
    if len(starts) >= 2:
        s2, f2, _, _ = window_tensors(frames_u8, starts[1], WINDOW_SIZE)
        slow2, fast2 = s2.to(device).float(), f2.to(device).float()
    return slow_t.to(device).float(), fast_t.to(device).float(), slow2, fast2, starts


def teacher_student_stages(teacher_raw, student_raw, slow_t, fast_t):
    x: Any = [slow_t, fast_t]
    tea_stages = []
    for bi, block in enumerate(teacher_raw.blocks):
        x = block(x)
        if bi >= 1 and isinstance(x, (list, tuple)):
            tea_stages.append(x[0][:, : WIDTHS[bi - 1]].float())
    tea_fast = x[1].float()
    smaps, sstages = student_raw.forward_maps_with_stages(slow_t, fast_t)
    return tea_stages, tea_fast, smaps, [s.float() for s in sstages]


@torch.no_grad()
def capture_branch2_boundaries(block: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    identity = x
    if getattr(block, "branch1_conv", None) is not None:
        identity = block.branch1_conv(x)
        if getattr(block, "branch1_norm", None) is not None:
            identity = block.branch1_norm(identity)
    b2 = block.branch2
    a = b2.act_a(b2.norm_a(b2.conv_a(x)))
    b = b2.act_b(b2.norm_b(b2.conv_b(a)))
    c = b2.norm_c(b2.conv_c(b))
    out = block.branch_fusion(identity, c)
    if getattr(block, "activation", None) is not None:
        out = block.activation(out)
    return {
        "after_reduce": a.float(),
        "after_temporal": b.float(),
        "after_expand": c.float(),
        "before_residual_add": c.float(),
        "after_residual_add": out.float(),
    }


def spatial_roi_masks(sealed: dict, h: int, w: int) -> dict[str, np.ndarray]:
    """Map sealed landmarks (norm-ish xy) onto HxW boolean masks for face/hands."""
    parts = np.asarray(sealed["joint_part"])
    lm = np.asarray(sealed["effective_landmarks"], dtype=np.float32)  # [T,J,3]
    masks: dict[str, np.ndarray] = {}
    for region in ("face", "left_hand", "right_hand"):
        idx = np.where(parts == region)[0]
        m = np.zeros((h, w), dtype=np.float32)
        if idx.size == 0:
            masks[region] = m
            continue
        pts = lm[:, idx, :2].reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.size == 0:
            masks[region] = m
            continue
        # map roughly-normalized coords in [-1,1] (or similar) to grid
        xs = ((pts[:, 0] + 1.0) * 0.5 * (w - 1)).clip(0, w - 1)
        ys = ((pts[:, 1] + 1.0) * 0.5 * (h - 1)).clip(0, h - 1)
        for x, y in zip(xs.astype(np.int64), ys.astype(np.int64)):
            m[y, x] += 1.0
        if m.sum() > 0:
            m /= m.sum()
        masks[region] = m
    return masks


def eval_fidelity_bundle(
    *,
    fid_ids: list[str],
    meta: dict,
    g2i: dict,
    resolver: RawVideoResolver,
    skel_root: Path,
    teacher_raw,
    student_raw,
    teacher_amp,
    student_amp,
    b4_head,
    device: torch.device,
    th: dict,
    collect_flips: bool = True,
) -> dict[str, Any]:
    kls, agree, tea_ok, stu_ok = [], 0, 0, 0
    stage_cos = [[] for _ in range(4)]
    roi_cos = {r: [] for r in REGIONS}
    temporal_cos, fast_abs = [], []
    nan_inf, n_ok = 0, 0
    flips: list[dict[str, Any]] = []

    for i, sid in enumerate(fid_ids):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        y = int(g2i[gloss])
        signer = str(row.get("signer_id") or "")
        tea_ex, sealed, frames, *_ = extract_tokens(teacher_amp, sid, row, resolver, skel_root)
        stu_ex, *_ = extract_tokens(student_amp, sid, row, resolver, skel_root)

        for r in REGIONS:
            t = torch.from_numpy(np.asarray(tea_ex["tokens"][r], dtype=np.float32))
            s = torch.from_numpy(np.asarray(stu_ex["tokens"][r], dtype=np.float32))
            L = min(int(t.shape[0]), int(s.shape[0]))
            roi_cos[r].append(cosine(t[:L], s[:L]))

        def logits_from_extract(ex: dict) -> torch.Tensor:
            tokens = {
                r: torch.from_numpy(np.asarray(ex["tokens"][r], dtype=np.float32)).to(device) for r in REGIONS
            }
            batch = build_b4_batch_from_projected(
                sid=sid, label=y, signer=signer, proj_tokens=tokens, sealed=sealed, device=device
            )
            return head_logits_eval(b4_head, batch)

        with torch.inference_mode():
            lt = logits_from_extract(tea_ex)
            ls = logits_from_extract(stu_ex)
            if (not torch.isfinite(lt).all()) or (not torch.isfinite(ls).all()):
                nan_inf += 1
                continue
            kls.append(kl_mean(lt, ls))
            yp_t = int(lt.argmax(-1))
            yp_s = int(ls.argmax(-1))
            agree += int(yp_t == yp_s)
            tea_ok += int(yp_t == y)
            stu_ok += int(yp_s == y)

            pt = F.softmax(lt.float(), dim=-1)[0]
            ps = F.softmax(ls.float(), dim=-1)[0]
            top2_t = pt.topk(2).values
            top2_s = ps.topk(2).values
            margin_t = float((top2_t[0] - top2_t[1]).item())
            margin_s = float((top2_s[0] - top2_s[1]).item())

            slow_t, fast_t, slow2, fast2, _starts = load_first_windows(frames, device)
            tea_stages, tea_fast, smaps, sstages = teacher_student_stages(
                teacher_raw, student_raw, slow_t, fast_t
            )
            fast_abs.append(float((tea_fast - smaps.fast.float()).abs().max().item()))
            for si in range(4):
                stage_cos[si].append(cosine(sstages[si], tea_stages[si]))
            motion_l2 = None
            if slow2 is not None:
                tea2, _, _, ss2 = teacher_student_stages(teacher_raw, student_raw, slow2, fast2)
                temporal_cos.append(cosine(ss2[-1] - sstages[-1], tea2[-1] - tea_stages[-1]))
                motion_l2 = float((tea2[-1] - tea_stages[-1]).float().norm().item())

            if collect_flips and yp_t != yp_s:
                flips.append(
                    {
                        "sample_id": sid,
                        "signer_id": signer,
                        "y_true": y,
                        "y_pred_teacher": yp_t,
                        "y_pred_student": yp_s,
                        "teacher_conf": float(pt[yp_t]),
                        "student_conf": float(ps[yp_s]),
                        "teacher_margin": margin_t,
                        "student_margin": margin_s,
                        "motion_l2": motion_l2,
                        "roi_cosine": {
                            r: float(roi_cos[r][-1]) for r in REGIONS
                        },
                    }
                )

        n_ok += 1
        if (i + 1) % 32 == 0:
            print(f"  fidelity {i+1}/{len(fid_ids)}", flush=True)

    coverage = n_ok / max(len(fid_ids), 1)
    agreement = agree / max(n_ok, 1)
    tea_top1 = tea_ok / max(n_ok, 1)
    stu_top1 = stu_ok / max(n_ok, 1)
    retention = (stu_top1 / tea_top1) if tea_top1 > 0 else float("nan")
    stage_means = [float(np.mean(xs)) if xs else float("nan") for xs in stage_cos]
    roi_means = {r: float(np.mean(roi_cos[r])) if roi_cos[r] else float("nan") for r in REGIONS}
    temporal_mean = float(np.mean(temporal_cos)) if temporal_cos else float("nan")
    kl_m = float(np.mean(kls)) if kls else float("nan")
    fast_max = float(np.max(fast_abs)) if fast_abs else float("nan")

    checks = {
        "forbid_nan_inf": nan_inf == 0,
        "sample_coverage_min": coverage >= float(th["sample_coverage_min"]),
        "logits_kl_mean_max": kl_m <= float(th["logits_kl_mean_max"]),
        "teacher_student_pred_agreement_min": agreement
        >= float(th["teacher_student_pred_agreement_min"]),
        "top1_retention_ratio_vs_teacher_min": retention
        >= float(th["top1_retention_ratio_vs_teacher_min"]),
        "stage_cosine_floor": all(
            stage_means[i] >= float(th["stage_cosine_floor"][i]) for i in range(4)
        ),
        "roi_fidelity_cosine_min": all(
            roi_means[r] >= float(th["roi_fidelity_cosine_min"][r]) for r in REGIONS
        ),
        "temporal_diff_cosine_min": (not temporal_cos)
        or (temporal_mean >= float(th["temporal_diff_cosine_min"])),
        "fast_pathway_max_abs_delta": fast_max <= float(th["fast_pathway_max_abs_delta"]),
    }
    return {
        "n_ok": n_ok,
        "coverage": coverage,
        "nan_inf_count": nan_inf,
        "metrics": {
            "logits_kl_mean": kl_m,
            "pred_agreement": agreement,
            "teacher_top1": tea_top1,
            "student_top1": stu_top1,
            "top1_retention_ratio_vs_teacher": retention,
            "stage_cosine_mean": stage_means,
            "roi_cosine_mean": roi_means,
            "temporal_diff_cosine_mean": temporal_mean,
            "temporal_n": len(temporal_cos),
            "fast_max_abs_delta": fast_max,
        },
        "checks": checks,
        "pass": all(checks.values()),
        "prediction_flips": flips,
        "n_flips": len(flips),
    }


def teacher_mid_activation(teacher_raw, slow_t, fast_t) -> torch.Tensor:
    """Mean mid (act_b) over stage5 Slow residual blocks: [B,512,T,H,W]."""
    x: Any = [slow_t, fast_t]
    for bi in range(4):
        x = teacher_raw.blocks[bi](x)
    stage5 = teacher_raw.blocks[4].multipathway_blocks[0]
    stores: list[list[torch.Tensor]] = []
    handles = []
    for rb in stage5.res_blocks:
        store: list[torch.Tensor] = []

        def make_hook(st):
            def _h(_m, _inp, out):
                st.append(out.detach().float())

            return _h

        handles.append(rb.branch2.act_b.register_forward_hook(make_hook(store)))
        stores.append(store)
    _ = teacher_raw.blocks[4](x)
    for h in handles:
        h.remove()
    mids = [s[0] for s in stores if s]
    return torch.stack(mids, dim=0).mean(0)


def main() -> int:
    ensure_train_cuda()
    if POST.exists():
        die(f"exists:{POST}")
    for p in (POLICY, CHAN, FID_IDS, FID_REP, BN_PROT, ARCH):
        if not p.is_file():
            die(f"missing:{p}")

    pol = json.loads(POLICY.read_text(encoding="utf-8"))
    chan = json.loads(CHAN.read_text(encoding="utf-8"))
    fid_doc = json.loads(FID_IDS.read_text(encoding="utf-8"))
    fid_rep = json.loads(FID_REP.read_text(encoding="utf-8"))
    bn_prot = json.loads(BN_PROT.read_text(encoding="utf-8"))
    arch = json.loads(ARCH.read_text(encoding="utf-8"))
    th = dict(pol["fidelity_protocol"]["thresholds"])

    device = torch.device("cuda")
    meta = load_meta()
    labels = json.loads((ROOT / pol["bindings"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels["gloss_to_id"]
    split = json.loads((ROOT / pol["bindings"]["split_manifest_rel"]).read_text(encoding="utf-8"))
    train_ids = load_split_role_ids(split, "confirm_train")
    val_ids = set(load_split_role_ids(split, "val"))
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    fid_ids = list(fid_doc["sample_ids"])
    if set(fid_ids) & val_ids or set(fid_ids) & test_ids:
        die("fidelity_set_leak")

    keep_idx = np.asarray(chan["stages"]["3"]["inner_idx"], dtype=np.int64)
    removed_idx = np.asarray(sorted(set(range(512)) - set(keep_idx.tolist())), dtype=np.int64)
    if len(removed_idx) != 64 or len(keep_idx) != 448:
        die(f"channel_counts:{len(keep_idx)},{len(removed_idx)}")

    teacher_raw = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["bindings"]["slowfast_rel"],
        expected_sha256=pol["bindings"]["slowfast_sha256"],
        device=device,
        repo_root=ROOT,
    )
    student_raw = build_v2_slowfast(
        teacher_raw,
        "V2_S5",
        device=device,
        requires_grad_slow=False,
        s5_c_mid=tuple(int(x) for x in pol["architecture_lock"]["C_mid_S5"]),
        eval_mode=True,
    )
    if student_raw.plan["stages"]["3"]["inner_idx_sha256"] != chan["stages"]["3"]["inner_idx_sha256"]:
        die("channel_lock_mismatch")

    ckpt_path = ROOT / arch["checkpoint"]["transplant_init_rel"]
    if not ckpt_path.is_file():
        durable = ROOT / str(arch["checkpoint"].get("durable_registry_rel", "")).replace("\\", "/")
        if durable.is_file():
            ckpt_path = durable
        else:
            die("missing_transplant_ckpt")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    student_raw.load_state_dict(payload["model"], strict=True)
    student_raw.eval()

    teacher_amp = AmpRoiFp32SlowFast(teacher_raw, channels_last=False)
    student_amp = AmpRoiFp32SlowFast(student_raw, channels_last=False)
    b4_head = build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt = torch.load(ROOT / pol["bindings"]["b4_teacher_rel"], map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()
    resolver = RawVideoResolver(ROOT)

    # ------------------------------------------------------------------
    # 1) BN-only counterfactual on a COPY (train-only; no grad; no FT)
    # ------------------------------------------------------------------
    bn_n = int(bn_prot["protocol"]["n_samples"])
    bn_seed = int(bn_prot["protocol"]["sample_order_seed"])
    bn_ids = stratified_sample_ids(train_ids, meta, bn_n, bn_seed, signer_field="signer_id")
    if set(bn_ids) & val_ids or set(bn_ids) & test_ids:
        die("bn_calib_leak")
    calib_ids_sha = hashlib.sha256(json.dumps(bn_ids, separators=(",", ":")).encode()).hexdigest()
    transplant_sha = str(arch["checkpoint"]["sha256"])

    if BN_CACHE.is_file():
        print("POSTMORTEM loading BN cache", flush=True)
        cache = torch.load(BN_CACHE, map_location=device, weights_only=False)
        if (
            cache.get("transplant_sha") != transplant_sha
            or cache.get("calib_ids_sha") != calib_ids_sha
            or cache.get("channel_sha") != chan["stages"]["3"]["inner_idx_sha256"]
        ):
            die("bn_cache_mismatch")
        student_bn = copy.deepcopy(student_raw)
        student_bn.load_state_dict(cache["student_bn_state"], strict=True)
        student_bn.eval()
        bn_before = cache["bn_before"]
        bn_after = cache["bn_after"]
        touched_bn = list(cache["touched_bn"])
        bn_fid = cache["bn_fid"]
        print("POSTMORTEM BN cache hit", flush=True)
    else:
        print("POSTMORTEM BN-only counterfactual", flush=True)
        student_bn = copy.deepcopy(student_raw)
        bn_before = sha_bn_stats(student_bn)
        touched_bn = set_stage5_slow_bn_train_stats_only(student_bn)
        with torch.no_grad():
            for i, sid in enumerate(bn_ids):
                row = meta[sid]
                _ex, _sealed, frames, *_ = extract_tokens(teacher_amp, sid, row, resolver, SKEL)
                slow_t, fast_t, *_ = load_first_windows(frames, device)
                _ = student_bn.forward_maps_with_stages(slow_t, fast_t)
                if (i + 1) % 64 == 0:
                    print(f"  bn_recal {i+1}/{bn_n}", flush=True)
        student_bn.eval()
        bn_after = sha_bn_stats(student_bn)
        student_bn_amp = AmpRoiFp32SlowFast(student_bn, channels_last=False)
        print("POSTMORTEM re-eval fidelity after BN-only", flush=True)
        bn_fid = eval_fidelity_bundle(
            fid_ids=fid_ids,
            meta=meta,
            g2i=g2i,
            resolver=resolver,
            skel_root=SKEL,
            teacher_raw=teacher_raw,
            student_raw=student_bn,
            teacher_amp=teacher_amp,
            student_amp=student_bn_amp,
            b4_head=b4_head,
            device=device,
            th=th,
            collect_flips=False,
        )
        torch.save(
            {
                "transplant_sha": transplant_sha,
                "calib_ids_sha": calib_ids_sha,
                "channel_sha": chan["stages"]["3"]["inner_idx_sha256"],
                "student_bn_state": student_bn.state_dict(),
                "bn_before": bn_before,
                "bn_after": bn_after,
                "touched_bn": touched_bn,
                "bn_fid": bn_fid,
            },
            BN_CACHE,
        )
        print(f"POSTMORTEM BN cache written:{BN_CACHE}", flush=True)

    # ------------------------------------------------------------------
    # Combined: transplant-init fidelity + flips + channel audit + boundaries
    # ------------------------------------------------------------------
    print("POSTMORTEM transplant-init fidelity + channel/boundary/flip audit", flush=True)
    energy = np.zeros(512, dtype=np.float64)
    tvar = np.zeros(512, dtype=np.float64)
    tderiv = np.zeros(512, dtype=np.float64)
    roi_energy = {r: np.zeros(512, dtype=np.float64) for r in ("face", "left_hand", "right_hand")}
    grad_sens = np.zeros(512, dtype=np.float64)
    n_act = 0
    n_grad = 0
    grad_budget = 32
    rem_energy_rows: list[np.ndarray] = []
    margin_rows: list[float] = []
    flip_flags: list[int] = []

    boundary_keys = [
        "s5_input",
        "after_reduce",
        "after_temporal",
        "after_expand",
        "before_residual_add",
        "after_residual_add",
        "roi_global",
    ]
    boundary_cos = {k: [] for k in boundary_keys}

    kls, agree, tea_ok, stu_ok = [], 0, 0, 0
    stage_cos = [[] for _ in range(4)]
    roi_cos = {r: [] for r in REGIONS}
    temporal_cos, fast_abs = [], []
    nan_inf, n_ok = 0, 0
    flips: list[dict[str, Any]] = []

    for i, sid in enumerate(fid_ids):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        y = int(g2i[gloss])
        signer = str(row.get("signer_id") or "")
        tea_ex, sealed, frames, *_ = extract_tokens(teacher_amp, sid, row, resolver, SKEL)
        stu_ex, *_ = extract_tokens(student_amp, sid, row, resolver, SKEL)

        for r in REGIONS:
            t = torch.from_numpy(np.asarray(tea_ex["tokens"][r], dtype=np.float32))
            s = torch.from_numpy(np.asarray(stu_ex["tokens"][r], dtype=np.float32))
            L = min(int(t.shape[0]), int(s.shape[0]))
            roi_cos[r].append(cosine(t[:L], s[:L]))

        tokens = {
            r: torch.from_numpy(np.asarray(tea_ex["tokens"][r], dtype=np.float32)).to(device) for r in REGIONS
        }
        stu_tokens = {
            r: torch.from_numpy(np.asarray(stu_ex["tokens"][r], dtype=np.float32)).to(device) for r in REGIONS
        }
        batch_t = build_b4_batch_from_projected(
            sid=sid, label=y, signer=signer, proj_tokens=tokens, sealed=sealed, device=device
        )
        batch_s = build_b4_batch_from_projected(
            sid=sid, label=y, signer=signer, proj_tokens=stu_tokens, sealed=sealed, device=device
        )

        # Use no_grad (not inference_mode) so tensors can later enter autograd for grad proxy.
        with torch.no_grad():
            lt = head_logits_eval(b4_head, batch_t)
            ls = head_logits_eval(b4_head, batch_s)
            if (not torch.isfinite(lt).all()) or (not torch.isfinite(ls).all()):
                nan_inf += 1
                continue
            kls.append(kl_mean(lt, ls))
            yp_t = int(lt.argmax(-1))
            yp_s = int(ls.argmax(-1))
            agree += int(yp_t == yp_s)
            tea_ok += int(yp_t == y)
            stu_ok += int(yp_s == y)
            pt = F.softmax(lt.float(), dim=-1)[0]
            ps = F.softmax(ls.float(), dim=-1)[0]
            margin_t = float((pt.topk(2).values[0] - pt.topk(2).values[1]).item())
            margin_s = float((ps.topk(2).values[0] - ps.topk(2).values[1]).item())

            slow_t, fast_t, slow2, fast2, _starts = load_first_windows(frames, device)
            tea_stages, tea_fast, smaps, sstages = teacher_student_stages(
                teacher_raw, student_raw, slow_t, fast_t
            )
            fast_abs.append(float((tea_fast - smaps.fast.float()).abs().max().item()))
            for si in range(4):
                stage_cos[si].append(cosine(sstages[si], tea_stages[si]))
            motion_l2 = None
            if slow2 is not None:
                tea2, _, _, ss2 = teacher_student_stages(teacher_raw, student_raw, slow2, fast2)
                temporal_cos.append(cosine(ss2[-1] - sstages[-1], tea2[-1] - tea_stages[-1]))
                motion_l2 = float((tea2[-1] - tea_stages[-1]).float().norm().item())

            if yp_t != yp_s:
                flips.append(
                    {
                        "sample_id": sid,
                        "signer_id": signer,
                        "y_true": y,
                        "y_pred_teacher": yp_t,
                        "y_pred_student": yp_s,
                        "teacher_conf": float(pt[yp_t]),
                        "student_conf": float(ps[yp_s]),
                        "teacher_margin": margin_t,
                        "student_margin": margin_s,
                        "motion_l2": motion_l2,
                        "roi_cosine": {r: float(roi_cos[r][-1]) for r in REGIONS},
                    }
                )

            mid = teacher_mid_activation(teacher_raw, slow_t, fast_t)
            e = mid.pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
            tv = mid.var(dim=2).mean(dim=(0, 2, 3)).cpu().numpy()
            if mid.shape[2] >= 2:
                td = (mid[:, :, 1:] - mid[:, :, :-1]).pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
            else:
                td = np.zeros(512, dtype=np.float64)
            if slow2 is not None:
                mid2 = teacher_mid_activation(teacher_raw, slow2, fast2)
                td = td + (mid2 - mid).pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
            energy += e
            tvar += tv
            tderiv += td
            n_act += 1

            hh, ww = int(mid.shape[3]), int(mid.shape[4])
            masks = spatial_roi_masks(sealed if isinstance(sealed, dict) else {}, hh, ww)
            mid_np = mid[0].pow(2).mean(dim=1).cpu().numpy()
            for region, m in masks.items():
                if m.sum() <= 0:
                    continue
                roi_energy[region] += (mid_np * m[None, :, :]).sum(axis=(1, 2))

            rem_energy_rows.append(e[removed_idx].copy())
            margin_rows.append(margin_t)
            flip_flags.append(int(yp_t != yp_s))

            xt: Any = [slow_t, fast_t]
            xs: Any = [slow_t, fast_t]
            for bi in range(4):
                xt = teacher_raw.blocks[bi](xt)
                xs = student_raw.blocks[bi](xs)
            boundary_cos["s5_input"].append(cosine(xs[0].float(), xt[0].float()))
            t_stage = teacher_raw.blocks[4].multipathway_blocks[0]
            s_stage = student_raw.blocks[4].multipathway_blocks[0]
            t_hold: dict[str, torch.Tensor] = {}
            s_hold: dict[str, torch.Tensor] = {}

            def thook(_m, inp, _out):
                t_hold.update(capture_branch2_boundaries(_m, inp[0]))

            def shook(_m, inp, _out):
                s_hold.update(capture_branch2_boundaries(_m, inp[0]))

            ht = t_stage.res_blocks[0].register_forward_hook(thook)
            hs = s_stage.res_blocks[0].register_forward_hook(shook)
            _ = teacher_raw.blocks[4](xt)
            _ = student_raw.blocks[4](xs)
            ht.remove()
            hs.remove()
            for key in ("after_reduce", "after_temporal"):
                boundary_cos[key].append(cosine(s_hold[key], t_hold[key][:, keep_idx]))
            for key in ("after_expand", "before_residual_add", "after_residual_add"):
                boundary_cos[key].append(cosine(s_hold[key], t_hold[key]))
            tg = torch.from_numpy(np.asarray(tea_ex["tokens"]["global"], dtype=np.float32))
            sg = torch.from_numpy(np.asarray(stu_ex["tokens"]["global"], dtype=np.float32))
            Lg = min(tg.shape[0], sg.shape[0])
            boundary_cos["roi_global"].append(cosine(sg[:Lg], tg[:Lg]))

        # Grad sensitivity proxy (INFERENCE): mid L2 w.r.t. inputs; not full teacher-logit path.
        if n_grad < grad_budget:
            teacher_raw.zero_grad(set_to_none=True)
            mid_holder: dict[str, torch.Tensor] = {}

            def retain_hook(_m, _inp, out):
                mid_holder["mid"] = out
                if out.requires_grad:
                    out.retain_grad()

            with torch.enable_grad():
                slow_req = slow_t.detach().clone().requires_grad_(True)
                fast_req = fast_t.detach().clone().requires_grad_(True)
                xg: Any = [slow_req, fast_req]
                for bi in range(4):
                    xg = teacher_raw.blocks[bi](xg)
                h = teacher_raw.blocks[4].multipathway_blocks[0].res_blocks[0].branch2.act_b.register_forward_hook(
                    retain_hook
                )
                _ = teacher_raw.blocks[4](xg)
                h.remove()
                if "mid" in mid_holder and mid_holder["mid"].requires_grad:
                    mid_g = mid_holder["mid"]
                    (mid_g.pow(2).mean()).backward()
                    if mid_g.grad is not None:
                        g = mid_g.grad.detach().abs().mean(dim=(0, 2, 3, 4)).cpu().numpy()
                        grad_sens += g
                        n_grad += 1
            teacher_raw.zero_grad(set_to_none=True)

        n_ok += 1
        if (i + 1) % 16 == 0:
            print(
                f"  audit {i+1}/{len(fid_ids)} agree={agree/max(n_ok,1):.3f} flips={len(flips)}",
                flush=True,
            )

    energy /= max(n_act, 1)
    tvar /= max(n_act, 1)
    tderiv /= max(n_act, 1)
    for r in roi_energy:
        roi_energy[r] /= max(n_act, 1)
    if n_grad > 0:
        grad_sens /= n_grad

    rem_e = energy[removed_idx]
    rem_td = tderiv[removed_idx]
    rem_tv = tvar[removed_idx]
    rem_g = grad_sens[removed_idx]
    e_med = float(np.median(energy))
    td_med = float(np.median(tderiv))
    low_e_high_td = [
        int(ch) for ch, e, td in zip(removed_idx, rem_e, rem_td) if e <= e_med and td >= td_med
    ]

    rem_mat = np.stack(rem_energy_rows, axis=0) if rem_energy_rows else np.zeros((0, 64))
    margins_arr = np.asarray(margin_rows, dtype=np.float64)
    flips_arr = np.asarray(flip_flags, dtype=np.float64)
    class_margin_corr = []
    if rem_mat.shape[0] >= 8:
        for j, ch in enumerate(removed_idx):
            x = rem_mat[:, j]
            if float(x.std()) < 1e-12 or float(margins_arr.std()) < 1e-12:
                corr_m = float("nan")
            else:
                corr_m = float(np.corrcoef(x, margins_arr)[0, 1])
            if float(x.std()) < 1e-12 or float(flips_arr.std()) < 1e-12:
                corr_f = float("nan")
            else:
                corr_f = float(np.corrcoef(x, flips_arr)[0, 1])
            class_margin_corr.append(
                {"channel": int(ch), "corr_with_teacher_margin": corr_m, "corr_with_flip": corr_f}
            )

    channel_audit = {
        "n_activation_clips": n_act,
        "n_grad_clips": n_grad,
        "removed_channels": removed_idx.tolist(),
        "kept_channels_sha256": chan["stages"]["3"]["inner_idx_sha256"],
        "removed_stats": {
            "activation_energy_mean": float(rem_e.mean()),
            "activation_energy_median": float(np.median(rem_e)),
            "temporal_variance_mean": float(rem_tv.mean()),
            "temporal_derivative_mean": float(rem_td.mean()),
            "grad_sensitivity_mean": float(rem_g.mean()) if n_grad else None,
            "kept_activation_energy_mean": float(energy[keep_idx].mean()),
            "kept_temporal_derivative_mean": float(tderiv[keep_idx].mean()),
            "removed_vs_kept_energy_ratio": float(rem_e.mean() / max(energy[keep_idx].mean(), 1e-12)),
            "removed_vs_kept_temporal_deriv_ratio": float(
                rem_td.mean() / max(tderiv[keep_idx].mean(), 1e-12)
            ),
            "hand_face_roi_contribution_removed_vs_kept": {
                r: {
                    "removed_mean": float(roi_energy[r][removed_idx].mean()),
                    "kept_mean": float(roi_energy[r][keep_idx].mean()),
                    "ratio": float(
                        roi_energy[r][removed_idx].mean()
                        / max(float(roi_energy[r][keep_idx].mean()), 1e-12)
                    ),
                }
                for r in ("face", "left_hand", "right_hand")
            },
        },
        "teacher_logit_grad_sensitivity": {
            "method": "local_mid_L2_energy_grad_proxy_not_full_head_path",
            "claim_class": "INFERENCE",
            "n_clips": n_grad,
            "removed_mean": float(rem_g.mean()) if n_grad else None,
            "kept_mean": float(grad_sens[keep_idx].mean()) if n_grad else None,
            "removed_vs_kept_ratio": float(rem_g.mean() / max(grad_sens[keep_idx].mean(), 1e-12))
            if n_grad
            else None,
        },
        "class_margin_contribution": {
            "method": "per_removed_channel_corr_energy_vs_teacher_margin_and_flip",
            "claim_class": "INFERENCE",
            "channels": class_margin_corr[:64],
            "mean_abs_corr_with_teacher_margin": float(
                np.nanmean([abs(c["corr_with_teacher_margin"]) for c in class_margin_corr])
            )
            if class_margin_corr
            else None,
            "mean_abs_corr_with_flip": float(
                np.nanmean([abs(c["corr_with_flip"]) for c in class_margin_corr])
            )
            if class_margin_corr
            else None,
        },
        "low_energy_high_temporal_removed_channels": low_e_high_td,
        "low_energy_high_temporal_fraction": float(len(low_e_high_td) / max(len(removed_idx), 1)),
        "top16_removed_by_temporal_derivative": [
            {
                "channel": int(removed_idx[j]),
                "temporal_derivative": float(rem_td[j]),
                "energy": float(rem_e[j]),
            }
            for j in np.argsort(-rem_td)[:16]
        ],
        "claim_class": "INFERENCE_supported_by_train_only_activation_stats",
    }

    boundary_means = {k: float(np.mean(v)) if v else float("nan") for k, v in boundary_cos.items()}
    chain = [
        "s5_input",
        "after_reduce",
        "after_temporal",
        "after_expand",
        "before_residual_add",
        "after_residual_add",
        "roi_global",
    ]
    drops = []
    for a, b in zip(chain, chain[1:]):
        drops.append({"from": a, "to": b, "delta": float(boundary_means[b] - boundary_means[a])})
    first_drop = min(drops, key=lambda d: d["delta"]) if drops else None

    coverage = n_ok / max(len(fid_ids), 1)
    agreement = agree / max(n_ok, 1)
    tea_top1 = tea_ok / max(n_ok, 1)
    stu_top1 = stu_ok / max(n_ok, 1)
    retention = (stu_top1 / tea_top1) if tea_top1 > 0 else float("nan")
    stage_means = [float(np.mean(xs)) if xs else float("nan") for xs in stage_cos]
    roi_means = {r: float(np.mean(roi_cos[r])) if roi_cos[r] else float("nan") for r in REGIONS}
    temporal_mean = float(np.mean(temporal_cos)) if temporal_cos else float("nan")
    kl_m = float(np.mean(kls)) if kls else float("nan")
    fast_max = float(np.max(fast_abs)) if fast_abs else float("nan")
    init_checks = {
        "forbid_nan_inf": nan_inf == 0,
        "sample_coverage_min": coverage >= float(th["sample_coverage_min"]),
        "logits_kl_mean_max": kl_m <= float(th["logits_kl_mean_max"]),
        "teacher_student_pred_agreement_min": agreement
        >= float(th["teacher_student_pred_agreement_min"]),
        "top1_retention_ratio_vs_teacher_min": retention
        >= float(th["top1_retention_ratio_vs_teacher_min"]),
        "stage_cosine_floor": all(
            stage_means[i] >= float(th["stage_cosine_floor"][i]) for i in range(4)
        ),
        "roi_fidelity_cosine_min": all(
            roi_means[r] >= float(th["roi_fidelity_cosine_min"][r]) for r in REGIONS
        ),
        "temporal_diff_cosine_min": (not temporal_cos)
        or (temporal_mean >= float(th["temporal_diff_cosine_min"])),
        "fast_pathway_max_abs_delta": fast_max <= float(th["fast_pathway_max_abs_delta"]),
    }
    init_fid = {
        "n_ok": n_ok,
        "metrics": {
            "logits_kl_mean": kl_m,
            "pred_agreement": agreement,
            "teacher_top1": tea_top1,
            "student_top1": stu_top1,
            "top1_retention_ratio_vs_teacher": retention,
            "stage_cosine_mean": stage_means,
            "roi_cosine_mean": roi_means,
            "temporal_diff_cosine_mean": temporal_mean,
            "temporal_n": len(temporal_cos),
            "fast_max_abs_delta": fast_max,
        },
        "checks": init_checks,
        "pass": all(init_checks.values()),
    }

    flip_audit = {
        "n_flips": len(flips),
        "flip_rate": float(len(flips) / max(n_ok, 1)),
        "mean_teacher_margin_on_flips": float(np.mean([f["teacher_margin"] for f in flips]))
        if flips
        else None,
        "mean_student_margin_on_flips": float(np.mean([f["student_margin"] for f in flips]))
        if flips
        else None,
        "mean_motion_l2_on_flips": float(
            np.nanmean([f["motion_l2"] for f in flips if f.get("motion_l2") is not None])
        )
        if any(f.get("motion_l2") is not None for f in flips)
        else None,
        "signer_flip_counts": dict(Counter(f["signer_id"] for f in flips).most_common(20)),
        "class_transition_top20": [
            {"teacher_pred": int(a), "student_pred": int(b), "n": int(n)}
            for (a, b), n in Counter((f["y_pred_teacher"], f["y_pred_student"]) for f in flips).most_common(
                20
            )
        ],
        "examples_head20": flips[:20],
        "note": "Train-only diagnostic; must not rewrite V2 thresholds or transplant verdict.",
    }

    bn_pass = bool(bn_fid["pass"])
    high_temporal_removed = float(channel_audit["low_energy_high_temporal_fraction"]) >= 0.25 or (
        float(channel_audit["removed_stats"]["removed_vs_kept_temporal_deriv_ratio"]) >= 1.1
    )
    boundary_at_proj = first_drop is not None and first_drop["to"] in (
        "after_reduce",
        "after_expand",
        "after_temporal",
    )

    if bn_pass:
        recommended = "OPEN_NEW_POLICY_S5_BN_FIRST"
        rationale = "BN-only counterfactual passes locked fidelity gates on the same train-only set."
    elif high_temporal_removed and not bn_pass:
        recommended = "TEMPORAL_ROI_AWARE_CHANNEL_SELECTION"
        rationale = (
            "Removed channels show elevated temporal sensitivity relative to kept set; "
            "BN-only did not clear gates."
        )
    elif boundary_at_proj and not bn_pass:
        recommended = "ACTIVATION_AWARE_LOW_RANK_OR_BLOCKWISE_TEACHER_REGRESSION"
        rationale = "Largest cosine drop occurs at reduce/temporal/expand projection boundary."
    else:
        recommended = "STOP_HARD_CHANNEL_PRUNING_STUDY_FACTORIZATION_KEEP_INTERFACE"
        rationale = (
            "BN-only, temporal-removed, and projection-boundary signals do not recover init fidelity."
        )

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {
        "schema": "ms_sta_v6_e4_b4sc_v2_s5_transplant_postmortem_v1",
        "created_at": now,
        "git_head": git_head,
        "parent_verdict": "V2_S5_TRANSPLANT_FIDELITY_NO_GO",
        "verdict_unchanged": True,
        "e4_ready": False,
        "S45_opened": False,
        "training_reopened": False,
        "thresholds_changed": False,
        "c2_gaussian_used": False,
        "scientific_statement": {
            "rejected": (
                "Hard channel selection for S5 C_mid 512→448 yields a student near enough to teacher "
                "at initialization under locked prediction/retention/temporal gates."
            ),
            "not_rejected": [
                "internal-bottleneck compression in general",
                "BN recalibration as a future arm",
                "KD/full training",
                "low-rank or temporal-aware initialization",
            ],
            "kd_attribution": "FORBIDDEN — KD was never opened",
            "param_delta_note": (
                "≈3.085M params (~7.7% deployed) is not a valid model-reduction claim because "
                "init fidelity failed and efficiency was not measured."
            ),
            "metric_interpretation": {
                "stage_roi_cosine": "Static average structure largely retained (INFERENCE).",
                "temporal_and_prediction_fail": (
                    "Removed channels may carry low-energy but decision-relevant motion "
                    "(INFERENCE, not FACT)."
                ),
                "open_causes": [
                    "BN statistics mismatch",
                    "Top-k selection drops motion-sensitive channels",
                    "Channel cut does not approximate residual nonlinear map",
                ],
            },
        },
        "inputs": {
            "policy_rel": str(POLICY.relative_to(ROOT)).replace("\\", "/"),
            "channel_lock_rel": str(CHAN.relative_to(ROOT)).replace("\\", "/"),
            "fidelity_set_rel": str(FID_IDS.relative_to(ROOT)).replace("\\", "/"),
            "fidelity_report_rel": str(FID_REP.relative_to(ROOT)).replace("\\", "/"),
            "transplant_ckpt_rel": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
            "original_fidelity_metrics": fid_rep.get("metrics"),
        },
        "bn_only_counterfactual": {
            "role": "cause_classification_only",
            "protocol_rel": str(BN_PROT.relative_to(ROOT)).replace("\\", "/"),
            "authorized_bn_scope": "blocks.4.multipathway_blocks.0.* BatchNorm only",
            "n_calib": bn_n,
            "calib_seed": bn_seed,
            "calib_ids_sha256": calib_ids_sha,
            "bn_cache_rel": str(BN_CACHE.relative_to(ROOT)).replace("\\", "/") if BN_CACHE.is_file() else None,
            "bn_modules_updated_n": len(touched_bn),
            "bn_modules_updated_head20": touched_bn[:20],
            "bn_stats_sha_before": bn_before,
            "bn_stats_sha_after": bn_after,
            "bn_stats_changed": bn_before != bn_after,
            "gradients": False,
            "optimizer_step": False,
            "val_used": False,
            "fidelity_after_bn": {
                "metrics": bn_fid["metrics"],
                "checks": bn_fid["checks"],
                "pass": bn_fid["pass"],
            },
            "does_not_override_verdict": True,
        },
        "removed_channel_audit": channel_audit,
        "block_boundary_localization": {
            "mean_cosine": boundary_means,
            "pairwise_drops": drops,
            "first_largest_drop": first_drop,
            "interpretation_claim_class": "INFERENCE",
        },
        "prediction_flip_audit": flip_audit,
        "transplant_init_reeval": {
            "metrics": init_fid["metrics"],
            "checks": init_fid["checks"],
            "pass": init_fid["pass"],
            "note": "Recomputed on locked transplant weights; expected still FAIL.",
        },
        "direction_rule": {
            "bn_only_passes_gates": bn_pass,
            "high_temporal_sensitivity_in_removed": high_temporal_removed,
            "projection_boundary_implicated": boundary_at_proj,
            "recommended_next": recommended,
            "rationale": rationale,
            "S45_remains_closed": True,
            "forbid_threshold_relaxation": True,
        },
    }
    create_only(POST, out)
    if BN_CACHE.is_file():
        BN_CACHE.unlink()
        print("POSTMORTEM BN cache removed after success", flush=True)

    st = json.loads(STATUS.read_text(encoding="utf-8"))
    st["status"] = "V2_S5_TRANSPLANT_FIDELITY_NO_GO_POSTMORTEM_COMPLETE"
    st["e4_ready"] = False
    st["updated_at"] = now
    st["s5"] = st.get("s5") or {}
    st["s5"]["postmortem_rel"] = str(POST.relative_to(ROOT)).replace("\\", "/")
    st["s5"]["recommended_next"] = recommended
    st["history"] = list(st.get("history") or [])
    st["history"].append(
        {
            "status": st["status"],
            "updated_at": now,
            "e4_ready": False,
            "note": f"Postmortem complete; verdict unchanged; next={recommended}",
        }
    )
    st["next"] = [
        "do_not_reopen_failed_S5_run",
        f"consider_new_policy_if_approved:{recommended}",
        "keep_S45_closed",
    ]
    STATUS.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("E4_B4SC_V2_S5_POSTMORTEM_OK", recommended, flush=True)
    print(
        json.dumps(
            {
                "bn_pass": bn_pass,
                "recommended": recommended,
                "first_drop": first_drop,
                "low_e_high_td_frac": channel_audit["low_energy_high_temporal_fraction"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
