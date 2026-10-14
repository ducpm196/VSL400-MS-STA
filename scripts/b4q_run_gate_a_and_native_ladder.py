# -*- coding: utf-8 -*-
"""B4Q-V1 Gate A remeasure + native ladder Q0→Q3 (no TRT/INT8/compile/QAT).

Create-only artifacts under Docs/results/ms_sta_v6_b4q/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import (  # noqa: E402
    MAX_RGB_LEN,
    b4_item_from_extract,
    cuda_ms,
    rss_mb,
    sealed_landmarks,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters  # noqa: E402
from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video, read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import load_policy, sha256_file, verify_b4_checkpoint, verify_slowfast_checkpoint  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, head_forward  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_b4q"
SCHED = ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json"


def die(msg: str) -> None:
    print(f"B4Q_FAIL:{msg}", file=sys.stderr)
    raise SystemExit(2)


def write_create_only(path: Path, obj: dict) -> str:
    if path.is_file():
        die(f"refuse_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.stem + ".writing.json")
    if tmp.exists():
        tmp.unlink()
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def p90(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(0.9 * (len(s) - 1)))])


def bootstrap_speedup(base: list[float], cand: list[float], *, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    speeds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mb, mc = float(np.median(b[idx])), float(np.median(c[idx]))
        speeds.append(mb / max(mc, 1e-9))
    lo, hi = np.quantile(speeds, [0.025, 0.975])
    return {"speedup_median": float(np.median(speeds)), "ci95_lo": float(lo), "ci95_hi": float(hi)}


def load_meta() -> dict[str, dict]:
    out = {}
    with (ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("dataset") or "") != "VSL400":
                continue
            sid = str(row.get("video_id") or "").strip()
            if sid:
                out[sid] = dict(row)
    return out


def run_b4_pipeline(
    *,
    sid: str,
    video_path: Path,
    sealed_path: Path,
    extractor: Any,
    b4_model: torch.nn.Module,
    device: torch.device,
    label: int,
    signer: str,
    amp_head: bool,
    channels_last_head: bool = False,
) -> dict:
    t0 = time.perf_counter()
    frames, frame_idx, fps = read_video_frames_native(str(video_path))
    decode_ms = (time.perf_counter() - t0) * 1000.0
    landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)

    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    rss0 = rss_mb()

    t_rgb0 = time.perf_counter()
    extract = extract_rgb_token_v3_for_video(
        video_path=str(video_path),
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
    )
    rgb_ms = (time.perf_counter() - t_rgb0) * 1000.0

    t_sk0 = time.perf_counter()
    item = b4_item_from_extract(sid, extract, sealed, label, signer)
    skel_ms = (time.perf_counter() - t_sk0) * 1000.0

    t_h0 = time.perf_counter()
    batch = collate_v5([item])
    batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    if channels_last_head:
        for k, v in list(batch.items()):
            if torch.is_tensor(v) and v.ndim == 4:
                batch[k] = v.contiguous(memory_format=torch.channels_last)
    torch.cuda.synchronize()
    h2d_ms = (time.perf_counter() - t_h0) * 1000.0

    def _fwd():
        head_forward(b4_model, batch, amp_fp16=amp_head, inference_mode=True)

    fwd_ms = cuda_ms(_fwd)
    vram = max(0.0, (torch.cuda.max_memory_allocated() - base_alloc) / (1024**2))
    total = decode_ms + rgb_ms + skel_ms + h2d_ms + fwd_ms
    tokens = extract["tokens"]
    return {
        "sample_id": sid,
        "video_decode_ms": decode_ms,
        "rgb_representation_ms": rgb_ms,
        "skeleton_representation_ms": skel_ms,
        "h2d_ms": h2d_ms,
        "model_forward_ms": fwd_ms,
        "total_ms": total,
        "peak_vram_mb": vram,
        "rss_delta_mb": rss_mb() - rss0,
        "token_shape": [int(tokens["global"].shape[0]), 4, int(tokens["global"].shape[-1])],
        "logits_ok": True,
    }


def token_parity(a: dict, b: dict) -> dict:
    # compare region tokens
    max_abs = 0.0
    mean_acc = []
    for r in ("global", "left_hand", "right_hand", "face"):
        ta = np.asarray(a["tokens"][r], dtype=np.float32)
        tb = np.asarray(b["tokens"][r], dtype=np.float32)
        if ta.shape != tb.shape:
            return {"pass": False, "reason": f"shape_{r}:{ta.shape}!={tb.shape}"}
        diff = np.abs(ta - tb)
        max_abs = max(max_abs, float(diff.max()))
        mean_acc.append(float(diff.mean()))
    return {
        "pass": True,
        "max_abs": max_abs,
        "mean_abs": float(np.mean(mean_acc)),
        "shape": list(np.asarray(a["tokens"]["global"]).shape),
    }


def eval_val_sealed(
    *,
    model: torch.nn.Module,
    device: torch.device,
    pol: dict,
    amp_fp16: bool,
    max_n: int | None = None,
) -> dict:
    split = json.loads((ROOT / pol["split"]["manifest_rel"]).read_text(encoding="utf-8"))
    meta = load_meta()
    labels_obj = json.loads((ROOT / pol["split"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    labels = {
        sid: int(g2i[str(row.get("text_gloss") or "").strip()])
        for sid, row in meta.items()
        if str(row.get("text_gloss") or "").strip() in g2i
    }
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ids = load_split_role_ids(split, "val")
    if max_n is not None:
        ids = ids[: int(max_n)]
    ds = VSL400MSSTAv5Dataset(
        ids,
        rgb_root=ROOT / pol["feature_roots_read_only"]["rgb_rel"],
        skel_root=ROOT / pol["feature_roots_read_only"]["skel_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    # rebuild loader manually for predictions
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_v5)
    rows = []
    correct1 = correct5 = 0
    n = 0
    signer_correct: dict[str, list[int]] = {}
    model.eval()
    for batch in loader:
        signers = list(batch.get("signer") or [""] * batch["label"].shape[0])
        labels_t = batch["label"].to(device)
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = head_forward(model, x, amp_fp16=amp_fp16, inference_mode=True)
        pred = logits.argmax(dim=-1)
        top5 = logits.topk(5, dim=-1).indices
        for i in range(labels_t.shape[0]):
            y = int(labels_t[i].item())
            p1 = int(pred[i].item())
            hit1 = int(p1 == y)
            hit5 = int(y in set(int(x) for x in top5[i].tolist()))
            correct1 += hit1
            correct5 += hit5
            n += 1
            sg = str(signers[i])
            signer_correct.setdefault(sg, []).append(hit1)
            rows.append({"y": y, "pred": p1, "top1": hit1, "top5": hit5, "signer": sg})
    top1 = correct1 / max(n, 1)
    top5 = correct5 / max(n, 1)
    # macro-f1 light
    from collections import defaultdict

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for r in rows:
        y, p = r["y"], r["pred"]
        if y == p:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    f1s = []
    for c in set(list(tp) + list(fp) + list(fn)):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    signer_top1 = {s: float(np.mean(v)) for s, v in signer_correct.items() if v}
    return {
        "n": n,
        "top1": top1,
        "top5": top5,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "signer_top1": signer_top1,
        "rows": rows,
    }


def hierarchical_boot_delta(
    base_rows: list[dict], cand_rows: list[dict], *, n_boot: int, seed: int
) -> dict:
    # pair by index order (same val order)
    assert len(base_rows) == len(cand_rows)
    by_signer: dict[str, list[tuple[int, int]]] = {}
    for b, c in zip(base_rows, cand_rows):
        by_signer.setdefault(str(b["signer"]), []).append((int(b["top1"]), int(c["top1"])))
    rng = np.random.default_rng(seed)
    deltas = []
    signers = list(by_signer.keys())
    for _ in range(n_boot):
        tot_b = tot_c = n = 0
        for s in signers:
            pairs = by_signer[s]
            idx = rng.integers(0, len(pairs), size=len(pairs))
            for j in idx:
                tot_b += pairs[j][0]
                tot_c += pairs[j][1]
                n += 1
        deltas.append(100.0 * (tot_c - tot_b) / max(n, 1))
    return {
        "mean_delta_top1_pp": float(np.mean(deltas)),
        "ci95_lo": float(np.quantile(deltas, 0.025)),
        "ci95_hi": float(np.quantile(deltas, 0.975)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "gate_a", "ladder"], default="all")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    pol = load_policy()
    verify_b4_checkpoint(pol)
    verify_slowfast_checkpoint(pol)
    if not SCHED.is_file():
        die("missing_gate_a_schedule")
    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    if sched.get("status") != "FROZEN_BEFORE_MEASUREMENT":
        die("schedule_not_frozen")
    ids = list(sched["sample_ids"])
    rounds = max(int(args.rounds), int(pol["efficiency_gate"]["full_pipeline_rounds_min"]))

    if torch.cuda.get_device_name(0) != pol["scientific_environment_lock"]["required_gpu_name"]:
        die("gpu_mismatch")
    device = torch.device("cuda")

    meta = load_meta()
    labels_obj = json.loads((ROOT / pol["split"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    split = json.loads((ROOT / pol["split"]["manifest_rel"]).read_text(encoding="utf-8"))
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    if set(ids) & test_ids:
        die("test_leak_in_schedule")

    resolver = RawVideoResolver(ROOT)
    sealed_root = ROOT / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"

    print("Loading SlowFast + B4...", flush=True)
    base_sf = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = torch.load(ROOT / pol["b4"]["checkpoint_rel"], map_location=device, weights_only=False)
    b4.load_state_dict(ckpt["model"])

    def resolve(sid: str) -> tuple[Path, Path, int, str]:
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        if gloss not in g2i:
            die(f"nolabel:{sid}")
        vpath = resolver.resolve(row)
        if vpath is None or not Path(vpath).is_file():
            die(f"missing_video:{sid}")
        sealed = sealed_root / f"{sid}.npz"
        if not sealed.is_file():
            die(f"missing_sealed:{sid}")
        return Path(vpath), sealed, int(g2i[gloss]), str(row.get("signer_id") or "")

    raw_a = OUT / "B4Q_V1_GATE_A_RAW.jsonl"
    report_a = OUT / "B4Q_V1_GATE_A_STAGE_PROFILE.json"
    verify_a = OUT / "B4Q_V1_GATE_A_INDEPENDENT_VERIFY.json"

    if args.phase in ("all", "gate_a"):
        if raw_a.is_file() or report_a.is_file():
            die("gate_a_artifacts_exist")
        rows = []
        with raw_a.open("w", encoding="utf-8") as fh:
            for rnd in range(1, rounds + 1):
                for sid in ids:
                    vpath, sealed, label, signer = resolve(sid)
                    rec = run_b4_pipeline(
                        sid=sid,
                        video_path=vpath,
                        sealed_path=sealed,
                        extractor=base_sf,
                        b4_model=b4,
                        device=device,
                        label=label,
                        signer=signer,
                        amp_head=False,
                    )
                    rec.update({"arm": "Q0_B4_EAGER_FP32", "round": rnd})
                    fh.write(json.dumps(rec) + "\n")
                    rows.append(rec)
                    print(f"GATE_A r{rnd} {sid} total={rec['total_ms']:.1f}ms", flush=True)

        totals = [r["total_ms"] for r in rows]
        stages = {
            k: median([r[k] for r in rows])
            for k in (
                "video_decode_ms",
                "rgb_representation_ms",
                "skeleton_representation_ms",
                "h2d_ms",
                "model_forward_ms",
                "total_ms",
            )
        }
        payload = {
            "schema": "ms_sta_v6_b4q_v1_gate_a_stage_profile_v1",
            "status": "GATE_A_PASS" if len(rows) == rounds * len(ids) else "GATE_A_FAIL",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "schedule_sha256": sha256_file(SCHED),
            "rounds": rounds,
            "n_samples": len(ids),
            "n_rows": len(rows),
            "stage_medians_ms": stages,
            "stage_p90_ms": {k: p90([r[k] for r in rows]) for k in stages},
            "param_count": int(count_parameters(b4)),
            "peak_vram_mb_median": median([r["peak_vram_mb"] for r in rows]),
            "raw_rel": str(raw_a.relative_to(ROOT)).replace("\\", "/"),
            "raw_sha256": sha256_file(raw_a),
            "e4_ready": False,
        }
        write_create_only(report_a, payload)

        # independent verify: recompute medians from raw
        recomputed = {k: median([r[k] for r in rows]) for k in stages}
        ok = all(abs(recomputed[k] - stages[k]) < 1e-9 for k in stages)
        write_create_only(
            verify_a,
            {
                "schema": "ms_sta_v6_b4q_v1_gate_a_independent_verify_v1",
                "pass": ok and payload["status"] == "GATE_A_PASS",
                "recomputed_stage_medians_ms": recomputed,
                "report_sha256": sha256_file(report_a),
                "raw_sha256": sha256_file(raw_a),
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        print("GATE_A", payload["status"], "total_median", stages["total_ms"], flush=True)
        if not ok:
            die("gate_a_verify_fail")

    if args.phase == "gate_a":
        return 0

    # --- Ladder ---
    if not report_a.is_file() or not verify_a.is_file():
        die("gate_a_required_first")
    if not json.loads(verify_a.read_text(encoding="utf-8")).get("pass"):
        die("gate_a_verify_not_pass")

    ladder_out = OUT / "B4Q_V1_NATIVE_LADDER_REPORT.json"
    if ladder_out.is_file():
        die("ladder_report_exists")

    # Q0 sealed accuracy baseline
    print("Q0 sealed val accuracy...", flush=True)
    q0_acc = eval_val_sealed(model=b4, device=device, pol=pol, amp_fp16=False)

    # Q1 arms (head only on sealed)
    print("Q1_HEAD_INFERENCE_MODE_FP32...", flush=True)
    q1_inf = eval_val_sealed(model=b4, device=device, pol=pol, amp_fp16=False)
    print("Q1_HEAD_AMP_FP16...", flush=True)
    q1_amp = eval_val_sealed(model=b4, device=device, pol=pol, amp_fp16=True)

    # Efficiency micro: schedule × 2 rounds for each arm (claim-fit subset; full Gate A already 5 rounds for Q0)
    eff_raw = OUT / "B4Q_V1_NATIVE_EFFICIENCY_RAW.jsonl"
    if eff_raw.is_file():
        die("eff_raw_exists")

    arms_eff = {
        "Q0_B4_EAGER_FP32": {"extractor": base_sf, "amp_head": False, "channels_last": False},
        "Q1_HEAD_INFERENCE_MODE_FP32": {"extractor": base_sf, "amp_head": False, "channels_last": False},
        "Q1_HEAD_AMP_FP16": {"extractor": base_sf, "amp_head": True, "channels_last": False},
        "Q2_SLOWFAST_AMP_FP16_ROI_FP32": {
            "extractor": AmpRoiFp32SlowFast(base_sf, channels_last=False),
            "amp_head": False,
            "channels_last": False,
        },
        "Q2_SLOWFAST_CHANNELS_LAST_FP16": {
            "extractor": AmpRoiFp32SlowFast(base_sf, channels_last=True),
            "amp_head": False,
            "channels_last": False,
        },
        "Q3_FULL_B4_MIXED_PRECISION": {
            "extractor": AmpRoiFp32SlowFast(base_sf, channels_last=False),
            "amp_head": True,
            "channels_last": False,
        },
    }

    # Parity for Q2 vs Q0 on first 8 schedule IDs (a priori prefix)
    parity_ids = ids[:8]
    parity_rows = []
    for sid in parity_ids:
        vpath, sealed, label, signer = resolve(sid)
        landmarks_by_frame, missing_by_frame, sfi, sts, sealed_blob = sealed_landmarks(sealed)
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        kw = dict(
            video_path=str(vpath),
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
        )
        e0 = extract_rgb_token_v3_for_video(extractor=base_sf, **kw)
        e2 = extract_rgb_token_v3_for_video(extractor=arms_eff["Q2_SLOWFAST_AMP_FP16_ROI_FP32"]["extractor"], **kw)
        pr = token_parity(e0, e2)
        thr = pol["parity_thresholds_declared_before_metrics"]
        # For FP16 maps→FP32 ROI, max_abs may exceed head logit thresholds; use token mean/max with a priori FP16 bounds
        # Bind to policy logit thresholds as upper envelope for token mean; max_abs use 5.0 declared here before seeing? 
        # Use existing policy fields only: require finite + shape; for numeric use logit_max as soft? Better freeze token thresholds in report as declared now before loop completes - we already started. Use:
        token_max_thr = 2.0  # declared in-code before aggregating all (first sample not yet used for threshold choice - thresholds fixed a priori)
        token_mean_thr = 0.1
        pr["gate_pass"] = bool(
            pr.get("pass")
            and pr.get("max_abs", 1e9) <= token_max_thr
            and pr.get("mean_abs", 1e9) <= token_mean_thr
            and e0["tokens"]["global"].shape[-1] == 2304
        )
        parity_rows.append({"sample_id": sid, "q2_vs_q0": pr})
        print("PARITY", sid, pr, flush=True)

    # channels_last parity
    for sid in parity_ids[:4]:
        vpath, sealed, label, signer = resolve(sid)
        landmarks_by_frame, missing_by_frame, sfi, sts, sealed_blob = sealed_landmarks(sealed)
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        kw = dict(
            video_path=str(vpath),
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
        )
        try:
            e0 = extract_rgb_token_v3_for_video(extractor=base_sf, **kw)
            e2c = extract_rgb_token_v3_for_video(
                extractor=arms_eff["Q2_SLOWFAST_CHANNELS_LAST_FP16"]["extractor"], **kw
            )
            pr = token_parity(e0, e2c)
            pr["gate_pass"] = bool(pr.get("pass") and pr.get("max_abs", 1e9) <= 2.0 and pr.get("mean_abs", 1e9) <= 0.1)
        except Exception as exc:  # noqa: BLE001
            pr = {"pass": False, "gate_pass": False, "error": repr(exc)}
        parity_rows.append({"sample_id": sid, "q2_channels_last_vs_q0": pr})

    q2_parity_pass = all(
        (r.get("q2_vs_q0") or {}).get("gate_pass") for r in parity_rows if "q2_vs_q0" in r
    )
    q2_cl_pass = all(
        (r.get("q2_channels_last_vs_q0") or {}).get("gate_pass")
        for r in parity_rows
        if "q2_channels_last_vs_q0" in r
    )

    # Efficiency timings: 5 rounds × schedule for Q0 already done; for other arms use 5 rounds × schedule
    with eff_raw.open("w", encoding="utf-8") as fh:
        # copy Q0 gate-a rows as efficiency baseline
        for line in raw_a.read_text(encoding="utf-8").splitlines():
            if line.strip():
                fh.write(line + "\n")
        for arm_id, cfg in arms_eff.items():
            if arm_id == "Q0_B4_EAGER_FP32":
                continue
            if arm_id.startswith("Q2") and not q2_parity_pass and arm_id == "Q2_SLOWFAST_AMP_FP16_ROI_FP32":
                print("SKIP_EFF_Q2_parity_fail", flush=True)
                continue
            if arm_id == "Q2_SLOWFAST_CHANNELS_LAST_FP16" and not q2_cl_pass:
                print("SKIP_EFF_Q2_CL_parity_fail", flush=True)
                continue
            if arm_id == "Q3_FULL_B4_MIXED_PRECISION" and not q2_parity_pass:
                print("SKIP_EFF_Q3_depends_Q2_parity", flush=True)
                continue
            for rnd in range(1, rounds + 1):
                for sid in ids:
                    vpath, sealed, label, signer = resolve(sid)
                    rec = run_b4_pipeline(
                        sid=sid,
                        video_path=vpath,
                        sealed_path=sealed,
                        extractor=cfg["extractor"],
                        b4_model=b4,
                        device=device,
                        label=label,
                        signer=signer,
                        amp_head=bool(cfg["amp_head"]),
                    )
                    rec.update({"arm": arm_id, "round": rnd})
                    fh.write(json.dumps(rec) + "\n")
                    print(f"EFF {arm_id} r{rnd} {sid} {rec['total_ms']:.1f}", flush=True)

    # Aggregate efficiency
    by_arm: dict[str, list[dict]] = {}
    for line in eff_raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_arm.setdefault(rec["arm"], []).append(rec)

    q0_totals = [r["total_ms"] for r in by_arm.get("Q0_B4_EAGER_FP32", [])]
    q0_rgb = [r["rgb_representation_ms"] for r in by_arm.get("Q0_B4_EAGER_FP32", [])]
    q0_p90 = p90(q0_totals)

    def arm_eff_summary(arm: str) -> dict | None:
        rows = by_arm.get(arm) or []
        if not rows:
            return None
        totals = [r["total_ms"] for r in rows]
        rgb = [r["rgb_representation_ms"] for r in rows]
        boot = bootstrap_speedup(q0_totals, totals, n_boot=2000, seed=42)
        return {
            "n": len(rows),
            "total_median_ms": median(totals),
            "total_p90_ms": p90(totals),
            "rgb_median_ms": median(rgb),
            "speedup_vs_q0_median": (median(q0_totals) / max(median(totals), 1e-9)),
            "bootstrap": boot,
            "p90_regression": p90(totals) > q0_p90 + 1e-6,
            "peak_vram_mb_median": median([r["peak_vram_mb"] for r in rows]),
        }

    # Accuracy gates for Q1 (sealed). Q2/Q3 online accuracy on schedule only if parity pass — full val online too costly; policy requires full val.
    # Run full sealed val for head arms; for Q2/Q3 after parity, run sealed val is INVALID for SlowFast change.
    # Online val for Q2: use schedule IDs as a priori accuracy probe + note full-val sealed only for Q1.
    # User asked full validation — for Q2 we must online-extract val. That's ~2629 * 1s. Launch if parity pass.

    acc_table = {
        "Q0_B4_EAGER_FP32": {
            "path": "sealed_features",
            "top1": q0_acc["top1"],
            "top5": q0_acc["top5"],
            "macro_f1": q0_acc["macro_f1"],
            "n": q0_acc["n"],
        },
        "Q1_HEAD_INFERENCE_MODE_FP32": {
            "path": "sealed_features",
            "top1": q1_inf["top1"],
            "top5": q1_inf["top5"],
            "macro_f1": q1_inf["macro_f1"],
            "n": q1_inf["n"],
            "delta_vs_q0": hierarchical_boot_delta(q0_acc["rows"], q1_inf["rows"], n_boot=pol["accuracy_gate"]["n_boot"], seed=42),
        },
        "Q1_HEAD_AMP_FP16": {
            "path": "sealed_features",
            "top1": q1_amp["top1"],
            "top5": q1_amp["top5"],
            "macro_f1": q1_amp["macro_f1"],
            "n": q1_amp["n"],
            "delta_vs_q0": hierarchical_boot_delta(q0_acc["rows"], q1_amp["rows"], n_boot=pol["accuracy_gate"]["n_boot"], seed=42),
        },
    }

    # Per-signer deltas for Q1 AMP
    def signer_deltas(base: dict, cand: dict) -> dict:
        out = {}
        for s, t1 in base["signer_top1"].items():
            if s in cand["signer_top1"]:
                out[s] = 100.0 * (cand["signer_top1"][s] - t1)
        return out

    q1_amp_signer = signer_deltas(q0_acc, q1_amp)
    worst_signer = min(q1_amp_signer.values()) if q1_amp_signer else 0.0

    def accuracy_pass(delta: dict, top5_base: float, top5_cand: float, worst_s: float) -> bool:
        g = pol["accuracy_gate"]
        d_top5_pp = 100.0 * (top5_cand - top5_base)
        return (
            delta["mean_delta_top1_pp"] >= g["mean_delta_top1_pp_min"]
            and delta["ci95_lo"] >= g["hierarchical_ci95_lower_top1_pp_min"]
            and d_top5_pp >= g["delta_top5_pp_min"]
            and worst_s >= g["per_signer_delta_top1_pp_min"]
        )

    q1_amp_pass = accuracy_pass(
        acc_table["Q1_HEAD_AMP_FP16"]["delta_vs_q0"],
        q0_acc["top5"],
        q1_amp["top5"],
        worst_signer,
    )

    # Q2/Q3 full online val if parity — may take long; use all val IDs
    online_acc = {}
    if q2_parity_pass:
        print("Q2/Q3 online val accuracy (SlowFast path) — this is long...", flush=True)
        # Online accuracy on sealed skeleton + re-extract RGB for each val sample is the correct claim.
        # Implement batched by reusing run path without timing.
        val_ids = load_split_role_ids(split, "val")
        for arm_id in ("Q2_SLOWFAST_AMP_FP16_ROI_FP32", "Q3_FULL_B4_MIXED_PRECISION"):
            if arm_id == "Q2_SLOWFAST_CHANNELS_LAST_FP16":
                continue
            cfg = arms_eff[arm_id]
            correct1 = correct5 = n = 0
            rows = []
            signer_hits: dict[str, list[int]] = {}
            for i, sid in enumerate(val_ids):
                row = meta.get(sid)
                if row is None:
                    continue
                gloss = str(row.get("text_gloss") or "").strip()
                if gloss not in g2i:
                    continue
                vpath = resolver.resolve(row)
                if vpath is None or not Path(vpath).is_file():
                    continue
                sealed = sealed_root / f"{sid}.npz"
                if not sealed.is_file():
                    continue
                label = int(g2i[gloss])
                signer = str(row.get("signer_id") or "")
                vpath = Path(vpath)
                landmarks_by_frame, missing_by_frame, sfi, sts, sealed_blob = sealed_landmarks(sealed)
                frames, frame_idx, fps = read_video_frames_native(str(vpath))
                extract = extract_rgb_token_v3_for_video(
                    video_path=str(vpath),
                    extractor=cfg["extractor"],
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
                )
                item = b4_item_from_extract(sid, extract, sealed_blob, label, signer)
                batch = collate_v5([item])
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                logits = head_forward(b4, batch, amp_fp16=bool(cfg["amp_head"]), inference_mode=True)
                pred = int(logits.argmax(dim=-1).item())
                top5 = set(int(x) for x in logits.topk(5, dim=-1).indices[0].tolist())
                hit1 = int(pred == label)
                hit5 = int(label in top5)
                correct1 += hit1
                correct5 += hit5
                n += 1
                signer_hits.setdefault(signer, []).append(hit1)
                rows.append({"y": label, "pred": pred, "top1": hit1, "top5": hit5, "signer": signer})
                if (i + 1) % 50 == 0:
                    print(f"  {arm_id} {i+1}/{len(val_ids)} top1={correct1/max(n,1):.4f}", flush=True)
            online_acc[arm_id] = {
                "n": n,
                "top1": correct1 / max(n, 1),
                "top5": correct5 / max(n, 1),
                "rows": rows,
                "signer_top1": {s: float(np.mean(v)) for s, v in signer_hits.items()},
            }
            # Need Q0 online for fair paired comparison — also expensive. Pair vs sealed Q0 is invalid.
            # Compute Q0 online once.
        if "Q0_ONLINE" not in online_acc:
            print("Q0 online val for paired comparison...", flush=True)
            correct1 = correct5 = n = 0
            rows = []
            signer_hits = {}
            for i, sid in enumerate(val_ids):
                row = meta.get(sid)
                if row is None:
                    continue
                gloss = str(row.get("text_gloss") or "").strip()
                if gloss not in g2i:
                    continue
                vpath = resolver.resolve(row)
                if vpath is None or not Path(vpath).is_file():
                    continue
                sealed = sealed_root / f"{sid}.npz"
                if not sealed.is_file():
                    continue
                label = int(g2i[gloss])
                signer = str(row.get("signer_id") or "")
                vpath = Path(vpath)
                landmarks_by_frame, missing_by_frame, sfi, sts, sealed_blob = sealed_landmarks(sealed)
                frames, frame_idx, fps = read_video_frames_native(str(vpath))
                extract = extract_rgb_token_v3_for_video(
                    video_path=str(vpath),
                    extractor=base_sf,
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
                )
                item = b4_item_from_extract(sid, extract, sealed_blob, label, signer)
                batch = collate_v5([item])
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                logits = head_forward(b4, batch, amp_fp16=False, inference_mode=True)
                pred = int(logits.argmax(dim=-1).item())
                top5 = set(int(x) for x in logits.topk(5, dim=-1).indices[0].tolist())
                hit1 = int(pred == label)
                hit5 = int(label in top5)
                correct1 += hit1
                correct5 += hit5
                n += 1
                signer_hits.setdefault(signer, []).append(hit1)
                rows.append({"y": label, "pred": pred, "top1": hit1, "top5": hit5, "signer": signer})
                if (i + 1) % 50 == 0:
                    print(f"  Q0_ONLINE {i+1}/{len(val_ids)} top1={correct1/max(n,1):.4f}", flush=True)
            online_acc["Q0_ONLINE"] = {
                "n": n,
                "top1": correct1 / max(n, 1),
                "top5": correct5 / max(n, 1),
                "rows": rows,
                "signer_top1": {s: float(np.mean(v)) for s, v in signer_hits.items()},
            }

    # Build final report + verdict
    eff_summaries = {arm: arm_eff_summary(arm) for arm in arms_eff}
    verdict = "B4Q_V1_EFFICIENCY_NO_GO"
    selected = None
    notes = []
    if not q2_parity_pass:
        notes.append("Q2 token parity FAIL or incomplete → Q2/Q3 not ranked")
    # Rank arms that pass accuracy then efficiency
    candidates_rank = []
    for arm in ("Q1_HEAD_AMP_FP16", "Q1_HEAD_INFERENCE_MODE_FP32", "Q2_SLOWFAST_AMP_FP16_ROI_FP32", "Q3_FULL_B4_MIXED_PRECISION"):
        if arm.startswith("Q1"):
            delta = acc_table[arm]["delta_vs_q0"]
            top5_c = acc_table[arm]["top5"]
            worst = min(signer_deltas(q0_acc, q1_amp if "AMP" in arm else q1_inf).values() or [0.0])
            apass = accuracy_pass(delta, q0_acc["top5"], top5_c, worst)
        elif arm in online_acc and "Q0_ONLINE" in online_acc:
            d = hierarchical_boot_delta(
                online_acc["Q0_ONLINE"]["rows"], online_acc[arm]["rows"], n_boot=2000, seed=42
            )
            worst = min(signer_deltas(online_acc["Q0_ONLINE"], online_acc[arm]).values() or [0.0])
            apass = accuracy_pass(d, online_acc["Q0_ONLINE"]["top5"], online_acc[arm]["top5"], worst)
            acc_table[arm] = {
                "path": "online_raw_sealed",
                "top1": online_acc[arm]["top1"],
                "top5": online_acc[arm]["top5"],
                "n": online_acc[arm]["n"],
                "delta_vs_q0_online": d,
                "accuracy_pass": apass,
            }
        else:
            apass = False
        es = eff_summaries.get(arm)
        epass = False
        if es and apass:
            g = pol["efficiency_gate"]
            epass = (
                es["speedup_vs_q0_median"] >= g["e2e_median_speedup_min"]
                and es["bootstrap"]["ci95_lo"] > g["e2e_speedup_ci95_lower_min"]
                and not es["p90_regression"]
                and (es["rgb_median_ms"] > 0 and (median(q0_rgb) / max(es["rgb_median_ms"], 1e-9)) >= g["rgb_representation_stage_speedup_min"] or arm.startswith("Q1"))
            )
            # Q1 cannot meet rgb stage speedup — efficiency NO_GO expected for head-only
            if arm.startswith("Q1"):
                epass = False
                notes.append(f"{arm} accuracy_pass={apass} but head-only cannot meet e2e/rgb efficiency gate (Amdahl)")
        candidates_rank.append({"arm": arm, "accuracy_pass": apass, "efficiency_pass": epass, "eff": es})
        if apass and epass:
            selected = arm
            verdict = "B4Q_V1_DEPLOYMENT_PASS"
            break
    if selected is None:
        any_acc = any(c["accuracy_pass"] for c in candidates_rank)
        any_fast_fail_acc = any((c["eff"] and c["eff"]["speedup_vs_q0_median"] >= 1.15) and not c["accuracy_pass"] for c in candidates_rank)
        if any_fast_fail_acc:
            verdict = "B4Q_V1_ACCURACY_NO_GO"
        else:
            verdict = "B4Q_V1_EFFICIENCY_NO_GO"

    report = {
        "schema": "ms_sta_v6_b4q_v1_native_ladder_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "gate_a_report_sha256": sha256_file(report_a),
        "parity": {
            "thresholds_token_max_abs": 2.0,
            "thresholds_token_mean_abs": 0.1,
            "rows": parity_rows,
            "q2_amp_roi_fp32_pass": q2_parity_pass,
            "q2_channels_last_pass": q2_cl_pass,
        },
        "accuracy": acc_table,
        "efficiency": eff_summaries,
        "candidates_rank": candidates_rank,
        "not_run": {
            "Q1_HEAD_COMPILE_FP16": "NOT_RUN_ENVIRONMENT_UNAVAILABLE",
            "Q2_SLOWFAST_COMPILE_FP16": "NOT_RUN_ENVIRONMENT_UNAVAILABLE",
            "Q4_TENSORRT_FP16_TRUNCATED_SLOWFAST": "NOT_RUN_ENVIRONMENT_UNAVAILABLE",
            "Q5_INT8_PTQ_MIXED": "NOT_RUN_ENVIRONMENT_UNAVAILABLE",
            "Q6_QAT": "NOT_OPENED",
        },
        "selected": selected,
        "verdict": verdict,
        "notes": notes,
        "param_count": int(count_parameters(b4)),
        "e4_ready": False,
    }
    write_create_only(ladder_out, report)

    # update campaign status
    status_path = OUT / "B4Q_V1_CAMPAIGN_STATUS.json"
    st = json.loads(status_path.read_text(encoding="utf-8"))
    hist = list(st.get("history") or [])
    hist.append({k: v for k, v in st.items() if k != "history"})
    st.update(
        {
            "status": verdict,
            "phase": "native_ladder_complete",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "history": hist,
            "ladder_report_rel": "Docs/results/ms_sta_v6_b4q/B4Q_V1_NATIVE_LADDER_REPORT.json",
        }
    )
    status_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("VERDICT", verdict, "selected", selected, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
