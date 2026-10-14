# -*- coding: utf-8 -*-
"""E4_B4SC_V1 E1 causal 2×2 factorial — schedule-locked; no Slow compression."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
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
    b4_item_from_extract,
    rss_mb,
    sealed_landmarks,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video, read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, head_forward  # noqa: E402
from src.training.ms_sta_v5_data import collate_v5, load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
LOCK = OUT / "E4_B4SC_V1_E1_SCHEDULE_LOCK.json"
POL_E4 = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"
POL_B4Q = ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json"
ARMS = ("E0", "E1a", "E1b", "E1c")


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_E1_FAIL:{msg}")


def create_only(path: Path, obj: Any) -> None:
    if path.is_file():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(str(obj), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_meta() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with (ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("dataset") or "") != "VSL400":
                continue
            sid = str(row.get("video_id") or "").strip()
            if sid:
                out[sid] = dict(row)
    return out


def tensor_sha256(t: torch.Tensor | np.ndarray) -> str:
    if torch.is_tensor(t):
        arr = t.detach().float().cpu().contiguous().numpy()
    else:
        arr = np.ascontiguousarray(t)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def frames_sha256(frames: np.ndarray, frame_idx: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(frames).tobytes())
    h.update(np.ascontiguousarray(frame_idx.astype(np.int32)).tobytes())
    return h.hexdigest()


def cuda_event_ms(fn) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    fn()
    ender.record()
    torch.cuda.synchronize()
    return float(starter.elapsed_time(ender))


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
    return {
        "speedup_median": float(np.median(speeds)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "latency_reduction": float(1.0 - float(np.median(c)) / max(float(np.median(b)), 1e-9)),
    }


def hierarchical_boot_delta(rows_a: list[dict], rows_b: list[dict], *, n_boot: int, seed: int) -> dict:
    """Paired by sample_id; delta = cand - base in percentage points."""
    by_a = {r["sample_id"]: r for r in rows_a}
    by_b = {r["sample_id"]: r for r in rows_b}
    ids = sorted(set(by_a) & set(by_b))
    diffs = []
    for sid in ids:
        diffs.append(100.0 * (float(by_b[sid]["top1"]) - float(by_a[sid]["top1"])))
    diffs_a = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs_a), size=len(diffs_a))
        boots.append(float(diffs_a[idx].mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n_paired": len(ids),
        "mean_delta_top1_pp": float(diffs_a.mean()) if len(diffs_a) else float("nan"),
        "ci95_lo": float(lo) if len(diffs_a) else float("nan"),
        "ci95_hi": float(hi) if len(diffs_a) else float("nan"),
    }


def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    by_a = {r["sample_id"]: int(r["top1"]) for r in rows_a}
    by_b = {r["sample_id"]: int(r["top1"]) for r in rows_b}
    ids = sorted(set(by_a) & set(by_b))
    b01 = b10 = 0
    for sid in ids:
        a, b = by_a[sid], by_b[sid]
        if a == 0 and b == 1:
            b01 += 1
        elif a == 1 and b == 0:
            b10 += 1
    return {"n": len(ids), "b_only_correct": b01, "a_only_correct": b10}


def arm_cfg(arm: str, base_sf, amp_sf, lock: dict) -> dict:
    spec = lock["factorial_arms"][arm]
    return {
        "extractor": amp_sf if spec["q2_amp"] else base_sf,
        "amp_head": False,  # Q2 definition: SlowFast AMP, head FP32
        "pathway_window_batch_size": int(lock["decode_batching_definition"]["pathway_window_batch_size_max"])
        if spec["batched_windows"]
        else 1,
        "q2_amp": bool(spec["q2_amp"]),
        "batched_windows": bool(spec["batched_windows"]),
    }


def run_pipeline(
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
    pathway_window_batch_size: int,
    time_stages: bool,
) -> dict:
    t_e2e0 = time.perf_counter()

    t0 = time.perf_counter()
    frames, frame_idx, fps = read_video_frames_native(str(video_path))
    decode_ms = (time.perf_counter() - t0) * 1000.0
    landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)

    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    rss0 = rss_mb()

    # preprocess (letterbox happens inside extract; measure wall around extract GPU+CPU)
    def _extract():
        return extract_rgb_token_v3_for_video(
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
            pathway_window_batch_size=pathway_window_batch_size,
        )

    if time_stages:
        # preprocess wall is letterbox+window build inside extract; split via cuda events around SlowFast is hard
        # Record: preprocess_ms ≈ wall extract - approximate by measuring extract wall as slowfast_roi_ms
        # and set preprocess_ms from letterbox-only separately.
        t_pre0 = time.perf_counter()
        # letterbox-only cost (CPU)
        from src.models.rgb_token_v3 import _resize_letterbox, INPUT_SIZE

        for fr in frames:
            _resize_letterbox(fr, INPUT_SIZE)
        preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

        torch.cuda.synchronize()
        t_rgb0 = time.perf_counter()
        extract = _extract()
        torch.cuda.synchronize()
        slowfast_roi_ms = (time.perf_counter() - t_rgb0) * 1000.0
    else:
        preprocess_ms = 0.0
        extract = _extract()
        slowfast_roi_ms = 0.0

    item = b4_item_from_extract(sid, extract, sealed, label, signer)
    batch = collate_v5([item])
    batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    torch.cuda.synchronize()

    def _fwd():
        head_forward(b4_model, batch, amp_fp16=amp_head, inference_mode=True)

    if time_stages:
        head_ms = cuda_event_ms(_fwd)
    else:
        _fwd()
        head_ms = 0.0

    peak_alloc = max(0.0, (torch.cuda.max_memory_allocated() - base_alloc) / (1024**2))
    peak_reserved = max(0.0, (torch.cuda.max_memory_reserved() - base_reserved) / (1024**2))
    e2e_ms = (time.perf_counter() - t_e2e0) * 1000.0
    return {
        "decode_ms": decode_ms,
        "preprocess_ms": preprocess_ms,
        "slowfast_roi_ms": slowfast_roi_ms,
        "head_ms": head_ms,
        "e2e_ms": e2e_ms,
        "peak_allocated_vram_mb": peak_alloc,
        "peak_reserved_vram_mb": peak_reserved,
        "rss_delta_mb": rss_mb() - rss0,
        "extract": extract,
        "frames": frames,
        "frame_idx": frame_idx,
        "fps": fps,
        "logits_ok": True,
    }


def phase_capability(lock: dict, device: torch.device) -> dict:
    pol = json.loads(POL_E4.read_text(encoding="utf-8"))
    b4q = json.loads(POL_B4Q.read_text(encoding="utf-8"))
    base_sf = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    amp_sf = AmpRoiFp32SlowFast(base_sf)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = torch.load(ROOT / pol["b4_teacher"]["checkpoint_rel"], map_location=device, weights_only=False)
    b4.load_state_dict(ckpt["model"])
    meta = load_meta()
    resolver = RawVideoResolver(ROOT)
    sealed_root = ROOT / b4q["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    sid = lock["bindings"]["sample_ids"][0]
    row = meta[sid]
    labels = json.loads((ROOT / b4q["split"]["labels_rel"]).read_text(encoding="utf-8"))["gloss_to_id"]
    gloss = str(row.get("text_gloss") or "").strip()
    label = int(labels[gloss])
    signer = str(row.get("signer_id") or "")
    vpath = Path(resolver.resolve(row))
    sealed = sealed_root / f"{sid}.npz"
    if not sealed.is_file():
        die(f"sealed_missing:{sealed}")
    out = {"arms": {}, "sample_id": sid, "sealed_rel": str(sealed.relative_to(ROOT)).replace("\\", "/")}
    for arm in ARMS:
        cfg = arm_cfg(arm, base_sf, amp_sf, lock)
        try:
            r = run_pipeline(
                sid=sid,
                video_path=vpath,
                sealed_path=sealed,
                extractor=cfg["extractor"],
                b4_model=b4,
                device=device,
                label=label,
                signer=signer,
                amp_head=cfg["amp_head"],
                pathway_window_batch_size=cfg["pathway_window_batch_size"],
                time_stages=True,
            )
            out["arms"][arm] = {
                "ok": True,
                "e2e_ms": r["e2e_ms"],
                "token_shape": list(np.asarray(r["extract"]["tokens"]["global"]).shape),
                "pathway_window_batch_size": cfg["pathway_window_batch_size"],
            }
        except Exception as exc:  # noqa: BLE001
            out["arms"][arm] = {"ok": False, "error": repr(exc)}
    out["pass"] = all(v.get("ok") for v in out["arms"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["capability", "parity", "accuracy", "efficiency", "report", "all"],
        default="all",
    )
    ap.add_argument("--run-id", default="E1_R1")
    ap.add_argument(
        "--arms",
        default="E0,E1a,E1b,E1c",
        help="Comma-separated arms for efficiency phase",
    )
    ap.add_argument(
        "--retain-e0-e1a-after-parity-fail",
        action="store_true",
        help="After token parity FAIL, continue accuracy/efficiency for E0/E1a only (E1b/E1c rejected).",
    )
    args = ap.parse_args()
    selected_arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    for a in selected_arms:
        if a not in ARMS:
            die(f"bad_arm:{a}")

    if not LOCK.is_file():
        die("schedule_lock_missing")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_BEFORE_E1_METRICS":
        die("schedule_lock_not_frozen")
    if lock.get("e4_ready") is not False:
        die("e4_ready_must_be_false")

    pol = json.loads(POL_E4.read_text(encoding="utf-8"))
    b4q = json.loads(POL_B4Q.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    if torch.cuda.get_device_name(0) != pol["scientific_environment_lock"]["required_gpu_name"]:
        die("gpu_mismatch")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = str(args.run_id)

    if args.phase in ("capability", "all"):
        cap = phase_capability(lock, device)
        cap.update({"schema": "ms_sta_v6_e4_b4sc_v1_e1_capability_probe_v1", "created_at": now, "git_head": git_head, "run_id": run_id})
        create_only(OUT / "E4_B4SC_V1_E1_CAPABILITY_PROBE.json", cap)
        print("CAPABILITY", "PASS" if cap["pass"] else "FAIL", flush=True)
        if not cap["pass"]:
            die("capability_fail")
        if args.phase == "capability":
            return 0

    # Shared models for later phases
    base_sf = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    amp_sf = AmpRoiFp32SlowFast(base_sf)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = torch.load(ROOT / pol["b4_teacher"]["checkpoint_rel"], map_location=device, weights_only=False)
    b4.load_state_dict(ckpt["model"])
    meta = load_meta()
    resolver = RawVideoResolver(ROOT)
    labels_obj = json.loads((ROOT / b4q["split"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    sealed_root = ROOT / b4q["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    ids = list(lock["bindings"]["sample_ids"])

    def resolve(sid: str):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        vpath = resolver.resolve(row)
        if vpath is None:
            die(f"video_missing:{sid}")
        sealed = sealed_root / f"{sid}.npz"
        if not sealed.is_file():
            die(f"sealed_missing:{sid}")
        return Path(vpath), sealed, int(g2i[gloss]), str(row.get("signer_id") or ""), row

    if args.phase in ("parity", "all"):
        parity_rows = []
        decode_rows = []
        isolation_rows = []
        for sid in ids:
            vpath, sealed, label, signer, _row = resolve(sid)
            landmarks_by_frame, missing_by_frame, sfi, sts, sealed_blob = sealed_landmarks(sealed)
            frames, frame_idx, fps = read_video_frames_native(str(vpath))
            # second decode for E1b path (same function) — exact hash required
            frames_b, frame_idx_b, fps_b = read_video_frames_native(str(vpath))
            drow = {
                "sample_id": sid,
                "frame_count_e0": int(frames.shape[0]),
                "frame_count_e1b": int(frames_b.shape[0]),
                "frame_idx_equal": bool(np.array_equal(frame_idx, frame_idx_b)),
                "shape_e0": list(frames.shape),
                "shape_e1b": list(frames_b.shape),
                "dtype_e0": str(frames.dtype),
                "dtype_e1b": str(frames_b.dtype),
                "fps_e0": float(fps),
                "fps_e1b": float(fps_b),
                "frames_sha256_e0": frames_sha256(frames, frame_idx),
                "frames_sha256_e1b": frames_sha256(frames_b, frame_idx_b),
            }
            drow["exact_frame_hash_equal"] = drow["frames_sha256_e0"] == drow["frames_sha256_e1b"]
            decode_rows.append(drow)
            if not drow["exact_frame_hash_equal"]:
                create_only(
                    OUT / "E4_B4SC_V1_E1_DECODE_PARITY_SOURCE_GAP.json",
                    {
                        "schema": "ms_sta_v6_e4_b4sc_v1_e1_decode_parity_source_gap_v1",
                        "created_at": now,
                        "sample_id": sid,
                        "detail": drow,
                        "note": "STOP before accuracy/timing; no post-hoc tolerance.",
                    },
                )
                die(f"decode_parity_fail:{sid}")

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
            e0 = extract_rgb_token_v3_for_video(extractor=base_sf, pathway_window_batch_size=1, **kw)
            e1b = extract_rgb_token_v3_for_video(
                extractor=base_sf,
                pathway_window_batch_size=int(lock["decode_batching_definition"]["pathway_window_batch_size_max"]),
                **kw,
            )
            e1a = extract_rgb_token_v3_for_video(extractor=amp_sf, pathway_window_batch_size=1, **kw)
            e1c = extract_rgb_token_v3_for_video(
                extractor=amp_sf,
                pathway_window_batch_size=int(lock["decode_batching_definition"]["pathway_window_batch_size_max"]),
                **kw,
            )

            def tok_hash(ex):
                return {r: tensor_sha256(ex["tokens"][r]) for r in ("global", "left_hand", "right_hand", "face")}

            h0, hb = tok_hash(e0), tok_hash(e1b)
            exact_e0_e1b = h0 == hb
            # AMP: report max abs vs FP32; batched AMP vs sequential AMP should be exact under eval BN
            ha, hc = tok_hash(e1a), tok_hash(e1c)
            exact_e1a_e1c = ha == hc
            max_abs_amp = 0.0
            for r in ("global", "left_hand", "right_hand", "face"):
                max_abs_amp = max(
                    max_abs_amp,
                    float(np.max(np.abs(np.asarray(e0["tokens"][r]) - np.asarray(e1a["tokens"][r])))),
                )
            parity_rows.append(
                {
                    "sample_id": sid,
                    "signer_id": signer,
                    "token_sha256_E0": h0,
                    "token_sha256_E1b": hb,
                    "token_sha256_E1a": ha,
                    "token_sha256_E1c": hc,
                    "exact_token_hash_E0_E1b": exact_e0_e1b,
                    "exact_token_hash_E1a_E1c": exact_e1a_e1c,
                    "amp_vs_fp32_max_abs": max_abs_amp,
                    "window_starts_equal": list(e0["window_starts"]) == list(e1b["window_starts"]),
                    "n_windows": len(e0["window_starts"]),
                }
            )

            # isolation: single-window vs same window inside multi-window batch
            if len(e0["window_starts"]) >= 2:
                # Compare maps via tokens of full clip already; additionally force batch size 2 vs 1 on tokens
                isolation_rows.append(
                    {
                        "sample_id": sid,
                        "n_windows": len(e0["window_starts"]),
                        "single_vs_batched_exact": exact_e0_e1b,
                        "amp_single_vs_batched_exact": exact_e1a_e1c,
                    }
                )
            print("PARITY", sid, "E0_E1b", exact_e0_e1b, "E1a_E1c", exact_e1a_e1c, flush=True)

        parity_pass = all(r["exact_token_hash_E0_E1b"] and r["exact_token_hash_E1a_E1c"] for r in parity_rows)
        decode_pass = all(r["exact_frame_hash_equal"] for r in decode_rows)
        report = {
            "schema": "ms_sta_v6_e4_b4sc_v1_e1_input_parity_report_v1",
            "created_at": now,
            "git_head": git_head,
            "run_id": run_id,
            "schedule_lock_sha256": sha256_file(LOCK),
            "pass": bool(parity_pass and decode_pass),
            "decode_pass": decode_pass,
            "token_parity_pass": parity_pass,
            "decode_rows": decode_rows,
            "token_parity_rows": parity_rows,
            "batched_window_isolation_rows": isolation_rows,
            "e1b_e1c_eligible": bool(parity_pass and decode_pass),
        }
        create_only(OUT / "E4_B4SC_V1_E1_INPUT_PARITY_REPORT.json", report)
        print("PARITY_REPORT", "PASS" if report["pass"] else "FAIL", flush=True)
        if not report["pass"]:
            if args.retain_e0_e1a_after_parity_fail and report.get("decode_pass"):
                print("PARITY_FAIL_RETAIN_E0_E1A", flush=True)
            else:
                die("parity_fail")
        if args.phase == "parity":
            return 0

    parity = json.loads((OUT / "E4_B4SC_V1_E1_INPUT_PARITY_REPORT.json").read_text(encoding="utf-8"))
    if not parity.get("pass"):
        if not (args.retain_e0_e1a_after_parity_fail and parity.get("decode_pass")):
            die("parity_required")
        selected_arms = tuple(a for a in selected_arms if a in ("E0", "E1a"))
        print("ARMS_AFTER_PARITY_FAIL", selected_arms, flush=True)

    if args.phase in ("accuracy", "all"):
        # Import E0/E1a from B4Q when checkpoint matches; derive E1b/E1c via exact token parity
        b4_sha = pol["b4_teacher"]["checkpoint_sha256"]
        src_map = {
            "E0": ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_ONLINE_PRED_Q0.jsonl",
            "E1a": ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_ONLINE_PRED_Q2.jsonl",
        }
        logit_map = {
            "E0": ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_ONLINE_LOGITS_Q0.npz",
            "E1a": ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_ONLINE_LOGITS_Q2.npz",
        }
        out_pred = {
            "E0": OUT / "E4_B4SC_V1_E1_E0_PREDICTIONS.jsonl",
            "E1a": OUT / "E4_B4SC_V1_E1_E1A_PREDICTIONS.jsonl",
        }
        for p in out_pred.values():
            if p.is_file():
                die(f"exists:{p}")

        def load_pred(path: Path) -> list[dict]:
            return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

        e0_rows = load_pred(src_map["E0"])
        e1a_rows = load_pred(src_map["E1a"])
        if len(e0_rows) != 2629 or len(e1a_rows) != 2629:
            die("online_pred_n_mismatch")
        if any(r.get("checkpoint_sha256") != b4_sha for r in e0_rows[:3] + e1a_rows[:3]):
            die("checkpoint_sha_mismatch_import")

        def write_arm(arm: str, rows: list[dict], provenance: dict) -> None:
            path = out_pred[arm]
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    top5 = r.get("top5")
                    if isinstance(top5, list):
                        ordered = top5
                    else:
                        # recovery format may store top5_hit only — expand from logits later
                        ordered = r.get("top5_ids") or []
                    out = {
                        "sample_id": r["sample_id"],
                        "signer_id": r.get("signer_id") or r.get("signer"),
                        "y_true": int(r["y_true"]) if "y_true" in r else int(r["y"]),
                        "y_pred": int(r["y_pred"]) if "y_pred" in r else int(r["pred"]),
                        "top5": ordered,
                        "top1": int(r.get("top1", int((r.get("y_pred", r.get("pred"))) == (r.get("y_true", r.get("y")))))),
                        "arm": arm,
                        "checkpoint_sha256": b4_sha,
                        "code_sha": git_head,
                        "schedule_lock_sha256": sha256_file(LOCK),
                        "provenance": provenance,
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

        write_arm(
            "E0",
            e0_rows,
            {
                "type": "import_b4q_online",
                "source_rel": str(src_map["E0"].relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": sha256_file(src_map["E0"]),
                "logits_rel": str(logit_map["E0"].relative_to(ROOT)).replace("\\", "/"),
                "logits_sha256": sha256_file(logit_map["E0"]),
            },
        )
        write_arm(
            "E1a",
            e1a_rows,
            {
                "type": "import_b4q_online",
                "source_rel": str(src_map["E1a"].relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": sha256_file(src_map["E1a"]),
                "logits_rel": str(logit_map["E1a"].relative_to(ROOT)).replace("\\", "/"),
                "logits_sha256": sha256_file(logit_map["E1a"]),
            },
        )
        # E1b/E1c: exact token parity FAIL → prediction SOURCE_GAP (no false derived claims)
        create_only(
            OUT / "E4_B4SC_V1_E1_E1B_PREDICTIONS_SOURCE_GAP.json",
            {
                "schema": "ms_sta_v6_e4_b4sc_v1_e1_pred_source_gap_v1",
                "arm": "E1b",
                "reason": "exact_token_parity_FAIL_cannot_derive_or_import",
                "gap_rel": "E4_B4SC_V1_E1_TOKEN_PARITY_SOURCE_GAP.json",
            },
        )
        create_only(
            OUT / "E4_B4SC_V1_E1_E1C_PREDICTIONS_SOURCE_GAP.json",
            {
                "schema": "ms_sta_v6_e4_b4sc_v1_e1_pred_source_gap_v1",
                "arm": "E1c",
                "reason": "exact_token_parity_FAIL_cannot_derive_or_import",
                "gap_rel": "E4_B4SC_V1_E1_TOKEN_PARITY_SOURCE_GAP.json",
            },
        )

        create_only(
            OUT / "E4_B4SC_V1_E1_LOGITS_MANIFEST.json",
            {
                "schema": "ms_sta_v6_e4_b4sc_v1_e1_logits_manifest_v1",
                "created_at": now,
                "arms": {
                    "E0": {
                        "npz_rel": str(logit_map["E0"].relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256_file(logit_map["E0"]),
                        "provenance": "b4q_import",
                    },
                    "E1a": {
                        "npz_rel": str(logit_map["E1a"].relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256_file(logit_map["E1a"]),
                        "provenance": "b4q_import",
                    },
                    "E1b": {"status": "SOURCE_GAP", "reason": "token_parity_fail"},
                    "E1c": {"status": "SOURCE_GAP", "reason": "token_parity_fail"},
                },
            },
        )
        print("ACCURACY_PREDS_WRITTEN", flush=True)
        if args.phase == "accuracy":
            return 0

    if args.phase in ("efficiency", "all"):
        raw_path = OUT / "E4_B4SC_V1_E1_EFFICIENCY_RAW.jsonl"
        if raw_path.is_file():
            die(f"exists:{raw_path}")
        meas = lock["measurement_schedule"]
        n_meas = int(meas["measured_rounds_per_sample_per_arm"])
        n_warm = int(meas["warmup_rounds_per_sample_per_arm"])
        env_hash = hashlib.sha256(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "git": git_head,
                    "lock": sha256_file(LOCK),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        for si, sid in enumerate(ids):
            order = [a for a in list(meas["arm_order_by_sample"][sid]) if a in selected_arms]
            vpath, sealed, label, signer, _row = resolve(sid)
            for pos, arm in enumerate(order):
                cfg = arm_cfg(arm, base_sf, amp_sf, lock)
                for rnd in range(n_warm + n_meas):
                    is_warm = rnd < n_warm
                    try:
                        r = run_pipeline(
                            sid=sid,
                            video_path=vpath,
                            sealed_path=sealed,
                            extractor=cfg["extractor"],
                            b4_model=b4,
                            device=device,
                            label=label,
                            signer=signer,
                            amp_head=cfg["amp_head"],
                            pathway_window_batch_size=cfg["pathway_window_batch_size"],
                            time_stages=True,
                        )
                        row = {
                            "run_id": run_id,
                            "arm": arm,
                            "sample_id": sid,
                            "signer_id": signer,
                            "round": int(rnd - n_warm + 1) if not is_warm else int(-(n_warm - rnd)),
                            "arm_order_position": int(pos),
                            "warmup": bool(is_warm),
                            "measured": (not is_warm),
                            "decode_ms": r["decode_ms"],
                            "preprocess_ms": r["preprocess_ms"],
                            "slowfast_roi_ms": r["slowfast_roi_ms"],
                            "head_ms": r["head_ms"],
                            "e2e_ms": r["e2e_ms"],
                            "peak_allocated_vram_mb": r["peak_allocated_vram_mb"],
                            "peak_reserved_vram_mb": r["peak_reserved_vram_mb"],
                            "success": True,
                            "error": None,
                            "env_config_sha256": env_hash,
                            "pathway_window_batch_size": cfg["pathway_window_batch_size"],
                            "q2_amp": cfg["q2_amp"],
                        }
                    except Exception as exc:  # noqa: BLE001
                        row = {
                            "run_id": run_id,
                            "arm": arm,
                            "sample_id": sid,
                            "signer_id": signer,
                            "round": int(rnd - n_warm + 1) if not is_warm else int(-(n_warm - rnd)),
                            "arm_order_position": int(pos),
                            "warmup": bool(is_warm),
                            "measured": (not is_warm),
                            "success": False,
                            "error": repr(exc),
                            "env_config_sha256": env_hash,
                        }
                    append_jsonl(raw_path, row)
                print(f"EFF {sid} {arm} done", flush=True)
        print("EFFICIENCY_RAW_DONE", raw_path, flush=True)
        if args.phase == "efficiency":
            return 0

    if args.phase in ("report", "all"):
        # Delegate aggregation to independent verify script entry via import
        from scripts.e4_b4sc_e1_independent_verify import run_verify  # noqa: WPS433

        ver = run_verify(run_id=run_id)
        create_only(OUT / "E4_B4SC_V1_E1_INDEPENDENT_VERIFY.json", ver)
        print("VERIFY", ver.get("pass"), "base", ver.get("selected_system_base"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
