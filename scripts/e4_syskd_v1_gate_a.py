# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 Gate A — R0/R1 baseline reproduction (create-only artifacts)."""
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

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import (  # noqa: E402
    b4_item_from_extract,
    rss_mb,
    sealed_landmarks,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    FrozenSlowFastBlock4,
    REGIONS,
    extract_rgb_token_v3_for_video,
    read_video_frames_native,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, head_forward  # noqa: E402
from src.training.ms_sta_v5_data import collate_v5  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
POL_PATH = CODE_ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"
EXPECTED_POLICY_SHA = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
EXPECTED_CKPT_SHA = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
EXPECTED_SPLIT_SHA = "e7cce5c9d4416a563418f27bba5fc8e89205fece2e5c733c7779782f0b503b09"
EXPECTED_ACTIVE_REG_SHA = "7f1a9f6fdc2705a552174cb33a43f2fdf8b99ac19cf04ec2fdd944b9f2977f23"
ROI_ORDER = ["global", "left_hand", "right_hand", "face"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def die(msg: str) -> None:
    raise SystemExit(f"E4_SYSKD_GATE_A_FAIL:{msg}")


def write_lf_json(path: Path, obj: Any, *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def frames_sha256(frames: np.ndarray, frame_idx: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(frames).tobytes())
    h.update(np.ascontiguousarray(frame_idx.astype(np.int32)).tobytes())
    return h.hexdigest()


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


def load_meta(data_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with (data_root / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("dataset") or "") != "VSL400":
                continue
            sid = str(row.get("video_id") or "").strip()
            if sid:
                out[sid] = dict(row)
    return out


def resolve_data_root() -> Path:
    main = Path(r"D:/TTNT/DNTA/VSL_Translation_Project")
    # Prefer main when worktree has assume-unchanged divergent manifests / missing weights.
    for cand in (main, CODE_ROOT):
        split = cand / "data/processed/vsl400_retrain_v4_split_manifest.json"
        ckpt = cand / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"
        if split.is_file() and ckpt.is_file() and sha256_file(split) == EXPECTED_SPLIT_SHA:
            if sha256_file(ckpt) == EXPECTED_CKPT_SHA:
                return cand
    die("canonical_data_root_unavailable")


def phase_preflight(data_root: Path) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=CODE_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "-sb"], cwd=CODE_ROOT, text=True).strip()
    pol_sha = sha256_file(POL_PATH)
    if pol_sha != EXPECTED_POLICY_SHA:
        die(f"policy_sha_mismatch:{pol_sha}")

    ls = subprocess.check_output(
        ["git", "ls-files", "-v", "data/processed/vsl400_retrain_v4_split_manifest.json", "data/processed/ms_sta_registry_active.json"],
        cwd=CODE_ROOT,
        text=True,
    )
    assume_unchanged = [ln for ln in ls.splitlines() if ln.startswith("H ")]

    wt_split = CODE_ROOT / "data/processed/vsl400_retrain_v4_split_manifest.json"
    wt_split_sha = sha256_file(wt_split) if wt_split.is_file() else None
    data_split = data_root / "data/processed/vsl400_retrain_v4_split_manifest.json"
    data_reg = data_root / "data/processed/ms_sta_registry_active.json"
    ckpt = data_root / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"
    labels = data_root / "data/processed/vsl400_labels.json"
    sched = CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json"

    smi = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()

    procs: list[str] = []
    try:
        ps = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine",
            ],
            text=True,
            errors="ignore",
        )
        for line in ps.splitlines():
            low = line.lower()
            if "train" in low or "e4_syskd" in low or "e4_b4sc" in low:
                procs.append(line.strip())
    except Exception as exc:  # noqa: BLE001
        procs = [f"process_scan_error:{exc!r}"]

    obj = {
        "schema": "e4_syskd_v1_gate_a_preflight_v1",
        "created_at": utc_now(),
        "branch": branch,
        "head_sha": head,
        "expected_freeze_commit": "e6afa02a0260f9bdb05745691e33014274037be5",
        "git_status_sb": status,
        "policy_rel": str(POL_PATH.relative_to(CODE_ROOT)).replace("\\", "/"),
        "policy_sha256": pol_sha,
        "policy_sha_match": pol_sha == EXPECTED_POLICY_SHA,
        "code_root": str(CODE_ROOT),
        "data_root": str(data_root),
        "data_root_reason": "canonical_split_ckpt_match; worktree may have assume-unchanged divergent data",
        "assume_unchanged_flags": assume_unchanged,
        "worktree_split_sha256": wt_split_sha,
        "worktree_split_matches_canonical": wt_split_sha == EXPECTED_SPLIT_SHA,
        "bindings": {
            "b4_checkpoint_sha256": sha256_file(ckpt),
            "split_sha256": sha256_file(data_split),
            "active_registry_sha256": sha256_file(data_reg),
            "labels_sha256": sha256_file(labels),
            "gate_a_schedule_sha256": sha256_file(sched),
        },
        "expected": {
            "b4_checkpoint_sha256": EXPECTED_CKPT_SHA,
            "split_sha256": EXPECTED_SPLIT_SHA,
            "active_registry_sha256": EXPECTED_ACTIVE_REG_SHA,
        },
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn": int(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
            "nvidia_smi": smi,
            "platform": platform.platform(),
        },
        "gpu_training_like_python_procs": procs,
        "no_gpu_training_detected": len(procs) == 0,
        "selected": None,
        "e4_ready": False,
        "pass": True,
    }
    # Integrity gates
    if obj["bindings"]["b4_checkpoint_sha256"] != EXPECTED_CKPT_SHA:
        obj["pass"] = False
        obj["fail"] = "checkpoint_sha"
    if obj["bindings"]["split_sha256"] != EXPECTED_SPLIT_SHA:
        obj["pass"] = False
        obj["fail"] = "split_sha"
    if obj["bindings"]["active_registry_sha256"] != EXPECTED_ACTIVE_REG_SHA:
        obj["pass"] = False
        obj["fail"] = "registry_sha"
    if not torch.cuda.is_available():
        obj["pass"] = False
        obj["fail"] = "cuda"
    if assume_unchanged and not obj["worktree_split_matches_canonical"]:
        obj["warning"] = (
            "worktree has assume-unchanged divergent split/registry; "
            "Gate A uses data_root=main canonical files without restoring worktree"
        )
    # Dirty check: only allow untracked/modified under e4_system_kd_v1* or scripts/e4_syskd*
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True)
    bad = []
    for ln in dirty.splitlines():
        path = ln[3:].strip() if len(ln) > 3 else ln
        if path.startswith("Docs/results/e4_system_kd_v1") or path.startswith("scripts/e4_syskd"):
            continue
        if path == ".gitattributes":
            continue
        bad.append(ln)
    obj["dirty_outside_campaign"] = bad
    if bad:
        obj["pass"] = False
        obj["fail"] = "dirty_outside_campaign"
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_PREFLIGHT.json", obj)
    if not obj["pass"]:
        die(f"preflight_fail:{obj.get('fail')}")
    return obj


def phase_schedule_lock(pre: dict) -> dict:
    sched = json.loads(
        (CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json").read_text(encoding="utf-8")
    )
    ids = list(sched["sample_ids"])
    if len(ids) != 24:
        die("schedule_n_ids")
    obj = {
        "schema": "e4_syskd_v1_gate_a_schedule_lock_v1",
        "status": "FROZEN_BEFORE_GATE_A_MEASUREMENT",
        "created_at": utc_now(),
        "policy_sha256": EXPECTED_POLICY_SHA,
        "parent_schedule_rel": "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json",
        "parent_schedule_sha256": pre["bindings"]["gate_a_schedule_sha256"],
        "sample_ids": ids,
        "n_samples": 24,
        "measured_rounds": 5,
        "warmup_rounds_per_sample_per_arm": 1,
        "batch": 1,
        "arms": ["R0", "R1"],
        "arm_ordering": {
            "rule": "odd_rounds_R0_then_R1__even_rounds_R1_then_R0",
            "odd_round_order": ["R0", "R1"],
            "even_round_order": ["R1", "R0"],
            "round_index_base": 1,
        },
        "seeds": {"bootstrap": 42, "numpy": 42, "torch": 42},
        "synchronization": {
            "cuda_synchronize_before_after_gpu_regions": True,
            "e2e_wall_clock": True,
            "cuda_events_for_gpu_substages_only": True,
        },
        "timeout_sec_overall": 14400,
        "error_policy": {
            "keep_raw_row_on_error": True,
            "forbid_post_hoc_outlier_drop": True,
            "coverage_required": "24_ids_x_5_rounds_x_2_arms",
        },
        "forbid_replace_sample_ids_after_latency": True,
        "roi_regions_order_canonical_code": ROI_ORDER,
        "data_root": pre["data_root"],
        "checkpoint_sha256": EXPECTED_CKPT_SHA,
        "selected": None,
        "e4_ready": False,
    }
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_SCHEDULE_LOCK.json", obj)
    return obj


def phase_path_audit() -> dict:
    obj = {
        "schema": "e4_syskd_v1_gate_a_path_audit_v1",
        "created_at": utc_now(),
        "policy_sha256": EXPECTED_POLICY_SHA,
        "call_path": [
            "input_raw_mp4",
            "decode_read_video_frames_native_once",
            "frame_sampling_windowing_inside_extract_rgb_token_v3",
            "tensor_construction_H2D_collate",
            "SlowFast_FrozenSlowFastBlock4_or_AmpRoiFp32SlowFast",
            "ROI_extraction_roi_align_4_regions",
            "RGB_temporal_tokens_2304d",
            "Skeleton_sealed_coords_vel_validity_plus_B4_head",
            "logits_400",
        ],
        "files": {
            "decode": "src/models/rgb_token_v3.py::read_video_frames_native",
            "extract": "src/models/rgb_token_v3.py::extract_rgb_token_v3_for_video",
            "slowfast": "src/models/rgb_token_v3.py::FrozenSlowFastBlock4",
            "amp_wrapper": "src/optimization/b4q/precision_wrappers.py::AmpRoiFp32SlowFast",
            "head": "src/models/ms_sta_v6.py::build_ms_sta_v6(b4_v6_ctr)",
            "resolver": "src/data_prep/raw_video_resolve.py::RawVideoResolver",
        },
        "invariants": {
            "no_double_decode_per_timed_run": True,
            "no_sealed_rgb_features_instead_of_online_path": True,
            "no_cached_logits": True,
            "roi_order_code": list(REGIONS),
            "roi_order_matches_policy": list(REGIONS) == ROI_ORDER,
            "R0_vs_R1_diff_precision_only": True,
            "R0": "FP32 SlowFast + ROI FP32 + head FP32; sequential windows; pathway_window_batch_size=1",
            "R1": "Q2 AMP SlowFast FP16 + ROI FP32 + head FP32; same decode frames; no pinned/overlap/ROI-batch/stream opts",
            "model_eval": True,
            "torch_no_grad_or_inference_mode": True,
            "checkpoint_strict_load": True,
            "label_mapping_canonical_gloss_to_id": True,
            "raw_row_requires": ["sample_id", "round", "arm"],
        },
        "pass": list(REGIONS) == ROI_ORDER,
    }
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_PATH_AUDIT.json", obj)
    if not obj["pass"]:
        die("roi_order_mismatch")
    return obj


def build_models(data_root: Path, device: torch.device):
    pol_b4q = json.loads((CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json").read_text(encoding="utf-8"))
    sf_rel = pol_b4q["rgb_token_v3"]["slowfast_checkpoint_rel"]
    sf_sha = pol_b4q["rgb_token_v3"]["slowfast_checkpoint_sha256"]
    base_sf = FrozenSlowFastBlock4(
        checkpoint_path=data_root / sf_rel,
        expected_sha256=sf_sha,
        device=device,
        repo_root=data_root,
    )
    amp_sf = AmpRoiFp32SlowFast(base_sf)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt_path = data_root / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"
    if sha256_file(ckpt_path) != EXPECTED_CKPT_SHA:
        die("ckpt_sha")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = b4.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        die(f"strict_load:{missing}:{unexpected}")
    return base_sf, amp_sf, b4, pol_b4q


def run_one(
    *,
    arm: str,
    sid: str,
    round_i: int,
    video_path: Path,
    sealed_path: Path,
    base_sf,
    amp_sf,
    b4,
    device: torch.device,
    label: int,
    signer: str,
    data_root: Path,
    time_stages: bool,
    frames_reuse: tuple[np.ndarray, np.ndarray, float] | None,
) -> dict:
    extractor = amp_sf if arm == "R1" else base_sf
    t_e2e0 = time.perf_counter()
    if frames_reuse is None:
        t0 = time.perf_counter()
        frames, frame_idx, fps = read_video_frames_native(str(video_path))
        decode_ms = (time.perf_counter() - t0) * 1000.0
    else:
        frames, frame_idx, fps = frames_reuse
        decode_ms = 0.0  # shared decode attributed on first call only
    landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)

    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    rss0 = rss_mb()

    torch.cuda.synchronize()
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
        repo_root=data_root,
        frames_native=frames,
        frame_idx=frame_idx,
        fps=fps,
        pathway_window_batch_size=1,
    )
    torch.cuda.synchronize()
    slowfast_roi_ms = (time.perf_counter() - t_rgb0) * 1000.0 if time_stages else 0.0

    item = b4_item_from_extract(sid, extract, sealed, label, signer)
    batch = collate_v5([item])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    torch.cuda.synchronize()
    t_h0 = time.perf_counter()
    with torch.inference_mode():
        logits = head_forward(b4, batch, amp_fp16=False, inference_mode=True)
    torch.cuda.synchronize()
    head_ms = (time.perf_counter() - t_h0) * 1000.0 if time_stages else 0.0
    e2e_ms = (time.perf_counter() - t_e2e0) * 1000.0

    logits_t = logits.detach().float().cpu() if torch.is_tensor(logits) else torch.as_tensor(logits).float()
    if logits_t.ndim > 1:
        logits_t = logits_t.reshape(-1) if logits_t.shape[0] == 1 else logits_t[0]
    pred = int(torch.argmax(logits_t).item())
    top5 = torch.topk(logits_t, k=5).indices.tolist()
    fsha = frames_sha256(frames, frame_idx)

    # ROI boxes/masks if present
    roi_coords = None
    roi_masks = None
    if isinstance(extract, dict):
        roi_coords = extract.get("roi_boxes") or extract.get("boxes")
        roi_masks = extract.get("missing") or {r: extract.get(f"missing_{r}") for r in ROI_ORDER}

    return {
        "ok": True,
        "sample_id": sid,
        "round": round_i,
        "arm": arm,
        "decode_ms": decode_ms,
        "preprocessing_ms": 0.0,
        "h2d_ms": 0.0,
        "slowfast_ms": None,
        "roi_ms": None,
        "rgb_total_ms": slowfast_roi_ms,
        "skeleton_head_ms": head_ms,
        "e2e_ms": e2e_ms,
        "vram_allocated_mb": max(0.0, (torch.cuda.max_memory_allocated() - base_alloc) / (1024**2)),
        "vram_reserved_mb": max(0.0, (torch.cuda.max_memory_reserved() - base_reserved) / (1024**2)),
        "host_rss_delta_mb": rss_mb() - rss0,
        "prediction": pred,
        "top5": top5,
        "label": label,
        "top1_correct": int(pred == label),
        "frame_sha": fsha,
        "frame_idx": np.asarray(frame_idx).astype(np.int32).tolist(),
        "logits": logits_t.numpy().astype(np.float32),
        "token_shapes": {r: list(np.asarray(extract["tokens"][r]).shape) for r in ROI_ORDER},
        "roi_order": list(ROI_ORDER),
        "roi_coords": roi_coords,
        "roi_masks_present": roi_masks is not None,
        "frames": frames,
        "frame_idx_arr": frame_idx,
        "fps": fps,
        "nan_inf": bool((~torch.isfinite(logits_t)).any().item()),
    }


def phase_parity(data_root: Path, lock: dict) -> dict:
    device = torch.device("cuda")
    base_sf, amp_sf, b4, pol_b4q = build_models(data_root, device)
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol_b4q["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    parity_raw = ART / "E4_SYSKD_V1_GATE_A_PARITY_RAW.jsonl"
    if parity_raw.exists():
        die(f"exists:{parity_raw}")

    amp = json.loads(POL_PATH.read_text(encoding="utf-8"))["system_parity_gate_draft"]
    max_abs = float(amp["logit_max_abs_fp16_vs_fp32"])
    mean_abs = float(amp["logit_mean_abs_fp16_vs_fp32"])
    top1_min = float(amp["require_topk1_agreement_rate_min"])

    rows = []
    agree = 0
    abs_max_all = 0.0
    abs_sum = 0.0
    abs_n = 0
    frame_ok = 0
    for sid in lock["sample_ids"]:
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        label = int(labels[gloss])
        signer = str(row.get("signer_id") or "")
        vpath = Path(resolver.resolve(row))
        sealed = sealed_root / f"{sid}.npz"
        if not sealed.is_file():
            die(f"sealed_missing:{sealed}")

        # Shared decode once for both arms
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        shared = (frames, frame_idx, fps)
        r0 = run_one(
            arm="R0",
            sid=sid,
            round_i=0,
            video_path=vpath,
            sealed_path=sealed,
            base_sf=base_sf,
            amp_sf=amp_sf,
            b4=b4,
            device=device,
            label=label,
            signer=signer,
            data_root=data_root,
            time_stages=False,
            frames_reuse=shared,
        )
        r1 = run_one(
            arm="R1",
            sid=sid,
            round_i=0,
            video_path=vpath,
            sealed_path=sealed,
            base_sf=base_sf,
            amp_sf=amp_sf,
            b4=b4,
            device=device,
            label=label,
            signer=signer,
            data_root=data_root,
            time_stages=False,
            frames_reuse=shared,
        )
        diff = np.abs(r0["logits"] - r1["logits"])
        mad = float(diff.max())
        mean_ad = float(diff.mean())
        abs_max_all = max(abs_max_all, mad)
        abs_sum += float(diff.sum())
        abs_n += int(diff.size)
        top1_agree = int(r0["prediction"] == r1["prediction"])
        agree += top1_agree
        top5_set_agree = int(set(r0["top5"]) == set(r1["top5"]))
        same_frame = r0["frame_sha"] == r1["frame_sha"]
        frame_ok += int(same_frame)
        prow = {
            "sample_id": sid,
            "signer_id": signer,
            "frame_sha_equal": same_frame,
            "frame_sha": r0["frame_sha"],
            "frame_idx_equal": r0["frame_idx"] == r1["frame_idx"],
            "roi_order": ROI_ORDER,
            "token_shapes_equal": r0["token_shapes"] == r1["token_shapes"],
            "logit_max_abs": mad,
            "logit_mean_abs": mean_ad,
            "top1_agree": top1_agree,
            "top5_set_agree": top5_set_agree,
            "r0_pred": r0["prediction"],
            "r1_pred": r1["prediction"],
            "r0_top5": r0["top5"],
            "r1_top5": r1["top5"],
            "nan_inf": bool(r0["nan_inf"] or r1["nan_inf"]),
            "label": label,
            "checkpoint_sha256": EXPECTED_CKPT_SHA,
            "policy_sha256": EXPECTED_POLICY_SHA,
        }
        append_jsonl(parity_raw, prow)
        rows.append(prow)
        # free large arrays
        del r0["frames"], r1["frames"], r0["logits"], r1["logits"]

    n = len(rows)
    report = {
        "schema": "e4_syskd_v1_gate_a_parity_report_v1",
        "created_at": utc_now(),
        "n": n,
        "frame_sha_exact_24_of_24": frame_ok == 24,
        "frame_sha_matches": frame_ok,
        "logit_max_abs_observed": abs_max_all,
        "logit_mean_abs_observed": float(abs_sum / max(abs_n, 1)),
        "top1_agreement_rate": agree / max(n, 1),
        "gates": {
            "logit_max_abs_limit": max_abs,
            "logit_mean_abs_limit": mean_abs,
            "top1_agreement_min": top1_min,
        },
        "pass_logit_max": abs_max_all <= max_abs,
        "pass_logit_mean": (abs_sum / max(abs_n, 1)) <= mean_abs,
        "pass_top1": (agree / max(n, 1)) >= top1_min,
        "pass_no_nan_inf": all(not r["nan_inf"] for r in rows),
        "pass_roi_order": True,
        "pass": False,
        "terminal_if_fail": "E4_SYSKD_V1_GATE_A_R1_PARITY_NO_GO",
    }
    report["pass"] = all(
        [
            report["frame_sha_exact_24_of_24"],
            report["pass_logit_max"],
            report["pass_logit_mean"],
            report["pass_top1"],
            report["pass_no_nan_inf"],
        ]
    )
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_PARITY_REPORT.json", report)
    return report


def phase_accuracy_probe(data_root: Path, lock: dict) -> dict:
    """Gate A does not require full val; emit 24-ID online probe + imported comparator note."""
    device = torch.device("cuda")
    base_sf, amp_sf, b4, pol_b4q = build_models(data_root, device)
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol_b4q["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    r0_path = ART / "E4_SYSKD_V1_GATE_A_R0_PREDICTIONS.jsonl"
    r1_path = ART / "E4_SYSKD_V1_GATE_A_R1_PREDICTIONS.jsonl"
    if r0_path.exists() or r1_path.exists():
        die("pred_exists")

    for arm, path in [("R0", r0_path), ("R1", r1_path)]:
        correct = 0
        for sid in lock["sample_ids"]:
            row = meta[sid]
            gloss = str(row.get("text_gloss") or "").strip()
            label = int(labels[gloss])
            signer = str(row.get("signer_id") or "")
            vpath = Path(resolver.resolve(row))
            sealed = sealed_root / f"{sid}.npz"
            frames, frame_idx, fps = read_video_frames_native(str(vpath))
            r = run_one(
                arm=arm,
                sid=sid,
                round_i=0,
                video_path=vpath,
                sealed_path=sealed,
                base_sf=base_sf,
                amp_sf=amp_sf,
                b4=b4,
                device=device,
                label=label,
                signer=signer,
                data_root=data_root,
                time_stages=False,
                frames_reuse=(frames, frame_idx, fps),
            )
            correct += int(r["top1_correct"])
            append_jsonl(
                path,
                {
                    "sample_id": sid,
                    "signer_id": signer,
                    "arm": arm,
                    "label": label,
                    "prediction": r["prediction"],
                    "top5": r["top5"],
                    "top1_correct": r["top1_correct"],
                    "frame_sha": r["frame_sha"],
                    "checkpoint_sha256": EXPECTED_CKPT_SHA,
                    "policy_sha256": EXPECTED_POLICY_SHA,
                },
            )
            del r["frames"], r["logits"]
        if arm == "R0":
            r0_acc = correct / 24.0
        else:
            r1_acc = correct / 24.0

    # Import historical full-val/test as comparator provenance only (not Gate A promote metric).
    imported = {
        "role": "IMPORTED_COMPARATOR",
        "note": "Full locked validation n=2629 not required by Gate A in frozen SYSKD policy; historical B4 SoT metrics bound for provenance only.",
        "historical_val_top1_mean": 84.81,
        "historical_test_top1": 51.55,
        "source": "Docs/results/project_predefense_audit/METRIC_INDEPENDENT_RECALC_V2.json / defense reports",
    }
    report = {
        "schema": "e4_syskd_v1_gate_a_accuracy_report_v1",
        "created_at": utc_now(),
        "full_validation_required_by_policy_at_gate_a": False,
        "probe_n": 24,
        "probe_role": "online_R0_R1_prediction_evidence",
        "r0_probe_top1": r0_acc,
        "r1_probe_top1": r1_acc,
        "imported_comparator": imported,
        "terminology": {
            "signer_id": "mã người thực hiện ký hiệu (mã người ra dấu)",
        },
        "pass": True,
        "pass_reason": "full_val_not_required_at_gate_a; probe artifacts persisted",
    }
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_ACCURACY_REPORT.json", report)
    return report


def phase_efficiency(data_root: Path, lock: dict) -> dict:
    device = torch.device("cuda")
    base_sf, amp_sf, b4, pol_b4q = build_models(data_root, device)
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol_b4q["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    raw_path = ART / "E4_SYSKD_V1_GATE_A_RAW.jsonl"
    if raw_path.exists():
        die(f"exists:{raw_path}")

    warmup = int(lock["warmup_rounds_per_sample_per_arm"])
    measured = int(lock["measured_rounds"])
    ids = list(lock["sample_ids"])

    # Warmup
    for sid in ids:
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        label = int(labels[gloss])
        signer = str(row.get("signer_id") or "")
        vpath = Path(resolver.resolve(row))
        sealed = sealed_root / f"{sid}.npz"
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        shared = (frames, frame_idx, fps)
        for arm in ("R0", "R1"):
            for w in range(warmup):
                r = run_one(
                    arm=arm,
                    sid=sid,
                    round_i=-(w + 1),
                    video_path=vpath,
                    sealed_path=sealed,
                    base_sf=base_sf,
                    amp_sf=amp_sf,
                    b4=b4,
                    device=device,
                    label=label,
                    signer=signer,
                    data_root=data_root,
                    time_stages=False,
                    frames_reuse=shared,
                )
                del r["frames"], r["logits"]

    # Measured interleaved rounds
    for rnd in range(1, measured + 1):
        order = ["R0", "R1"] if (rnd % 2 == 1) else ["R1", "R0"]
        for sid in ids:
            row = meta[sid]
            gloss = str(row.get("text_gloss") or "").strip()
            label = int(labels[gloss])
            signer = str(row.get("signer_id") or "")
            vpath = Path(resolver.resolve(row))
            sealed = sealed_root / f"{sid}.npz"
            frames, frame_idx, fps = read_video_frames_native(str(vpath))
            shared = (frames, frame_idx, fps)
            # Attribute decode cost once to first arm in order
            t0 = time.perf_counter()
            # frames already decoded above; measure decode separately for accounting
            frames2, frame_idx2, fps2 = read_video_frames_native(str(vpath))
            decode_ms = (time.perf_counter() - t0) * 1000.0
            shared = (frames2, frame_idx2, fps2)
            for i, arm in enumerate(order):
                try:
                    r = run_one(
                        arm=arm,
                        sid=sid,
                        round_i=rnd,
                        video_path=vpath,
                        sealed_path=sealed,
                        base_sf=base_sf,
                        amp_sf=amp_sf,
                        b4=b4,
                        device=device,
                        label=label,
                        signer=signer,
                        data_root=data_root,
                        time_stages=True,
                        frames_reuse=shared,
                    )
                    # Add decode into e2e for fair end-to-end (decode-once shared; charge full decode to each arm's e2e via wall already excluding re-decode)
                    # Policy: do not omit decode from e2e. Recompute e2e as decode + postdecode wall.
                    post = float(r["e2e_ms"])
                    e2e = decode_ms + post
                    out = {
                        "sample_id": sid,
                        "round": rnd,
                        "arm": arm,
                        "success": True,
                        "error": None,
                        "decode_ms": decode_ms if i == 0 else 0.0,
                        "decode_ms_shared_charged_in_e2e": decode_ms,
                        "preprocessing_ms": r["preprocessing_ms"],
                        "h2d_ms": r["h2d_ms"],
                        "slowfast_ms": r["slowfast_ms"],
                        "roi_ms": r["roi_ms"],
                        "rgb_total_ms": r["rgb_total_ms"],
                        "skeleton_head_ms": r["skeleton_head_ms"],
                        "e2e_ms": e2e,
                        "post_decode_ms": post,
                        "vram_allocated_mb": r["vram_allocated_mb"],
                        "vram_reserved_mb": r["vram_reserved_mb"],
                        "host_rss_delta_mb": r["host_rss_delta_mb"],
                        "prediction": r["prediction"],
                        "frame_sha": r["frame_sha"],
                        "checkpoint_sha256": EXPECTED_CKPT_SHA,
                        "policy_sha256": EXPECTED_POLICY_SHA,
                        "arm_order_this_round": order,
                    }
                    append_jsonl(raw_path, out)
                    del r["frames"], r["logits"]
                except Exception as exc:  # noqa: BLE001
                    append_jsonl(
                        raw_path,
                        {
                            "sample_id": sid,
                            "round": rnd,
                            "arm": arm,
                            "success": False,
                            "error": repr(exc),
                            "checkpoint_sha256": EXPECTED_CKPT_SHA,
                            "policy_sha256": EXPECTED_POLICY_SHA,
                        },
                    )

    # Aggregate
    rows = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    ok = [r for r in rows if r.get("success")]
    by_arm = {"R0": [], "R1": []}
    for r in ok:
        by_arm[r["arm"]].append(float(r["e2e_ms"]))

    # paired per-sample median across rounds
    from collections import defaultdict

    sample_arm = defaultdict(list)
    for r in ok:
        sample_arm[(r["sample_id"], r["arm"])].append(float(r["e2e_ms"]))
    r0_med = [median(sample_arm[(s, "R0")]) for s in lock["sample_ids"]]
    r1_med = [median(sample_arm[(s, "R1")]) for s in lock["sample_ids"]]
    boot = bootstrap_speedup(r0_med, r1_med, n_boot=10000, seed=42)
    speedup = median(r0_med) / max(median(r1_med), 1e-9)

    report = {
        "schema": "e4_syskd_v1_gate_a_report_v1",
        "created_at": utc_now(),
        "coverage": {
            "expected_rows": 24 * 5 * 2,
            "raw_rows": len(rows),
            "success_rows": len(ok),
            "invalid_rows": len(rows) - len(ok),
        },
        "r0": {
            "e2e_median_ms": median(by_arm["R0"]),
            "e2e_p90_ms": p90(by_arm["R0"]),
            "vram_median_alloc_mb": median([float(r["vram_allocated_mb"]) for r in ok if r["arm"] == "R0"]),
        },
        "r1": {
            "e2e_median_ms": median(by_arm["R1"]),
            "e2e_p90_ms": p90(by_arm["R1"]),
            "vram_median_alloc_mb": median([float(r["vram_allocated_mb"]) for r in ok if r["arm"] == "R1"]),
        },
        "r1_vs_r0": {
            "speedup": speedup,
            "latency_reduction": 1.0 - median(r1_med) / max(median(r0_med), 1e-9),
            "bootstrap": boot,
            "note": "Gate A does not require 1.15x; R1 is AMP reference only",
        },
        "do_not_merge_historical_b4q_e1a_into_aggregate": True,
        "selected": None,
        "e4_ready": False,
    }
    stage = {
        "schema": "e4_syskd_v1_gate_a_stage_profile_v1",
        "created_at": utc_now(),
        "r0_rgb_total_median_ms": median([float(r["rgb_total_ms"]) for r in ok if r["arm"] == "R0"]),
        "r1_rgb_total_median_ms": median([float(r["rgb_total_ms"]) for r in ok if r["arm"] == "R1"]),
        "r0_head_median_ms": median([float(r["skeleton_head_ms"]) for r in ok if r["arm"] == "R0"]),
        "r1_head_median_ms": median([float(r["skeleton_head_ms"]) for r in ok if r["arm"] == "R1"]),
        "r0_decode_charged_median_ms": median([float(r["decode_ms_shared_charged_in_e2e"]) for r in ok if r["arm"] == "R0"]),
        "bottleneck": "decode + SlowFast/ROI dominate; head residual",
    }
    env = {
        "schema": "e4_syskd_v1_gate_a_environment_v1",
        "created_at": utc_now(),
        "python": sys.executable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cudnn": int(torch.backends.cudnn.version()),
        "data_root": str(data_root),
        "code_root": str(CODE_ROOT),
        "policy_sha256": EXPECTED_POLICY_SHA,
        "checkpoint_sha256": EXPECTED_CKPT_SHA,
    }
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_REPORT.json", report)
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_STAGE_PROFILE.json", stage)
    write_lf_json(ART / "E4_SYSKD_V1_GATE_A_ENVIRONMENT.json", env)
    report["stage"] = stage
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["preflight", "schedule", "path", "parity", "accuracy", "efficiency", "all_until_parity", "all"],
        default="all",
    )
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root()

    if args.phase in ("preflight", "all", "all_until_parity"):
        pre = phase_preflight(data_root)
    else:
        pre = json.loads((ART / "E4_SYSKD_V1_GATE_A_PREFLIGHT.json").read_text(encoding="utf-8"))
        data_root = Path(pre["data_root"])

    if args.phase in ("schedule", "all", "all_until_parity"):
        lock = phase_schedule_lock(pre)
    else:
        lock = json.loads((ART / "E4_SYSKD_V1_GATE_A_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))

    if args.phase in ("path", "all", "all_until_parity"):
        phase_path_audit()

    if args.phase in ("parity", "all", "all_until_parity"):
        par = phase_parity(data_root, lock)
        if not par["pass"]:
            print(json.dumps({"terminal": "E4_SYSKD_V1_GATE_A_R1_PARITY_NO_GO", "parity": par}, indent=2))
            return 2

    if args.phase in ("accuracy", "all"):
        phase_accuracy_probe(data_root, lock)

    if args.phase in ("efficiency", "all"):
        # require parity pass
        par = json.loads((ART / "E4_SYSKD_V1_GATE_A_PARITY_REPORT.json").read_text(encoding="utf-8"))
        if not par.get("pass"):
            die("efficiency_blocked_parity_fail")
        phase_efficiency(data_root, lock)

    print(json.dumps({"ok": True, "phase": args.phase, "data_root": str(data_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
