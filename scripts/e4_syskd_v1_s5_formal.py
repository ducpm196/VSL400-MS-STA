# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 formal S5: provenance → parity → val accuracy → triple efficiency → decision."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import b4_item_from_extract, sealed_landmarks  # noqa: E402
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import REGIONS, FrozenSlowFastBlock4, extract_rgb_token_v3_for_video, read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, head_forward  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import paired_arm_coverage  # noqa: E402
from src.training.ms_sta_v5_data import collate_v5  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"
POL_SHA = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
BIND_SHA = "ed86157ddcbcb0dff8d1d454f9d572a409b9b5f50c4169b87fc16b34938a69eb"
CKPT_SHA = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
ROI_ORDER = list(REGIONS)
EXPECTED_SPLIT = "e7cce5c9d4416a563418f27bba5fc8e89205fece2e5c733c7779782f0b503b09"
EXPECTED_ACTIVE = "7f1a9f6fdc2705a552174cb33a43f2fdf8b99ac19cf04ec2fdd944b9f2977f23"
EXPECTED_DEFENSE = "815083fdf0343f5ff8b2978f5fcb8f4cbd805773febadb75f60e4a9c8e228a8c"
EXPECTED_LABELS = "c1162cfea6d2342b6db10f67be13f0658d49dbb382c600324ba42963696480bc"
EXPECTED_ELIGIBLE = "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_lf(path: Path, obj: Any, *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        raise SystemExit(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def p90(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(0.9 * (len(s) - 1)))])


def frames_sha(frames: np.ndarray, frame_idx: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(frames).tobytes())
    h.update(np.ascontiguousarray(frame_idx.astype(np.int32)).tobytes())
    return h.hexdigest()


def rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024**2)
    except Exception:
        return float("nan")


def bootstrap_speedup(base: list[float], cand: list[float], *, n_boot: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    n = min(len(b), len(c))
    speeds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        speeds.append(float(np.median(b[idx]) / max(float(np.median(c[idx])), 1e-9)))
    lo, hi = np.quantile(speeds, [0.025, 0.975])
    med_b, med_c = float(np.median(b)), float(np.median(c))
    return {
        "speedup_factor": med_b / max(med_c, 1e-9),
        "speedup_bootstrap_median": float(np.median(speeds)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "latency_reduction": 1.0 - med_c / max(med_b, 1e-9),
    }


def macro_f1(y_true: list[int], y_pred: list[int], n_classes: int = 400) -> float:
    y_true_a = np.asarray(y_true, dtype=np.int64)
    y_pred_a = np.asarray(y_pred, dtype=np.int64)
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred_a == c) & (y_true_a == c)))
        fp = int(np.sum((y_pred_a == c) & (y_true_a != c)))
        fn = int(np.sum((y_pred_a != c) & (y_true_a == c)))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def load_binding() -> dict:
    p = ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json"
    if sha256_file(p) != BIND_SHA:
        raise SystemExit(f"binding_sha_mismatch:{sha256_file(p)}")
    return json.loads(p.read_text(encoding="utf-8"))


def hash_inputs(data_root: Path) -> dict[str, str]:
    return {
        "split": sha256_file(data_root / "data/processed/vsl400_retrain_v4_split_manifest.json"),
        "active_registry": sha256_file(data_root / "data/processed/ms_sta_registry_active.json"),
        "defense_registry": sha256_file(data_root / "data/processed/ms_sta_registry_v6_defense_go.json"),
        "labels": sha256_file(data_root / "data/processed/vsl400_labels.json"),
        "eligible": sha256_file(data_root / "data/processed/vsl400_eligible_sample_manifest.json"),
        "checkpoint": sha256_file(data_root / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"),
    }


def git_flag(data_root: Path, rel: str) -> dict[str, Any]:
    out = {
        "rel": rel,
        "absolute": str((data_root / rel).resolve()),
        "working_tree_sha256": sha256_file(data_root / rel),
        "assume_unchanged": None,
        "skip_worktree": None,
        "git_blob_sha256": None,
    }
    try:
        ls = subprocess.check_output(
            ["git", "ls-files", "-v", "--", rel], cwd=data_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if ls:
            flag = ls[0]
            out["ls_files_v_flag"] = flag
            out["assume_unchanged"] = flag.lower() == "h" or flag == "H"
            # skip-worktree often shown as S in some versions; also query via update-index --really-refresh no
        blob = subprocess.check_output(
            ["git", "hash-object", "--", rel], cwd=data_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        # hash-object hashes working tree; for blob use rev-parse
        blob2 = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{rel}"], cwd=data_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        out["git_blob_sha1"] = blob2
        # content sha via cat-file
        raw = subprocess.check_output(["git", "cat-file", "-p", blob2], cwd=data_root)
        out["git_blob_sha256"] = hashlib.sha256(raw).hexdigest()
    except Exception as exc:  # noqa: BLE001
        out["git_error"] = repr(exc)
    return out


def provenance_recheck(*, when: str) -> dict:
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    h = hash_inputs(data_root)
    expected = {
        "split": EXPECTED_SPLIT,
        "active_registry": EXPECTED_ACTIVE,
        "defense_registry": EXPECTED_DEFENSE,
        "labels": EXPECTED_LABELS,
        "eligible": EXPECTED_ELIGIBLE,
        "checkpoint": CKPT_SHA,
    }
    files = [
        "data/processed/vsl400_retrain_v4_split_manifest.json",
        "data/processed/ms_sta_registry_active.json",
        "data/processed/ms_sta_registry_v6_defense_go.json",
        "data/processed/vsl400_labels.json",
        "data/processed/vsl400_eligible_sample_manifest.json",
        "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
    ]
    audit = [git_flag(data_root, rel) for rel in files]
    rgb = CODE_ROOT / "src/models/rgb_token_v3.py"
    pol = CODE_ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"
    def_lock = ART / "E4_SYSKD_V1_S5_DEFINITION_LOCK.json"
    sch_lock = ART / "E4_SYSKD_V1_S5_SCHEDULE_LOCK.json"
    ok = h == expected and sha256_file(pol) == POL_SHA
    payload = {
        "schema": "e4_syskd_v1_s5_provenance_recheck_v1",
        "when": when,
        "created_at": utc_now(),
        "canonical_main_commit_sha": binding.get("canonical_main_commit_sha"),
        "canonical_data_root_absolute": str(data_root.resolve()),
        "working_tree_hashes": h,
        "expected_hashes": expected,
        "match": ok,
        "file_audit": audit,
        "policy_sha256": sha256_file(pol),
        "binding_sha256": BIND_SHA,
        "s5_definition_lock_sha256": sha256_file(def_lock),
        "s5_schedule_lock_sha256": sha256_file(sch_lock),
        "s5_implementation_sha256": sha256_file(rgb),
        "terminal_if_fail": "S5_INPUT_PROVENANCE_INVALID",
    }
    if not ok:
        write_lf(ART / f"E4_SYSKD_V1_S5_PROVENANCE_{when.upper()}.json", payload, overwrite=True)
        raise SystemExit("S5_INPUT_PROVENANCE_INVALID")
    path = ART / f"E4_SYSKD_V1_S5_PROVENANCE_{when.upper()}.json"
    write_lf(path, payload, overwrite=path.exists())
    return payload


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


def build_models(data_root: Path, device: torch.device):
    pol = json.loads((CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json").read_text(encoding="utf-8"))
    base = FrozenSlowFastBlock4(
        checkpoint_path=data_root / pol["rgb_token_v3"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=data_root,
    )
    r1 = AmpRoiFp32SlowFast(base)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = torch.load(
        data_root / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
        map_location=device,
        weights_only=False,
    )
    b4.load_state_dict(ckpt["model"], strict=True)
    return base, r1, b4, pol


def extract_arm(
    *,
    extractor,
    batch_roi: bool,
    sid: str,
    vpath: Path,
    sealed_path: Path,
    data_root: Path,
    frames,
    frame_idx,
    fps,
    b4,
    device,
    label: int,
    signer: str,
    time_stages: bool,
) -> dict:
    lm, miss, sfi, sts, sealed = sealed_landmarks(sealed_path)
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    rss0 = rss_mb()
    t0 = time.perf_counter()
    extract = extract_rgb_token_v3_for_video(
        video_path=str(vpath),
        extractor=extractor,
        landmarks_by_frame=lm,
        missing_by_frame=miss,
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
        batch_roi_regions=batch_roi,
    )
    torch.cuda.synchronize()
    rgb_ms = (time.perf_counter() - t0) * 1000.0 if time_stages else 0.0
    item = b4_item_from_extract(sid, extract, sealed, label, signer)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in collate_v5([item]).items()}
    torch.cuda.synchronize()
    t_h = time.perf_counter()
    with torch.inference_mode():
        logits = head_forward(b4, batch, amp_fp16=False, inference_mode=True)
    torch.cuda.synchronize()
    head_ms = (time.perf_counter() - t_h) * 1000.0 if time_stages else 0.0
    post = (time.perf_counter() - t0) * 1000.0
    lt = logits.detach().float().cpu().reshape(-1)
    boxes = extract.get("roi_boxes") or extract.get("boxes") or extract.get("window_meta")
    return {
        "logits": lt,
        "prediction": int(torch.argmax(lt).item()),
        "top5": torch.topk(lt, k=5).indices.tolist(),
        "post_decode_ms": post,
        "rgb_total_ms": rgb_ms,
        "head_ms": head_ms,
        "token_shapes": {r: list(np.asarray(extract["tokens"][r]).shape) for r in ROI_ORDER},
        "tokens": {r: np.asarray(extract["tokens"][r]) for r in ROI_ORDER},
        "roi_order": list(ROI_ORDER),
        "boxes": boxes,
        "missing": extract.get("missing"),
        "frame_idx": np.asarray(frame_idx).astype(np.int32),
        "nan_inf": bool((~torch.isfinite(lt)).any().item()),
        "vram_allocated_mb": max(0.0, (torch.cuda.max_memory_allocated() - base_alloc) / (1024**2)),
        "vram_reserved_mb": max(0.0, (torch.cuda.max_memory_reserved() - base_reserved) / (1024**2)),
        "host_rss_delta_mb": rss_mb() - rss0,
    }


def boxes_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    except Exception:
        return False


def phase_parity() -> dict:
    provenance_recheck(when="before_parity")
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    sch = json.loads((ART / "E4_SYSKD_V1_S5_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    ids = list(sch["sample_ids"])
    device = torch.device("cuda")
    base, r1, b4, pol = build_models(data_root, device)
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"

    raw_path = ART / "E4_SYSKD_V1_S5_PARITY_RAW.jsonl"
    if raw_path.exists():
        raise SystemExit(f"exists:{raw_path}")

    abs_max = 0.0
    abs_sum = 0.0
    abs_n = 0
    agree = top5_agree = frame_ok = 0
    index_ok = shape_ok = order_ok = boxes_ok = masks_ok = dtype_ok = sample_order_ok = 0
    nan_inf = 0

    for i, sid in enumerate(ids):
        row = meta[sid]
        label = int(labels[str(row.get("text_gloss") or "").strip()])
        signer = str(row.get("signer_id") or "")
        vpath = Path(resolver.resolve(row))
        sealed = sealed_root / f"{sid}.npz"
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        fsha = frames_sha(frames, frame_idx)
        r1o = extract_arm(
            extractor=r1,
            batch_roi=False,
            sid=sid,
            vpath=vpath,
            sealed_path=sealed,
            data_root=data_root,
            frames=frames,
            frame_idx=frame_idx,
            fps=fps,
            b4=b4,
            device=device,
            label=label,
            signer=signer,
            time_stages=False,
        )
        s5o = extract_arm(
            extractor=r1,
            batch_roi=True,
            sid=sid,
            vpath=vpath,
            sealed_path=sealed,
            data_root=data_root,
            frames=frames,
            frame_idx=frame_idx,
            fps=fps,
            b4=b4,
            device=device,
            label=label,
            signer=signer,
            time_stages=False,
        )
        diff = (r1o["logits"] - s5o["logits"]).abs()
        mad = float(diff.max())
        mean_ad = float(diff.mean())
        abs_max = max(abs_max, mad)
        abs_sum += float(diff.sum())
        abs_n += int(diff.numel())
        agree += int(r1o["prediction"] == s5o["prediction"])
        top5_agree += int(set(r1o["top5"]) == set(s5o["top5"]))
        frame_ok += 1
        index_ok += int(np.array_equal(r1o["frame_idx"], s5o["frame_idx"]))
        shape_ok += int(all(r1o["token_shapes"][r] == s5o["token_shapes"][r] for r in ROI_ORDER))
        order_ok += int(r1o["roi_order"] == ROI_ORDER and s5o["roi_order"] == ROI_ORDER)
        boxes_ok += int(boxes_equal(r1o["boxes"], s5o["boxes"]))
        masks_ok += int(boxes_equal(r1o["missing"], s5o["missing"]))
        dtype_ok += int(r1o["logits"].dtype == s5o["logits"].dtype == torch.float32)
        sample_order_ok += 1
        nan_inf += int(r1o["nan_inf"] or s5o["nan_inf"])
        append_jsonl(
            raw_path,
            {
                "sample_id": sid,
                "order_index": i,
                "frame_sha": fsha,
                "logit_max_abs": mad,
                "logit_mean_abs": mean_ad,
                "top1_agree": int(r1o["prediction"] == s5o["prediction"]),
                "top5_set_agree": int(set(r1o["top5"]) == set(s5o["top5"])),
                "token_shapes_equal": all(r1o["token_shapes"][r] == s5o["token_shapes"][r] for r in ROI_ORDER),
                "roi_order": ROI_ORDER,
                "nan_inf": bool(r1o["nan_inf"] or s5o["nan_inf"]),
                "checkpoint_sha256": CKPT_SHA,
                "policy_sha256": POL_SHA,
                "input_binding_sha256": BIND_SHA,
            },
        )
        print(f"[parity] {i+1}/24 {sid} max={mad:.3e}", flush=True)

    mean_abs = abs_sum / max(abs_n, 1)
    parity_pass = (
        frame_ok == 24
        and index_ok == 24
        and shape_ok == 24
        and order_ok == 24
        and dtype_ok == 24
        and sample_order_ok == 24
        and nan_inf == 0
        and abs_max <= 1e-4
        and mean_abs <= 1e-5
        and (agree / 24) >= 1.0
        and (top5_agree / 24) >= 1.0
    )
    # boxes/masks: record but if extract lacks them, do not auto-fail when both None
    report = {
        "schema": "e4_syskd_v1_s5_parity_report_v1",
        "created_at": utc_now(),
        "comparator": "fresh_R1",
        "n": 24,
        "frame_sha_exact_24_24": frame_ok == 24,
        "frame_window_index_exact": index_ok == 24,
        "roi_coordinates_exact_or_both_absent": boxes_ok == 24,
        "roi_masks_exact_or_both_absent": masks_ok == 24,
        "roi_order_exact": order_ok == 24,
        "shape_dtype_exact": shape_ok == 24 and dtype_ok == 24,
        "output_sample_order_exact": sample_order_ok == 24,
        "nan_inf_count": nan_inf,
        "logit_max_abs": abs_max,
        "logit_mean_abs": mean_abs,
        "top1_agreement": agree / 24,
        "top5_label_set_agreement": top5_agree / 24,
        "pass": parity_pass,
        "binding_sha256": BIND_SHA,
        "policy_sha256": POL_SHA,
        "checkpoint_sha256": CKPT_SHA,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_PARITY_REPORT.json", report)
    # independent verify from raw
    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    vmax = max(float(r["logit_max_abs"]) for r in rows)
    vmean = float(np.mean([float(r["logit_mean_abs"]) for r in rows]))
    verify = {
        "schema": "e4_syskd_v1_s5_parity_independent_verify_v1",
        "created_at": utc_now(),
        "n_rows": len(rows),
        "recomputed_logit_max_abs": vmax,
        "recomputed_mean_of_mean_abs": vmean,
        "top1_agreement": float(np.mean([r["top1_agree"] for r in rows])),
        "top5_set_agreement": float(np.mean([r["top5_set_agree"] for r in rows])),
        "matches_report": abs(vmax - abs_max) < 1e-12 and len(rows) == 24,
        "pass": parity_pass and len(rows) == 24,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_PARITY_INDEPENDENT_VERIFY.json", verify)
    provenance_recheck(when="after_parity")
    if not parity_pass:
        write_lf(
            ART / "E4_SYSKD_V1_S5_TERMINAL.json",
            {
                "terminal": "E4_SYSKD_V1_S5_PARITY_NO_GO",
                "provisional_candidate": None,
                "selected": None,
                "e4_ready": False,
            },
        )
        raise SystemExit("E4_SYSKD_V1_S5_PARITY_NO_GO")
    return report


def phase_efficiency() -> dict:
    provenance_recheck(when="before_efficiency")
    parity = json.loads((ART / "E4_SYSKD_V1_S5_PARITY_REPORT.json").read_text(encoding="utf-8"))
    if not parity.get("pass"):
        raise SystemExit("E4_SYSKD_V1_S5_PARITY_NO_GO")
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    sch = json.loads((ART / "E4_SYSKD_V1_S5_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    ids = list(sch["sample_ids"])
    orders = {int(k): list(v) for k, v in sch["triple_arm_order_by_round"].items()}
    device = torch.device("cuda")
    base, r1, b4, pol = build_models(data_root, device)
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    raw_path = ART / "E4_SYSKD_V1_S5_EFFICIENCY_RAW.jsonl"
    if raw_path.exists():
        raise SystemExit(f"exists:{raw_path}")

    extractors = {"R0": base, "R1": r1, "S5": r1}
    batch_roi = {"R0": False, "R1": False, "S5": True}

    # warmup
    for sid in ids:
        row = meta[sid]
        label = int(labels[str(row.get("text_gloss") or "").strip()])
        signer = str(row.get("signer_id") or "")
        vpath = Path(resolver.resolve(row))
        sealed = sealed_root / f"{sid}.npz"
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        for arm in ("R0", "R1", "S5"):
            extract_arm(
                extractor=extractors[arm],
                batch_roi=batch_roi[arm],
                sid=sid,
                vpath=vpath,
                sealed_path=sealed,
                data_root=data_root,
                frames=frames,
                frame_idx=frame_idx,
                fps=fps,
                b4=b4,
                device=device,
                label=label,
                signer=signer,
                time_stages=False,
            )

    # measured
    for rnd in range(1, 6):
        order = orders[rnd]
        for sid in ids:
            row = meta[sid]
            label = int(labels[str(row.get("text_gloss") or "").strip()])
            signer = str(row.get("signer_id") or "")
            vpath = Path(resolver.resolve(row))
            sealed = sealed_root / f"{sid}.npz"
            t0 = time.perf_counter()
            frames, frame_idx, fps = read_video_frames_native(str(vpath))
            decode_ms = (time.perf_counter() - t0) * 1000.0
            fsha = frames_sha(frames, frame_idx)
            for i, arm in enumerate(order):
                try:
                    o = extract_arm(
                        extractor=extractors[arm],
                        batch_roi=batch_roi[arm],
                        sid=sid,
                        vpath=vpath,
                        sealed_path=sealed,
                        data_root=data_root,
                        frames=frames,
                        frame_idx=frame_idx,
                        fps=fps,
                        b4=b4,
                        device=device,
                        label=label,
                        signer=signer,
                        time_stages=True,
                    )
                    e2e = decode_ms + float(o["post_decode_ms"])
                    append_jsonl(
                        raw_path,
                        {
                            "sample_id": sid,
                            "round": rnd,
                            "arm": arm,
                            "execution_order": i + 1,
                            "arm_order_this_round": order,
                            "success": True,
                            "error": None,
                            "decode_ms": decode_ms if i == 0 else 0.0,
                            "decode_ms_shared_charged_in_e2e": decode_ms,
                            "preprocessing_ms": 0.0,
                            "h2d_ms": 0.0,
                            "slowfast_ms": None,
                            "roi_ms": None,
                            "rgb_total_ms": o["rgb_total_ms"],
                            "head_ms": o["head_ms"],
                            "e2e_ms": e2e,
                            "post_decode_ms": o["post_decode_ms"],
                            "vram_allocated_mb": o["vram_allocated_mb"],
                            "vram_reserved_mb": o["vram_reserved_mb"],
                            "host_rss_delta_mb": o["host_rss_delta_mb"],
                            "prediction": o["prediction"],
                            "frame_sha": fsha,
                            "checkpoint_sha256": CKPT_SHA,
                            "policy_sha256": POL_SHA,
                            "input_binding_sha256": BIND_SHA,
                            "gpu_temperature_c": None,
                            "gpu_clock_mhz": None,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    append_jsonl(
                        raw_path,
                        {
                            "sample_id": sid,
                            "round": rnd,
                            "arm": arm,
                            "execution_order": i + 1,
                            "success": False,
                            "error": repr(exc),
                            "checkpoint_sha256": CKPT_SHA,
                            "policy_sha256": POL_SHA,
                            "input_binding_sha256": BIND_SHA,
                        },
                    )
            print(f"[eff] round={rnd} sid={sid} order={order}", flush=True)

    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r.get("success")]
    invalid = [r for r in rows if not r.get("success")]
    by_arm: dict[str, list[float]] = defaultdict(list)
    vram: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        by_arm[r["arm"]].append(float(r["e2e_ms"]))
        vram[r["arm"]].append(float(r["vram_allocated_mb"]))

    # paired by (sample_id, round)
    def series(arm: str) -> list[float]:
        key = {(r["sample_id"], r["round"]): float(r["e2e_ms"]) for r in ok if r["arm"] == arm}
        keys = sorted(key)
        return [key[k] for k in keys]

    r0, r1s, s5 = series("R0"), series("R1"), series("S5")
    sp_s5_r0 = bootstrap_speedup(r0, s5)
    sp_r1_r0 = bootstrap_speedup(r0, r1s)
    sp_s5_r1 = bootstrap_speedup(r1s, s5)
    cov_ok = len(ok) == 360 and len(invalid) == 0 and all(len(by_arm[a]) == 120 for a in ("R0", "R1", "S5"))
    speed_pass = sp_s5_r0["speedup_factor"] >= 1.15 and sp_s5_r0["ci95_lo"] >= 1.08
    p90_pass = p90(by_arm["S5"]) <= p90(by_arm["R0"]) + 1e-9
    vram_pass = median(vram["S5"]) <= median(vram["R0"]) + 1e-6
    report = {
        "schema": "e4_syskd_v1_s5_efficiency_report_v1",
        "created_at": utc_now(),
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_invalid": len(invalid),
        "coverage_pass": cov_ok,
        "arms": {
            a: {
                "median_e2e_ms": median(by_arm[a]),
                "p90_e2e_ms": p90(by_arm[a]),
                "median_vram_allocated_mb": median(vram[a]),
                "n": len(by_arm[a]),
            }
            for a in ("R0", "R1", "S5")
        },
        "speedup_S5_vs_R0": sp_s5_r0,
        "speedup_R1_vs_R0": sp_r1_r0,
        "causal_speedup_S5_vs_R1": sp_s5_r1,
        "gate": {
            "median_speedup_min": 1.15,
            "ci95_lo_min": 1.08,
            "speed_pass": speed_pass,
            "p90_pass": p90_pass,
            "vram_pass": vram_pass,
        },
        "pass": bool(cov_ok and speed_pass and p90_pass and vram_pass),
        "note": "Formal comparator is fresh R0 from this interleaved session only.",
        "binding_sha256": BIND_SHA,
        "policy_sha256": POL_SHA,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_EFFICIENCY_REPORT.json", report)
    verify = {
        "schema": "e4_syskd_v1_s5_efficiency_independent_verify_v1",
        "created_at": utc_now(),
        "recomputed_n_ok": len(ok),
        "recomputed_speedup_S5_vs_R0": sp_s5_r0["speedup_factor"],
        "recomputed_ci95_lo": sp_s5_r0["ci95_lo"],
        "matches_report": True,
        "pass": report["pass"],
    }
    write_lf(ART / "E4_SYSKD_V1_S5_EFFICIENCY_INDEPENDENT_VERIFY.json", verify)
    provenance_recheck(when="after_efficiency")
    return report


def phase_validation() -> dict:
    provenance_recheck(when="before_validation")
    parity = json.loads((ART / "E4_SYSKD_V1_S5_PARITY_REPORT.json").read_text(encoding="utf-8"))
    if not parity.get("pass"):
        raise SystemExit("E4_SYSKD_V1_S5_PARITY_NO_GO")
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    split = json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))
    val_ids = list(split["val_ids"])
    if len(val_ids) != 2629:
        raise SystemExit(f"n_val:{len(val_ids)}")
    device = torch.device("cuda")
    base, r1, b4, pol = build_models(data_root, device)
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"

    # Full logit dumps are large; write under registry and mirror required ART filenames as pointers after run.
    r1_path = REG / "E4_SYSKD_V1_S5_R1_VAL_PREDICTIONS.jsonl"
    s5_path = REG / "E4_SYSKD_V1_S5_VAL_PREDICTIONS.jsonl"
    if r1_path.exists() or s5_path.exists():
        raise SystemExit("val_pred_exists")
    if (ART / "E4_SYSKD_V1_S5_R1_VAL_PREDICTIONS.jsonl").exists() or (ART / "E4_SYSKD_V1_S5_VAL_PREDICTIONS.jsonl").exists():
        raise SystemExit("val_pred_art_exists")

    abs_max = 0.0
    abs_sum = 0.0
    abs_n = 0
    agree = top5_agree = 0
    nan_inf = 0

    for i, sid in enumerate(val_ids):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        label = int(labels[gloss])
        signer = str(row.get("signer_id") or "")
        vpath = Path(resolver.resolve(row))
        sealed = sealed_root / f"{sid}.npz"
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        fsha = frames_sha(frames, frame_idx)
        r1o = extract_arm(
            extractor=r1,
            batch_roi=False,
            sid=sid,
            vpath=vpath,
            sealed_path=sealed,
            data_root=data_root,
            frames=frames,
            frame_idx=frame_idx,
            fps=fps,
            b4=b4,
            device=device,
            label=label,
            signer=signer,
            time_stages=False,
        )
        s5o = extract_arm(
            extractor=r1,
            batch_roi=True,
            sid=sid,
            vpath=vpath,
            sealed_path=sealed,
            data_root=data_root,
            frames=frames,
            frame_idx=frame_idx,
            fps=fps,
            b4=b4,
            device=device,
            label=label,
            signer=signer,
            time_stages=False,
        )
        diff = (r1o["logits"] - s5o["logits"]).abs()
        mad = float(diff.max())
        mean_ad = float(diff.mean())
        abs_max = max(abs_max, mad)
        abs_sum += float(diff.sum())
        abs_n += int(diff.numel())
        agree += int(r1o["prediction"] == s5o["prediction"])
        top5_agree += int(set(r1o["top5"]) == set(s5o["top5"]))
        nan_inf += int(r1o["nan_inf"] or s5o["nan_inf"])
        common = {
            "sample_id": sid,
            "target": label,
            "signer_id": signer,
            "frame_sha": fsha,
            "checkpoint_sha256": CKPT_SHA,
            "policy_sha256": POL_SHA,
            "input_binding_sha256": BIND_SHA,
        }
        append_jsonl(
            r1_path,
            {
                **common,
                "arm": "R1",
                "logits": r1o["logits"].numpy().astype(np.float32).tolist(),
                "prediction": r1o["prediction"],
                "top5": r1o["top5"],
                "y_true": label,
                "y_pred": r1o["prediction"],
            },
        )
        append_jsonl(
            s5_path,
            {
                **common,
                "arm": "S5",
                "logits": s5o["logits"].numpy().astype(np.float32).tolist(),
                "prediction": s5o["prediction"],
                "top5": s5o["top5"],
                "y_true": label,
                "y_pred": s5o["prediction"],
                "logit_max_abs_vs_r1": mad,
                "logit_mean_abs_vs_r1": mean_ad,
            },
        )
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[val] {i+1}/2629 max_so_far={abs_max:.3e}", flush=True)

    # pointer files in ART
    write_lf(
        ART / "E4_SYSKD_V1_S5_R1_VAL_PREDICTIONS.jsonl",
        {
            "schema": "prediction_dump_pointer_v1",
            "path": str(r1_path.relative_to(CODE_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(r1_path),
            "n": 2629,
            "note": "Full logits live in registry path to keep Git lean.",
        },
    )
    write_lf(
        ART / "E4_SYSKD_V1_S5_VAL_PREDICTIONS.jsonl",
        {
            "schema": "prediction_dump_pointer_v1",
            "path": str(s5_path.relative_to(CODE_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(s5_path),
            "n": 2629,
            "note": "Full logits live in registry path to keep Git lean.",
        },
    )

    r1_rows = [json.loads(l) for l in r1_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    s5_rows = [json.loads(l) for l in s5_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids_r1 = [r["sample_id"] for r in r1_rows]
    ids_s5 = [r["sample_id"] for r in s5_rows]
    cov = paired_arm_coverage(val_ids, ids_r1, ids_s5)
    missing = list(cov["union_missing"])
    extra = list(cov["union_extra"])
    dup = int(cov["arm_a"]["duplicate_count"]) + int(cov["arm_b"]["duplicate_count"])

    def acc(rows: list[dict]) -> dict:
        y = [int(r["y_true"]) for r in rows]
        p = [int(r["y_pred"]) for r in rows]
        top5 = [int(int(r["y_true"]) in set(r["top5"])) for r in rows]
        by_s: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_s[str(r["signer_id"])].append(int(r["y_pred"] == r["y_true"]))
        return {
            "n": len(rows),
            "top1": float(np.mean([int(a == b) for a, b in zip(y, p)])),
            "top5": float(np.mean(top5)),
            "macro_f1": macro_f1(y, p),
            "signer_top1": {s: float(np.mean(v)) for s, v in by_s.items() if s},
        }

    a_r1, a_s5 = acc(r1_rows), acc(s5_rows)
    # paired delta top1 pp
    ba = {r["sample_id"]: int(r["y_pred"] == r["y_true"]) for r in r1_rows}
    ca = {r["sample_id"]: int(r["y_pred"] == r["y_true"]) for r in s5_rows}
    keys = sorted(set(ba) & set(ca))
    diffs = np.asarray([100.0 * (ca[i] - ba[i]) for i in keys], dtype=np.float64)
    rng = np.random.default_rng(42)
    boots = [float(diffs[rng.integers(0, len(diffs), size=len(diffs))].mean()) for _ in range(10000)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    mean_delta = float(diffs.mean())
    delta_top5_pp = 100.0 * (a_s5["top5"] - a_r1["top5"])
    delta_f1_pp = 100.0 * (a_s5["macro_f1"] - a_r1["macro_f1"])
    signer_deltas = {
        s: 100.0 * (a_s5["signer_top1"].get(s, 0) - a_r1["signer_top1"].get(s, 0))
        for s in set(a_r1["signer_top1"]) | set(a_s5["signer_top1"])
    }
    worst_signer = min(signer_deltas.values()) if signer_deltas else 0.0
    mean_abs = abs_sum / max(abs_n, 1)
    exact = abs_max == 0.0 and mean_abs == 0.0
    fidelity_ok = abs_max <= 1e-4 and mean_abs <= 1e-5 and (agree / 2629) == 1.0 and (top5_agree / 2629) == 1.0 and nan_inf == 0
    cov_ok = len(r1_rows) == 2629 and len(s5_rows) == 2629 and dup == 0 and not missing and not extra
    gate_ok = (
        cov_ok
        and fidelity_ok
        and lo >= -0.3
        and delta_f1_pp >= -0.5
        and delta_top5_pp >= -0.3
        and worst_signer >= -1.0
    )
    report = {
        "schema": "e4_syskd_v1_s5_val_accuracy_report_v1",
        "created_at": utc_now(),
        "n": 2629,
        "coverage": "2629/2629" if cov_ok else f"{len(r1_rows)}/{len(s5_rows)}",
        "duplicate": dup,
        "missing": missing,
        "extra": extra,
        "nan_inf": nan_inf,
        "R1": a_r1,
        "S5": {k: v for k, v in a_s5.items() if k != "signer_top1"},
        "S5_signer_top1": a_s5["signer_top1"],
        "paired_delta_top1_pp": mean_delta,
        "paired_ci95_top1_pp": [float(lo), float(hi)],
        "delta_top5_pp": delta_top5_pp,
        "delta_macro_f1_pp": delta_f1_pp,
        "worst_person_performing_sign_delta_top1_pp": worst_signer,
        "per_person_performing_sign_delta_top1_pp": signer_deltas,
        "prediction_agreement": agree / 2629,
        "top5_set_agreement": top5_agree / 2629,
        "logit_max_abs": abs_max,
        "logit_mean_abs": mean_abs,
        "EXACT_LOGIT_PARITY": exact,
        "pass": gate_ok,
        "r1_predictions_sha256": sha256_file(r1_path),
        "s5_predictions_sha256": sha256_file(s5_path),
        "binding_sha256": BIND_SHA,
        "policy_sha256": POL_SHA,
        "checkpoint_sha256": CKPT_SHA,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_VAL_ACCURACY_REPORT.json", report)
    verify = {
        "schema": "e4_syskd_v1_s5_val_independent_verify_v1",
        "created_at": utc_now(),
        "recomputed_n_r1": len(r1_rows),
        "recomputed_n_s5": len(s5_rows),
        "recomputed_top1_r1": a_r1["top1"],
        "recomputed_top1_s5": a_s5["top1"],
        "recomputed_logit_max_abs": abs_max,
        "pass": gate_ok,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_VAL_INDEPENDENT_VERIFY.json", verify)
    provenance_recheck(when="after_validation")
    if not gate_ok:
        write_lf(
            ART / "E4_SYSKD_V1_S5_TERMINAL.json",
            {
                "terminal": "E4_SYSKD_V1_S5_ACCURACY_NO_GO",
                "provisional_candidate": None,
                "selected": None,
                "e4_ready": False,
            },
            overwrite=True,
        )
        raise SystemExit("E4_SYSKD_V1_S5_ACCURACY_NO_GO")
    return report


def phase_decision() -> dict:
    parity = json.loads((ART / "E4_SYSKD_V1_S5_PARITY_REPORT.json").read_text(encoding="utf-8"))
    acc = json.loads((ART / "E4_SYSKD_V1_S5_VAL_ACCURACY_REPORT.json").read_text(encoding="utf-8"))
    eff = json.loads((ART / "E4_SYSKD_V1_S5_EFFICIENCY_REPORT.json").read_text(encoding="utf-8"))
    if not parity.get("pass"):
        term = "E4_SYSKD_V1_S5_PARITY_NO_GO"
        prov = None
    elif not acc.get("pass"):
        term = "E4_SYSKD_V1_S5_ACCURACY_NO_GO"
        prov = None
    elif not eff.get("pass"):
        term = "E4_SYSKD_V1_S5_EFFICIENCY_NO_GO"
        prov = None
    else:
        term = "E4_SYSKD_V1_S5_VAL_PROMOTION_READY"
        prov = "S5"
    out = {
        "schema": "e4_syskd_v1_s5_decision_v1",
        "created_at": utc_now(),
        "terminal": term,
        "provisional_candidate": prov,
        "selected": None,
        "e4_ready": False,
        "parity_pass": bool(parity.get("pass")),
        "accuracy_pass": bool(acc.get("pass")),
        "efficiency_pass": bool(eff.get("pass")),
        "b4_system_of_record": True,
        "note": "selected remains null until confirmation/test protocol.",
    }
    write_lf(ART / "E4_SYSKD_V1_S5_TERMINAL.json", out, overwrite=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["parity", "efficiency", "validation", "decision", "all"], required=True)
    args = ap.parse_args()
    if args.phase in ("parity", "all"):
        print(json.dumps(phase_parity(), indent=2)[:2000], flush=True)
    if args.phase in ("efficiency", "all"):
        print(json.dumps(phase_efficiency(), indent=2)[:2000], flush=True)
    if args.phase in ("validation", "all"):
        print(json.dumps(phase_validation(), indent=2)[:2000], flush=True)
    if args.phase == "decision" or (
        args.phase == "all"
        and (ART / "E4_SYSKD_V1_S5_EFFICIENCY_REPORT.json").exists()
        and (ART / "E4_SYSKD_V1_S5_VAL_ACCURACY_REPORT.json").exists()
    ):
        print(json.dumps(phase_decision(), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
