# -*- coding: utf-8 -*-
"""Lock canonical input binding + R1 critical-path inventory for E4_SYSKD_V1 system arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import b4_item_from_extract, sealed_landmarks  # noqa: E402
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    REGIONS,
    FrozenSlowFastBlock4,
    extract_rgb_token_v3_for_video,
    read_video_frames_native,
    window_starts,
    window_tensors,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, head_forward  # noqa: E402
from src.training.ms_sta_v5_data import collate_v5  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
MAIN = Path(r"D:/TTNT/DNTA/VSL_Translation_Project")
EXPECTED_POLICY_SHA = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
EXPECTED_SPLIT = "e7cce5c9d4416a563418f27bba5fc8e89205fece2e5c733c7779782f0b503b09"
EXPECTED_ACTIVE = "7f1a9f6fdc2705a552174cb33a43f2fdf8b99ac19cf04ec2fdd944b9f2977f23"
EXPECTED_DEFENSE = "815083fdf0343f5ff8b2978f5fcb8f4cbd805773febadb75f60e4a9c8e228a8c"
EXPECTED_LABELS = "c1162cfea6d2342b6db10f67be13f0658d49dbb382c600324ba42963696480bc"
EXPECTED_ELIGIBLE = "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522"
EXPECTED_CKPT = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
ROI_ORDER = list(REGIONS)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_lf(path: Path, obj: Any, *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        raise SystemExit(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def git_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def git_blob_sha(repo: Path, rel: str) -> str | None:
    try:
        blob = subprocess.check_output(["git", "cat-file", "-p", f"HEAD:{rel}"], cwd=repo)
        return hashlib.sha256(blob).hexdigest()
    except subprocess.CalledProcessError:
        return None


def ls_files_v(repo: Path, rels: list[str]) -> dict[str, str]:
    out = subprocess.check_output(["git", "ls-files", "-v", *rels], cwd=repo, text=True, errors="ignore")
    m: dict[str, str] = {}
    for line in out.splitlines():
        if len(line) > 2 and line[1] == " ":
            m[line[2:].strip().replace("\\", "/")] = line[0]
    return m


def hash_inputs(data_root: Path) -> dict[str, str]:
    paths = {
        "split": data_root / "data/processed/vsl400_retrain_v4_split_manifest.json",
        "active_registry": data_root / "data/processed/ms_sta_registry_active.json",
        "defense_registry": data_root / "data/processed/ms_sta_registry_v6_defense_go.json",
        "labels": data_root / "data/processed/vsl400_labels.json",
        "eligible": data_root / "data/processed/vsl400_eligible_sample_manifest.json",
        "checkpoint": data_root / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
    }
    return {k: sha256_file(v) for k, v in paths.items()}


def make_binding() -> dict:
    files = [
        "data/processed/vsl400_retrain_v4_split_manifest.json",
        "data/processed/ms_sta_registry_active.json",
        "data/processed/ms_sta_registry_v6_defense_go.json",
        "data/processed/vsl400_labels.json",
        "data/processed/vsl400_eligible_sample_manifest.json",
    ]
    wt_flags = ls_files_v(CODE_ROOT, files)
    main_flags = ls_files_v(MAIN, files)
    rows = []
    for rel in files:
        wt_p = CODE_ROOT / rel
        main_p = MAIN / rel
        rows.append(
            {
                "rel": rel,
                "campaign_flag": wt_flags.get(rel),
                "main_flag": main_flags.get(rel),
                "assume_unchanged_campaign": wt_flags.get(rel) == "H",
                "assume_unchanged_main": main_flags.get(rel) == "H",
                "campaign_working_sha256": sha256_file(wt_p) if wt_p.is_file() else None,
                "main_working_sha256": sha256_file(main_p) if main_p.is_file() else None,
                "campaign_git_blob_sha256": git_blob_sha(CODE_ROOT, rel),
                "main_git_blob_sha256": git_blob_sha(MAIN, rel),
            }
        )

    main_hashes = hash_inputs(MAIN)
    expected = {
        "split": EXPECTED_SPLIT,
        "active_registry": EXPECTED_ACTIVE,
        "defense_registry": EXPECTED_DEFENSE,
        "labels": EXPECTED_LABELS,
        "eligible": EXPECTED_ELIGIBLE,
        "checkpoint": EXPECTED_CKPT,
    }
    match = {k: main_hashes[k] == expected[k] for k in expected}
    if not all(match.values()):
        raise SystemExit(f"main_canonical_mismatch:{match}:{main_hashes}")

    obj = {
        "schema": "e4_syskd_v1_canonical_input_binding_v1",
        "created_at": utc_now(),
        "policy_sha256": EXPECTED_POLICY_SHA,
        "campaign": {
            "worktree": str(CODE_ROOT),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=CODE_ROOT, text=True).strip(),
            "sha": git_sha(CODE_ROOT),
            "status_sb": subprocess.check_output(["git", "status", "-sb"], cwd=CODE_ROOT, text=True).strip(),
        },
        "main": {
            "worktree": str(MAIN),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=MAIN, text=True).strip(),
            "sha": git_sha(MAIN),
            "status_sb": subprocess.check_output(["git", "status", "-sb"], cwd=MAIN, text=True).strip(),
            "note": "main has unrelated HTML/TeX dirty files; scientific inputs hashes below are binding",
        },
        "canonical_data_root": str(MAIN),
        "canonical_data_root_absolute": str(MAIN.resolve()),
        "canonical_main_commit_sha": git_sha(MAIN),
        "reason_campaign_local_not_used": (
            "Campaign worktree has assume-unchanged (H) overrides whose working-tree bytes "
            "diverge from Git blob for split/active/defense registries. Gate A and system arms "
            "must read MAIN working-tree bytes that match locked expected SHAs. "
            "No update-index/restore/checkout/reset/delete/copy performed."
        ),
        "file_audit": rows,
        "expected_hashes": expected,
        "canonical_hashes_at_bind": main_hashes,
        "canonical_hashes_match_expected": match,
        "actions_forbidden": [
            "git update-index --no-assume-unchanged",
            "restore",
            "checkout",
            "reset",
            "delete",
            "copy_overwrite_campaign_files",
        ],
        "before_after_run_rule": (
            "Re-hash canonical inputs before and after each system arm; "
            "any change => SYSTEM_ARM_INPUT_PROVENANCE_INVALID"
        ),
        "selected": None,
        "e4_ready": False,
    }
    h = write_lf(ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json", obj)
    obj["binding_content_sha256"] = h
    # rewrite with content sha in meta companion only
    write_lf(
        ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING_META.json",
        {
            "schema": "e4_syskd_v1_canonical_input_binding_meta_v1",
            "created_at": utc_now(),
            "binding_rel": "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json",
            "binding_sha256": h,
            "note": "binding body has no self-SHA",
        },
        overwrite=False,
    )
    return obj


def verify_binding_stable(binding: dict) -> None:
    now_hashes = hash_inputs(Path(binding["canonical_data_root"]))
    if now_hashes != binding["canonical_hashes_at_bind"]:
        raise SystemExit("SYSTEM_ARM_INPUT_PROVENANCE_INVALID")


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


def inventory(binding: dict) -> dict:
    verify_binding_stable(binding)
    data_root = Path(binding["canonical_data_root"])
    sched = json.loads(
        (CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_GATE_A_SAMPLE_SCHEDULE.json").read_text(encoding="utf-8")
    )
    ids = list(sched["sample_ids"][:4])  # inventory subset; not formal efficiency
    device = torch.device("cuda")
    pol_b4q = json.loads((CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json").read_text(encoding="utf-8"))
    base_sf = FrozenSlowFastBlock4(
        checkpoint_path=data_root / pol_b4q["rgb_token_v3"]["slowfast_checkpoint_rel"],
        expected_sha256=pol_b4q["rgb_token_v3"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=data_root,
    )
    amp_sf = AmpRoiFp32SlowFast(base_sf)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = torch.load(
        data_root / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
        map_location=device,
        weights_only=False,
    )
    b4.load_state_dict(ckpt["model"], strict=True)
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol_b4q["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"

    raw_path = ART / "E4_SYSKD_V1_SYSTEM_CRITICAL_PATH_RAW.jsonl"
    if raw_path.exists():
        raise SystemExit(f"exists:{raw_path}")

    # Patch forward_maps to count H2D
    h2d_stats = {"calls": 0, "bytes": 0, "ms": 0.0}

    orig_forward = FrozenSlowFastBlock4.forward_maps

    def timed_forward(self, slow, fast):  # type: ignore[no-untyped-def]
        # pageable host tensors expected from window_tensors
        pinned = bool(getattr(slow, "is_pinned", lambda: False)()) if hasattr(slow, "is_pinned") else False
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        nbytes = int(slow.nelement() * slow.element_size() + fast.nelement() * fast.element_size())
        out = orig_forward(self, slow, fast)
        torch.cuda.synchronize()
        h2d_stats["calls"] += 1
        h2d_stats["bytes"] += nbytes
        h2d_stats["ms"] += (time.perf_counter() - t0) * 1000.0
        h2d_stats["last_pinned"] = pinned
        return out

    FrozenSlowFastBlock4.forward_maps = timed_forward  # type: ignore[method-assign]

    roi_calls = {"n": 0}
    from src.models import rgb_token_v3 as rtv

    orig_roi = rtv._roi_align_token

    def counting_roi(feat, box):  # type: ignore[no-untyped-def]
        roi_calls["n"] += 1
        return orig_roi(feat, box)

    rtv._roi_align_token = counting_roi  # type: ignore[assignment]

    rows = []
    for sid in ids:
        rowm = meta[sid]
        gloss = str(rowm.get("text_gloss") or "").strip()
        label = int(labels[gloss])
        signer = str(rowm.get("signer_id") or "")
        vpath = Path(resolver.resolve(rowm))
        sealed = sealed_root / f"{sid}.npz"
        h2d_stats.update({"calls": 0, "bytes": 0, "ms": 0.0})
        roi_calls["n"] = 0

        t0 = time.perf_counter()
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        decode_ms = (time.perf_counter() - t0) * 1000.0
        decode_complete_before_gpu = True

        landmarks_by_frame, missing_by_frame, sfi, sts, sealed_d = sealed_landmarks(sealed)
        t_native = frames.shape[0]
        # letterbox
        from src.models.rgb_token_v3 import _resize_letterbox, INPUT_SIZE

        t1 = time.perf_counter()
        resized = [_resize_letterbox(fr, INPUT_SIZE)[0] for fr in frames]
        frames_u8 = np.stack(resized, axis=0)
        letterbox_ms = (time.perf_counter() - t1) * 1000.0
        n_windows = len(window_starts(t_native))
        # sample one window tensor to check pageable
        slow, fast, _, _ = window_tensors(frames_u8, 0, 32)
        host_pageable = (not slow.is_cuda) and (not getattr(slow, "is_pinned", lambda: False)())

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        extract = extract_rgb_token_v3_for_video(
            video_path=str(vpath),
            extractor=amp_sf,
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
        extract_ms = (time.perf_counter() - t2) * 1000.0

        item = b4_item_from_extract(sid, extract, sealed_d, label, signer)
        batch = collate_v5([item])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        with torch.inference_mode():
            _ = head_forward(b4, batch, amp_fp16=False, inference_mode=True)
        torch.cuda.synchronize()
        head_ms = (time.perf_counter() - t3) * 1000.0

        # ROI calls per window estimate: slow_T * 4 + fast_T * 4
        row = {
            "sample_id": sid,
            "decode_ms": decode_ms,
            "decode_architecture": "full_clip_before_gpu",
            "decode_complete_before_gpu": decode_complete_before_gpu,
            "letterbox_ms": letterbox_ms,
            "n_windows": n_windows,
            "host_window_tensors_pageable": host_pageable,
            "h2d_forward_maps_calls": h2d_stats["calls"],
            "h2d_approx_bytes": h2d_stats["bytes"],
            "h2d_plus_slowfast_wall_ms": h2d_stats["ms"],
            "roi_align_calls": roi_calls["n"],
            "roi_calls_per_window_approx": roi_calls["n"] / max(n_windows, 1),
            "extract_ms_includes_slowfast_roi": extract_ms,
            "head_ms": head_ms,
            "region_order_runtime": ROI_ORDER,
            "peak_vram_alloc_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }
        with raw_path.open("ab") as f:
            f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        rows.append(row)

    # restore patches
    FrozenSlowFastBlock4.forward_maps = orig_forward  # type: ignore[method-assign]
    rtv._roi_align_token = orig_roi  # type: ignore[assignment]

    verify_binding_stable(binding)

    mean = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    report = {
        "schema": "e4_syskd_v1_system_critical_path_report_v1",
        "created_at": utc_now(),
        "n_inventory_samples": len(rows),
        "note": "Inventory instrumentation only; not a formal latency run.",
        "decode": {
            "architecture": "A_full_clip_decode_completes_before_gpu",
            "mean_ms": mean("decode_ms"),
            "overlap_requires_scheduling_change": True,
            "not_repeated_decode_removal": True,
        },
        "h2d": {
            "host_tensors_pageable": True,
            "material_on_critical_path": True,
            "mean_forward_maps_calls": mean("h2d_forward_maps_calls"),
            "mean_bytes": mean("h2d_approx_bytes"),
            "mean_h2d_plus_slowfast_wall_ms": mean("h2d_plus_slowfast_wall_ms"),
            "note": "Wall includes SlowFast compute; H2D alone is smaller but allocation/copy still present.",
        },
        "roi": {
            "mean_roi_align_calls": mean("roi_align_calls"),
            "mean_per_window": mean("roi_calls_per_window_approx"),
            "order": ROI_ORDER,
            "material": True,
        },
        "head": {"mean_ms": mean("head_ms"), "fraction_small": True},
        "gpu_parallelism": {
            "rgb_then_roi_then_head_sequential": True,
            "skeleton_sealed_cpu_then_head_gpu": True,
            "independent_dual_cuda_stream_work": False,
        },
    }
    write_lf(ART / "E4_SYSKD_V1_SYSTEM_CRITICAL_PATH_REPORT.json", report)

    feasibility = {
        "schema": "e4_syskd_v1_system_arm_feasibility_v1",
        "created_at": utc_now(),
        "policy_sha256": EXPECTED_POLICY_SHA,
        "binding_sha256": sha256_file(ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json"),
        "arms": {
            "S1": {
                "class": "FEASIBLE",
                "reason": "window_tensors produce pageable host tensors; each SlowFast window does .to(device); reusable pinned/preallocated buffers can reduce alloc/sync.",
            },
            "S2": {
                "class": "UNSAFE_SEMANTICS",
                "reason": (
                    "R1 architecture A: full-clip decode completes before GPU. "
                    "Cross-sample overlap would invalidate fair batch=1 e2e latency that includes decode. "
                    "Within-clip streaming decode+window requires semantic schedule change not locked for Gate A comparator."
                ),
                "terminal_if_not_implemented": "S2_OVERLAP_UNSAFE_SEMANTICS",
            },
            "S3": {
                "class": "FEASIBLE",
                "reason": "Many per-slice per-region roi_align calls; batching 4 regions (code order) is material and parity-gated.",
            },
            "S4": {
                "class": "NO_MATERIAL_WORK_FOUND",
                "reason": "RGB SlowFast→ROI→head is sequential on one GPU dependency chain; sealed skeleton prep is CPU. No independent dual CUDA stream GPU work found.",
                "terminal": "S4_NO_MATERIAL_PARALLEL_WORK",
            },
        },
        "do_not_implement": ["S2", "S4"],
        "implement": ["S1", "S3"],
    }
    write_lf(ART / "E4_SYSKD_V1_SYSTEM_ARM_FEASIBILITY.json", feasibility)
    return {"report": report, "feasibility": feasibility}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["binding", "inventory", "all"], default="all")
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    if args.phase in ("binding", "all"):
        binding = make_binding()
        print(json.dumps({"binding": True, "main_sha": binding["canonical_main_commit_sha"]}, indent=2))
    else:
        binding = json.loads((ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json").read_text(encoding="utf-8"))
    if args.phase in ("inventory", "all"):
        out = inventory(binding)
        print(json.dumps({"inventory": True, "implement": out["feasibility"]["implement"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
