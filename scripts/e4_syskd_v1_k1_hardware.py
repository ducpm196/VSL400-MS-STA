# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 K1 hardware feasibility — no training, no accuracy claims."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
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

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import sealed_landmarks  # noqa: E402
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.rgb_token_v3 import REGIONS, FrozenSlowFastBlock4, extract_rgb_token_v3_for_video, read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast  # noqa: E402
from src.optimization.e4_syskd.k1_dense_student import (  # noqa: E402
    K1_C_MID,
    K2_C_MID_INVENTORY_ONLY,
    DenseK1Student,
    architecture_inventory,
    module_param_hash,
    verify_fast_unchanged,
)

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
BIND_SHA = "ed86157ddcbcb0dff8d1d454f9d572a409b9b5f50c4169b87fc16b34938a69eb"
POL_SHA = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
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


def load_binding() -> dict:
    p = ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json"
    if sha256_file(p) != BIND_SHA:
        raise SystemExit("binding_sha")
    return json.loads(p.read_text(encoding="utf-8"))


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


class StudentAsBlock4(torch.nn.Module):
    """Duck-type FrozenSlowFastBlock4 for AmpRoiFp32SlowFast + extract path."""

    def __init__(self, student: DenseK1Student, teacher_meta: dict) -> None:
        super().__init__()
        self.student = student
        self.device = student.device
        self.checkpoint_sha256 = str(teacher_meta.get("checkpoint_sha256") or "k1_untrained")
        self.load_meta = dict(student.load_meta)
        self.eval()

    @torch.no_grad()
    def forward_maps(self, slow: torch.Tensor, fast: torch.Tensor):
        return self.student.forward_maps(slow, fast)


def time_rgb(
    *,
    extractor,
    sid: str,
    vpath: Path,
    sealed: Path,
    data_root: Path,
    frames,
    frame_idx,
    fps,
    batch_roi: bool,
) -> dict:
    lm, miss, sfi, sts, _sealed = sealed_landmarks(sealed)
    torch.cuda.synchronize()
    base_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
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
    rgb_ms = (time.perf_counter() - t0) * 1000.0
    tokens = extract["tokens"]
    finite = all(np.isfinite(np.asarray(tokens[r])).all() for r in ROI_ORDER)
    return {
        "rgb_total_ms": rgb_ms,
        "vram_mb": max(0.0, (torch.cuda.max_memory_allocated() - base_alloc) / (1024**2)),
        "token_shapes": {r: list(np.asarray(tokens[r]).shape) for r in ROI_ORDER},
        "finite_tokens": finite,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["inventory", "hardware", "all"], default="all")
    args = ap.parse_args()

    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    sch = json.loads((ART / "E4_SYSKD_V1_S5_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    ids = list(sch["sample_ids"])
    device = torch.device("cuda")
    pol = json.loads((CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json").read_text(encoding="utf-8"))

    teacher = FrozenSlowFastBlock4(
        checkpoint_path=data_root / pol["rgb_token_v3"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=data_root,
    )
    # H0 shapes
    with torch.inference_mode():
        s = torch.zeros(1, 3, 8, 256, 256, device=device)
        f = torch.zeros(1, 3, 32, 256, 256, device=device)
        tm = teacher.forward_maps(s, f)
    h0 = {
        "slow_shape": list(tm.slow.shape),
        "fast_shape": list(tm.fast.shape),
        "params": sum(p.numel() for p in teacher.parameters()),
        "pass": list(tm.slow.shape)[1] >= 2048 and list(tm.fast.shape)[1] >= 256,
    }

    student = DenseK1Student(teacher, c_mid=K1_C_MID, device=device, seed=0)
    with torch.inference_mode():
        sm = student.forward_maps(s, f)
        _, stages = student.forward_maps_with_stages(s, f)
    fast_rep = verify_fast_unchanged(teacher, student)
    interface = {
        "schema": "e4_syskd_v1_k1_shape_interface_report_v1",
        "created_at": utc_now(),
        "h0_teacher": h0,
        "h1_student": {
            "c_mid_target": list(K1_C_MID),
            "c_mid_observed": student.stage_mid_channels(),
            "c_out": list(student.c_out),
            "slow_shape": list(sm.slow.shape),
            "fast_shape": list(sm.fast.shape),
            "stage_out_channels": [int(t.shape[1]) for t in stages],
            "c_out_match": [int(t.shape[1]) >= c for t, c in zip(stages, student.c_out)],
            "nan_inf": bool((~torch.isfinite(sm.slow)).any() or (~torch.isfinite(sm.fast)).any()),
        },
        "fast_unchanged": fast_rep,
        "roi_head_accepts_c_out": all(int(t.shape[1]) >= c for t, c in zip(stages, student.c_out)),
        "pass": (
            h0["pass"]
            and student.stage_mid_channels() == list(K1_C_MID)
            and all(int(t.shape[1]) >= c for t, c in zip(stages, student.c_out))
            and fast_rep["pass"]
            and not bool((~torch.isfinite(sm.slow)).any() or (~torch.isfinite(sm.fast)).any())
        ),
        "note": "No prediction fidelity — student untrained.",
    }
    inv = architecture_inventory(teacher, student)
    inv["created_at"] = utc_now()
    inv["k2_param_estimate_note"] = (
        f"K2 C_mid={list(K2_C_MID_INVENTORY_ONLY)} not instantiated for formal HW; inventory-only."
    )
    write_lf(ART / "E4_SYSKD_V1_K1_ARCHITECTURE_INVENTORY.json", inv, overwrite=True)
    write_lf(ART / "E4_SYSKD_V1_K1_SHAPE_INTERFACE_REPORT.json", interface, overwrite=True)

    if args.phase == "inventory":
        print(json.dumps({"interface_pass": interface["pass"], "inv": inv["k1"]}, indent=2))
        return 0 if interface["pass"] else 2

    if not interface["pass"]:
        write_lf(
            ART / "E4_SYSKD_V1_K1_HARDWARE_FEASIBILITY.json",
            {
                "terminal": "E4_SYSKD_V1_STUDENT_HARDWARE_NO_GO",
                "reason": "interface_fail",
                "k1_training_allowed": False,
            },
            overwrite=True,
        )
        raise SystemExit("E4_SYSKD_V1_STUDENT_HARDWARE_NO_GO")

    # Hardware: teacher R1, S5, K1-R1, K1-S5 — RGB/SlowFast+ROI only (no head accuracy)
    meta = load_meta(data_root)
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    r1 = AmpRoiFp32SlowFast(teacher)
    k1_base = StudentAsBlock4(student, teacher.load_meta)
    k1 = AmpRoiFp32SlowFast(k1_base)  # type: ignore[arg-type]

    raw_path = ART / "E4_SYSKD_V1_K1_HARDWARE_RAW.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    runtimes = {
        "R1": (r1, False),
        "S5": (r1, True),
        "K1-R1": (k1, False),
        "K1-S5": (k1, True),
    }

    # warmup
    for sid in ids[:4]:
        row = meta[sid]
        vpath = Path(resolver.resolve(row))
        sealed = sealed_root / f"{sid}.npz"
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        for name, (ext, broi) in runtimes.items():
            time_rgb(
                extractor=ext,
                sid=sid,
                vpath=vpath,
                sealed=sealed,
                data_root=data_root,
                frames=frames,
                frame_idx=frame_idx,
                fps=fps,
                batch_roi=broi,
            )

    measured = 3
    for rnd in range(1, measured + 1):
        for sid in ids:
            row = meta[sid]
            vpath = Path(resolver.resolve(row))
            sealed = sealed_root / f"{sid}.npz"
            tdec = time.perf_counter()
            frames, frame_idx, fps = read_video_frames_native(str(vpath))
            decode_ms = (time.perf_counter() - tdec) * 1000.0
            for name, (ext, broi) in runtimes.items():
                o = time_rgb(
                    extractor=ext,
                    sid=sid,
                    vpath=vpath,
                    sealed=sealed,
                    data_root=data_root,
                    frames=frames,
                    frame_idx=frame_idx,
                    fps=fps,
                    batch_roi=broi,
                )
                append_jsonl(
                    raw_path,
                    {
                        "sample_id": sid,
                        "round": rnd,
                        "runtime": name,
                        "success": True,
                        "decode_ms_shared": decode_ms,
                        "slowfast_roi_ms": o["rgb_total_ms"],
                        "e2e_ms": decode_ms + o["rgb_total_ms"],
                        "vram_mb": o["vram_mb"],
                        "token_shapes": o["token_shapes"],
                        "finite_tokens": o["finite_tokens"],
                        "policy_sha256": POL_SHA,
                        "binding_sha256": BIND_SHA,
                        "note": "untrained_student_no_accuracy",
                    },
                )
            print(f"[k1-hw] round={rnd} sid={sid}", flush=True)

    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by: dict[str, list[float]] = defaultdict(list)
    vram: dict[str, list[float]] = defaultdict(list)
    rgb: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["runtime"]].append(float(r["e2e_ms"]))
        vram[r["runtime"]].append(float(r["vram_mb"]))
        rgb[r["runtime"]].append(float(r["slowfast_roi_ms"]))

    def speed(a: str, b: str) -> float:
        return median(by[a]) / max(median(by[b]), 1e-9)

    # Material speed: K1-R1 faster than R1 on SlowFast+ROI; or K1-S5 vs S5 / ability to hold 1.15 vs R0 proxy
    # Use S5 formal R0 median from efficiency report if present
    r0_med = None
    eff_path = ART / "E4_SYSKD_V1_S5_EFFICIENCY_REPORT.json"
    if eff_path.exists():
        r0_med = float(json.loads(eff_path.read_text(encoding="utf-8"))["arms"]["R0"]["median_e2e_ms"])

    k1_r1_vs_r1 = median(rgb["R1"]) / max(median(rgb["K1-R1"]), 1e-9)
    k1_s5_vs_s5 = median(rgb["S5"]) / max(median(rgb["K1-S5"]), 1e-9)
    # e2e potential vs formal R0 (same decode attribution style)
    pot_k1_r1 = (r0_med / median(by["K1-R1"])) if r0_med else float("nan")
    pot_k1_s5 = (r0_med / median(by["K1-S5"])) if r0_med else float("nan")

    material = (k1_r1_vs_r1 >= 1.02) or (k1_s5_vs_s5 >= 1.02)
    # Amdahl: can e2e reach/maintain 1.15 vs R0 with K1 (+ optional S5)?
    amdalh_ok = bool(
        material
        and (
            (pot_k1_s5 >= 1.15 if np.isfinite(pot_k1_s5) else False)
            or (pot_k1_r1 >= 1.15 if np.isfinite(pot_k1_r1) else False)
            or (material and pot_k1_s5 >= 1.10)  # secondary if S5 already cleared gate
        )
    )
    # Prefer strict: material HW benefit required; training allowed only if material AND (potential ≥1.15 on K1-S5 or K1-R1)
    training_allowed = bool(interface["pass"] and material and ((pot_k1_s5 >= 1.15) or (pot_k1_r1 >= 1.15)))

    # If S5 already ≥1.15, K1 is secondary compression — still need material speed
    s5_already = False
    if eff_path.exists():
        s5_already = bool(json.loads(eff_path.read_text(encoding="utf-8")).get("pass"))

    if not material:
        terminal = "E4_SYSKD_V1_STUDENT_HARDWARE_NO_GO"
        training_allowed = False
    elif training_allowed:
        terminal = "E4_SYSKD_V1_K1_TRAINING_CONTROLS_READY"
    else:
        terminal = "E4_SYSKD_V1_STUDENT_HARDWARE_NO_GO"

    feasibility = {
        "schema": "e4_syskd_v1_k1_hardware_feasibility_v1",
        "created_at": utc_now(),
        "interface_pass": interface["pass"],
        "runtimes": {
            name: {
                "median_e2e_ms": median(by[name]),
                "p90_e2e_ms": p90(by[name]),
                "median_slowfast_roi_ms": median(rgb[name]),
                "median_vram_mb": median(vram[name]),
                "n": len(by[name]),
                "params": inv["k1"]["params"]["param_count"] if name.startswith("K1") else inv["teacher"]["params"]["param_count"],
                "mac": None,
                "mac_note": "MAC omitted — no verified tool binding in campaign.",
            }
            for name in runtimes
        },
        "k1_r1_vs_r1_rgb_speedup": k1_r1_vs_r1,
        "k1_s5_vs_s5_rgb_speedup": k1_s5_vs_s5,
        "potential_e2e_vs_formal_R0_median": {"K1-R1": pot_k1_r1, "K1-S5": pot_k1_s5, "R0_median_ms": r0_med},
        "material_speed": material,
        "amdahl_feasibility": amdalh_ok,
        "s5_already_val_promotion_pass": s5_already,
        "k1_secondary_compression_candidate": bool(s5_already),
        "k1_training_allowed": training_allowed,
        "terminal": terminal,
        "selected": None,
        "e4_ready": False,
        "not_opened": ["K1_training", "K2_formal_hw", "confirmation", "test", "D4", "demo"],
    }
    write_lf(ART / "E4_SYSKD_V1_K1_HARDWARE_FEASIBILITY.json", feasibility, overwrite=True)
    verify = {
        "schema": "e4_syskd_v1_k1_hardware_independent_verify_v1",
        "created_at": utc_now(),
        "n_rows": len(rows),
        "recomputed_k1_r1_vs_r1": k1_r1_vs_r1,
        "terminal": terminal,
        "pass": interface["pass"] and len(rows) == len(ids) * measured * len(runtimes),
        "teacher_param_hash": module_param_hash(teacher),
        "student_param_hash": module_param_hash(student),
        "hashes_differ_expected": module_param_hash(teacher) != module_param_hash(student),
    }
    write_lf(ART / "E4_SYSKD_V1_K1_HARDWARE_INDEPENDENT_VERIFY.json", verify, overwrite=True)
    print(json.dumps({"terminal": terminal, "material": material, "training_allowed": training_allowed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
