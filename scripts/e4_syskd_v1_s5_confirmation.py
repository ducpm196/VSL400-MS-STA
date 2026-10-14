# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 S5 confirmation: per-seed accuracy + efficiency (no locked test)."""
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

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import b4_item_from_extract, sealed_landmarks  # noqa: E402
from scripts.e4_syskd_v1_s5_formal import (  # noqa: E402
    BIND_SHA,
    POL_SHA,
    ROI_ORDER,
    bootstrap_speedup,
    extract_arm,
    frames_sha,
    hash_inputs,
    load_binding,
    load_meta,
    macro_f1,
    median,
    p90,
    provenance_recheck,
    write_lf,
    append_jsonl,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import FrozenSlowFastBlock4, read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import paired_arm_coverage  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_locks() -> tuple[dict, dict]:
    sch = json.loads((ART / "E4_SYSKD_V1_S5_CONFIRMATION_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    ck = json.loads((ART / "E4_SYSKD_V1_S5_CONFIRMATION_CHECKPOINT_LOCK.json").read_text(encoding="utf-8"))
    return sch, ck


def ckpt_for_seed(ck: dict, seed: int) -> dict:
    for row in ck["checkpoints"]:
        if int(row["seed"]) == int(seed):
            return row
    raise SystemExit(f"S5_CONFIRMATION_CHECKPOINT_SOURCE_GAP:{seed}")


def build_models_for_seed(data_root: Path, device: torch.device, ckpt_abs: Path, expected_sha: str):
    if sha256_file(ckpt_abs) != expected_sha:
        raise SystemExit(f"ckpt_sha:{sha256_file(ckpt_abs)}")
    pol = json.loads((CODE_ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json").read_text(encoding="utf-8"))
    base = FrozenSlowFastBlock4(
        checkpoint_path=data_root / pol["rgb_token_v3"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=data_root,
    )
    r1 = AmpRoiFp32SlowFast(base)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    blob = torch.load(ckpt_abs, map_location=device, weights_only=False)
    b4.load_state_dict(blob["model"], strict=True)
    return base, r1, b4, pol


def phase_accuracy_seed(seed: int) -> dict:
    provenance_recheck(when=f"before_confirm_acc_seed{seed}")
    sch, ck = load_locks()
    row = ckpt_for_seed(ck, seed)
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    split = json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))
    val_ids = list(split["val_ids"])
    if len(val_ids) != 2629:
        raise SystemExit("n_val")
    device = torch.device("cuda")
    base, r1, b4, pol = build_models_for_seed(data_root, device, Path(row["path_absolute"]), row["sha256"])
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"

    r1_path = REG / f"E4_SYSKD_V1_S5_CONFIRMATION_PREDICTIONS_SEED{seed}_R1.jsonl"
    s5_path = REG / f"E4_SYSKD_V1_S5_CONFIRMATION_PREDICTIONS_SEED{seed}_S5.jsonl"
    if r1_path.exists() or s5_path.exists():
        raise SystemExit(f"exists_pred:{seed}")

    abs_max = 0.0
    abs_sum = 0.0
    abs_n = 0
    agree = top5_agree = nan_inf = 0
    frame_ok = 0

    for i, sid in enumerate(val_ids):
        m = meta[sid]
        gloss = str(m.get("text_gloss") or "").strip()
        label = int(labels[gloss])
        signer = str(m.get("signer_id") or "")
        vpath = Path(resolver.resolve(m))
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
        frame_ok += 1
        common = {
            "sample_id": sid,
            "target": label,
            "signer_id": signer,
            "frame_sha": fsha,
            "checkpoint_sha256": row["sha256"],
            "policy_sha256": POL_SHA,
            "input_binding_sha256": BIND_SHA,
            "seed": seed,
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
            print(f"[confirm-acc seed={seed}] {i+1}/2629 max={abs_max:.3e}", flush=True)

    write_lf(
        ART / f"E4_SYSKD_V1_S5_CONFIRMATION_PREDICTIONS_SEED{seed}.json",
        {
            "schema": "confirmation_prediction_pointer_v1",
            "seed": seed,
            "r1_path": str(r1_path.relative_to(CODE_ROOT)).replace("\\", "/"),
            "s5_path": str(s5_path.relative_to(CODE_ROOT)).replace("\\", "/"),
            "r1_sha256": sha256_file(r1_path),
            "s5_sha256": sha256_file(s5_path),
            "r1_bytes": r1_path.stat().st_size,
            "s5_bytes": s5_path.stat().st_size,
            "n": 2629,
            "checkpoint_sha256": row["sha256"],
        },
        overwrite=True,
    )

    r1_rows = [json.loads(l) for l in r1_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    s5_rows = [json.loads(l) for l in s5_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cov = paired_arm_coverage(val_ids, [r["sample_id"] for r in r1_rows], [r["sample_id"] for r in s5_rows])

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
    fidelity_ok = (
        abs_max <= 1e-4
        and mean_abs <= 1e-5
        and (agree / 2629) == 1.0
        and (top5_agree / 2629) == 1.0
        and nan_inf == 0
        and frame_ok == 2629
    )
    gate_ok = (
        bool(cov["exact_coverage"])
        and fidelity_ok
        and lo >= -0.3
        and delta_f1_pp >= -0.5
        and delta_top5_pp >= -0.3
        and worst_signer >= -1.0
    )
    out = {
        "seed": seed,
        "checkpoint_sha256": row["sha256"],
        "coverage": cov,
        "nan_inf": nan_inf,
        "frame_sha_rows": frame_ok,
        "R1": {k: v for k, v in a_r1.items() if k != "signer_top1"},
        "S5": {k: v for k, v in a_s5.items() if k != "signer_top1"},
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
        "r1_sha256": sha256_file(r1_path),
        "s5_sha256": sha256_file(s5_path),
    }
    write_lf(ART / f"E4_SYSKD_V1_S5_CONFIRMATION_ACCURACY_SEED{seed}.json", out, overwrite=True)
    provenance_recheck(when=f"after_confirm_acc_seed{seed}")
    return out


def phase_efficiency_seed(seed: int) -> dict:
    provenance_recheck(when=f"before_confirm_eff_seed{seed}")
    sch, ck = load_locks()
    row = ckpt_for_seed(ck, seed)
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    ids = list(sch["efficiency"]["sample_ids"])
    orders = {int(k): list(v) for k, v in sch["efficiency"]["triple_arm_order_by_round"].items()}
    device = torch.device("cuda")
    base, r1, b4, pol = build_models_for_seed(data_root, device, Path(row["path_absolute"]), row["sha256"])
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
    raw_path = ART / "E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_RAW.jsonl"
    extractors = {"R0": base, "R1": r1, "S5": r1}
    batch_roi = {"R0": False, "R1": False, "S5": True}

    # warmup
    for sid in ids:
        m = meta[sid]
        label = int(labels[str(m.get("text_gloss") or "").strip()])
        signer = str(m.get("signer_id") or "")
        vpath = Path(resolver.resolve(m))
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

    for rnd in range(1, 6):
        order = orders[rnd]
        for sid in ids:
            m = meta[sid]
            label = int(labels[str(m.get("text_gloss") or "").strip()])
            signer = str(m.get("signer_id") or "")
            vpath = Path(resolver.resolve(m))
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
                            "seed": seed,
                            "sample_id": sid,
                            "round": rnd,
                            "arm": arm,
                            "execution_order": i + 1,
                            "arm_order_this_round": order,
                            "success": True,
                            "error": None,
                            "decode_ms": decode_ms if i == 0 else 0.0,
                            "decode_ms_shared_charged_in_e2e": decode_ms,
                            "rgb_total_ms": o["rgb_total_ms"],
                            "head_ms": o["head_ms"],
                            "e2e_ms": e2e,
                            "post_decode_ms": o["post_decode_ms"],
                            "vram_allocated_mb": o["vram_allocated_mb"],
                            "prediction": o["prediction"],
                            "frame_sha": fsha,
                            "checkpoint_sha256": row["sha256"],
                            "policy_sha256": POL_SHA,
                            "input_binding_sha256": BIND_SHA,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    append_jsonl(
                        raw_path,
                        {
                            "seed": seed,
                            "sample_id": sid,
                            "round": rnd,
                            "arm": arm,
                            "success": False,
                            "error": repr(exc),
                            "checkpoint_sha256": row["sha256"],
                        },
                    )
            print(f"[confirm-eff seed={seed}] round={rnd} sid={sid}", flush=True)

    rows = [
        json.loads(l)
        for l in raw_path.read_text(encoding="utf-8").splitlines()
        if l.strip() and json.loads(l).get("seed") == seed
    ]
    ok = [r for r in rows if r.get("success")]
    invalid = [r for r in rows if not r.get("success")]
    by_arm: dict[str, list[float]] = defaultdict(list)
    vram: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        by_arm[r["arm"]].append(float(r["e2e_ms"]))
        vram[r["arm"]].append(float(r.get("vram_allocated_mb") or 0.0))

    def series(arm: str) -> list[float]:
        key = {(r["sample_id"], r["round"]): float(r["e2e_ms"]) for r in ok if r["arm"] == arm}
        return [key[k] for k in sorted(key)]

    r0, r1s, s5 = series("R0"), series("R1"), series("S5")
    sp = bootstrap_speedup(r0, s5)
    cov_ok = len(ok) == 360 and len(invalid) == 0
    speed_pass = sp["speedup_factor"] >= 1.15 and sp["ci95_lo"] >= 1.08
    p90_pass = p90(by_arm["S5"]) <= p90(by_arm["R0"]) + 1e-9
    vram_pass = median(vram["S5"]) <= median(vram["R0"]) + 1e-6
    out = {
        "seed": seed,
        "checkpoint_sha256": row["sha256"],
        "n_ok": len(ok),
        "n_invalid": len(invalid),
        "coverage_pass": cov_ok,
        "arms": {
            a: {
                "median_e2e_ms": median(by_arm[a]),
                "p90_e2e_ms": p90(by_arm[a]),
                "median_vram_allocated_mb": median(vram[a]),
            }
            for a in ("R0", "R1", "S5")
        },
        "speedup_S5_vs_R0": sp,
        "gate": {"speed_pass": speed_pass, "p90_pass": p90_pass, "vram_pass": vram_pass},
        "pass": bool(cov_ok and speed_pass and p90_pass and vram_pass),
    }
    write_lf(ART / f"E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_SEED{seed}.json", out, overwrite=True)
    provenance_recheck(when=f"after_confirm_eff_seed{seed}")
    return out


def aggregate_accuracy() -> dict:
    seeds = [42, 123, 456]
    per = []
    for s in seeds:
        p = ART / f"E4_SYSKD_V1_S5_CONFIRMATION_ACCURACY_SEED{s}.json"
        if not p.exists():
            raise SystemExit(f"missing_acc:{s}")
        per.append(json.loads(p.read_text(encoding="utf-8")))
    t1 = [float(x["S5"]["top1"]) for x in per]
    t5 = [float(x["S5"]["top5"]) for x in per]
    f1 = [float(x["S5"]["macro_f1"]) for x in per]
    report = {
        "schema": "e4_syskd_v1_s5_confirmation_accuracy_report_v1",
        "created_at": utc_now(),
        "seeds": seeds,
        "per_seed": per,
        "mean_top1": float(np.mean(t1)),
        "std_top1": float(np.std(t1, ddof=1)),
        "mean_top5": float(np.mean(t5)),
        "std_top5": float(np.std(t5, ddof=1)),
        "mean_macro_f1": float(np.mean(f1)),
        "std_macro_f1": float(np.std(f1, ddof=1)),
        "worst_seed_top1": int(seeds[int(np.argmin(t1))]),
        "worst_person_performing_sign_delta_across_seeds": min(
            float(x["worst_person_performing_sign_delta_top1_pp"]) for x in per
        ),
        "parity_pass_3_of_3": all(bool(x["pass"]) and bool(x.get("EXACT_LOGIT_PARITY") or x["logit_max_abs"] <= 1e-4) for x in per),
        "checkpoint_provenance_3_of_3": True,
        "pass": all(bool(x["pass"]) for x in per),
        "policy_sha256": POL_SHA,
        "binding_sha256": BIND_SHA,
        "note": "Do not select best seed as conclusion.",
    }
    write_lf(ART / "E4_SYSKD_V1_S5_CONFIRMATION_ACCURACY_REPORT.json", report, overwrite=True)
    return report


def aggregate_efficiency() -> dict:
    seeds = [42, 123, 456]
    per = []
    for s in seeds:
        p = ART / f"E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_SEED{s}.json"
        if not p.exists():
            raise SystemExit(f"missing_eff:{s}")
        per.append(json.loads(p.read_text(encoding="utf-8")))
    speeds = [float(x["speedup_S5_vs_R0"]["speedup_factor"]) for x in per]
    ci_lo = [float(x["speedup_S5_vs_R0"]["ci95_lo"]) for x in per]
    report = {
        "schema": "e4_syskd_v1_s5_confirmation_efficiency_report_v1",
        "created_at": utc_now(),
        "seeds": seeds,
        "per_seed": per,
        "mean_speedup": float(np.mean(speeds)),
        "std_speedup": float(np.std(speeds, ddof=1)),
        "worst_seed_speedup": int(seeds[int(np.argmin(speeds))]),
        "worst_seed_speedup_value": float(min(speeds)),
        "worst_ci_lower_bound": float(min(ci_lo)),
        "worst_p90_s5": float(max(float(x["arms"]["S5"]["p90_e2e_ms"]) for x in per)),
        "all_seed_gate_status": all(bool(x["pass"]) for x in per),
        "pass": all(bool(x["pass"]) for x in per),
        "note": "Per-seed sessions only; raw rows not pooled across seeds for median.",
        "policy_sha256": POL_SHA,
        "binding_sha256": BIND_SHA,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_REPORT.json", report, overwrite=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["accuracy", "efficiency", "aggregate_accuracy", "aggregate_efficiency"])
    ap.add_argument("--seed", type=int, choices=[42, 123, 456])
    args = ap.parse_args()
    if args.phase == "accuracy":
        if args.seed is None:
            raise SystemExit("seed_required")
        print(json.dumps(phase_accuracy_seed(args.seed), indent=2)[:2500], flush=True)
    elif args.phase == "efficiency":
        if args.seed is None:
            raise SystemExit("seed_required")
        print(json.dumps(phase_efficiency_seed(args.seed), indent=2)[:2500], flush=True)
    elif args.phase == "aggregate_accuracy":
        print(json.dumps(aggregate_accuracy(), indent=2)[:2500], flush=True)
    else:
        print(json.dumps(aggregate_efficiency(), indent=2)[:2500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
