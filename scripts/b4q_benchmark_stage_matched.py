# -*- coding: utf-8 -*-
"""B4Q-V1 baseline stage profile (create-only) + Amdahl bounds.

Remeasures cache-backed B4 head with CUDA events (warmup≥100, trials≥500).
Binds prior R4B stage-matched e2e breakdown as supporting evidence (hashed).
Does not mutate B4 checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters  # noqa: E402
from src.optimization.b4q.invariants import load_policy, sha256_file, verify_b4_checkpoint  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_BASELINE_STAGE_PROFILE.json"
RAW = ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_BASELINE_HEAD_TIMINGS.jsonl"
WARMUP = 100
TRIALS = 500


def load_meta(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def cuda_ms(fn, *, n: int, warmup: int) -> list[float]:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs: list[float] = []
    for _ in range(n):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        xs.append(float(starter.elapsed_time(ender)))
    return xs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()
    if OUT.is_file():
        raise SystemExit(f"refuse_overwrite:{OUT}")
    if RAW.is_file():
        raise SystemExit(f"refuse_overwrite:{RAW}")

    pol = load_policy()
    verify_b4_checkpoint(pol)
    if args.warmup < int(pol["efficiency_gate"]["warmup_gpu_stages_min"]):
        raise SystemExit("warmup_below_policy")
    if args.trials < int(pol["efficiency_gate"]["measured_trials_gpu_stages_min"]):
        raise SystemExit("trials_below_policy")

    device = torch.device("cuda")
    if torch.cuda.get_device_name(0) != pol["scientific_environment_lock"]["required_gpu_name"]:
        raise SystemExit("gpu_mismatch")

    split = json.loads((ROOT / pol["split"]["manifest_rel"]).read_text(encoding="utf-8"))
    meta = load_meta(ROOT / "metadata.csv")
    labels_obj = json.loads((ROOT / pol["split"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    labels = {
        sid: int(g2i[str(row.get("text_gloss") or "").strip()])
        for sid, row in meta.items()
        if str(row.get("text_gloss") or "").strip() in g2i
    }
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ids = load_split_role_ids(split, "val")[:8]
    ds = VSL400MSSTAv5Dataset(
        ids,
        rgb_root=ROOT / pol["feature_roots_read_only"]["rgb_rel"],
        skel_root=ROOT / pol["feature_roots_read_only"]["skel_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    batch = collate_v5([ds[0]])
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    model = build_ms_sta_v6("b4_v6_ctr").to(device)
    ckpt = torch.load(ROOT / pol["b4"]["checkpoint_rel"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def fwd() -> None:
        with torch.inference_mode():
            out = model(x)
            if not torch.isfinite(out["logits"]).all():
                raise RuntimeError("non_finite_logits")

    timings = cuda_ms(fwd, n=int(args.trials), warmup=int(args.warmup))
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with RAW.open("w", encoding="utf-8") as f:
        for i, ms in enumerate(timings):
            f.write(json.dumps({"i": i, "model_forward_ms": ms}) + "\n")

    prior_rel = pol["amdahl_prior_from_r4b_stage_breakdown"]["source_rel"]
    prior = json.loads((ROOT / prior_rel).read_text(encoding="utf-8"))
    prior_stages = prior["baseline_b4_hot"]["stage_medians_ms"]
    prior_total = float(prior["baseline_b4_hot"]["total_median_ms"])
    head_med = float(statistics.median(timings))
    amdahl = {
        "head_only_max_e2e_fraction_if_head_instant": float(prior_stages["model_forward"]) / prior_total,
        "remeasured_head_median_ms": head_med,
        "remeasured_head_as_fraction_of_prior_e2e": head_med / prior_total,
        "rgb_slowfast_roi_fraction_prior": float(prior_stages["rgb_representation"]) / prior_total,
        "decode_fraction_prior": float(prior_stages["video_decode"]) / prior_total,
        "interpretation": "Head-only optimization cannot meet 15% e2e gate; target SlowFast/ROI then decode.",
    }

    report = {
        "schema": "ms_sta_v6_b4q_v1_baseline_stage_profile_v1",
        "status": "PARTIAL_HEAD_REMEASURE_PLUS_R4B_PRIOR",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip(),
        "policy_sha256": sha256_file(ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json"),
        "baseline_lock_sha256": sha256_file(ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_BASELINE_LOCK.json"),
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cache_backed_model_forward": {
            "warmup": int(args.warmup),
            "trials": int(args.trials),
            "batch_size": 1,
            "median_ms": head_med,
            "mean_ms": float(statistics.mean(timings)),
            "p90_ms": float(sorted(timings)[int(0.9 * (len(timings) - 1))]),
            "min_ms": float(min(timings)),
            "max_ms": float(max(timings)),
            "raw_jsonl_rel": "Docs/results/ms_sta_v6_b4q/B4Q_V1_BASELINE_HEAD_TIMINGS.jsonl",
            "raw_sha256": sha256_file(RAW),
            "param_count": int(count_parameters(model)),
        },
        "prior_r4b_stage_matched_e2e": {
            "source_rel": prior_rel,
            "source_sha256": sha256_file(ROOT / prior_rel),
            "total_median_ms": prior_total,
            "stage_medians_ms": prior_stages,
            "note": "Supporting prior for Amdahl/redirect; full raw e2e remeasure under B4Q protocol still required for Gate A complete.",
        },
        "amdahl": amdahl,
        "gate_a": "PARTIAL",
        "remaining_for_gate_a_pass": [
            "remeasure_video_decode_cpu_wall",
            "remeasure_rgb_slowfast_roi_cuda_events",
            "remeasure_skeleton_h2d",
            "full_raw_or_sealed_e2e_ge_5_rounds",
        ],
        "e4_ready": False,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "STAGE_PROFILE_PARTIAL",
        "head_median_ms",
        head_med,
        "amdahl_head_frac",
        f"{amdahl['head_only_max_e2e_fraction_if_head_instant']:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
