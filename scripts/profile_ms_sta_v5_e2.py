#!/usr/bin/env python3
"""Phase-1 profiler: B4* vs B4-E module timing + score accounting (no train/test)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_v5 import (  # noqa: E402
    build_ms_sta_v5,
    count_parameters,
    score_element_reduction,
)
from src.models.ms_sta_v5_e2 import build_ms_sta_v5_e2, count_parameters as count_e2  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json, load_split_role_ids  # noqa: E402
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402

OUT = _ROOT / "Docs/results/ms_sta_v5_e2/E2_PROFILE_REPORT.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def _p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[int(0.9 * (len(s) - 1))]


def _paired_ci(deltas: list[float], n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return sum(deltas) / n, means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]


def _load_batch(policy: dict, batch_size: int) -> dict[str, Any]:
    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    val_ids = load_split_role_ids(split, "val")[:batch_size]
    ds = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    return collate_v5([ds[i] for i in range(len(ds))])


def _time_forward(model, batch, device, warmup: int, trials: int, use_amp: bool) -> dict[str, Any]:
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(x)
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return {
        "median_sec": _median(times),
        "p90_sec": _p90(times),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "times_sec": times,
    }


def _module_profile(model, batch, device) -> dict[str, float]:
    """Rough sequential stage timing via CUDA events on one forward with hooks."""
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    stages: dict[str, float] = {}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def make_hooks(name: str, mod: torch.nn.Module) -> None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        def pre(_m, _inp):
            start.record()

        def post(_m, _inp, _out):
            end.record()
            events.append((name, start, end))

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    if hasattr(model, "skel_encoder"):
        make_hooks("joint_encoder", model.skel_encoder)
    if getattr(model, "skel_temporal", None) is not None:
        make_hooks("skel_temporal", model.skel_temporal)
    if getattr(model, "rgb_temporal", None) is not None:
        make_hooks("rgb_temporal", model.rgb_temporal)
    if getattr(model, "cross", None) is not None:
        make_hooks("cross_attention", model.cross)
    if getattr(model, "gate", None) is not None:
        make_hooks("gate", model.gate)
    if getattr(model, "cls", None) is not None:
        make_hooks("classifier", model.cls)
    if getattr(model, "early_reducer", None) is not None:
        make_hooks("early_reducer", model.early_reducer)

    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()
        events.clear()
        _ = model(x)
        torch.cuda.synchronize()
    for name, s, e in events:
        stages[name] = stages.get(name, 0.0) + float(s.elapsed_time(e))
    # average if multiple (hooks fire once per forward here)
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("profile_requires_cuda")

    policy = load_json(_ROOT / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json")
    e_policy = load_json(_ROOT / "Docs/results/ms_sta_v5_e_policy/V5_E_POLICY.json")
    freeze = load_json(_ROOT / "Docs/results/ms_sta_v5_oneshot/PRETEST_FREEZE.json")
    b4_meta = freeze["checkpoints_seed42"]["b4_star_fixed"]
    e_confirm = load_json(_ROOT / "Docs/results/ms_sta_v5_e_runs/CONFIRM_RESULTS.json")["trials"]
    e42 = next(t for t in e_confirm if int(t["seed"]) == 42)

    device = torch.device("cuda")
    report: dict[str, Any] = {
        "schema": "ms_sta_v5_e2_profile_report_v1",
        "gpu": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "trials": args.trials,
        "score_formula": "1 - ceil(T/s)^2 / T^2",
        "score_examples": {
            "T32_s4": score_element_reduction(32, 4),
            "T33_s4": score_element_reduction(33, 4),
        },
        "models": {},
        "verdict_notes": [],
    }

    # Score distribution on val lengths.
    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    val_ids = load_split_role_ids(split, "val")
    ds_all = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    reductions = [float(score_element_reduction(int(ds_all[i]["skel_len"]), 4)) for i in range(len(ds_all))]
    report["val_score_reduction_s4"] = {
        "n": len(reductions),
        "mean": sum(reductions) / len(reductions),
        "min": min(reductions),
        "max": max(reductions),
        "exact_9375_fraction": sum(1 for r in reductions if abs(r - 0.9375) < 1e-9) / len(reductions),
    }

    configs = [
        ("b4_star_fixed", "b4", b4_meta["checkpoint_rel"], float(b4_meta["dropout"]), policy),
        ("b4_star_e_stride4", "e1", e42["best_checkpoint"], float(e42["dropout"]), e_policy),
    ]
    # Also profile fresh e2_lite_s2 randomly init for module share (no ckpt).
    for batch_size in (1, 8):
        batch = _load_batch(policy, batch_size)
        for variant, key, ckpt_rel, dropout, pol in configs:
            model = build_ms_sta_v5(
                variant,
                num_classes=400,
                d_model=int(pol["model"]["d_model"]),
                nhead=int(pol["model"]["nhead"]),
                temporal_layers=int(pol["model"]["temporal_layers"]),
                dropout=dropout,
                alpha_max=float(pol["model"]["alpha_max"]),
            ).to(device)
            ckpt = torch.load(_ROOT / ckpt_rel, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()
            entry = report["models"].setdefault(
                variant,
                {
                    "param_count": count_parameters(model),
                    "checkpoint_rel": ckpt_rel,
                    "checkpoint_sha256": sha256_file(_ROOT / ckpt_rel),
                },
            )
            for amp in (False, True):
                tag = "amp" if amp else "fp32"
                # Interleaved A/B with B4 for paired delta when comparing e to b4
                stats = _time_forward(model, batch, device, args.warmup, args.trials, amp)
                entry[f"batch{batch_size}_{tag}"] = {
                    "median_sec": stats["median_sec"],
                    "p90_sec": stats["p90_sec"],
                    "peak_vram_bytes": stats["peak_vram_bytes"],
                }
                if key == "e1" and "b4_star_fixed" in report["models"]:
                    b4_times = report["models"]["b4_star_fixed"].get(f"_raw_b{batch_size}_{tag}")
                    if b4_times:
                        deltas = [e - b for e, b in zip(stats["times_sec"], b4_times)]
                        mean_d, lo, hi = _paired_ci(deltas)
                        entry[f"batch{batch_size}_{tag}"]["paired_delta_e_minus_b4"] = {
                            "mean_sec": mean_d,
                            "ci95_sec": [lo, hi],
                            "median_reduction": 1.0
                            - (
                                stats["median_sec"]
                                / max(
                                    report["models"]["b4_star_fixed"][f"batch{batch_size}_{tag}"][
                                        "median_sec"
                                    ],
                                    1e-12,
                                )
                            ),
                        }
                entry[f"_raw_b{batch_size}_{tag}"] = stats["times_sec"]
            entry[f"module_ms_batch{batch_size}"] = _module_profile(model, batch, device)
            del model
            torch.cuda.empty_cache()

        # Fresh E2 lite init profile (architecture share only).
        e2 = build_ms_sta_v5_e2("e2_lite_s2", dropout=0.1).to(device).eval()
        e2_stats = _time_forward(e2, batch, device, min(10, args.warmup), min(50, args.trials), False)
        e2_entry = report["models"].setdefault(
            "e2_lite_s2_random_init",
            {
                "param_count": count_e2(e2),
                "note": "untrained architecture probe only",
            },
        )
        e2_entry[f"batch{batch_size}_fp32"] = {
            "median_sec": e2_stats["median_sec"],
            "p90_sec": e2_stats["p90_sec"],
            "peak_vram_bytes": e2_stats["peak_vram_bytes"],
        }
        e2_entry[f"module_ms_batch{batch_size}"] = _module_profile(e2, batch, device)
        del e2
        torch.cuda.empty_cache()

    # Strip raw times from final report (keep paired stats).
    for _v, ent in report["models"].items():
        for k in list(ent.keys()):
            if k.startswith("_raw_"):
                del ent[k]

    # Verdict: if skel temporal share of B4* is small, prefer lite width candidates.
    mod = report["models"]["b4_star_fixed"].get("module_ms_batch8") or {}
    total = sum(mod.values()) or 1.0
    skel_share = float(mod.get("skel_temporal", 0.0)) / total
    joint_share = float(mod.get("joint_encoder", 0.0)) / total
    report["b4_module_share_batch8"] = {k: float(v) / total for k, v in mod.items()}
    if skel_share < 0.15:
        report["verdict_notes"].append(
            "Skeleton temporal share <15% of profiled modules → prefer e2_lite width/layer reduction; "
            "do not assume attention-score cut alone yields speed gate."
        )
        report["preferred_pilot_emphasis"] = "lite"
    else:
        report["verdict_notes"].append("Skeleton temporal is material; early reduction remains justified.")
        report["preferred_pilot_emphasis"] = "balanced"
    report["joint_encoder_share"] = joint_share
    report["skel_temporal_share"] = skel_share
    report["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    sha = atomic_write(OUT, report)
    print(json.dumps({"event": "E2_PROFILE_WRITTEN", "sha256": sha, "skel_share": skel_share, "joint_share": joint_share, "preferred": report["preferred_pilot_emphasis"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
