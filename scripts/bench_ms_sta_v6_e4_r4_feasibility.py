# -*- coding: utf-8 -*-
"""E4-R4 pilot CUDA feasibility benchmark (create-only evidence).

Uses protocol-locked candidates. Remeasures B4 in-session.
Does not train, extract full corpus, or mutate B4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.e4_r4_joint_layouts import layout_indices  # noqa: E402
from src.data_prep.e4_r4_rgb_contract import pixels_per_frame  # noqa: E402
from src.data_prep.e4_r4_skel_compact import compact_skeleton_arrays, load_sealed_skeleton  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters as count_b4  # noqa: E402
from src.models.ms_sta_v6_e4_r4 import build_ms_sta_v6_e4_r4, count_parameters  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json  # noqa: E402
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4"
PROTOCOL = OUT / "E4_R4_FEASIBILITY_PROTOCOL.json"


def die(msg: str) -> None:
    print(f"E4_R4_FEASIBILITY_FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_create_only(path: Path, payload: dict | str) -> str:
    if path.is_file():
        die(f"refuse_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return float(s[len(s) // 2])


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    return float(s[max(0, int(0.9 * (len(s) - 1)))])


def mean(xs: list[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def bootstrap_speedup_ci(
    base: list[float],
    cand: list[float],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(cand, dtype=np.float64)
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    speeds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mb = float(np.median(b[idx]))
        mc = float(np.median(c[idx]))
        speeds.append((mb - mc) / max(mb, 1e-9))
    lo, hi = np.quantile(speeds, [0.025, 0.975])
    return {
        "speedup_median": float(np.median(speeds)),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }


def cuda_time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def pad_skel_to(t_max: int, arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    coords = arrays["coordinates"]
    t = int(coords.shape[0])
    j = int(coords.shape[1])
    out_c = np.zeros((t_max, j, 3), dtype=np.float32)
    out_v = np.zeros((t_max, j, 3), dtype=np.float32)
    out_val = np.zeros((t_max, j), dtype=np.float32)
    pad = np.ones((t_max,), dtype=bool)
    n = min(t, t_max)
    out_c[:n] = coords[:n]
    out_v[:n] = arrays["velocity"][:n]
    out_val[:n] = arrays["validity"][:n]
    pad[:n] = False
    return {
        "coordinates": torch.from_numpy(out_c),
        "velocity": torch.from_numpy(out_v),
        "validity": torch.from_numpy(out_val),
        "skel_pad_mask": torch.from_numpy(pad),
    }


def load_e4_batch(feature_root: Path, sid: str, rgb_id: str, skel: str, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(feature_root / f"rgb_{rgb_id}" / f"{sid}.npz", allow_pickle=False) as z:
        rgb = np.asarray(z["rgb_tensor"], dtype=np.float32)
        tvalid = np.asarray(z["temporal_valid"], dtype=bool)
        seq = int(np.asarray(z["sequence_length"]).reshape(-1)[0])
    with np.load(feature_root / f"skel_{skel}" / f"{sid}.npz", allow_pickle=False) as z:
        sk = {k: z[k] for k in ("coordinates", "velocity", "validity")}
        sk_seq = int(np.asarray(z["sequence_length"]).reshape(-1)[0])
    t_max = int(rgb.shape[0])
    skel_t = pad_skel_to(t_max, sk)
    # Align validity: RGB pad + skel pad
    batch = {
        "rgb_tensor": torch.from_numpy(rgb).unsqueeze(0).to(device, non_blocking=True),
        "temporal_valid": torch.from_numpy(tvalid).unsqueeze(0).to(device, non_blocking=True),
        "coordinates": skel_t["coordinates"].unsqueeze(0).to(device, non_blocking=True),
        "velocity": skel_t["velocity"].unsqueeze(0).to(device, non_blocking=True),
        "validity": skel_t["validity"].unsqueeze(0).to(device, non_blocking=True),
        "skel_pad_mask": skel_t["skel_pad_mask"].unsqueeze(0).to(device, non_blocking=True),
        "seq_len": torch.tensor([min(seq, sk_seq)], device=device),
    }
    return batch


def load_b4_batch(ds: VSL400MSSTAv5Dataset, index: int, device: torch.device) -> dict[str, Any]:
    batch = collate_v5([ds[index]])
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def peak_vram() -> dict[str, float]:
    return {
        "allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }


def correctness_e4(model, batch, device) -> list[str]:
    fails = []
    model.eval()
    with torch.inference_mode():
        out1 = model(batch)
        out2 = model(batch)
    logits = out1["logits"]
    if tuple(logits.shape) != (1, 400):
        fails.append(f"bad_shape:{tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        fails.append("nonfinite")
    if not torch.allclose(out1["logits"], out2["logits"], atol=1e-5, rtol=0):
        fails.append("nondeterministic")
    # padding invariance: zeroing already-padded frames should not change logits much
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    pad = b2["skel_pad_mask"]
    if pad.any():
        b2["coordinates"] = b2["coordinates"].masked_fill(pad.unsqueeze(-1).unsqueeze(-1), 123.0)
        with torch.inference_mode():
            out3 = model(b2)
        if not torch.allclose(out1["logits"], out3["logits"], atol=1e-4, rtol=0):
            fails.append("pad_invariance_fail")
    # missing region smoke: zero validity on half joints
    b3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b3["validity"][:, :, ::2] = 0
    with torch.inference_mode():
        out4 = model(b3)
    if not torch.isfinite(out4["logits"]).all():
        fails.append("missing_region_nonfinite")
    if logits.device.type != "cuda":
        fails.append("cpu_fallback")
    return fails


def correctness_b4(model, batch) -> list[str]:
    fails = []
    model.eval()
    with torch.inference_mode():
        out1 = model(batch)
        out2 = model(batch)
    logits = out1["logits"]
    if tuple(logits.shape) != (1, 400):
        fails.append(f"bad_shape:{tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        fails.append("nonfinite")
    if not torch.allclose(out1["logits"], out2["logits"], atol=1e-5, rtol=0):
        fails.append("nondeterministic")
    if logits.device.type != "cuda":
        fails.append("cpu_fallback")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    if not PROTOCOL.is_file():
        die("missing_protocol_write_before_measure")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_FIRST_MEASUREMENT":
        die("protocol_not_frozen")

    if not torch.cuda.is_available():
        die("cuda_unavailable")
    gpu = torch.cuda.get_device_name(0)
    if gpu != protocol["required_gpu_name"]:
        die(f"gpu_mismatch:{gpu}")

    # Bind SHA checks
    binds = protocol["binds"]
    checks = {
        "representation_audit_sha256": OUT / "E4_R4_PILOT_REPRESENTATION_AUDIT.json",
        "alignment_audit_sha256": OUT / "E4_R4_PILOT_ALIGNMENT_AUDIT.json",
        "visual_qa_sha256": OUT / "E4_R4_PILOT_MODEL_ASSISTED_VISUAL_QA.json",
        "policy_amendment_sha256": ROOT
        / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4_POLICY_AMENDMENT_CAMPAIGN_ORDER.json",
        "pilot_freeze_sha256": OUT / "E4_R4_PILOT_SAMPLE_FREEZE.json",
    }
    for key, path in checks.items():
        got = sha256_file(path)
        if got != binds[key]:
            die(f"sha_mismatch:{key}:{got}")

    freeze = json.loads((OUT / "E4_R4_PILOT_SAMPLE_FREEZE.json").read_text(encoding="utf-8"))
    ids = list(freeze["pilot_train_ids"]) + list(freeze["pilot_val_ids"])
    split = load_json(ROOT / "data/processed/vsl400_retrain_v4_split_manifest.json")
    test_ids = set(map(str, split["test_ids"]))
    if set(ids) & test_ids:
        die("test_id_leak")

    device = torch.device("cuda:0")
    warmup = int(args.warmup or protocol["benchmark"]["primary"]["warmup_min"])
    trials = int(args.trials or protocol["benchmark"]["primary"]["trials_min"])
    if warmup < 100 or trials < 500:
        die("warmup_or_trials_below_protocol")

    feature_root = ROOT / binds["pilot_feature_root_rel"]
    meta = load_metadata_by_id(ROOT / "metadata.csv")
    labels_obj = load_json(ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}

    b4_ds = VSL400MSSTAv5Dataset(
        ids,
        rgb_root=ROOT / binds["b4_rgb_root_rel"],
        skel_root=ROOT / binds["b4_skel_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
        require_all=True,
    )

    # Build models
    torch.manual_seed(42)
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    candidates = protocol["candidates"]
    e4_models = {}
    for c in candidates:
        m = build_ms_sta_v6_e4_r4(rgb_id=c["rgb"], skel_layout=c["skel"], d_model=128)
        e4_models[c["candidate_id"]] = m.to(device).eval()

    # Identity / complexity table
    identity = []
    for c in candidates:
        cid = c["candidate_id"]
        m = e4_models[cid]
        n_j = len(layout_indices(c["skel"]))
        identity.append(
            {
                "candidate_id": cid,
                "rgb": c["rgb"],
                "skel": c["skel"],
                "promote_forbidden": c.get("promote_forbidden", False),
                "param_count": count_parameters(m),
                "n_joints": n_j,
                "rgb_pixels_per_frame": pixels_per_frame(c["rgb"]),
                "rgb_token_proxy": 32,  # temporal tokens after encode
                "skel_token_proxy": 32,
                "temporal_attn_score_elements_rgb": 32 * 32,
                "checkpoint_state": "untrained_feasibility_model",
            }
        )
    identity.append(
        {
            "candidate_id": "B4_V6_CTR",
            "param_count": count_b4(b4),
            "rgb_tokens": "4_regions_x_L_x_2304",
            "skel_joints": 75,
            "checkpoint_state": "untrained_feasibility_comparator_same_arch_as_teacher",
        }
    )

    # Preload one batch per id for forward timing (cycle through ids)
    print("Preloading B4/E4 batches...", flush=True)
    b4_batches = [load_b4_batch(b4_ds, i, device) for i in range(len(ids))]
    # Use first id per candidate for correctness; cycle all for timing
    sample_sid = ids[0]

    correctness = {"B4_V6_CTR": correctness_b4(b4, b4_batches[0])}
    for c in candidates:
        cid = c["candidate_id"]
        batch = load_e4_batch(feature_root, sample_sid, c["rgb"], c["skel"], device)
        correctness[cid] = correctness_e4(e4_models[cid], batch, device)
        if correctness[cid]:
            print("CORRECTNESS_FAIL", cid, correctness[cid], flush=True)

    # Cache E4 batches for all ids (memory heavy but 132*8 ok)
    e4_batches: dict[str, list] = {c["candidate_id"]: [] for c in candidates}
    for i, sid in enumerate(ids):
        for c in candidates:
            e4_batches[c["candidate_id"]].append(
                load_e4_batch(feature_root, sid, c["rgb"], c["skel"], device)
            )
        if (i + 1) % 20 == 0:
            print(f" loaded {i+1}/{len(ids)}", flush=True)

    model_ids = ["B4_V6_CTR"] + [c["candidate_id"] for c in candidates]
    # Drop correctness failures from latency promotion path (still measure for evidence)
    rng = random.Random(42)

    def run_forward(mid: str, idx: int) -> None:
        if mid == "B4_V6_CTR":
            with torch.inference_mode():
                b4(b4_batches[idx % len(b4_batches)])
        else:
            with torch.inference_mode():
                e4_models[mid](e4_batches[mid][idx % len(e4_batches[mid])])

    # Warmup interleaved
    print("Warmup...", flush=True)
    for w in range(warmup):
        for mid in model_ids:
            run_forward(mid, w)

    raw_path = OUT / "E4_R4_FEASIBILITY_RAW_RESULTS.jsonl"
    if raw_path.is_file():
        die("raw_results_exists")
    raw_f = raw_path.open("w", encoding="utf-8")

    timings: dict[str, list[float]] = {m: [] for m in model_ids}
    print(f"Primary trials={trials}...", flush=True)
    for t in range(trials):
        order = model_ids[:]
        rng.shuffle(order)
        for mid in order:
            torch.cuda.reset_peak_memory_stats()
            ms = cuda_time_ms(lambda mid=mid, t=t: run_forward(mid, t))
            timings[mid].append(ms)
            rec = {
                "tier": "A_model_forward",
                "trial": t,
                "model_id": mid,
                "latency_ms": ms,
                "sample_index": t % len(ids),
                "vram": peak_vram(),
            }
            raw_f.write(json.dumps(rec) + "\n")
        if (t + 1) % 50 == 0:
            print(f"  trial {t+1}/{trials}", flush=True)

    # Cache-backed total: load from disk + h2d + forward (subset trials)
    cache_trials = min(100, trials)
    cache_timings: dict[str, list[float]] = {m: [] for m in model_ids}
    print(f"Cache-backed trials={cache_trials}...", flush=True)
    for t in range(cache_trials):
        sid = ids[t % len(ids)]
        order = model_ids[:]
        rng.shuffle(order)
        for mid in order:
            def work(mid=mid, sid=sid, t=t):
                if mid == "B4_V6_CTR":
                    batch = load_b4_batch(b4_ds, t % len(ids), device)
                    with torch.inference_mode():
                        b4(batch)
                else:
                    c = next(x for x in candidates if x["candidate_id"] == mid)
                    batch = load_e4_batch(feature_root, sid, c["rgb"], c["skel"], device)
                    with torch.inference_mode():
                        e4_models[mid](batch)

            ms = cuda_time_ms(work)
            cache_timings[mid].append(ms)
            raw_f.write(
                json.dumps(
                    {
                        "tier": "B_cache_backed_total",
                        "trial": t,
                        "model_id": mid,
                        "latency_ms": ms,
                        "sample_id": sid,
                    }
                )
                + "\n"
            )

    # Preprocess subset (skel compact only — RGB pack is CPU/video; measure skel+npz stat)
    prep_n = int(protocol.get("preprocess_subset_n", 24))
    prep_ids = ids[:prep_n]
    sealed_root = ROOT / binds["b4_skel_root_rel"] / "skeleton_v3"
    prep_ms = {"skel_compact_S_A": [], "skel_compact_S_B": []}
    for sid in prep_ids:
        sealed = load_sealed_skeleton(sealed_root / f"{sid}.npz")
        for layout, key in (("S_A", "skel_compact_S_A"), ("S_B", "skel_compact_S_B")):
            t0 = time.perf_counter()
            compact_skeleton_arrays(sealed, layout)
            prep_ms[key].append((time.perf_counter() - t0) * 1000.0)
            raw_f.write(
                json.dumps(
                    {
                        "tier": "C_representation_preprocess",
                        "model_id": key,
                        "sample_id": sid,
                        "latency_ms": prep_ms[key][-1],
                    }
                )
                + "\n"
            )
    raw_f.close()

    # Secondary batch=8 / AMP (not primary gate)
    secondary = {}
    try:
        # batch 8 forward B4 vs one fast-looking candidate later filled
        idxs = list(range(min(8, len(ids))))
        b4_b8 = collate_v5([b4_ds[i] for i in idxs])
        b4_b8 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b4_b8.items()}
        # pad e4 to batch 8 for R0_S_A
        cid0 = "R0_S_A_d128"
        e4_list = [e4_batches[cid0][i] for i in idxs]

        def stack_e4(batches):
            keys = ["rgb_tensor", "temporal_valid", "coordinates", "velocity", "validity", "skel_pad_mask"]
            return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

        e4_b8 = stack_e4(e4_list)
        for _ in range(20):
            with torch.inference_mode():
                b4(b4_b8)
                e4_models[cid0](e4_b8)
        b4_b8_ms = [cuda_time_ms(lambda: b4(b4_b8)) for _ in range(50)]
        e4_b8_ms = [cuda_time_ms(lambda: e4_models[cid0](e4_b8)) for _ in range(50)]
        secondary["batch8_fp32"] = {
            "B4_V6_CTR_median_ms": median(b4_b8_ms),
            f"{cid0}_median_ms": median(e4_b8_ms),
        }
        # AMP batch1
        amp_b4 = []
        amp_e4 = []
        for i in range(50):
            with torch.inference_mode(), torch.cuda.amp.autocast():
                amp_b4.append(cuda_time_ms(lambda i=i: b4(b4_batches[i % len(b4_batches)])))
                amp_e4.append(
                    cuda_time_ms(lambda i=i: e4_models[cid0](e4_batches[cid0][i % len(ids)]))
                )
        secondary["batch1_amp"] = {
            "B4_V6_CTR_median_ms": median(amp_b4),
            f"{cid0}_median_ms": median(amp_e4),
            "note": "secondary_not_primary_gate",
        }
    except Exception as e:
        secondary["error"] = str(e)

    gate = protocol["feasibility_gate"]
    b4_med = median(timings["B4_V6_CTR"])
    b4_p90 = p90(timings["B4_V6_CTR"])
    b4_cache = median(cache_timings["B4_V6_CTR"])

    rows = []
    promoted = []
    for c in candidates:
        cid = c["candidate_id"]
        corr_fail = correctness[cid]
        med = median(timings[cid])
        p90v = p90(timings[cid])
        speedup = (b4_med - med) / max(b4_med, 1e-9)
        ci = bootstrap_speedup_ci(
            timings["B4_V6_CTR"],
            timings[cid],
            n_boot=int(gate["n_bootstrap"]),
            seed=int(gate["bootstrap_seed"]),
        )
        cache_med = median(cache_timings[cid])
        cache_speedup = (b4_cache - cache_med) / max(b4_cache, 1e-9)
        # VRAM from last forward peaks stored approximately via remeasure
        torch.cuda.reset_peak_memory_stats()
        run_forward(cid, 0)
        vram = peak_vram()
        torch.cuda.reset_peak_memory_stats()
        run_forward("B4_V6_CTR", 0)
        vram_b4 = peak_vram()
        vram_regress = (vram["allocated_mb"] - vram_b4["allocated_mb"]) / max(vram_b4["allocated_mb"], 1e-9)

        reasons = []
        if corr_fail:
            reasons.append("correctness_fail")
        if c.get("promote_forbidden"):
            reasons.append("promote_forbidden_policy")
        if speedup < float(gate["primary_median_speedup_min"]):
            reasons.append("median_speedup_lt_8pct")
        if p90v > b4_p90:
            reasons.append("p90_regress")
        if ci["ci95_lo"] <= 0:
            reasons.append("bootstrap_ci_lo_le_0")
        if vram_regress > float(gate["peak_vram_material_regress_max_frac"]):
            reasons.append("vram_regress")
        if cache_med > b4_cache:
            reasons.append("cache_backed_slower_than_b4")

        ok = len(reasons) == 0
        row = {
            "candidate_id": cid,
            "correctness_pass": not corr_fail,
            "correctness_failures": corr_fail,
            "model_forward_median_ms": med,
            "model_forward_p90_ms": p90v,
            "model_forward_mean_ms": mean(timings[cid]),
            "speedup_vs_b4_median": speedup,
            "bootstrap_ci": ci,
            "cache_backed_median_ms": cache_med,
            "cache_backed_speedup_vs_b4": cache_speedup,
            "vram": vram,
            "vram_regress_frac_vs_b4": vram_regress,
            "param_count": count_parameters(e4_models[cid]),
            "promote_forbidden": c.get("promote_forbidden", False),
            "gate_pass": ok,
            "reject_reasons": reasons,
        }
        rows.append(row)
        if ok:
            promoted.append(row)

    # Rank promote
    promoted_sorted = sorted(
        promoted,
        key=lambda r: (
            -r["cache_backed_speedup_vs_b4"],
            r["model_forward_p90_ms"],
            r["vram"]["allocated_mb"],
            r["param_count"],
            r["candidate_id"],
        ),
    )
    promote_max = int(gate["promote_max"])
    selected = promoted_sorted[:promote_max]
    verdict = "E4_R4_FEASIBILITY_NO_GO" if not selected else "E4_R4_FEASIBILITY_PROMOTE"
    if not selected:
        status = "E4_R4_FEASIBILITY_NO_GO"
    else:
        status = "E4_R4_FEASIBILITY_PROMOTE"

    # Recompute aggregates from raw jsonl
    recomputed = {m: [] for m in model_ids}
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("tier") == "A_model_forward":
                recomputed[rec["model_id"]].append(float(rec["latency_ms"]))
    for m in model_ids:
        if len(recomputed[m]) != trials:
            die(f"raw_trial_count_mismatch:{m}:{len(recomputed[m])}")
        if abs(median(recomputed[m]) - median(timings[m])) > 1e-9:
            die(f"aggregate_mismatch:{m}")

    env = {
        "schema": "ms_sta_v6_e4_r4_feasibility_environment_v1",
        "gpu_name": gpu,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    env_sha = write_create_only(OUT / "E4_R4_FEASIBILITY_ENVIRONMENT.json", env)

    report = {
        "schema": "ms_sta_v6_e4_r4_feasibility_report_v1",
        "status": status,
        "verdict": verdict,
        "created_at": env["created_at"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "environment_sha256": env_sha,
        "raw_results_sha256": sha256_file(raw_path),
        "binds": binds,
        "warmup": warmup,
        "trials_primary": trials,
        "cache_trials": cache_trials,
        "baseline_b4": {
            "model_forward_median_ms": b4_med,
            "model_forward_p90_ms": b4_p90,
            "model_forward_mean_ms": mean(timings["B4_V6_CTR"]),
            "cache_backed_median_ms": b4_cache,
            "correctness_pass": not correctness["B4_V6_CTR"],
            "correctness_failures": correctness["B4_V6_CTR"],
            "param_count": count_b4(b4),
        },
        "candidate_identity": identity,
        "candidate_rows": rows,
        "promoted": [r["candidate_id"] for r in selected],
        "promoted_detail": selected,
        "preprocess_subset_median_ms": {k: median(v) for k, v in prep_ms.items()},
        "secondary": secondary,
        "gate": gate,
        "e4_ready": False,
        "forbid_full_corpus_in_this_task": True,
        "note": "Model-forward is not end-to-end raw-video latency.",
    }
    report_sha = write_create_only(OUT / "E4_R4_FEASIBILITY_REPORT.json", report)

    closeout = {
        "schema": "ms_sta_v6_e4_r4_feasibility_closeout_v1",
        "status": status,
        "verdict": verdict,
        "promoted_candidate_ids": [r["candidate_id"] for r in selected],
        "n_promoted": len(selected),
        "next_phase_allowed": (
            "await_full_corpus_authorization_for_promoted_only"
            if selected
            else "stop_no_go_no_full_corpus_no_hpo"
        ),
        "e4_ready": False,
        "report_sha256": report_sha,
        "created_at": env["created_at"],
    }
    close_sha = write_create_only(OUT / "E4_R4_FEASIBILITY_CLOSEOUT.json", closeout)

    evidence = {
        "schema": "ms_sta_v6_e4_r4_evidence_index_v1",
        "created_at": env["created_at"],
        "artifacts": {
            "protocol": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4_FEASIBILITY_PROTOCOL.json", "sha256": sha256_file(PROTOCOL)},
            "raw_results": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4_FEASIBILITY_RAW_RESULTS.jsonl", "sha256": sha256_file(raw_path)},
            "report": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4_FEASIBILITY_REPORT.json", "sha256": report_sha},
            "environment": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4_FEASIBILITY_ENVIRONMENT.json", "sha256": env_sha},
            "closeout": {"rel": "Docs/results/ms_sta_v6_e4_r4/E4_R4_FEASIBILITY_CLOSEOUT.json", "sha256": close_sha},
        },
        "verdict": verdict,
    }
    write_create_only(OUT / "EVIDENCE_INDEX.json", evidence)

    print("VERDICT", verdict)
    print("B4_median_ms", round(b4_med, 3))
    for r in rows:
        print(
            r["candidate_id"],
            "med",
            round(r["model_forward_median_ms"], 3),
            "speedup",
            round(r["speedup_vs_b4_median"], 4),
            "pass",
            r["gate_pass"],
            r["reject_reasons"],
        )
    print("PROMOTED", [r["candidate_id"] for r in selected])
    return 0 if selected else 3


if __name__ == "__main__":
    raise SystemExit(main())
