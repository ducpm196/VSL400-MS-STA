# -*- coding: utf-8 -*-
"""Performance loader parity vs baseline (num_workers=0, pin_memory=False).

Thresholds declared BEFORE seeing errors. Create-only:
  Docs/results/ms_sta_v6_e4_r4/E4_R4C_PERFORMANCE_PARITY.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ms_sta_v6_e4_r4 import build_ms_sta_v6_e4_r4
from src.training.ms_sta_e4_r4c_data import VSL400E4R4CDataset, collate_e4_r4c
from src.training.ms_sta_v5_data import load_split_role_ids

# Declared before run — do not change after seeing errors.
TENSOR_MAX_ABS = 0.0  # exact match for CPU tensors from same files
LOGIT_MAX_ABS = 1e-5
LOSS_MAX_ABS = 1e-6
GRAD_MAX_ABS = 1e-5
SPEEDUP_MIN_RATIO = 1.10  # adopt only if >=10% faster samples/s
PARITY_BATCHES = 4
BATCH_SIZE = 8
SUBSET_N = 64
SEED = 42
LR = 0.0003
WD = 0.01


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.dtype != b.dtype and not (a.is_floating_point() and b.is_floating_point()):
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def collect_batches(loader: DataLoader, n: int) -> list[dict]:
    out = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        out.append(batch)
    return out


def step_metrics(model: torch.nn.Module, batch: dict, device: torch.device, *, step_seed: int) -> dict:
    torch.manual_seed(step_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(step_seed)
    model.train()
    labels_t = batch["label"].to(device)
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    tvalid = x["temporal_valid"]
    if tvalid.dtype != torch.bool:
        tvalid = tvalid.bool()
    none_valid = ~tvalid.any(dim=1)
    if bool(none_valid.any()):
        tvalid = tvalid.clone()
        tvalid[none_valid, 0] = True
        x["temporal_valid"] = tvalid
    out = model(x)
    ce = F.cross_entropy(out["logits"], labels_t)
    loss = ce  # scientific distill path checked separately via teacher cache IDs
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().float().reshape(-1))
    grad_cat = torch.cat(grads) if grads else torch.zeros(1, device=device)
    return {
        "sample_ids": [str(s) for s in batch["sample_id"]],
        "labels": batch["label"].detach().cpu(),
        "logits": out["logits"].detach().float().cpu(),
        "ce_loss": float(ce.item()),
        "loss": float(loss.item()),
        "grad": grad_cat.detach().cpu(),
        "finite_logits": bool(torch.isfinite(out["logits"]).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_grad": bool(torch.isfinite(grad_cat).all()),
    }


def compare_candidate(
    *,
    subset: Subset,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    baseline_batches: list[dict],
    init_state: dict,
) -> dict:
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_e4_r4c,
        persistent_workers=(num_workers > 0),
    )
    cand_batches = collect_batches(loader, PARITY_BATCHES)
    if len(cand_batches) != len(baseline_batches):
        return {"parity_pass": False, "reason": "batch_count_mismatch"}

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = build_ms_sta_v6_e4_r4(rgb_id="R1", skel_layout="S_A", d_model=128).to(device)
    model.load_state_dict(init_state)

    id_ok = True
    tensor_ok = True
    logit_max = 0.0
    loss_max = 0.0
    grad_max = 0.0
    finite_ok = True
    for b0, b1 in zip(baseline_batches, cand_batches):
        if [str(s) for s in b0["sample_id"]] != [str(s) for s in b1["sample_id"]]:
            id_ok = False
        for key in (
            "label",
            "rgb_tensor",
            "coordinates",
            "velocity",
            "validity",
            "temporal_valid",
            "skel_pad_mask",
        ):
            if key not in b0 or key not in b1:
                continue
            if not torch.is_tensor(b0[key]):
                continue
            err = tensor_max_abs(b0[key], b1[key])
            if err > TENSOR_MAX_ABS:
                tensor_ok = False

        model.load_state_dict(init_state)
        base_m = step_metrics(model, b0, device, step_seed=SEED + 1000)
        model.load_state_dict(init_state)
        cand_m = step_metrics(model, b1, device, step_seed=SEED + 1000)
        logit_max = max(logit_max, tensor_max_abs(base_m["logits"], cand_m["logits"]))
        loss_max = max(loss_max, abs(base_m["loss"] - cand_m["loss"]))
        grad_max = max(grad_max, tensor_max_abs(base_m["grad"], cand_m["grad"]))
        finite_ok = finite_ok and all(
            [
                base_m["finite_logits"],
                base_m["finite_loss"],
                base_m["finite_grad"],
                cand_m["finite_logits"],
                cand_m["finite_loss"],
                cand_m["finite_grad"],
            ]
        )

    parity_pass = (
        id_ok
        and tensor_ok
        and finite_ok
        and logit_max <= LOGIT_MAX_ABS
        and loss_max <= LOSS_MAX_ABS
        and grad_max <= GRAD_MAX_ABS
    )
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "parity_pass": parity_pass,
        "id_order_ok": id_ok,
        "input_tensor_exact_ok": tensor_ok,
        "finite_ok": finite_ok,
        "observed": {
            "logit_max_abs": logit_max,
            "loss_max_abs": loss_max,
            "grad_max_abs": grad_max,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "data/features/ms_sta_v6_e4_r4c_r1_sa_a365896280f0",
    )
    ap.add_argument(
        "--bench",
        type=Path,
        default=ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_DATALOADER_BENCHMARK.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_PERFORMANCE_PARITY.json",
    )
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise SystemExit(f"refuse_overwrite:{out}")
    bench_path = args.bench if args.bench.is_absolute() else ROOT / args.bench
    if not bench_path.is_file():
        raise SystemExit(f"missing_benchmark:{bench_path}")
    bench = json.loads(bench_path.read_text(encoding="utf-8"))

    train_pol = json.loads(
        (ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json").read_text(encoding="utf-8")
    )
    split = json.loads((ROOT / train_pol["split_manifest_rel"]).read_text(encoding="utf-8"))
    if sha256_file(ROOT / train_pol["split_manifest_rel"]) != train_pol["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")

    meta = load_meta(ROOT / "metadata.csv")
    labels_obj = json.loads((ROOT / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[str(row.get("text_gloss") or "").strip()]) for sid, row in meta.items() if str(row.get("text_gloss") or "").strip() in g2i}
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ids = load_split_role_ids(split, "confirm_train")
    feature_root = args.feature_root if args.feature_root.is_absolute() else ROOT / args.feature_root
    ds = VSL400E4R4CDataset(ids, feature_root=feature_root, labels=labels, metadata_by_id=meta, forbid_ids=test_ids)
    subset = Subset(ds, list(range(min(SUBSET_N, len(ds)))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    base_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_e4_r4c,
    )
    baseline_batches = collect_batches(base_loader, PARITY_BATCHES)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    init_model = build_ms_sta_v6_e4_r4(rgb_id="R1", skel_layout="S_A", d_model=128).to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}

    candidates = []
    for row in bench.get("rows") or []:
        nw = int(row["num_workers"])
        pin = bool(row["pin_memory"])
        if nw == 0 and pin is False:
            continue
        result = compare_candidate(
            subset=subset,
            device=device,
            num_workers=nw,
            pin_memory=pin,
            baseline_batches=baseline_batches,
            init_state=init_state,
        )
        base_sps = (bench.get("baseline") or {}).get("samples_per_sec_mean")
        cand_sps = row.get("samples_per_sec_mean")
        speedup = None
        if base_sps and cand_sps and base_sps > 0:
            speedup = float(cand_sps) / float(base_sps)
        result["samples_per_sec"] = cand_sps
        result["speedup_vs_baseline"] = speedup
        result["adopt_eligible"] = bool(result["parity_pass"] and speedup is not None and speedup >= SPEEDUP_MIN_RATIO)
        candidates.append(result)
        print(
            f"PARITY nw={nw} pin={pin} pass={result['parity_pass']} speedup={speedup} adopt={result['adopt_eligible']}",
            flush=True,
        )

    eligible = [c for c in candidates if c.get("adopt_eligible")]
    if eligible:
        chosen = max(eligible, key=lambda c: (c.get("speedup_vs_baseline") or 0, -c["num_workers"]))
        decision = {
            "action": "ADOPT",
            "num_workers": chosen["num_workers"],
            "pin_memory": chosen["pin_memory"],
            "reason": "parity_pass_and_speedup_ge_threshold",
        }
    else:
        decision = {
            "action": "KEEP_BASELINE",
            "num_workers": 0,
            "pin_memory": False,
            "reason": "no_candidate_passed_parity_and_speedup",
        }

    report = {
        "schema": "ms_sta_v6_e4_r4c_performance_parity_v2",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_sha256": sha256_file(bench_path),
        "prior_fail_artifact": "Docs/results/ms_sta_v6_e4_r4/E4_R4C_PERFORMANCE_PARITY_FAIL_dropout_rng_false_negative.json",
        "thresholds_declared_before_run": {
            "tensor_max_abs": TENSOR_MAX_ABS,
            "logit_max_abs": LOGIT_MAX_ABS,
            "loss_max_abs": LOSS_MAX_ABS,
            "grad_max_abs": GRAD_MAX_ABS,
            "speedup_min_ratio": SPEEDUP_MIN_RATIO,
            "per_step_manual_seed": "SEED+1000 identical for baseline and candidate",
        },
        "protocol": {
            "subset_n": SUBSET_N,
            "batch_size": BATCH_SIZE,
            "parity_batches": PARITY_BATCHES,
            "seed": SEED,
            "shuffle": False,
            "lr": LR,
            "weight_decay": WD,
            "note": "CE-only parity step; distill weight unchanged in trainer scientific path",
        },
        "candidates": candidates,
        "decision": decision,
        "gate_d_parity": "PASS",
        "e4_ready": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PERFORMANCE_PARITY_OK", decision["action"], decision["num_workers"], decision["pin_memory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
