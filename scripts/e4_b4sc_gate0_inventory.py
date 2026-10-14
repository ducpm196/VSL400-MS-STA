# -*- coding: utf-8 -*-
"""E4_B4SC_V1 Gate 0 inventory — measure only; no compression / no weight mutation.

Creates create-only JSON artifacts under Docs/results/ms_sta_v6_e4_b4sc/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters  # noqa: E402
from src.models.rgb_token_v3 import FrozenSlowFastBlock4  # noqa: E402
from src.optimization.b4q.invariants import load_policy as load_b4q_policy, sha256_file  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POL_E4 = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"
PROTO = OUT / "E4_B4SC_V1_GATE0_INVENTORY_PROTOCOL.json"


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_GATE0_FAIL:{msg}")


def create_only(path: Path, obj: dict) -> None:
    if path.is_file():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_meta() -> dict[str, dict]:
    import csv

    path = ROOT / "metadata.csv"
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["video_id"]).strip()] = dict(row)
    return out


def numerical_rank(mat: np.ndarray, energy: float = 0.99) -> dict:
    """mat: (n, c) — returns SVD energy ranks (proxy, not final target)."""
    if mat.size == 0:
        return {"n": 0}
    x = mat.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"error": "svd_failed", "n_rows": int(x.shape[0]), "n_cols": int(x.shape[1])}
    s2 = s ** 2
    tot = float(s2.sum()) + 1e-12
    csum = np.cumsum(s2) / tot
    r99 = int(np.searchsorted(csum, energy) + 1)
    return {
        "n_rows": int(x.shape[0]),
        "n_cols": int(x.shape[1]),
        "singular_values_head": [float(v) for v in s[:16].tolist()],
        f"rank_energy_{int(energy*100)}": r99,
        "effective_rank_exp": float(np.exp(-np.sum((s2 / tot) * np.log((s2 / tot) + 1e-12)))),
    }


def cuda_ms(fn, *, n: int, warmup: int) -> list[float]:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        xs.append(float(starter.elapsed_time(ender)))
    return xs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=8, help="Sealed val samples for head/rank probes")
    ap.add_argument("--head-warmup", type=int, default=50)
    ap.add_argument("--head-trials", type=int, default=100)
    args = ap.parse_args()

    pol = json.loads(POL_E4.read_text(encoding="utf-8"))
    if pol.get("status") != "FROZEN":
        die("policy_not_frozen")
    proto = json.loads(PROTO.read_text(encoding="utf-8"))
    b4q = load_b4q_policy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("cuda_required")
    if torch.cuda.get_device_name(0) != pol["scientific_environment_lock"]["required_gpu_name"]:
        die("gpu_mismatch")

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- SlowFast module inventory (shapes) ---
    sf = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    shapes = {"slowfast_modules": [], "notes": []}
    for name, mod in sf.named_modules():
        n_param = sum(p.numel() for p in mod.parameters(recurse=False))
        if n_param == 0 and not any(True for _ in mod.children()):
            continue
        entry = {"name": name, "type": type(mod).__name__, "params_direct": int(n_param)}
        if hasattr(mod, "weight") and torch.is_tensor(getattr(mod, "weight", None)):
            entry["weight_shape"] = list(mod.weight.shape)
        shapes["slowfast_modules"].append(entry)
    shapes["slowfast_total_params"] = int(sum(p.numel() for p in sf.parameters()))
    shapes["load_meta"] = getattr(sf, "load_meta", {})

    # Synthetic pathway probe if forward_maps accepts fixed tensors
    try:
        # Common SlowFast 8x8: slow T'=8, fast T'=32 after sample — sizes depend on impl
        # Probe via a tiny random clip through public API if available
        slow = torch.randn(1, 3, 8, 224, 224, device=device)
        fast = torch.randn(1, 3, 32, 224, 224, device=device)
        with torch.inference_mode():
            maps = sf.forward_maps(slow, fast)
        shapes["forward_maps_probe"] = {
            "slow_in": list(slow.shape),
            "fast_in": list(fast.shape),
            "slow_map": list(maps.slow.shape),
            "fast_map": list(maps.fast.shape),
        }
    except Exception as exc:  # noqa: BLE001
        shapes["forward_maps_probe"] = {"error": repr(exc)}
        shapes["notes"].append("forward_maps_probe_failed — record real extract shapes in Gate0b if needed")

    # --- B4 head sealed probe ---
    b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
    ckpt = torch.load(ROOT / pol["b4_teacher"]["checkpoint_rel"], map_location=device, weights_only=False)
    b4.load_state_dict(ckpt["model"])
    shapes["b4_param_count"] = int(count_parameters(b4))

    split = json.loads((ROOT / pol["split"]["manifest_rel"]).read_text(encoding="utf-8"))
    meta = load_meta()
    labels_obj = json.loads((ROOT / b4q["split"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    labels = {
        sid: int(g2i[str(row.get("text_gloss") or "").strip()])
        for sid, row in meta.items()
        if str(row.get("text_gloss") or "").strip() in g2i
    }
    ids = load_split_role_ids(split, "val")[: int(args.n_samples)]
    ds = VSL400MSSTAv5Dataset(
        ids,
        rgb_root=ROOT / b4q["feature_roots_read_only"]["rgb_rel"],
        skel_root=ROOT / b4q["feature_roots_read_only"]["skel_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    batch = collate_v5([ds[i] for i in range(len(ds))])
    batch_g = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # token shapes from batch
    token_shapes = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            token_shapes[k] = list(v.shape)
    shapes["sealed_batch_tensor_shapes"] = token_shapes

    def head_fn():
        with torch.inference_mode():
            _ = b4(batch_g)

    head_ms = cuda_ms(head_fn, n=int(args.head_trials), warmup=int(args.head_warmup))
    latency = {
        "schema": "ms_sta_v6_e4_b4sc_v1_gate0_stage_latency_v1",
        "created_at": now,
        "git_head": git_head,
        "note": "Head CUDA-event only in Gate0a; full e2e decode+SlowFast uses B4Q Gate A prior until Gate0b online probe.",
        "b4_head_ms": {
            "n": len(head_ms),
            "warmup": int(args.head_warmup),
            "median": median(head_ms),
            "p90": sorted(head_ms)[int(0.9 * (len(head_ms) - 1))],
        },
        "b4q_gate_a_prior_ms": pol["b4q_latency_prior_locked"],
    }

    # Activation rank proxies on sealed RGB tokens (per region)
    rank_report = {"schema": "ms_sta_v6_e4_b4sc_v1_gate0_activation_rank_v1", "created_at": now, "regions": {}}
    for region in pol["rgb_token_v3_invariants"]["regions_order_canonical"]:
        key = f"token_{region}"
        if key not in batch:
            continue
        t = batch[key].detach().float().cpu().numpy()  # (B,L,C) or similar
        flat = t.reshape(-1, t.shape[-1])
        # subsample rows for SVD stability
        if flat.shape[0] > 2048:
            rng = np.random.default_rng(42)
            idx = rng.choice(flat.shape[0], size=2048, replace=False)
            flat = flat[idx]
        rank_report["regions"][region] = numerical_rank(flat, energy=0.99)
    rank_report["interpretation"] = (
        "Energy ranks are inventory proxies for region-aware bottleneck design; "
        "NOT locked compression targets. Lock targets only after review + kernel rounding."
    )

    # Joint / region inventory (static from policy + skeleton tensor)
    skel = batch["coordinates"].detach().cpu().numpy()
    joint_inv = {
        "schema": "ms_sta_v6_e4_b4sc_v1_gate0_region_joint_inventory_v1",
        "created_at": now,
        "regions_order": pol["rgb_token_v3_invariants"]["regions_order_canonical"],
        "skeleton_coordinates_shape": list(batch["coordinates"].shape),
        "n_joints_dim": int(skel.shape[-2]) if skel.ndim >= 3 else None,
        "policy_n_joints": pol["skeleton_v3_invariants"]["n_joints"],
        "groups_round1": {
            "hands_fingers": "DO_NOT_CUT",
            "face_head": "DO_NOT_CUT_BLIND",
            "torso_lower": "MAPPING_ONLY_AFTER_COUNTERFACTUAL",
        },
        "region_aware_projection_plan": {
            "left_hand": "highest_rank",
            "right_hand": "highest_rank",
            "face": "high_rank",
            "global_or_body": "may_be_lower",
            "forbid_mix_hand_face_before_projection": True,
        },
    }

    report = {
        "schema": "ms_sta_v6_e4_b4sc_v1_gate0_inventory_report_v1",
        "created_at": now,
        "git_head": git_head,
        "policy_sha256": sha256_file(POL_E4),
        "protocol_sha256": sha256_file(PROTO),
        "pass": True,
        "model_weights_mutated": False,
        "compression_applied": False,
        "artifacts": {
            "tensor_shapes": "E4_B4SC_V1_GATE0_TENSOR_SHAPES.json",
            "stage_latency": "E4_B4SC_V1_GATE0_STAGE_LATENCY.json",
            "activation_rank": "E4_B4SC_V1_GATE0_ACTIVATION_RANK.json",
            "region_joint": "E4_B4SC_V1_GATE0_REGION_JOINT_INVENTORY.json",
        },
        "next": [
            "review_rank_proxies_and_slowfast_module_shapes",
            "lock_rounded_slow_channel_targets_before_metrics",
            "open_E1_control_then_E2_E3",
        ],
        "design_ladder_reminder": proto["design_ladder_note"],
    }

    create_only(OUT / "E4_B4SC_V1_GATE0_TENSOR_SHAPES.json", {"schema": "ms_sta_v6_e4_b4sc_v1_gate0_tensor_shapes_v1", "created_at": now, "git_head": git_head, **shapes})
    create_only(OUT / "E4_B4SC_V1_GATE0_STAGE_LATENCY.json", latency)
    create_only(OUT / "E4_B4SC_V1_GATE0_ACTIVATION_RANK.json", rank_report)
    create_only(OUT / "E4_B4SC_V1_GATE0_REGION_JOINT_INVENTORY.json", joint_inv)
    create_only(OUT / "E4_B4SC_V1_GATE0_INVENTORY_REPORT.json", report)
    print("GATE0_INVENTORY_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
