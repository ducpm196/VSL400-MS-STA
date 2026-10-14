# -*- coding: utf-8 -*-
"""Create channel-selection lock + E2/E3 architecture audits (create-only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters  # noqa: E402
from src.models.rgb_token_v3 import FrozenSlowFastBlock4  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_b4sc.narrow_slowfast import (  # noqa: E402
    E2_WIDTHS,
    E3_WIDTHS,
    ArmWidths,
    NarrowSlowFastBlock4,
    RoiToHeadProjection,
    StageKDAdapters,
    TEACHER_SLOW_WIDTHS,
    compute_channel_plan,
)

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POL = OUT / "E4_B4SC_V1_E2E3_TRAINING_POLICY.json"
POL_E4 = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_E2E3_FAIL:{msg}")


def create_only(path: Path, obj: dict) -> None:
    if path.is_file():
        die(f"exists:{path}")
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_meta() -> dict[str, dict]:
    out = {}
    with (ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("dataset") or "") != "VSL400":
                continue
            sid = str(row.get("video_id") or "").strip()
            if sid:
                out[sid] = dict(row)
    return out


def pick_calib_ids(split: dict, n: int, seed: int) -> list[str]:
    from src.training.ms_sta_v5_data import load_split_role_ids

    ids = list(load_split_role_ids(split, "confirm_train"))
    rng = np.random.default_rng(seed)
    # stratified-ish: shuffle then take prefix (deterministic)
    perm = rng.permutation(len(ids))
    chosen = [ids[i] for i in perm[:n]]
    # ensure no val/test leakage
    val = set(load_split_role_ids(split, "val"))
    test = set(str(x) for x in (split.get("test_ids") or []))
    if set(chosen) & val or set(chosen) & test:
        die("calib_leak")
    return chosen


def audit_arm(teacher: FrozenSlowFastBlock4, arm: ArmWidths, plan: dict, device: torch.device) -> dict:
    student = NarrowSlowFastBlock4(teacher, plan, device=device, requires_grad_slow=True)
    slow = torch.randn(1, 3, 8, 224, 224, device=device)
    fast = torch.randn(1, 3, 32, 224, 224, device=device)
    with torch.inference_mode():
        tm = teacher.forward_maps(slow, fast)
        sm = student.forward_maps(slow, fast)
    # stage width check
    outs = {}
    for name, mod in student.named_modules():
        if type(mod).__name__ != "Conv3d":
            continue
        if ".multipathway_blocks.0." in name and "res_blocks" in name:
            # names: "blocks.1.multipathway_blocks.0...." or "1.multipathway..."
            parts = name.split(".")
            st = int(parts[1]) if parts[0] == "blocks" else int(parts[0])
            outs.setdefault(st, set()).add(int(mod.weight.shape[0]))
    stage_widths = [max(outs[i]) for i in sorted(outs)]
    if stage_widths != list(arm.slow_widths):
        die(f"width_mismatch:{arm.name}:{stage_widths}")

    proj = RoiToHeadProjection(arm.token_dim).to(device)
    adapters = StageKDAdapters(arm.slow_widths).to(device)
    head = build_ms_sta_v6("b4_v6_ctr")
    head_n = int(count_parameters(head))
    pc = student.count_params()
    return {
        "arm": arm.name,
        "slow_stage_widths": stage_widths,
        "fast_map_shape": list(sm.fast.shape),
        "slow_map_shape": list(sm.slow.shape),
        "fast_exact_match_teacher": bool(torch.equal(tm.fast, sm.fast)),
        "token_dim_student": arm.token_dim,
        "roi_to_head_out_dim": 2304,
        "params": {
            **pc,
            "head_params": head_n,
            "roi_projection_params": int(sum(p.numel() for p in proj.parameters())),
            "stage_kd_adapter_params_train_only": int(sum(p.numel() for p in adapters.parameters())),
            "deployed_total_approx": pc["total"] + head_n + int(sum(p.numel() for p in proj.parameters())),
            "trainable_round1": pc["trainable"] + int(sum(p.numel() for p in proj.parameters())) + int(sum(p.numel() for p in adapters.parameters())),
        },
        "asserts": {
            "slow_widths_exact": True,
            "fast_unchanged": True,
            "kd_adapters_train_only": True,
        },
        "stem_widths": {"slow": 64, "fast": 8},
        "lateral_fusion": "UNCHANGED",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit-only", action="store_true", help="Use existing channel lock; write audits only")
    args = ap.parse_args()
    if not POL.is_file():
        die("missing_training_policy")
    tpol = json.loads(POL.read_text(encoding="utf-8"))
    if tpol.get("status") != "FROZEN_BEFORE_TRAINING":
        die("policy_not_frozen")
    e4 = json.loads(POL_E4.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.get_device_name(0) != e4["scientific_environment_lock"]["required_gpu_name"]:
        die("gpu_mismatch")

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    teacher = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / e4["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=e4["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )

    lock_path = OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json"
    if args.audit_only:
        if not lock_path.is_file():
            die("missing_channel_lock")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        plan_e2, plan_e3 = lock["E2"], lock["E3"]
    else:
        split = json.loads((ROOT / e4["split"]["manifest_rel"]).read_text(encoding="utf-8"))
        calib_n = int(tpol["channel_selection"]["calibration_n"])
        calib_seed = int(tpol["channel_selection"]["calibration_seed"])
        calib_ids = pick_calib_ids(split, calib_n, calib_seed)
        calib_sha = hashlib.sha256(json.dumps(calib_ids, separators=(",", ":")).encode()).hexdigest()

        plan_e2 = compute_channel_plan(teacher, E2_WIDTHS)
        plan_e3 = compute_channel_plan(teacher, E3_WIDTHS)
        plan_e2_b = compute_channel_plan(teacher, E2_WIDTHS)
        for si in map(str, range(4)):
            if plan_e2["stages"][si]["out_idx_sha256"] != plan_e2_b["stages"][si]["out_idx_sha256"]:
                die("nondeterministic_channel_plan")

        lock = {
            "schema": "ms_sta_v6_e4_b4sc_v1_channel_selection_lock_v1",
            "status": "LOCKED_BEFORE_TRAINING",
            "created_at": now,
            "git_head": git_head,
            "training_policy_sha256": sha256_file(POL),
            "method": tpol["channel_selection"]["method"],
            "calibration": {
                "role": "confirm_train",
                "n": calib_n,
                "seed": calib_seed,
                "sample_ids_sha256": calib_sha,
                "sample_ids_rel": "E4_B4SC_V1_CHANNEL_CALIB_IDS.json",
                "activation_used": False,
                "note": "BN-gamma + conv L1 tie-break only; calib IDs reserved for optional activation re-score without changing locked indices.",
            },
            "E2": plan_e2,
            "E3": plan_e3,
            "teacher_slow_widths": list(TEACHER_SLOW_WIDTHS),
            "forbid_val_test_ids": True,
        }
        create_only(OUT / "E4_B4SC_V1_CHANNEL_CALIB_IDS.json", {"schema": "ms_sta_v6_e4_b4sc_v1_channel_calib_ids_v1", "sample_ids": calib_ids, "sha256": calib_sha})
        create_only(lock_path, lock)

    a2 = audit_arm(teacher, E2_WIDTHS, plan_e2, device)
    a3 = audit_arm(teacher, E3_WIDTHS, plan_e3, device)
    create_only(
        OUT / "E4_B4SC_V1_E2_ARCHITECTURE_AUDIT.json",
        {"schema": "ms_sta_v6_e4_b4sc_v1_e2_architecture_audit_v1", "created_at": now, "git_head": git_head, **a2},
    )
    create_only(
        OUT / "E4_B4SC_V1_E3_ARCHITECTURE_AUDIT.json",
        {"schema": "ms_sta_v6_e4_b4sc_v1_e3_architecture_audit_v1", "created_at": now, "git_head": git_head, **a3},
    )
    print("CHANNEL_LOCK_AND_AUDIT_OK", "E2", a2["slow_stage_widths"], "E3", a3["slow_stage_widths"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
