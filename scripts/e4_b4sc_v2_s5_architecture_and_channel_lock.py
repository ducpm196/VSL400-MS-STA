
# -*- coding: utf-8 -*-
"""E4_B4SC_V2 S5 architecture audit + channel lock (create-only; train-only selection)."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e4_b4sc_train_e2_screen import FrozenSlowFastBlock4  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_b4sc.narrow_slowfast import StageKDAdapters  # noqa: E402
from src.optimization.e4_b4sc_v2.factory import build_v2_slowfast, s5_mid_plan, tensor_sha256  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POLICY = OUT / "E4_B4SC_V2_S5_POLICY_LOCK.json"
ARCH = OUT / "E4_B4SC_V2_S5_ARCHITECTURE_AUDIT.json"
CHAN = OUT / "E4_B4SC_V2_S5_CHANNEL_LOCK.json"
CKPT = OUT / "checkpoints_v2_s5" / "S5_transplant_init.pt"


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_V2_S5_ARCH_FAIL:{msg}")


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_params(m: nn.Module) -> dict[str, int]:
    total = sum(int(p.numel()) for p in m.parameters())
    trainable = sum(int(p.numel()) for p in m.parameters() if p.requires_grad)
    slow = sum(int(p.numel()) for n, p in m.named_parameters() if ".multipathway_blocks.0." in n)
    fast = sum(int(p.numel()) for n, p in m.named_parameters() if ".multipathway_blocks.1." in n)
    return {"total": total, "trainable": trainable, "slow_pathway": slow, "fast_pathway": fast, "other": total - slow - fast}


def stage5_mid_shapes(module: nn.Module) -> list[dict[str, Any]]:
    stage = module.blocks[4].multipathway_blocks[0]
    rows = []
    for i, rb in enumerate(stage.res_blocks):
        rows.append({
            "res_block": i,
            "norm_b_features": int(rb.branch2.norm_b.num_features),
            "norm_c_features": int(rb.branch2.norm_c.num_features),
            "conv_a_shape": list(rb.branch2.conv_a.weight.shape),
            "conv_b_shape": list(rb.branch2.conv_b.weight.shape),
            "conv_c_shape": list(rb.branch2.conv_c.weight.shape),
        })
    return rows


def theoretical_stage5_branch2_macs(c_in: int, c_mid: int, c_out: int, t: int, h: int, w: int, n_blocks: int) -> int:
    per = c_in * c_mid * t * h * w + c_mid * c_mid * 3 * t * h * w + c_mid * c_out * t * h * w
    return int(per * n_blocks)


def main() -> int:
    if not POLICY.is_file():
        die("missing_s5_policy_lock")
    pol = json.loads(POLICY.read_text(encoding="utf-8"))
    if pol.get("status") != "FROZEN_BEFORE_FIRST_S5_METRIC":
        die(f"policy_status:{pol.get('status')}")
    if not pol["bindings"].get("hash_bindings_ok"):
        die("policy_hash_bindings_not_ok")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("cuda_required")

    teacher = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["bindings"]["slowfast_rel"],
        expected_sha256=pol["bindings"]["slowfast_sha256"],
        device=device,
        repo_root=ROOT,
    )
    c_mid = tuple(int(x) for x in pol["architecture_lock"]["C_mid_S5"])
    plan = s5_mid_plan(teacher, c_mid, allow_prefix_stub=False)

    for si in range(3):
        st = plan["stages"][str(si)]
        if int(st["student_inner"]) != int(st["teacher_inner"]) or bool(st["compress_mid_only"]):
            die(f"early_stage_compressed:{si}")
    st5 = plan["stages"]["3"]
    if int(st5["student_inner"]) != 448 or int(st5["teacher_inner"]) != 512:
        die("stage5_mid_not_448")
    if int(st5["student_out"]) != 2048 or int(st5["teacher_out"]) != 2048:
        die("stage5_out_changed")
    if st5["inner_selection_method"] != "bn_gamma_l1_then_conv_l1_tiebreak":
        die(f"channel_method:{st5['inner_selection_method']}")

    student = build_v2_slowfast(
        teacher, "V2_S5", device=device, requires_grad_slow=False, s5_c_mid=c_mid, eval_mode=True
    )
    if not hasattr(student, "plan"):
        die("student_missing_plan")
    if student.plan["stages"]["3"]["inner_idx_sha256"] != st5["inner_idx_sha256"]:
        die("nondeterministic_channel_selection")

    adapters = StageKDAdapters(tuple(plan["slow_widths"]), tuple(plan["slow_widths"])).to(device)

    slow = torch.randn(1, 3, 8, 224, 224, device=device)
    fast = torch.randn(1, 3, 32, 224, 224, device=device)
    with torch.inference_mode():
        tmaps = teacher.forward_maps(slow, fast)
        smaps, _ = student.forward_maps_with_stages(slow, fast)
    if tuple(smaps.slow.shape) != tuple(tmaps.slow.shape):
        die(f"slow_shape:{tuple(smaps.slow.shape)}")
    if tuple(smaps.fast.shape) != tuple(tmaps.fast.shape):
        die(f"fast_shape:{tuple(smaps.fast.shape)}")
    fast_delta = float((smaps.fast.float() - tmaps.fast.float()).abs().max().item())
    if fast_delta != 0.0:
        die(f"fast_changed:{fast_delta}")
    if int(smaps.slow.shape[1]) + int(smaps.fast.shape[1]) != int(pol["architecture_lock"]["roi_input_dim"]):
        die("roi_dim")

    before = stage5_mid_shapes(teacher)
    after = stage5_mid_shapes(student)
    for a in after:
        if a["norm_b_features"] != 448 or a["norm_c_features"] != 2048:
            die(f"mid_out_shape:{a}")

    t_fast = {n: tensor_sha256(p.data) for n, p in teacher.named_parameters() if ".multipathway_blocks.1." in n}
    s_fast = {n: tensor_sha256(p.data) for n, p in student.named_parameters() if ".multipathway_blocks.1." in n}
    if t_fast != s_fast:
        die("fast_param_hash_mismatch")

    t_counts = count_params(teacher)
    s_counts = count_params(student)
    adapter_params = int(sum(p.numel() for p in adapters.parameters()))

    payload = {
        "schema": "ms_sta_v6_e4_b4sc_v2_s5_transplant_init_v1",
        "arm": "V2_S5",
        "plan_stage5_inner_idx_sha256": st5["inner_idx_sha256"],
        "model": student.state_dict(),
        "e4_ready": False,
        "c2_gaussian_used": False,
    }
    if CKPT.exists():
        die(f"exists:{CKPT}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CKPT)
    ckpt_sha = sha256_file(CKPT)

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    channel_lock = {
        "schema": "ms_sta_v6_e4_b4sc_v2_s5_channel_lock_v1",
        "status": "LOCKED_BEFORE_S5_FIDELITY_METRICS",
        "created_at": now,
        "git_head": git_head,
        "policy_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_POLICY_LOCK.json",
        "method": "bn_gamma_l1_then_conv_l1_tiebreak",
        "calibration": {
            "role": "confirm_train",
            "n": pol["channel_selection"]["calibration_n"],
            "seed": pol["channel_selection"]["calibration_seed"],
            "activation_used": False,
            "note": "BN-gamma + conv L1 tie-break only; no val/test metrics.",
        },
        "C_mid": plan["c_mid"],
        "C_out": plan["slow_widths"],
        "stages": plan["stages"],
        "forbid_val_test": True,
        "c2_gaussian_used": False,
        "e4_ready": False,
    }
    create_only(CHAN, channel_lock)

    arch = {
        "schema": "ms_sta_v6_e4_b4sc_v2_s5_architecture_audit_v1",
        "created_at": now,
        "git_head": git_head,
        "policy_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_POLICY_LOCK.json",
        "channel_lock_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_CHANNEL_LOCK.json",
        "arm": "V2_S5",
        "affected_blocks": pol["architecture_lock"]["stage5_only"]["affected_exact_modules"],
        "tensor_shapes": {"teacher_stage5": before, "student_stage5": after},
        "interfaces": {
            "residual_out_channels": 2048,
            "lateral_fusion_unchanged": True,
            "roi_input_dim": 2304,
            "roi_formula": "2048_Slow+256_Fast",
            "fast_max_abs_delta": fast_delta,
            "fast_param_hashes_equal": True,
            "head_interface_unchanged": True,
            "deploy_excludes_kd_adapters": True,
        },
        "parameter_counts": {
            "teacher_slowfast": t_counts,
            "student_slowfast": s_counts,
            "delta_total": int(s_counts["total"] - t_counts["total"]),
            "stage_kd_adapters_train_only": adapter_params,
        },
        "macs_stage5_branch2_theoretical": {
            "note": "Approx Conv3d MACs for stage5 branch2 only at feature map 8x7x7; not a full-network claim.",
            "teacher": theoretical_stage5_branch2_macs(2048, 512, 2048, 8, 7, 7, 3),
            "student": theoretical_stage5_branch2_macs(2048, 448, 2048, 8, 7, 7, 3),
        },
        "checkpoint": {
            "transplant_init_rel": CKPT.relative_to(ROOT).as_posix(),
            "sha256": ckpt_sha,
            "bytes": int(CKPT.stat().st_size),
        },
        "assertions_pass": True,
        "e4_ready": False,
        "c2_gaussian_used": False,
    }
    create_only(ARCH, arch)
    print("E4_B4SC_V2_S5_ARCH_OK", flush=True)
    print(json.dumps({"delta_params": arch["parameter_counts"]["delta_total"], "ckpt_sha16": ckpt_sha[:16], "mid": after[0]["norm_b_features"], "out": after[0]["norm_c_features"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
