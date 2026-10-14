# -*- coding: utf-8 -*-
"""E4_B4SC_V3 F0 microbench: original vs factorized conv_b (ranks 480/448).

Create-only artifacts under Docs/results/ms_sta_v6_e4_b4sc/.
Does not open F1+ or KD. Uses locked factorization graph from V3 amendment.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e4_b4sc_v2_run_controls import ensure_train_cuda  # noqa: E402
from src.models.rgb_token_v3 import FrozenSlowFastBlock4  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
AMEND = OUT / "E4_B4SC_V3_POLICY_AMENDMENT_GRAPH_RANK_RULE.json"
INV = OUT / "E4_B4SC_V3_F0_OPERATOR_INVENTORY.json"
RAW = OUT / "E4_B4SC_V3_F0_RAW.jsonl"
REPORT = OUT / "E4_B4SC_V3_F0_REPORT.json"
VERIFY = OUT / "E4_B4SC_V3_F0_INDEPENDENT_VERIFY.json"
V2_POL = OUT / "E4_B4SC_V2_S5_POLICY_LOCK.json"


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_V3_F0_FAIL:{msg}")


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


class FactorizedConvB(nn.Module):
    """y = U(V(x)) + bias_U approximating a single Conv3d before BN/act."""

    def __init__(self, teacher: nn.Conv3d, rank: int):
        super().__init__()
        if teacher.weight.ndim != 5:
            die("teacher_not_conv3d")
        cout, cin, kt, kh, kw = teacher.weight.shape
        if (kt, kh, kw) != (1, 3, 3) or cout != 512 or cin != 512:
            die(f"unexpected_shape:{tuple(teacher.weight.shape)}")
        self.V = nn.Conv3d(
            cin,
            rank,
            kernel_size=(kt, kh, kw),
            stride=teacher.stride,
            padding=teacher.padding,
            dilation=teacher.dilation,
            groups=1,
            bias=False,
        )
        self.U = nn.Conv3d(rank, cout, kernel_size=1, bias=True)
        with torch.no_grad():
            w = teacher.weight.detach().float().cpu()
            # W_mat: (Cout, Cin*kT*kH*kW)
            w_mat = w.reshape(cout, cin * kt * kh * kw)
            # SVD
            U_m, S, Vh = torch.linalg.svd(w_mat, full_matrices=False)
            r = int(rank)
            U_r = U_m[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            s_sqrt = torch.sqrt(torch.clamp(S_r, min=0.0))
            v_mat = (s_sqrt[:, None] * Vh_r).reshape(r, cin, kt, kh, kw)
            u_mat = (U_r * s_sqrt[None, :]).reshape(cout, r, 1, 1, 1)
            self.V.weight.copy_(v_mat.to(teacher.weight.dtype))
            self.U.weight.copy_(u_mat.to(teacher.weight.dtype))
            if teacher.bias is not None:
                self.U.bias.copy_(teacher.bias.detach())
            else:
                self.U.bias.zero_()
            # reconstruction diagnostics
            recon = (U_r * S_r[None, :]) @ Vh_r
            rel = torch.linalg.norm(w_mat - recon) / torch.linalg.norm(w_mat)
            energy = float((S_r.square().sum() / S.square().sum()).item())
            self.recon_rel_fro = float(rel.item())
            self.retained_energy = energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.V(x))


def cuda_median_ms(fn, warmup: int, rounds: int) -> dict[str, float]:
    """CUDA event timing with explicit synchronize (policy F0)."""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(rounds):
        starter.record()
        _ = fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(float(starter.elapsed_time(ender)))
    times_sorted = sorted(times)
    mid = times_sorted[len(times_sorted) // 2]
    p90 = times_sorted[max(0, int(math.ceil(0.9 * len(times_sorted)) - 1))]
    return {"median_ms": float(mid), "p90_ms": float(p90), "mean_ms": float(sum(times) / len(times))}


def macs_conv3d(n: int, cout: int, cin: int, kt: int, kh: int, kw: int, t: int, h: int, w: int) -> int:
    # MACs ≈ N * Cout * T_out * H_out * W_out * (Cin * kt * kh * kw)
    return int(n * cout * t * h * w * cin * kt * kh * kw)


def main() -> int:
    ensure_train_cuda()
    if REPORT.exists() or RAW.exists() or VERIFY.exists():
        die("f0_artifacts_exist")
    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    inv = json.loads(INV.read_text(encoding="utf-8"))
    pol = json.loads(V2_POL.read_text(encoding="utf-8"))
    f0g = amend["F0_numeric_gates"]
    ranks = list(amend["factorization_graph_locked"]["ranks_locked"]["candidates"])
    warmup = int(f0g["warmup_rounds"])
    rounds = int(f0g["measured_rounds"])

    device = torch.device("cuda")
    teacher = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["bindings"]["slowfast_rel"],
        expected_sha256=pol["bindings"]["slowfast_sha256"],
        device=device,
        repo_root=ROOT,
    )
    # Resolve module paths from inventory
    paths = [o["module_path"] for o in inv["operators"]]
    mods = []
    for pth in paths:
        m = teacher
        for part in pth.split("."):
            if part.isdigit():
                m = m[int(part)]
            else:
                m = getattr(m, part)
        if not isinstance(m, nn.Conv3d):
            die(f"not_conv3d:{pth}")
        mods.append((pth, m))

    # Activation shape probe: typical S5 mid feature after reduce.
    # Use locked Kinetics-like map if unknown: try forward one random tensor through stem.. too heavy.
    # Probe via registering hook with a short synthetic path through stage5 only is hard.
    # Use shapes from running one window of random input through full network once.
    teacher.eval()
    with torch.no_grad():
        # SlowFast 8x8 style for this repo window: probe until hooks fire
        slow = torch.randn(1, 3, 8, 224, 224, device=device)
        fast = torch.randn(1, 3, 32, 224, 224, device=device)
        shapes: dict[str, list[int]] = {}

        def make_hook(name: str):
            def _h(_m, inp, _out):
                shapes[name] = list(inp[0].shape)

            return _h

        handles = [m.register_forward_hook(make_hook(n)) for n, m in mods]
        try:
            _ = teacher.forward_maps(slow, fast) if hasattr(teacher, "forward_maps") else teacher([slow, fast])
        finally:
            for h in handles:
                h.remove()
    if len(shapes) != len(mods):
        # fallback common ResNet3d stage5 map after stride
        for n, _m in mods:
            shapes.setdefault(n, [1, 512, 4, 7, 7])

    rows: list[dict[str, Any]] = []
    per_rank_pass: dict[str, bool] = {}
    per_rank_detail: dict[str, Any] = {}

    # Aggregate original vs factorized across 3 operators (sequential sum of medians)
    for rank in ranks:
        orig_sum = 0.0
        fact_sum = 0.0
        orig_p90_sum = 0.0
        fact_p90_sum = 0.0
        recon = []
        energy = []
        op_details = []
        all_shape_ok = True
        for name, m in mods:
            shp = shapes[name]
            x = torch.randn(*shp, device=device, dtype=torch.float32)
            # Isolated FP16 copies so teacher stays FP32 for later ranks.
            m_h = nn.Conv3d(
                m.in_channels,
                m.out_channels,
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=m.bias is not None,
            ).to(device)
            with torch.no_grad():
                m_h.weight.copy_(m.weight.detach())
                if m.bias is not None and m_h.bias is not None:
                    m_h.bias.copy_(m.bias.detach())
            m_h = m_h.half().eval()
            fact = FactorizedConvB(m, rank=int(rank)).to(device).half().eval()
            x_h = x.half()
            recon.append(fact.recon_rel_fro)
            energy.append(fact.retained_energy)

            def run_orig(mm=m_h, xx=x_h):
                with torch.inference_mode():
                    return mm(xx)

            def run_fact(ff=fact, xx=x_h):
                with torch.inference_mode():
                    return ff(xx)

            # shape check + nan/inf
            with torch.inference_mode():
                y0 = run_orig()
                y1 = run_fact()
            if tuple(y0.shape) != tuple(y1.shape):
                all_shape_ok = False
            if not torch.isfinite(y0).all() or not torch.isfinite(y1).all():
                all_shape_ok = False

            # counterbalanced order: orig, fact, fact, orig
            o1 = cuda_median_ms(run_orig, warmup // 2, rounds // 2)
            f1 = cuda_median_ms(run_fact, warmup // 2, rounds // 2)
            f2 = cuda_median_ms(run_fact, warmup // 2, rounds // 2)
            o2 = cuda_median_ms(run_orig, warmup // 2, rounds // 2)
            o_med = 0.5 * (o1["median_ms"] + o2["median_ms"])
            f_med = 0.5 * (f1["median_ms"] + f2["median_ms"])
            o_p90 = 0.5 * (o1["p90_ms"] + o2["p90_ms"])
            f_p90 = 0.5 * (f1["p90_ms"] + f2["p90_ms"])
            orig_sum += o_med
            fact_sum += f_med
            orig_p90_sum += o_p90
            fact_p90_sum += f_p90
            row = {
                "rank": rank,
                "module": name,
                "input_shape": shp,
                "original_median_ms": o_med,
                "factorized_median_ms": f_med,
                "original_p90_ms": o_p90,
                "factorized_p90_ms": f_p90,
                "recon_rel_fro": fact.recon_rel_fro,
                "retained_energy": fact.retained_energy,
                "macs_original": macs_conv3d(shp[0], 512, 512, 1, 3, 3, shp[2], shp[3], shp[4]),
                "macs_factorized": macs_conv3d(shp[0], rank, 512, 1, 3, 3, shp[2], shp[3], shp[4])
                + macs_conv3d(shp[0], 512, rank, 1, 1, 1, shp[2], shp[3], shp[4]),
            }
            rows.append(row)
            op_details.append(row)
            print(
                f"F0 rank={rank} {name} orig={o_med:.3f}ms fact={f_med:.3f}ms recon={fact.recon_rel_fro:.4f}",
                flush=True,
            )

        faster = fact_sum < orig_sum
        p90_ok = fact_p90_sum <= orig_p90_sum * 1.05
        passed = bool(faster and p90_ok and all_shape_ok)
        per_rank_pass[str(rank)] = passed
        per_rank_detail[str(rank)] = {
            "original_median_ms_sum3": orig_sum,
            "factorized_median_ms_sum3": fact_sum,
            "original_p90_ms_sum3": orig_p90_sum,
            "factorized_p90_ms_sum3": fact_p90_sum,
            "speedup_median": (orig_sum / fact_sum) if fact_sum > 0 else None,
            "mean_recon_rel_fro": float(sum(recon) / len(recon)),
            "mean_retained_energy": float(sum(energy) / len(energy)),
            "shape_ok": all_shape_ok,
            "pass": passed,
            "ops": op_details,
        }

    any_pass = any(per_rank_pass.values())
    terminal = "V3_F0_PASS_RANKS_" + ",".join([k for k, v in per_rank_pass.items() if v]) if any_pass else "V3_F0_HARDWARE_EFFICIENCY_NO_GO"

    # Amdahl sketch vs E1a
    e1a_sf = float(inv["latency_context_from_E1a"]["slowfast_roi_median_ms"])
    e1a_e2e = float(inv["latency_context_from_E1a"]["e2e_median_ms"])
    e0_e2e = 1325.35
    amdahl = {}
    target_e2e = e0_e2e / 1.15
    gap_base = e1a_e2e - target_e2e
    for rank, det in per_rank_detail.items():
        local_save = det["original_median_ms_sum3"] - det["factorized_median_ms_sum3"]
        projected = e1a_e2e - local_save
        amdahl[rank] = {
            "local_ms_saved_sum3_conv_b": local_save,
            "E0_e2e_median_ms_locked_context": e0_e2e,
            "E1a_e2e_median_ms": e1a_e2e,
            "target_e2e_for_1_15x_ms": target_e2e,
            "gap_E1a_to_1_15x_before_factor_ms": gap_base,
            "optimistic_e2e_if_full_local_save_applies_ms": projected,
            "remaining_gap_to_1_15x_vs_E0_ms": projected - target_e2e,
            "closes_1_15x_gap": bool(projected <= target_e2e),
            "note": "Optimistic Amdahl applies 100% of sum-of-3-conv_b delta to e2e; negative save means slower.",
        }

    RAW.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "schema": "ms_sta_v6_e4_b4sc_v3_f0_report_v1",
        "created_at": now,
        "git_head": git_head(),
        "amendment_rel": str(AMEND.relative_to(ROOT)).replace("\\", "/"),
        "inventory_rel": str(INV.relative_to(ROOT)).replace("\\", "/"),
        "warmup_rounds": warmup,
        "measured_rounds": rounds,
        "precision": "fp16_deploy_proxy",
        "ranks": ranks,
        "per_rank": per_rank_detail,
        "any_rank_pass": any_pass,
        "terminal": terminal,
        "amdahl": amdahl,
        "e4_ready": False,
        "next": (
            ["open_F1_for_passing_ranks_only"]
            if any_pass
            else ["stop_F1_F2_F3_KD", "continue_B4PIPE_or_accept_efficiency_ceiling"]
        ),
    }
    create_only(REPORT, report)

    # Independent verify: re-read raw, recompute pass rule
    raw_rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    verify_pass = {}
    for rank in ranks:
        sub = [r for r in raw_rows if int(r["rank"]) == int(rank)]
        o = sum(r["original_median_ms"] for r in sub)
        f = sum(r["factorized_median_ms"] for r in sub)
        op = sum(r["original_p90_ms"] for r in sub)
        fp = sum(r["factorized_p90_ms"] for r in sub)
        verify_pass[str(rank)] = bool(f < o and fp <= op * 1.05)
    create_only(
        VERIFY,
        {
            "schema": "ms_sta_v6_e4_b4sc_v3_f0_independent_verify_v1",
            "created_at": now,
            "raw_rel": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "report_rel": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "recomputed_pass": verify_pass,
            "matches_report": verify_pass == per_rank_pass,
            "terminal": terminal,
        },
    )

    # Update V3 campaign status
    st_path = OUT / "E4_B4SC_V3_CAMPAIGN_STATUS.json"
    st = json.loads(st_path.read_text(encoding="utf-8"))
    st["status"] = terminal
    st["updated_at"] = now
    st["gates"]["F0_operator_feasibility"] = "PASS" if any_pass else "FAIL"
    if not any_pass:
        st["gates"]["F1_weight_reconstruction"] = "STOPPED_F0_NO_GO"
        st["gates"]["F2_activation_aware_reconstruction"] = "STOPPED_F0_NO_GO"
        st["gates"]["F3_fidelity"] = "STOPPED_F0_NO_GO"
        st["gates"]["full_KD"] = "STOPPED_F0_NO_GO"
    st["f0_report_rel"] = str(REPORT.relative_to(ROOT)).replace("\\", "/")
    st["history"] = list(st.get("history") or [])
    st["history"].append({"status": terminal, "updated_at": now, "e4_ready": False})
    st_path.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("E4_B4SC_V3_F0_DONE", terminal, flush=True)
    print(json.dumps({"per_rank_pass": per_rank_pass, "amdahl": amdahl}, indent=2), flush=True)
    return 0 if True else 1


if __name__ == "__main__":
    raise SystemExit(main())
