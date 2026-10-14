# -*- coding: utf-8 -*-
"""E4_B4SC_V2 S5 transplant-init fidelity (confirm_train only; create-only)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e4_b4sc_train_e2_screen import (  # noqa: E402
    AmpRoiFp32SlowFast,
    build_b4_batch_from_projected,
    build_ms_sta_v6,
    head_logits_eval,
    load_meta,
)
from scripts.e4_b4sc_v2_run_controls import (  # noqa: E402
    REGIONS,
    FrozenSlowFastBlock4,
    RawVideoResolver,
    cosine,
    ensure_train_cuda,
    extract_tokens,
    load_split_role_ids,
    stratified_sample_ids,
)
from src.optimization.e4_b4sc_v2.factory import build_v2_slowfast  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POLICY = OUT / "E4_B4SC_V2_S5_POLICY_LOCK.json"
CHAN = OUT / "E4_B4SC_V2_S5_CHANNEL_LOCK.json"
FID_IDS = OUT / "E4_B4SC_V2_S5_FIDELITY_SET_IDS.json"
FID_REP = OUT / "E4_B4SC_V2_S5_TRANSPLANT_FIDELITY.json"
TERM = OUT / "E4_B4SC_V2_S5_TERMINAL.json"
STATUS = OUT / "E4_B4SC_V2_CAMPAIGN_STATUS.json"


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_V2_S5_FID_FAIL:{msg}")


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def kl_mean(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
    p = F.softmax(teacher_logits, dim=-1)
    log_q = F.log_softmax(student_logits, dim=-1)
    return float(F.kl_div(log_q, p, reduction="batchmean").item())


def main() -> int:
    ensure_train_cuda()
    pol = json.loads(POLICY.read_text(encoding="utf-8"))
    chan = json.loads(CHAN.read_text(encoding="utf-8"))
    th = pol["fidelity_protocol"]["thresholds"]
    fset = pol["fidelity_protocol"]["fidelity_set"]
    if not pol.get("c2_gaussian_perturbation", {}).get("forbid_as_S5_transplant_init", False):
        die("policy_missing_c2_forbid")

    device = torch.device("cuda")
    meta = load_meta()
    labels = json.loads((ROOT / pol["bindings"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels["gloss_to_id"]
    split = json.loads((ROOT / pol["bindings"]["split_manifest_rel"]).read_text(encoding="utf-8"))
    train_ids = load_split_role_ids(split, pol["bindings"].get("train_role", "confirm_train"))
    # controls helper may need role names used by V2 policy
    if not train_ids:
        train_ids = load_split_role_ids(split, "confirm_train")
    val_ids = set(load_split_role_ids(split, "val"))
    test_ids = set(str(x) for x in (split.get("test_ids") or []))

    fid_ids = stratified_sample_ids(
        train_ids, meta, int(fset["n"]), int(fset["seed"]), signer_field="signer_id"
    )
    if (set(fid_ids) & val_ids) or (set(fid_ids) & test_ids):
        die("fidelity_set_leak")
    create_only(
        FID_IDS,
        {
            "schema": "ms_sta_v6_e4_b4sc_v2_s5_fidelity_set_ids_v1",
            "role": "confirm_train",
            "n": len(fid_ids),
            "seed": fset["seed"],
            "sample_ids": fid_ids,
            "sha256": hashlib.sha256(json.dumps(fid_ids, separators=(",", ":")).encode()).hexdigest(),
            "forbid_val_test": True,
        },
    )

    resolver = RawVideoResolver(ROOT)
    skel_root = ROOT / "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3"

    teacher_raw = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["bindings"]["slowfast_rel"],
        expected_sha256=pol["bindings"]["slowfast_sha256"],
        device=device,
        repo_root=ROOT,
    )
    student_raw = build_v2_slowfast(
        teacher_raw,
        "V2_S5",
        device=device,
        requires_grad_slow=False,
        s5_c_mid=tuple(int(x) for x in pol["architecture_lock"]["C_mid_S5"]),
        eval_mode=True,
    )
    if student_raw.plan["stages"]["3"]["inner_idx_sha256"] != chan["stages"]["3"]["inner_idx_sha256"]:
        die("channel_lock_mismatch")

    teacher_amp = AmpRoiFp32SlowFast(teacher_raw, channels_last=False)
    student_amp = AmpRoiFp32SlowFast(student_raw, channels_last=False)

    b4_head = build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt = torch.load(ROOT / pol["bindings"]["b4_teacher_rel"], map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()

    from src.models.rgb_token_v3 import (
        INPUT_SIZE,
        WINDOW_SIZE,
        WINDOW_STRIDE,
        _resize_letterbox,
        window_starts,
        window_tensors,
    )

    widths = [256, 512, 1024, 2048]
    kls: list[float] = []
    agree = 0
    tea_correct = 0
    stu_correct = 0
    stage_cos = [[] for _ in range(4)]
    roi_cos = {r: [] for r in REGIONS}
    temporal_cos: list[float] = []
    fast_abs: list[float] = []
    nan_inf = 0
    n_ok = 0

    for i, sid in enumerate(fid_ids):
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        y = int(g2i[gloss])
        signer = str(row.get("signer_id") or "")
        tea_ex, sealed, frames, _frame_idx, _fps, _a, _b, _c, _d = extract_tokens(
            teacher_amp, sid, row, resolver, skel_root
        )
        stu_ex, _, _, _, _, _, _, _, _ = extract_tokens(student_amp, sid, row, resolver, skel_root)

        for r in REGIONS:
            t = torch.from_numpy(np.asarray(tea_ex["tokens"][r], dtype=np.float32))
            s = torch.from_numpy(np.asarray(stu_ex["tokens"][r], dtype=np.float32))
            L = min(int(t.shape[0]), int(s.shape[0]))
            roi_cos[r].append(cosine(t[:L], s[:L]))

        def logits_from_extract(ex: dict) -> torch.Tensor:
            tokens = {
                r: torch.from_numpy(np.asarray(ex["tokens"][r], dtype=np.float32)).to(device) for r in REGIONS
            }
            batch = build_b4_batch_from_projected(
                sid=sid, label=y, signer=signer, proj_tokens=tokens, sealed=sealed, device=device
            )
            return head_logits_eval(b4_head, batch)

        with torch.inference_mode():
            lt = logits_from_extract(tea_ex)
            ls = logits_from_extract(stu_ex)
            if (not torch.isfinite(lt).all()) or (not torch.isfinite(ls).all()):
                nan_inf += 1
                continue
            kls.append(kl_mean(lt, ls))
            yp_t = int(lt.argmax(-1))
            yp_s = int(ls.argmax(-1))
            agree += int(yp_t == yp_s)
            tea_correct += int(yp_t == y)
            stu_correct += int(yp_s == y)

            resized = []
            for fr in frames:
                canvas, _ = _resize_letterbox(fr, INPUT_SIZE)
                resized.append(canvas)
            frames_u8 = np.stack(resized, axis=0)
            starts = window_starts(frames_u8.shape[0], window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE)

            def stages_for(start: int):
                slow_t, fast_t, _, _ = window_tensors(frames_u8, start, WINDOW_SIZE)
                slow_t = slow_t.to(device).float()
                fast_t = fast_t.to(device).float()
                x: Any = [slow_t, fast_t]
                tea_stages = []
                for bi, block in enumerate(teacher_raw.blocks):
                    x = block(x)
                    if bi >= 1 and isinstance(x, (list, tuple)):
                        tea_stages.append(x[0][:, : widths[bi - 1]].float())
                tea_fast = x[1].float()
                smaps, sstages = student_raw.forward_maps_with_stages(slow_t, fast_t)
                return tea_stages, tea_fast, smaps, [s.float() for s in sstages]

            tea_stages, tea_fast, smaps, sstages = stages_for(starts[0])
            fast_abs.append(float((tea_fast - smaps.fast.float()).abs().max().item()))
            for si in range(4):
                stage_cos[si].append(cosine(sstages[si], tea_stages[si]))
            if len(starts) >= 2:
                tea2, _, _, ss2 = stages_for(starts[1])
                temporal_cos.append(cosine(ss2[-1] - sstages[-1], tea2[-1] - tea_stages[-1]))

        n_ok += 1
        if (i + 1) % 16 == 0:
            print(
                f"FIDELITY {i+1}/{len(fid_ids)} kl={float(np.mean(kls)):.4f} agree={agree/max(n_ok,1):.3f}",
                flush=True,
            )

    coverage = n_ok / max(len(fid_ids), 1)
    agreement = agree / max(n_ok, 1)
    tea_top1 = tea_correct / max(n_ok, 1)
    stu_top1 = stu_correct / max(n_ok, 1)
    retention = (stu_top1 / tea_top1) if tea_top1 > 0 else float("nan")
    stage_means = [float(np.mean(xs)) if xs else float("nan") for xs in stage_cos]
    roi_means = {r: float(np.mean(roi_cos[r])) if roi_cos[r] else float("nan") for r in REGIONS}
    # policy uses left_hand/right_hand; REGIONS uses left_hand/right_hand — map if needed
    roi_policy = {}
    for k, v in th["roi_fidelity_cosine_min"].items():
        key = k
        if key not in roi_means and key.replace("hand", "_hand") in roi_means:
            key = key.replace("hand", "_hand")
        if key not in roi_means and k == "left_hand":
            key = "left_hand"
        if key not in roi_means and k == "right_hand":
            key = "right_hand"
        roi_policy[k] = (roi_means.get(key, roi_means.get(k, float("nan"))), float(v))

    temporal_mean = float(np.mean(temporal_cos)) if temporal_cos else float("nan")
    kl_m = float(np.mean(kls)) if kls else float("nan")
    fast_max = float(np.max(fast_abs)) if fast_abs else float("nan")

    checks = {
        "forbid_nan_inf": nan_inf == 0 and bool(th["forbid_nan_inf"]),
        "sample_coverage_min": coverage >= float(th["sample_coverage_min"]),
        "logits_kl_mean_max": kl_m <= float(th["logits_kl_mean_max"]),
        "teacher_student_pred_agreement_min": agreement >= float(th["teacher_student_pred_agreement_min"]),
        "top1_retention_ratio_vs_teacher_min": retention >= float(th["top1_retention_ratio_vs_teacher_min"]),
        "stage_cosine_floor": all(stage_means[i] >= float(th["stage_cosine_floor"][i]) for i in range(4)),
        "roi_fidelity_cosine_min": all(val >= floor for val, floor in roi_policy.values()),
        "temporal_diff_cosine_min": (not temporal_cos)
        or (temporal_mean >= float(th["temporal_diff_cosine_min"])),
        "fast_pathway_max_abs_delta": fast_max <= float(th["fast_pathway_max_abs_delta"]),
        "fast_pathway_exact_hash_match": fast_max == 0.0,
    }
    passed = all(checks.values())
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "schema": "ms_sta_v6_e4_b4sc_v2_s5_transplant_fidelity_v1",
        "created_at": now,
        "git_head": git_head,
        "phase": "transplant_init",
        "bn_recalibrated": False,
        "c2_gaussian_used": False,
        "e4_ready": False,
        "fidelity_set_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_FIDELITY_SET_IDS.json",
        "channel_lock_rel": "Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_CHANNEL_LOCK.json",
        "n_requested": len(fid_ids),
        "n_ok": n_ok,
        "coverage": coverage,
        "nan_inf_count": nan_inf,
        "metrics": {
            "logits_kl_mean": kl_m,
            "pred_agreement": agreement,
            "teacher_top1": tea_top1,
            "student_top1": stu_top1,
            "top1_retention_ratio_vs_teacher": retention,
            "stage_cosine_mean": stage_means,
            "roi_cosine_mean": roi_means,
            "temporal_diff_cosine_mean": temporal_mean,
            "temporal_n": len(temporal_cos),
            "fast_max_abs_delta": fast_max,
        },
        "thresholds": th,
        "checks": checks,
        "pass": passed,
        "terminal_if_fail": "V2_S5_TRANSPLANT_FIDELITY_NO_GO",
    }
    create_only(FID_REP, report)

    st = json.loads(STATUS.read_text(encoding="utf-8"))
    if not passed:
        create_only(
            TERM,
            {
                "schema": "ms_sta_v6_e4_b4sc_v2_s5_terminal_v1",
                "terminal_state": "V2_S5_TRANSPLANT_FIDELITY_NO_GO",
                "e4_ready": False,
                "S45_ready": False,
                "checks_failed": [k for k, v in checks.items() if not v],
                "created_at": now,
            },
        )
        st["status"] = "V2_S5_TRANSPLANT_FIDELITY_NO_GO"
        st["e4_ready"] = False
        st["updated_at"] = now
        st["history"].append({"status": st["status"], "updated_at": now, "e4_ready": False})
        st["next"] = ["stop_before_bn_and_training"]
        STATUS.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("V2_S5_TRANSPLANT_FIDELITY_NO_GO", flush=True)
        return 2

    st["status"] = "V2_S5_TRANSPLANT_FIDELITY_PASS_AWAITING_BN"
    st["e4_ready"] = False
    st["updated_at"] = now
    st["history"].append({"status": st["status"], "updated_at": now, "e4_ready": False})
    st["next"] = ["run_bn_recalibration_train_only", "evaluate_bn_fidelity_same_set"]
    STATUS.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("E4_B4SC_V2_S5_TRANSPLANT_FIDELITY_PASS", flush=True)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
