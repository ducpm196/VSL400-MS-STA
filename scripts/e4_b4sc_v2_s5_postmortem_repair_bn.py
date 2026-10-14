# -*- coding: utf-8 -*-
"""Repair invalid BN-only section of S5 transplant postmortem."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.e4_b4sc_v2_s5_transplant_postmortem as pm  # noqa: E402


def main() -> int:
    pm.ensure_train_cuda()
    if not pm.POST.is_file():
        pm.die(f"missing:{pm.POST}")

    pol = json.loads(pm.POLICY.read_text(encoding="utf-8"))
    chan = json.loads(pm.CHAN.read_text(encoding="utf-8"))
    fid_doc = json.loads(pm.FID_IDS.read_text(encoding="utf-8"))
    bn_prot = json.loads(pm.BN_PROT.read_text(encoding="utf-8"))
    arch = json.loads(pm.ARCH.read_text(encoding="utf-8"))
    post = json.loads(pm.POST.read_text(encoding="utf-8"))
    th = dict(pol["fidelity_protocol"]["thresholds"])

    device = torch.device("cuda")
    meta = pm.load_meta()
    labels = json.loads((pm.ROOT / pol["bindings"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels["gloss_to_id"]
    split = json.loads((pm.ROOT / pol["bindings"]["split_manifest_rel"]).read_text(encoding="utf-8"))
    train_ids = pm.load_split_role_ids(split, "confirm_train")
    val_ids = set(pm.load_split_role_ids(split, "val"))
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    fid_ids = list(fid_doc["sample_ids"])

    teacher_raw = pm.FrozenSlowFastBlock4(
        checkpoint_path=pm.ROOT / pol["bindings"]["slowfast_rel"],
        expected_sha256=pol["bindings"]["slowfast_sha256"],
        device=device,
        repo_root=pm.ROOT,
    )
    student_raw = pm.build_v2_slowfast(
        teacher_raw,
        "V2_S5",
        device=device,
        requires_grad_slow=False,
        s5_c_mid=tuple(int(x) for x in pol["architecture_lock"]["C_mid_S5"]),
        eval_mode=True,
    )
    if student_raw.plan["stages"]["3"]["inner_idx_sha256"] != chan["stages"]["3"]["inner_idx_sha256"]:
        pm.die("channel_lock_mismatch")

    ckpt_path = pm.ROOT / arch["checkpoint"]["transplant_init_rel"]
    if not ckpt_path.is_file():
        durable = pm.ROOT / str(arch["checkpoint"].get("durable_registry_rel", "")).replace("\\", "/")
        if durable.is_file():
            ckpt_path = durable
        else:
            pm.die("missing_transplant_ckpt")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    student_raw.load_state_dict(payload["model"], strict=True)
    student_raw.eval()

    teacher_amp = pm.AmpRoiFp32SlowFast(teacher_raw, channels_last=False)
    b4_head = pm.build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt = torch.load(pm.ROOT / pol["bindings"]["b4_teacher_rel"], map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()
    resolver = pm.RawVideoResolver(pm.ROOT)

    print("REPAIR BN-only counterfactual", flush=True)
    student_bn = copy.deepcopy(student_raw)
    bn_before = pm.sha_bn_stats(student_bn)
    touched = pm.set_stage5_slow_bn_train_stats_only(student_bn)
    print(f"  touched_bn={len(touched)}", flush=True)
    if not touched:
        pm.die("bn_scope_still_empty")

    bn_n = int(bn_prot["protocol"]["n_samples"])
    bn_seed = int(bn_prot["protocol"]["sample_order_seed"])
    bn_ids = pm.stratified_sample_ids(train_ids, meta, bn_n, bn_seed, signer_field="signer_id")
    if set(bn_ids) & val_ids or set(bn_ids) & test_ids:
        pm.die("bn_calib_leak")

    with torch.no_grad():
        for i, sid in enumerate(bn_ids):
            row = meta[sid]
            _ex, _sealed, frames, *_ = pm.extract_tokens(teacher_amp, sid, row, resolver, pm.SKEL)
            slow_t, fast_t, *_ = pm.load_first_windows(frames, device)
            _ = student_bn.forward_maps_with_stages(slow_t, fast_t)
            if (i + 1) % 64 == 0:
                print(f"  bn_recal {i+1}/{bn_n}", flush=True)
    student_bn.eval()
    bn_after = pm.sha_bn_stats(student_bn)
    student_bn_amp = pm.AmpRoiFp32SlowFast(student_bn, channels_last=False)
    print("REPAIR re-eval fidelity after BN", flush=True)
    bn_fid = pm.eval_fidelity_bundle(
        fid_ids=fid_ids,
        meta=meta,
        g2i=g2i,
        resolver=resolver,
        skel_root=pm.SKEL,
        teacher_raw=teacher_raw,
        student_raw=student_bn,
        teacher_amp=teacher_amp,
        student_amp=student_bn_amp,
        b4_head=b4_head,
        device=device,
        th=th,
        collect_flips=False,
    )

    rem = post["removed_channel_audit"]["removed_stats"]
    low_frac = float(post["removed_channel_audit"]["low_energy_high_temporal_fraction"])
    high_temporal = low_frac >= 0.25 or float(rem["removed_vs_kept_temporal_deriv_ratio"]) >= 1.1
    first_drop = post["block_boundary_localization"]["first_largest_drop"]
    boundary_at_proj = first_drop is not None and first_drop["to"] in ("after_reduce", "after_expand")
    bn_pass = bool(bn_fid["pass"])

    if bn_pass:
        recommended = "OPEN_NEW_POLICY_S5_BN_FIRST"
        rationale = "BN-only counterfactual passes locked fidelity gates on the same train-only set."
    elif high_temporal and not bn_pass:
        recommended = "TEMPORAL_ROI_AWARE_CHANNEL_SELECTION"
        rationale = "Removed channels show elevated temporal sensitivity; BN-only did not clear gates."
    elif boundary_at_proj and not bn_pass:
        recommended = "ACTIVATION_AWARE_LOW_RANK_OR_BLOCKWISE_TEACHER_REGRESSION"
        rationale = "Largest cosine drop occurs at reduce/expand projection boundary."
    else:
        recommended = "STOP_HARD_CHANNEL_PRUNING_STUDY_FACTORIZATION_KEEP_INTERFACE"
        rationale = (
            "BN-only did not clear gates; removed-channel temporal elevation is weak; "
            f"largest drop is at {first_drop['from']}→{first_drop['to']} "
            "(temporal/spatial mid-conv), not reduce/expand projection."
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    post["bn_only_counterfactual"] = {
        "role": "cause_classification_only",
        "repaired_at": now,
        "repair_reason": (
            "Prior BN counterfactual matched 0 modules due to leading-dot name bug; "
            "repaired matcher startswith('blocks.4.multipathway_blocks.0.')."
        ),
        "protocol_rel": str(pm.BN_PROT.relative_to(pm.ROOT)).replace("\\", "/"),
        "authorized_bn_scope": "blocks.4.multipathway_blocks.0.* BatchNorm only",
        "n_calib": bn_n,
        "calib_seed": bn_seed,
        "calib_ids_sha256": hashlib.sha256(json.dumps(bn_ids, separators=(",", ":")).encode()).hexdigest(),
        "bn_modules_updated_n": len(touched),
        "bn_modules_updated_head20": touched[:20],
        "bn_stats_sha_before": bn_before,
        "bn_stats_sha_after": bn_after,
        "bn_stats_changed": bn_before != bn_after,
        "gradients": False,
        "optimizer_step": False,
        "val_used": False,
        "fidelity_after_bn": {
            "metrics": bn_fid["metrics"],
            "checks": bn_fid["checks"],
            "pass": bn_fid["pass"],
        },
        "does_not_override_verdict": True,
    }
    post["direction_rule"] = {
        "bn_only_passes_gates": bn_pass,
        "high_temporal_sensitivity_in_removed": high_temporal,
        "projection_boundary_implicated": boundary_at_proj,
        "largest_drop": first_drop,
        "recommended_next": recommended,
        "rationale": rationale,
        "S45_remains_closed": True,
        "forbid_threshold_relaxation": True,
        "repaired_at": now,
    }
    post["verdict_unchanged"] = True
    post["e4_ready"] = False
    post["training_reopened"] = False
    post["thresholds_changed"] = False
    post["updated_at"] = now
    post["bn_repair_note"] = (
        "Prior BN counterfactual was invalid (0 BN modules touched). "
        "This repair re-ran BN-only on locked transplant copy; audit sections retained. "
        "Direction rule treats reduce/expand only as projection boundary."
    )
    pm.POST.write_text(json.dumps(post, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    st = json.loads(pm.STATUS.read_text(encoding="utf-8"))
    st["status"] = "V2_S5_TRANSPLANT_FIDELITY_NO_GO_POSTMORTEM_COMPLETE"
    st["e4_ready"] = False
    st["updated_at"] = now
    st["s5"] = st.get("s5") or {}
    st["s5"]["postmortem_rel"] = str(pm.POST.relative_to(pm.ROOT)).replace("\\", "/")
    st["s5"]["recommended_next"] = recommended
    st["s5"]["bn_repair_at"] = now
    st["history"] = list(st.get("history") or [])
    st["history"].append(
        {
            "status": st["status"],
            "updated_at": now,
            "e4_ready": False,
            "note": f"BN postmortem repaired; verdict unchanged; next={recommended}",
        }
    )
    st["next"] = [
        "do_not_reopen_failed_S5_run",
        f"consider_new_policy_if_approved:{recommended}",
        "keep_S45_closed",
    ]
    pm.STATUS.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("E4_B4SC_V2_S5_POSTMORTEM_BN_REPAIR_OK", recommended, flush=True)
    print(
        json.dumps(
            {
                "bn_pass": bn_pass,
                "bn_changed": bn_before != bn_after,
                "n_bn": len(touched),
                "recommended": recommended,
                "bn_metrics": bn_fid["metrics"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
