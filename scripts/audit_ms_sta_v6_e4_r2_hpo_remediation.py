#!/usr/bin/env python3
"""E4-R2 HPO audit remediation: fair HPO teacher comparator (no wave-1 retrain)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]

from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.training.ms_sta_e4_r2_provenance import sha256_file  # noqa: E402
from src.training.ms_sta_v5_data import (  # noqa: E402
    VSL400MSSTAv5Dataset,
    collate_v5,
    load_json,
    load_split_role_ids,
)
from src.training.ms_sta_v5_train import evaluate, load_metadata_by_id  # noqa: E402


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    if path.is_file():
        raise RuntimeError(f"refuse_overwrite:{path}")
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    rem = load_json(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_HPO_REMEDIATION_POLICY.json")
    if rem.get("status") != "FROZEN":
        raise SystemExit("remediation_policy_not_frozen")
    train_pol = load_json(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_TRAIN_POLICY.json")
    rows = load_json(_ROOT / rem["wave1_rows_rel"])["rows"]
    if len(rows) != int(rem["exact_wave1_trials"]):
        raise SystemExit("wave1_trial_count_mismatch")

    # Bind fair teacher
    t_info = rem["fair_hpo_teacher"]
    t_path = _ROOT / t_info["checkpoint_rel"]
    t_sha = sha256_file(t_path)
    if t_sha != t_info["checkpoint_sha256"]:
        raise SystemExit(f"fair_teacher_sha_mismatch:{t_sha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required")

    split = load_json(_ROOT / train_pol["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    tune_ids = load_split_role_ids(split, "tune")
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ds = VSL400MSSTAv5Dataset(
        tune_ids,
        rgb_root=_ROOT / train_pol["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / train_pol["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_v5)

    teacher = build_ms_sta_v6("b4_v6_ctr").to(device)
    ckpt = torch.load(t_path, map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    t_metrics = evaluate(teacher, loader, device, collect_rows=False)
    teacher_tune_top1 = float(t_metrics["top1"])
    # Sanity vs reported
    reported = float(t_info["reported_tune_top1"])
    if abs(teacher_tune_top1 - reported) > 0.005:
        # Allow small float drift but flag large mismatch
        raise SystemExit(
            f"fair_teacher_tune_top1_mismatch:measured={teacher_tune_top1}:reported={reported}"
        )

    # Reuse prior audit latency if present (selection ranking only)
    prior_lat: dict[str, dict[str, float]] = {}
    prior_audit_path = _ROOT / "Docs/results/ms_sta_v6_e4_r2/HPO_AUDIT.json"
    if prior_audit_path.is_file():
        prior = load_json(prior_audit_path)
        for c in prior.get("configs") or []:
            prior_lat[c["config_id"]] = {
                "batch1_median_sec": float(c["batch1_median_sec"]),
                "batch1_p90_sec": float(c["batch1_p90_sec"]),
                "batch1_speedup_vs_b4": float(c.get("batch1_speedup_vs_b4", float("nan"))),
            }

    elig_min = float(rem["eligibility_delta_pp_min"])
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cfg_id = f"{r['variant']}_lr{r['lr']}_dw{r['distill_weight']}_do{r['dropout']}_wd{r['weight_decay']}"
        by_cfg[cfg_id].append(r)

    configs = []
    for cfg_id, rs in sorted(by_cfg.items()):
        seeds = sorted(int(x["seed"]) for x in rs)
        if seeds != [42, 123]:
            raise SystemExit(f"bad_seeds:{cfg_id}:{seeds}")
        mean_top1 = sum(float(x["best_val_top1"]) for x in rs) / len(rs)
        mean_top5 = sum(float(x["best_val_top5"]) for x in rs) / len(rs)
        delta_pp = 100.0 * (mean_top1 - teacher_tune_top1)
        eligible = delta_pp >= elig_min
        lat = prior_lat.get(cfg_id, {})
        configs.append(
            {
                "config_id": cfg_id,
                "variant": rs[0]["variant"],
                "lr": rs[0]["lr"],
                "dropout": rs[0]["dropout"],
                "weight_decay": rs[0]["weight_decay"],
                "distill_weight": rs[0]["distill_weight"],
                "seeds": seeds,
                "mean_tune_top1": mean_top1,
                "mean_tune_top5": mean_top5,
                "delta_top1_pp_vs_fair_hpo_teacher_tune": delta_pp,
                "accuracy_eligible": eligible,
                "batch1_median_sec": lat.get("batch1_median_sec"),
                "batch1_p90_sec": lat.get("batch1_p90_sec"),
                "batch1_speedup_vs_b4_prior_audit": lat.get("batch1_speedup_vs_b4"),
                "trials": [
                    {
                        "seed": x["seed"],
                        "tag": x["tag"],
                        "best_val_top1": x["best_val_top1"],
                        "out_rel": x["out_rel"],
                    }
                    for x in sorted(rs, key=lambda z: int(z["seed"]))
                ],
            }
        )

    eligible_cfgs = [c for c in configs if c["accuracy_eligible"]]
    eligible_cfgs.sort(
        key=lambda c: (
            float("inf") if c["batch1_median_sec"] is None else c["batch1_median_sec"],
            float("inf") if c["batch1_p90_sec"] is None else c["batch1_p90_sec"],
            c["config_id"],
        )
    )
    selected = eligible_cfgs[0] if eligible_cfgs else None
    expected = rem["expected_eligible_config_id"]
    if selected and selected["config_id"] != expected:
        # Not a hard fail if another eligible is faster; record mismatch for audit.
        expected_mismatch = True
    else:
        expected_mismatch = bool(selected is None)

    audit = {
        "schema": "ms_sta_v6_e4_r2_hpo_audit_remediated_v1",
        "canonical": True,
        "campaign": "E4-R2",
        "remediation": True,
        "invalid_prior_comparator": "INVALID_COMPARATOR_CONFIRM_TRAIN_OVERLAPS_TUNE",
        "fair_hpo_teacher_checkpoint_sha256": t_sha,
        "fair_hpo_teacher_tune_top1_measured": teacher_tune_top1,
        "fair_hpo_teacher_tune_top1_reported": reported,
        "fair_hpo_teacher_tune_n": int(t_metrics["n"]),
        "eligibility_delta_pp_min": elig_min,
        "n_trials": len(rows),
        "reused_wave1_artifacts": True,
        "no_wave1_retrain": True,
        "configs": configs,
        "n_accuracy_eligible": len(eligible_cfgs),
        "selected_config_id": None if selected is None else selected["config_id"],
        "expected_eligible_config_id": expected,
        "expected_config_selected": bool(selected and selected["config_id"] == expected),
        "expected_mismatch_flag": expected_mismatch and not (selected and selected["config_id"] == expected),
        "latency_source": "prior_HPO_AUDIT_screening_only",
        "note": "Remediated eligibility only. Confirm still required before accuracy gate. Pilot/audit latency not efficiency-gate evidence.",
    }
    selected_doc = {
        "schema": "ms_sta_v6_e4_r2_hpo_selected_remediated_v1",
        "canonical": True,
        "campaign": "E4-R2",
        "remediation": True,
        "status": "SELECTED" if selected is not None else "NO_ELIGIBLE_AFTER_REMEDIATION",
        "selection_order": rem["selection_order"],
        "selected": selected,
        "fair_hpo_teacher_checkpoint_sha256": t_sha,
        "fair_hpo_teacher_tune_top1_measured": teacher_tune_top1,
    }

    a_sha = atomic_write(_ROOT / rem["remediation_audit_rel"], audit)
    s_sha = atomic_write(_ROOT / rem["remediation_selected_rel"], selected_doc)
    print(
        json.dumps(
            {
                "event": "hpo_audit_remediated",
                "audit_sha256": a_sha,
                "selected_sha256": s_sha,
                "teacher_tune_top1": teacher_tune_top1,
                "n_eligible": len(eligible_cfgs),
                "selected": None if selected is None else selected["config_id"],
                "delta_pp": None if selected is None else selected["delta_top1_pp_vs_fair_hpo_teacher_tune"],
                "eligible_ids": [c["config_id"] for c in eligible_cfgs],
            },
            indent=2,
        )
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
