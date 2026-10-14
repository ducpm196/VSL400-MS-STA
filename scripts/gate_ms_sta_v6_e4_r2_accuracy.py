#!/usr/bin/env python3
"""E4-R2 confirm close + frozen accuracy gate vs B4-V6 (create-only)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]

from src.eval.ms_sta_v5_m0_strong_baseline import (  # noqa: E402
    PredRow,
    hierarchical_bootstrap_ci,
    load_json,
    per_signer_delta_top1_pp,
)
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.training.ms_sta_e4_r2_provenance import sha256_file  # noqa: E402
from src.training.ms_sta_e4_r2_status import (  # noqa: E402
    build_campaign_status,
    load_json as load_status_json,
    write_campaign_status,
)
from src.training.ms_sta_v5_data import (  # noqa: E402
    VSL400MSSTAv5Dataset,
    collate_v5,
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


def load_pred_rows(path: Path) -> dict[str, PredRow]:
    obj = load_json(path)
    out: dict[str, PredRow] = {}
    for r in obj["rows"]:
        sid = str(r["sample_id"])
        out[sid] = PredRow(
            sample_id=sid,
            signer=str(r.get("signer") or ""),
            label=int(r["label"]),
            correct=bool(r["correct"]),
            top5_hit=bool(r["top5_hit"]),
        )
    return out


def export_b4_val_preds(seed: int, out_path: Path, train_pol: dict, device: torch.device) -> dict[str, PredRow]:
    """Evaluate B4 confirm checkpoint on val; write create-only if missing."""
    if out_path.is_file():
        return load_pred_rows(out_path)
    ckpt_path = _ROOT / f"Docs/results/ms_sta_v6_runs/confirm/b4_v6_ctr_seed{seed}/best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(f"missing_b4_confirm_ckpt:{ckpt_path}")
    split = load_json(_ROOT / train_pol["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    val_ids = load_split_role_ids(split, "val")
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ds = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / train_pol["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / train_pol["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_v5)
    model = build_ms_sta_v6("b4_v6_ctr").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, loader, device, collect_rows=True)
    payload = {
        "schema": "ms_sta_v5_m0_val_preds_v1",
        "variant": "b4_v6_ctr",
        "phase": "confirm_export_for_e4_r2_gate",
        "seed": seed,
        "n": int(metrics["n"]),
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "macro_f1": metrics["macro_f1"],
        "checkpoint_sha256": sha256_file(ckpt_path),
        "rows": metrics["rows"],
    }
    atomic_write(out_path, payload)
    return load_pred_rows(out_path)


def recompute_from_rows(preds: dict[str, PredRow]) -> dict[str, float]:
    n = max(len(preds), 1)
    top1 = sum(1 for p in preds.values() if p.correct) / n
    top5 = sum(1 for p in preds.values() if p.top5_hit) / n
    # Macro-F1 requires per-class; use SUMMARY only as secondary — here approximate via labels
    return {"top1": top1, "top5": top5, "n": float(len(preds))}


def main() -> int:
    sci = load_json(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_POLICY.json")
    train_pol = load_json(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_TRAIN_POLICY.json")
    rem = load_json(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_HPO_REMEDIATION_POLICY.json")
    selected = load_json(_ROOT / rem["remediation_selected_rel"])["selected"]
    confirm_rows = load_json(_ROOT / rem["runs_root_rel"] / "confirm" / "CONFIRM_ROWS.json")["rows"]
    gates = sci["accuracy_gate"]
    seeds = [42, 123, 456]
    if len(confirm_rows) != 3:
        raise SystemExit("expected_3_confirm_rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required_for_b4_export")

    out_root = _ROOT / "Docs/results/ms_sta_v6_e4_r2"
    b4_export_root = out_root / "b4_confirm_val_preds_export"
    b4_export_root.mkdir(parents=True, exist_ok=True)

    e_panels: dict[int, dict[str, PredRow]] = {}
    t_panels: dict[int, dict[str, PredRow]] = {}
    e_top1, t_top1, e_top5, t_top5, e_f1, t_f1 = [], [], [], [], [], []
    bindings_ok = True
    seed_reports = []

    for row in confirm_rows:
        seed = int(row["seed"])
        e_dir = _ROOT / row["out_rel"]
        e_preds_path = e_dir / "VAL_PREDS.json"
        e_sum = load_json(e_dir / "SUMMARY.json")
        if int(e_sum.get("epochs_run") or 0) != 16:
            raise SystemExit(f"epochs_not_16:{seed}:{e_sum.get('epochs_run')}")
        if not e_sum.get("canonical") or e_sum.get("git_dirty") is not False:
            raise SystemExit(f"noncanonical_confirm:{seed}")
        if e_sum.get("git_revision") != "56142e40f0d14fecc559881b8f3888610ff82f98":
            raise SystemExit(f"git_mismatch:{seed}")
        if e_sum.get("executable_tree_sha256") != "b4f6a01f539a1806c7394d308b107648d671e09566397d9cbea6fc0db3c3c70d":
            raise SystemExit(f"tree_mismatch:{seed}")
        e_preds = load_pred_rows(e_preds_path)
        recomputed = recompute_from_rows(e_preds)
        if abs(recomputed["top1"] - float(e_sum["best_val_top1"])) > 1e-9:
            # best_val_top1 is from best checkpoint; VAL_PREDS is final eval of best — should match closely
            if abs(recomputed["top1"] - float(e_sum["best_val_top1"])) > 1e-6:
                raise SystemExit(f"top1_recompute_mismatch:{seed}:{recomputed['top1']}:{e_sum['best_val_top1']}")
        t_path = b4_export_root / f"b4_v6_ctr_seed{seed}_VAL_PREDS.json"
        t_preds = export_b4_val_preds(seed, t_path, train_pol, device)
        if set(e_preds) != set(t_preds):
            raise SystemExit(f"id_mismatch_seed_{seed}")
        if len(e_preds) != 2629:
            raise SystemExit(f"unexpected_n_val:{seed}:{len(e_preds)}")

        e_panels[seed] = e_preds
        t_panels[seed] = t_preds
        e_top1.append(recomputed["top1"])
        t_top1.append(recompute_from_rows(t_preds)["top1"])
        e_top5.append(recomputed["top5"])
        t_top5.append(recompute_from_rows(t_preds)["top5"])
        e_f1.append(float(e_sum["best_val_macro_f1"]))
        t_f1.append(float(load_json(t_path)["macro_f1"]))
        seed_reports.append(
            {
                "seed": seed,
                "e4_out_rel": row["out_rel"],
                "e4_checkpoint_sha256": row["checkpoint_sha256"],
                "e4_top1_recomputed": e_top1[-1],
                "e4_top5_recomputed": e_top5[-1],
                "e4_macro_f1_summary": e_f1[-1],
                "b4_top1": t_top1[-1],
                "b4_top5": t_top5[-1],
                "b4_macro_f1": t_f1[-1],
                "b4_preds_rel": str(t_path.relative_to(_ROOT)).replace("\\", "/"),
                "b4_checkpoint_sha256": sha256_file(
                    _ROOT / f"Docs/results/ms_sta_v6_runs/confirm/b4_v6_ctr_seed{seed}/best.pt"
                ),
                "n_val": len(e_preds),
                "epochs_run": 16,
            }
        )

    ids = sorted(e_panels[seeds[0]].keys())
    d1, signers = [], []
    for sid in ids:
        vals = [float(e_panels[s][sid].correct) - float(t_panels[s][sid].correct) for s in seeds]
        d1.append(sum(vals) / len(vals))
        signers.append(e_panels[seeds[0]][sid].signer)
    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(
        d1, signers, n_boot=int(gates["n_boot"]), seed=int(gates["boot_seed"])
    )
    signer_deltas = per_signer_delta_top1_pp(
        ids, {sid: e_panels[seeds[0]][sid].signer for sid in ids}, e_panels, t_panels
    )
    mean_d1 = 100.0 * (sum(e_top1[i] - t_top1[i] for i in range(3)) / 3.0)
    mean_d5 = 100.0 * (sum(e_top5[i] - t_top5[i] for i in range(3)) / 3.0)
    mean_df1 = 100.0 * (sum(e_f1[i] - t_f1[i] for i in range(3)) / 3.0)
    e_mean = sum(e_top1) / 3.0
    t_mean = sum(t_top1) / 3.0
    e_std = math.sqrt(sum((x - e_mean) ** 2 for x in e_top1) / 2.0)  # sample std n-1

    checks = {
        "mean_delta_top1_pp": mean_d1 >= float(gates["mean_delta_top1_pp_min"]),
        "hierarchical_ci95_lower_pp": ci_lo >= float(gates["hierarchical_ci95_lower_pp_min"]),
        "mean_delta_top5_pp": mean_d5 >= float(gates["mean_delta_top5_pp_min"]),
        "no_signer_catastrophe": all(
            v >= float(gates["no_signer_catastrophe_pp"]) for v in signer_deltas.values()
        ),
    }
    verdict = "E4_R2_ACCURACY_PASS" if all(checks.values()) else str(gates["fail_status"])

    report = {
        "schema": "ms_sta_v6_e4_r2_confirm_validation_report_v1",
        "canonical": True,
        "campaign": "E4-R2",
        "selected_config_id": selected["config_id"],
        "git_revision": "56142e40f0d14fecc559881b8f3888610ff82f98",
        "executable_tree_sha256": "b4f6a01f539a1806c7394d308b107648d671e09566397d9cbea6fc0db3c3c70d",
        "seeds": seeds,
        "n_val": 2629,
        "homogeneous_bindings": bindings_ok,
        "seed_reports": seed_reports,
        "e4_top1_pct": [100.0 * x for x in e_top1],
        "b4_top1_pct": [100.0 * x for x in t_top1],
        "e4_mean_top1_pct": 100.0 * e_mean,
        "e4_sample_std_top1_pct": 100.0 * e_std,
        "b4_mean_top1_pct": 100.0 * t_mean,
        "mean_delta_top1_pp": mean_d1,
        "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
        "mean_delta_top5_pp": mean_d5,
        "mean_delta_macro_f1_pp": mean_df1,
        "signer_deltas_pp": signer_deltas,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    gate = {
        "schema": "ms_sta_v6_e4_r2_accuracy_gate_v1",
        "canonical": True,
        "campaign": "E4-R2",
        "verdict": verdict,
        "thresholds": gates,
        "checks": checks,
        "mean_delta_top1_pp": mean_d1,
        "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
        "mean_delta_top5_pp": mean_d5,
        "mean_delta_macro_f1_pp": mean_df1,
        "e4_top1": e_top1,
        "b4_top1": t_top1,
        "signer_deltas_pp": signer_deltas,
        "selected_config_id": selected["config_id"],
        "fail_status": gates["fail_status"],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Thresholds frozen; not modified after seeing results.",
    }
    completion = {
        "schema": "ms_sta_v6_e4_r2_confirm_completion_v1",
        "canonical": True,
        "status": "CONFIRM_COMPLETE",
        "accuracy_verdict": verdict,
        "n_seeds": 3,
        "epochs_each": 16,
        "selected_config_id": selected["config_id"],
        "confirm_rows_rel": "Docs/results/ms_sta_v6_e4_r2_runs_canonical/confirm/CONFIRM_ROWS.json",
        "validation_report_rel": "Docs/results/ms_sta_v6_e4_r2/CONFIRM_VALIDATION_REPORT.json",
        "accuracy_gate_rel": "Docs/results/ms_sta_v6_e4_r2/ACCURACY_GATE.json",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    index = {
        "schema": "ms_sta_v6_e4_r2_evidence_index_v1",
        "campaign": "E4-R2",
        "remediation": True,
        "invalid_comparator_companion_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_NO_GO_COMPANION_INVALID_COMPARATOR.json",
        "hpo_audit_remediated_rel": "Docs/results/ms_sta_v6_e4_r2/HPO_AUDIT_REMEDIATED.json",
        "hpo_selected_remediated_rel": "Docs/results/ms_sta_v6_e4_r2/HPO_SELECTED_REMEDIATED.json",
        "confirm_validation_report_rel": "Docs/results/ms_sta_v6_e4_r2/CONFIRM_VALIDATION_REPORT.json",
        "accuracy_gate_rel": "Docs/results/ms_sta_v6_e4_r2/ACCURACY_GATE.json",
        "confirm_completion_rel": "Docs/results/ms_sta_v6_e4_r2/CONFIRM_COMPLETION.json",
        "accuracy_verdict": verdict,
        "b4_primary": True,
        "e4_ready": False,
    }

    r_sha = atomic_write(out_root / "CONFIRM_VALIDATION_REPORT.json", report)
    g_sha = atomic_write(out_root / "ACCURACY_GATE.json", gate)
    c_sha = atomic_write(out_root / "CONFIRM_COMPLETION.json", completion)
    i_sha = atomic_write(out_root / "EVIDENCE_INDEX.json", index)

    # NO-GO artifact if failed
    if verdict != "E4_R2_ACCURACY_PASS":
        nogo = {
            "schema": "ms_sta_v6_e4_r2_accuracy_no_go_v1",
            "status": verdict,
            "canonical": True,
            "accuracy_gate_rel": "Docs/results/ms_sta_v6_e4_r2/ACCURACY_GATE.json",
            "mean_delta_top1_pp": mean_d1,
            "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
            "checks": checks,
            "keep_b4_primary": True,
            "e4_ready": False,
            "forbid_efficiency_benchmark": True,
            "forbid_oneshot_test": True,
            "forbid_d4": True,
            "prior_invalid_comparator_preserved": True,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write(out_root / "E4_R2_ACCURACY_NO_GO.json", nogo)

    runs_root = _ROOT / "Docs/results/ms_sta_v6_e4_r2_runs_canonical"
    prev_top = load_status_json(runs_root / "STATUS.json") if (runs_root / "STATUS.json").is_file() else None
    confirm_path = runs_root / "confirm" / "STATUS.json"
    confirm = load_status_json(confirm_path) if confirm_path.is_file() else {"status": "CONFIRM_COMPLETE"}
    write_campaign_status(
        runs_root=runs_root,
        sealed_rel_root=out_root,
        payload=build_campaign_status(
            status=verdict,
            phase="accuracy_closeout",
            git_revision=str(report.get("git_revision") or ""),
            executable_tree_sha256=str(report.get("executable_tree_sha256") or ""),
            prev=prev_top,
            extra={
                "accuracy_gate_rel": "Docs/results/ms_sta_v6_e4_r2/ACCURACY_GATE.json",
                "confirm_completion_rel": "Docs/results/ms_sta_v6_e4_r2/CONFIRM_COMPLETION.json",
                **(
                    {
                        "accuracy_no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_ACCURACY_NO_GO.json",
                        "forbid_efficiency_benchmark": True,
                    }
                    if verdict != "E4_R2_ACCURACY_PASS"
                    else {"forbid_efficiency_benchmark": False}
                ),
                "note": "Top-level STATUS supersedes stale E4_R2_HPO_NO_SELECT; historical NO_GO preserved.",
            },
        ),
        confirm=confirm,
    )

    print(
        json.dumps(
            {
                "event": "accuracy_gate_complete",
                "verdict": verdict,
                "mean_delta_top1_pp": mean_d1,
                "ci95": [ci_lo, ci_hi],
                "mean_delta_top5_pp": mean_d5,
                "e4_mean_top1_pct": 100.0 * e_mean,
                "b4_mean_top1_pct": 100.0 * t_mean,
                "checks": checks,
                "report_sha256": r_sha,
                "gate_sha256": g_sha,
                "completion_sha256": c_sha,
                "index_sha256": i_sha,
            },
            indent=2,
        )
    )
    return 0 if verdict == "E4_R2_ACCURACY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
