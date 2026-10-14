#!/usr/bin/env python3
"""E4-R2 wave-1 HPO audit + selection (create-only). Pre-lock forbidden."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]

from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.ms_sta_v6_e4_r2 import build_ms_sta_v6_e4_r2  # noqa: E402
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


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def _p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[max(0, int(0.9 * (len(s) - 1)))]


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _load_tune_loader(policy: dict, batch_size: int = 8):
    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    ids = load_split_role_ids(split, policy["hpo"]["val_role"])
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    ds = VSL400MSSTAv5Dataset(
        ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=test_ids,
    )
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_v5), len(ds)


def _bench(model, batch, *, warmup: int = 40, trials: int = 200) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return {"median_sec": _median(times), "p90_sec": _p90(times)}


def main() -> int:
    policy = load_json(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_TRAIN_POLICY.json")
    rows_obj = load_json(_ROOT / policy["runs_root_rel"] / "HPO_WAVE1_ROWS.json")
    rows = rows_obj["rows"]
    if len(rows) != 8:
        raise SystemExit(f"expected_8_rows_got_{len(rows)}")

    # Homogeneity
    gits, trees, pols, teachers, splits = set(), set(), set(), set(), set()
    verified_rows = []
    for r in rows:
        out = _ROOT / r["out_rel"]
        summary = load_json(out / "SUMMARY.json")
        prov = load_json(out / "PROVENANCE.json")
        ckpt = out / "best.pt"
        if not ckpt.is_file():
            raise SystemExit(f"missing_ckpt:{out}")
        if not summary.get("canonical") or summary.get("git_dirty") is not False:
            raise SystemExit(f"noncanonical_summary:{out}")
        gits.add(summary["git_revision"])
        trees.add(summary["executable_tree_sha256"])
        pols.add(summary["policy_sha256"])
        teachers.add(summary["teacher_checkpoint_sha256"])
        splits.add(summary["split_manifest_sha256"])
        verified_rows.append(
            {
                **r,
                "checkpoint_sha256": sha256_file(ckpt),
                "summary_sha256": sha256_file(out / "SUMMARY.json"),
                "provenance_sha256": sha256_file(out / "PROVENANCE.json"),
                "bindings_git": prov.get("git_revision"),
                "bindings_tree": prov.get("executable_tree_sha256"),
            }
        )
    if len(gits) != 1 or len(trees) != 1 or len(pols) != 1 or len(teachers) != 1 or len(splits) != 1:
        raise SystemExit("heterogeneous_bindings")
    if next(iter(gits)) != "eb4bed23a4118365c5449fa84f14bf02c93ebeb6":
        raise SystemExit("git_revision_mismatch_expected_eb4bed2")
    if next(iter(trees)) != "c2b9412786cc78eb8f96cd3ab2b27250ec19bd41350c66d1fbadf7492911349c":
        raise SystemExit("tree_mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("cuda_required_for_hpo_audit")
    if torch.cuda.get_device_name(0) != policy["scientific_environment_lock"]["required_gpu_name"]:
        raise SystemExit("gpu_mismatch")

    # Teacher tune Top-1 (HPO eligibility reference)
    tune_loader, tune_n = _load_tune_loader(policy)
    teacher = build_ms_sta_v6("b4_v6_ctr").to(device)
    t_ckpt = torch.load(_ROOT / policy["teacher_checkpoint_rel"], map_location=device, weights_only=False)
    teacher.load_state_dict(t_ckpt["model"])
    teacher.eval()
    t_metrics = evaluate(teacher, tune_loader, device, collect_rows=False)
    teacher_tune_top1 = float(t_metrics["top1"])
    elig_min_pp = float(policy["hpo"]["accuracy_eligibility_tune_delta_top1_pp_min"])

    # Aggregate configs over seeds
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in verified_rows:
        cfg_id = f"{r['variant']}_lr{r['lr']}_dw{r['distill_weight']}_do{r['dropout']}_wd{r['weight_decay']}"
        by_cfg[cfg_id].append(r)

    # Latency bench: one batch from tune, seed-42 checkpoint per config
    batch = next(iter(tune_loader))
    batch = _to_device(batch, device)
    # Teacher latency baseline for ranking only (not efficiency gate)
    b4_lat = _bench(teacher, batch)

    configs = []
    for cfg_id, rs in sorted(by_cfg.items()):
        seeds = sorted(int(x["seed"]) for x in rs)
        if seeds != [42, 123]:
            raise SystemExit(f"bad_seeds:{cfg_id}:{seeds}")
        mean_top1 = sum(float(x["best_val_top1"]) for x in rs) / len(rs)
        mean_top5 = sum(float(x["best_val_top5"]) for x in rs) / len(rs)
        delta_pp = 100.0 * (mean_top1 - teacher_tune_top1)
        eligible = delta_pp >= elig_min_pp
        # latency from seed 42 ckpt
        r42 = next(x for x in rs if int(x["seed"]) == 42)
        arm = r42["variant"].split("_", 1)[1]
        m = build_ms_sta_v6_e4_r2(arm, dropout=0.0).to(device)
        sd = torch.load(_ROOT / r42["out_rel"] / "best.pt", map_location=device, weights_only=False)
        m.load_state_dict(sd["model"])
        lat = _bench(m, batch)
        del m
        torch.cuda.empty_cache()
        configs.append(
            {
                "config_id": cfg_id,
                "variant": r42["variant"],
                "lr": r42["lr"],
                "dropout": r42["dropout"],
                "weight_decay": r42["weight_decay"],
                "distill_weight": r42["distill_weight"],
                "seeds": seeds,
                "mean_tune_top1": mean_top1,
                "mean_tune_top5": mean_top5,
                "delta_top1_pp_vs_teacher_tune": delta_pp,
                "accuracy_eligible": eligible,
                "batch1_median_sec": lat["median_sec"],
                "batch1_p90_sec": lat["p90_sec"],
                "batch1_speedup_vs_b4": (b4_lat["median_sec"] / lat["median_sec"]) - 1.0,
                "seed42_out_rel": r42["out_rel"],
                "trials": [
                    {
                        "seed": x["seed"],
                        "tag": x["tag"],
                        "best_val_top1": x["best_val_top1"],
                        "checkpoint_sha256": x["checkpoint_sha256"],
                        "out_rel": x["out_rel"],
                    }
                    for x in sorted(rs, key=lambda z: int(z["seed"]))
                ],
            }
        )

    # Selection: eligible first, then faster median, then better p90, then config_id
    eligible_cfgs = [c for c in configs if c["accuracy_eligible"]]
    eligible_cfgs.sort(
        key=lambda c: (
            c["batch1_median_sec"],
            c["batch1_p90_sec"],
            c["config_id"],
        )
    )
    selected = eligible_cfgs[0] if eligible_cfgs else None

    audit = {
        "schema": "ms_sta_v6_e4_r2_hpo_audit_v1",
        "canonical": True,
        "campaign": "E4-R2",
        "wave": 1,
        "git_revision": next(iter(gits)),
        "executable_tree_sha256": next(iter(trees)),
        "policy_sha256": next(iter(pols)),
        "teacher_checkpoint_sha256": next(iter(teachers)),
        "split_manifest_sha256": next(iter(splits)),
        "n_trials": len(verified_rows),
        "homogeneous_bindings": True,
        "teacher_tune_top1": teacher_tune_top1,
        "teacher_tune_n": tune_n,
        "eligibility_delta_pp_min": elig_min_pp,
        "b4_batch1_median_sec_audit": b4_lat["median_sec"],
        "note_latency": "Audit microbench for selection ranking only; not efficiency-gate evidence.",
        "configs": configs,
        "n_accuracy_eligible": len(eligible_cfgs),
        "wave2_authorized": False,
        "wave2_note": "Policy does not authorize auto wave-2; A/B/C not launched.",
        "selected_config_id": None if selected is None else selected["config_id"],
    }
    selected_doc = {
        "schema": "ms_sta_v6_e4_r2_hpo_selected_v1",
        "canonical": True,
        "campaign": "E4-R2",
        "selection_order": policy["hpo"]["selection_order"],
        "selected": selected,
        "status": "SELECTED" if selected is not None else "NO_ELIGIBLE_WAVE1",
        "git_revision": next(iter(gits)),
        "executable_tree_sha256": next(iter(trees)),
    }

    runs = _ROOT / policy["runs_root_rel"]
    a_sha = atomic_write(runs / "HPO_AUDIT.json", audit)
    s_sha = atomic_write(runs / "HPO_SELECTED.json", selected_doc)
    print(
        json.dumps(
            {
                "event": "hpo_audit_written",
                "audit_sha256": a_sha,
                "selected_sha256": s_sha,
                "teacher_tune_top1": teacher_tune_top1,
                "n_eligible": len(eligible_cfgs),
                "selected": None if selected is None else selected["config_id"],
                "eligible": [
                    {
                        "id": c["config_id"],
                        "delta_pp": c["delta_top1_pp_vs_teacher_tune"],
                        "med": c["batch1_median_sec"],
                    }
                    for c in eligible_cfgs
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
