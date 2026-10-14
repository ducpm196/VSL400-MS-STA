# -*- coding: utf-8 -*-
"""E4-R4C Phase A/B: HPO boundary audit + dual-seed selection adjudication.

Create-only artifacts under Docs/results/ms_sta_v6_e4_r4/.
Does not modify HPO_SELECTION.json or frozen policies.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical"
HPO_DIR = CANON / "hpo"
OUT_DIR = ROOT / "Docs/results/ms_sta_v6_e4_r4"
SCREENING = (11, 29)
REQUIRED_FILES = ("best.pt", "history.json", "summary.json", "val_predictions.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(cmd: list[str]) -> str:
    return subprocess.check_output(["git", *cmd], cwd=ROOT, text=True).strip()


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def sample_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)  # sample std (n-1)


def group_hparams(rows: list[dict[str, Any]]) -> dict[tuple[float, float], list[dict[str, Any]]]:
    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (float(row["lr"]), float(row["weight_decay"]))
        groups[key].append(row)
    return dict(groups)


def validate_group_seeds(group_rows: list[dict[str, Any]], required: tuple[int, ...] = SCREENING) -> None:
    seeds = [int(r["seed"]) for r in group_rows]
    if sorted(seeds) != sorted(required):
        raise ValueError(f"seed_set_mismatch got={seeds} need={list(required)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate_seed:{seeds}")
    for r in group_rows:
        if not is_finite(r["val_top1"]) or not is_finite(r["val_top5"]):
            raise ValueError(f"non_finite_metric:{r.get('config_id')}")


def rank_groups(groups: dict[tuple[float, float], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    ranked = []
    for (lr, wd), rows in groups.items():
        validate_group_seeds(rows)
        top1s = [float(r["val_top1"]) for r in rows]
        top5s = [float(r["val_top5"]) for r in rows]
        config_ids = sorted(str(r["config_id"]) for r in rows)
        ranked.append(
            {
                "lr": lr,
                "weight_decay": wd,
                "mean_top1": float(statistics.mean(top1s)),
                "std_top1": float(sample_std(top1s)),
                "mean_top5": float(statistics.mean(top5s)),
                "std_top5": float(sample_std(top5s)),
                "raw_rows": sorted(rows, key=lambda r: int(r["seed"])),
                "stable_config_id": config_ids[0],
            }
        )
    ranked.sort(
        key=lambda g: (-g["mean_top1"], -g["mean_top5"], g["stable_config_id"])
    )
    return ranked


def load_canonical_rows() -> list[dict[str, Any]]:
    index = json.loads((CANON / "HPO_INDEX.json").read_text(encoding="utf-8"))
    rows = []
    for trial in index["trials"]:
        cid = trial["config_id"]
        summary = json.loads((HPO_DIR / cid / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "config_id": cid,
                "lr": float(trial["lr"]),
                "weight_decay": float(trial["weight_decay"]),
                "seed": int(trial["seed"]),
                "val_top1": float(summary["best_val_top1"]),
                "val_top5": float(summary.get("final_top5", summary.get("best_val_top5", float("nan")))),
                "summary_path": str((HPO_DIR / cid / "summary.json").relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return rows


def audit_predictions(split: dict[str, Any]) -> list[dict[str, Any]]:
    tune_ids = set(str(x) for x in (split.get("tune_ids") or split.get("roles", {}).get("tune") or []))
    # Prefer explicit role lists if nested
    if not tune_ids and "role_ids" in split:
        tune_ids = set(str(x) for x in split["role_ids"].get("tune", []))
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    issues = []
    for d in sorted(HPO_DIR.iterdir()):
        if not d.is_dir():
            continue
        pred_path = d / "val_predictions.json"
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        rows = pred.get("rows") or []
        ids = [str(r["sample_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            issues.append({"config_id": d.name, "error": "duplicate_sample_ids"})
        if any(not is_finite(r.get("prob_true")) for r in rows):
            issues.append({"config_id": d.name, "error": "non_finite_prob"})
        leak = sorted(set(ids) & test_ids)
        if leak:
            issues.append({"config_id": d.name, "error": "test_id_leakage", "n": len(leak)})
        if tune_ids and set(ids) - tune_ids:
            # allow if role not encoded as tune_ids; record soft warning
            issues.append(
                {
                    "config_id": d.name,
                    "warning": "ids_outside_tune_set",
                    "n_outside": len(set(ids) - tune_ids),
                }
            )
        issues.append(
            {
                "config_id": d.name,
                "n_rows": len(ids),
                "unique_ids": len(set(ids)),
                "top1": pred.get("top1"),
                "top5": pred.get("top5"),
            }
        )
    return issues


def resolve_tune_ids(split: dict[str, Any]) -> set[str]:
    if "tune_ids" in split:
        return set(str(x) for x in split["tune_ids"])
    roles = split.get("roles") or {}
    if "tune" in roles:
        return set(str(x) for x in roles["tune"])
    # Path-B style nested
    for key in ("splits", "by_role"):
        block = split.get(key) or {}
        if "tune" in block:
            return set(str(x) for x in block["tune"])
    return set()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = OUT_DIR / "E4_R4C_HPO_BOUNDARY_AUDIT.json"
    adj_path = OUT_DIR / "E4_R4C_HPO_SELECTION_ADJUDICATION.json"
    if audit_path.exists():
        raise SystemExit(f"refuse_overwrite:{audit_path}")
    if adj_path.exists():
        raise SystemExit(f"refuse_overwrite:{adj_path}")

    head = git_rev(["rev-parse", "HEAD"])
    lock = json.loads((OUT_DIR / "E4_R4C_HPO_CODE_LOCK.json").read_text(encoding="utf-8"))
    lock_head = lock["git_head"]
    # ancestor check
    merge_base = git_rev(["merge-base", lock_head, head])
    lock_is_ancestor = merge_base == lock_head

    index = json.loads((CANON / "HPO_INDEX.json").read_text(encoding="utf-8"))
    selection = json.loads((CANON / "HPO_SELECTION.json").read_text(encoding="utf-8"))
    train_pol = json.loads(
        (ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json").read_text(encoding="utf-8")
    )
    split_path = ROOT / train_pol["split_manifest_rel"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha = sha256_file(split_path)
    teacher_path = ROOT / train_pol["teacher_checkpoint_rel"]
    teacher_sha = sha256_file(teacher_path)

    # artifact presence
    run_audit = []
    hard_errors = []
    for trial in index["trials"]:
        cid = trial["config_id"]
        run_dir = HPO_DIR / cid
        present = {name: (run_dir / name).is_file() for name in REQUIRED_FILES}
        hashes = {
            name: sha256_file(run_dir / name) if present[name] else None for name in REQUIRED_FILES
        }
        missing = [k for k, v in present.items() if not v]
        if missing:
            hard_errors.append({"config_id": cid, "missing": missing})
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if present["summary.json"] else {}
        if present["summary.json"]:
            if not is_finite(summary.get("best_val_top1")):
                hard_errors.append({"config_id": cid, "error": "non_finite_best_val_top1"})
        run_audit.append(
            {
                "config_id": cid,
                "present": present,
                "sha256": hashes,
                "best_val_top1": summary.get("best_val_top1"),
                "final_top5": summary.get("final_top5"),
                "seed": summary.get("seed", trial["seed"]),
            }
        )

    # invalidated prelock markers
    prelock_readme = ROOT / "Docs/results/ms_sta_v6_e4_r4/_r4c_hpo_invalid_prelock/README.md"
    prelock_ok = prelock_readme.is_file() and "Do not" in prelock_readme.read_text(encoding="utf-8")

    # prediction checks (test leakage hard)
    tune_ids = resolve_tune_ids(split)
    test_ids = set(str(x) for x in (split.get("test_ids") or []))
    pred_checks = []
    for trial in index["trials"]:
        cid = trial["config_id"]
        pred = json.loads((HPO_DIR / cid / "val_predictions.json").read_text(encoding="utf-8"))
        ids = [str(r["sample_id"]) for r in pred["rows"]]
        uniq = set(ids)
        leak = sorted(uniq & test_ids)
        entry = {
            "config_id": cid,
            "n": len(ids),
            "unique": len(uniq),
            "duplicate": len(ids) != len(uniq),
            "test_leak_n": len(leak),
            "role_hint": "tune",
            "ids_subset_of_tune": (uniq <= tune_ids) if tune_ids else None,
        }
        if entry["duplicate"] or entry["test_leak_n"]:
            hard_errors.append({"config_id": cid, "error": "prediction_integrity", **entry})
        pred_checks.append(entry)

    confirm_dir = CANON / "confirm"
    confirm_exists = confirm_dir.is_dir() and any(confirm_dir.iterdir())

    gate_a_pass = (
        len(index["trials"]) == 8
        and not hard_errors
        and lock_is_ancestor
        and teacher_sha == train_pol["teacher_checkpoint_sha256"]
        and split_sha == train_pol["split_manifest_sha256"]
        and prelock_ok
        and not confirm_exists
    )

    audit = {
        "schema": "ms_sta_v6_e4_r4c_hpo_boundary_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "branch": git_rev(["branch", "--show-current"]),
        "git_head": head,
        "hpo_code_lock_git_head": lock_head,
        "hpo_code_lock_is_ancestor_of_head": lock_is_ancestor,
        "hpo_code_lock_tree_sha256": lock.get("executable_tree_sha256"),
        "n_index_trials": len(index["trials"]),
        "runs": run_audit,
        "prediction_checks": pred_checks,
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_sha_matches_policy": teacher_sha == train_pol["teacher_checkpoint_sha256"],
        "split_manifest_rel": train_pol["split_manifest_rel"],
        "split_manifest_sha256": split_sha,
        "split_sha_matches_policy": split_sha == train_pol["split_manifest_sha256"],
        "invalidated_prelock_readme_present": prelock_ok,
        "confirm_dir_nonempty": confirm_exists,
        "hard_errors": hard_errors,
        "hashes": {
            "HPO_INDEX.json": sha256_file(CANON / "HPO_INDEX.json"),
            "HPO_SELECTION.json": sha256_file(CANON / "HPO_SELECTION.json"),
            "E4_R4C_HPO_CODE_LOCK.json": sha256_file(OUT_DIR / "E4_R4C_HPO_CODE_LOCK.json"),
            "E4_R4C_POST_HPO_BOUNDARY_POLICY.json": sha256_file(
                ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_POST_HPO_BOUNDARY_POLICY.json"
            ),
        },
        "gate_a": "PASS" if gate_a_pass else "FAIL",
        "e4_ready": False,
        "auto_launch_confirm": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Phase B adjudication
    rows = load_canonical_rows()
    groups = group_hparams(rows)
    ranked = rank_groups(groups)
    winner = ranked[0]
    sel_w = selection["winner"]
    same_hparams = (
        abs(float(sel_w["lr"]) - float(winner["lr"])) < 1e-15
        and abs(float(sel_w["weight_decay"]) - float(winner["weight_decay"])) < 1e-15
    )
    if same_hparams:
        materiality = "SEED_PRESENTATION_DEFECT_NONMATERIAL_TO_HPARAM_CHOICE"
        gate_b = "PASS"
        confirm_hparams = {"lr": winner["lr"], "weight_decay": winner["weight_decay"]}
        blocked = False
    else:
        materiality = "MATERIAL_HPARAM_MISMATCH_REQUIRES_INDEPENDENT_ADJUDICATION"
        gate_b = "BLOCKED"
        confirm_hparams = None
        blocked = True

    adjudication = {
        "schema": "ms_sta_v6_e4_r4c_hpo_selection_adjudication_v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_rule": [
            "group_by_(lr,weight_decay)",
            "require_seeds_11_and_29",
            "rank_by_mean_top1_then_mean_top5_then_stable_config_id",
            "do_not_overwrite_HPO_SELECTION.json",
            "confirm_uses_group_hparams_not_screening_seed_29_as_final",
        ],
        "source_hashes": {
            "HPO_INDEX.json": audit["hashes"]["HPO_INDEX.json"],
            "HPO_SELECTION.json": audit["hashes"]["HPO_SELECTION.json"],
            "summaries": {r["config_id"]: sha256_file(HPO_DIR / r["config_id"] / "summary.json") for r in rows},
        },
        "raw_rows": rows,
        "grouped_ranked": [
            {
                "lr": g["lr"],
                "weight_decay": g["weight_decay"],
                "mean_top1": g["mean_top1"],
                "std_top1": g["std_top1"],
                "mean_top5": g["mean_top5"],
                "std_top5": g["std_top5"],
                "stable_config_id": g["stable_config_id"],
                "raw_config_ids": [r["config_id"] for r in g["raw_rows"]],
                "raw_top1": [r["val_top1"] for r in g["raw_rows"]],
                "raw_top5": [r["val_top5"] for r in g["raw_rows"]],
            }
            for g in ranked
        ],
        "group_winner": {
            "lr": winner["lr"],
            "weight_decay": winner["weight_decay"],
            "mean_top1": winner["mean_top1"],
            "mean_top5": winner["mean_top5"],
        },
        "hpo_selection_winner_hparams": {
            "lr": sel_w["lr"],
            "weight_decay": sel_w["weight_decay"],
            "seed": sel_w["seed"],
            "config_id": sel_w["config_id"],
        },
        "same_hparams_as_selection_file": same_hparams,
        "materiality": materiality,
        "confirm_hyperparameters": confirm_hparams,
        "forbid_final_seed_29": True,
        "confirm_seeds": [42, 123, 456],
        "gate_b": gate_b,
        "blocked": blocked,
        "e4_ready": False,
        "notes": [
            "HPO_SELECTION.json ranks single runs (seed presentation); confirm must use grouped mean hparams.",
            "Expected checksum direction: (3e-4, 1e-2) mean Top-1 ~24.54% > (3e-4, 1e-3) ~22.88%.",
        ],
    }
    adj_path.write_text(json.dumps(adjudication, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"gate_a": audit["gate_a"], "gate_b": gate_b, "materiality": materiality, "confirm_hparams": confirm_hparams}, indent=2))
    return 0 if gate_a_pass and not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
