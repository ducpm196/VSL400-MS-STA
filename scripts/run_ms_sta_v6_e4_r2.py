#!/usr/bin/env python3
"""MS-STA E4-R2 orchestrator — distill latency-promising arms, finite HPO.

Canonical mode requires clean worktree, HEAD==upstream, code-lock match,
and refuse-existing trial outputs. Does not open test / D4 / registry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.eval.ms_sta_v5_m0_strong_baseline import load_json  # noqa: E402
from src.training.ms_sta_e4_r2_provenance import (  # noqa: E402
    assert_no_drift,
    require_clean_code_lock,
    sha256_file,
    verify_sealed_code_lock,
)


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def write_status(runs_root: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write(runs_root / "STATUS.json", payload)


def _python() -> Path:
    for rel in ("_venvs/train_cuda/Scripts/python.exe", "env/Scripts/python.exe"):
        p = _ROOT / rel
        if p.is_file():
            return p
    return Path(sys.executable)


def hpo_grid(policy: dict[str, Any], *, wave: str) -> list[dict[str, Any]]:
    hpo = policy["hpo"]
    if wave == "1":
        cands = list(policy.get("wave1_candidates") or policy["candidates"][:2])
    else:
        cands = list(policy["candidates"])
    seeds = [int(s) for s in hpo["screening_seeds"]]
    trials = []
    for variant, lr, dw, seed in itertools.product(
        cands, hpo["lr"], hpo["distill_weight"], seeds
    ):
        trials.append(
            {
                "variant": variant,
                "lr": float(lr),
                "dropout": float(hpo["dropout"]),
                "weight_decay": float(hpo["weight_decay"]),
                "distill_weight": float(dw),
                "seed": int(seed),
            }
        )
    if len(trials) > int(hpo["max_runs"]):
        raise RuntimeError(f"hpo_grid_exceeds_max_runs:{len(trials)}")
    return trials


def run_trial(policy_path: Path, trial: dict[str, Any], phase: str, out_dir: Path) -> dict[str, Any]:
    # Never reuse: trainer also refuse_existing_outputs.
    if out_dir.exists() and any((out_dir / n).is_file() for n in ("SUMMARY.json", "best.pt", "VAL_PREDS.json")):
        raise RuntimeError(f"refuse_existing_trial_dir:{out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(_python()),
        str(_ROOT / "src" / "training" / "ms_sta_v5_train.py"),
        "--policy",
        str(policy_path),
        "--variant",
        trial["variant"],
        "--phase",
        phase,
        "--seed",
        str(trial["seed"]),
        "--lr",
        str(trial["lr"]),
        "--dropout",
        str(trial["dropout"]),
        "--weight-decay",
        str(trial["weight_decay"]),
        "--distill-weight",
        str(trial.get("distill_weight", 0.5)),
        "--out-dir",
        str(out_dir),
        "--device",
        "cuda",
    ]
    with (out_dir / "train.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    summary_path = out_dir / "SUMMARY.json"
    if proc.returncode != 0 or not summary_path.is_file():
        raise RuntimeError(f"trial_failed:{out_dir}:rc={proc.returncode}")
    summary = load_json(summary_path)
    if not summary.get("canonical"):
        raise RuntimeError(f"trial_not_canonical:{out_dir}")
    if not summary.get("git_revision") or summary.get("git_dirty") is not False:
        raise RuntimeError(f"trial_provenance_incomplete:{out_dir}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy",
        default=str(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_TRAIN_POLICY.json"),
    )
    ap.add_argument("--wave", choices=["1", "all"], default="1")
    ap.add_argument("--phase", choices=["hpo", "status"], default="hpo")
    args = ap.parse_args()

    policy_path = Path(args.policy)
    policy = load_json(policy_path)
    if policy.get("status") != "FROZEN" or not policy.get("train_allowed"):
        raise SystemExit("train_policy_not_frozen")
    if not policy.get("forbid_final_test"):
        raise SystemExit("forbid_final_test_required")
    if not policy.get("require_canonical_provenance"):
        raise SystemExit("require_canonical_provenance_must_be_true")

    code_lock = require_clean_code_lock(
        root=_ROOT, expect_branch=str(policy.get("branch") or "research/ms-sta-v6-e4-r2")
    )
    sealed_path = _ROOT / policy["code_lock_rel"]
    if not sealed_path.is_file():
        raise SystemExit(f"code_lock_missing:{policy['code_lock_rel']}")
    sealed = load_json(sealed_path)
    policy_sha = sha256_file(policy_path)
    verify_sealed_code_lock(root=_ROOT, sealed=sealed, live=code_lock, policy_sha=policy_sha)
    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)
    # Canonical runs root must not contain pre-lock diagnostic rows.
    if "prelock" in str(runs_root).lower() or runs_root.name == "ms_sta_v6_e4_r2_runs":
        raise SystemExit("canonical_runs_root_must_not_be_prelock_or_legacy")

    pilot = load_json(_ROOT / policy["pilot_bench_rel"])
    if pilot.get("canonical") is True:
        raise SystemExit("pilot_bench_must_remain_non_canonical_screening")
    promising = set(pilot.get("latency_promising_arms") or [])
    write_status(
        runs_root,
        {
            "campaign": "E4-R2",
            "phase": f"hpo_wave_{args.wave}",
            "status": "RUNNING",
            "canonical": True,
            "latency_promising_arms": sorted(promising),
            "python": str(_python()),
            "bindings": code_lock,
            "code_lock_rel": policy["code_lock_rel"],
            "train_policy_sha256": policy_sha,
        },
    )

    if args.phase == "status":
        print(json.dumps(load_json(runs_root / "STATUS.json"), indent=2))
        return 0

    trials = hpo_grid(policy, wave=args.wave)
    filtered = []
    for t in trials:
        arm = t["variant"].split("_", 1)[1]
        if arm not in promising:
            continue
        filtered.append(t)
    trials = filtered

    rows: list[dict[str, Any]] = []
    rows_name = "HPO_WAVE1_ROWS.json" if args.wave == "1" else "HPO_ROWS.json"
    for i, trial in enumerate(trials, 1):
        assert_no_drift(code_lock, root=_ROOT)
        tag = (
            f"{trial['variant']}_lr{trial['lr']}_dw{trial['distill_weight']}"
            f"_do{trial['dropout']}_wd{trial['weight_decay']}_s{trial['seed']}"
        )
        out_dir = runs_root / "hpo" / tag
        write_status(
            runs_root,
            {
                "campaign": "E4-R2",
                "phase": f"hpo_wave_{args.wave}",
                "status": "RUNNING",
                "canonical": True,
                "trial_index": i,
                "trial_total": len(trials),
                "current_tag": tag,
                "git_revision": code_lock["git_revision"],
                "executable_tree_sha256": code_lock["executable_tree_sha256"],
            },
        )
        print(json.dumps({"event": "trial_start", "i": i, "n": len(trials), "tag": tag, "canonical": True}))
        summary = run_trial(policy_path, trial, "hpo", out_dir)
        row = {
            **trial,
            "tag": tag,
            "best_val_top1": summary["best_val_top1"],
            "best_val_top5": summary.get("best_val_top5"),
            "best_val_macro_f1": summary.get("best_val_macro_f1"),
            "epochs_run": summary.get("epochs_run"),
            "out_rel": str(out_dir.relative_to(_ROOT)).replace("\\", "/"),
            "canonical": True,
            "git_revision": summary.get("git_revision"),
            "executable_tree_sha256": summary.get("executable_tree_sha256"),
            "policy_sha256": summary.get("policy_sha256"),
        }
        rows.append(row)
        atomic_write(runs_root / rows_name, {"schema": "e4_r2_hpo_rows_v1", "canonical": True, "rows": rows})

    write_status(
        runs_root,
        {
            "campaign": "E4-R2",
            "phase": f"hpo_wave_{args.wave}",
            "status": "HPO_WAVE_COMPLETE",
            "canonical": True,
            "n_trials": len(rows),
            "rows_rel": str((runs_root / rows_name).relative_to(_ROOT)).replace("\\", "/"),
            "git_revision": code_lock["git_revision"],
            "executable_tree_sha256": code_lock["executable_tree_sha256"],
            "note": "Selection only after wave audit; pre-lock diagnostic forbidden.",
        },
    )
    print(json.dumps({"event": "hpo_wave_complete", "n": len(rows), "canonical": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
