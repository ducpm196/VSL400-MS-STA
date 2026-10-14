# -*- coding: utf-8 -*-
"""E4-R4C finite HPO / confirm orchestrator — single candidate R1_S_A_d128."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def atomic_write(path: Path, obj: dict) -> None:
    if path.is_file():
        raise SystemExit(f"refuse_overwrite:{path}")
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)


def py() -> Path:
    p = _ROOT / "_venvs/train_cuda/Scripts/python.exe"
    return p if p.is_file() else Path(sys.executable)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--train-policy", type=Path, default=_ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_TRAIN_POLICY.json")
    ap.add_argument("--phase", choices=["hpo", "confirm"], required=True)
    ap.add_argument("--require-hpo-code-lock", action="store_true", default=True)
    ap.add_argument("--allow-unlocked", action="store_true")
    args = ap.parse_args()

    if args.phase == "confirm" and args.allow_unlocked:
        raise SystemExit("allow_unlocked_forbidden_for_confirm")

    policy = json.loads(Path(args.train_policy).read_text(encoding="utf-8"))
    quality = _ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_FULL_CORPUS_QUALITY.json"
    if not quality.is_file() or json.loads(quality.read_text(encoding="utf-8")).get("status") != "PASS":
        raise SystemExit("quality_not_pass")
    lock = json.loads((_ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4B_FEASIBILITY_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("verdict") != "E4_R4B_STAGE_MATCHED_PASS":
        raise SystemExit("feasibility_lock_not_pass")

    if args.phase == "hpo" and args.require_hpo_code_lock and not args.allow_unlocked:
        cl_path = _ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_CODE_LOCK.json"
        if not cl_path.is_file():
            raise SystemExit("missing_hpo_code_lock")
        cl = json.loads(cl_path.read_text(encoding="utf-8"))
        if cl.get("status") != "HPO_CODE_LOCK_PASS":
            raise SystemExit("hpo_code_lock_not_pass")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
        lock_head = str(cl.get("git_head") or "")
        if lock_head != head:
            anc = subprocess.run(
                ["git", "merge-base", "--is-ancestor", lock_head, head],
                cwd=_ROOT,
                check=False,
            )
            if anc.returncode != 0:
                raise SystemExit(f"hpo_code_lock_git_mismatch:{lock_head}!={head}")
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True).strip()
        if dirty:
            raise SystemExit("worktree_dirty_during_hpo")
        h = hashlib.sha256()
        for rel in cl["tree_files"]:
            p = _ROOT / rel
            h.update(rel.encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
        if h.hexdigest() != cl.get("executable_tree_sha256"):
            raise SystemExit("hpo_code_lock_tree_mismatch")

    perf_lock = None
    if args.phase == "confirm":
        pl_path = _ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_PERFORMANCE_CODE_LOCK.json"
        if not pl_path.is_file():
            raise SystemExit("missing_performance_code_lock")
        perf_lock = json.loads(pl_path.read_text(encoding="utf-8"))
        if perf_lock.get("status") != "PERFORMANCE_CODE_LOCK_PASS":
            raise SystemExit("performance_code_lock_not_pass")
        if not perf_lock.get("forbid_allow_unlocked", True):
            raise SystemExit("performance_lock_must_forbid_allow_unlocked")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
        lock_head = str(perf_lock.get("git_head") or "")
        if lock_head != head:
            anc = subprocess.run(
                ["git", "merge-base", "--is-ancestor", lock_head, head],
                cwd=_ROOT,
                check=False,
            )
            if anc.returncode != 0:
                raise SystemExit(f"performance_code_lock_git_mismatch:{lock_head}!={head}")
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True).strip()
        if dirty:
            raise SystemExit("worktree_dirty_during_confirm")
        # Tree hash must still match locked executable set (performance-only files).
        h = hashlib.sha256()
        for rel in perf_lock["tree_files"]:
            p = _ROOT / rel
            h.update(rel.encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
        if h.hexdigest() != perf_lock.get("executable_tree_sha256"):
            raise SystemExit("performance_code_lock_tree_mismatch")
        teacher_parity = json.loads(
            (_ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_TEACHER_LOGITS_PARITY.json").read_text(encoding="utf-8")
        )
        if teacher_parity.get("gate_c") != "PASS":
            raise SystemExit("teacher_parity_not_pass")

    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)
    feature_root = args.feature_root if args.feature_root.is_absolute() else (_ROOT / args.feature_root)

    if args.phase == "hpo":
        hpo = policy["hpo"]
        trials = []
        for lr, wd, seed in itertools.product(hpo["lr"], hpo["weight_decay"], hpo["screening_seeds"]):
            cfg = f"R1_S_A_d128_lr{lr}_wd{wd}_s{seed}"
            trials.append({"config_id": cfg, "lr": float(lr), "weight_decay": float(wd), "seed": int(seed)})
        index_path = runs_root / "HPO_INDEX.json"
        if not index_path.is_file():
            atomic_write(
                index_path,
                {
                    "schema": "ms_sta_v6_e4_r4c_hpo_index_v1",
                    "n_trials": len(trials),
                    "trials": trials,
                    "feature_root": str(feature_root.relative_to(_ROOT)).replace("\\", "/"),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        results = []
        for t in trials:
            out_dir = runs_root / "hpo" / t["config_id"]
            if (out_dir / "summary.json").is_file():
                summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            else:
                cmd = [
                    str(py()),
                    "-m",
                    "src.training.ms_sta_e4_r4c_train",
                    "--train-policy",
                    str(args.train_policy),
                    "--feature-root",
                    str(feature_root),
                    "--out-dir",
                    str(out_dir),
                    "--train-role",
                    hpo["train_role"],
                    "--val-role",
                    hpo["val_role"],
                    "--seed",
                    str(t["seed"]),
                    "--lr",
                    str(t["lr"]),
                    "--weight-decay",
                    str(t["weight_decay"]),
                    "--config-id",
                    t["config_id"],
                    "--early-stopping",
                    "--patience",
                    str(hpo["patience"]),
                ]
                print("RUN", " ".join(cmd), flush=True)
                subprocess.check_call(cmd, cwd=_ROOT)
                summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            results.append({**t, "val_top1": summary["best_val_top1"], "val_top5": summary.get("final_top5")})
        ranked = sorted(results, key=lambda r: (-r["val_top1"], -(r.get("val_top5") or 0), r["config_id"]))
        winner = ranked[0]
        sel_path = runs_root / "HPO_SELECTION.json"
        if not sel_path.is_file():
            atomic_write(
                sel_path,
                {
                    "schema": "ms_sta_v6_e4_r4c_hpo_selection_v1",
                    "winner": winner,
                    "ranked": ranked,
                    "selection_order": hpo["selection_order"],
                    "e4_ready": False,
                },
            )
        print("HPO_WINNER", winner["config_id"], winner["val_top1"])
        return 0

    # confirm
    assert perf_lock is not None
    sel = json.loads((runs_root / "HPO_SELECTION.json").read_text(encoding="utf-8"))
    w = sel["winner"]
    adj_path = _ROOT / "Docs/results/ms_sta_v6_e4_r4/E4_R4C_HPO_SELECTION_ADJUDICATION.json"
    if adj_path.is_file():
        adj = json.loads(adj_path.read_text(encoding="utf-8"))
        conf_hp = adj.get("confirm_hyperparameters") or adj.get("adjudicated_hyperparameters") or {}
        if "lr" in conf_hp and "weight_decay" in conf_hp:
            w = {**w, "lr": float(conf_hp["lr"]), "weight_decay": float(conf_hp["weight_decay"])}
    conf = policy["confirm"]
    dl = perf_lock.get("dataloader") or {}
    nw = int(dl.get("num_workers", 0))
    pin = bool(dl.get("pin_memory", False))
    teacher_cache = _ROOT / "Docs/results/ms_sta_v6_e4_r4c_runs_canonical/teacher_logits_confirm_train.npz"
    if not teacher_cache.is_file():
        raise SystemExit("missing_teacher_logits_cache")
    for seed in conf["seeds"]:
        if int(seed) == 29:
            raise SystemExit("forbid_screening_seed_29_as_confirm_seed")
        out_dir = runs_root / "confirm" / f"seed{seed}"
        if (out_dir / "summary.json").is_file():
            print("skip_existing", out_dir)
            continue
        cmd = [
            str(py()),
            "-m",
            "src.training.ms_sta_e4_r4c_train",
            "--train-policy",
            str(args.train_policy),
            "--feature-root",
            str(feature_root),
            "--out-dir",
            str(out_dir),
            "--train-role",
            conf["train_role"],
            "--val-role",
            conf["val_role"],
            "--seed",
            str(seed),
            "--lr",
            str(w["lr"]),
            "--weight-decay",
            str(w["weight_decay"]),
            "--epochs",
            str(conf["epochs"]),
            "--batch-size",
            str(conf["batch_size"]),
            "--config-id",
            f"R1_S_A_d128_lr{w['lr']}_wd{w['weight_decay']}_confirm_s{seed}",
            "--num-workers",
            str(nw),
            "--teacher-logits-cache",
            str(teacher_cache),
        ]
        if pin:
            cmd.append("--pin-memory")
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=_ROOT)
    print("CONFIRM_PHASE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
