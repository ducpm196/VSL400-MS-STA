#!/usr/bin/env python3
"""MS-STA v5 M0 orchestrator — HPO → confirm → accuracy gate (no final test)."""

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

from ms_sta_v5_d0_lib import git_porcelain, git_rev  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def hpo_grid(policy: dict[str, Any]) -> list[dict[str, Any]]:
    grid = policy["hpo"]["grid"]
    trials = []
    for variant, lr, dropout, wd in itertools.product(
        policy["variants"], grid["lr"], grid["dropout"], grid["weight_decay"]
    ):
        trials.append(
            {
                "variant": variant,
                "lr": lr,
                "dropout": dropout,
                "weight_decay": wd,
                "seed": 42,
            }
        )
    return trials


def run_trial(policy_path: Path, trial: dict[str, Any], phase: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "SUMMARY.json"
    if summary_path.is_file() and (out_dir / "best.pt").is_file() and (out_dir / "VAL_PREDS.json").is_file():
        return load_json(summary_path)
    cmd = [
        str(_ROOT / "_venvs" / "train_cuda" / "Scripts" / "python.exe"),
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
        "--out-dir",
        str(out_dir),
        "--device",
        "cuda",
    ]
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0 or not summary_path.is_file():
        raise RuntimeError(f"trial_failed:{out_dir}:rc={proc.returncode}")
    return load_json(summary_path)


def select_best_per_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        v = row["variant"]
        if v not in best or row["best_val_top1"] > best[v]["best_val_top1"]:
            best[v] = row
    return best


def _signer_top1(rows: list[dict[str, Any]]) -> dict[str, float]:
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(str(r.get("signer") or "unknown"), []).append(bool(r["correct"]))
    return {s: (sum(v) / max(len(v), 1)) for s, v in by.items()}


def paired_bootstrap_ci_lower(
    b3_correct: list[bool], b4_correct: list[bool], *, n_boot: int = 2000, seed: int = 0
) -> float:
    """Paired accuracy delta (B4-B3) bootstrap 95% CI lower bound in probability units."""
    import random as _random

    n = len(b3_correct)
    if n == 0 or n != len(b4_correct):
        raise ValueError("paired_ci_length_mismatch")
    deltas = [float(b4_correct[i]) - float(b3_correct[i]) for i in range(n)]
    rng = _random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        acc = 0.0
        for _j in range(n):
            acc += deltas[rng.randrange(n)]
        boots.append(acc / n)
    boots.sort()
    idx = int(0.025 * (len(boots) - 1))
    return boots[idx]


def evaluate_accuracy_gate(policy: dict[str, Any], confirm_rows: list[dict[str, Any]], runs_root: Path) -> dict[str, Any]:
    gates = policy["accuracy_gates"]
    by_var_seed: dict[str, dict[int, dict[str, Any]]] = {}
    for r in confirm_rows:
        by_var_seed.setdefault(r["variant"], {})[int(r["seed"])] = r
    seeds = [int(s) for s in policy["seeds_confirm"]]
    b3_top1 = []
    b4_top1 = []
    b3_top5 = []
    b4_top5 = []
    b3_f1 = []
    b4_f1 = []
    paired_ci_lowers = []
    signer_delta_mins = []
    for seed in seeds:
        b3 = by_var_seed["b3_star_fixed"][seed]
        b4 = by_var_seed["b4_star_fixed"][seed]
        b3_top1.append(float(b3["best_val_top1"]))
        b4_top1.append(float(b4["best_val_top1"]))
        b3_top5.append(float(b3.get("best_val_top5", 0.0)))
        b4_top5.append(float(b4.get("best_val_top5", 0.0)))
        b3_f1.append(float(b3.get("best_val_macro_f1", 0.0)))
        b4_f1.append(float(b4.get("best_val_macro_f1", 0.0)))
        p3 = load_json(_ROOT / b3["val_preds"])
        p4 = load_json(_ROOT / b4["val_preds"])
        m3 = {r["sample_id"]: r for r in p3["rows"]}
        m4 = {r["sample_id"]: r for r in p4["rows"]}
        if set(m3) != set(m4):
            raise RuntimeError(f"val_pred_id_mismatch_seed_{seed}")
        ids = sorted(m3)
        b3_c = [bool(m3[i]["correct"]) for i in ids]
        b4_c = [bool(m4[i]["correct"]) for i in ids]
        paired_ci_lowers.append(paired_bootstrap_ci_lower(b3_c, b4_c, seed=seed))
        s3 = _signer_top1(list(m3.values()))
        s4 = _signer_top1(list(m4.values()))
        common = sorted(set(s3) & set(s4))
        if not common:
            raise RuntimeError("signer_metrics_empty")
        signer_delta_mins.append(min(100.0 * (s4[s] - s3[s]) for s in common))

    deltas = [b4_top1[i] - b3_top1[i] for i in range(len(seeds))]
    mean_delta_pp = 100.0 * (sum(deltas) / len(deltas))
    mean_delta_top5_pp = 100.0 * (sum(b4_top5[i] - b3_top5[i] for i in range(len(seeds))) / len(seeds))
    mean_b3_f1 = sum(b3_f1) / len(b3_f1)
    mean_b4_f1 = sum(b4_f1) / len(b4_f1)
    every = all(d > 0 for d in deltas)
    ci_ok = all(x > 0.0 for x in paired_ci_lowers) if gates.get("paired_ci95_lower_gt_0") else True
    top5_ok = mean_delta_top5_pp >= float(gates["mean_delta_top5_pp_min"])
    f1_ok = (mean_b4_f1 > mean_b3_f1) if gates.get("macro_f1_b4_gt_b3") else True
    signer_ok = all(d >= float(gates["no_signer_catastrophe_pp"]) for d in signer_delta_mins)
    mean_ok = mean_delta_pp >= float(gates["mean_delta_top1_pp_min"])
    verdict = (
        "VALIDATION_PASS"
        if (every and mean_ok and ci_ok and top5_ok and f1_ok and signer_ok)
        else "VALIDATION_NO_GO"
    )
    return {
        "schema": "ms_sta_v5_m0_accuracy_gate_v1",
        "verdict": verdict,
        "seeds": seeds,
        "b3_top1": b3_top1,
        "b4_top1": b4_top1,
        "deltas": deltas,
        "mean_delta_top1_pp": mean_delta_pp,
        "b3_top5": b3_top5,
        "b4_top5": b4_top5,
        "mean_delta_top5_pp": mean_delta_top5_pp,
        "b3_macro_f1": b3_f1,
        "b4_macro_f1": b4_f1,
        "mean_b3_macro_f1": mean_b3_f1,
        "mean_b4_macro_f1": mean_b4_f1,
        "paired_ci95_lower": paired_ci_lowers,
        "signer_min_delta_top1_pp": signer_delta_mins,
        "b4_gt_b3_every_seed": every,
        "checks": {
            "b4_gt_b3_every_seed": every,
            "mean_delta_top1_pp": mean_ok,
            "paired_ci95_lower_gt_0": ci_ok,
            "mean_delta_top5_pp": top5_ok,
            "macro_f1_b4_gt_b3": f1_ok,
            "no_signer_catastrophe": signer_ok,
        },
        "metrics_complete": True,
        "notes": [
            "Gate uses final selected confirm checkpoints via VAL_PREDS.json.",
            "Final test remains forbidden.",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs_root": str(runs_root.relative_to(_ROOT)).replace("\\", "/"),
    }


def write_status(runs_root: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write(runs_root / "STATUS.json", payload)
    atomic_write(_ROOT / "logs" / "ms_sta_m0" / "STATUS.json", payload)


def run_efficiency_gate(policy: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    """Latency/VRAM/params for B4* reference. B4*-E stride-4 is not in Stage-A freeze."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "efficiency_requires_cuda: rerun with _venvs/train_cuda/Scripts/python.exe "
            "scripts/run_ms_sta_v5_m0.py --phase efficiency"
        )
    from src.models.ms_sta_v5 import build_ms_sta_v5, count_parameters
    from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json, load_split_role_ids
    from src.training.ms_sta_v5_train import load_metadata_by_id

    gate = load_json(runs_root / "ACCURACY_GATE.json")
    if gate.get("verdict") != "VALIDATION_PASS":
        raise SystemExit("efficiency_requires_accuracy_pass")
    confirm = load_json(runs_root / "CONFIRM_RESULTS.json")["trials"]
    b4 = [r for r in confirm if r["variant"] == "b4_star_fixed" and int(r["seed"]) == 42]
    if not b4:
        raise SystemExit("b4_seed42_missing")
    ckpt_path = _ROOT / b4[0]["best_checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ms_sta_v5(
        "b4_star_fixed",
        num_classes=400,
        d_model=int(policy["model"]["d_model"]),
        nhead=int(policy["model"]["nhead"]),
        temporal_layers=int(policy["model"]["temporal_layers"]),
        dropout=float(b4[0]["dropout"]),
        alpha_max=float(policy["model"]["alpha_max"]),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    split = load_json(_ROOT / policy["split_manifest_rel"])
    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
    val_ids = load_split_role_ids(split, "val")[: int(policy["confirm"]["batch_size"])]
    ds = VSL400MSSTAv5Dataset(
        val_ids,
        rgb_root=_ROOT / policy["feature_roots"]["rgb_root_rel"],
        skel_root=_ROOT / policy["feature_roots"]["skeleton_root_rel"],
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(str(x) for x in (split.get("test_ids") or [])),
    )
    batch = collate_v5([ds[i] for i in range(len(ds))])
    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    warmup = int(policy["efficiency"]["warmup"])
    trials = int(policy["efficiency"]["trials"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    p90 = times_sorted[int(0.9 * (len(times_sorted) - 1))]
    peak_vram = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    report = {
        "schema": "ms_sta_v5_m0_efficiency_gate_v1",
        "verdict": "EFFICIENCY_RECORDED_NO_93_75_CLAIM",
        "b4_star_reference": {
            "checkpoint": str(ckpt_path.relative_to(_ROOT)).replace("\\", "/"),
            "param_count": count_parameters(model),
            "batch_size": len(ds),
            "device": str(device),
            "median_latency_sec": median,
            "p90_latency_sec": p90,
            "peak_vram_bytes": peak_vram,
            "warmup": warmup,
            "trials": trials,
        },
        "b4_star_e_stride4": {
            "implemented": False,
            "reason": "Stage-A architecture freeze lists b4_star_fixed only; stride-4 E-path not authorized in frozen variants.",
        },
        "claim_attention_score_reduction_93_75": False,
        "claim_total_system_flops_reduction_93_75": False,
        "notes": [
            "Efficiency instrumentation recorded for reference B4*.",
            "Do not interpret as 93.75% total FLOPs or attention-score reduction.",
            "Paired accuracy non-inferiority vs E-path deferred until E-path is separately frozen.",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(runs_root / "EFFICIENCY_GATE.json", report)
    atomic_write(_ROOT / "Docs/results/ms_sta_v5_m0/EFFICIENCY_GATE.json", report)
    return report


def create_validation_lock(policy: dict[str, Any], runs_root: Path, head: str) -> dict[str, Any]:
    gate = load_json(runs_root / "ACCURACY_GATE.json")
    if gate.get("verdict") != "VALIDATION_PASS":
        raise SystemExit("lock_requires_accuracy_pass")
    if not (runs_root / "EFFICIENCY_GATE.json").is_file():
        raise SystemExit("lock_requires_efficiency_artifact")
    selected = load_json(runs_root / "HPO_SELECTED.json")
    confirm = load_json(runs_root / "CONFIRM_RESULTS.json")
    ckpt_map = {}
    for row in confirm["trials"]:
        key = f"{row['variant']}_seed{row['seed']}"
        ckpt_map[key] = {
            "best_checkpoint": row.get("best_checkpoint"),
            "val_preds": row.get("val_preds"),
            "best_val_top1": row.get("best_val_top1"),
            "best_val_top5": row.get("best_val_top5"),
            "best_val_macro_f1": row.get("best_val_macro_f1"),
            "lr": row.get("lr"),
            "dropout": row.get("dropout"),
            "weight_decay": row.get("weight_decay"),
        }
    lock = {
        "schema": "ms_sta_v5_m0_validation_lock_v1",
        "status": "LOCKED",
        "immutable": True,
        "forbid_final_test": True,
        "head": head,
        "policy_sha256": sha256_file(_ROOT / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json"),
        "accuracy_gate_sha256": sha256_file(runs_root / "ACCURACY_GATE.json"),
        "efficiency_gate_sha256": sha256_file(runs_root / "EFFICIENCY_GATE.json"),
        "hpo_selected": selected,
        "checkpoint_map": ckpt_map,
        "accuracy_verdict": gate["verdict"],
        "next_requires_separate_oneshot_test_auth": True,
        "oneshot_test_command_template": (
            "ONE_SHOT_TEST_AUTH required separately; do not invoke until authorized. "
            "Artifact: Docs/results/ms_sta_v5_m0/VALIDATION_LOCK.json"
        ),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    lock_path = runs_root / "VALIDATION_LOCK.json"
    lock_sha = atomic_write(lock_path, lock)
    lock["lock_sha256"] = lock_sha
    atomic_write(lock_path, lock)
    atomic_write(_ROOT / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK.json", lock)
    return lock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--policy",
        default="Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json",
    )
    ap.add_argument(
        "--phase",
        choices=["dry-plan", "hpo", "confirm", "gate", "efficiency", "lock", "all"],
        default="dry-plan",
    )
    ap.add_argument("--max-trials", type=int, default=0, help="Optional HPO cap for smoke (0=all)")
    args = ap.parse_args()

    policy_path = _ROOT / args.policy
    policy = load_json(policy_path)
    if policy.get("status") != "FROZEN":
        raise SystemExit("policy_not_frozen")
    auth = load_json(_ROOT / "Docs/results/ms_sta_v5_m0/M0_MASTER_CAMPAIGN_AUTHORIZATION.json")
    if auth.get("status") != "AUTHORIZED" or auth.get("allow_final_test") is not False:
        raise SystemExit("master_auth_invalid")

    head = git_rev(_ROOT)
    porcelain = git_porcelain(_ROOT).strip()
    runs_root = _ROOT / policy["runs_root_rel"]
    runs_root.mkdir(parents=True, exist_ok=True)

    trials = hpo_grid(policy)
    if args.max_trials and args.max_trials > 0:
        trials = trials[: int(args.max_trials)]

    plan = {
        "schema": "ms_sta_v5_m0_orchestrator_plan_v1",
        "phase": args.phase,
        "head": head,
        "porcelain_clean": porcelain == "",
        "n_hpo_trials": len(trials),
        "variants": policy["variants"],
        "confirm_seeds": policy["seeds_confirm"],
        "forbid_final_test": True,
        "policy_sha256": sha256_file(policy_path),
    }
    if args.phase == "dry-plan":
        print(json.dumps(plan, indent=2))
        return 0

    if porcelain:
        raise SystemExit(f"dirty_worktree_forbidden:{porcelain[:200]}")

    if args.phase in {"hpo", "all"}:
        hpo_rows = []
        for i, trial in enumerate(trials):
            tag = (
                f"{trial['variant']}_lr{trial['lr']}_do{trial['dropout']}_wd{trial['weight_decay']}_s{trial['seed']}"
            )
            out_dir = runs_root / "hpo" / tag
            write_status(
                runs_root,
                {
                    "phase": "hpo",
                    "i": i,
                    "n": len(trials),
                    "tag": tag,
                    "head": head,
                    "forbid_final_test": True,
                },
            )
            print(json.dumps({"event": "hpo_trial_start", "i": i, "n": len(trials), "tag": tag}), flush=True)
            summary = run_trial(policy_path, trial, "hpo", out_dir)
            summary["tag"] = tag
            summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
            hpo_rows.append(summary)
            atomic_write(runs_root / "HPO_LEADERBOARD.json", {"trials": hpo_rows, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        best = select_best_per_variant(hpo_rows)
        atomic_write(runs_root / "HPO_SELECTED.json", {"selected": best})
        print(json.dumps({"event": "hpo_complete", "selected": {k: v["best_val_top1"] for k, v in best.items()}}), flush=True)

    if args.phase in {"confirm", "all"}:
        selected_path = runs_root / "HPO_SELECTED.json"
        if not selected_path.is_file():
            raise SystemExit("hpo_selected_missing")
        selected = load_json(selected_path)["selected"]
        confirm_rows = []
        for variant, cfg in selected.items():
            for seed in policy["seeds_confirm"]:
                trial = {
                    "variant": variant,
                    "lr": cfg["lr"],
                    "dropout": cfg["dropout"],
                    "weight_decay": cfg["weight_decay"],
                    "seed": seed,
                }
                out_dir = runs_root / "confirm" / f"{variant}_seed{seed}"
                write_status(
                    runs_root,
                    {
                        "phase": "confirm",
                        "variant": variant,
                        "seed": seed,
                        "head": head,
                        "forbid_final_test": True,
                    },
                )
                print(json.dumps({"event": "confirm_start", "variant": variant, "seed": seed}), flush=True)
                summary = run_trial(policy_path, trial, "confirm", out_dir)
                summary["out_dir"] = str(out_dir.relative_to(_ROOT)).replace("\\", "/")
                confirm_rows.append(summary)
                atomic_write(runs_root / "CONFIRM_RESULTS.json", {"trials": confirm_rows})
        print(json.dumps({"event": "confirm_complete", "n": len(confirm_rows)}), flush=True)

    accuracy_pass = False
    if args.phase in {"gate", "all"}:
        confirm_path = runs_root / "CONFIRM_RESULTS.json"
        if not confirm_path.is_file():
            raise SystemExit("confirm_results_missing")
        rows = load_json(confirm_path)["trials"]
        report = evaluate_accuracy_gate(policy, rows, runs_root)
        atomic_write(runs_root / "ACCURACY_GATE.json", report)
        atomic_write(_ROOT / "Docs/results/ms_sta_v5_m0/ACCURACY_GATE.json", report)
        write_status(runs_root, {"phase": "gate", "verdict": report["verdict"], "head": head})
        print(json.dumps(report, indent=2), flush=True)
        if report["verdict"] != "VALIDATION_PASS":
            return 2
        accuracy_pass = True

    if args.phase in {"efficiency", "all"}:
        if args.phase == "efficiency" or accuracy_pass or (
            (runs_root / "ACCURACY_GATE.json").is_file()
            and load_json(runs_root / "ACCURACY_GATE.json").get("verdict") == "VALIDATION_PASS"
        ):
            eff = run_efficiency_gate(policy, runs_root)
            write_status(runs_root, {"phase": "efficiency", "verdict": eff["verdict"], "head": head})
            print(json.dumps(eff, indent=2), flush=True)
        elif args.phase == "all":
            return 2

    if args.phase in {"lock", "all"}:
        lock = create_validation_lock(policy, runs_root, head)
        write_status(
            runs_root,
            {
                "phase": "validation_lock",
                "status": "LOCKED",
                "lock_sha256": lock.get("lock_sha256"),
                "stop_before_final_test": True,
                "head": head,
            },
        )
        print(json.dumps({"event": "validation_lock_complete", "lock_sha256": lock.get("lock_sha256")}, indent=2), flush=True)
        print(
            json.dumps(
                {
                    "event": "STOP_BEFORE_FINAL_TEST",
                    "oneshot_requires": "separate signer-disjoint test authorization",
                    "lock_artifact": "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK.json",
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
