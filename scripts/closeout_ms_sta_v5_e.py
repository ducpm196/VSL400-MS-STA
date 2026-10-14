#!/usr/bin/env python3
"""Create-only V5-E close-out + evidence index (B4-E supplementary, B4* primary).

Preserves V5_E_NO_GO bit-exact. Does not retrain, open test, or overwrite locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_porcelain, git_rev  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _mean_std_pct(vals: list[float]) -> tuple[float, float]:
    mean = 100.0 * statistics.mean(vals)
    std = 100.0 * statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def build_closeout(*, head: str) -> dict[str, Any]:
    no_go_path = _ROOT / "Docs/results/ms_sta_v5_e/V5_E_NO_GO.json"
    gate_path = _ROOT / "Docs/results/ms_sta_v5_e/ACCURACY_NONINFERIORITY_GATE.json"
    auth_path = _ROOT / "Docs/results/ms_sta_v5_e/V5_E_AUTHORIZATION.json"
    policy_path = _ROOT / "Docs/results/ms_sta_v5_e_policy/V5_E_POLICY.json"
    confirm_e = _ROOT / "Docs/results/ms_sta_v5_e_runs/CONFIRM_RESULTS.json"
    confirm_b4 = _ROOT / "Docs/results/ms_sta_v5_m0_runs/CONFIRM_RESULTS.json"
    lock_v2 = _ROOT / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json"
    hpo_sel = _ROOT / "Docs/results/ms_sta_v5_e_runs/HPO_SELECTED.json"
    hpo_lb = _ROOT / "Docs/results/ms_sta_v5_e_runs/HPO_LEADERBOARD.json"

    no_go = load_json(no_go_path)
    gate = load_json(gate_path)
    if no_go.get("status") != "V5_E_NO_GO" or gate.get("verdict") != "V5_E_NO_GO":
        raise SystemExit("closeout_requires_v5_e_no_go")

    e_trials = [t for t in load_json(confirm_e)["trials"] if t["variant"] == "b4_star_e_stride4"]
    b4_trials = [t for t in load_json(confirm_b4)["trials"] if t["variant"] == "b4_star_fixed"]
    if len(e_trials) != 3 or len(b4_trials) != 3:
        raise SystemExit("confirm_seed_count_mismatch")

    e_top1 = [float(t["best_val_top1"]) for t in e_trials]
    e_top5 = [float(t["best_val_top5"]) for t in e_trials]
    e_f1 = [float(t["best_val_macro_f1"]) for t in e_trials]
    b4_top1 = [float(t["best_val_top1"]) for t in b4_trials]
    b4_top5 = [float(t["best_val_top5"]) for t in b4_trials]
    b4_f1 = [float(t["best_val_macro_f1"]) for t in b4_trials]

    e_m1, e_s1 = _mean_std_pct(e_top1)
    b4_m1, b4_s1 = _mean_std_pct(b4_top1)
    e_m5, _ = _mean_std_pct(e_top5)
    b4_m5, _ = _mean_std_pct(b4_top5)
    e_mf, _ = _mean_std_pct(e_f1)
    b4_mf, _ = _mean_std_pct(b4_f1)

    failed = [k for k, v in gate["checks"].items() if not v]
    passed = [k for k, v in gate["checks"].items() if v]

    return {
        "schema": "ms_sta_v5_e_closeout_v1",
        "status": "CLOSED_SUPPLEMENTARY_NO_GO",
        "immutable": True,
        "scientific_identity": {
            "proposed_final_model": "b4_star_fixed",
            "supplementary_ablation": "b4_star_e_stride4",
            "b4_e_is_proposed_solution": False,
            "b4_e_replaces_b4": False,
            "b4_e_on_final_test": False,
            "b4_e_demo_engine": False,
            "project_no_go_from_e_no_go": False,
            "allowed_93_75_claim": (
                "Skeleton temporal-attention score-matrix element reduction only "
                "when matrix length is divisible by 4 or padded with invalid mask "
                "to a multiple of 4"
            ),
            "forbidden_claims": [
                "B4-E preserves accuracy / accuracy-equivalent",
                "B4-E is the proposed or final model",
                "93.75% total FLOPs / latency / VRAM / whole-system reduction",
                "V5_E_NO_GO overrides B4* Lock V2 canonical status",
            ],
        },
        "verdict": "V5_E_NO_GO",
        "retain_final_model": "b4_star_fixed",
        "bound_validation_lock_v2_sha256": sha256_file(lock_v2),
        "head_at_closeout": head,
        "artifacts": {
            "v5_e_no_go": {"rel": "Docs/results/ms_sta_v5_e/V5_E_NO_GO.json", "sha256": sha256_file(no_go_path)},
            "accuracy_non_inferiority_gate": {
                "rel": "Docs/results/ms_sta_v5_e/ACCURACY_NONINFERIORITY_GATE.json",
                "sha256": sha256_file(gate_path),
            },
            "authorization": {"rel": "Docs/results/ms_sta_v5_e/V5_E_AUTHORIZATION.json", "sha256": sha256_file(auth_path)},
            "policy": {"rel": "Docs/results/ms_sta_v5_e_policy/V5_E_POLICY.json", "sha256": sha256_file(policy_path)},
            "confirm_e": {"rel": "Docs/results/ms_sta_v5_e_runs/CONFIRM_RESULTS.json", "sha256": sha256_file(confirm_e)},
            "hpo_selected": {"rel": "Docs/results/ms_sta_v5_e_runs/HPO_SELECTED.json", "sha256": sha256_file(hpo_sel)},
            "hpo_leaderboard": {
                "rel": "Docs/results/ms_sta_v5_e_runs/HPO_LEADERBOARD.json",
                "sha256": sha256_file(hpo_lb),
            },
            "confirm_b4_reference": {
                "rel": "Docs/results/ms_sta_v5_m0_runs/CONFIRM_RESULTS.json",
                "sha256": sha256_file(confirm_b4),
            },
            "validation_lock_v2": {
                "rel": "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json",
                "sha256": sha256_file(lock_v2),
            },
        },
        "hpo_selected_hyperparams": {
            k: load_json(hpo_sel)["selected"]["b4_star_e_stride4"][k]
            for k in (
                "variant",
                "seed",
                "lr",
                "dropout",
                "weight_decay",
                "best_val_top1",
                "best_val_top5",
                "best_val_macro_f1",
                "tag",
                "out_dir",
            )
        },
        "b4_e_confirm": {
            "seeds": [int(t["seed"]) for t in e_trials],
            "top1": e_top1,
            "top5": e_top5,
            "macro_f1": e_f1,
            "mean_top1_pct": e_m1,
            "std_top1_pct": e_s1,
            "mean_top5_pct": e_m5,
            "mean_macro_f1_pct": e_mf,
            "hyperparams": {
                "lr": e_trials[0]["lr"],
                "dropout": e_trials[0]["dropout"],
                "weight_decay": e_trials[0]["weight_decay"],
            },
        },
        "b4_star_confirm_reference": {
            "seeds": [int(t["seed"]) for t in b4_trials],
            "top1": b4_top1,
            "top5": b4_top5,
            "macro_f1": b4_f1,
            "mean_top1_pct": b4_m1,
            "std_top1_pct": b4_s1,
            "mean_top5_pct": b4_m5,
            "mean_macro_f1_pct": b4_mf,
        },
        "non_inferiority": {
            "mean_delta_top1_pp": gate["mean_delta_top1_pp"],
            "hierarchical_ci95_top1_pp": gate["hierarchical_ci95_top1_pp"],
            "mean_delta_top5_pp": gate["mean_delta_top5_pp"],
            "hierarchical_ci95_top5_pp": gate["hierarchical_ci95_top5_pp"],
            "macro_f1_regression_pp": gate["macro_f1_regression_pp"],
            "per_signer_delta_top1_pp": gate["per_signer_delta_top1_pp"],
            "seed_deltas_top1": gate["deltas_top1"],
            "checks": gate["checks"],
            "checks_failed": failed,
            "checks_passed": passed,
            "accuracy_equivalence_claim": False,
        },
        "explicit_statements": [
            "B4-E is a supplementary validation-only efficiency ablation, not the proposed solution.",
            "B4-E is not the final model and must not be loaded on final test or used as demo engine.",
            "No accuracy-equivalence claim: non-inferiority vs B4* failed.",
            "93.75% applies only to Skeleton temporal-attention score elements under stated length conditions.",
            "V5_E_NO_GO does not make the thesis NO-GO; B4* Lock V2 remains the primary contribution.",
        ],
        "next": {
            "optional_diagnostic_efficiency_bench": True,
            "b4_star_oneshot_test_requires_separate_auth": True,
            "forbid_e2_distillation": True,
            "forbid_further_efficiency_architecture_search": True,
        },
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_evidence_index(*, closeout_sha: str, head: str) -> dict[str, Any]:
    def entry(rel: str) -> dict[str, str]:
        return {"rel": rel, "sha256": sha256_file(_ROOT / rel)}

    return {
        "schema": "ms_sta_v5_e_evidence_index_v1",
        "status": "CLOSED_SUPPLEMENTARY",
        "proposed_final_model": "b4_star_fixed",
        "supplementary_ablation": "b4_star_e_stride4",
        "verdict": "V5_E_NO_GO",
        "b4_canonical_lock": entry("Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json"),
        "closeout": {
            "rel": "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json",
            "sha256": closeout_sha,
        },
        "no_go": entry("Docs/results/ms_sta_v5_e/V5_E_NO_GO.json"),
        "accuracy_gate": entry("Docs/results/ms_sta_v5_e/ACCURACY_NONINFERIORITY_GATE.json"),
        "policy": entry("Docs/results/ms_sta_v5_e_policy/V5_E_POLICY.json"),
        "authorization_phase_a": entry("Docs/results/ms_sta_v5_e/V5_E_AUTHORIZATION.json"),
        "confirm_results": entry("Docs/results/ms_sta_v5_e_runs/CONFIRM_RESULTS.json"),
        "hpo_selected": entry("Docs/results/ms_sta_v5_e_runs/HPO_SELECTED.json"),
        "hpo_leaderboard": entry("Docs/results/ms_sta_v5_e_runs/HPO_LEADERBOARD.json"),
        "claim_policy": entry("Docs/results/ms_sta_v5_e/DEFENSE_CLAIM_POLICY.json"),
        "head_at_index": head,
        "forbid_final_test_for_b4_e": True,
        "b4_star_final_test_requires_oneshot_auth": True,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_claim_policy() -> dict[str, Any]:
    return {
        "schema": "ms_sta_v5_defense_claim_policy_v1",
        "status": "FROZEN",
        "main_thesis": {
            "proposed_model": "MS-STA B4* (b4_star_fixed)",
            "primary_contrast": "b4_star_fixed vs b3_mlp_roi_fixed",
            "validation_lock": "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json",
        },
        "supplement": {
            "name": "B4-E stride-4 (b4_star_e_stride4)",
            "role": "efficiency ablation / engineering trade-off analysis",
            "not_proposed_solution": True,
            "closeout": "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json",
        },
        "document_hierarchy": {
            "main_chapters_or_slides": [
                "Problem and local signer-disjoint protocol",
                "D0 RGB/Skeleton representation",
                "B4* block diagram",
                "Fair B0/B3/B3+MLP/B4* comparison",
                "Multi-seed validation",
                "Retrospective signer-disjoint test",
                "Demo and limitations",
                "Conclusion",
            ],
            "appendix_only": [
                "B4-E stride-4 formulation",
                "score-element equation (93.75% under stated conditions)",
                "validation accuracy trade-off vs B4*",
                "optional latency/VRAM diagnostic benchmark",
                "V5_E_NO_GO decision",
            ],
        },
        "reject_if": [
            "calls B4-E the final or proposed model",
            "mixes B4* and B4-E metrics without labeling roles",
            "states B4-E accuracy equivalence",
            "describes 93.75% as total FLOPs, latency, or VRAM",
            "lets B4-E NO-GO override B4* canonical status",
        ],
        "approved_conclusion_template": (
            "B4* is the proposed architecture and shows consistent validation gains. "
            "B4-E is a supplementary efficiency experiment: stride 4 reduces Skeleton "
            "temporal-attention score-matrix elements by 93.75% under stated length "
            "conditions, but did not meet accuracy non-inferiority; therefore B4-E is "
            "not selected as the final model."
        ),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    if root.resolve() != _ROOT.resolve():
        raise SystemExit("root_mismatch")

    porcelain = git_porcelain(_ROOT).strip()
    # Allow untracked E evidence + intentional .gitignore update for e_runs *.pt.
    dirty_tracked = []
    for line in porcelain.splitlines():
        if line.startswith("??") or line.startswith("!!"):
            continue
        # porcelain: XY PATH (XY may include leading space)
        path = line[3:].strip() if len(line) >= 4 and line[2] == " " else line.split(maxsplit=1)[-1]
        if path.replace("\\", "/") == ".gitignore":
            continue
        dirty_tracked.append(line)
    if dirty_tracked:
        raise SystemExit(f"dirty_tracked_forbidden:{dirty_tracked[:5]}")

    head = git_rev(_ROOT)
    claim_path = _ROOT / "Docs/results/ms_sta_v5_e/DEFENSE_CLAIM_POLICY.json"
    close_path = _ROOT / "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json"
    idx_path = _ROOT / "Docs/results/ms_sta_v5_e/EVIDENCE_INDEX.json"

    claim_sha = atomic_write(claim_path, build_claim_policy())
    closeout = build_closeout(head=head)
    close_sha = atomic_write(close_path, closeout)
    # rebuild closeout artifact hash entry after claim exists
    idx = build_evidence_index(closeout_sha=close_sha, head=head)
    idx["claim_policy"]["sha256"] = claim_sha
    idx_sha = atomic_write(idx_path, idx)

    print(
        json.dumps(
            {
                "event": "V5_E_CLOSEOUT_WRITTEN",
                "closeout_sha256": close_sha,
                "evidence_index_sha256": idx_sha,
                "claim_policy_sha256": claim_sha,
                "verdict": "V5_E_NO_GO",
                "proposed_final_model": "b4_star_fixed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
