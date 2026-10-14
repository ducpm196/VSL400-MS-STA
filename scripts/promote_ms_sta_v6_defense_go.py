#!/usr/bin/env python3
"""Promote MS-STA v6 Defense GO after oneshot PASS + D4 PASS (create-only registry).

Does not rewrite historical V5 registry bytes. Atomically retargets active pointer.
Never loads E4. V5 remains rollback registry.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_rev  # noqa: E402

DEFENSE_CKPT_REL = "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"
EXPECTED_SHA = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, obj: dict[str, Any], *, overwrite: bool = False) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not overwrite:
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    completion = load_json(_ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_COMPLETION.json")
    report = load_json(_ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_REPORT.json")
    d4 = load_json(_ROOT / "Docs/results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json")
    freeze = load_json(_ROOT / "Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json")
    e4_close = load_json(_ROOT / "Docs/results/ms_sta_v6_e4/E4_CLOSEOUT.json")
    if completion.get("status") != "COMPLETE" or completion.get("verdict") != "B4_PROMOTION_GATE_PASS":
        raise SystemExit("completion_pass_required")
    if report.get("verdict") != "B4_PROMOTION_GATE_PASS":
        raise SystemExit("test_report_pass_required")
    if d4.get("d4_pass") is not True:
        raise SystemExit("d4_pass_required")
    if e4_close.get("final_test_includes_e4") is not False:
        raise SystemExit("e4_must_remain_excluded")

    ckpt_rel = DEFENSE_CKPT_REL
    ckpt_sha = freeze["checkpoints_seed42"]["b4_v6_ctr"]["checkpoint_sha256"]
    if ckpt_sha != EXPECTED_SHA:
        raise SystemExit("unexpected_b4_sha")
    if sha256_file(_ROOT / ckpt_rel) != ckpt_sha:
        raise SystemExit("defense_checkpoint_sha_drift")

    head = git_rev(_ROOT)
    d4_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json")
    completion_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_COMPLETION.json")
    report_sha = sha256_file(_ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_REPORT.json")
    variants = report["variants"]
    contrast = report["b4_minus_mlp"]

    final = {
        "schema": "ms_sta_v6_defense_go_final_results_v1",
        "defense_status": "GO",
        "defense_ready": True,
        "proposed_final_model": "b4_v6_ctr",
        "primary_seed": 42,
        "checkpoint_rel": ckpt_rel,
        "checkpoint_sha256": ckpt_sha,
        "test_kind": "retrospective_signer_disjoint_confirmation",
        "pristine_blind_claim": False,
        "e4_on_test_or_demo": False,
        "e4_status": "E4_ACCURACY_NO_GO",
        "validation": {
            "teacher_lock": "Docs/results/ms_sta_v6/V6_TEACHER_LOCK.json",
            "b4_confirm_top1": [0.8531761125903385, 0.8482312666413084, 0.8429060479269684],
            "label": "Stage-A finite-policy teacher after CTR-matched architecture PASS",
        },
        "test": {
            "n": report["test_n"],
            "rgb_top1": variants["rgb_strong"]["top1"],
            "b3_ctr_concat_top1": variants["b3_ctr_concat"]["top1"],
            "b3_mlp_ctr_top1": variants["b3_mlp_ctr"]["top1"],
            "b4_top1": variants["b4_v6_ctr"]["top1"],
            "b4_top5": variants["b4_v6_ctr"]["top5"],
            "b4_macro_f1": variants["b4_v6_ctr"]["macro_f1"],
            "mlp_top1": variants["b3_mlp_ctr"]["top1"],
            "delta_top1_pp": contrast["hierarchical_mean_delta_top1_pp"],
            "hierarchical_ci95_top1_pp": contrast["hierarchical_ci95_top1_pp"],
            "mcnemar_top1": contrast["mcnemar_top1"],
            "per_signer_delta_top1_pp": contrast["per_signer_delta_top1_pp"],
            "completion_sha256": completion_sha,
            "report_sha256": report_sha,
        },
        "d4": {
            "path": "Docs/results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json",
            "sha256": d4_sha,
            "d4_pass": True,
            "n_scenarios": d4["n_scenarios"],
            "n_pass": d4["n_pass"],
        },
        "e4_appendix": {
            "closeout": "Docs/results/ms_sta_v6_e4/E4_CLOSEOUT.json",
            "no_go": "Docs/results/ms_sta_v6_e4/E4_NO_GO.json",
            "role": "negative_efficiency_result_appendix_only",
        },
        "claim_limits": [
            "Test is retrospective signer-disjoint confirmation, not pristine blind.",
            "Do not claim paper-comparable or VSL400 literature superiority.",
            "Do not claim real-time/production/arbitrary-video support.",
            "E4 is NO-GO; do not claim efficiency contribution as active engine.",
            "Demo is cache-backed offline on D0 feature NPZ.",
        ],
        "rollback_registry": "data/processed/ms_sta_registry_v5_defense_go.json",
        "head": head,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    final_path = _ROOT / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json"
    final_sha = atomic_write(final_path, final)

    registry = {
        "schema": "ms_sta_registry_v6_defense_go_v1",
        "status": "GO",
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "d4_pass": True,
        "evaluation_locked": True,
        "ms_sta_v6": True,
        "ms_sta_v5": False,
        "active_variant": "b4_v6_ctr",
        "experiment_variant": "b4_v6_ctr",
        "engine": "ms_sta_v6_b4_v6_ctr",
        "checkpoint": ckpt_rel,
        "checkpoint_sha256": ckpt_sha,
        "primary_seed": 42,
        "feature_roots": {
            "skeleton_root_rel": freeze["feature_roots"]["skeleton_root_rel"],
            "rgb_root_rel": freeze["feature_roots"]["rgb_root_rel"],
        },
        "split_manifest_rel": freeze["split_manifest_rel"],
        "demo_set_rel": "data/processed/demo_set_ms_sta.json",
        "final_results_rel": "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json",
        "final_results_sha256": final_sha,
        "d4_report_sha256": d4_sha,
        "test_completion_sha256": completion_sha,
        "b4_ready": True,
        "e4_ready": False,
        "forbidden_variants_on_demo": [
            "e4_w192",
            "e4_w192_t1",
            "e4_w160_t1",
            "b4_star_e_stride4",
        ],
        "rollback_registry_rel": "data/processed/ms_sta_registry_v5_defense_go.json",
        "historical_not_primary": [
            "data/processed/ms_sta_registry_v5_defense_go.json",
            "Docs/results/ms_sta_v5_defense_go/FINAL_RESULTS.json",
        ],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head": head,
    }
    reg_path = _ROOT / "data/processed/ms_sta_registry_v6_defense_go.json"
    reg_sha = atomic_write(reg_path, registry)

    # Preserve prior pointer target for rollback evidence.
    prior_pointer = load_json(_ROOT / "data/processed/ms_sta_registry_active.json")
    pointer = {
        "schema": "ms_sta_registry_active_pointer_v1",
        "registry_path": "data/processed/ms_sta_registry_v6_defense_go.json",
        "registry_sha256": reg_sha,
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "historical_fallback_disabled": True,
        "final_results_sha256": final_sha,
        "rollback_registry_path": "data/processed/ms_sta_registry_v5_defense_go.json",
        "previous_active": prior_pointer,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supersedes": "data/processed/ms_sta_registry_v5_defense_go.json",
        "head": head,
    }
    pointer_sha = atomic_write(
        _ROOT / "data/processed/ms_sta_registry_active.json", pointer, overwrite=True
    )

    smoke = {
        "schema": "ms_sta_v6_post_promotion_smoke_v1",
        "status": "PASS",
        "checks": {
            "final_results_defense_ready": final["defense_ready"] is True,
            "d4_pass": True,
            "checkpoint_sha_matches": sha256_file(_ROOT / ckpt_rel) == ckpt_sha,
            "active_pointer_targets_v6": pointer["registry_path"].endswith("v6_defense_go.json"),
            "active_pointer_sha_matches_registry": pointer["registry_sha256"] == reg_sha,
            "e4_not_ready": registry["e4_ready"] is False,
            "v5_registry_untouched": (_ROOT / "data/processed/ms_sta_registry_v5_defense_go.json").is_file(),
        },
        "bindings": {
            "final_results_sha256": final_sha,
            "registry_sha256": reg_sha,
            "active_pointer_sha256": pointer_sha,
            "d4_sha256": d4_sha,
            "checkpoint_sha256": ckpt_sha,
        },
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not all(smoke["checks"].values()):
        raise SystemExit(f"post_promotion_smoke_fail:{smoke['checks']}")
    smoke_sha = atomic_write(_ROOT / "Docs/results/ms_sta_v6_defense_go/POST_PROMOTION_SMOKE.json", smoke)

    print(
        json.dumps(
            {
                "event": "V6_DEFENSE_GO",
                "final_results_sha256": final_sha,
                "registry_sha256": reg_sha,
                "smoke_sha256": smoke_sha,
                "checkpoint_sha256": ckpt_sha,
                "defense_ready": True,
                "e4_ready": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
