"""Canonical MS-STA retrain-v3 policy helpers (Path B).

Pure functions: no training side effects. Gate3 warm-start is forbidden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

INIT_RANDOM = "random"
GATE3_INIT_PREFIX = "gate3_"

HPO_VARIANTS = ("b0", "b3", "b4")
CONFIRM_VARIANTS = ("b0", "b1", "b2", "b3", "b4", "b4e", "b5")
INHERIT_FROM_B4 = ("b1", "b2", "b4e", "b5")
SCREEN_SEED = 11
FULL_SEEDS = (11, 29)
CONFIRM_SEEDS = (42, 123, 456)
PRIMARY_SEED = 42

LEARNING_RATES = (1e-4, 3e-4, 6e-4, 1e-3)
WEIGHT_DECAYS = (0.01, 0.05)

PATH_B_SIGNERS = {
    "hpo_train": [
        "signer_1",
        "signer_2",
        "signer_3",
        "signer_4",
        "signer_6",
        "signer_7",
        "signer_8",
        "signer_9",
        "signer_10",
        "signer_11",
        "signer_12",
        "signer_13",
        "signer_14",
        "signer_15",
        "signer_16",
    ],
    "tune": ["signer_18", "signer_19", "signer_20"],
    "confirm_train": [
        "signer_1",
        "signer_2",
        "signer_3",
        "signer_4",
        "signer_6",
        "signer_7",
        "signer_8",
        "signer_9",
        "signer_10",
        "signer_11",
        "signer_12",
        "signer_13",
        "signer_14",
        "signer_15",
        "signer_16",
        "signer_18",
        "signer_19",
        "signer_20",
    ],
    "val": ["signer_21", "signer_22", "signer_23"],
    "test": ["signer_24", "signer_25", "signer_26", "signer_27", "signer_28"],
    "missing": ["signer_5", "signer_17"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids: list[str]) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_frozen_v3_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema") != "ms_sta_retrain_v3_policy_v1":
        raise ValueError(f"unexpected retrain-v3 schema: {policy.get('schema')!r}")
    if policy.get("status") != "FROZEN":
        raise ValueError("retrain-v3 policy status != FROZEN")
    if policy.get("path") != "B":
        raise ValueError(f"this helper locks Path B only; got path={policy.get('path')!r}")
    claim = str(policy.get("test_claim_label") or "")
    if "retrospective" not in claim.lower():
        raise ValueError("Path B requires retrospective test_claim_label")


def format_config_id(lr: float, weight_decay: float) -> str:
    lr_map = {1e-4: "1e-4", 3e-4: "3e-4", 6e-4: "6e-4", 1e-3: "1e-3"}
    wd_map = {0.01: "0.01", 0.05: "0.05"}
    lr_s = lr_map.get(float(lr))
    wd_s = wd_map.get(float(weight_decay))
    if lr_s is None or wd_s is None:
        raise ValueError(f"lr/wd outside frozen search space: lr={lr} wd={weight_decay}")
    return f"lr{lr_s}_wd{wd_s}"


def expand_search_space(policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    space = (policy or {}).get("hpo_search_space") or {}
    lrs = list(space.get("learning_rate") or LEARNING_RATES)
    wds = list(space.get("weight_decay") or WEIGHT_DECAYS)
    configs: list[dict[str, Any]] = []
    for lr in lrs:
        for wd in wds:
            cid = format_config_id(float(lr), float(wd))
            configs.append(
                {
                    "config_id": cid,
                    "lr": float(lr),
                    "weight_decay": float(wd),
                }
            )
    configs.sort(key=lambda c: c["config_id"])
    if len(configs) != 8:
        raise ValueError(f"expected 8 HPO configs, got {len(configs)}")
    return configs


def hpo_screen_plan(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """24 screen trials: B0/B3/B4 × 8 configs × seed 11."""
    cfgs = expand_search_space(policy)
    seed = int(((policy.get("hpo") or {}).get("screen_seed") or SCREEN_SEED))
    epochs = int(((policy.get("hpo") or {}).get("screen_epochs") or 12))
    plan = []
    for variant in HPO_VARIANTS:
        for cfg in cfgs:
            plan.append(
                {
                    "phase": "hpo_screen",
                    "variant": variant,
                    "config_id": cfg["config_id"],
                    "lr": cfg["lr"],
                    "weight_decay": cfg["weight_decay"],
                    "seed": seed,
                    "epochs": epochs,
                    "train_role": "hpo_train",
                    "val_role": "tune",
                    "skel_temporal_stride": 1,
                }
            )
    if len(plan) != 24:
        raise ValueError(f"expected 24 screen trials, got {len(plan)}")
    return plan


def hpo_full_plan(
    policy: dict[str, Any],
    *,
    top_configs_by_variant: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """12 full-budget trials: top-2 configs × seeds {11,29} per HPO variant."""
    cfg_index = {c["config_id"]: c for c in expand_search_space(policy)}
    seeds = list(((policy.get("hpo") or {}).get("full_seeds") or FULL_SEEDS))
    epochs = int(((policy.get("hpo") or {}).get("full_epochs") or 16))
    plan: list[dict[str, Any]] = []
    for variant in HPO_VARIANTS:
        tops = top_configs_by_variant.get(variant) or []
        if len(tops) != 2:
            raise ValueError(f"{variant} needs exactly 2 promoted configs, got {tops}")
        for cid in tops:
            if cid not in cfg_index:
                raise ValueError(f"unknown promoted config_id={cid}")
            cfg = cfg_index[cid]
            for seed in seeds:
                plan.append(
                    {
                        "phase": "hpo_full",
                        "variant": variant,
                        "config_id": cid,
                        "lr": cfg["lr"],
                        "weight_decay": cfg["weight_decay"],
                        "seed": int(seed),
                        "epochs": epochs,
                        "train_role": "hpo_train",
                        "val_role": "tune",
                        "skel_temporal_stride": 1,
                    }
                )
    if len(plan) != 12:
        raise ValueError(f"expected 12 full trials, got {len(plan)}")
    return plan


def confirm_plan(
    policy: dict[str, Any],
    *,
    selected_config_by_variant: dict[str, str],
) -> list[dict[str, Any]]:
    """21 confirmation runs across B0/B1/B2/B3/B4/B4-E/B5 × seeds 42/123/456."""
    cfg_index = {c["config_id"]: c for c in expand_search_space(policy)}
    seeds = list(((policy.get("confirmation") or {}).get("seeds") or CONFIRM_SEEDS))
    epochs = int(((policy.get("confirmation") or {}).get("epochs") or 16))
    b4_cid = selected_config_by_variant.get("b4")
    if not b4_cid:
        raise ValueError("confirmation requires selected b4 config_id")
    plan: list[dict[str, Any]] = []
    for variant in CONFIRM_VARIANTS:
        if variant in HPO_VARIANTS:
            cid = selected_config_by_variant.get(variant)
            if not cid:
                raise ValueError(f"missing selected config for {variant}")
        else:
            cid = b4_cid
        cfg = cfg_index[cid]
        model_variant = "b4" if variant == "b4e" else variant
        stride = 4 if variant == "b4e" else 1
        for seed in seeds:
            plan.append(
                {
                    "phase": "confirm",
                    "variant": variant,
                    "model_variant": model_variant,
                    "config_id": cid,
                    "lr": cfg["lr"],
                    "weight_decay": cfg["weight_decay"],
                    "seed": int(seed),
                    "epochs": epochs,
                    "train_role": "confirm_train",
                    "val_role": "val",
                    "skel_temporal_stride": stride,
                }
            )
    if len(plan) != 21:
        raise ValueError(f"expected 21 confirmation runs, got {len(plan)}")
    return plan


def refuse_gate3_init(init_mode: str) -> None:
    mode = str(init_mode)
    if mode.startswith(GATE3_INIT_PREFIX) or mode != INIT_RANDOM:
        raise ValueError(
            f"retrain-v3 canonical mode refuses Gate3/non-random init ({mode!r}); "
            "all VSL modules must use fresh random initialization"
        )


def select_config_from_full_rows(
    rows: list[dict[str, Any]],
    *,
    required_seeds: tuple[int, ...] = FULL_SEEDS,
    tie_epsilon_pp: float = 0.25,
) -> str:
    """Select one config_id by mean tune Top-1; require every seed.

    Tie within ``tie_epsilon_pp``: higher mean Top-5, then lexicographically smaller
    config_id. Latency is intentionally excluded from HPO tie-break (LR/WD do not
    change inference architecture); latency is measured later on final B0/B3/B4/B4-E.
    """
    by_cfg: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        cid = str(row["config_id"])
        if row.get("best_val_top1") is None:
            raise ValueError(f"full-budget row missing best_val_top1 for config={cid}")
        seed = int(row["seed"])
        slot = by_cfg.setdefault(cid, {})
        if seed in slot:
            raise ValueError(f"duplicate full-budget row for config={cid} seed={seed}")
        slot[seed] = row

    if not by_cfg:
        raise ValueError("no completed full-budget rows to select from")

    required = tuple(int(s) for s in required_seeds)
    for cid, by_seed in by_cfg.items():
        have = set(by_seed)
        need = set(required)
        if have != need:
            raise ValueError(
                f"config {cid} incomplete full seeds: have={sorted(have)} need={sorted(need)}"
            )

    def mean_top1(cid: str) -> float:
        return sum(float(by_cfg[cid][s]["best_val_top1"]) for s in required) / len(required)

    def mean_top5(cid: str) -> float:
        vals = []
        for s in required:
            t5 = by_cfg[cid][s].get("best_val_top5")
            if t5 is None:
                raise ValueError(f"config {cid} seed={s} missing best_val_top5")
            vals.append(float(t5))
        return sum(vals) / len(vals)

    best_m1 = max(mean_top1(cid) for cid in by_cfg)
    tied = [cid for cid in by_cfg if abs(mean_top1(cid) - best_m1) <= float(tie_epsilon_pp)]
    tied.sort(key=lambda cid: (-mean_top5(cid), cid))
    return tied[0]


def promote_candidate_checkpoint(
    candidate_path: Path,
    final_path: Path,
    *,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Copy val-Top-1 candidate into a final artifact with source/final SHA lifecycle."""
    import torch

    if not candidate_path.is_file():
        raise FileNotFoundError(f"missing candidate checkpoint: {candidate_path}")
    source_sha = sha256_file(candidate_path)
    payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")
    if payload.get("checkpoint_status") not in (None, "candidate", "final_selected_checkpoint"):
        raise ValueError(f"unexpected checkpoint_status={payload.get('checkpoint_status')!r}")
    ts = now_iso or __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    payload["checkpoint_status"] = "final_selected_checkpoint"
    payload["lifecycle"] = {
        "source_candidate_path": str(candidate_path).replace("\\", "/"),
        "source_candidate_sha256": source_sha,
        "selection": str(payload.get("selection") or "val_top1"),
        "promoted_at": ts,
    }
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, final_path)
    final_sha = sha256_file(final_path)
    return {
        "candidate_path": str(candidate_path).replace("\\", "/"),
        "final_path": str(final_path).replace("\\", "/"),
        "source_candidate_sha256": source_sha,
        "final_sha256": final_sha,
        "checkpoint_status": "final_selected_checkpoint",
    }


def promote_top_k_from_screen(
    rows: list[dict[str, Any]],
    *,
    k: int = 2,
    expected_configs: int = 8,
) -> list[str]:
    by_cfg: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row["config_id"])
        if row.get("best_val_top1") is None:
            raise ValueError(f"screen row missing best_val_top1 for config={cid}")
        if cid in by_cfg:
            raise ValueError(f"duplicate screen row for config={cid}")
        by_cfg[cid] = row
    if len(by_cfg) != expected_configs:
        raise ValueError(
            f"screen incomplete for promotion: have {len(by_cfg)} configs, need {expected_configs}"
        )
    scored = sorted(
        by_cfg.items(),
        key=lambda kv: (
            -float(kv[1]["best_val_top1"]),
            -float(kv[1].get("best_val_top5") or 0.0),
            kv[0],
        ),
    )
    out = [cid for cid, _ in scored[:k]]
    if len(out) != k:
        raise ValueError(f"need {k} promoted configs, got {out}")
    return out


def efficiency_gate_inputs_ok(
    *,
    mean_delta_top1_pp: float,
    latency_reduction_pct: float,
    param_count_delta: int,
    score_elems_before: int = 4096,
    score_elems_after: int = 256,
) -> bool:
    reduction = 100.0 * (1.0 - (score_elems_after / score_elems_before))
    if abs(reduction - 93.75) > 1e-6:
        return False
    if mean_delta_top1_pp < -1.0:
        return False
    if latency_reduction_pct < 20.0:
        return False
    if param_count_delta > 0:
        return False
    return True


def evaluate_efficiency_gate(measurement: dict[str, Any]) -> dict[str, Any]:
    """Derive efficiency gate from a measurement artifact (not operator booleans)."""
    before = int(measurement.get("score_elems_before") or skeleton_attention_score_elements(64, 1))
    after = int(measurement.get("score_elems_after") or skeleton_attention_score_elements(64, 4))
    mean_delta = float(measurement["mean_delta_top1_pp"])
    latency_reduction = float(measurement["latency_reduction_pct"])
    param_delta = int(measurement["param_count_delta"])
    passed = efficiency_gate_inputs_ok(
        mean_delta_top1_pp=mean_delta,
        latency_reduction_pct=latency_reduction,
        param_count_delta=param_delta,
        score_elems_before=before,
        score_elems_after=after,
    )
    return {
        "passed": passed,
        "score_elems_before": before,
        "score_elems_after": after,
        "score_reduction_pct": 100.0 * (1.0 - after / before),
        "mean_delta_top1_pp": mean_delta,
        "latency_reduction_pct": latency_reduction,
        "param_count_delta": param_delta,
        "vram_b4_mb": measurement.get("vram_b4_mb"),
        "vram_b4e_mb": measurement.get("vram_b4e_mb"),
        "active_variant": active_variant_from_efficiency_gate(passed),
    }


def _mean_sample_std(xs: list[float]) -> tuple[float | None, float | None]:
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def evaluate_architecture_gates(
    confirm_rows: list[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
    seeds: tuple[int, ...] = CONFIRM_SEEDS,
) -> dict[str, Any]:
    """Evaluate frozen architecture gates from confirmation validation Top-1/Top-5.

    Does not open test. ``passed`` is false if any required clause fails.
    """
    arch = ((policy or {}).get("gates") or {}).get("architecture") or {}
    require_b4_gt_b0 = bool(arch.get("b4_top1_gt_b0_all_seeds", True))
    require_b4_gt_b3 = bool(arch.get("b4_top1_gt_b3_all_seeds", True))
    require_mean_pos = bool(arch.get("mean_delta_top1_positive_vs_b0_and_b3", True))
    top5_limit = float(arch.get("mean_top5_regression_limit_pp", -1.0))

    by: dict[str, dict[int, dict[str, Any]]] = {}
    for row in confirm_rows:
        if int(row.get("exit_code", 1)) != 0:
            continue
        by.setdefault(str(row["variant"]), {})[int(row["seed"])] = row

    missing = []
    for v in ("b0", "b3", "b4"):
        for s in seeds:
            if s not in (by.get(v) or {}):
                missing.append(f"{v}/seed{s}")
    if missing:
        return {
            "passed": False,
            "error": f"missing confirm rows: {missing}",
            "clauses": {},
        }

    per_seed: dict[str, dict[str, float]] = {}
    deltas_b4_b0: list[float] = []
    deltas_b4_b3: list[float] = []
    deltas_top5_b4_b3: list[float] = []
    b4_gt_b0_flags: list[bool] = []
    b4_gt_b3_flags: list[bool] = []
    for s in seeds:
        b0 = float(by["b0"][s]["best_val_top1"])
        b3 = float(by["b3"][s]["best_val_top1"])
        b4 = float(by["b4"][s]["best_val_top1"])
        b3_5 = float(by["b3"][s].get("best_val_top5") or 0.0)
        b4_5 = float(by["b4"][s].get("best_val_top5") or 0.0)
        d40 = b4 - b0
        d43 = b4 - b3
        d5 = b4_5 - b3_5
        deltas_b4_b0.append(d40)
        deltas_b4_b3.append(d43)
        deltas_top5_b4_b3.append(d5)
        b4_gt_b0_flags.append(b4 > b0)
        b4_gt_b3_flags.append(b4 > b3)
        per_seed[str(s)] = {
            "b0_top1": b0,
            "b3_top1": b3,
            "b4_top1": b4,
            "b3_top5": b3_5,
            "b4_top5": b4_5,
            "b4_minus_b0": d40,
            "b4_minus_b3": d43,
            "b4_minus_b3_top5": d5,
        }

    mean_d40, _ = _mean_sample_std(deltas_b4_b0)
    mean_d43, _ = _mean_sample_std(deltas_b4_b3)
    mean_d5, _ = _mean_sample_std(deltas_top5_b4_b3)
    assert mean_d40 is not None and mean_d43 is not None and mean_d5 is not None

    clause_b4_gt_b0 = all(b4_gt_b0_flags)
    clause_b4_gt_b3 = all(b4_gt_b3_flags)
    clause_mean_pos = (mean_d40 > 0.0) and (mean_d43 > 0.0)
    clause_top5 = mean_d5 >= top5_limit

    required = {
        "b4_top1_gt_b0_all_seeds": (not require_b4_gt_b0) or clause_b4_gt_b0,
        "b4_top1_gt_b3_all_seeds": (not require_b4_gt_b3) or clause_b4_gt_b3,
        "mean_delta_top1_positive_vs_b0_and_b3": (not require_mean_pos) or clause_mean_pos,
        "mean_top5_regression_limit_pp": clause_top5,
    }
    passed = all(required.values())

    b4e_deltas: list[float] = []
    if all(s in (by.get("b4e") or {}) for s in seeds):
        for s in seeds:
            b4e_deltas.append(
                float(by["b4e"][s]["best_val_top1"]) - float(by["b4"][s]["best_val_top1"])
            )
    mean_b4e, _ = _mean_sample_std(b4e_deltas) if b4e_deltas else (None, None)

    return {
        "passed": passed,
        "clauses": {
            "b4_top1_gt_b0_all_seeds": clause_b4_gt_b0,
            "b4_top1_gt_b3_all_seeds": clause_b4_gt_b3,
            "mean_delta_top1_positive_vs_b0_and_b3": clause_mean_pos,
            "mean_top5_regression_vs_b3_pp": mean_d5,
            "mean_top5_regression_limit_pp": top5_limit,
            "mean_top5_ok": clause_top5,
        },
        "required_results": required,
        "per_seed": per_seed,
        "mean_b4_minus_b0_pp": mean_d40,
        "mean_b4_minus_b3_pp": mean_d43,
        "mean_b4e_minus_b4_pp": mean_b4e,
        "seeds": list(seeds),
    }


def active_variant_from_efficiency_gate(passed: bool) -> str:
    return "b4e" if passed else "b4"


def run_key(trial: dict[str, Any]) -> str:
    variant = trial["variant"]
    return (
        f"{trial['phase']}__{variant}__{trial['config_id']}__seed{trial['seed']}"
        f"__stride{trial.get('skel_temporal_stride', 1)}"
    )


def model_variant_name(variant: str) -> str:
    return "b4" if variant == "b4e" else variant


def final_stem_from_candidate_stem(stem: str) -> str:
    return f"{stem}__final"


def require_complete_phase_rows(
    rows: list[dict[str, Any]],
    *,
    expected_n: int,
    phase: str,
) -> None:
    ok = [r for r in rows if int(r.get("exit_code", 1)) == 0 and r.get("best_val_top1") is not None]
    if len(ok) != expected_n:
        raise ValueError(f"{phase} incomplete: have {len(ok)} ok rows, need {expected_n}")
    keys = [r.get("run_key") for r in ok]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{phase} has duplicate run_key entries")


def require_homogeneous_git_revision(
    rows: list[dict[str, Any]],
    *,
    expected_revision: str,
    phase: str,
) -> None:
    """Fail-closed: every successful trial must share the campaign code-lock revision."""
    if not expected_revision:
        raise ValueError(f"{phase} missing expected git_revision")
    ok = [r for r in rows if int(r.get("exit_code", 1)) == 0]
    if not ok:
        raise ValueError(f"{phase} has no successful rows to verify git_revision")
    for row in ok:
        rev = row.get("git_revision")
        if not rev:
            raise ValueError(f"{phase} row missing git_revision: {row.get('run_key')}")
        if rev != expected_revision:
            raise ValueError(
                f"{phase} git_revision drift for {row.get('run_key')}: "
                f"{rev} != {expected_revision}"
            )


def require_hpo_candidate_lifecycle(rows: list[dict[str, Any]], *, phase: str) -> None:
    """HPO checkpoints remain disposable candidates; never final_selected_checkpoint."""
    for row in rows:
        status = row.get("checkpoint_status") or "candidate"
        if status == "final_selected_checkpoint":
            raise ValueError(
                f"{phase} must keep candidate lifecycle; got final_selected_checkpoint for {row.get('run_key')}"
            )
        if row.get("final_checkpoint"):
            raise ValueError(f"{phase} must not emit final_checkpoint for {row.get('run_key')}")


def require_confirm_final_lifecycle(rows: list[dict[str, Any]]) -> None:
    """Every confirmation run must promote a final_selected_checkpoint artifact."""
    for row in rows:
        if row.get("checkpoint_status") != "final_selected_checkpoint":
            raise ValueError(
                f"confirm missing final_selected_checkpoint for {row.get('run_key')}: "
                f"{row.get('checkpoint_status')!r}"
            )
        if not row.get("final_checkpoint") or not row.get("final_checkpoint_sha256"):
            raise ValueError(f"confirm missing final path/sha for {row.get('run_key')}")


def verify_test_matrix_artifacts(status_dir: Path, *, expected_n: int = 6165) -> dict[str, Any]:
    """Fail-closed verification of evaluator outputs before marking TEST COMPLETED."""
    from src.training.retrain_v3_stats import REQUIRED_TEST_VARIANTS, require_exact_variants

    matrix_path = Path(status_dir) / "TEST_MATRIX.json"
    rows_path = Path(status_dir) / "TEST_PER_SAMPLE.jsonl"
    if not matrix_path.is_file():
        raise ValueError(f"missing {matrix_path}")
    if not rows_path.is_file():
        raise ValueError(f"missing {rows_path}")
    tm = json.loads(matrix_path.read_text(encoding="utf-8"))
    require_exact_variants(set(tm.get("variants_present") or []), REQUIRED_TEST_VARIANTS)
    if set(tm.get("required_variants") or []) != set(REQUIRED_TEST_VARIANTS):
        raise ValueError("TEST_MATRIX required_variants mismatch")
    if int(tm.get("n_test") or -1) != int(expected_n):
        raise ValueError(f"TEST_MATRIX n_test={tm.get('n_test')} != {expected_n}")
    if int(tm.get("common_pair_n") or -1) != int(expected_n):
        raise ValueError(f"TEST_MATRIX common_pair_n={tm.get('common_pair_n')} != {expected_n}")
    for key in (
        "mcnemar",
        "holm",
        "b4e_vs_b4_noninferiority",
        "hypothesis_decisions",
        "variants",
        "paired_deltas",
    ):
        if key not in tm:
            raise ValueError(f"TEST_MATRIX missing {key}")
    variants = tm["variants"] or {}
    require_exact_variants(set(variants), REQUIRED_TEST_VARIANTS)
    for v in REQUIRED_TEST_VARIANTS:
        metrics = variants[v] or {}
        for mk in ("top1", "top5", "macro_f1", "per_signer", "worst_signer", "bootstrap_ci_top1"):
            if mk not in metrics:
                raise ValueError(f"TEST_MATRIX variant {v} missing {mk}")
    ni = tm.get("b4e_vs_b4_noninferiority") or {}
    if "ci_low" not in ni or "decision_rule" not in ni:
        raise ValueError("b4e_vs_b4_noninferiority must use paired CI decision_rule")
    paired = tm.get("paired_deltas") or {}
    for pk in ("top1_b4_minus_b0", "top1_b4_minus_b3", "top1_b4e_minus_b4", "top5_b4_minus_b0", "top5_b4e_minus_b4"):
        if pk not in paired:
            raise ValueError(f"TEST_MATRIX paired_deltas missing {pk}")
    n_rows = 0
    with rows_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_rows += 1
                row = json.loads(line)
                for k in (
                    "video_id",
                    "y",
                    "pred_b0",
                    "pred_b3",
                    "pred_b4",
                    "pred_b4e",
                    "top5_b0",
                    "top5_b3",
                    "top5_b4",
                    "top5_b4e",
                    "top5_hit_b0",
                    "top5_hit_b3",
                    "top5_hit_b4",
                    "top5_hit_b4e",
                ):
                    if k not in row:
                        raise ValueError(f"TEST_PER_SAMPLE missing {k}")
    if n_rows != int(expected_n):
        raise ValueError(f"TEST_PER_SAMPLE rows={n_rows} != {expected_n}")
    return {
        "n_test": int(expected_n),
        "variants": list(REQUIRED_TEST_VARIANTS),
        "matrix": str(matrix_path).replace("\\", "/"),
        "per_sample": str(rows_path).replace("\\", "/"),
    }


def build_demo_set_ms_sta_v3(
    rows: list[dict[str, Any]],
    *,
    active_variant: str,
    id_to_gloss: dict[int, str],
    test_matrix_sha256: str,
    policy_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Build defense demo scenarios from v3 TEST_PER_SAMPLE rows (reuse_metrics=false)."""
    if active_variant not in ("b4", "b4e"):
        raise ValueError(f"demo active_variant must be b4 or b4e, got {active_variant}")
    pred_k = f"pred_{active_variant}"
    top5_k = f"top5_{active_variant}"
    hit_k = f"top5_hit_{active_variant}"

    top1_correct = None
    top5_assist = None
    for row in rows:
        y = int(row["y"])
        pred_a = int(row[pred_k])
        pred_b0 = int(row["pred_b0"])
        hit_a = bool(row[hit_k])
        if top1_correct is None and pred_a == y and pred_b0 != y:
            top1_correct = row
        if top5_assist is None and pred_a != y and hit_a:
            top5_assist = row
        if top1_correct is not None and top5_assist is not None:
            break
    if top1_correct is None:
        for row in rows:
            if int(row[pred_k]) == int(row["y"]):
                top1_correct = row
                break
    if top5_assist is None:
        for row in rows:
            if int(row[pred_k]) != int(row["y"]) and bool(row[hit_k]):
                top5_assist = row
                break
    if top1_correct is None or top5_assist is None:
        raise ValueError("unable to select v3 demo scenarios from TEST_PER_SAMPLE")

    def _scenario_row(row: dict[str, Any], role: str) -> dict[str, Any]:
        y = int(row["y"])
        pred_a = int(row[pred_k])
        return {
            "video_id": str(row["video_id"]),
            "signer_id": row.get("signer_id"),
            "gold": id_to_gloss.get(y, str(y)),
            "gold_class_id": y,
            "pred_active_class_id": pred_a,
            "b0_pred": id_to_gloss.get(int(row["pred_b0"]), str(row["pred_b0"])),
            "b0_top1": bool(int(row["pred_b0"]) == y),
            "b0_top5_hit": bool(row["top5_hit_b0"]),
            f"{active_variant}_pred": id_to_gloss.get(pred_a, str(pred_a)),
            f"{active_variant}_top1": bool(pred_a == y),
            f"{active_variant}_top5_hit": bool(row[hit_k]),
            f"{active_variant}_top5": row[top5_k],
            "top5_active": [int(x) for x in row[top5_k]],
            "role": role,
        }

    return {
        "schema": "ms_sta_demo_set_v3",
        "gate": "D4",
        "retrain_v3": True,
        "reuse_metrics": False,
        "reuse_e2_demo": False,
        "active_variant": active_variant,
        "source": "logs/ms_sta_retrain_v3/TEST_PER_SAMPLE.jsonl",
        "test_matrix_sha256": test_matrix_sha256,
        "policy_sha256": policy_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "n": 2,
        "video_ids": [str(top1_correct["video_id"]), str(top5_assist["video_id"])],
        "scenarios": {
            "top1_correct": str(top1_correct["video_id"]),
            "top5_assist": str(top5_assist["video_id"]),
        },
        "scenario_rows": {
            "top1_correct": _scenario_row(top1_correct, "top1_correct"),
            "top5_assist": _scenario_row(top5_assist, "top5_assist"),
        },
        "note": "Built exclusively from retrain-v3 TEST_PER_SAMPLE; historical demo_set_ms_sta.json is not used.",
    }


def verify_d4_acceptance_artifacts(status_dir: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Fail-closed verification of D4 report + promote marker before COMPLETED."""
    repo = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    report_path = Path(status_dir) / "D4_ACCEPTANCE_REPORT.json"
    if not report_path.is_file():
        raise ValueError(f"missing {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("pass"):
        raise ValueError("D4_ACCEPTANCE_REPORT.pass is false")
    if not report.get("registry_updated"):
        raise ValueError("D4 did not update v3 registry after PASS")
    if report.get("historical_registry_mutated"):
        raise ValueError("D4 must not mutate historical registry")
    if not report.get("defense_checkpoint") or not report.get("defense_sha256"):
        raise ValueError("D4 missing defense checkpoint identity")
    if not report.get("registry_cutover_path"):
        raise ValueError("D4 missing registry cutover path")
    demo_rel = report.get("demo_set_v3") or "data/processed/demo_set_ms_sta_v3.json"
    demo_path = repo / str(demo_rel)
    if not demo_path.is_file():
        raise ValueError("missing demo_set_ms_sta_v3.json")
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    if demo.get("retrain_v3") is not True or demo.get("reuse_e2_demo") is not False:
        raise ValueError("demo_set_ms_sta_v3 lineage invalid")
    marker = Path(status_dir) / "REGISTRY_V3_PROMOTED.json"
    if not marker.is_file():
        raise ValueError("missing REGISTRY_V3_PROMOTED.json")
    cutover = repo / "data" / "processed" / "ms_sta_registry_active.json"
    if not cutover.is_file():
        raise ValueError("missing ms_sta_registry_active.json cutover pointer")
    scenarios = report.get("scenarios") or []
    if not scenarios or not all(bool(s.get("pass")) for s in scenarios):
        raise ValueError("D4 scenarios incomplete or not all PASS")
    names = {s.get("name") for s in scenarios}
    for required in (
        "health_candidate_vs_ready",
        "cache_backed_predict_clip_ms_sta",
        "payload_engine_identity_retrain_v3_final",
        "demo_set_v3_lineage",
        "active_stride_identity",
        "predict_identity_top1_correct",
        "predict_identity_top5_assist",
        "defense_exact_sha_equals_source",
        "health_defense_post_promotion",
        "defense_predict_identity_top1_correct",
    ):
        if required not in names:
            raise ValueError(f"D4 missing required scenario {required}")
    if not report.get("exact_byte_copy"):
        raise ValueError("D4 must report exact_byte_copy=true")
    if report.get("defense_sha256") != report.get("source_sha256"):
        raise ValueError("D4 defense_sha256 must equal source_sha256")
    exp = report.get("active_experiment_variant") or report.get("active_variant")
    if exp == "b4e" and "b4e_stride_identity" not in names:
        raise ValueError("D4 missing required scenario b4e_stride_identity")
    return {
        "n_scenarios": len(scenarios),
        "report": str(report_path).replace("\\", "/"),
        "active_variant": report.get("active_variant"),
    }


def skeleton_attention_score_elements(temporal_tokens: int = 64, stride: int = 1) -> int:
    """Self-attention score elements for Skeleton temporal tokens after optional pool."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    tokens = (temporal_tokens + stride - 1) // stride if stride > 1 else temporal_tokens
    return int(tokens * tokens)
