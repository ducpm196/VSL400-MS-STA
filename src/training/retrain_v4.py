"""Canonical MS-STA retrain-v4 policy helpers (Path B).

Pure functions: planning, selection, aux-loss halt, early-stop contract.
Training side effects live in the trainer/orchestrator only after FROZEN.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.training.retrain_v3 import (
    final_stem_from_candidate_stem,
    format_config_id,
    select_config_from_full_rows,
    sha256_file,
)

EVALUATION_LOCKED_SCHEMA = "ms_sta_retrain_v4_evaluation_locked_v1"

HPO_VARIANTS = ("b0", "b3", "b4", "b4r")
CONFIRM_VARIANTS = ("b0", "b3", "b4", "b4r")
TUNE_SEEDS = (11, 29)
CONFIRM_SEEDS = (42, 123, 456)
PRIMARY_SEED = 42
LEARNING_RATES = (1e-4, 3e-4, 6e-4)
WEIGHT_DECAYS = (0.01, 0.05)

DEFAULT_RESIDUAL_INIT = "zero_delta_head"
DRAFT_POLICY_REL = "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json"
FROZEN_POLICY_REL = "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY.json"


def load_policy(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def refuse_train_until_frozen(policy: dict[str, Any]) -> str | None:
    if policy.get("train_allowed") is not True:
        return "train_allowed is not true"
    if str(policy.get("status") or "") != "FROZEN":
        return f"policy status={policy.get('status')!r} is not FROZEN"
    schema = str(policy.get("schema") or "")
    if "draft" in schema.lower():
        return f"policy schema={schema!r} is still a draft schema"
    return None


def _require_hex64(value: object, *, label: str) -> str:
    s = str(value or "").strip().lower()
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"{label} must be a 64-hex sha256, got {value!r}")
    return s


def _require_rel_file(root: Path, rel: object, *, label: str) -> Path:
    if not rel:
        raise ValueError(f"{label} path missing")
    path = root / str(rel)
    if not path.is_file():
        raise ValueError(f"{label} file missing: {path}")
    return path


def require_frozen_v4_policy(policy: dict[str, Any], *, root: Path | None = None) -> None:
    """Fail-closed frozen contract: status + full SHA/runtime locks (no draft keys)."""
    refuse = refuse_train_until_frozen(policy)
    if refuse:
        raise ValueError(refuse)
    if policy.get("path") != "B":
        raise ValueError(f"retrain-v4 locks Path B only; got path={policy.get('path')!r}")
    schema = str(policy.get("schema") or "")
    if schema != "ms_sta_retrain_v4_policy_v1":
        raise ValueError(f"frozen schema must be ms_sta_retrain_v4_policy_v1, got {schema!r}")
    if "lineage_draft" in policy or "runtime_locks_draft" in policy:
        raise ValueError("frozen policy must not contain lineage_draft/runtime_locks_draft")
    banned = _frozen_lifecycle_banned_substrings(policy)
    if banned:
        raise ValueError(f"frozen policy contains banned lifecycle language: {banned[:5]}")
    orch_status = str((policy.get("orchestrator") or {}).get("implementation_status") or "")
    if orch_status != "frozen_hpo_ready":
        raise ValueError(
            "frozen policy requires orchestrator.implementation_status='frozen_hpo_ready', "
            f"got {orch_status!r}"
        )
    if policy.get("train_allowed") is not True or str(policy.get("status")) != "FROZEN":
        raise ValueError("frozen policy requires status=FROZEN and train_allowed=true")
    if "deferred_implementation" in policy:
        raise ValueError(
            "frozen policy must not keep deferred_implementation; use implementation landed map"
        )
    impl = policy.get("implementation")
    if not isinstance(impl, dict) or not impl:
        raise ValueError("frozen policy requires implementation landed map")
    claim = str(
        (policy.get("test_protocol") or {}).get("label")
        or policy.get("test_claim_label")
        or ""
    )
    if "retrospective" not in claim.lower():
        raise ValueError("Path B requires retrospective test_claim_label")

    lineage = policy.get("lineage")
    if not isinstance(lineage, dict) or not lineage:
        raise ValueError("frozen policy requires lineage (not lineage_draft)")
    data_locks = policy.get("data_locks")
    if not isinstance(data_locks, dict) or not data_locks:
        raise ValueError("frozen policy requires data_locks")
    runtime = policy.get("runtime_locks")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("frozen policy requires runtime_locks")
    code_lock = policy.get("code_lock")
    if not isinstance(code_lock, dict) or not code_lock:
        raise ValueError("frozen policy requires code_lock")

    for key in (
        "split_manifest",
        "split_manifest_sha256",
        "eligible_sample_manifest",
        "eligible_sample_manifest_sha256",
        "labels",
        "labels_sha256",
        "metadata",
        "metadata_sha256",
    ):
        if not lineage.get(key) and not data_locks.get(key):
            raise ValueError(f"frozen lineage/data_locks missing {key}")

    for key in (
        "split_manifest_sha256",
        "eligible_sample_manifest_sha256",
        "labels_sha256",
        "metadata_sha256",
    ):
        _require_hex64(lineage.get(key) or data_locks.get(key), label=key)

    for key in (
        "cuda_provenance_sha256",
        "pip_freeze_sha256",
        "backbone_checkpoint_sha256",
        "skel_manifest_sha256",
        "temporal_manifest_sha256",
    ):
        _require_hex64(runtime.get(key), label=f"runtime_locks.{key}")
    if not runtime.get("active_skel_dir") or not runtime.get("active_temporal_dir"):
        raise ValueError("runtime_locks requires active_skel_dir and active_temporal_dir")
    if str(runtime.get("device_expected") or "") != "cuda":
        raise ValueError("runtime_locks.device_expected must be cuda")

    require_code_lock_fields(code_lock)

    hpo = policy.get("hpo") or {}
    conf = policy.get("confirmation") or {}
    if int(hpo.get("runs_total") or 0) != 48:
        raise ValueError("hpo.runs_total must be 48")
    if int(conf.get("runs_total") or 0) != 12:
        raise ValueError("confirmation.runs_total must be 12")
    if conf.get("early_stopping_allowed") is not False:
        raise ValueError("confirmation.early_stopping_allowed must be false")
    tie_eps = hpo.get("tie_epsilon_pp", None)
    if tie_eps is None or float(tie_eps) != 0.0:
        raise ValueError("hpo.tie_epsilon_pp must be exactly 0.0 for best-config selection")
    variants = list(hpo.get("variants") or [])
    if variants != list(HPO_VARIANTS):
        raise ValueError(f"hpo.variants must be {HPO_VARIANTS}, got {variants}")

    if root is not None:
        validate_frozen_v4_bindings(policy, root=root)


def validate_frozen_v4_bindings(policy: dict[str, Any], *, root: Path) -> None:
    """Verify on-disk SHA bindings for a frozen policy (fail-closed)."""
    lineage = policy.get("lineage") or {}
    data_locks = policy.get("data_locks") or {}
    runtime = policy.get("runtime_locks") or {}

    checks = [
        (
            lineage.get("split_manifest") or data_locks.get("split_manifest"),
            lineage.get("split_manifest_sha256") or data_locks.get("split_manifest_sha256"),
            "split_manifest",
        ),
        (
            lineage.get("eligible_sample_manifest") or data_locks.get("eligible_sample_manifest"),
            lineage.get("eligible_sample_manifest_sha256")
            or data_locks.get("eligible_sample_manifest_sha256"),
            "eligible_sample_manifest",
        ),
        (
            lineage.get("labels") or data_locks.get("labels"),
            lineage.get("labels_sha256") or data_locks.get("labels_sha256"),
            "labels",
        ),
        (
            lineage.get("metadata") or data_locks.get("metadata"),
            lineage.get("metadata_sha256") or data_locks.get("metadata_sha256"),
            "metadata",
        ),
    ]
    for rel, expected, label in checks:
        path = _require_rel_file(root, rel, label=label)
        actual = sha256_file(path)
        if actual != str(expected):
            raise ValueError(f"{label} sha mismatch: {actual} != {expected}")

    for key in ("cuda_provenance",):
        rel = runtime.get(key)
        expected = runtime.get(f"{key}_sha256")
        if rel:
            path = _require_rel_file(root, rel, label=key)
            actual = sha256_file(path)
            if actual != str(expected):
                raise ValueError(f"{key} sha mismatch: {actual} != {expected}")

    code_lock = policy.get("code_lock") or {}
    expected_tree = str(code_lock.get("executable_tree_sha256") or "")
    actual_tree = executable_tree_sha256(root)
    if expected_tree and actual_tree != expected_tree:
        raise ValueError(
            f"executable_tree_sha256 mismatch: {actual_tree} != {expected_tree}"
        )


def lineage_block(policy: dict[str, Any]) -> dict[str, Any]:
    return dict(policy.get("lineage") or policy.get("lineage_draft") or {})


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
    expected = int(space.get("configs_per_variant") or len(lrs) * len(wds))
    if len(configs) != expected:
        raise ValueError(f"expected {expected} HPO configs per variant, got {len(configs)}")
    if len(configs) != 6:
        raise ValueError(f"v4 accuracy HPO requires exactly 6 configs/variant, got {len(configs)}")
    return configs


def residual_init_for_variant(variant: str, policy: dict[str, Any] | None = None) -> str:
    v = str(variant)
    if v != "b4r":
        return "none"
    arch = (policy or {}).get("architecture") or {}
    vctrl = ((policy or {}).get("variant_controls") or {}).get("b4r") or {}
    return str(
        vctrl.get("residual_init")
        or arch.get("default_residual_init")
        or DEFAULT_RESIDUAL_INIT
    )


def hpo_plan(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat accuracy HPO: 4 variants × 6 configs × 2 tune seeds = 48."""
    cfgs = expand_search_space(policy)
    hpo = policy.get("hpo") or {}
    seeds = list(hpo.get("tune_seeds") or TUNE_SEEDS)
    epochs = int(hpo.get("epochs") or (policy.get("hpo_search_space") or {}).get("epochs") or 16)
    patience = hpo.get("patience_early_stopping")
    if patience is None:
        patience = (policy.get("hpo_search_space") or {}).get("patience_early_stopping", 5)
    variants = list(hpo.get("variants") or HPO_VARIANTS)
    plan: list[dict[str, Any]] = []
    for variant in variants:
        for cfg in cfgs:
            for seed in seeds:
                plan.append(
                    {
                        "phase": "hpo",
                        "variant": variant,
                        "model_variant": variant,
                        "config_id": cfg["config_id"],
                        "lr": cfg["lr"],
                        "weight_decay": cfg["weight_decay"],
                        "seed": int(seed),
                        "epochs": epochs,
                        "patience": int(patience),
                        "early_stopping": True,
                        "train_role": "hpo_train",
                        "val_role": "tune",
                        "skel_temporal_stride": 1,
                        "residual_init": residual_init_for_variant(variant, policy),
                    }
                )
    expected = int(hpo.get("runs_total") or 48)
    if len(plan) != expected:
        raise ValueError(f"expected {expected} HPO runs, got {len(plan)}")
    return plan


def select_configs_per_variant(
    rows: list[dict[str, Any]],
    *,
    variants: tuple[str, ...] | list[str] = HPO_VARIANTS,
    required_seeds: tuple[int, ...] = TUNE_SEEDS,
    tie_epsilon_pp: float = 0.0,
) -> dict[str, str]:
    """Independent HPO selection per variant.

    Default ``tie_epsilon_pp=0.0``: higher mean Top-1 always wins; Top-5 / config_id
    only break exact Top-1 ties.
    """
    selected: dict[str, str] = {}
    for variant in variants:
        subset = [r for r in rows if str(r.get("variant")) == variant]
        selected[variant] = select_config_from_full_rows(
            subset,
            required_seeds=required_seeds,
            tie_epsilon_pp=float(tie_epsilon_pp),
        )
    return selected


def mean_tune_top1_for_config(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    config_id: str,
    required_seeds: tuple[int, ...] = TUNE_SEEDS,
) -> float:
    vals: list[float] = []
    by_seed = {
        int(r["seed"]): r
        for r in rows
        if str(r.get("variant")) == variant and str(r.get("config_id")) == config_id
    }
    for s in required_seeds:
        if s not in by_seed or by_seed[s].get("best_val_top1") is None:
            raise ValueError(f"missing tune row for {variant}/{config_id}/seed={s}")
        vals.append(float(by_seed[s]["best_val_top1"]))
    return sum(vals) / len(vals)


def aux_loss_amendment_required(
    rows: list[dict[str, Any]],
    selected: dict[str, str],
    *,
    required_seeds: tuple[int, ...] = TUNE_SEEDS,
) -> dict[str, Any]:
    """Halt-for-amendment trigger after finite HPO (no silent aux enable)."""
    b3_cid = selected.get("b3")
    b4r_cid = selected.get("b4r")
    if not b3_cid or not b4r_cid:
        raise ValueError("aux trigger requires selected b3 and b4r config_ids")
    b3_mean = mean_tune_top1_for_config(
        rows, variant="b3", config_id=b3_cid, required_seeds=required_seeds
    )
    b4r_mean = mean_tune_top1_for_config(
        rows, variant="b4r", config_id=b4r_cid, required_seeds=required_seeds
    )
    triggered = bool(b4r_mean <= b3_mean)
    return {
        "trigger": "mean_tune_top1_b4r_le_mean_tune_top1_b3_after_finite_hpo",
        "triggered": triggered,
        "b3_config_id": b3_cid,
        "b4r_config_id": b4r_cid,
        "b3_mean_tune_top1": b3_mean,
        "b4r_mean_tune_top1": b4r_mean,
        "action_if_triggered": "halt_campaign_for_policy_amendment",
        "forbid_silent_enable": True,
        "gradient_norms_are_diagnostic_only": True,
    }


def confirm_plan(
    policy: dict[str, Any],
    *,
    selected_config_by_variant: dict[str, str],
) -> list[dict[str, Any]]:
    """12 confirmation runs: B0/B3/B4/B4R × seeds 42/123/456; full 16 epochs."""
    cfg_index = {c["config_id"]: c for c in expand_search_space(policy)}
    conf = policy.get("confirmation") or {}
    seeds = list(conf.get("seeds") or CONFIRM_SEEDS)
    epochs = int(conf.get("epochs") or 16)
    variants = list(conf.get("variants") or CONFIRM_VARIANTS)
    early = conf.get("early_stopping_allowed")
    if early is None:
        early = False
    if early:
        raise ValueError("v4 confirmation forbids early stopping")
    plan: list[dict[str, Any]] = []
    for variant in variants:
        cid = selected_config_by_variant.get(variant)
        if not cid:
            raise ValueError(f"missing selected config for {variant}")
        if cid not in cfg_index:
            raise ValueError(f"unknown selected config_id={cid} for {variant}")
        cfg = cfg_index[cid]
        for seed in seeds:
            plan.append(
                {
                    "phase": "confirm",
                    "variant": variant,
                    "model_variant": variant,
                    "config_id": cid,
                    "lr": cfg["lr"],
                    "weight_decay": cfg["weight_decay"],
                    "seed": int(seed),
                    "epochs": epochs,
                    "patience": None,
                    "early_stopping": False,
                    "train_role": "confirm_train",
                    "val_role": "val",
                    "skel_temporal_stride": 1,
                    "residual_init": residual_init_for_variant(variant, policy),
                    "checkpoint_rule": conf.get(
                        "checkpoint_rule", "best_val_top1_within_full_16_epochs"
                    ),
                }
            )
    expected = int(conf.get("runs_total") or 12)
    if len(plan) != expected:
        raise ValueError(f"expected {expected} confirmation runs, got {len(plan)}")
    return plan


def should_early_stop(*, bad_epochs: int, patience: int | None, early_stopping: bool) -> bool:
    if not early_stopping:
        return False
    if patience is None or int(patience) <= 0:
        return False
    return int(bad_epochs) >= int(patience)


def build_optimizer_and_scheduler(
    parameters,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    torch_module,
):
    """AdamW + 1-epoch linear warmup then cosine (v4 training contract)."""
    opt = torch_module.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))
    spe = max(1, int(steps_per_epoch))
    warmup_steps = spe
    remain_steps = max(1, (int(epochs) - 1) * spe)
    warmup = torch_module.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch_module.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remain_steps)
    sched = torch_module.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    for pg in opt.param_groups:
        pg["lr"] = float(lr) * 0.01
    return opt, sched, {
        "optimizer": "AdamW",
        "scheduler": "linear_warmup_1_epoch_then_cosine",
        "warmup_steps": warmup_steps,
        "cosine_steps": remain_steps,
        "start_factor": 0.01,
    }


def checkpoint_hyperparameters(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    grad_clip: float | None,
    patience: int | None,
    early_stopping: bool,
    residual_init: str,
    skel_temporal_stride: int,
) -> dict[str, Any]:
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "label_smoothing": float(label_smoothing),
        "grad_clip": float(grad_clip) if grad_clip is not None else None,
        "patience": patience,
        "early_stopping": bool(early_stopping),
        "residual_init": str(residual_init),
        "skel_temporal_stride": int(skel_temporal_stride),
        "optimizer": "AdamW",
        "optimizer_lr_warmup_epochs": 1,
        "scheduler": "linear_warmup_1_epoch_then_cosine",
    }


def policy_sha256(path: Path) -> str:
    return sha256_file(path)


def protocol_seed_from_string(protocol: str) -> int:
    return int(hashlib.sha256(protocol.encode("utf-8")).hexdigest()[:8], 16)


def run_key(trial: dict[str, Any]) -> str:
    return (
        f"{trial['phase']}__{trial['variant']}__{trial['config_id']}__seed{trial['seed']}"
        f"__stride{trial.get('skel_temporal_stride', 1)}"
    )


def require_hex64(value: object, *, label: str) -> str:
    """Public alias for 64-hex SHA checks (same contract as ``_require_hex64``)."""
    return _require_hex64(value, label=label)


def require_exact_str(actual: object, expected: object, *, label: str) -> None:
    if actual is None or actual == "":
        raise ValueError(f"{label} missing (None/empty not allowed)")
    if str(actual) != str(expected):
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def verify_checkpoint_payload_identity(
    payload: dict[str, Any],
    *,
    phase: str,
    variant: str,
    config_id: str,
    seed: int,
    expected_policy_sha: str,
    expected_split_sha: str,
    expected_git_rev: str,
    require_final: bool,
) -> None:
    """Fail-closed identity checks on a loaded checkpoint payload."""
    require_exact_str(payload.get("phase"), phase, label="payload.phase")
    require_exact_str(
        payload.get("experiment_variant") or payload.get("variant"),
        variant,
        label="payload.variant",
    )
    require_exact_str(payload.get("config_id"), config_id, label="payload.config_id")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"payload.seed mismatch: {payload.get('seed')!r} != {seed!r}")
    require_exact_str(
        payload.get("retrain_v4_policy_sha256"),
        expected_policy_sha,
        label="payload.retrain_v4_policy_sha256",
    )
    require_exact_str(
        payload.get("manifest_sha256"),
        expected_split_sha,
        label="payload.manifest_sha256",
    )
    require_exact_str(
        payload.get("git_revision"),
        expected_git_rev,
        label="payload.git_revision",
    )
    if require_final:
        require_exact_str(
            payload.get("checkpoint_status"),
            "final_selected_checkpoint",
            label="payload.checkpoint_status",
        )
    else:
        status = str(payload.get("checkpoint_status") or "")
        if status not in ("candidate", "final_selected_checkpoint"):
            raise ValueError(f"unexpected payload.checkpoint_status={status!r}")


def trial_is_complete(
    *,
    phase: str,
    variant: str,
    config_id: str,
    seed: int,
    status_path: Path,
    ckpt_path: Path,
    final_path: Path | None,
    expected_policy_sha: str,
    expected_split_sha: str,
    expected_git_rev: str,
) -> bool:
    """Strict completeness: status+ckpt (+final for confirm) with required identity."""
    import json

    import torch

    # Required identity args — never optional / None (fail-closed).
    require_hex64(expected_policy_sha, label="expected_policy_sha")
    require_hex64(expected_split_sha, label="expected_split_sha")
    if not str(expected_git_rev or "").strip():
        raise ValueError("expected_git_rev missing (None/empty not allowed)")

    if not status_path.is_file() or not ckpt_path.is_file():
        return False
    if phase == "confirm" and (final_path is None or not final_path.is_file()):
        return False

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("best_val_top1") is None:
        return False

    require_exact_str(
        status.get("retrain_v4_policy_sha256"),
        expected_policy_sha,
        label="status.retrain_v4_policy_sha256",
    )
    require_exact_str(
        status.get("manifest_sha256"),
        expected_split_sha,
        label="status.manifest_sha256",
    )
    require_exact_str(
        status.get("git_revision"),
        expected_git_rev,
        label="status.git_revision",
    )
    require_exact_str(status.get("phase"), phase, label="status.phase")
    require_exact_str(
        status.get("experiment_variant") or status.get("variant"),
        variant,
        label="status.variant",
    )
    require_exact_str(status.get("config_id"), config_id, label="status.config_id")
    if int(status.get("seed", -1)) != int(seed):
        raise ValueError(f"status.seed mismatch: {status.get('seed')!r} != {seed!r}")

    cand_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(cand_payload, dict):
        raise ValueError(f"checkpoint payload must be dict: {ckpt_path}")
    verify_checkpoint_payload_identity(
        cand_payload,
        phase=phase,
        variant=variant,
        config_id=config_id,
        seed=seed,
        expected_policy_sha=expected_policy_sha,
        expected_split_sha=expected_split_sha,
        expected_git_rev=expected_git_rev,
        require_final=False,
    )
    ckpt_sha = sha256_file(ckpt_path)

    if phase == "confirm":
        assert final_path is not None
        require_exact_str(
            status.get("checkpoint_status"),
            "final_selected_checkpoint",
            label="status.checkpoint_status",
        )
        final_sha = sha256_file(final_path)
        require_exact_str(
            status.get("final_checkpoint_sha256"),
            final_sha,
            label="status.final_checkpoint_sha256",
        )
        require_exact_str(
            status.get("source_candidate_sha256"),
            ckpt_sha,
            label="status.source_candidate_sha256",
        )
        final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if not isinstance(final_payload, dict):
            raise ValueError(f"final payload must be dict: {final_path}")
        verify_checkpoint_payload_identity(
            final_payload,
            phase=phase,
            variant=variant,
            config_id=config_id,
            seed=seed,
            expected_policy_sha=expected_policy_sha,
            expected_split_sha=expected_split_sha,
            expected_git_rev=expected_git_rev,
            require_final=True,
        )
    return True


def executable_tree_sha256(root: Path) -> str:
    """Content hash of *tracked* ``src/`` + ``scripts/`` bytes via ``git ls-files``.

    Ignored/generated paths (e.g. ``*.pyc``) must not participate in the code-lock.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "src", "scripts"],
            cwd=root,
            text=False,
        )
    except Exception as exc:
        raise ValueError(f"git ls-files failed for executable tree hash: {exc}") from exc
    rels = [p.decode("utf-8", errors="surrogateescape") for p in out.split(b"\0") if p]
    files: list[str] = []
    for rel in rels:
        name = Path(rel).name
        if "RETRAIN_POLICY" in name:
            continue
        path = root / rel
        if not path.is_file():
            raise ValueError(f"tracked executable path missing on disk: {rel}")
        files.append(rel.replace("\\", "/"))
    if not files:
        raise ValueError("git ls-files returned no tracked src/scripts files")
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with (root / rel).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\n")
    return h.hexdigest()


def exact_id_list_sha256(ids: list[str]) -> str:
    """Stable SHA of an exact ordered ID list (newline-joined UTF-8)."""
    payload = "\n".join(str(x) for x in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_freeze_binding(
    binding: dict[str, Any],
    *,
    root: Path,
    policy_path: Path,
    policy_sha: str,
    head_rev: str,
) -> None:
    """Fail-closed check of external POLICY_FREEZE_BINDING.json (post-finalize)."""
    schema = str(binding.get("schema") or "")
    if schema != "ms_sta_retrain_v4_policy_freeze_binding_v1":
        raise ValueError(f"unexpected freeze binding schema: {schema!r}")
    require_exact_str(binding.get("policy_sha256"), policy_sha, label="binding.policy_sha256")
    require_hex64(binding.get("executable_tree_sha256"), label="binding.executable_tree_sha256")
    tree_actual = executable_tree_sha256(root)
    require_exact_str(
        binding.get("executable_tree_sha256"),
        tree_actual,
        label="binding.executable_tree_sha256 vs live tree",
    )
    parent = str(binding.get("executable_code_lock") or "").strip()
    if not parent or parent.startswith("pending"):
        raise ValueError("binding.executable_code_lock must be bound parent code revision")
    policy_commit = str(binding.get("policy_commit") or "").strip()
    if not policy_commit or policy_commit.startswith("pending"):
        raise ValueError(
            "binding.policy_commit still pending; run finalize after committing the policy"
        )
    import subprocess

    def _git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    try:
        _git("rev-parse", "--verify", f"{policy_commit}^{{commit}}")
    except Exception as exc:
        raise ValueError(f"binding.policy_commit not a git commit: {policy_commit}") from exc
    # Policy bytes at policy_commit must match the frozen SHA.
    rel = str(binding.get("policy_path") or "").replace("\\", "/")
    if not rel:
        rel = str(policy_path.resolve().relative_to(root.resolve())).replace("\\", "/")
    try:
        blob = subprocess.check_output(
            ["git", "show", f"{policy_commit}:{rel}"],
            cwd=root,
        )
    except Exception as exc:
        raise ValueError(
            f"cannot read {rel} at policy_commit={policy_commit}: {exc}"
        ) from exc
    blob_sha = hashlib.sha256(blob).hexdigest()
    if blob_sha != policy_sha:
        raise ValueError(
            f"policy bytes at {policy_commit}:{rel} sha={blob_sha} != binding {policy_sha}"
        )
    # HEAD must be a descendant of policy_commit (or equal).
    try:
        _git("merge-base", "--is-ancestor", policy_commit, head_rev)
    except Exception as exc:
        raise ValueError(
            f"HEAD={head_rev} is not a descendant of policy_commit={policy_commit}"
        ) from exc
    if tree_actual != str(binding.get("executable_tree_sha256")):
        raise ValueError("executable tree drifted after freeze binding")


def load_per_sample_manifest(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "ms_sta_retrain_v4_per_sample_manifest_v1":
        raise ValueError(f"unexpected per-sample manifest schema: {payload.get('schema')!r}")
    return payload


def require_per_sample_manifest_contract(
    manifest: dict[str, Any],
    *,
    expected_policy_sha: str,
    expected_split_sha: str,
    expected_git_rev: str,
    expected_variants: tuple[str, ...] = ("b3", "b4r"),
) -> dict[tuple[str, int], str]:
    """Validate per-sample manifest bindings; return final ckpt sha map."""
    require_exact_str(
        manifest.get("policy_sha256"), expected_policy_sha, label="manifest.policy_sha256"
    )
    require_exact_str(
        manifest.get("split_manifest_sha256"),
        expected_split_sha,
        label="manifest.split_manifest_sha256",
    )
    require_exact_str(
        manifest.get("git_revision"), expected_git_rev, label="manifest.git_revision"
    )
    require_hex64(manifest.get("exact_id_sha256"), label="manifest.exact_id_sha256")
    require_hex64(
        manifest.get("signer_metadata_sha256"), label="manifest.signer_metadata_sha256"
    )
    n_val = int(manifest.get("n_val") or 0)
    if n_val <= 0:
        raise ValueError("manifest.n_val must be > 0")
    jsonl = manifest.get("jsonl_sha256_by_variant_seed") or {}
    finals = manifest.get("final_checkpoint_sha256_by_variant_seed") or {}
    expected_keys = {f"{v}/seed{s}" for v in expected_variants for s in CONFIRM_SEEDS}
    if set(jsonl) != expected_keys:
        raise ValueError(f"manifest jsonl keys {sorted(jsonl)} != {sorted(expected_keys)}")
    if set(finals) != expected_keys:
        raise ValueError(f"manifest final keys {sorted(finals)} != {sorted(expected_keys)}")
    out: dict[tuple[str, int], str] = {}
    for key, sha in finals.items():
        require_hex64(sha, label=f"manifest.final[{key}]")
        require_hex64(jsonl[key], label=f"manifest.jsonl[{key}]")
        variant, seed_s = key.split("/seed", 1)
        out[(variant, int(seed_s))] = str(sha)
    return out


def confirm_final_checkpoint_rel(
    *,
    variant: str,
    config_id: str,
    seed: int,
    checkpoints_rel: str,
) -> str:
    """Repo-relative path for a confirm final checkpoint (matches orchestrator lock keys)."""
    stem = run_key(
        {
            "phase": "confirm",
            "variant": variant,
            "config_id": config_id,
            "seed": int(seed),
            "skel_temporal_stride": 1,
        }
    )
    base = str(checkpoints_rel).replace("\\", "/").rstrip("/")
    return f"{base}/{final_stem_from_candidate_stem(stem)}.pth"


def primary_seed_final_sha_by_variant(
    *,
    selected: dict[str, str],
    final_sha_by_path: dict[str, str],
    checkpoints_rel: str,
    variants: tuple[str, ...] = CONFIRM_VARIANTS,
    primary_seed: int = PRIMARY_SEED,
) -> dict[str, str]:
    """Derive exact {variant → primary-seed final SHA} from selected configs + lock map."""
    if set(selected) != set(variants):
        raise ValueError(f"selected variants {sorted(selected)} != {list(variants)}")
    out: dict[str, str] = {}
    for variant in variants:
        rel = confirm_final_checkpoint_rel(
            variant=variant,
            config_id=str(selected[variant]),
            seed=int(primary_seed),
            checkpoints_rel=checkpoints_rel,
        )
        sha = final_sha_by_path.get(rel)
        if not sha:
            raise ValueError(
                f"lock.final_checkpoint_sha256_by_path missing primary final for "
                f"{variant} at {rel}"
            )
        out[variant] = require_hex64(sha, label=f"primary final[{variant}]")
    return out


def require_evaluation_locked_contract(
    lock: dict[str, Any],
    *,
    policy_sha: str,
    split_sha: str,
    git_revision: str,
    report_sha: str,
    selected_sha: str,
    man_sha: str,
    freeze_binding_sha: str,
    final_shas: dict[str, str],
    jsonl_shas: dict[str, str],
    run_state_sha: str | None = None,
    primary_seed: int = PRIMARY_SEED,
) -> None:
    """Shared fail-closed contract for lock create and ``lock --resume``.

    ``run_state_sha`` when provided must match ``lock.run_state_sha256`` (pre-lock
    snapshot). When None, only require the lock field to be a 64-hex digest.
    """
    if lock.get("schema") != EVALUATION_LOCKED_SCHEMA:
        raise ValueError(f"lock.schema={lock.get('schema')!r} != {EVALUATION_LOCKED_SCHEMA!r}")
    if lock.get("architecture_gate_pass") is not True:
        raise ValueError("lock.architecture_gate_pass is not true")
    if int(lock.get("primary_seed") or -1) != int(primary_seed):
        raise ValueError(f"lock.primary_seed={lock.get('primary_seed')!r} != {primary_seed}")
    require_exact_str(lock.get("policy_sha256"), policy_sha, label="lock.policy_sha256")
    require_exact_str(
        lock.get("split_manifest_sha256"), split_sha, label="lock.split_manifest_sha256"
    )
    require_exact_str(lock.get("git_revision"), git_revision, label="lock.git_revision")
    require_exact_str(
        lock.get("confirm_validation_report_sha256"),
        report_sha,
        label="lock.confirm_validation_report_sha256",
    )
    require_exact_str(
        lock.get("selected_configs_sha256"),
        selected_sha,
        label="lock.selected_configs_sha256",
    )
    require_exact_str(
        lock.get("per_sample_manifest_sha256"),
        man_sha,
        label="lock.per_sample_manifest_sha256",
    )
    require_exact_str(
        lock.get("freeze_binding_sha256"),
        freeze_binding_sha,
        label="lock.freeze_binding_sha256",
    )
    lock_run_state = require_hex64(lock.get("run_state_sha256"), label="lock.run_state_sha256")
    if run_state_sha is not None:
        require_exact_str(lock_run_state, run_state_sha, label="lock.run_state_sha256")

    lock_finals = dict(lock.get("final_checkpoint_sha256_by_path") or {})
    if len(lock_finals) != 12:
        raise ValueError(f"lock finals n={len(lock_finals)} != 12")
    if set(lock_finals) != set(final_shas):
        raise ValueError(
            f"lock final paths mismatch: {sorted(lock_finals)} != {sorted(final_shas)}"
        )
    for path, sha in final_shas.items():
        require_hex64(sha, label=f"expected final[{path}]")
        require_exact_str(lock_finals.get(path), sha, label=f"lock.final[{path}]")

    lock_jsonl = dict(lock.get("per_sample_jsonl_sha256_by_variant_seed") or {})
    if len(lock_jsonl) != 6:
        raise ValueError(f"lock jsonl n={len(lock_jsonl)} != 6")
    if set(lock_jsonl) != set(jsonl_shas):
        raise ValueError(
            f"lock jsonl keys mismatch: {sorted(lock_jsonl)} != {sorted(jsonl_shas)}"
        )
    for key, sha in jsonl_shas.items():
        require_hex64(sha, label=f"expected jsonl[{key}]")
        require_exact_str(lock_jsonl.get(key), sha, label=f"lock.jsonl[{key}]")


def allowed_post_code_lock_paths() -> frozenset[str]:
    """Paths allowed to change after executable_code_lock without breaking the tree lock."""
    return frozenset(
        {
            "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY.json",
            "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json",
            "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json",
            "logs/ms_sta_retrain_v4/RUN_STATE.json",
            "logs/ms_sta_retrain_v4/EVALUATION_LOCKED.json",
            "logs/ms_sta_retrain_v4/CONFIRM_VALIDATION_REPORT.json",
            "logs/ms_sta_retrain_v4/SELECTED_CONFIGS.json",
            "logs/ms_sta_retrain_v4/per_sample_val/PER_SAMPLE_MANIFEST.json",
            "logs/ms_sta_retrain_v4/TEST_REPORT.json",
            "logs/ms_sta_retrain_v4/TEST_PER_SAMPLE.jsonl",
            "logs/ms_sta_retrain_v4/TEST_COMPLETION.json",
        }
    )


def require_code_lock_fields(code_lock: dict[str, Any]) -> None:
    """Freeze/runtime code-lock contract (no self-referential policy commit)."""
    parent = str(code_lock.get("executable_code_lock") or "").strip()
    if not parent or parent.startswith("pending"):
        raise ValueError("code_lock.executable_code_lock must be parent code git revision")
    if len(parent) < 7:
        raise ValueError("code_lock.executable_code_lock looks too short for a git revision")
    tree = str(code_lock.get("executable_tree_sha256") or "").strip()
    if tree.startswith("pending"):
        raise ValueError("code_lock.executable_tree_sha256 must be bound (not pending)")
    require_hex64(tree, label="code_lock.executable_tree_sha256")
    if code_lock.get("require_clean_worktree") is not True:
        raise ValueError("code_lock.require_clean_worktree must be true")
    # policy_commit is recorded outside the policy file after freeze; must not be required inside.
    if "policy_commit" in code_lock:
        raise ValueError(
            "code_lock.policy_commit must not live inside the policy; "
            "bind in POLICY_FREEZE_BINDING.json after the freeze commit"
        )


def verify_code_lock_against_repo(
    *,
    root: Path,
    code_lock: dict[str, Any],
    head_rev: str,
    changed_paths: list[str] | None = None,
) -> None:
    """Verify executable tree hash and that HEAD diffs from code lock are policy/evidence-only."""
    require_code_lock_fields(code_lock)
    expected_tree = str(code_lock["executable_tree_sha256"])
    actual_tree = executable_tree_sha256(root)
    if actual_tree != expected_tree:
        raise ValueError(
            f"runtime executable_tree_sha256 mismatch: {actual_tree} != {expected_tree}"
        )
    parent = str(code_lock["executable_code_lock"])
    if head_rev == parent:
        return
    if changed_paths is None:
        return
    allowed = allowed_post_code_lock_paths()
    bad = [p for p in changed_paths if p.replace("\\", "/") not in allowed]
    if bad:
        raise ValueError(
            f"HEAD={head_rev} diverges from executable_code_lock={parent} "
            f"with non-policy paths: {bad[:10]}"
        )


def evaluate_architecture_gates_v4(
    *,
    per_seed: dict[str, dict[str, float]],
    paired_ci: dict[str, Any],
    mean_delta_top5_pp: float,
    macro_f1_b3: float,
    macro_f1_b4r: float,
    mean_delta_top1_ge_pp: float = 1.0,
    mean_delta_top5_ge_pp: float = -1.0,
) -> dict[str, Any]:
    """Apply registered accuracy gates for B4R vs B3.

    ``mean_delta_top1_ge_pp`` defaults to +1.0 pp (practical \"outperform\" bar).
    Positive but sub-threshold deltas are \"improve\" language only — not PASS.
    """
    seeds = sorted(per_seed.keys(), key=lambda s: int(s))
    if [int(s) for s in seeds] != list(CONFIRM_SEEDS):
        raise ValueError(f"per_seed keys must be {CONFIRM_SEEDS}, got {seeds}")
    all_gt = all(float(per_seed[s]["b4r_top1"]) > float(per_seed[s]["b3_top1"]) for s in seeds)
    deltas = [float(per_seed[s]["b4r_top1"]) - float(per_seed[s]["b3_top1"]) for s in seeds]
    mean_delta = float(sum(deltas) / len(deltas))
    ci_ok = bool(float(paired_ci.get("ci_low", -1e9)) > 0.0)
    top5_ok = bool(float(mean_delta_top5_pp) >= float(mean_delta_top5_ge_pp))
    f1_ok = bool(float(macro_f1_b4r) > float(macro_f1_b3))
    mean_ok = bool(mean_delta >= float(mean_delta_top1_ge_pp))
    passed = bool(all_gt and mean_ok and ci_ok and top5_ok and f1_ok)
    if passed:
        claim = "outperform"
    elif mean_delta > 0.0:
        claim = "improve_not_outperform"
    else:
        claim = "no_gain"
    return {
        "b4r_top1_gt_b3_all_confirm_seeds": all_gt,
        "mean_delta_top1_gt_0": mean_delta > 0.0,
        "mean_delta_top1_ge_pp": mean_ok,
        "mean_delta_top1_threshold_pp": float(mean_delta_top1_ge_pp),
        "mean_delta_top1_pp": mean_delta,
        "paired_ci95_delta_top1_lower_bound_gt_0": ci_ok,
        "mean_delta_top5_ge_pp_-1": top5_ok,
        "mean_delta_top5_threshold_pp": float(mean_delta_top5_ge_pp),
        "mean_delta_top5_pp": float(mean_delta_top5_pp),
        "macro_f1_b4r_gt_b3": f1_ok,
        "macro_f1_b3": float(macro_f1_b3),
        "macro_f1_b4r": float(macro_f1_b4r),
        "claim_language": claim,
        "architecture_gate_pass": passed,
        "forbid_test_if_any_gate_fails": True,
    }


def promote_draft_to_freeze_candidate(
    draft: dict[str, Any],
    *,
    head_rev: str,
    executable_tree_sha: str,
) -> dict[str, Any]:
    """Deterministic draft → freeze-candidate promotion (no I/O).

    Promotes ``lineage_draft`` / ``runtime_locks_draft``, builds ``data_locks``,
    sets frozen schema/status/train_allowed, and binds executable code-lock fields.
    Caller must still run freeze helper under a clean worktree to write
    ``RETRAIN_POLICY.json``.
    """
    import copy

    if not isinstance(draft, dict) or not draft:
        raise ValueError("draft policy missing")
    out = copy.deepcopy(draft)
    if "lineage_draft" in out:
        lineage = dict(out.pop("lineage_draft") or {})
        out["lineage"] = lineage
    elif not isinstance(out.get("lineage"), dict) or not out.get("lineage"):
        raise ValueError("draft missing lineage_draft/lineage")
    if "runtime_locks_draft" in out:
        out["runtime_locks"] = dict(out.pop("runtime_locks_draft") or {})
    elif not isinstance(out.get("runtime_locks"), dict) or not out.get("runtime_locks"):
        raise ValueError("draft missing runtime_locks_draft/runtime_locks")

    lineage = dict(out["lineage"])
    data_locks = dict(out.get("data_locks") or {})
    for key in (
        "eligible_sample_manifest",
        "eligible_sample_manifest_sha256",
        "labels",
        "labels_sha256",
        "metadata",
        "metadata_sha256",
        "split_manifest",
        "split_manifest_sha256",
    ):
        if lineage.get(key) and not data_locks.get(key):
            data_locks[key] = lineage[key]
    out["data_locks"] = data_locks

    runtime = dict(out["runtime_locks"])
    for key in ("skel_manifest_sha256", "temporal_manifest_sha256"):
        if lineage.get(key) and not runtime.get(key):
            runtime[key] = lineage[key]
    runtime["sha_binding_status"] = "FROZEN"
    out["runtime_locks"] = runtime
    if lineage.get("sha_binding_status"):
        lineage["sha_binding_status"] = "FROZEN"
        out["lineage"] = lineage

    out["schema"] = "ms_sta_retrain_v4_policy_v1"
    out["status"] = "FROZEN"
    out["train_allowed"] = True
    out.pop("phase", None)

    code_lock = dict(out.get("code_lock") or {})
    code_lock.pop("policy_commit", None)
    tip = str(head_rev or "").strip()
    if not tip:
        raise ValueError("head_rev required for freeze candidate")
    tree = require_hex64(executable_tree_sha, label="executable_tree_sha256")
    code_lock["executable_code_lock"] = tip
    code_lock["executable_tree_sha256"] = tree
    code_lock["current_scaffold_tip"] = tip[:12] if len(tip) >= 12 else tip
    code_lock["require_clean_worktree"] = True
    out["code_lock"] = code_lock

    # Keep claim language thresholds explicit on the candidate.
    gates = dict(out.get("accuracy_gates") or {})
    gates.setdefault("mean_delta_top1_ge_pp", 1.0)
    gates.setdefault("mean_delta_top5_ge_pp", -1.0)
    gates.setdefault("claim_language", {})
    claim = dict(gates["claim_language"])
    claim.setdefault("outperform_requires_mean_delta_top1_ge_pp", 1.0)
    claim.setdefault(
        "improve_only_when_mean_delta_top1_in_open_interval_0_1",
        True,
    )
    claim.setdefault("forbid_outperform_wording_below_threshold", True)
    gates["claim_language"] = claim
    out["accuracy_gates"] = gates

    # Frozen candidate lifecycle SoT: ready for HPO after freeze+binding, not "still refused".
    orch = dict(out.get("orchestrator") or {})
    orch["implementation_status"] = "frozen_hpo_ready"
    out["orchestrator"] = orch

    out.pop("deferred_implementation", None)
    out["implementation"] = {
        "trainer_v4_residual_init_plumbing": True,
        "trainer_v4_b4r_epoch_diagnostics": True,
        "gradient_clipping_v4": True,
        "orchestrator_phase_launchers": True,
        "aligned_validation_evaluator_v4": True,
        "per_sample_exporter_v4": True,
        "frozen_policy_validator": True,
        "deterministic_policy_promotion": True,
        "immutable_policy_freeze": True,
        "status": "landed",
        "note": (
            "Capabilities required for the primary campaign are implemented. "
            "HPO opens only after promote→freeze→finalize binding on a clean worktree."
        ),
    }

    out["notes"] = [
        "Primary campaign: HPO 48 + Confirm 12; Lock manual; Test retrospective one-shot after accuracy PASS. warm_start_diagnostic deferred (runs_planned=0).",
        "Live SoT remains VALIDATION_NO_GO / B3 62.29% until v4 accuracy gate passes.",
        "Do not open retrain-v3 lock/test/D4.",
        "Historical E1-v2/E2/D4 remain historical_not_primary.",
        "Efficiency is campaign v4e only after accuracy PASS.",
        "B4 = historical attention fusion control; B4R = residual MS-STA proposal. Never overwrite B4 identity.",
        "v4 compares B4R vs freshly trained B3 under the same split/HPO/budget — not against the v3 B3 checkpoint.",
        "Claim language: 0 < mean ΔTop-1 < 1.0 pp => improve_not_outperform (architecture_gate_pass=false); outperform only when mean ΔTop-1 ≥ +1.0 pp and all other gates PASS.",
        "Gradient clipping applies for v3 or v4; HPO tie_epsilon_pp=0.0; evaluator requires exact val IDs.",
        "No auto-chain: after HPO 48 STOP for review; confirm only after HPO review; lock/test only after multi-seed gates PASS.",
    ]

    banned = _frozen_lifecycle_banned_substrings(out)
    if banned:
        raise ValueError(f"frozen candidate still contains banned lifecycle language: {banned[:5]}")
    if str((out.get("orchestrator") or {}).get("implementation_status")) != "frozen_hpo_ready":
        raise ValueError("frozen candidate requires orchestrator.implementation_status=frozen_hpo_ready")
    return out


_FROZEN_BANNED_SUBSTRINGS = (
    "still_refused",
    "train_still_refused",
    "train_allowed=false until",
    "train_allowed remains false",
    "freeze helper still required",
    "draft_not_frozen",
    "contracts_strengthened_train_still_refused",
    "lifecycle_contracts_landed_train_still_refused",
)


def _frozen_lifecycle_banned_substrings(obj: Any) -> list[str]:
    """Recursively collect banned draft/refuse phrases from a candidate/policy tree."""
    hits: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            low = node.lower()
            for ban in _FROZEN_BANNED_SUBSTRINGS:
                if ban in low and ban not in hits:
                    hits.append(ban)

    walk(obj)
    return hits
