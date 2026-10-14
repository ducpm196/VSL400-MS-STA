"""Retrain-v4 orchestrator — executable plans; training refused until FROZEN.

Phases: hpo → (aux halt check) → confirm → lock → test.
Atomic RUN_STATE + _run_contract; resume requires matching policy/code/split SHAs.
Never auto-chains confirm → test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import final_stem_from_candidate_stem, sha256_file
from src.training.retrain_v4 import (
    CONFIRM_SEEDS,
    CONFIRM_VARIANTS,
    EVALUATION_LOCKED_SCHEMA,
    PRIMARY_SEED,
    aux_loss_amendment_required,
    confirm_plan,
    hpo_plan,
    lineage_block,
    load_per_sample_manifest,
    load_policy,
    refuse_train_until_frozen,
    require_evaluation_locked_contract,
    require_exact_str,
    require_freeze_binding,
    require_frozen_v4_policy,
    require_hex64,
    require_per_sample_manifest_contract,
    run_key,
    select_configs_per_variant,
    trial_is_complete,
    verify_code_lock_against_repo,
)

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json"
TRAIN_SCRIPT = _ROOT / "src/training/train_ms_sta_vsl400.py"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_ROOT.resolve()).as_posix()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _artifact_dirs(policy: dict) -> tuple[Path, Path, Path]:
    art = policy.get("artifact_dirs") or {}
    ckpt = _ROOT / str(art.get("checkpoints") or "data/processed/ms_sta_retrain_v4")
    status = _ROOT / str(art.get("status") or "logs/ms_sta_retrain_v4")
    fail = status / "failures"
    return ckpt, status, fail


def _stem(trial: dict) -> str:
    return run_key(trial)


def _trial_cmd(policy_path: Path, trial: dict, stem: str) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--retrain-v4-policy",
        str(policy_path),
        "--variant",
        str(trial["model_variant"]),
        "--experiment-variant",
        str(trial["variant"]),
        "--phase",
        str(trial["phase"]),
        "--config-id",
        str(trial["config_id"]),
        "--seed",
        str(trial["seed"]),
        "--epochs",
        str(trial["epochs"]),
        "--lr",
        str(trial["lr"]),
        "--weight-decay",
        str(trial["weight_decay"]),
        "--train-role",
        str(trial["train_role"]),
        "--val-role",
        str(trial["val_role"]),
        "--skel-temporal-stride",
        str(trial.get("skel_temporal_stride", 1)),
        "--artifact-stem",
        stem,
        "--grad-clip",
        "1.0",
    ]
    if trial.get("early_stopping"):
        cmd.extend(["--patience", str(int(trial["patience"]))])
    else:
        cmd.append("--disable-early-stopping")
    if trial.get("residual_init") and trial["residual_init"] != "none":
        cmd.extend(["--residual-init", str(trial["residual_init"])])
    return cmd


def assess_hpo_readiness(
    policy: dict,
    *,
    root: Path,
    policy_path: Path,
    binding_path: Path | None = None,
) -> dict:
    """Fail-closed readiness: policy_ready (static) vs hpo_ready (launch gates).

    ``hpo_ready`` is true only after ``require_frozen_v4_policy``, clean worktree +
    executable-tree code-lock preflight, and ``require_freeze_binding``.
    """
    reasons: list[str] = []
    refuse = refuse_train_until_frozen(policy)
    if refuse:
        return {
            "policy_ready": False,
            "hpo_ready": False,
            "train_refused": refuse,
            "reasons": [refuse],
        }

    try:
        require_frozen_v4_policy(policy, root=root)
    except ValueError as exc:
        msg = str(exc)
        return {
            "policy_ready": False,
            "hpo_ready": False,
            "train_refused": None,
            "reasons": [msg],
        }

    # Clean worktree + executable tree vs policy code_lock.
    code_lock = dict(policy.get("code_lock") or {})
    try:
        if code_lock.get("require_clean_worktree", True) and _git_dirty():
            raise ValueError("clean worktree required for hpo_ready")
        verify_code_lock_against_repo(
            root=root, code_lock=code_lock, head_rev=_git_rev()
        )
    except ValueError as exc:
        return {
            "policy_ready": True,
            "hpo_ready": False,
            "train_refused": None,
            "reasons": [f"code_lock_preflight: {exc}"],
        }

    bind = binding_path or (
        root / "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json"
    )
    if not bind.is_file():
        return {
            "policy_ready": True,
            "hpo_ready": False,
            "train_refused": None,
            "reasons": ["missing POLICY_FREEZE_BINDING.json"],
        }
    if not policy_path.is_file():
        return {
            "policy_ready": True,
            "hpo_ready": False,
            "train_refused": None,
            "reasons": [f"missing policy file for sha binding: {policy_path}"],
        }
    try:
        require_freeze_binding(
            json.loads(bind.read_text(encoding="utf-8")),
            root=root,
            policy_path=policy_path,
            policy_sha=sha256_file(policy_path),
            head_rev=_git_rev(),
        )
    except ValueError as exc:
        return {
            "policy_ready": True,
            "hpo_ready": False,
            "train_refused": None,
            "reasons": [f"freeze_binding: {exc}"],
        }

    return {
        "policy_ready": True,
        "hpo_ready": True,
        "train_refused": None,
        "reasons": [],
    }


def dry_plan(
    policy: dict,
    *,
    root: Path | None = None,
    policy_path: Path | None = None,
    binding_path: Path | None = None,
) -> dict:
    """Build dry-plan payload.

    ``policy_ready`` reflects static frozen-policy checks.
    ``hpo_ready`` is true only when launch gates (code-lock + binding) also pass.
    Without ``policy_path`` on disk, ``hpo_ready`` stays false even if policy_ready.
    """
    hpo = hpo_plan(policy)
    placeholder = {v: "lr1e-4_wd0.01" for v in ("b0", "b3", "b4", "b4r")}
    conf = confirm_plan(policy, selected_config_by_variant=placeholder)
    lineage = lineage_block(policy)
    arch = dict(policy.get("architecture") or {})

    root_p = root if root is not None else _ROOT
    if policy_path is not None and policy_path.is_file():
        readiness = assess_hpo_readiness(
            policy,
            root=root_p,
            policy_path=policy_path,
            binding_path=binding_path,
        )
    else:
        refuse = refuse_train_until_frozen(policy)
        if refuse:
            readiness = {
                "policy_ready": False,
                "hpo_ready": False,
                "train_refused": refuse,
                "reasons": [refuse],
            }
        else:
            try:
                require_frozen_v4_policy(policy, root=root_p)
                readiness = {
                    "policy_ready": True,
                    "hpo_ready": False,
                    "train_refused": None,
                    "reasons": [
                        "policy_ready only: binding/code-lock preflight not verified "
                        "(no on-disk policy_path)"
                    ],
                }
            except ValueError as exc:
                readiness = {
                    "policy_ready": False,
                    "hpo_ready": False,
                    "train_refused": None,
                    "reasons": [str(exc)],
                }

    policy_ready = bool(readiness["policy_ready"])
    hpo_ready = bool(readiness["hpo_ready"])
    if hpo_ready:
        implementation_status = str(
            (policy.get("orchestrator") or {}).get("implementation_status")
            or "frozen_hpo_ready"
        )
        note = (
            "HPO_READY: frozen policy + freeze binding + code-lock preflight verified. "
            "Dry-plan does not start training; launch HPO explicitly after audit PASS."
        )
    elif policy_ready:
        implementation_status = str(
            (policy.get("orchestrator") or {}).get("implementation_status")
            or "frozen_policy_not_launch_ready"
        )
        note = (
            "policy_ready=true but hpo_ready=false: "
            + "; ".join(readiness.get("reasons") or ["launch gates incomplete"])
        )
    else:
        implementation_status = "train_refused_until_frozen"
        note = (
            "Training refused until immutable FROZEN policy + clean code-lock ("
            + "; ".join(readiness.get("reasons") or ["not frozen"])
            + ")."
        )

    return {
        "schema": "ms_sta_retrain_v4_dry_plan_v6",
        "train_allowed": bool(policy.get("train_allowed")),
        "status": policy.get("status"),
        "path": policy.get("path"),
        "primary_variant": policy.get("primary_variant"),
        "residual_init_default": arch.get("default_residual_init"),
        "architecture": {
            "formula": arch.get("formula"),
            "delta_head": arch.get("delta_head"),
            "param_count_contract": arch.get("param_count_contract"),
            "forbid_duplicate_delta_cls_module": bool(
                arch.get("forbid_duplicate_delta_cls_module")
            ),
        },
        "primary_comparison": dict(policy.get("primary_comparison") or {}),
        "warm_start_diagnostic": dict(policy.get("warm_start_diagnostic") or {}),
        "hpo": {
            "runs_total": len(hpo),
            "configs_per_variant": 6,
            "variant_config_pairs": 24,
            "variants": sorted({t["variant"] for t in hpo}),
            "tune_seeds": sorted({t["seed"] for t in hpo}),
            "early_stopping": True,
            "tie_epsilon_pp": (policy.get("hpo") or {}).get("tie_epsilon_pp", 0.0),
            "selection_metric": (policy.get("hpo") or {}).get("selection_metric"),
            "tie_break": (policy.get("hpo") or {}).get("tie_break"),
        },
        "confirm": {
            "n_runs": len(conf),
            "run_full_epochs": True,
            "early_stopping_allowed": False,
            "epochs": conf[0]["epochs"] if conf else None,
            "require_aligned_per_sample_validation_rows": True,
        },
        "aux_loss": {
            "silent_enable_forbidden": True,
            "halt_for_amendment_if_b4r_le_b3_on_tune": True,
        },
        "lineage": lineage,
        "accuracy_gates": dict(policy.get("accuracy_gates") or {}),
        "efficiency_campaign": {
            "campaign_id": (policy.get("efficiency_campaign") or {}).get("campaign_id"),
            "opens_only_after_accuracy_gate_pass": True,
            "forbid_mixing_into_accuracy_hpo": True,
        },
        "orchestrator_contracts": {
            "atomic_run_contract": True,
            "atomic_run_state": True,
            "resume_requires_matching_hashes": True,
            "confirm_requires_final_checkpoint": True,
            "lock_and_test_human_reviewed": True,
        },
        "code_lock": dict(policy.get("code_lock") or {}),
        "live_sot_until_pass": {
            "campaign": "VALIDATION_NO_GO",
            "current_best": "b3",
            "top1_mean_pct": 62.28644829800899,
        },
        "implementation_status": implementation_status,
        "policy_ready": policy_ready,
        "hpo_ready": hpo_ready,
        "readiness_reasons": list(readiness.get("reasons") or []),
        "note": note,
    }


def _load_status_rows(status_dir: Path, phase: str) -> list[dict]:
    rows = []
    if not status_dir.is_dir():
        return rows
    for path in sorted(status_dir.glob(f"{phase}__*.json")):
        if path.name.endswith("__final.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_key"] = path.stem
        rows.append(payload)
    return rows


def _preflight_clean_and_code_lock(policy: dict) -> None:
    """Common launch gate for *all* phases including lock/test early returns."""
    code_lock = policy.get("code_lock") or {}
    dirty = _git_dirty()
    if code_lock.get("require_clean_worktree", True) and dirty:
        raise SystemExit("[FAIL] clean worktree required for retrain-v4 launch")
    try:
        verify_code_lock_against_repo(
            root=_ROOT, code_lock=code_lock, head_rev=_git_rev()
        )
    except ValueError as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc


def _ensure_state(
    *,
    state_path: Path,
    contract_path: Path,
    policy: dict,
    policy_path: Path,
    policy_sha: str,
    phase: str,
    plan: list[dict],
    resume: bool,
) -> dict:
    _preflight_clean_and_code_lock(policy)
    head = _git_rev()
    dirty = _git_dirty()
    lineage = lineage_block(policy)
    split_sha = lineage.get("split_manifest_sha256")
    state = {
        "schema": "ms_sta_retrain_v4_run_state_v2",
        "policy": _rel(policy_path),
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": head,
        "git_dirty": dirty,
        "path": "B",
        "phases": {},
        "selected_configs": {},
        "selected_configs_sha256": None,
        "evaluation_locked": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if state_path.is_file():
        prev = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise SystemExit("[FAIL] RUN_STATE exists; pass --resume or clear versioned outputs")
        if prev.get("policy_sha256") != policy_sha:
            raise SystemExit("[FAIL] resume policy sha mismatch")
        if prev.get("git_revision") != head:
            raise SystemExit("[FAIL] resume git revision mismatch")
        if prev.get("split_manifest_sha256") != split_sha:
            raise SystemExit("[FAIL] resume split sha mismatch")
        state = prev
    # Per-phase immutable contract (never overwrite another phase's contract).
    if contract_path.is_file():
        prev_c = json.loads(contract_path.read_text(encoding="utf-8"))
        if (
            prev_c.get("policy_sha256") != policy_sha
            or prev_c.get("split_manifest_sha256") != split_sha
            or prev_c.get("git_revision") != head
            or prev_c.get("phase") != phase
            or int(prev_c.get("plan_n") or -1) != len(plan)
            or prev_c.get("plan_keys") != [run_key(t) for t in plan]
        ):
            raise SystemExit(
                f"[FAIL] existing phase contract mismatch for {contract_path.name}; "
                "refuse overwrite of immutable campaign contract"
            )
    else:
        contract = {
            "schema": "ms_sta_retrain_v4_run_contract_v2",
            "phase": phase,
            "policy_sha256": policy_sha,
            "split_manifest_sha256": split_sha,
            "git_revision": head,
            "git_dirty": dirty,
            "plan_n": len(plan),
            "plan_keys": [run_key(t) for t in plan],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "forbid_auto_chain_to_test": True,
            "resume_requires_matching_hashes": True,
            "immutable_after_write": True,
        }
        _atomic_write(contract_path, contract)
    phases = dict(state.get("phases") or {})
    phase_rec = dict(phases.get(phase) or {})
    phase_rec.update(
        {
            "contract": _rel(contract_path),
            "plan_n": len(plan),
            "started_at": phase_rec.get("started_at")
            or datetime.now(timezone.utc).isoformat(),
            "status": phase_rec.get("status") or "in_progress",
            "completed_keys": list(phase_rec.get("completed_keys") or []),
        }
    )
    phases[phase] = phase_rec
    state["phases"] = phases
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)
    return state


def _mark_trial_complete(state: dict, state_path: Path, phase: str, stem: str) -> None:
    phases = dict(state.get("phases") or {})
    phase_rec = dict(phases.get(phase) or {})
    done = list(phase_rec.get("completed_keys") or [])
    if stem not in done:
        done.append(stem)
    phase_rec["completed_keys"] = done
    phase_rec["n_complete"] = len(done)
    plan_n = int(phase_rec.get("plan_n") or 0)
    if plan_n and len(done) >= plan_n:
        phase_rec["status"] = "complete"
        phase_rec["completed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        phase_rec["status"] = "in_progress"
    phases[phase] = phase_rec
    state["phases"] = phases
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)


def _gather_lock_disk_evidence(
    *,
    policy: dict,
    policy_sha: str,
    status_dir: Path,
    state: dict,
    verify_trials_complete: bool = True,
) -> dict:
    """Recompute evaluation-lock identity from on-disk confirm artifacts (shared)."""
    report_path = status_dir / "CONFIRM_VALIDATION_REPORT.json"
    selected_path = status_dir / "SELECTED_CONFIGS.json"
    per_sample_manifest_path = status_dir / "per_sample_val" / "PER_SAMPLE_MANIFEST.json"
    binding_path = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json"
    ckpt_dir, _, _ = _artifact_dirs(policy)
    for required, label in (
        (report_path, "CONFIRM_VALIDATION_REPORT.json"),
        (selected_path, "SELECTED_CONFIGS.json"),
        (per_sample_manifest_path, "per_sample_val/PER_SAMPLE_MANIFEST.json"),
        (binding_path, "POLICY_FREEZE_BINDING.json"),
    ):
        if not required.is_file():
            raise ValueError(f"missing {label}")

    lineage = lineage_block(policy)
    split_sha = require_hex64(
        lineage.get("split_manifest_sha256"), label="lineage.split_manifest_sha256"
    )
    git_revision = str(state.get("git_revision") or "")
    if not git_revision:
        raise ValueError("RUN_STATE.git_revision missing")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    configs = selected.get("selected") or {}
    if set(configs) != set(CONFIRM_VARIANTS):
        raise ValueError(f"selected variants {sorted(configs)} != {list(CONFIRM_VARIANTS)}")

    manifest = load_per_sample_manifest(per_sample_manifest_path)
    ckpt_map = require_per_sample_manifest_contract(
        manifest,
        expected_policy_sha=policy_sha,
        expected_split_sha=split_sha,
        expected_git_rev=git_revision,
        expected_variants=("b3", "b4r"),
    )
    man_sha = sha256_file(per_sample_manifest_path)
    path_map = manifest.get("jsonl_paths_by_variant_seed") or {}
    jsonl_shas = dict(manifest.get("jsonl_sha256_by_variant_seed") or {})
    if len(jsonl_shas) != 6:
        raise ValueError(f"expected 6 B3/B4R jsonl entries, got {len(jsonl_shas)}")
    for key, expected_sha in jsonl_shas.items():
        rel = path_map.get(key)
        if not rel:
            raise ValueError(f"missing jsonl path for {key}")
        path = _ROOT / str(rel)
        if not path.is_file():
            raise ValueError(f"missing jsonl file {path}")
        require_exact_str(sha256_file(path), expected_sha, label=f"jsonl[{key}]")
        require_hex64(expected_sha, label=f"jsonl_sha[{key}]")

    final_shas: dict[str, str] = {}
    for variant, cid in configs.items():
        for seed in CONFIRM_SEEDS:
            stem = run_key(
                {
                    "phase": "confirm",
                    "variant": variant,
                    "config_id": cid,
                    "seed": seed,
                    "skel_temporal_stride": 1,
                }
            )
            status_path = status_dir / f"{stem}.json"
            cand_path = ckpt_dir / f"{stem}.pth"
            final_path = ckpt_dir / f"{final_stem_from_candidate_stem(stem)}.pth"
            if verify_trials_complete:
                if not trial_is_complete(
                    phase="confirm",
                    variant=str(variant),
                    config_id=str(cid),
                    seed=int(seed),
                    status_path=status_path,
                    ckpt_path=cand_path,
                    final_path=final_path,
                    expected_policy_sha=policy_sha,
                    expected_split_sha=split_sha,
                    expected_git_rev=git_revision,
                ):
                    raise ValueError(f"confirm trial incomplete for lock: {stem}")
            elif not final_path.is_file():
                raise ValueError(f"missing final checkpoint {final_path}")
            final_sha = sha256_file(final_path)
            final_shas[_rel(final_path)] = final_sha
            if variant in ("b3", "b4r"):
                key = f"{variant}/seed{seed}"
                require_exact_str(
                    ckpt_map[(variant, int(seed))],
                    final_sha,
                    label=f"manifest final vs disk {key}",
                )

    if len(final_shas) != 12:
        raise ValueError(f"expected 12 final checkpoints, got {len(final_shas)}")

    return {
        "split_sha": split_sha,
        "git_revision": git_revision,
        "report_path": report_path,
        "selected_path": selected_path,
        "per_sample_manifest_path": per_sample_manifest_path,
        "binding_path": binding_path,
        "report_sha": sha256_file(report_path),
        "selected_sha": sha256_file(selected_path),
        "man_sha": man_sha,
        "freeze_binding_sha": sha256_file(binding_path),
        "final_shas": final_shas,
        "jsonl_shas": jsonl_shas,
        "configs": configs,
        "manifest": manifest,
    }


def _apply_fresh_lock_to_run_state(
    *,
    state_path: Path,
    lock_path: Path,
    report_sha: str,
    man_sha: str,
) -> dict:
    """Write lock fields into RUN_STATE after a *new* EVALUATION_LOCKED.json was created."""
    if not state_path.is_file():
        raise ValueError("RUN_STATE.json missing after lock write")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    lock_sha = sha256_file(lock_path)
    state["evaluation_locked"] = True
    state["evaluation_locked_sha256"] = lock_sha
    state["phases"] = dict(state.get("phases") or {})
    state["phases"]["lock"] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "lock_artifact": _rel(lock_path),
        "report_sha256": report_sha,
        "per_sample_manifest_sha256": man_sha,
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)
    return state


def _resume_lock_heal(
    *,
    policy: dict,
    policy_path: Path,
    policy_sha: str,
    status_dir: Path,
    state_path: Path,
    lock_path: Path,
) -> int:
    """Verify existing lock; fill only missing RUN_STATE lock fields.

    Requires RUN_STATE to already exist. Never creates state from {}. Never
    overwrites an existing evaluation_locked_sha256 (mismatch → exit 2).
    Pre-lock heal requires state file sha == lock.run_state_sha256.
    Full lock contract (schema / selected / 12 finals / 6 jsonl / primary_seed)
    is verified via the same disk evidence path as lock create.
    """
    if not state_path.is_file():
        print("[FAIL] lock --resume requires existing RUN_STATE.json (refuse create-from-empty)")
        return 2
    if not lock_path.is_file():
        print("[FAIL] lock --resume requires existing EVALUATION_LOCKED.json")
        return 2

    state = json.loads(state_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha = sha256_file(lock_path)
    head = _git_rev()
    confirm_phase = (state.get("phases") or {}).get("confirm") or {}

    try:
        require_hex64(policy_sha, label="policy_sha")
        require_exact_str(state.get("policy_sha256"), policy_sha, label="RUN_STATE.policy_sha256")
        require_exact_str(state.get("git_revision"), head, label="RUN_STATE.git_revision")
        require_exact_str(confirm_phase.get("status"), "complete", label="confirm.status")
        if int(confirm_phase.get("n_complete") or -1) != 12:
            raise ValueError(f"confirm.n_complete={confirm_phase.get('n_complete')!r} != 12")

        evidence = _gather_lock_disk_evidence(
            policy=policy,
            policy_sha=policy_sha,
            status_dir=status_dir,
            state=state,
            verify_trials_complete=True,
        )
        require_exact_str(
            state.get("split_manifest_sha256"),
            evidence["split_sha"],
            label="RUN_STATE.split_manifest_sha256",
        )
        require_freeze_binding(
            json.loads(evidence["binding_path"].read_text(encoding="utf-8")),
            root=_ROOT,
            policy_path=policy_path,
            policy_sha=policy_sha,
            head_rev=head,
        )

        already_locked = bool(state.get("evaluation_locked")) or bool(
            state.get("evaluation_locked_sha256")
        )
        # Pre-lock: seal requires current state bytes == lock.run_state_sha256.
        # Already-locked: only require lock field is hex64 (state mutated after seal).
        run_state_sha = None if already_locked else sha256_file(state_path)
        require_evaluation_locked_contract(
            lock,
            policy_sha=policy_sha,
            split_sha=evidence["split_sha"],
            git_revision=evidence["git_revision"],
            report_sha=evidence["report_sha"],
            selected_sha=evidence["selected_sha"],
            man_sha=evidence["man_sha"],
            freeze_binding_sha=evidence["freeze_binding_sha"],
            final_shas=evidence["final_shas"],
            jsonl_shas=evidence["jsonl_shas"],
            run_state_sha=run_state_sha,
        )

        report_sha = evidence["report_sha"]
        man_sha = evidence["man_sha"]
        if already_locked:
            existing = str(state.get("evaluation_locked_sha256") or "").strip()
            if not existing:
                raise ValueError(
                    "RUN_STATE.evaluation_locked set but evaluation_locked_sha256 missing"
                )
            if existing != lock_sha:
                raise ValueError(
                    f"CONFLICTING evaluation_locked_sha256: state={existing} lock_file={lock_sha}"
                )
            phases = dict(state.get("phases") or {})
            lock_phase = dict(phases.get("lock") or {})
            changed = False
            if lock_phase.get("status") != "complete":
                lock_phase["status"] = "complete"
                changed = True
            if not lock_phase.get("lock_artifact"):
                lock_phase["lock_artifact"] = _rel(lock_path)
                changed = True
            if not lock_phase.get("report_sha256"):
                lock_phase["report_sha256"] = report_sha
                changed = True
            if not lock_phase.get("per_sample_manifest_sha256"):
                lock_phase["per_sample_manifest_sha256"] = man_sha
                changed = True
            if not lock_phase.get("completed_at"):
                lock_phase["completed_at"] = datetime.now(timezone.utc).isoformat()
                changed = True
            if state.get("evaluation_locked") is not True:
                state["evaluation_locked"] = True
                changed = True
            if changed:
                phases["lock"] = lock_phase
                state["phases"] = phases
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                _atomic_write(state_path, state)
        else:
            state["evaluation_locked"] = True
            state["evaluation_locked_sha256"] = lock_sha
            phases = dict(state.get("phases") or {})
            phases["lock"] = {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "lock_artifact": _rel(lock_path),
                "report_sha256": report_sha,
                "per_sample_manifest_sha256": man_sha,
            }
            state["phases"] = phases
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write(state_path, state)
    except ValueError as exc:
        print(f"[FAIL] lock --resume verify failed: {exc}")
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "lock_resume_healed": True,
                "evaluation_locked": _rel(lock_path),
                "evaluation_locked_sha256": lock_sha,
                "note": "Existing lock fully verified; only missing RUN_STATE lock fields filled.",
            },
            indent=2,
        )
    )
    return 0


def _lock_evaluation(
    *,
    policy: dict,
    policy_path: Path,
    policy_sha: str,
    status_dir: Path,
    ckpt_dir: Path,
    state_path: Path,
    resume: bool,
) -> int:
    """Verify report/policy/split/per-sample/final hashes; write EVALUATION_LOCKED.json."""
    report_path = status_dir / "CONFIRM_VALIDATION_REPORT.json"
    selected_path = status_dir / "SELECTED_CONFIGS.json"
    lock_path = status_dir / "EVALUATION_LOCKED.json"
    per_sample_manifest_path = status_dir / "per_sample_val" / "PER_SAMPLE_MANIFEST.json"
    binding_path = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json"
    if lock_path.is_file():
        if not resume:
            print(
                "[FAIL] EVALUATION_LOCKED.json already exists; refuse overwrite "
                "(pass --resume to verify lock and heal RUN_STATE only)"
            )
            return 2
        return _resume_lock_heal(
            policy=policy,
            policy_path=policy_path,
            policy_sha=policy_sha,
            status_dir=status_dir,
            state_path=state_path,
            lock_path=lock_path,
        )
    for required, label in (
        (report_path, "CONFIRM_VALIDATION_REPORT.json"),
        (selected_path, "SELECTED_CONFIGS.json"),
        (state_path, "RUN_STATE.json"),
        (per_sample_manifest_path, "per_sample_val/PER_SAMPLE_MANIFEST.json"),
        (binding_path, "POLICY_FREEZE_BINDING.json"),
    ):
        if not required.is_file():
            print(f"[FAIL] lock requires {label}")
            return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    gates = report.get("architecture_gates") or {}
    if not gates.get("architecture_gate_pass"):
        print("[FAIL] lock refused: architecture gate did not PASS")
        print("[INFO] Campaign may close VALIDATION_NO_GO without opening test.")
        return 2
    bindings = report.get("bindings") or {}
    if bindings.get("canonical_producer") is not True:
        print("[FAIL] report missing canonical_producer binding (refuse hand-edited report)")
        return 2
    if bindings.get("evaluator") != "scripts/evaluate_ms_sta_retrain_v4_validation.py":
        print("[FAIL] report evaluator binding mismatch")
        return 2
    if report.get("schema") != "ms_sta_retrain_v4_paired_gate_v3":
        print("[FAIL] report schema is not canonical paired_gate_v3")
        return 2

    state = json.loads(state_path.read_text(encoding="utf-8"))
    lineage = lineage_block(policy)
    split_sha = str(lineage.get("split_manifest_sha256") or "")
    head = _git_rev()
    confirm_phase = (state.get("phases") or {}).get("confirm") or {}
    try:
        require_hex64(policy_sha, label="policy_sha")
        require_hex64(split_sha, label="split_sha")
        require_exact_str(state.get("policy_sha256"), policy_sha, label="RUN_STATE.policy_sha256")
        require_exact_str(
            state.get("split_manifest_sha256"), split_sha, label="RUN_STATE.split_sha"
        )
        require_exact_str(confirm_phase.get("status"), "complete", label="confirm.status")
        if int(confirm_phase.get("n_complete") or -1) != 12:
            raise ValueError(
                f"confirm.n_complete={confirm_phase.get('n_complete')!r} != 12"
            )
        require_exact_str(
            bindings.get("policy_sha256"), policy_sha, label="report.bindings.policy_sha256"
        )
        require_exact_str(
            bindings.get("split_manifest_sha256"),
            split_sha,
            label="report.bindings.split_manifest_sha256",
        )
        require_exact_str(
            bindings.get("git_revision"),
            state.get("git_revision"),
            label="report.bindings.git_revision",
        )
        freeze_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        require_freeze_binding(
            freeze_binding,
            root=_ROOT,
            policy_path=policy_path,
            policy_sha=policy_sha,
            head_rev=head,
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    try:
        evidence = _gather_lock_disk_evidence(
            policy=policy,
            policy_sha=policy_sha,
            status_dir=status_dir,
            state=state,
            verify_trials_complete=True,
        )
        report_sha = evidence["report_sha"]
        selected_sha = evidence["selected_sha"]
        man_sha = evidence["man_sha"]
        final_shas = evidence["final_shas"]
        jsonl_shas = evidence["jsonl_shas"]
        split_sha = evidence["split_sha"]
        require_exact_str(
            bindings.get("per_sample_manifest_sha256"),
            man_sha,
            label="report.bindings.per_sample_manifest_sha256",
        )
        require_exact_str(
            bindings.get("exact_id_sha256"),
            evidence["manifest"].get("exact_id_sha256"),
            label="report.bindings.exact_id_sha256",
        )
        for variant in ("b3", "b4r"):
            for seed in CONFIRM_SEEDS:
                key = f"{variant}/seed{seed}"
                # Path key for primary/other seeds already in final_shas; match report binding.
                # Find sha by scanning finals for this variant/seed via configs.
                cid = evidence["configs"][variant]
                stem = run_key(
                    {
                        "phase": "confirm",
                        "variant": variant,
                        "config_id": cid,
                        "seed": seed,
                        "skel_temporal_stride": 1,
                    }
                )
                final_rel = _rel(ckpt_dir / f"{final_stem_from_candidate_stem(stem)}.pth")
                require_exact_str(
                    (bindings.get("final_checkpoint_sha256_by_variant_seed") or {}).get(key),
                    final_shas[final_rel],
                    label=f"report final binding {key}",
                )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    run_state_sha = sha256_file(state_path)
    lock_payload = {
        "schema": EVALUATION_LOCKED_SCHEMA,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "policy": _rel(policy_path),
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": state.get("git_revision"),
        "confirm_validation_report": _rel(report_path),
        "confirm_validation_report_sha256": report_sha,
        "selected_configs": _rel(selected_path),
        "selected_configs_sha256": selected_sha,
        "run_state": _rel(state_path),
        "run_state_sha256": run_state_sha,
        "per_sample_manifest": _rel(per_sample_manifest_path),
        "per_sample_manifest_sha256": man_sha,
        "final_checkpoint_sha256_by_path": final_shas,
        "per_sample_jsonl_sha256_by_variant_seed": jsonl_shas,
        "primary_seed": PRIMARY_SEED,
        "architecture_gate_pass": True,
        "forbid_auto_chain_to_test": True,
        "test_requires_explicit_launch": True,
        "freeze_binding": _rel(binding_path),
        "freeze_binding_sha256": evidence["freeze_binding_sha"],
    }
    try:
        require_evaluation_locked_contract(
            lock_payload,
            policy_sha=policy_sha,
            split_sha=split_sha,
            git_revision=str(state.get("git_revision") or ""),
            report_sha=report_sha,
            selected_sha=selected_sha,
            man_sha=man_sha,
            freeze_binding_sha=evidence["freeze_binding_sha"],
            final_shas=final_shas,
            jsonl_shas=jsonl_shas,
            run_state_sha=run_state_sha,
        )
    except ValueError as exc:
        print(f"[FAIL] lock payload self-check failed: {exc}")
        return 2
    _atomic_write(lock_path, lock_payload)
    _apply_fresh_lock_to_run_state(
        state_path=state_path,
        lock_path=lock_path,
        report_sha=report_sha,
        man_sha=man_sha,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "evaluation_locked": _rel(lock_path),
                "report_sha256": report_sha,
                "n_final_checkpoints": len(final_shas),
                "n_per_sample_jsonl": len(jsonl_shas),
            },
            indent=2,
        )
    )
    return 0


def _heal_phases_test_from_completion(
    *,
    state_path: Path,
    status_dir: Path,
    completion_path: Path,
) -> dict:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    report_path = status_dir / "TEST_REPORT.json"
    rows_path = status_dir / "TEST_PER_SAMPLE.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phases"] = dict(state.get("phases") or {})
    state["phases"]["test"] = {
        "status": "complete",
        "completed_at": completion.get("completed_at")
        or datetime.now(timezone.utc).isoformat(),
        "test_report": _rel(report_path),
        "test_report_sha256": completion.get("test_report_sha256") or sha256_file(report_path),
        "test_per_sample_jsonl": _rel(rows_path),
        "test_per_sample_jsonl_sha256": completion.get("test_per_sample_jsonl_sha256")
        or sha256_file(rows_path),
        "test_completion": _rel(completion_path),
        "test_completion_sha256": sha256_file(completion_path),
    }
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(state_path, state)
    return state


def _phase_test(*, policy_path: Path, status_dir: Path, state_path: Path, resume: bool) -> int:
    lock_flag = status_dir / "EVALUATION_LOCKED.json"
    if not lock_flag.is_file():
        print("[FAIL] test refused until EVALUATION_LOCKED.json after architecture PASS")
        return 2
    if not state_path.is_file():
        print("[FAIL] test requires RUN_STATE.json")
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not state.get("evaluation_locked"):
        # Allow heal path when lock file exists but RUN_STATE crashed mid-lock.
        if resume and lock_flag.is_file():
            print("[INFO] evaluation_locked missing in RUN_STATE; lock file present — heal via lock --resume first")
        print("[FAIL] RUN_STATE.evaluation_locked is not true")
        return 2
    try:
        require_exact_str(
            state.get("evaluation_locked_sha256"),
            sha256_file(lock_flag),
            label="RUN_STATE.evaluation_locked_sha256",
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    report_path = status_dir / "TEST_REPORT.json"
    rows_path = status_dir / "TEST_PER_SAMPLE.jsonl"
    completion_path = status_dir / "TEST_COMPLETION.json"
    phase_test = (state.get("phases") or {}).get("test")
    if phase_test or report_path.is_file() or rows_path.is_file() or completion_path.is_file():
        if not resume:
            print(
                "[FAIL] test already recorded/artifacts exist; refuse overwrite "
                "(pass --resume to verify bit-exact only)"
            )
            return 2
    cmd = [
        sys.executable,
        "-u",
        str(_ROOT / "scripts/evaluate_ms_sta_retrain_v4_test.py"),
        "--policy",
        str(policy_path),
        "--eval-lock",
        str(lock_flag),
        "--confirm-retrospective-test",
    ]
    if resume:
        cmd.append("--resume")
    print(f"[launch] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=_ROOT, check=False)
    if proc.returncode != 0:
        return int(proc.returncode) or 1
    if not report_path.is_file():
        print("[FAIL] test evaluator did not write/keep TEST_REPORT.json")
        return 2
    if not rows_path.is_file():
        print("[FAIL] test evaluator did not write/keep TEST_PER_SAMPLE.jsonl")
        return 2
    if not completion_path.is_file():
        print("[FAIL] test evaluator did not write/keep TEST_COMPLETION.json")
        return 2
    # Always heal phases.test from immutable completion (covers crash-after-completion).
    _heal_phases_test_from_completion(
        state_path=state_path,
        status_dir=status_dir,
        completion_path=completion_path,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "phase": "test",
                "resume_verified": bool(resume),
                "report": _rel(report_path),
                "completion": _rel(completion_path),
            },
            indent=2,
        )
    )
    return 0


def run_phase(policy: dict, policy_path: Path, phase: str, *, resume: bool) -> int:
    refuse = refuse_train_until_frozen(policy)
    if refuse:
        print(f"[FAIL] refuse phase={phase}: {refuse}")
        print("[FAIL] Live SoT remains VALIDATION_NO_GO / B3 until v4 accuracy gate passes.")
        return 2
    require_frozen_v4_policy(policy, root=_ROOT)
    # Common preflight before *any* phase early-return (lock/test included).
    _preflight_clean_and_code_lock(policy)
    policy_sha = sha256_file(policy_path)
    binding_path = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json"
    if not binding_path.is_file():
        print("[FAIL] missing POLICY_FREEZE_BINDING.json; freeze+finalize before launch")
        return 2
    try:
        require_freeze_binding(
            json.loads(binding_path.read_text(encoding="utf-8")),
            root=_ROOT,
            policy_path=policy_path,
            policy_sha=policy_sha,
            head_rev=_git_rev(),
        )
    except ValueError as exc:
        print(f"[FAIL] freeze binding: {exc}")
        return 2
    ckpt_dir, status_dir, fail_dir = _artifact_dirs(policy)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    state_path = status_dir / "RUN_STATE.json"
    contract_path = status_dir / f"_run_contract_{phase}.json"
    lineage = lineage_block(policy)
    split_sha = str(lineage.get("split_manifest_sha256") or "")
    head = _git_rev()

    if phase == "hpo":
        trials = hpo_plan(policy)
    elif phase == "confirm":
        selected_path = status_dir / "SELECTED_CONFIGS.json"
        if not selected_path.is_file():
            print("[FAIL] missing SELECTED_CONFIGS.json; run hpo selection first")
            return 2
        selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
        if selected_payload.get("aux_loss_check", {}).get("triggered"):
            print("[FAIL] aux-loss amendment pending; confirm refused")
            return 3
        trials = confirm_plan(
            policy, selected_config_by_variant=selected_payload["selected"]
        )
    elif phase == "lock":
        return _lock_evaluation(
            policy=policy,
            policy_path=policy_path,
            policy_sha=policy_sha,
            status_dir=status_dir,
            ckpt_dir=ckpt_dir,
            state_path=state_path,
            resume=resume,
        )
    elif phase == "test":
        return _phase_test(
            policy_path=policy_path,
            status_dir=status_dir,
            state_path=state_path,
            resume=resume,
        )
    else:
        print(f"[FAIL] unknown phase={phase}")
        return 2

    state = _ensure_state(
        state_path=state_path,
        contract_path=contract_path,
        policy=policy,
        policy_path=policy_path,
        policy_sha=policy_sha,
        phase=phase,
        plan=trials,
        resume=resume,
    )

    for trial in trials:
        stem = _stem(trial)
        status_path = status_dir / f"{stem}.json"
        ckpt_path = ckpt_dir / f"{stem}.pth"
        final_path = ckpt_dir / f"{final_stem_from_candidate_stem(stem)}.pth"
        try:
            done = trial_is_complete(
                phase=phase,
                variant=str(trial["variant"]),
                config_id=str(trial["config_id"]),
                seed=int(trial["seed"]),
                status_path=status_path,
                ckpt_path=ckpt_path,
                final_path=final_path if phase == "confirm" else None,
                expected_policy_sha=policy_sha,
                expected_split_sha=split_sha,
                expected_git_rev=head,
            )
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 2
        if done:
            print(f"[skip] complete {stem}")
            _mark_trial_complete(state, state_path, phase, stem)
            continue
        if status_path.is_file() or ckpt_path.is_file() or (
            phase == "confirm" and final_path.is_file()
        ):
            print(f"[FAIL] refuse partial overwrite for {stem}")
            return 2
        cmd = _trial_cmd(policy_path, trial, stem)
        print(f"[launch] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=_ROOT, check=False)
        if proc.returncode != 0:
            _atomic_write(
                fail_dir / f"{stem}.json",
                {"stem": stem, "exit_code": proc.returncode, "cmd": cmd},
            )
            return int(proc.returncode)
        try:
            if not trial_is_complete(
                phase=phase,
                variant=str(trial["variant"]),
                config_id=str(trial["config_id"]),
                seed=int(trial["seed"]),
                status_path=status_path,
                ckpt_path=ckpt_path,
                final_path=final_path if phase == "confirm" else None,
                expected_policy_sha=policy_sha,
                expected_split_sha=split_sha,
                expected_git_rev=head,
            ):
                print(f"[FAIL] trial did not produce complete artifacts: {stem}")
                return 2
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 2
        _mark_trial_complete(state, state_path, phase, stem)

    if phase == "hpo":
        rows = _load_status_rows(status_dir, "hpo")
        selected = select_configs_per_variant(rows)
        aux = aux_loss_amendment_required(rows, selected)
        selected_path = status_dir / "SELECTED_CONFIGS.json"
        _atomic_write(
            selected_path,
            {
                "schema": "ms_sta_retrain_v4_selected_configs_v1",
                "selected": selected,
                "aux_loss_check": aux,
                "tie_epsilon_pp": 0.0,
            },
        )
        state["selected_configs"] = selected
        state["selected_configs_sha256"] = sha256_file(selected_path)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(state_path, state)
        if aux["triggered"]:
            print("[HALT] aux-loss amendment required; campaign stopped before confirm")
            print(json.dumps(aux, indent=2))
            return 3
        print(json.dumps({"selected": selected, "aux_loss_check": aux}, indent=2))
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--dry-plan", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--phase",
        choices=("hpo", "hpo_screen", "hpo_full", "confirm", "lock", "test"),
        default=None,
    )
    args = ap.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    if not policy_path.is_file():
        print(f"[FAIL] missing policy: {policy_path}")
        return 2
    policy = load_policy(policy_path)

    if args.dry_plan or args.phase is None:
        plan = dry_plan(policy, root=_ROOT, policy_path=policy_path)
        refuse = refuse_train_until_frozen(policy)
        # Fail-closed: frozen/policy_ready without verified binding/code-lock → exit 2.
        # Draft (not policy_ready) remains exit 0 with train_refused explanation.
        if plan.get("policy_ready") and not plan.get("hpo_ready"):
            print(
                json.dumps(
                    {
                        **plan,
                        "ok": False,
                        "train_refused": refuse,
                        "fail": "hpo_ready_gates_failed",
                    },
                    indent=2,
                )
            )
            return 2
        print(
            json.dumps(
                {**plan, "ok": True, "train_refused": refuse},
                indent=2,
            )
        )
        return 0

    phase = "hpo" if args.phase in ("hpo_screen", "hpo_full", "hpo") else args.phase
    return run_phase(policy, policy_path, phase, resume=bool(args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
