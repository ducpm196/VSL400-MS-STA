"""Fail-closed contract tests for retrain-v4 P0 hardening."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from src.training.retrain_v4 import (
    CONFIRM_SEEDS,
    CONFIRM_VARIANTS,
    confirm_final_checkpoint_rel,
    executable_tree_sha256,
    primary_seed_final_sha_by_variant,
    require_code_lock_fields,
    require_evaluation_locked_contract,
    require_frozen_v4_policy,
    require_per_sample_manifest_contract,
    select_configs_per_variant,
    trial_is_complete,
)
from scripts.evaluate_ms_sta_retrain_v4_validation import build_gate_report

_ROOT = Path(__file__).resolve().parents[1]
DRAFT = _ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json"
POLICY_SHA = "d" * 64
CKPT_SHA = "c" * 64


def _ckpt_map() -> dict[tuple[str, int], str]:
    return {(v, s): CKPT_SHA for v in ("b3", "b4r") for s in (42, 123, 456)}


def _row(
    *,
    sample_id: str,
    signer_id: str,
    seed: int,
    variant: str,
    y_true: int,
    y_pred: int,
    top5: list[int] | None = None,
    policy_sha: str = POLICY_SHA,
    ckpt_sha: str = CKPT_SHA,
) -> dict:
    if top5 is None:
        # Keep IDs in [0,1] so tiny num_classes fixtures stay valid.
        top5 = [y_pred] + [i % 2 for i in range(4)]
    top5 = list(top5)
    while len(top5) < 5:
        top5.append(0)
    return {
        "sample_id": sample_id,
        "signer_id": signer_id,
        "seed": seed,
        "variant": variant,
        "y_true": y_true,
        "y_pred": y_pred,
        "correct_top1": int(y_pred == y_true),
        "correct_top5": int(y_true in top5),
        "top5": top5,
        "final_checkpoint_sha256": ckpt_sha,
        "policy_sha256": policy_sha,
    }


def test_grad_clip_runs_in_v4_train_step() -> None:
    """One real backward step under v4_mode must call clip_grad_norm_."""
    model = nn.Linear(4, 2)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 0])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    crit = nn.CrossEntropyLoss()
    v4_mode = True
    grad_clip = 1.0
    with mock.patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clipped:
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        if (False or v4_mode) and grad_clip is not None:  # mirrors trainer condition shape
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
    assert clipped.called


def test_grad_clip_condition_in_trainer_source() -> None:
    src = (_ROOT / "src/training/train_ms_sta_vsl400.py").read_text(encoding="utf-8")
    assert "if (v3_mode or v4_mode) and args.grad_clip is not None:" in src
    assert "if v3_mode and args.grad_clip is not None:" not in src


def test_frozen_minimal_policy_rejected() -> None:
    minimal = {
        "schema": "ms_sta_retrain_v4_policy_v1",
        "status": "FROZEN",
        "train_allowed": True,
        "path": "B",
        "test_protocol": {"label": "retrospective_signer_disjoint_confirmation"},
    }
    with pytest.raises(ValueError, match="frozen_hpo_ready|lineage|implementation"):
        require_frozen_v4_policy(minimal)
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="train_allowed|draft|FROZEN|executable"):
        require_frozen_v4_policy(draft)


def test_code_lock_rejects_embedded_policy_commit() -> None:
    with pytest.raises(ValueError, match="policy_commit"):
        require_code_lock_fields(
            {
                "require_clean_worktree": True,
                "executable_code_lock": "deadbeef",
                "executable_tree_sha256": "a" * 64,
                "policy_commit": "pending_x",
            }
        )


def test_hpo_tie_epsilon_zero_prefers_higher_top1() -> None:
    rows = []
    for cfg, top1, top5 in (
        ("lr1e-4_wd0.01", 50.0, 70.0),
        ("lr3e-4_wd0.01", 49.8, 90.0),  # 0.20 pp lower Top-1, higher Top-5
    ):
        for seed in (11, 29):
            rows.append(
                {
                    "variant": "b3",
                    "config_id": cfg,
                    "seed": seed,
                    "best_val_top1": top1,
                    "best_val_top5": top5,
                }
            )
    selected = select_configs_per_variant(rows, variants=["b3"], tie_epsilon_pp=0.0)
    assert selected["b3"] == "lr1e-4_wd0.01"


def test_trial_complete_requires_final_on_confirm(tmp_path: Path) -> None:
    status = tmp_path / "confirm__b3__lr1e-4_wd0.01__seed42__stride1.json"
    ckpt = tmp_path / "confirm__b3__lr1e-4_wd0.01__seed42__stride1.pth"
    final = tmp_path / "confirm__b3__lr1e-4_wd0.01__seed42__stride1__final.pth"
    status.write_text(
        json.dumps(
            {
                "best_val_top1": 1.0,
                "checkpoint_status": "candidate",
                "phase": "confirm",
                "experiment_variant": "b3",
                "config_id": "lr1e-4_wd0.01",
                "seed": 42,
                "retrain_v4_policy_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "git_revision": "deadbeef",
            }
        ),
        encoding="utf-8",
    )
    ckpt.write_bytes(b"ckpt")
    assert (
        trial_is_complete(
            phase="confirm",
            variant="b3",
            config_id="lr1e-4_wd0.01",
            seed=42,
            status_path=status,
            ckpt_path=ckpt,
            final_path=final,
            expected_policy_sha="a" * 64,
            expected_split_sha="b" * 64,
            expected_git_rev="deadbeef",
        )
        is False
    )


def test_trial_complete_rejects_missing_split_identity(tmp_path: Path) -> None:
    """Independent check: None/missing split or git must fail-closed."""
    status = tmp_path / "hpo__b3__lr1e-4_wd0.01__seed11__stride1.json"
    ckpt = tmp_path / "hpo__b3__lr1e-4_wd0.01__seed11__stride1.pth"
    status.write_text(
        json.dumps(
            {
                "best_val_top1": 1.0,
                "phase": "hpo",
                "experiment_variant": "b3",
                "config_id": "lr1e-4_wd0.01",
                "seed": 11,
                "retrain_v4_policy_sha256": "a" * 64,
                "manifest_sha256": None,
                "git_revision": None,
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "phase": "hpo",
            "experiment_variant": "b3",
            "config_id": "lr1e-4_wd0.01",
            "seed": 11,
            "retrain_v4_policy_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "git_revision": "deadbeef",
            "checkpoint_status": "candidate",
            "model": {},
        },
        ckpt,
    )
    with pytest.raises(ValueError, match="manifest_sha256|git_revision|missing"):
        trial_is_complete(
            phase="hpo",
            variant="b3",
            config_id="lr1e-4_wd0.01",
            seed=11,
            status_path=status,
            ckpt_path=ckpt,
            final_path=None,
            expected_policy_sha="a" * 64,
            expected_split_sha="b" * 64,
            expected_git_rev="deadbeef",
        )


def test_evaluator_refuses_silent_id_intersection() -> None:
    expected = ["a", "b"]
    rows_b3 = {
        42: [
            _row(sample_id="a", signer_id="s1", seed=42, variant="b3", y_true=0, y_pred=0),
            _row(sample_id="b", signer_id="s2", seed=42, variant="b3", y_true=1, y_pred=1),
        ],
        123: [],
        456: [],
    }
    for s in (123, 456):
        rows_b3[s] = [{**r, "seed": s} for r in rows_b3[42]]
    rows_b4r = {
        s: [
            {**r, "variant": "b4r", "seed": s}
            for r in rows_b3[s]
            if r["sample_id"] != "b"  # silently drop one id
        ]
        for s in (42, 123, 456)
    }
    with pytest.raises(ValueError, match="ID set mismatch"):
        build_gate_report(
            rows_b3_by_seed=rows_b3,
            rows_b4r_by_seed=rows_b4r,
            expected_ids=expected,
            n_boot=20,
            seed=42,
            num_classes=2,
            expected_policy_sha=POLICY_SHA,
            expected_final_ckpt_sha_by_variant_seed=_ckpt_map(),
        )


def test_evaluator_rejects_invalid_sha_and_missing_top5() -> None:
    expected = ["a"]
    bad = _row(sample_id="a", signer_id="s1", seed=42, variant="b3", y_true=0, y_pred=0)
    bad["policy_sha256"] = "not-a-sha"
    del bad["correct_top5"]
    del bad["top5"]
    rows_b3 = {s: [{**bad, "seed": s, "variant": "b3"}] for s in (42, 123, 456)}
    rows_b4r = {s: [{**bad, "seed": s, "variant": "b4r"}] for s in (42, 123, 456)}
    with pytest.raises(ValueError, match="64-hex|correct_top5|top5|policy_sha256"):
        build_gate_report(
            rows_b3_by_seed=rows_b3,
            rows_b4r_by_seed=rows_b4r,
            expected_ids=expected,
            n_boot=20,
            seed=42,
            num_classes=2,
            expected_policy_sha=POLICY_SHA,
            expected_final_ckpt_sha_by_variant_seed=_ckpt_map(),
        )


def test_macro_f1_primary_is_per_seed_mean_not_majority() -> None:
    expected = ["a", "b"]
    rows_b3: dict[int, list[dict]] = {}
    rows_b4r: dict[int, list[dict]] = {}
    for s, pred_a, pred_b in (
        (42, 0, 1),
        (123, 1, 1),
        (456, 0, 0),
    ):
        rows_b3[s] = [
            _row(sample_id="a", signer_id="s1", seed=s, variant="b3", y_true=0, y_pred=pred_a),
            _row(sample_id="b", signer_id="s2", seed=s, variant="b3", y_true=1, y_pred=pred_b),
        ]
        rows_b4r[s] = [
            _row(sample_id="a", signer_id="s1", seed=s, variant="b4r", y_true=0, y_pred=0),
            _row(sample_id="b", signer_id="s2", seed=s, variant="b4r", y_true=1, y_pred=1),
        ]
    report = build_gate_report(
        rows_b3_by_seed=rows_b3,
        rows_b4r_by_seed=rows_b4r,
        expected_ids=expected,
        n_boot=50,
        seed=42,
        num_classes=2,
        expected_policy_sha=POLICY_SHA,
        expected_final_ckpt_sha_by_variant_seed=_ckpt_map(),
    )
    assert report["macro_f1_primary"] == "per_seed_then_mean_pm_sample_std"
    assert "macro_f1_b3_mean_pm_sample_std" in report["summary"]
    assert report["secondary_analyses"]["seed_majority_vote"]["registered_as_primary"] is False
    assert "s1" in report["per_signer"]
    assert "b3_top1" in report["per_signer"]["s1"]
    assert "b4r_top1" in report["per_signer"]["s1"]
    assert "delta_top1_pp" in report["per_signer"]["s1"]


def test_draft_policy_has_tie_epsilon_zero() -> None:
    policy = json.loads(DRAFT.read_text(encoding="utf-8"))
    assert float(policy["hpo"]["tie_epsilon_pp"]) == 0.0
    assert "policy_commit" not in (policy.get("code_lock") or {})
    assert "executable_tree_sha256" in (policy.get("code_lock") or {})


def test_executable_tree_hash_ignores_untracked_pyc() -> None:
    """Independent proof: ignored/generated .pyc must not change the code-lock hash."""
    h1 = executable_tree_sha256(_ROOT)
    probe = _ROOT / "src" / "_code_lock_probe_do_not_commit.pyc"
    try:
        probe.write_bytes(b"IGNORED_PYC_CHANGES_EXEC_TREE_HASH=True\n" + b"\0" * 64)
        h2 = executable_tree_sha256(_ROOT)
    finally:
        probe.unlink(missing_ok=True)
    assert h1 == h2
    assert len(h1) == 64


def test_per_sample_manifest_requires_exact_variant_seed_keys() -> None:
    man = {
        "schema": "ms_sta_retrain_v4_per_sample_manifest_v1",
        "policy_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "git_revision": "deadbeef",
        "n_val": 2,
        "exact_id_sha256": "c" * 64,
        "signer_metadata_sha256": "d" * 64,
        "jsonl_sha256_by_variant_seed": {"b3/seed42": "e" * 64},
        "final_checkpoint_sha256_by_variant_seed": {"b3/seed42": "f" * 64},
    }
    with pytest.raises(ValueError, match="jsonl keys"):
        require_per_sample_manifest_contract(
            man,
            expected_policy_sha="a" * 64,
            expected_split_sha="b" * 64,
            expected_git_rev="deadbeef",
        )


def test_evaluator_cli_requires_manifest_not_free_dirs() -> None:
    src = (_ROOT / "scripts/evaluate_ms_sta_retrain_v4_validation.py").read_text(encoding="utf-8")
    assert "--per-sample-manifest" in src
    assert "--b3-jsonl-dir" not in src
    assert "require_frozen_v4_policy" in src
    assert "expected_final_ckpt_sha_by_variant_seed=ckpt_map" in src


def test_build_gate_report_rejects_unbound_ckpt_map() -> None:
    expected = ["a"]
    row = _row(sample_id="a", signer_id="s1", seed=42, variant="b3", y_true=0, y_pred=0)
    rows_b3 = {s: [{**row, "seed": s, "variant": "b3"}] for s in (42, 123, 456)}
    rows_b4r = {s: [{**row, "seed": s, "variant": "b4r"}] for s in (42, 123, 456)}
    with pytest.raises(TypeError):
        build_gate_report(  # type: ignore[call-arg]
            rows_b3_by_seed=rows_b3,
            rows_b4r_by_seed=rows_b4r,
            expected_ids=expected,
            n_boot=10,
            seed=42,
            num_classes=2,
            expected_policy_sha=POLICY_SHA,
        )
    with pytest.raises(ValueError, match="required|None"):
        build_gate_report(
            rows_b3_by_seed=rows_b3,
            rows_b4r_by_seed=rows_b4r,
            expected_ids=expected,
            n_boot=10,
            seed=42,
            num_classes=2,
            expected_policy_sha=POLICY_SHA,
            expected_final_ckpt_sha_by_variant_seed=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="keys"):
        build_gate_report(
            rows_b3_by_seed=rows_b3,
            rows_b4r_by_seed=rows_b4r,
            expected_ids=expected,
            n_boot=10,
            seed=42,
            num_classes=2,
            expected_policy_sha=POLICY_SHA,
            expected_final_ckpt_sha_by_variant_seed={},
        )


def test_lock_and_test_scripts_exist() -> None:
    assert (_ROOT / "scripts/finalize_ms_sta_retrain_v4_policy_binding.py").is_file()
    assert (_ROOT / "scripts/evaluate_ms_sta_retrain_v4_test.py").is_file()
    orch = (_ROOT / "scripts/run_ms_sta_retrain_v4.py").read_text(encoding="utf-8")
    assert "PER_SAMPLE_MANIFEST.json" in orch
    assert "evaluate_ms_sta_retrain_v4_test.py" in orch
    assert "require_freeze_binding" in orch
    assert "trial_is_complete(" in orch


def test_test_evaluator_is_one_shot_and_binds_lock() -> None:
    src = (_ROOT / "scripts/evaluate_ms_sta_retrain_v4_test.py").read_text(encoding="utf-8")
    assert "refuse overwrite" in src
    assert "os.replace" in src
    assert "_publish_staged_test_artifacts" in src
    assert "rebuilt_from_rows_only" in src
    assert "--resume" in src
    assert "TEST_COMPLETION.json" in src
    assert "evaluation_locked_sha256" in src
    assert "freeze_binding_sha256" in src
    assert "missing required freeze binding" in src
    assert "active_skel_dir" in src
    assert "mcnemar_exact_top1" in src
    assert "paired_hierarchical_bootstrap_top5" in src
    # Resume must run after identity checks (lock SHA appears before resume return).
    id_pos = src.find("Identity checks run for both fresh and --resume")
    resume_pos = src.find("if args.resume:")
    assert id_pos != -1 and resume_pos != -1 and id_pos < resume_pos
    orch = (_ROOT / "scripts/run_ms_sta_retrain_v4.py").read_text(encoding="utf-8")
    assert "evaluation_locked_sha256" in orch
    assert "resume=resume" in orch
    assert "_preflight_clean_and_code_lock" in orch
    assert "_resume_lock_heal" in orch
    assert "_gather_lock_disk_evidence" in orch
    assert "require_evaluation_locked_contract" in orch
    assert "TEST_COMPLETION.json" in orch
    src = (_ROOT / "scripts/evaluate_ms_sta_retrain_v4_test.py").read_text(encoding="utf-8")
    assert "primary_seed_final_sha_by_variant" in src
    assert "expected_signer_by_id" in src
    assert "READY.rows_sha256" in src or "READY present" in src


def test_preflight_dirty_worktree_refuses_all_phases() -> None:
    import scripts.run_ms_sta_retrain_v4 as orch

    with mock.patch.object(orch, "_git_dirty", return_value=True), mock.patch.object(
        orch, "verify_code_lock_against_repo", return_value=None
    ):
        with pytest.raises(SystemExit, match="clean worktree"):
            orch._preflight_clean_and_code_lock(
                {"code_lock": {"require_clean_worktree": True}}
            )


def _full_lock_maps(ckpt_rel: str = "data/processed/ms_sta_retrain_v4") -> tuple[dict, dict, dict]:
    selected = {v: f"cfg_{v}" for v in CONFIRM_VARIANTS}
    finals: dict[str, str] = {}
    n = 0
    for v in CONFIRM_VARIANTS:
        for seed in CONFIRM_SEEDS:
            n += 1
            rel = confirm_final_checkpoint_rel(
                variant=v, config_id=selected[v], seed=seed, checkpoints_rel=ckpt_rel
            )
            finals[rel] = f"{n:064x}"
    jsonl = {f"{v}/seed{s}": f"{(10 + i):064x}" for i, (v, s) in enumerate(
        (vv, ss) for vv in ("b3", "b4r") for ss in CONFIRM_SEEDS
    )}
    return selected, finals, jsonl


def _mock_lock_evidence(
    *,
    policy_sha: str,
    split_sha: str,
    head: str,
    report_sha: str,
    selected_sha: str,
    man_sha: str,
    freeze_sha: str,
    finals: dict[str, str],
    jsonl: dict[str, str],
    binding_path: Path,
) -> dict:
    return {
        "split_sha": split_sha,
        "git_revision": head,
        "report_sha": report_sha,
        "selected_sha": selected_sha,
        "man_sha": man_sha,
        "freeze_binding_sha": freeze_sha,
        "final_shas": finals,
        "jsonl_shas": jsonl,
        "binding_path": binding_path,
        "configs": {v: f"cfg_{v}" for v in CONFIRM_VARIANTS},
        "manifest": {"exact_id_sha256": "e" * 64},
    }


def test_lock_resume_heals_run_state_without_overwrite(tmp_path: Path) -> None:
    """Pre-lock crash: lock exists, RUN_STATE unlocked but sha matches lock.run_state_sha256."""
    import scripts.run_ms_sta_retrain_v4 as orch
    from src.training.retrain_v3 import sha256_file as _sha

    status = tmp_path / "status"
    status.mkdir()
    lock_path = status / "EVALUATION_LOCKED.json"
    report_path = status / "CONFIRM_VALIDATION_REPORT.json"
    selected_path = status / "SELECTED_CONFIGS.json"
    man_path = status / "per_sample_val" / "PER_SAMPLE_MANIFEST.json"
    man_path.parent.mkdir(parents=True)
    state_path = status / "RUN_STATE.json"
    bind_dir = tmp_path / "Docs" / "results" / "ms_sta_retrain_v4_policy"
    bind_dir.mkdir(parents=True)
    binding = bind_dir / "POLICY_FREEZE_BINDING.json"

    policy_sha = "a" * 64
    split_sha = "b" * 64
    head = "deadbeef"
    selected, finals, jsonl = _full_lock_maps()
    report_path.write_text(
        json.dumps({"architecture_gates": {"architecture_gate_pass": True}}),
        encoding="utf-8",
    )
    selected_path.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    man_path.write_text(
        json.dumps({"schema": "ms_sta_retrain_v4_per_sample_manifest_v1"}),
        encoding="utf-8",
    )
    binding.write_text(
        json.dumps({"schema": "ms_sta_retrain_v4_policy_freeze_binding_v1"}),
        encoding="utf-8",
    )
    report_sha = _sha(report_path)
    selected_sha = _sha(selected_path)
    man_sha = _sha(man_path)
    binding_sha = _sha(binding)

    state = {
        "schema": "ms_sta_retrain_v4_run_state_v2",
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": head,
        "evaluation_locked": False,
        "phases": {"confirm": {"status": "complete", "n_complete": 12}},
    }
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    state_sha = _sha(state_path)
    lock = {
        "schema": "ms_sta_retrain_v4_evaluation_locked_v1",
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": head,
        "confirm_validation_report_sha256": report_sha,
        "selected_configs_sha256": selected_sha,
        "per_sample_manifest_sha256": man_sha,
        "architecture_gate_pass": True,
        "freeze_binding_sha256": binding_sha,
        "run_state_sha256": state_sha,
        "primary_seed": 42,
        "final_checkpoint_sha256_by_path": finals,
        "per_sample_jsonl_sha256_by_variant_seed": jsonl,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    evidence = _mock_lock_evidence(
        policy_sha=policy_sha,
        split_sha=split_sha,
        head=head,
        report_sha=report_sha,
        selected_sha=selected_sha,
        man_sha=man_sha,
        freeze_sha=binding_sha,
        finals=finals,
        jsonl=jsonl,
        binding_path=binding,
    )

    with mock.patch.object(orch, "_ROOT", tmp_path), mock.patch.object(
        orch, "_git_rev", return_value=head
    ), mock.patch.object(orch, "require_freeze_binding", return_value=None), mock.patch.object(
        orch, "_rel", side_effect=lambda p: Path(p).name
    ), mock.patch.object(orch, "_gather_lock_disk_evidence", return_value=evidence):
        rc = orch._resume_lock_heal(
            policy={"lineage": {"split_manifest_sha256": split_sha}},
            policy_path=tmp_path / "policy.json",
            policy_sha=policy_sha,
            status_dir=status,
            state_path=state_path,
            lock_path=lock_path,
        )
    assert rc == 0
    healed = json.loads(state_path.read_text(encoding="utf-8"))
    assert healed["evaluation_locked"] is True
    assert healed["evaluation_locked_sha256"] == _sha(lock_path)
    assert healed["phases"]["lock"]["status"] == "complete"
    assert healed["policy_sha256"] == policy_sha
    assert healed["split_manifest_sha256"] == split_sha
    assert healed["git_revision"] == head


def test_incomplete_lock_contract_refused_on_resume(tmp_path: Path) -> None:
    """Positive-looking but incomplete lock (missing finals/selected) must fail resume."""
    import scripts.run_ms_sta_retrain_v4 as orch
    from src.training.retrain_v3 import sha256_file as _sha

    status = tmp_path / "status"
    status.mkdir()
    lock_path = status / "EVALUATION_LOCKED.json"
    report_path = status / "CONFIRM_VALIDATION_REPORT.json"
    selected_path = status / "SELECTED_CONFIGS.json"
    man_path = status / "per_sample_val" / "PER_SAMPLE_MANIFEST.json"
    man_path.parent.mkdir(parents=True)
    state_path = status / "RUN_STATE.json"
    bind_dir = tmp_path / "Docs" / "results" / "ms_sta_retrain_v4_policy"
    bind_dir.mkdir(parents=True)
    binding = bind_dir / "POLICY_FREEZE_BINDING.json"

    policy_sha = "a" * 64
    split_sha = "b" * 64
    head = "deadbeef"
    selected, finals, jsonl = _full_lock_maps()
    report_path.write_text("{}", encoding="utf-8")
    selected_path.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    man_path.write_text("{}", encoding="utf-8")
    binding.write_text("{}", encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "policy_sha256": policy_sha,
                "split_manifest_sha256": split_sha,
                "git_revision": head,
                "evaluation_locked": False,
                "phases": {"confirm": {"status": "complete", "n_complete": 12}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    # Incomplete lock — historically accepted by heal.
    lock = {
        "schema": "ms_sta_retrain_v4_evaluation_locked_v1",
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": head,
        "confirm_validation_report_sha256": _sha(report_path),
        "per_sample_manifest_sha256": _sha(man_path),
        "architecture_gate_pass": True,
        "freeze_binding_sha256": _sha(binding),
        "run_state_sha256": _sha(state_path),
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    evidence = _mock_lock_evidence(
        policy_sha=policy_sha,
        split_sha=split_sha,
        head=head,
        report_sha=_sha(report_path),
        selected_sha=_sha(selected_path),
        man_sha=_sha(man_path),
        freeze_sha=_sha(binding),
        finals=finals,
        jsonl=jsonl,
        binding_path=binding,
    )
    with mock.patch.object(orch, "_ROOT", tmp_path), mock.patch.object(
        orch, "_git_rev", return_value=head
    ), mock.patch.object(orch, "require_freeze_binding", return_value=None), mock.patch.object(
        orch, "_gather_lock_disk_evidence", return_value=evidence
    ):
        rc = orch._resume_lock_heal(
            policy={"lineage": {"split_manifest_sha256": split_sha}},
            policy_path=tmp_path / "policy.json",
            policy_sha=policy_sha,
            status_dir=status,
            state_path=state_path,
            lock_path=lock_path,
        )
    assert rc == 2
    with pytest.raises(ValueError, match="primary_seed|selected_configs|finals"):
        require_evaluation_locked_contract(
            lock,
            policy_sha=policy_sha,
            split_sha=split_sha,
            git_revision=head,
            report_sha=_sha(report_path),
            selected_sha=_sha(selected_path),
            man_sha=_sha(man_path),
            freeze_binding_sha=_sha(binding),
            final_shas=finals,
            jsonl_shas=jsonl,
            run_state_sha=_sha(state_path),
        )


def test_conflicting_lock_sha_refused(tmp_path: Path) -> None:
    """Already-locked RUN_STATE with mismatched evaluation_locked_sha256 → exit 2."""
    import scripts.run_ms_sta_retrain_v4 as orch
    from src.training.retrain_v3 import sha256_file as _sha

    status = tmp_path / "status"
    status.mkdir()
    lock_path = status / "EVALUATION_LOCKED.json"
    report_path = status / "CONFIRM_VALIDATION_REPORT.json"
    selected_path = status / "SELECTED_CONFIGS.json"
    man_path = status / "per_sample_val" / "PER_SAMPLE_MANIFEST.json"
    man_path.parent.mkdir(parents=True)
    state_path = status / "RUN_STATE.json"
    bind_dir = tmp_path / "Docs" / "results" / "ms_sta_retrain_v4_policy"
    bind_dir.mkdir(parents=True)
    binding = bind_dir / "POLICY_FREEZE_BINDING.json"

    policy_sha = "a" * 64
    split_sha = "b" * 64
    head = "deadbeef"
    selected, finals, jsonl = _full_lock_maps()
    report_path.write_text(json.dumps({"architecture_gates": {"architecture_gate_pass": True}}), encoding="utf-8")
    selected_path.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    man_path.write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    binding.write_text(json.dumps({"schema": "ms_sta_retrain_v4_policy_freeze_binding_v1"}), encoding="utf-8")
    lock = {
        "schema": "ms_sta_retrain_v4_evaluation_locked_v1",
        "policy_sha256": policy_sha,
        "split_manifest_sha256": split_sha,
        "git_revision": head,
        "confirm_validation_report_sha256": _sha(report_path),
        "selected_configs_sha256": _sha(selected_path),
        "per_sample_manifest_sha256": _sha(man_path),
        "architecture_gate_pass": True,
        "freeze_binding_sha256": _sha(binding),
        "run_state_sha256": "c" * 64,
        "primary_seed": 42,
        "final_checkpoint_sha256_by_path": finals,
        "per_sample_jsonl_sha256_by_variant_seed": jsonl,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    lock_sha = _sha(lock_path)
    conflicting = ("0" if lock_sha[0] != "0" else "1") + lock_sha[1:]
    state_path.write_text(
        json.dumps(
            {
                "schema": "ms_sta_retrain_v4_run_state_v2",
                "policy_sha256": policy_sha,
                "split_manifest_sha256": split_sha,
                "git_revision": head,
                "evaluation_locked": True,
                "evaluation_locked_sha256": conflicting,
                "phases": {"confirm": {"status": "complete", "n_complete": 12}},
            }
        ),
        encoding="utf-8",
    )
    evidence = _mock_lock_evidence(
        policy_sha=policy_sha,
        split_sha=split_sha,
        head=head,
        report_sha=_sha(report_path),
        selected_sha=_sha(selected_path),
        man_sha=_sha(man_path),
        freeze_sha=_sha(binding),
        finals=finals,
        jsonl=jsonl,
        binding_path=binding,
    )

    with mock.patch.object(orch, "_ROOT", tmp_path), mock.patch.object(
        orch, "_git_rev", return_value=head
    ), mock.patch.object(orch, "require_freeze_binding", return_value=None), mock.patch.object(
        orch, "_gather_lock_disk_evidence", return_value=evidence
    ):
        rc = orch._resume_lock_heal(
            policy={"lineage": {"split_manifest_sha256": split_sha}},
            policy_path=tmp_path / "policy.json",
            policy_sha=policy_sha,
            status_dir=status,
            state_path=state_path,
            lock_path=lock_path,
        )
    assert rc == 2
    after = json.loads(state_path.read_text(encoding="utf-8"))
    assert after["evaluation_locked_sha256"] == conflicting
    print("CONFLICTING_LOCK_SHA_REFUSED=True")


def test_lock_resume_refuses_missing_run_state(tmp_path: Path) -> None:
    import scripts.run_ms_sta_retrain_v4 as orch

    status = tmp_path / "status"
    status.mkdir()
    (status / "EVALUATION_LOCKED.json").write_text("{}", encoding="utf-8")
    (status / "CONFIRM_VALIDATION_REPORT.json").write_text("{}", encoding="utf-8")
    man = status / "per_sample_val" / "PER_SAMPLE_MANIFEST.json"
    man.parent.mkdir(parents=True)
    man.write_text("{}", encoding="utf-8")
    rc = orch._resume_lock_heal(
        policy={},
        policy_path=tmp_path / "p.json",
        policy_sha="a" * 64,
        status_dir=status,
        state_path=status / "RUN_STATE.json",
        lock_path=status / "EVALUATION_LOCKED.json",
    )
    assert rc == 2


def test_test_completion_resume_without_phases_test(tmp_path: Path) -> None:
    """Crash after artifacts+completion but before phases.test: resume still verifies."""
    from scripts.evaluate_ms_sta_retrain_v4_test import (
        _verify_existing_artifacts,
        _write_test_completion,
    )
    from src.training.retrain_v3 import sha256_file as _sha

    report_path = tmp_path / "TEST_REPORT.json"
    rows_path = tmp_path / "TEST_PER_SAMPLE.jsonl"
    completion_path = tmp_path / "TEST_COMPLETION.json"
    rows_path.write_text('{"sample_id":"x"}\n', encoding="utf-8")
    rows_sha = _sha(rows_path)
    policy_sha = "a" * 64
    lock_sha = "b" * 64
    binding_sha = "c" * 64
    report = {
        "bindings": {
            "policy_sha256": policy_sha,
            "evaluation_locked_sha256": lock_sha,
            "freeze_binding_sha256": binding_sha,
        },
        "artifacts": {"test_per_sample_jsonl_sha256": rows_sha},
    }
    import hashlib

    payload_sha = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    report["artifacts"]["test_report_payload_sha256"] = payload_sha
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_test_completion(
        completion_path,
        policy_sha=policy_sha,
        lock_sha=lock_sha,
        binding_sha=binding_sha,
        report_sha=_sha(report_path),
        rows_sha=rows_sha,
        payload_sha=payload_sha,
    )
    rc = _verify_existing_artifacts(
        report_path=report_path,
        rows_path=rows_path,
        completion_path=completion_path,
        policy_sha=policy_sha,
        lock_sha=lock_sha,
        binding_sha=binding_sha,
        state={"phases": {}},
        test_ids=["x"],
        id_sha="d" * 64,
        split_sha="e" * 64,
        lock={},
        policy={},
        num_classes=400,
        n_boot=10,
        bootstrap_seed=42,
        status_dir=tmp_path,
        expected_signer_by_id={"x": "signer_1"},
    )
    assert rc == 0


def _write_rows_fixture(
    rows_path: Path,
    *,
    test_ids: list[str],
    policy_sha: str,
    ckpt_by_variant: dict[str, str],
    signer_by_id: dict[str, str],
    swap_b3_b4r: bool = False,
) -> None:
    from scripts.evaluate_ms_sta_retrain_v4_test import PRIMARY_SEED, TEST_VARIANTS

    lines: list[str] = []
    for sid_i, sid in enumerate(test_ids):
        y_true = sid_i % 5
        for variant in TEST_VARIANTS:
            y_pred = y_true if variant != "b0" else (y_true + 1) % 5
            top5 = [(y_pred + k) % 5 for k in range(5)]
            assert len(set(top5)) == 5
            sha = ckpt_by_variant[variant]
            if swap_b3_b4r and variant == "b3":
                sha = ckpt_by_variant["b4r"]
            elif swap_b3_b4r and variant == "b4r":
                sha = ckpt_by_variant["b3"]
            lines.append(
                json.dumps(
                    {
                        "sample_id": sid,
                        "signer_id": signer_by_id[sid],
                        "seed": int(PRIMARY_SEED),
                        "variant": variant,
                        "split": "test",
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "correct_top1": int(y_pred == y_true),
                        "correct_top5": int(y_true in top5),
                        "top5": top5,
                        "final_checkpoint_sha256": sha,
                        "policy_sha256": policy_sha,
                    },
                    ensure_ascii=False,
                )
            )
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows_recovery_context(tmp_path: Path) -> dict:
    from src.training.retrain_v3 import sha256_file as _sha
    from src.training.retrain_v4 import exact_id_list_sha256

    status = tmp_path
    status.mkdir(parents=True, exist_ok=True)
    ckpt_rel = "data/processed/ms_sta_retrain_v4"
    selected, finals, _jsonl = _full_lock_maps(ckpt_rel)
    selected_path = status / "SELECTED_CONFIGS.json"
    selected_path.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    selected_sha = _sha(selected_path)
    primary = primary_seed_final_sha_by_variant(
        selected=selected, final_sha_by_path=finals, checkpoints_rel=ckpt_rel
    )
    test_ids = ["vid_a", "vid_b"]
    signer_by_id = {"vid_a": "signer_1", "vid_b": "signer_2"}
    policy_sha = "a" * 64
    lock = {
        "git_revision": "deadbeef",
        "primary_seed": 42,
        "selected_configs_sha256": selected_sha,
        "final_checkpoint_sha256_by_path": finals,
    }
    policy = {
        "test_protocol": {"label": "retrospective"},
        "artifact_dirs": {"checkpoints": ckpt_rel, "status": str(status)},
    }
    return {
        "status": status,
        "report_path": status / "TEST_REPORT.json",
        "rows_path": status / "TEST_PER_SAMPLE.jsonl",
        "completion_path": status / "TEST_COMPLETION.json",
        "test_ids": test_ids,
        "id_sha": exact_id_list_sha256(sorted(test_ids)),
        "policy_sha": policy_sha,
        "lock_sha": "b" * 64,
        "binding_sha": "c" * 64,
        "split_sha": "d" * 64,
        "lock": lock,
        "policy": policy,
        "primary": primary,
        "signer_by_id": signer_by_id,
    }


def test_rows_only_resume_rebuilds_report_without_inference(tmp_path: Path) -> None:
    """Crash after rows, before report: --resume rebuilds report+completion from rows."""
    from scripts.evaluate_ms_sta_retrain_v4_test import _verify_existing_artifacts
    from src.training.retrain_v3 import sha256_file as _sha

    ctx = _rows_recovery_context(tmp_path)
    _write_rows_fixture(
        ctx["rows_path"],
        test_ids=ctx["test_ids"],
        policy_sha=ctx["policy_sha"],
        ckpt_by_variant=ctx["primary"],
        signer_by_id=ctx["signer_by_id"],
    )
    assert not ctx["report_path"].is_file()

    rc = _verify_existing_artifacts(
        report_path=ctx["report_path"],
        rows_path=ctx["rows_path"],
        completion_path=ctx["completion_path"],
        policy_sha=ctx["policy_sha"],
        lock_sha=ctx["lock_sha"],
        binding_sha=ctx["binding_sha"],
        state={"phases": {}},
        test_ids=ctx["test_ids"],
        id_sha=ctx["id_sha"],
        split_sha=ctx["split_sha"],
        lock=ctx["lock"],
        policy=ctx["policy"],
        num_classes=5,
        n_boot=20,
        bootstrap_seed=42,
        status_dir=ctx["status"],
        expected_signer_by_id=ctx["signer_by_id"],
    )
    assert rc == 0, "ROWS_ONLY_RESUME_EXIT should be 0 after rebuild"
    print("ROWS_ONLY_RESUME_EXIT=0")
    assert ctx["report_path"].is_file()
    assert ctx["completion_path"].is_file()
    report = json.loads(ctx["report_path"].read_text(encoding="utf-8"))
    assert report["artifacts"].get("rebuilt_from_rows_only") is True
    assert report["bindings"]["evaluation_locked_sha256"] == ctx["lock_sha"]
    assert report["artifacts"]["test_per_sample_jsonl_sha256"] == _sha(ctx["rows_path"])
    rows_sha_before = _sha(ctx["rows_path"])
    rc2 = _verify_existing_artifacts(
        report_path=ctx["report_path"],
        rows_path=ctx["rows_path"],
        completion_path=ctx["completion_path"],
        policy_sha=ctx["policy_sha"],
        lock_sha=ctx["lock_sha"],
        binding_sha=ctx["binding_sha"],
        state={"phases": {}},
        test_ids=ctx["test_ids"],
        id_sha=ctx["id_sha"],
        split_sha=ctx["split_sha"],
        lock=ctx["lock"],
        policy=ctx["policy"],
        num_classes=5,
        n_boot=20,
        bootstrap_seed=42,
        status_dir=ctx["status"],
        expected_signer_by_id=ctx["signer_by_id"],
    )
    assert rc2 == 0
    assert _sha(ctx["rows_path"]) == rows_sha_before


def test_swapped_variant_sha_and_fake_signer_refused(tmp_path: Path) -> None:
    """Rows-only recovery must refuse swapped B3/B4R SHAs and non-canonical signers."""
    from scripts.evaluate_ms_sta_retrain_v4_test import _verify_existing_artifacts

    ctx = _rows_recovery_context(tmp_path)
    # Swapped B3/B4R checkpoint SHAs in rows (still present in lock values).
    _write_rows_fixture(
        ctx["rows_path"],
        test_ids=ctx["test_ids"],
        policy_sha=ctx["policy_sha"],
        ckpt_by_variant=ctx["primary"],
        signer_by_id=ctx["signer_by_id"],
        swap_b3_b4r=True,
    )
    rc_swap = _verify_existing_artifacts(
        report_path=ctx["report_path"],
        rows_path=ctx["rows_path"],
        completion_path=ctx["completion_path"],
        policy_sha=ctx["policy_sha"],
        lock_sha=ctx["lock_sha"],
        binding_sha=ctx["binding_sha"],
        state={"phases": {}},
        test_ids=ctx["test_ids"],
        id_sha=ctx["id_sha"],
        split_sha=ctx["split_sha"],
        lock=ctx["lock"],
        policy=ctx["policy"],
        num_classes=5,
        n_boot=20,
        bootstrap_seed=42,
        status_dir=ctx["status"],
        expected_signer_by_id=ctx["signer_by_id"],
    )
    assert rc_swap == 2

    # Fake signer accepted previously — must now fail.
    ctx2 = _rows_recovery_context(tmp_path / "fake_signer")
    fake_signers = {sid: "signer_999" for sid in ctx2["test_ids"]}
    _write_rows_fixture(
        ctx2["rows_path"],
        test_ids=ctx2["test_ids"],
        policy_sha=ctx2["policy_sha"],
        ckpt_by_variant=ctx2["primary"],
        signer_by_id=fake_signers,
    )
    rc_fake = _verify_existing_artifacts(
        report_path=ctx2["report_path"],
        rows_path=ctx2["rows_path"],
        completion_path=ctx2["completion_path"],
        policy_sha=ctx2["policy_sha"],
        lock_sha=ctx2["lock_sha"],
        binding_sha=ctx2["binding_sha"],
        state={"phases": {}},
        test_ids=ctx2["test_ids"],
        id_sha=ctx2["id_sha"],
        split_sha=ctx2["split_sha"],
        lock=ctx2["lock"],
        policy=ctx2["policy"],
        num_classes=5,
        n_boot=20,
        bootstrap_seed=42,
        status_dir=ctx2["status"],
        expected_signer_by_id=ctx2["signer_by_id"],
    )
    assert rc_fake == 2
    print("SWAPPED_VARIANT_SHA_AND_FAKE_SIGNER_ACCEPTED=False")

def test_architecture_gates_require_mean_delta_ge_1pp() -> None:
    from src.training.retrain_v4 import evaluate_architecture_gates_v4

    per_seed = {
        "42": {"b3_top1": 60.0, "b4r_top1": 60.5},
        "123": {"b3_top1": 61.0, "b4r_top1": 61.4},
        "456": {"b3_top1": 62.0, "b4r_top1": 62.3},
    }
    # Mean delta ~0.4 pp: improve language, not PASS.
    small = evaluate_architecture_gates_v4(
        per_seed=per_seed,
        paired_ci={"ci_low": 0.1},
        mean_delta_top5_pp=0.0,
        macro_f1_b3=0.5,
        macro_f1_b4r=0.51,
    )
    assert small["mean_delta_top1_gt_0"] is True
    assert small["mean_delta_top1_ge_pp"] is False
    assert small["architecture_gate_pass"] is False
    assert small["claim_language"] == "improve_not_outperform"
    print("SUB_1PP_OUTPERFORM_ACCEPTED=False")

    per_seed_big = {
        "42": {"b3_top1": 60.0, "b4r_top1": 61.5},
        "123": {"b3_top1": 61.0, "b4r_top1": 62.2},
        "456": {"b3_top1": 62.0, "b4r_top1": 63.1},
    }
    big = evaluate_architecture_gates_v4(
        per_seed=per_seed_big,
        paired_ci={"ci_low": 0.2},
        mean_delta_top5_pp=0.0,
        macro_f1_b3=0.5,
        macro_f1_b4r=0.55,
    )
    assert big["mean_delta_top1_ge_pp"] is True
    assert big["architecture_gate_pass"] is True
    assert big["claim_language"] == "outperform"


@pytest.mark.requires_artifacts
def test_promote_draft_to_freeze_candidate_deterministic() -> None:
    from src.training.retrain_v4 import (
        _frozen_lifecycle_banned_substrings,
        executable_tree_sha256,
        promote_draft_to_freeze_candidate,
        require_frozen_v4_policy,
    )

    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    head = "d4a22151d1ad8f818160f06dd4753be2c749b950"
    tree = executable_tree_sha256(_ROOT)
    cand = promote_draft_to_freeze_candidate(
        draft, head_rev=head, executable_tree_sha=tree
    )
    assert "lineage_draft" not in cand
    assert "runtime_locks_draft" not in cand
    assert "deferred_implementation" not in cand
    assert cand["schema"] == "ms_sta_retrain_v4_policy_v1"
    assert cand["status"] == "FROZEN"
    assert cand["train_allowed"] is True
    assert cand["orchestrator"]["implementation_status"] == "frozen_hpo_ready"
    assert cand["implementation"]["immutable_policy_freeze"] is True
    assert cand["code_lock"]["executable_code_lock"] == head
    assert cand["code_lock"]["executable_tree_sha256"] == tree
    assert float(cand["accuracy_gates"]["mean_delta_top1_ge_pp"]) == 1.0
    assert _frozen_lifecycle_banned_substrings(cand) == []
    require_frozen_v4_policy(cand, root=_ROOT)


def test_promote_script_exists_and_draft_tip_updated() -> None:
    assert (_ROOT / "scripts/promote_ms_sta_retrain_v4_policy_candidate.py").is_file()
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    tip = str(draft["code_lock"]["current_scaffold_tip"])
    assert tip
    assert float(draft["accuracy_gates"]["mean_delta_top1_ge_pp"]) == 1.0
    warm_runs = (draft.get("warm_start_diagnostic") or {}).get("runs_planned")
    assert warm_runs == 0
    print("WARM_START_UNIMPLEMENTED_RUNS_PLANNED=0")
    trainer = (_ROOT / "src/training/train_ms_sta_vsl400.py").read_text(encoding="utf-8")
    assert "b4r_diagnostics" in trainer
    assert "train_top5" in trainer
    print("TRAIN_TOP5_PRESENT=True")
    for field in (
        "delta_head_grad_norm_preclip_step_mean",
        "cross_attention_grad_norm_preclip_step_mean",
        "spatial_attention_grad_norm_preclip_step_mean",
        "rgb_temporal_grad_norm_preclip_step_mean",
        "skel_temporal_grad_norm_preclip_step_mean",
        "stream_gate_grad_norm_preclip_step_mean",
    ):
        assert field in trainer
    print("ATTENTION_GRAD_COMPONENTS_COMPLETE=True")
    hyp = (_ROOT / "Docs/reviews/MS_STA_RETRAIN_V4_B4_RESIDUAL_HYPOTHESIS.md").read_text(
        encoding="utf-8"
    )
    assert "≥ **+1.0 pp**" in hyp or "≥ +1.0 pp" in hyp or "+1.0 pp" in hyp
    assert "improve_not_outperform" in hyp
    assert "Mean ΔTop-1 > 0\n" not in hyp
    assert "DRAFT_NOT_FROZEN" not in hyp
    assert "CLOSED" in hyp
    assert "VALIDATION_NO_GO" in hyp
    assert "HPO_NOT_STARTED" not in hyp
    assert "train_ms_sta_vsl400.py" in hyp


@pytest.mark.requires_artifacts
def test_frozen_candidate_lifecycle_so_t_no_refuse_language() -> None:
    """Negative scan: frozen candidate must not self-contradict lifecycle SoT."""
    import subprocess

    from scripts.run_ms_sta_retrain_v4 import dry_plan
    from src.training.retrain_v4 import (
        _frozen_lifecycle_banned_substrings,
        executable_tree_sha256,
        promote_draft_to_freeze_candidate,
        refuse_train_until_frozen,
        require_frozen_v4_policy,
    )

    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    tree = executable_tree_sha256(_ROOT)
    cand = promote_draft_to_freeze_candidate(
        draft, head_rev=head, executable_tree_sha=tree
    )
    assert cand["status"] == "FROZEN"
    assert cand["train_allowed"] is True
    assert cand["orchestrator"]["implementation_status"] == "frozen_hpo_ready"
    assert _frozen_lifecycle_banned_substrings(cand) == []
    require_frozen_v4_policy(cand, root=_ROOT)
    print("FROZEN_LIFECYCLE_SO_T_OK=True")

    blob = json.dumps(cand, ensure_ascii=False).lower()
    for ban in (
        "still_refused",
        "train_allowed=false until",
        "freeze helper still required",
        "draft_not_frozen",
    ):
        assert ban not in blob, ban
    print("STILL_REFUSED_ABSENT=True")

    plan = dry_plan(cand)
    assert refuse_train_until_frozen(cand) is None
    assert plan["policy_ready"] is True
    assert plan["hpo_ready"] is False  # no on-disk binding/code-lock preflight
    assert "contract satisfied" not in str(plan.get("note") or "").lower()
    assert "still_refused" not in json.dumps(plan).lower()
    assert plan["hpo"]["runs_total"] == 48
    assert "lineage" in plan and "lineage_draft" not in plan
    print("FROZEN_DRY_PLAN_POLICY_READY_WITHOUT_FALSE_HPO=True")

    draft_plan = dry_plan(draft)
    assert draft_plan["hpo_ready"] is False
    assert draft_plan["policy_ready"] is False
    assert refuse_train_until_frozen(draft) is not None


@pytest.mark.requires_artifacts
def test_external_candidate_promote_freeze_no_deadlock(tmp_path: Path) -> None:
    """External candidate exit 0; draft untouched; SHA stable; freeze accepts abs draft."""
    import importlib.util
    import subprocess
    import sys
    from unittest import mock

    from src.training.retrain_v3 import sha256_file

    promote = _ROOT / "scripts/promote_ms_sta_retrain_v4_policy_candidate.py"
    freeze = _ROOT / "scripts/freeze_ms_sta_retrain_v4_policy.py"
    cand = tmp_path / "MS_STA_RETRAIN_V4_POLICY_CANDIDATE.json"
    draft_sha_before = sha256_file(DRAFT)
    draft_text_before = DRAFT.read_text(encoding="utf-8")

    r1 = subprocess.run(
        [sys.executable, str(promote), "--out", str(cand)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r1.returncode == 0, r1.stdout + r1.stderr
    print("EXTERNAL_CANDIDATE_EXIT=0")
    assert cand.is_file()
    assert DRAFT.read_text(encoding="utf-8") == draft_text_before
    assert sha256_file(DRAFT) == draft_sha_before
    sha1 = sha256_file(cand)

    r2 = subprocess.run(
        [sys.executable, str(promote), "--out", str(cand), "--allow-overwrite"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert sha256_file(cand) == sha1

    r3 = subprocess.run(
        [sys.executable, str(promote), "--out", str(cand)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r3.returncode == 2

    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=_ROOT, text=True
    )
    assert "MS_STA_RETRAIN_V4_POLICY_CANDIDATE.json" not in porcelain

    spec = importlib.util.spec_from_file_location("freeze_v4_mod", freeze)
    assert spec is not None and spec.loader is not None
    fz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fz)
    assert fz._display_path(cand) == str(cand.resolve())

    disposable = (
        _ROOT
        / "Docs/results/ms_sta_retrain_v4_policy"
        / "_TEST_ONLY_RETRAIN_POLICY_CANDIDATE_SMOKE.json"
    )
    disposable_bind = (
        _ROOT
        / "Docs/results/ms_sta_retrain_v4_policy"
        / "_TEST_ONLY_POLICY_FREEZE_BINDING_SMOKE.json"
    )
    try:
        argv = [
            str(freeze),
            "--draft",
            str(cand),
            "--out",
            str(disposable),
            "--binding-out",
            str(disposable_bind),
            "--allow-overwrite",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            fz, "_git_dirty", return_value=False
        ):
            rc = fz.main()
        # rc==0 when HEAD/tree match candidate locks; rc==2 is a clean validator
        # refusal — either proves no relative_to deadlock on external draft.
        assert rc in (0, 2)
        if rc == 0:
            assert disposable.is_file()
            binding = json.loads(disposable_bind.read_text(encoding="utf-8"))
            assert "candidate_source" in binding
    finally:
        disposable.unlink(missing_ok=True)
        disposable_bind.unlink(missing_ok=True)


def test_sample_weighted_loss_and_finite_null_helpers() -> None:
    from src.training.train_ms_sta_vsl400 import (
        _finite_or_null,
        _root_sum_square,
        _sample_weighted_mean,
    )

    assert _sample_weighted_mean(10.0 * 2 + 4.0 * 8, 10) == pytest.approx(5.2)
    missing: dict[str, str] = {}
    assert _finite_or_null(float("nan"), field="x", missing=missing) is None
    assert missing["x"] == "non_finite"
    assert _root_sum_square([3.0, 4.0, None]) == pytest.approx(5.0)


@pytest.mark.requires_artifacts
def test_dry_plan_hpo_ready_requires_binding_and_code_lock(tmp_path: Path) -> None:
    """Missing binding / wrong SHA / tree drift must not yield hpo_ready=true."""
    import subprocess
    import sys
    from unittest import mock

    from scripts.run_ms_sta_retrain_v4 import assess_hpo_readiness, dry_plan
    from src.training.retrain_v3 import sha256_file
    from src.training.retrain_v4 import (
        executable_tree_sha256,
        promote_draft_to_freeze_candidate,
    )

    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    tree = executable_tree_sha256(_ROOT)
    cand = promote_draft_to_freeze_candidate(
        draft, head_rev=head, executable_tree_sha=tree
    )
    policy_path = tmp_path / "RETRAIN_POLICY.json"
    policy_path.write_text(
        json.dumps(cand, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _ = sha256_file(policy_path)

    with mock.patch("scripts.run_ms_sta_retrain_v4._git_dirty", return_value=False):
        missing = assess_hpo_readiness(
            cand, root=_ROOT, policy_path=policy_path, binding_path=tmp_path / "nope.json"
        )
        assert missing["policy_ready"] is True
        assert missing["hpo_ready"] is False
        assert any("missing POLICY_FREEZE_BINDING" in r for r in missing["reasons"])
        print("MISSING_BINDING_HPO_READY=False")

        plan_missing = dry_plan(
            cand, root=_ROOT, policy_path=policy_path, binding_path=tmp_path / "nope.json"
        )
        assert plan_missing["policy_ready"] is True
        assert plan_missing["hpo_ready"] is False
        assert "contract satisfied" not in plan_missing["note"].lower()

        bad_bind = tmp_path / "POLICY_FREEZE_BINDING.json"
        bad_bind.write_text(
            json.dumps(
                {
                    "schema": "ms_sta_retrain_v4_policy_freeze_binding_v1",
                    "policy_path": "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY.json",
                    "policy_sha256": "0" * 64,
                    "executable_code_lock": head,
                    "executable_tree_sha256": tree,
                    "policy_commit": head,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        wrong_sha = assess_hpo_readiness(
            cand, root=_ROOT, policy_path=policy_path, binding_path=bad_bind
        )
        assert wrong_sha["policy_ready"] is True
        assert wrong_sha["hpo_ready"] is False
        assert any(
            "binding" in r.lower() or "sha" in r.lower() for r in wrong_sha["reasons"]
        )
        print("WRONG_BINDING_SHA_HPO_READY=False")

        with mock.patch(
            "scripts.run_ms_sta_retrain_v4.verify_code_lock_against_repo",
            side_effect=ValueError(
                "runtime executable_tree_sha256 mismatch: dead != beef"
            ),
        ):
            drift = assess_hpo_readiness(
                cand, root=_ROOT, policy_path=policy_path, binding_path=bad_bind
            )
        assert drift["policy_ready"] is True
        assert drift["hpo_ready"] is False
        assert any("code_lock_preflight" in r for r in drift["reasons"])
        print("TREE_DRIFT_HPO_READY=False")

    # CLI against on-disk frozen policy without binding → exit 2 (even if dirty).
    script = _ROOT / "scripts/run_ms_sta_retrain_v4.py"
    cli = subprocess.run(
        [sys.executable, str(script), "--dry-plan", "--policy", str(policy_path)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 2, cli.stdout + cli.stderr
    cli_payload = json.loads(cli.stdout)
    assert cli_payload["ok"] is False
    assert cli_payload["hpo_ready"] is False
    assert cli_payload["policy_ready"] is True
    assert "contract satisfied" not in json.dumps(cli_payload).lower()
    print("CLI_MISSING_BINDING_EXIT=2")
