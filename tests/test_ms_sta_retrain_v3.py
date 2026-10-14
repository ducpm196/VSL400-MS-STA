"""Hermetic contracts for MS-STA retrain-v3 Path B (P1 reopen #2 blockers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import (
    CONFIRM_VARIANTS,
    FULL_SEEDS,
    HPO_VARIANTS,
    PATH_B_SIGNERS,
    active_variant_from_efficiency_gate,
    build_demo_set_ms_sta_v3,
    confirm_plan,
    evaluate_architecture_gates,
    evaluate_efficiency_gate,
    expand_search_space,
    hpo_full_plan,
    hpo_screen_plan,
    promote_candidate_checkpoint,
    promote_top_k_from_screen,
    refuse_gate3_init,
    require_complete_phase_rows,
    require_confirm_final_lifecycle,
    require_frozen_v3_policy,
    require_homogeneous_git_revision,
    require_hpo_candidate_lifecycle,
    select_config_from_full_rows,
    sha256_file,
    skeleton_attention_score_elements,
    verify_d4_acceptance_artifacts,
    verify_test_matrix_artifacts,
)
from src.training.retrain_v3_stats import (
    REQUIRED_TEST_VARIANTS,
    holm_correction,
    macro_f1,
    mcnemar,
    noninferiority_from_paired_ci,
    paired_signer_bootstrap_delta_ci,
    require_exact_variants,
    require_sha_present_and_equal,
)

POLICY_PATH = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3_policy" / "RETRAIN_POLICY.json"
SPLIT_PATH = _ROOT / "data" / "processed" / "vsl400_retrain_v3_split_manifest.json"
ELIGIBLE_PATH = _ROOT / "data" / "processed" / "vsl400_eligible_sample_manifest.json"


@pytest.fixture(scope="module")
def policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def split_man() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def test_policy_frozen_path_b_reuse_contract(policy: dict) -> None:
    require_frozen_v3_policy(policy)
    assert policy["path"] == "B"
    assert "retrospective" in policy["test_claim_label"]
    assert policy["reuse_contract"]["reuse_metrics"] is False
    assert policy["reuse_contract"]["reuse_checkpoints"] is False
    assert policy["reuse_contract"]["reuse_vsl_weights"] is False
    assert policy["reuse_contract"]["reuse_label_free_feature_caches"] is True
    assert policy["initialization"]["forbid_gate3_warm_start"] is True
    assert policy["initialization"]["optimizer_lr_warmup_start_factor"] == 0.01
    assert policy["hpo"]["tie_break_excludes_latency"] is True
    assert "median_forward_latency" not in " ".join(policy["hpo"]["tie_break"])
    assert policy["final_test"]["bootstrap_replicates"] == 10000
    assert "ci_low" in policy["final_test"]["noninferiority_decision_rule"]
    for v in ("b0", "b1", "b2", "b3", "b4", "b4e", "b5"):
        assert policy["variant_controls"][v]["init"] == "random"


def test_policy_and_split_hash_match(policy: dict) -> None:
    actual = sha256_file(SPLIT_PATH)
    assert actual == policy["lineage"]["split_manifest_sha256"]
    assert actual == policy["data_locks"]["split_manifest_sha256"]
    elig = sha256_file(ELIGIBLE_PATH)
    assert elig == policy["lineage"]["eligible_sample_manifest_sha256"]
    assert elig == policy["data_locks"]["eligible_sample_manifest_sha256"]


def test_split_roles_path_b(split_man: dict) -> None:
    assert split_man["path"] == "B"
    assert split_man["signers"]["hpo_train"] == PATH_B_SIGNERS["hpo_train"]
    assert split_man["signers"]["tune"] == PATH_B_SIGNERS["tune"]
    assert split_man["signers"]["val"] == PATH_B_SIGNERS["val"]
    assert split_man["signers"]["test"] == PATH_B_SIGNERS["test"]
    assert set(split_man["missing_signers"]) == {"signer_5", "signer_17"}
    assert split_man["sealed_holdout_signers"] == []
    assert split_man["n_confirm_train"] == 15992
    assert split_man["n_val"] == 2595
    assert split_man["n_test"] == 6165
    assert split_man["n_hpo_train"] + split_man["n_tune"] == split_man["n_confirm_train"]
    assert not set(split_man["hpo_train_ids"]) & set(split_man["tune_ids"])
    assert not set(split_man["confirm_train_ids"]) & set(split_man["test_ids"])
    assert not set(split_man["val_ids"]) & set(split_man["test_ids"])
    assert not set(split_man["tune_ids"]) & set(split_man["val_ids"])


def test_signer_overlap_and_sealed_holdout_rejection(split_man: dict) -> None:
    all_dev = set(split_man["confirm_train_ids"]) | set(split_man["val_ids"])
    assert not all_dev & set(split_man["test_ids"])
    roles = ("hpo_train", "tune", "confirm_train", "val", "test")
    for s in ("signer_5", "signer_17"):
        assert s not in set(sum((split_man["signers"][k] for k in roles), []))


def test_split_ids_subset_of_eligible(split_man: dict) -> None:
    elig = json.loads(ELIGIBLE_PATH.read_text(encoding="utf-8"))
    eligible = set(map(str, elig["eligible_ids"]))
    for role in ("hpo_train_ids", "tune_ids", "confirm_train_ids", "val_ids", "test_ids"):
        ids = set(map(str, split_man[role]))
        assert ids <= eligible, role
    assert len(split_man["test_ids"]) == int(elig["eligible_by_split"]["test"])


def test_gate3_warm_start_rejected() -> None:
    with pytest.raises(ValueError, match="refuses Gate3"):
        refuse_gate3_init("gate3_rgb_proj_and_classifier")
    with pytest.raises(ValueError, match="refuses Gate3"):
        refuse_gate3_init("gate3_rgb_proj_only")
    refuse_gate3_init("random")


def test_equal_hpo_and_confirm_counts(policy: dict) -> None:
    screen = hpo_screen_plan(policy)
    assert len(screen) == 24
    assert len(expand_search_space(policy)) == 8
    for v in HPO_VARIANTS:
        assert sum(1 for t in screen if t["variant"] == v) == 8
    promoted = {v: [c["config_id"] for c in expand_search_space(policy)[:2]] for v in HPO_VARIANTS}
    assert len(hpo_full_plan(policy, top_configs_by_variant=promoted)) == 12
    selected = {v: expand_search_space(policy)[0]["config_id"] for v in HPO_VARIANTS}
    plan = confirm_plan(policy, selected_config_by_variant=selected)
    assert len(plan) == 21
    assert set(CONFIRM_VARIANTS) == {t["variant"] for t in plan}


def test_policy_inheritance_b1_b2_b5_b4e(policy: dict) -> None:
    selected = {"b0": "lr1e-4_wd0.01", "b3": "lr1e-4_wd0.01", "b4": "lr3e-4_wd0.05"}
    plan = confirm_plan(policy, selected_config_by_variant=selected)
    assert len(plan) == 21
    for t in plan:
        if t["variant"] in ("b1", "b2", "b4e", "b5"):
            assert t["config_id"] == selected["b4"]
        if t["variant"] == "b4e":
            assert t["skel_temporal_stride"] == 4
            assert t["model_variant"] == "b4"
        else:
            assert t["skel_temporal_stride"] == 1
        assert t["train_role"] == "confirm_train"
        assert t["val_role"] == "val"


def test_full_selector_requires_both_seeds_and_excludes_latency() -> None:
    with pytest.raises(ValueError, match="incomplete full seeds"):
        select_config_from_full_rows(
            [{"config_id": "lr1e-4_wd0.01", "seed": 11, "best_val_top1": 10.0, "best_val_top5": 20.0}]
        )
    with pytest.raises(ValueError, match="duplicate"):
        select_config_from_full_rows(
            [
                {"config_id": "a", "seed": 11, "best_val_top1": 10.0, "best_val_top5": 20.0},
                {"config_id": "a", "seed": 11, "best_val_top1": 11.0, "best_val_top5": 21.0},
                {"config_id": "a", "seed": 29, "best_val_top1": 10.0, "best_val_top5": 20.0},
            ]
        )
    # Within 0.25 pp Top-1: prefer higher Top-5, then lex id. Latency fields must be ignored.
    rows = [
        {"config_id": "lr3e-4_wd0.01", "seed": 11, "best_val_top1": 12.00, "best_val_top5": 20.0, "median_forward_latency_ms": 10.0},
        {"config_id": "lr3e-4_wd0.01", "seed": 29, "best_val_top1": 12.00, "best_val_top5": 20.0, "median_forward_latency_ms": 10.0},
        {"config_id": "lr1e-4_wd0.05", "seed": 11, "best_val_top1": 12.10, "best_val_top5": 19.0, "median_forward_latency_ms": 5.0},
        {"config_id": "lr1e-4_wd0.05", "seed": 29, "best_val_top1": 12.10, "best_val_top5": 19.0, "median_forward_latency_ms": 5.0},
        {"config_id": "lr1e-3_wd0.01", "seed": 11, "best_val_top1": 12.05, "best_val_top5": 22.0, "median_forward_latency_ms": 8.0},
        {"config_id": "lr1e-3_wd0.01", "seed": 29, "best_val_top1": 12.05, "best_val_top5": 22.0, "median_forward_latency_ms": 8.0},
    ]
    assert select_config_from_full_rows(rows, required_seeds=FULL_SEEDS, tie_epsilon_pp=0.25) == "lr1e-3_wd0.01"
    # Same Top-1/Top-5 window: lex smaller wins even if latency is worse.
    tied = [
        {"config_id": "lr1e-4_wd0.05", "seed": 11, "best_val_top1": 12.0, "best_val_top5": 22.0, "median_forward_latency_ms": 1.0},
        {"config_id": "lr1e-4_wd0.05", "seed": 29, "best_val_top1": 12.0, "best_val_top5": 22.0, "median_forward_latency_ms": 1.0},
        {"config_id": "lr1e-3_wd0.01", "seed": 11, "best_val_top1": 12.0, "best_val_top5": 22.0, "median_forward_latency_ms": 100.0},
        {"config_id": "lr1e-3_wd0.01", "seed": 29, "best_val_top1": 12.0, "best_val_top5": 22.0, "median_forward_latency_ms": 100.0},
    ]
    assert select_config_from_full_rows(tied, required_seeds=FULL_SEEDS, tie_epsilon_pp=0.25) == "lr1e-3_wd0.01"


def test_screen_promote_requires_all_eight_configs() -> None:
    rows = [
        {"config_id": f"c{i}", "best_val_top1": float(i), "best_val_top5": float(i)}
        for i in range(7)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        promote_top_k_from_screen(rows, k=2, expected_configs=8)
    ok = [
        {"config_id": f"c{i}", "best_val_top1": float(i), "best_val_top5": float(i)}
        for i in range(8)
    ]
    assert promote_top_k_from_screen(ok, k=2, expected_configs=8) == ["c7", "c6"]


def test_efficiency_gate_from_measurement_not_operator_bool() -> None:
    assert skeleton_attention_score_elements(64, 1) == 4096
    assert skeleton_attention_score_elements(64, 4) == 256
    gate = evaluate_efficiency_gate(
        {
            "mean_delta_top1_pp": -0.5,
            "latency_reduction_pct": 25.0,
            "param_count_delta": 0,
            "score_elems_before": 4096,
            "score_elems_after": 256,
        }
    )
    assert gate["passed"] is True
    assert gate["active_variant"] == "b4e"
    gate2 = evaluate_efficiency_gate(
        {
            "mean_delta_top1_pp": -1.5,
            "latency_reduction_pct": 25.0,
            "param_count_delta": 0,
        }
    )
    assert gate2["passed"] is False
    assert active_variant_from_efficiency_gate(False) == "b4"


def test_promote_candidate_writes_final_lifecycle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    cand = tmp_path / "cand.pth"
    final = tmp_path / "final.pth"
    torch.save(
        {
            "checkpoint_status": "candidate",
            "selection": "val_top1",
            "model": {"w": torch.tensor([1.0])},
            "retrain_v3": True,
        },
        cand,
    )
    promo = promote_candidate_checkpoint(cand, final)
    assert promo["checkpoint_status"] == "final_selected_checkpoint"
    assert Path(promo["final_path"]).is_file()
    loaded = torch.load(final, map_location="cpu", weights_only=False)
    assert loaded["checkpoint_status"] == "final_selected_checkpoint"
    assert loaded["lifecycle"]["source_candidate_sha256"] == promo["source_candidate_sha256"]
    assert promo["final_sha256"] == sha256_file(final)


def test_trainer_promotes_final_only_on_confirm_phase() -> None:
    text = (_ROOT / "src/training/train_ms_sta_vsl400.py").read_text(encoding="utf-8")
    assert 'str(args.phase) == "confirm"' in text
    assert "final_stem_from_candidate_stem" in text
    assert 'hpo_disposable' in text


def test_hpo_vs_confirm_lifecycle_guards() -> None:
    hpo_rows = [
        {
            "run_key": "hpo",
            "exit_code": 0,
            "best_val_top1": 1.0,
            "checkpoint_status": "candidate",
            "final_checkpoint": None,
        }
    ]
    require_hpo_candidate_lifecycle(hpo_rows, phase="hpo_full")
    with pytest.raises(ValueError, match="candidate lifecycle"):
        require_hpo_candidate_lifecycle(
            [{**hpo_rows[0], "checkpoint_status": "final_selected_checkpoint"}],
            phase="hpo_screen",
        )
    confirm_rows = [
        {
            "run_key": "c",
            "exit_code": 0,
            "best_val_top1": 1.0,
            "checkpoint_status": "final_selected_checkpoint",
            "final_checkpoint": "x.pth",
            "final_checkpoint_sha256": "abc",
        }
    ]
    require_confirm_final_lifecycle(confirm_rows)
    with pytest.raises(ValueError, match="missing final_selected_checkpoint"):
        require_confirm_final_lifecycle([{**confirm_rows[0], "checkpoint_status": "candidate"}])


def test_required_scripts_exist() -> None:
    for rel in (
        "scripts/build_ms_sta_retrain_v3_splits.py",
        "scripts/run_ms_sta_retrain_v3.py",
        "scripts/monitor_ms_sta_retrain_v3.py",
        "scripts/watch_ms_sta_retrain_v3.ps1",
        "scripts/canonicalize_ms_sta_retrain_v3.py",
        "scripts/build_confirm_validation_report_v3.py",
        "scripts/evaluate_ms_sta_retrain_v3_validation.py",
        "scripts/diagnose_ms_sta_b3_b4_v3.py",
        "scripts/evaluate_ms_sta_retrain_v3_validation.py",
        "scripts/benchmark_ms_sta_retrain_v3_efficiency.py",
        "scripts/evaluate_ms_sta_retrain_v3.py",
        "scripts/run_d4_retrain_v3_acceptance.py",
        "src/training/retrain_v3_stats.py",
    ):
        assert (_ROOT / rel).is_file()


def test_efficiency_benchmark_forbids_cli_checkpoint_override() -> None:
    text = (_ROOT / "scripts/benchmark_ms_sta_retrain_v3_efficiency.py").read_text(encoding="utf-8")
    assert "--b4-ckpt" not in text
    assert "--b4e-ckpt" not in text
    assert "cli_checkpoint_override_forbidden" in text
    assert "resolved_from" in text
    assert "raw_latency_ms" in text
    assert "environment" in text


def test_watchdog_persists_restart_and_refuses_live_tree() -> None:
    text = (_ROOT / "scripts/watch_ms_sta_retrain_v3.ps1").read_text(encoding="utf-8")
    assert "watchdog_restarts.json" in text
    assert "refuse resume; live process tree" in text
    assert "Wait-ProcessTreeDead" in text
    assert "Get-DescendantPids" in text


def test_no_test_access_from_hpo_confirmation(policy: dict) -> None:
    for t in hpo_screen_plan(policy):
        assert t["val_role"] == "tune"
        assert t["train_role"] == "hpo_train"
    selected = {"b0": "lr1e-3_wd0.05", "b3": "lr1e-3_wd0.05", "b4": "lr1e-3_wd0.05"}
    for t in confirm_plan(policy, selected_config_by_variant=selected):
        assert t["val_role"] == "val"


def test_require_complete_phase_rows() -> None:
    rows = [{"run_key": f"k{i}", "exit_code": 0, "best_val_top1": 1.0} for i in range(3)]
    require_complete_phase_rows(rows, expected_n=3, phase="x")
    with pytest.raises(ValueError, match="incomplete"):
        require_complete_phase_rows(rows[:2], expected_n=3, phase="x")


def test_require_homogeneous_git_revision() -> None:
    rows = [
        {"run_key": "a", "exit_code": 0, "git_revision": "abc"},
        {"run_key": "b", "exit_code": 0, "git_revision": "abc"},
    ]
    require_homogeneous_git_revision(rows, expected_revision="abc", phase="hpo_screen")
    with pytest.raises(ValueError, match="drift"):
        require_homogeneous_git_revision(
            [{**rows[0], "git_revision": "zzz"}, rows[1]],
            expected_revision="abc",
            phase="hpo_screen",
        )
    with pytest.raises(ValueError, match="missing git_revision"):
        require_homogeneous_git_revision(
            [{"run_key": "a", "exit_code": 0}],
            expected_revision="abc",
            phase="hpo_screen",
        )


def test_evaluate_architecture_gates_b4_vs_b3() -> None:
    seeds = (42, 123, 456)
    rows = []
    for s in seeds:
        rows.append({"variant": "b0", "seed": s, "exit_code": 0, "best_val_top1": 40.0, "best_val_top5": 70.0})
        rows.append({"variant": "b3", "seed": s, "exit_code": 0, "best_val_top1": 62.0, "best_val_top5": 88.0})
        rows.append({"variant": "b4", "seed": s, "exit_code": 0, "best_val_top1": 60.0, "best_val_top5": 87.5})
        rows.append({"variant": "b4e", "seed": s, "exit_code": 0, "best_val_top1": 58.5, "best_val_top5": 86.0})
    policy = {
        "gates": {
            "architecture": {
                "b4_top1_gt_b0_all_seeds": True,
                "b4_top1_gt_b3_all_seeds": True,
                "mean_delta_top1_positive_vs_b0_and_b3": True,
                "mean_top5_regression_limit_pp": -1.0,
            }
        }
    }
    gate = evaluate_architecture_gates(rows, policy=policy, seeds=seeds)
    assert gate["clauses"]["b4_top1_gt_b0_all_seeds"] is True
    assert gate["clauses"]["b4_top1_gt_b3_all_seeds"] is False
    assert gate["passed"] is False
    assert gate["mean_b4_minus_b3_pp"] == pytest.approx(-2.0)
    assert gate["mean_b4e_minus_b4_pp"] == pytest.approx(-1.5)

    # Flip B4 above B3 -> pass
    for r in rows:
        if r["variant"] == "b4":
            r["best_val_top1"] = 63.0
            r["best_val_top5"] = 88.0
    gate2 = evaluate_architecture_gates(rows, policy=policy, seeds=seeds)
    assert gate2["passed"] is True


def test_require_exact_variants_negative() -> None:
    require_exact_variants(set(REQUIRED_TEST_VARIANTS))
    with pytest.raises(ValueError, match="missing"):
        require_exact_variants({"b0", "b3", "b4"})
    with pytest.raises(ValueError, match="extra"):
        require_exact_variants({"b0", "b3", "b4", "b4e", "b5"})


def test_stats_helpers_mcnemar_holm_noninferiority() -> None:
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    p = np.array([0, 1, 1, 0, 0, 2], dtype=np.int64)
    assert macro_f1(y, p, num_classes=3) > 0
    base = np.array([True, True, False, False])
    other = np.array([True, False, True, False])
    m = mcnemar(base, other)
    assert m["baseline_only_correct"] == 1
    assert m["other_only_correct"] == 1
    assert m["n_common_pairs"] == 4
    assert 0.0 <= m["p_value_exact"] <= 1.0
    big = mcnemar(np.zeros(2000, dtype=bool), np.ones(2000, dtype=bool))
    assert big["n_discordant"] == 2000
    assert big["p_value_exact"] < 1e-10
    holm = holm_correction([("a", 0.01), ("b", 0.04)])
    assert holm["tests"]["a"]["reject_null_at_0.05"] is True

    # Point estimate alone is not enough: CI lower bound decides NI.
    signers = [f"s{i}" for i in range(20)]
    ref = np.ones(20, dtype=bool)
    cand_ok = np.ones(20, dtype=bool)
    cand_ok[0] = False  # tiny deficit
    delta = paired_signer_bootstrap_delta_ci(
        ref, cand_ok, signers, n_boot=200, seed=42, label_a="b4", label_b="b4e"
    )
    ni = noninferiority_from_paired_ci(delta, margin_pp=-1.0)
    assert "ci_low" in ni and ni["decision_rule"] == "ci_low >= margin_pp"
    # Clear inferiority: all false vs all true.
    delta_bad = paired_signer_bootstrap_delta_ci(
        np.ones(20, dtype=bool),
        np.zeros(20, dtype=bool),
        signers,
        n_boot=200,
        seed=42,
        label_a="b4",
        label_b="b4e",
    )
    ni_bad = noninferiority_from_paired_ci(delta_bad, margin_pp=-1.0)
    assert ni_bad["noninferior"] is False
    assert ni_bad["ci_low"] < -1.0

    with pytest.raises(ValueError, match="SHA missing"):
        require_sha_present_and_equal(None, "abc", label="policy")
    with pytest.raises(ValueError, match="mismatch"):
        require_sha_present_and_equal("aaa", "bbb", label="policy")
    require_sha_present_and_equal("abc", "abc", label="policy")


def _full_matrix_fixture(n: int = 2) -> tuple[dict, list[dict]]:
    variants = {}
    for v in REQUIRED_TEST_VARIANTS:
        variants[v] = {
            "top1": 10.0,
            "top5": 20.0,
            "macro_f1": 5.0,
            "per_signer": {"signer_1": {"n": 1, "top1": 10.0}},
            "worst_signer": {"signer": "signer_1", "n": 1, "top1": 10.0},
            "bootstrap_ci_top1": {"top1_point": 10.0, "ci_low": 1.0, "ci_high": 20.0},
        }
    paired = {
        "top1_b4_minus_b0": {"delta_pp": 1.0, "ci_low": 0.1, "ci_high": 2.0},
        "top1_b4_minus_b3": {"delta_pp": 0.5, "ci_low": 0.0, "ci_high": 1.0},
        "top1_b4e_minus_b4": {"delta_pp": -0.2, "ci_low": -0.5, "ci_high": 0.1},
        "top5_b4_minus_b0": {"delta_pp": 1.0, "ci_low": 0.1, "ci_high": 2.0},
        "top5_b4e_minus_b4": {"delta_pp": 0.0, "ci_low": -0.2, "ci_high": 0.2},
    }
    matrix = {
        "n_test": n,
        "common_pair_n": n,
        "required_variants": list(REQUIRED_TEST_VARIANTS),
        "variants_present": list(REQUIRED_TEST_VARIANTS),
        "mcnemar": {},
        "holm": {},
        "b4e_vs_b4_noninferiority": {
            "delta_pp": -0.2,
            "ci_low": -0.5,
            "decision_rule": "ci_low >= margin_pp",
            "noninferior": True,
        },
        "hypothesis_decisions": {},
        "paired_deltas": paired,
        "variants": variants,
    }
    rows = []
    for i in range(n):
        rows.append(
            {
                "video_id": str(i + 1),
                "signer_id": "signer_1",
                "y": i % 2,
                "pred_b0": 0,
                "pred_b3": i % 2,
                "pred_b4": i % 2,
                "pred_b4e": i % 2,
                "top5_b0": [0, 1, 2, 3, 4],
                "top5_b3": [0, 1, 2, 3, 4],
                "top5_b4": [0, 1, 2, 3, 4],
                "top5_b4e": [0, 1, 2, 3, 4],
                "top5_hit_b0": True,
                "top5_hit_b3": True,
                "top5_hit_b4": True,
                "top5_hit_b4e": True,
            }
        )
    return matrix, rows


def test_verify_test_matrix_artifacts_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        verify_test_matrix_artifacts(tmp_path)
    matrix = {
        "n_test": 2,
        "common_pair_n": 2,
        "required_variants": list(REQUIRED_TEST_VARIANTS),
        "variants_present": ["b0", "b3", "b4"],
        "mcnemar": {},
        "holm": {},
        "b4e_vs_b4_noninferiority": {},
        "hypothesis_decisions": {},
        "variants": {},
    }
    (tmp_path / "TEST_MATRIX.json").write_text(json.dumps(matrix), encoding="utf-8")
    (tmp_path / "TEST_PER_SAMPLE.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        verify_test_matrix_artifacts(tmp_path, expected_n=2)


def test_verify_test_matrix_artifacts_positive(tmp_path: Path) -> None:
    matrix, rows = _full_matrix_fixture(2)
    (tmp_path / "TEST_MATRIX.json").write_text(json.dumps(matrix), encoding="utf-8")
    (tmp_path / "TEST_PER_SAMPLE.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out = verify_test_matrix_artifacts(tmp_path, expected_n=2)
    assert out["n_test"] == 2


def test_build_demo_set_from_v3_rows() -> None:
    rows = [
        {
            "video_id": "100",
            "signer_id": "signer_24",
            "y": 1,
            "pred_b0": 0,
            "pred_b4": 1,
            "pred_b4e": 1,
            "top5_b4": [1, 2, 3, 4, 5],
            "top5_hit_b0": False,
            "top5_hit_b4": True,
            "top5_hit_b4e": True,
        },
        {
            "video_id": "200",
            "signer_id": "signer_25",
            "y": 7,
            "pred_b0": 8,
            "pred_b4": 8,
            "pred_b4e": 8,
            "top5_b4": [8, 7, 1, 2, 3],
            "top5_hit_b0": True,
            "top5_hit_b4": True,
            "top5_hit_b4e": True,
        },
    ]
    demo = build_demo_set_ms_sta_v3(
        rows,
        active_variant="b4",
        id_to_gloss={0: "A", 1: "B", 7: "G", 8: "H"},
        test_matrix_sha256="t" * 64,
        policy_sha256="p" * 64,
        split_manifest_sha256="s" * 64,
    )
    assert demo["retrain_v3"] is True
    assert demo["reuse_e2_demo"] is False
    assert demo["scenarios"]["top1_correct"] == "100"
    assert demo["scenarios"]["top5_assist"] == "200"
    assert demo["scenario_rows"]["top1_correct"]["gold_class_id"] == 1
    assert demo["scenario_rows"]["top1_correct"]["pred_active_class_id"] == 1
    assert demo["scenario_rows"]["top5_assist"]["gold_class_id"] == 7


def test_verify_d4_acceptance_negative_and_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        verify_d4_acceptance_artifacts(tmp_path, root=tmp_path)
    bad = {
        "pass": True,
        "registry_updated": False,
        "historical_registry_mutated": False,
        "defense_checkpoint": None,
        "scenarios": [],
    }
    (tmp_path / "D4_ACCEPTANCE_REPORT.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="registry"):
        verify_d4_acceptance_artifacts(tmp_path, root=tmp_path)

    demo_dir = tmp_path / "data" / "processed"
    demo_dir.mkdir(parents=True)
    (demo_dir / "demo_set_ms_sta_v3.json").write_text(
        json.dumps({"retrain_v3": True, "reuse_e2_demo": False}),
        encoding="utf-8",
    )
    (demo_dir / "ms_sta_registry_active.json").write_text("{}", encoding="utf-8")
    good = {
        "pass": True,
        "registry_updated": True,
        "historical_registry_mutated": False,
        "defense_checkpoint": "x.pth",
        "defense_sha256": "abc",
        "source_sha256": "abc",
        "exact_byte_copy": True,
        "registry_cutover_path": "data/processed/ms_sta_registry_active.json",
        "demo_set_v3": "data/processed/demo_set_ms_sta_v3.json",
        "active_variant": "b4",
        "active_experiment_variant": "b4",
        "scenarios": [
            {"name": "health_candidate_vs_ready", "pass": True},
            {"name": "cache_backed_predict_clip_ms_sta", "pass": True},
            {"name": "payload_engine_identity_retrain_v3_final", "pass": True},
            {"name": "demo_set_v3_lineage", "pass": True},
            {"name": "active_stride_identity", "pass": True},
            {"name": "predict_identity_top1_correct", "pass": True},
            {"name": "predict_identity_top5_assist", "pass": True},
            {"name": "defense_exact_sha_equals_source", "pass": True},
            {"name": "health_defense_post_promotion", "pass": True},
            {"name": "defense_predict_identity_top1_correct", "pass": True},
        ],
    }
    (tmp_path / "D4_ACCEPTANCE_REPORT.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "REGISTRY_V3_PROMOTED.json").write_text("{}", encoding="utf-8")
    out = verify_d4_acceptance_artifacts(tmp_path, root=tmp_path)
    assert out["n_scenarios"] == 10


def test_backend_v3_defense_loads_exact_final_without_payload_defense_flag(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    from src.models.ms_sta_wordlevel import build_ms_sta
    import src.deployment.backend_api as api

    model = build_ms_sta("b4", num_classes=20, skel_temporal_stride=4)
    final = tmp_path / "final.pth"
    policy_sha = "c" * 64
    torch.save(
        {
            "retrain_v3": True,
            "experiment_variant": "b4e",
            "defense_ready": False,  # exact final-test payload; registry owns defense lifecycle
            "e1_selected": False,
            "checkpoint_status": "final_selected_checkpoint",
            "variant": "b4",
            "seed": 42,
            "skel_temporal_stride": 4,
            "model": model.state_dict(),
            "num_classes": 20,
            "retrain_v3_policy_sha256": policy_sha,
        },
        final,
    )
    digest = sha256_file(final)
    reg = {
        "retrain_v3": True,
        "defense_ready": True,
        "d4_pass": True,
        "evaluation_locked": True,
        "active_variant": "b4",
        "experiment_variant": "b4e",
        "checkpoint": final.name,
        "checkpoint_sha256": digest,
        "policy_sha256": policy_sha,
        "ckpts_dir": str(tmp_path),
    }
    monkeypatch.setattr(api, "_resolve_ms_sta_ckpt_dir", lambda r: tmp_path)
    loaded, variant, mode = api._load_ms_sta_from_registry(reg)
    assert loaded is not None
    assert mode == "defense"
    assert variant == "b4"
    assert int(loaded.skel_temporal_stride) == 4
    assert api._ms_sta_experiment_variant == "b4e"
    assert api._ms_sta_skel_temporal_stride == 4


def test_d4_stage_exact_copy_refuses_mutated_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from scripts.run_d4_retrain_v3_acceptance import _stage_exact_defense

    src = tmp_path / "src.pth"
    torch.save({"retrain_v3": True, "checkpoint_status": "final_selected_checkpoint", "x": 1}, src)
    src_sha = sha256_file(src)
    # Mutating then saving must not be used; staging compares against source sha.
    ckpt_dir = tmp_path / "ckpts"
    staged = _stage_exact_defense(
        active="b4",
        src=src,
        src_sha=src_sha,
        ckpt_dir=ckpt_dir,
        policy_sha="p" * 64,
        lock={"test_claim_label": "retrospective_signer_disjoint_confirmation"},
    )
    assert staged["defense_sha256"] == src_sha
    assert staged["exact_byte_copy"] is True
    with pytest.raises(RuntimeError, match="sha mismatch"):
        _stage_exact_defense(
            active="b4",
            src=src,
            src_sha="0" * 64,
            ckpt_dir=ckpt_dir / "bad",
            policy_sha="p" * 64,
            lock={},
        )


def test_d4_promote_only_after_all_pass_contract() -> None:
    text = (_ROOT / "scripts/run_d4_retrain_v3_acceptance.py").read_text(encoding="utf-8")
    assert "_stage_exact_defense" in text
    assert "_run_defense_post_promotion_smoke" in text
    assert "_cutover_active_pointer" in text
    assert text.index("if candidate_pass:") < text.index("staged = _stage_exact_defense(")
    assert text.index("post = _run_defense_post_promotion_smoke(") < text.index(
        "_cutover_active_pointer(staged=staged"
    )
    assert "shutil.copy2(src, tmp_ckpt)" in text
    assert 'payload["defense_ready"] = True' not in text
    assert "predict_identity_top1_correct" in text
    assert "ms_sta_experiment_variant" in text
    assert "demo_set_ms_sta_v3" in text
    assert "health_defense_post_promotion" in text


def test_backend_v3_b4e_candidate_from_promoted_payload(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from src.models.ms_sta_wordlevel import build_ms_sta
    import src.deployment.backend_api as api

    model = build_ms_sta("b4", num_classes=20, skel_temporal_stride=4)
    cand = tmp_path / "cand.pth"
    final = tmp_path / "final.pth"
    policy_sha = "a" * 64
    torch.save(
        {
            "retrain_v3": True,
            "experiment_variant": "b4e",
            "defense_ready": False,
            "e1_selected": False,
            "checkpoint_status": "candidate",
            "variant": "b4",
            "seed": 42,
            "skel_temporal_stride": 4,
            "model": model.state_dict(),
            "num_classes": 20,
            "retrain_v3_policy_sha256": policy_sha,
        },
        cand,
    )
    promote_candidate_checkpoint(cand, final)
    digest = sha256_file(final)
    reg = {
        "retrain_v3": True,
        "defense_ready": False,
        "evaluation_locked": True,
        "d4_candidate_allowed": True,
        "active_variant": "b4",
        "experiment_variant": "b4e",
        "checkpoint": final.name,
        "checkpoint_sha256": digest,
        "policy_sha256": policy_sha,
        "ckpts_dir": str(tmp_path),
    }
    monkeypatch.setattr(api, "_resolve_ms_sta_ckpt_dir", lambda r: tmp_path)
    loaded, variant, mode = api._load_ms_sta_from_registry(reg)
    assert loaded is not None and mode == "candidate" and int(loaded.skel_temporal_stride) == 4


def test_backend_resolves_v3_cutover_pointer() -> None:
    text = (_ROOT / "src/deployment/backend_api.py").read_text(encoding="utf-8")
    assert "_resolve_ms_sta_registry_path" in text
    assert "MS_STA_REGISTRY_PATH" in text
    assert "ms_sta_registry_active.json" in text
    assert "skel_temporal_stride" in text
    assert "retrain_v3" in text
    assert "historical registry present but fallback disabled" in text
    assert "_load_v3_closeout_kpi" in text
    assert "MS_STA_ALLOW_HISTORICAL_KPI" in text


def test_active_registry_is_v6_defense_go() -> None:
    active = json.loads(
        (_ROOT / "data/processed/ms_sta_registry_active.json").read_text(encoding="utf-8")
    )
    assert active.get("campaign_status") == "DEFENSE_GO"
    assert active.get("defense_ready") is True
    assert active.get("demo_allowed") is True
    assert active.get("historical_fallback_disabled") is True
    assert active.get("rollback_registry_path") == "data/processed/ms_sta_registry_v5_defense_go.json"
    reg_path = _ROOT / str(active["registry_path"])
    assert sha256_file(reg_path) == active["registry_sha256"]
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert reg.get("schema") == "ms_sta_registry_v6_defense_go_v1"
    assert reg.get("status") == "GO"
    assert reg.get("campaign_status") == "DEFENSE_GO"
    assert reg.get("defense_ready") is True
    assert reg.get("demo_allowed") is True
    assert reg.get("active_variant") == "b4_v6_ctr"
    assert reg.get("e4_ready") is False
    assert reg.get("b4_ready") is True
    assert (
        reg.get("checkpoint_sha256")
        == "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
    )
    v5 = _ROOT / "data/processed/ms_sta_registry_v5_defense_go.json"
    assert v5.is_file()
    assert Path(active["registry_path"]).name != "ms_sta_registry.json"

def test_orchestrator_test_completed_requires_artifact_verify() -> None:
    text = (_ROOT / "scripts/run_ms_sta_retrain_v3.py").read_text(encoding="utf-8")
    assert "verify_test_matrix_artifacts" in text
    assert "verify_d4_acceptance_artifacts" in text
    assert "artifacts_verified" in text
    assert "require_hpo_candidate_lifecycle" in text
    assert "require_confirm_final_lifecycle" in text


def test_orchestrator_dry_plan_counts(policy: dict) -> None:
    import subprocess

    py = _ROOT / "_venvs" / "ci" / "Scripts" / "python.exe"
    if not py.is_file():
        py = Path(sys.executable)
    proc = subprocess.run(
        [str(py), str(_ROOT / "scripts/run_ms_sta_retrain_v3.py"), "--campaign-dry-plan"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["counts"]["hpo_screen"] == 24
    assert payload["training_runs_total"] == 57


def test_output_collision_helper_still_imported() -> None:
    from src.training.e1_policy_enforce import refuse_output_collision

    p = _ROOT / "tmp_retrain_v3_collision_test.pth"
    s = _ROOT / "tmp_retrain_v3_collision_test.json"
    try:
        if p.exists():
            p.unlink()
        if s.exists():
            s.unlink()
        refuse_output_collision(checkpoint_path=p, status_path=s, allow_overwrite=False, allow_pilot=False)
        p.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="collision|exists|refuse"):
            refuse_output_collision(checkpoint_path=p, status_path=s, allow_overwrite=False, allow_pilot=False)
    finally:
        if p.exists():
            p.unlink()
        if s.exists():
            s.unlink()
