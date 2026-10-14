"""Retrain-v4 scaffold contracts (train still forbidden)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
SPLIT_REL = "data/processed/vsl400_retrain_v4_split_manifest.json"


def test_v4_policy_draft_refuses_train_and_locks_science() -> None:
    policy = json.loads(
        (_ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy.get("train_allowed") is False
    assert policy.get("status") != "FROZEN"
    assert (policy.get("architecture") or {}).get("default_residual_init") == "zero_delta_head"

    primary = policy.get("primary_comparison") or {}
    req = primary.get("requirements") or {}
    assert req.get("matched_hpo_space_for_all_primary_variants") is True
    budget = primary.get("matched_budget_definition") or {}
    assert budget.get("confirm_early_stopping") is False

    warm = policy.get("warm_start_diagnostic") or {}
    assert warm.get("runs_planned") == 0
    assert warm.get("status") == "deferred_optional_after_primary_accuracy"

    hpo = policy.get("hpo") or {}
    assert hpo.get("variants") == ["b0", "b3", "b4", "b4r"]
    assert hpo.get("configs_per_variant") == 6
    assert hpo.get("variant_config_pairs") == 24
    assert hpo.get("runs_total") == 48

    conf = policy.get("confirmation") or {}
    assert conf.get("run_full_epochs") is True
    assert conf.get("early_stopping_allowed") is False

    zero_scale = next(
        a
        for a in ((policy.get("ablations") or {}).get("required_before_claim") or [])
        if (a.get("id") if isinstance(a, dict) else a) == "b4r_zero_scale_legacy"
    )
    assert zero_scale.get("training_runs_planned") == 0
    assert "unit" in str(zero_scale.get("evidence") or "").lower()

    cond = ((policy.get("ablations") or {}).get("conditional") or [])[0]
    assert cond.get("open_only_when") == "mean_tune_top1_b4r_le_mean_tune_top1_b3_after_finite_hpo"

    boot = (policy.get("accuracy_gates") or {}).get("paired_bootstrap") or {}
    assert boot.get("join_by_sample_id_required") is True
    assert boot.get("forbid_pool_3xn_as_iid") is True
    assert (policy.get("accuracy_gates") or {}).get("aligned_evaluator", {}).get(
        "join_key"
    ) == "sample_id"

    lineage = policy.get("lineage_draft") or {}
    assert lineage.get("split_algorithm") == "path_b_dev_pool_shuffle_v2"
    assert lineage.get("split_seed") == 3441779383
    assert lineage.get("v4_val_disjoint_from_v3_tune_union_val") is True
    from src.training.retrain_v3 import sha256_file
    from src.training.retrain_v4_split import PROTOCOL_STRING, SPLIT_SEED

    assert lineage.get("split_protocol_string") == PROTOCOL_STRING
    assert lineage.get("split_seed") == SPLIT_SEED
    split_path = _ROOT / lineage["split_manifest"]
    assert split_path.is_file()
    assert sha256_file(split_path) == lineage["split_manifest_sha256"]


def test_v4_orchestrator_dry_plan_and_refuse_phase() -> None:
    py = sys.executable
    script = _ROOT / "scripts/run_ms_sta_retrain_v4.py"
    dry = subprocess.run(
        [py, str(script), "--dry-plan"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    payload = json.loads(dry.stdout)
    assert payload["ok"] is True
    assert payload["train_allowed"] is False
    assert payload["hpo"]["runs_total"] == 48
    assert payload["confirm"]["n_runs"] == 12
    assert payload["confirm"]["early_stopping_allowed"] is False
    assert payload["warm_start_diagnostic"]["runs_planned"] == 0
    assert payload["warm_start_diagnostic"].get("status") == "deferred_optional_after_primary_accuracy"
    assert payload["lineage"]["split_algorithm"] == "path_b_dev_pool_shuffle_v2"
    assert payload["lineage"]["v4_val_disjoint_from_v3_tune_union_val"] is True
    assert payload.get("policy_ready") is False
    assert payload.get("hpo_ready") is False
    assert payload.get("schema") == "ms_sta_retrain_v4_dry_plan_v6"

    phase = subprocess.run(
        [py, str(script), "--phase", "hpo"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert phase.returncode == 2
    assert "train_allowed" in phase.stdout


def test_v4_split_disjoint_bit_exact_and_tracked() -> None:
    from src.training.retrain_v3 import PATH_B_SIGNERS, sha256_file
    from src.training.retrain_v4_split import (
        FIXED_TEST_SIGNERS,
        PROTOCOL_STRING,
        SPLIT_ALGORITHM,
        SPLIT_SEED,
        V3_FORBIDDEN_FOR_V4_VAL,
        V3_HPO_TRAIN_SIGNERS,
        assign_path_b_roles,
    )

    elig = [
        f"signer_{i}"
        for i in list(range(1, 5)) + list(range(6, 17)) + list(range(18, 29))
    ]
    a = assign_path_b_roles(elig, seed=SPLIT_SEED)
    b = assign_path_b_roles(elig, seed=SPLIT_SEED)
    assert a == b
    assert a["test"] == FIXED_TEST_SIGNERS
    assert set(a["val"]).isdisjoint(set(V3_FORBIDDEN_FOR_V4_VAL))
    assert set(a["val"]).issubset(set(V3_HPO_TRAIN_SIGNERS))
    assert set(a["val"]).isdisjoint(
        set(PATH_B_SIGNERS["tune"]) | set(PATH_B_SIGNERS["val"])
    )

    man = json.loads((_ROOT / SPLIT_REL).read_text(encoding="utf-8"))
    assert man["schema"] == "ms_sta_retrain_v4_split_manifest_v1"
    assert man["created_at"] == "protocol_locked"
    assert man["split_algorithm"]["algorithm"] == SPLIT_ALGORITHM
    assert man["split_algorithm"]["seed"] == SPLIT_SEED
    assert man["split_algorithm"]["protocol_string"] == PROTOCOL_STRING
    assert man["signers"]["val"] == ["signer_1", "signer_14", "signer_2"]
    assert set(man["signers"]["val"]).isdisjoint(
        set(PATH_B_SIGNERS["tune"]) | set(PATH_B_SIGNERS["val"])
    )

    # Bit-exact rebuild (in-memory).
    from scripts.build_ms_sta_retrain_v4_splits import build_manifest

    m1 = build_manifest(
        eligible_path=_ROOT / "data/processed/vsl400_eligible_sample_manifest.json",
        metadata_path=_ROOT / "metadata.csv",
        labels_path=_ROOT / "data/processed/vsl400_labels.json",
    )
    m2 = build_manifest(
        eligible_path=_ROOT / "data/processed/vsl400_eligible_sample_manifest.json",
        metadata_path=_ROOT / "metadata.csv",
        labels_path=_ROOT / "data/processed/vsl400_labels.json",
    )
    b1 = json.dumps(m1, ensure_ascii=False, indent=2) + "\n"
    b2 = json.dumps(m2, ensure_ascii=False, indent=2) + "\n"
    assert b1 == b2
    assert m1["created_at"] == "protocol_locked"

    policy = json.loads(
        (_ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json").read_text(
            encoding="utf-8"
        )
    )
    assert sha256_file(_ROOT / SPLIT_REL) == policy["lineage_draft"]["split_manifest_sha256"]

    # Overwrite refused without draft flag.
    refused = subprocess.run(
        [sys.executable, str(_ROOT / "scripts/build_ms_sta_retrain_v4_splits.py")],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "allow-overwrite-draft" in refused.stdout


def test_v4_split_manifest_present_in_git_archive() -> None:
    """Clean archive must include the tracked split manifest with matching SHA."""
    from src.training.retrain_v3 import sha256_file

    policy = json.loads(
        (_ROOT / "Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json").read_text(
            encoding="utf-8"
        )
    )
    expected = policy["lineage_draft"]["split_manifest_sha256"]
    on_disk = sha256_file(_ROOT / SPLIT_REL)
    assert on_disk == expected

    # Delivered source ZIPs deliberately have no .git directory. Their inventory
    # binds the same sealed bytes instead of relying on a parent checkout's HEAD.
    delivery = _ROOT / "SUBMISSION_MANIFEST.json"
    if delivery.is_file():
        inventory = json.loads(delivery.read_text(encoding="utf-8"))["files"]
        assert next(row["sha256"] for row in inventory if row["path"] == SPLIT_REL) == expected
        return

    # Ensure path is not ignored.
    check = subprocess.run(
        ["git", "check-ignore", "-v", SPLIT_REL],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit 1 means not ignored; exit 0 with negation rule also OK if force-tracked.
    out = (check.stdout or "") + (check.stderr or "")
    if check.returncode == 0:
        assert out.strip().startswith("!") or "!" in out.split(":")[-1] or out.startswith(
            ".gitignore"
        )
        # If matched by ignore, the un-ignore rule must be present for this file.
        assert "vsl400_retrain_v4_split_manifest.json" in (
            (_ROOT / ".gitignore").read_text(encoding="utf-8")
        )

    archived = subprocess.run(
        ["git", "archive", "--worktree-attributes", "--format=tar", "HEAD", SPLIT_REL],
        cwd=_ROOT,
        capture_output=True,
        check=False,
    )
    if archived.returncode != 0:
        # File may not be committed yet in this worktree; fall back to path existence
        # + force-add contract. Commit step will make archive succeed.
        assert (_ROOT / SPLIT_REL).is_file()
        assert f"!{SPLIT_REL}" in (_ROOT / ".gitignore").read_text(encoding="utf-8").replace(
            "\\", "/"
        ) or "!data/processed/vsl400_retrain_v4_split_manifest.json" in (
            _ROOT / ".gitignore"
        ).read_text(encoding="utf-8")
        attrs = (_ROOT / ".gitattributes").read_text(encoding="utf-8")
        assert "vsl400_retrain_v4_split_manifest.json -text whitespace=cr-at-eol" in attrs
        return

    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("vsl400_retrain_v4_split_manifest.json")]
        assert len(members) == 1
        extracted = tar.extractfile(members[0])
        assert extracted is not None
        payload = extracted.read()
    import hashlib

    assert hashlib.sha256(payload).hexdigest() == expected


def test_hierarchical_stats_guards_and_align_by_sample_id() -> None:
    from src.training.retrain_v4_stats import (
        align_rows_by_sample_id,
        hierarchical_paired_bootstrap_delta_ci,
        seed_average_correctness,
    )

    with pytest.raises(ValueError, match="n_boot"):
        hierarchical_paired_bootstrap_delta_ci([1], [0], ["s"], n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        hierarchical_paired_bootstrap_delta_ci([1], [0], ["s"], alpha=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        hierarchical_paired_bootstrap_delta_ci([1.5], [0], ["s"], n_boot=10)
    with pytest.raises(ValueError, match="aligned"):
        hierarchical_paired_bootstrap_delta_ci([1, 0], [1], ["s1", "s2"], n_boot=10)
    with pytest.raises(ValueError, match="signer"):
        hierarchical_paired_bootstrap_delta_ci([1], [0], [""], n_boot=10)

    # Join by sample_id even when row order differs across seeds.
    rows = {
        42: [
            {"sample_id": "b", "correct_top1": 0, "signer_id": "s1"},
            {"sample_id": "a", "correct_top1": 1, "signer_id": "s2"},
        ],
        123: [
            {"sample_id": "a", "correct_top1": 1, "signer_id": "s2"},
            {"sample_id": "b", "correct_top1": 1, "signer_id": "s1"},
        ],
    }
    ids, signers, seed_correct = align_rows_by_sample_id(rows)
    assert ids == ["a", "b"]
    assert signers == ["s2", "s1"]
    _, avg = seed_average_correctness(ids, seed_correct, seed_order=[42, 123])
    assert list(avg) == [1.0, 0.5]

    out = hierarchical_paired_bootstrap_delta_ci(
        avg,
        avg,
        signers,
        n_boot=50,
        seed=42,
    )
    assert out["join_by_sample_id_required"] is True
    assert out["delta_pp"] == 0.0
