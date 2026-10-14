"""Retrain-v4 Path-B split helpers (deterministic; no training).

Creates a *new* signer-disjoint development partition so v3 tune+val
(used while designing B4R) are not reused in v4 confirmation validation.
Test signers 24–28 stay fixed and remain retrospective.

v4 val is drawn only from the v3 HPO-train pool and must be disjoint from
``v3_tune ∪ v3_val``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from src.training.retrain_v3 import PATH_B_SIGNERS

# Locked before any v4 training. Changing this requires a new manifest SHA.
SPLIT_ALGORITHM = "path_b_dev_pool_shuffle_v2"
PROTOCOL_STRING = (
    "ms_sta_retrain_v4|path_b_dev_pool_shuffle_v2|"
    "n_val=3|n_tune=3|"
    "test=signer_24..28|"
    "val_from=v3_hpo_train_only|"
    "val_disjoint=v3_tune_union_v3_val|"
    "shuffle=random.Random(seed).shuffle"
)
# First 8 hex digits of SHA-256(protocol) → deterministic uint32 seed (auditable).
SPLIT_SEED = int(hashlib.sha256(PROTOCOL_STRING.encode("utf-8")).hexdigest()[:8], 16)
# Scientific artifact timestamp is protocol-locked (not wall-clock).
ARTIFACT_CREATED_AT = "protocol_locked"

N_VAL_SIGNERS = 3
N_TUNE_SIGNERS = 3

FIXED_TEST_SIGNERS = list(PATH_B_SIGNERS["test"])
FIXED_MISSING_SIGNERS = list(PATH_B_SIGNERS["missing"])
V3_HPO_TRAIN_SIGNERS = list(PATH_B_SIGNERS["hpo_train"])
V3_TUNE_SIGNERS = list(PATH_B_SIGNERS["tune"])
V3_VAL_SIGNERS = list(PATH_B_SIGNERS["val"])
V3_FORBIDDEN_FOR_V4_VAL = sorted(set(V3_TUNE_SIGNERS) | set(V3_VAL_SIGNERS))


def assign_path_b_roles(
    eligible_signers: list[str],
    *,
    seed: int = SPLIT_SEED,
    n_val: int = N_VAL_SIGNERS,
    n_tune: int = N_TUNE_SIGNERS,
) -> dict[str, list[str]]:
    """Deterministically partition eligible signers into Path-B roles.

    - test / missing are fixed.
    - v4 val ⊆ v3 hpo_train and disjoint from v3 tune∪val.
    - remaining development signers are shuffled into tune then hpo_train.
    - confirm_train = hpo_train ∪ tune.
    """
    elig = {str(s) for s in eligible_signers}
    test = list(FIXED_TEST_SIGNERS)
    missing = list(FIXED_MISSING_SIGNERS)
    for s in test:
        if s not in elig:
            raise ValueError(f"fixed test signer missing from eligible set: {s}")
    for s in missing:
        if s in elig:
            raise ValueError(f"missing signer unexpectedly has eligible samples: {s}")

    forbidden_val = set(V3_FORBIDDEN_FOR_V4_VAL)
    val_pool = sorted(
        (set(V3_HPO_TRAIN_SIGNERS) & elig) - forbidden_val - set(test)
    )
    if len(val_pool) < n_val:
        raise ValueError(
            f"v4 val pool too small: |pool|={len(val_pool)} need >= {n_val}"
        )

    rng = random.Random(int(seed))
    val_order = list(val_pool)
    rng.shuffle(val_order)
    val = sorted(val_order[:n_val])
    if set(val) & forbidden_val:
        raise ValueError(f"v4 val overlaps v3 tune∪val: {sorted(set(val) & forbidden_val)}")

    dev = sorted(elig - set(test))
    remainder = sorted(set(dev) - set(val))
    if len(remainder) < n_tune + 1:
        raise ValueError(
            f"remainder pool too small: |rem|={len(remainder)} need >= {n_tune + 1}"
        )
    rem_order = list(remainder)
    rng.shuffle(rem_order)
    tune = sorted(rem_order[:n_tune])
    hpo_train = sorted(rem_order[n_tune:])
    confirm_train = sorted(set(hpo_train) | set(tune))

    roles = {
        "hpo_train": hpo_train,
        "tune": tune,
        "confirm_train": confirm_train,
        "val": val,
        "test": test,
        "missing": missing,
    }
    _validate_roles(roles, elig)
    return roles


def _validate_roles(roles: dict[str, list[str]], elig: set[str]) -> None:
    hpo = set(roles["hpo_train"])
    tune = set(roles["tune"])
    confirm = set(roles["confirm_train"])
    val = set(roles["val"])
    test = set(roles["test"])
    missing = set(roles["missing"])
    forbidden_val = set(V3_FORBIDDEN_FOR_V4_VAL)

    if hpo & tune:
        raise ValueError("hpo_train and tune must be disjoint")
    if hpo | tune != confirm:
        raise ValueError("hpo_train ∪ tune must equal confirm_train")
    if confirm & val:
        raise ValueError("confirm_train overlaps val")
    if (confirm | val) & test:
        raise ValueError("development overlaps test")
    if missing & (confirm | val | test | hpo | tune):
        raise ValueError("missing signer appears in a role")
    if confirm | val | test != elig:
        raise ValueError("confirm_train∪val∪test must equal eligible signer set")
    if val & forbidden_val:
        raise ValueError(
            "v4 val must be disjoint from v3 tune∪val; "
            f"overlap={sorted(val & forbidden_val)}"
        )
    if not val.issubset(set(V3_HPO_TRAIN_SIGNERS)):
        raise ValueError(
            "v4 val must be subset of v3 hpo_train pool; "
            f"extras={sorted(val - set(V3_HPO_TRAIN_SIGNERS))}"
        )


def split_algorithm_metadata() -> dict[str, Any]:
    return {
        "algorithm": SPLIT_ALGORITHM,
        "protocol_string": PROTOCOL_STRING,
        "seed": SPLIT_SEED,
        "seed_derivation": "int(sha256(protocol_string).hexdigest()[:8], 16)",
        "artifact_created_at": ARTIFACT_CREATED_AT,
        "n_val_signers": N_VAL_SIGNERS,
        "n_tune_signers": N_TUNE_SIGNERS,
        "fixed_test_signers": list(FIXED_TEST_SIGNERS),
        "fixed_missing_signers": list(FIXED_MISSING_SIGNERS),
        "v3_hpo_train_pool_for_val": list(V3_HPO_TRAIN_SIGNERS),
        "v3_tune_signers": list(V3_TUNE_SIGNERS),
        "v3_val_signers": list(V3_VAL_SIGNERS),
        "v4_val_disjoint_from_v3_tune_union_val": True,
        "shuffle_impl": "random.Random(seed).shuffle(val_pool then remainder)",
        "test_claim_label": "retrospective_signer_disjoint_confirmation",
        "forbid_reuse_v3_validation_as_independent_confirmation": True,
    }
