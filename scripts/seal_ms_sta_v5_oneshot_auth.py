#!/usr/bin/env python3
"""Create-only Phase-B pretest freeze + one-shot test authorization (B4* primary).

Does not run inference. B4-E is excluded from final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_porcelain, git_rev  # noqa: E402

TEST_VARIANTS = ("rgb_fixed", "b3_star_fixed", "b3_mlp_roi_fixed", "b4_star_fixed")
PRIMARY_SEED = 42
FORBIDDEN_ON_TEST = ("b4_star_e_stride4",)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    del args

    porcelain = git_porcelain(_ROOT).strip()
    dirty = [ln for ln in porcelain.splitlines() if not ln.startswith("??") and not ln.startswith("!!")]
    if dirty:
        raise SystemExit(f"dirty_tracked_forbidden:{dirty[:5]}")

    head = git_rev(_ROOT)
    lock_v2 = _ROOT / "Docs/results/ms_sta_v5_m0/VALIDATION_LOCK_V2.json"
    closeout = _ROOT / "Docs/results/ms_sta_v5_e/V5_E_CLOSEOUT.json"
    claim = _ROOT / "Docs/results/ms_sta_v5_e/DEFENSE_CLAIM_POLICY.json"
    confirm = load_json(_ROOT / "Docs/results/ms_sta_v5_m0_runs/CONFIRM_RESULTS.json")
    policy = load_json(_ROOT / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json")
    split_rel = policy["split_manifest_rel"]
    split_path = _ROOT / split_rel
    split_sha = sha256_file(split_path)
    if split_sha != policy["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")

    close = load_json(closeout)
    if close.get("status") != "CLOSED_SUPPLEMENTARY_NO_GO":
        raise SystemExit("requires_e_closeout")
    if close.get("retain_final_model") != "b4_star_fixed":
        raise SystemExit("retain_model_mismatch")

    ckpts: dict[str, dict[str, Any]] = {}
    for variant in TEST_VARIANTS:
        rows = [t for t in confirm["trials"] if t["variant"] == variant and int(t["seed"]) == PRIMARY_SEED]
        if len(rows) != 1:
            raise SystemExit(f"missing_seed42:{variant}")
        rel = str(rows[0]["best_checkpoint"]).replace("\\", "/")
        path = _ROOT / rel
        if not path.is_file():
            raise SystemExit(f"checkpoint_missing:{rel}")
        ckpts[variant] = {
            "checkpoint_rel": rel,
            "checkpoint_sha256": sha256_file(path),
            "dropout": float(rows[0]["dropout"]),
            "lr": float(rows[0]["lr"]),
            "weight_decay": float(rows[0]["weight_decay"]),
            "best_val_top1": float(rows[0]["best_val_top1"]),
        }

    freeze = {
        "schema": "ms_sta_v5_pretest_freeze_v1",
        "status": "FROZEN",
        "immutable": True,
        "proposed_final_model": "b4_star_fixed",
        "demo_checkpoint": "b4_star_fixed",
        "primary_seed": PRIMARY_SEED,
        "test_variants": list(TEST_VARIANTS),
        "forbidden_on_test": list(FORBIDDEN_ON_TEST),
        "primary_contrast": "b4_star_fixed_vs_b3_mlp_roi_fixed",
        "head_at_freeze": head,
        "validation_lock_v2_sha256": sha256_file(lock_v2),
        "v5_e_closeout_sha256": sha256_file(closeout),
        "defense_claim_policy_sha256": sha256_file(claim),
        "split_manifest_rel": split_rel,
        "split_manifest_sha256": split_sha,
        "expected_test_n": 6165,
        "expected_test_signers": ["signer_24", "signer_25", "signer_26", "signer_27", "signer_28"],
        "checkpoints_seed42": ckpts,
        "notes": [
            "B4-E is closed as supplementary NO-GO and must not be evaluated on test.",
            "One-shot retrospective signer-disjoint test only after ONE_SHOT_TEST_AUTHORIZATION.",
            "No retrain; no hyperparameter change after this freeze.",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    freeze_path = _ROOT / "Docs/results/ms_sta_v5_oneshot/PRETEST_FREEZE.json"
    freeze_sha = atomic_write(freeze_path, freeze)

    auth = {
        "schema": "ms_sta_v5_oneshot_test_authorization_v1",
        "status": "AUTHORIZED",
        "allow_final_test": True,
        "allow_demo": False,
        "confirm_retrospective_test": True,
        "proposed_final_model": "b4_star_fixed",
        "primary_seed": PRIMARY_SEED,
        "test_variants": list(TEST_VARIANTS),
        "forbidden_on_test": list(FORBIDDEN_ON_TEST),
        "primary_contrast": "b4_star_fixed_vs_b3_mlp_roi_fixed",
        "pretest_freeze_rel": "Docs/results/ms_sta_v5_oneshot/PRETEST_FREEZE.json",
        "pretest_freeze_sha256": freeze_sha,
        "validation_lock_v2_sha256": freeze["validation_lock_v2_sha256"],
        "v5_e_closeout_sha256": freeze["v5_e_closeout_sha256"],
        "split_manifest_sha256": split_sha,
        "expected_test_n": 6165,
        "execution_head_at_auth": head,
        "oneshot_command": (
            "_venvs/train_cuda/Scripts/python.exe scripts/run_ms_sta_v5_oneshot_test.py "
            "--confirm-retrospective-test"
        ),
        "notes": [
            "Single authorized opening of signer-disjoint test for frozen seed-42 checkpoints.",
            "B4-E must not be loaded.",
            "Create-only TEST_REPORT; refuse overwrite.",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    auth_path = _ROOT / "Docs/results/ms_sta_v5_oneshot/ONE_SHOT_TEST_AUTHORIZATION.json"
    auth_sha = atomic_write(auth_path, auth)

    print(
        json.dumps(
            {
                "event": "ONESHOT_AUTH_SEALED",
                "pretest_freeze_sha256": freeze_sha,
                "oneshot_auth_sha256": auth_sha,
                "head": head,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
