# -*- coding: utf-8 -*-
"""B4Q-V1 invariant helpers — fail-closed hash/schema checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_policy(path: Path | None = None) -> dict[str, Any]:
    p = path or (ROOT / "Docs/results/ms_sta_v6_b4q_policy/B4Q_V1_POLICY.json")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("status") != "FROZEN" or not obj.get("immutable"):
        raise RuntimeError("b4q_policy_not_frozen")
    return obj


def verify_b4_checkpoint(policy: dict[str, Any] | None = None) -> str:
    pol = policy or load_policy()
    ckpt = ROOT / pol["b4"]["checkpoint_rel"]
    digest = sha256_file(ckpt)
    if digest != pol["b4"]["checkpoint_sha256"]:
        raise RuntimeError(f"b4_checkpoint_sha_mismatch:{digest}")
    return digest


def verify_slowfast_checkpoint(policy: dict[str, Any] | None = None) -> str:
    pol = policy or load_policy()
    ckpt = ROOT / pol["rgb_token_v3"]["slowfast_checkpoint_rel"]
    digest = sha256_file(ckpt)
    if digest != pol["rgb_token_v3"]["slowfast_checkpoint_sha256"]:
        raise RuntimeError(f"slowfast_sha_mismatch:{digest}")
    return digest


def verify_split_hashes(policy: dict[str, Any] | None = None) -> None:
    pol = policy or load_policy()
    split = pol["split"]
    for key_rel, key_sha in (
        ("manifest_rel", "manifest_sha256"),
        ("eligible_manifest_rel", "eligible_manifest_sha256"),
        ("labels_rel", "labels_sha256"),
    ):
        digest = sha256_file(ROOT / split[key_rel])
        if digest != split[key_sha]:
            raise RuntimeError(f"split_hash_mismatch:{key_rel}:{digest}")


def region_order_ok(regions: list[str], policy: dict[str, Any] | None = None) -> bool:
    pol = policy or load_policy()
    return list(regions) == list(pol["rgb_token_v3"]["regions_order_canonical"])
