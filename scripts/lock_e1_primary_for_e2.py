"""Create E2 defense lock copies from E1-v2 primary-seed checkpoints.

Does NOT modify E1-v2 source checkpoints under data/processed/ms_sta_e1_v2/.
Writes locked copies under data/processed/ms_sta_e2_lock/ and updates registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

E1_V2_DIR = _ROOT / "data" / "processed" / "ms_sta_e1_v2"
DEFENSE_DIR = _ROOT / "data" / "processed" / "ms_sta_e2_lock"
REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
POLICY = _ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY.json"
LOCK_MANIFEST = _ROOT / "logs" / "gate_e2" / "E2_LOCK_MANIFEST.json"
AGGREGATE = _ROOT / "logs" / "gate_e1_v2" / "MATRIX_AGGREGATE.json"

PRIMARY_SEED = 42
ACTIVE_VARIANT = "b4"
POLICY_SHA = "12cf1176c94c43b9bef87d1da65cb5c911961a0afccedf2414f09fed963ae53a"
ELIGIBLE_SHA = "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_ROOT.resolve()).as_posix()


def _lock_copy(
    *,
    source: Path,
    dest: Path,
    variant: str,
    b0_identity_role: str | None,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == dest.resolve():
        raise ValueError("refuse in-place lock of source checkpoint")
    # Never write into E1-v2 source dir or preserved E1-v1 dir.
    forbidden_parents = {
        (E1_V2_DIR.resolve()),
        (_ROOT / "data" / "processed" / "ms_sta").resolve(),
    }
    if dest.resolve().parent in forbidden_parents:
        raise ValueError(f"refuse writing lock copy into preserved/source dir: {dest.parent}")

    source_sha = _sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if str(payload.get("variant")) != variant:
        raise ValueError(f"source variant={payload.get('variant')!r} != {variant!r}")
    if int(payload.get("seed")) != PRIMARY_SEED:
        raise ValueError(f"source seed={payload.get('seed')!r} != {PRIMARY_SEED}")
    if payload.get("e1_policy_sha256") != POLICY_SHA:
        raise ValueError("source e1_policy_sha256 mismatch vs E1-v2 policy")
    if payload.get("sample_manifest_sha256") != ELIGIBLE_SHA:
        raise ValueError("source sample_manifest_sha256 mismatch")
    if variant == "b0" and payload.get("b0_identity_role") != "b0_e1":
        raise ValueError("B0 source missing b0_identity_role=b0_e1")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        raise FileExistsError(f"defense copy already exists (refuse overwrite): {dest}")

    payload = dict(payload)
    payload["e1_selected"] = True
    payload["evaluation_locked"] = True
    payload["defense_ready"] = False
    payload["seed"] = PRIMARY_SEED
    payload["variant"] = variant
    payload["e1_version"] = payload.get("e1_version") or "v2"
    payload["schema_version"] = max(int(payload.get("schema_version") or 0), 2)
    payload["source_checkpoint"] = _rel(source)
    payload["source_checkpoint_sha256"] = source_sha
    payload["locked_checkpoint"] = _rel(dest)
    payload["lock_created_at"] = datetime.now(timezone.utc).isoformat()
    payload["lock_primary_seed"] = PRIMARY_SEED
    payload["test_eval"] = "deferred"
    if b0_identity_role is not None:
        payload["b0_identity_role"] = b0_identity_role
    # First save without self-hash; second save attaches locked sha of first bytes.
    payload.pop("locked_checkpoint_sha256", None)
    torch.save(payload, dest)
    locked_sha_content = _sha256(dest)
    payload["locked_checkpoint_sha256"] = locked_sha_content
    torch.save(payload, dest)
    locked_sha_file = _sha256(dest)
    return {
        "variant": variant,
        "seed": PRIMARY_SEED,
        "source_checkpoint": _rel(source),
        "source_checkpoint_sha256": source_sha,
        "locked_checkpoint": _rel(dest),
        "locked_checkpoint_sha256": locked_sha_file,
        "locked_checkpoint_content_sha256_before_self_hash_field": locked_sha_content,
        "b0_identity_role": payload.get("b0_identity_role"),
        "val": payload.get("val"),
        "e1_policy_sha256": payload.get("e1_policy_sha256"),
        "sample_manifest_sha256": payload.get("sample_manifest_sha256"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Lock E1-v2 primary seed artifacts for E2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not POLICY.is_file() or _sha256(POLICY) != POLICY_SHA:
        print(f"[FAIL] E1 policy sha mismatch; expected {POLICY_SHA}")
        return 2
    if not AGGREGATE.is_file():
        print(f"[FAIL] missing E1-v2 aggregate: {AGGREGATE}")
        return 2
    agg = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    gate = agg.get("consistent_improvement_gate") or {}
    if gate.get("pass") is not True:
        print("[FAIL] E1-v2 consistent_improvement_gate.pass != true")
        return 2

    src_b0 = E1_V2_DIR / f"b0_seed{PRIMARY_SEED}.pth"
    src_b4 = E1_V2_DIR / f"{ACTIVE_VARIANT}_seed{PRIMARY_SEED}.pth"
    dst_b0 = DEFENSE_DIR / f"b0_seed{PRIMARY_SEED}.pth"
    dst_b4 = DEFENSE_DIR / f"{ACTIVE_VARIANT}_seed{PRIMARY_SEED}.pth"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "would_lock": [
                        {"src": _rel(src_b0), "dst": _rel(dst_b0)},
                        {"src": _rel(src_b4), "dst": _rel(dst_b4)},
                    ]
                },
                indent=2,
            )
        )
        return 0

    # Integrity: source files must remain unchanged — copy via load/save only to dest.
    b0_info = _lock_copy(
        source=src_b0, dest=dst_b0, variant="b0", b0_identity_role="b0_e1"
    )
    b4_info = _lock_copy(
        source=src_b4, dest=dst_b4, variant=ACTIVE_VARIANT, b0_identity_role=None
    )

    # Confirm sources untouched (still same sha as recorded).
    if _sha256(src_b0) != b0_info["source_checkpoint_sha256"]:
        print("[FAIL] B0 source mutated during lock")
        return 2
    if _sha256(src_b4) != b4_info["source_checkpoint_sha256"]:
        print("[FAIL] B4 source mutated during lock")
        return 2

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    reg["status"] = "NO-GO"
    reg["defense_ready"] = False
    reg["e1_pass"] = True
    reg["e1_selected"] = True
    reg["evaluation_locked"] = True
    reg["e2_pass"] = False
    reg["e2_unlocked"] = False
    reg["active_variant"] = ACTIVE_VARIANT
    reg["checkpoint"] = dst_b4.name
    reg["checkpoint_sha256"] = b4_info["locked_checkpoint_sha256"]
    reg["b0_checkpoint"] = dst_b0.name
    reg["b0_checkpoint_sha256"] = b0_info["locked_checkpoint_sha256"]
    reg["ckpts_dir"] = _rel(DEFENSE_DIR)
    reg["eligible_sample_manifest"] = "data/processed/vsl400_eligible_sample_manifest.json"
    reg["eligible_sample_manifest_sha256"] = ELIGIBLE_SHA
    reg["e2_lock"] = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "primary_seed": PRIMARY_SEED,
        "active_variant": ACTIVE_VARIANT,
        "b0_identity_role": "b0_e1",
        "gate3_forbidden_as_e2_b0": True,
        "e1_v2_sources_immutable": True,
        "b0": b0_info,
        "active": b4_info,
        "claim_limits": {
            "temporal_b1_gt_b0": "consistent_on_val",
            "structured_fusion_b4_gt_b3": "consistent_on_val",
            "spatial_attention_b4_vs_b5": "mixed_inconclusive",
        },
        "aggregate": _rel(AGGREGATE),
    }
    reg["checkpoints"] = {
        "b0": {
            "path": _rel(dst_b0),
            "sha256": b0_info["locked_checkpoint_sha256"],
            "role": "b0_e1",
            "seed": PRIMARY_SEED,
            "source_checkpoint_sha256": b0_info["source_checkpoint_sha256"],
        },
        "b4": {
            "path": _rel(dst_b4),
            "sha256": b4_info["locked_checkpoint_sha256"],
            "role": "active_variant",
            "seed": PRIMARY_SEED,
            "source_checkpoint_sha256": b4_info["source_checkpoint_sha256"],
        },
    }
    reasons = reg.get("reasons") or []
    note = (
        "E1-v2 primary seed 42 locked for E2 (B4 active, B0-E1 counterpart); "
        "Gate3 forbidden as E2 B0; defense_ready=false until E2"
    )
    reg["reasons"] = [note] + [r for r in reasons if "locked for E2" not in r]
    reg["updated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    LOCK_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "option_b_e2_lock_manifest_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "primary_seed": PRIMARY_SEED,
        "active_variant": ACTIVE_VARIANT,
        "b0_identity_role": "b0_e1",
        "gate3_forbidden_as_e2_b0": True,
        "ckpts_dir": _rel(DEFENSE_DIR),
        "b0": b0_info,
        "active": b4_info,
        "registry": {
            "e1_pass": True,
            "e1_selected": True,
            "evaluation_locked": True,
            "defense_ready": False,
            "e2_pass": False,
            "status": "NO-GO",
            "checkpoint": dst_b4.name,
            "checkpoint_sha256": b4_info["locked_checkpoint_sha256"],
            "b0_checkpoint": dst_b0.name,
            "b0_checkpoint_sha256": b0_info["locked_checkpoint_sha256"],
        },
    }
    LOCK_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["registry"] | {"manifest": _rel(LOCK_MANIFEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
