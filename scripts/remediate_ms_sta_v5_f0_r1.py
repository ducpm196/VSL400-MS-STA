#!/usr/bin/env python3
"""F0-R1 remediation: archive broken VideoMAE-small safetensors and regenerate with metadata.

Preserves original F0 JSON artifacts. Writes versioned R1 weights/conversion/realization.
Does not run GPU. Does not mutate F0_FINAL_AUDIT / report / completion / consumption.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from f0_feasibility.campaign_paths import F0_R1_PATHS  # noqa: E402
from ms_sta_v5_f0_lib import (  # noqa: E402
    ENV_REALIZATION_REL,
    EXPECTED_INDEX_POLICY_SHA256,
    EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
    EXPECTED_PROTOCOL_PLAN_SHA256,
    EXPECTED_SNAPSHOT_CONTRACT_SHA256,
    PROTOCOL_PLAN_REL,
    REALIZATION_REL,
    WEIGHTS_DIR_REL,
    WEIGHTS_REL,
    load_json,
    sha256_file,
    validate_realization_manifest,
    validate_realization_manifest_audit,
    validate_videomae_small_conversion_evidence,
    validate_weights_manifest,
)
from realize_ms_sta_v5_f0 import _convert_bin_to_safetensors  # noqa: E402

_ARCHIVE_REL = "Docs/results/ms_sta_v5_f0/archive/f0_fail_90820dc/videomae_vit_small"
_SOURCE_BIN_REL = f"{WEIGHTS_DIR_REL}/videomae_vit_small/pytorch_model.bin"
_LIVE_ST_REL = f"{WEIGHTS_DIR_REL}/videomae_vit_small/model.safetensors"
_PINNED_SOURCE_SHA = "d69585904fec7e507bf2edfba4a7abe2b92def9afc76460d1dc14dbbf864bcfc"
_HISTORICAL_CONVERTED_SHA = "aeae93f62f724e206527b88c36cdfc8d80613a2aac57839f9ce21314ca33428b"


def _write_json(rel: str, obj: dict) -> str:
    path = _ROOT / rel
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError(f"crlf_forbidden:{rel}")
    if path.is_file():
        raise RuntimeError(f"already_exists:{rel}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256_file(path)


def _smoke_from_pretrained(converted: Path) -> dict:
    """Local-only Transformers load; network must stay disabled."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import VideoMAEForVideoClassification

    model = VideoMAEForVideoClassification.from_pretrained(
        str(converted.parent),
        local_files_only=True,
    )
    n = sum(p.numel() for p in model.parameters())
    return {"ok": True, "param_count": int(n), "local_files_only": True}


def main() -> int:
    source = _ROOT / _SOURCE_BIN_REL
    live = _ROOT / _LIVE_ST_REL
    if not source.is_file():
        raise SystemExit(f"missing_source_bin:{_SOURCE_BIN_REL}")
    source_sha = sha256_file(source)
    if source_sha != _PINNED_SOURCE_SHA:
        raise SystemExit(f"source_sha_drift:{source_sha}")

    archive_dir = _ROOT / _ARCHIVE_REL
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / "model.safetensors"
    if live.is_file() and not archived.is_file():
        shutil.copy2(live, archived)
    if archived.is_file():
        archived_sha = sha256_file(archived)
        if archived_sha != _HISTORICAL_CONVERTED_SHA and sha256_file(live) == _HISTORICAL_CONVERTED_SHA:
            # Prefer exact historical bytes when live still matches FAIL-era digest.
            shutil.copy2(live, archived)
            archived_sha = sha256_file(archived)
        note = {
            "schema": "ms_sta_v5_f0_r1_archive_note_v1",
            "status": "ARCHIVED",
            "reason": "f0_fail_videomae_safetensors_metadata_none",
            "historical_converted_sha256": _HISTORICAL_CONVERTED_SHA,
            "archived_sha256": sha256_file(archived),
            "source_sha256": source_sha,
            "original_f0_final_audit": "Docs/results/ms_sta_v5_f0/F0_FINAL_AUDIT.json",
            "original_conversion_evidence": (
                "Docs/results/ms_sta_v5_f0/F0_VIDEOMAE_SMALL_CONVERSION_EVIDENCE.json"
            ),
            "immutable": True,
        }
        note_path = archive_dir / "ARCHIVE_NOTE.json"
        if not note_path.is_file():
            note_path.write_bytes(
                (json.dumps(note, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            )

    # Regenerate live contract local_name with required safetensors metadata.
    tmp = live.with_suffix(".safetensors.r1_tmp")
    if tmp.is_file():
        tmp.unlink()
    _convert_bin_to_safetensors(source, tmp)
    os.replace(tmp, live)
    converted_sha = sha256_file(live)

    from safetensors import safe_open

    with safe_open(str(live), framework="pt") as handle:
        meta = handle.metadata()
    if meta != {"format": "pt"}:
        raise SystemExit(f"metadata_not_format_pt:{meta}")

    smoke = _smoke_from_pretrained(live)

    # Build R1 weights manifest from original entries with updated videomae sha.
    old_weights = load_json(_ROOT / WEIGHTS_REL)
    entries = []
    for e in old_weights.get("entries") or []:
        entry = dict(e)
        if (
            entry.get("candidate_id") == "videomae_vit_small"
            and entry.get("role") == "weights"
            and str(entry.get("local_path", "")).endswith("model.safetensors")
        ):
            entry["sha256"] = converted_sha
            conv = dict(entry.get("conversion") or {})
            conv["source_sha256"] = source_sha
            conv["conversion"] = "torch_state_dict_to_safetensors"
            conv["safetensors_metadata"] = {"format": "pt"}
            conv["reason"] = "f0_r1_remediate_missing_safetensors_metadata"
            entry["conversion"] = conv
        entries.append(entry)

    r1_weights = {
        "schema": "ms_sta_v5_f0_weights_manifest_v1",
        "status": "REALIZED",
        "campaign": "f0_r1",
        "parent_f0_weights_manifest": WEIGHTS_REL,
        "parent_f0_weights_manifest_sha256": sha256_file(_ROOT / WEIGHTS_REL),
        "approval_sha256": old_weights.get("approval_sha256"),
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "bind_snapshot_contract": True,
        "snapshot_contract_sha256": EXPECTED_SNAPSHOT_CONTRACT_SHA256,
        "index_policy_sha256": EXPECTED_INDEX_POLICY_SHA256,
        "entries": entries,
        "remediation": {
            "videomae_vit_small": "regenerated_safetensors_metadata_format_pt",
            "source_sha256_pinned": source_sha,
            "converted_sha256": converted_sha,
        },
    }
    weights_sha = _write_json(F0_R1_PATHS.weights_manifest_rel, r1_weights)
    plan = load_json(_ROOT / PROTOCOL_PLAN_REL)
    validate_weights_manifest(
        r1_weights, root=_ROOT, plan=plan, plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256
    )

    # R1 conversion evidence (versioned; does not overwrite F0 evidence).
    import torch
    from safetensors.torch import load_file

    src = torch.load(source, map_location="cpu")
    if isinstance(src, dict) and "state_dict" in src:
        src = src["state_dict"]
    dst = load_file(str(live))
    src_keys = sorted(src.keys())
    dst_keys = sorted(dst.keys())
    if src_keys != dst_keys:
        raise SystemExit("keysets_not_equal")
    for k in src_keys:
        if not torch.equal(src[k].contiguous(), dst[k].contiguous()):
            raise SystemExit(f"tensor_not_bit_exact:{k}")

    evidence = {
        "schema": "ms_sta_v5_f0_videomae_small_conversion_evidence_v1",
        "status": "VERIFIED",
        "campaign": "f0_r1",
        "candidate_id": "videomae_vit_small",
        "repo_id": "MCG-NJU/videomae-small-finetuned-kinetics",
        "revision": "8fc0b88efda5274a4c2e74d3157c02f1ec0764a7",
        "source_hf_path": "pytorch_model.bin",
        "source_local_path": _SOURCE_BIN_REL,
        "converted_local_name": "model.safetensors",
        "converted_local_path": _LIVE_ST_REL,
        "conversion_method": "torch_state_dict_to_safetensors",
        "reason": "f0_r1_remediate_missing_safetensors_metadata",
        "source_sha256": source_sha,
        "converted_sha256": converted_sha,
        "source_key_count": len(src_keys),
        "converted_key_count": len(dst_keys),
        "keysets_equal": True,
        "dtype_shape_equal": True,
        "tensors_bit_exact": True,
        "safetensors_metadata": {"format": "pt"},
        "safetensors_metadata_verified": True,
        "from_pretrained_local_smoke": smoke,
        "weights_manifest_sha256": weights_sha,
        "parent_f0_conversion_evidence": (
            "Docs/results/ms_sta_v5_f0/F0_VIDEOMAE_SMALL_CONVERSION_EVIDENCE.json"
        ),
        "archived_broken_safetensors": f"{_ARCHIVE_REL}/model.safetensors",
    }
    conv_sha = _write_json(F0_R1_PATHS.conversion_evidence_rel, evidence)
    validate_videomae_small_conversion_evidence(
        evidence,
        root=_ROOT,
        weights_manifest_sha256=weights_sha,
        recompute_tensors=True,
        weights_manifest_rel=F0_R1_PATHS.weights_manifest_rel,
    )

    # R1 realization binds R1 weights; reuses F0 env realization digests/evidence.
    old_realization = load_json(_ROOT / REALIZATION_REL)
    env_sha = sha256_file(_ROOT / ENV_REALIZATION_REL)
    r1_realization = dict(old_realization)
    r1_realization["campaign"] = "f0_r1"
    r1_realization["weights_manifest_sha256"] = weights_sha
    r1_realization["env_realization_sha256"] = env_sha
    r1_realization["parent_f0_realization_manifest"] = REALIZATION_REL
    r1_realization["parent_f0_realization_manifest_sha256"] = sha256_file(_ROOT / REALIZATION_REL)
    r1_realization["remediation"] = {
        "slowfast": "module_boundary_freeze_matcher",
        "videomae_vit_small": "safetensors_metadata_format_pt",
    }
    man_sha = _write_json(F0_R1_PATHS.realization_manifest_rel, r1_realization)
    validate_realization_manifest(
        r1_realization,
        root=_ROOT,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        weights_manifest_sha256=weights_sha,
        env_realization_sha256=env_sha,
    )

    r1_audit = {
        "schema": "ms_sta_v5_f0_realization_manifest_audit_v1",
        "audit_verdict": "PASS",
        "campaign": "f0_r1",
        "realization_manifest": F0_R1_PATHS.realization_manifest_rel,
        "plan_sha256": EXPECTED_PROTOCOL_PLAN_SHA256,
        "plan_lock_sha256": EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        "weights_manifest_sha256": weights_sha,
        "env_realization_sha256": env_sha,
        "realization_manifest_sha256": man_sha,
        "gpu_feasibility_authorized": False,
        "execution_approval_authorized": False,
        "auto_chain": False,
        "notes": [
            "R1 realization audit after adapter remediation; GPU still requires R1 execution seal.",
        ],
    }
    audit_sha = _write_json(F0_R1_PATHS.realization_manifest_audit_rel, r1_audit)
    validate_realization_manifest_audit(
        r1_audit,
        root=_ROOT,
        plan_sha256=EXPECTED_PROTOCOL_PLAN_SHA256,
        plan_lock_sha256=EXPECTED_PROTOCOL_PLAN_LOCK_SHA256,
        weights_manifest_sha256=weights_sha,
        env_realization_sha256=env_sha,
        realization_manifest_sha256=man_sha,
        realization_manifest_rel=F0_R1_PATHS.realization_manifest_rel,
    )

    out = {
        "ok": True,
        "campaign": "f0_r1",
        "source_sha256": source_sha,
        "converted_sha256": converted_sha,
        "weights_manifest_sha256": weights_sha,
        "conversion_evidence_sha256": conv_sha,
        "realization_manifest_sha256": man_sha,
        "realization_manifest_audit_sha256": audit_sha,
        "from_pretrained_local_smoke": smoke,
        "preserved_f0_final_audit": "Docs/results/ms_sta_v5_f0/F0_FINAL_AUDIT.json",
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
