"""Rebuild MS-STA defense companion docs WITHOUT mutating immutable E2/D4 bytes.

Immutable science artifacts (do not overwrite):
  Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json
  Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json
  Docs/results/ms_sta_e1_policy/E1_POLICY.json  (frozen sha 12cf1176…)

Companion outputs (safe to rewrite):
  Docs/results/ms_sta_defense_go/FINAL_RESULTS.json
  Docs/results/ms_sta_defense_go/EVIDENCE_INDEX.json
  Docs/results/ms_sta_defense_go/ARTIFACT_MANIFEST.json
  Docs/results/ms_sta_e1_policy/E1_POLICY.md + CHECKSUMS.json
  data/processed/ms_sta_registry.json reasons / relative paths / pointers
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ABS_RE = re.compile(r"^[A-Za-z]:\\")

# Locked immutable hashes (CRLF working-tree / binary git)
_E1_V2 = "69aed7858ed716476da6fe786f2836c160b7e45dda3e1c84abe02076bb489c41"
_E2 = "408fc077d92746dc043dc3d457575701b22c7bf4f19b50484450df6b95e31e79"
_D4 = "f5375f225cc1c09c84b8b544990734aef543111781b21bd7cc9233c1d9d126b8"
_POLICY = "12cf1176c94c43b9bef87d1da65cb5c911961a0afccedf2414f09fed963ae53a"
_POLICY_V1 = "30c454d4781750ee778c7935bb151035bdc9b48445a6458db83723bc95501b66"
_SPLIT = "7c11b4119a16ea07c18bfcb870e47369e0e7194194a4770218d0d7d8804d4882"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _rel_path(value: str) -> str:
    if not isinstance(value, str):
        return value
    if not _ABS_RE.match(value):
        return value.replace("\\", "/")
    marker = "VSL_Translation_Project"
    idx = value.replace("/", "\\").lower().find(marker.lower())
    if idx >= 0:
        tail = value[idx + len(marker) :].lstrip("\\/")
        return tail.replace("\\", "/")
    return value.replace("\\", "/")


def _walk_rel(obj):
    if isinstance(obj, dict):
        return {k: _walk_rel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_rel(v) for v in obj]
    if isinstance(obj, str):
        return _rel_path(obj)
    return obj


def _require_sha(path: Path, expect: str, label: str) -> str:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    got = _sha(path)
    if got != expect:
        raise SystemExit(f"{label} sha mismatch expect={expect[:12]} got={got[:12]} path={path}")
    return got


def main() -> int:
    e1_src = _ROOT / "logs" / "gate_e1_v2" / "MATRIX_AGGREGATE.json"
    e1_dst = _ROOT / "Docs" / "results" / "ms_sta_e1_v2" / "MATRIX_AGGREGATE.json"
    e1_dst.parent.mkdir(parents=True, exist_ok=True)
    if e1_src.is_file():
        # Copy only if destination missing or already matches expected; never alter bytes.
        if not e1_dst.is_file() or _sha(e1_dst) != _E1_V2:
            if _sha(e1_src) == _E1_V2:
                shutil.copy2(e1_src, e1_dst)
    e1_sha = _require_sha(e1_dst, _E1_V2, "E1-v2")

    e2_path = _ROOT / "Docs" / "results" / "ms_sta_e2" / "E2_CANONICAL_REPORT.json"
    d4_path = _ROOT / "Docs" / "results" / "ms_sta_d4" / "D4_ACCEPTANCE_REPORT.json"
    policy_path = _ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY.json"
    e2_sha = _require_sha(e2_path, _E2, "E2")
    d4_sha = _require_sha(d4_path, _D4, "D4")
    policy_sha = _require_sha(policy_path, _POLICY, "E1 policy")

    e2 = json.loads(e2_path.read_text(encoding="utf-8-sig"))
    d4 = json.loads(d4_path.read_text(encoding="utf-8-sig"))
    agg = json.loads(e1_dst.read_text(encoding="utf-8-sig"))
    gate = agg.get("consistent_improvement_gate") or {}

    # Companion only — does not mutate E2/D4 files.
    final = {
        "schema": "ms_sta_final_results_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "immutable_git_tip": "785f43194663d736c6130d7c3614a184d69f20dc",
        "defense_status": "GO",
        "publication_source_package": "NO-GO",
        "active_variant": "b4",
        "primary_seed": 42,
        "derived_from": {
            "e2_canonical_sha256": e2_sha,
            "d4_acceptance_sha256": d4_sha,
            "e1_v2_aggregate_sha256": e1_sha,
            "note": (
                "E2/D4 files are immutable science artifacts. "
                "This companion records post-D4 defense GO without rewriting them. "
                "E2.defense_status may remain NO-GO (E2-time); trust this file for defense."
            ),
        },
        "e1_v2": {
            "path": "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json",
            "sha256": e1_sha,
            "split": "validation_only",
            "b0_top1_mean": agg["summary"]["b0"]["top1"]["mean"],
            "b4_top1_mean": agg["summary"]["b4"]["top1"]["mean"],
            "b4_minus_b0_mean_pp": gate.get("mean_delta_top1_vs_b0"),
            "b4_minus_b3_mean_pp": gate.get("mean_delta_top1_vs_b3"),
            "consistent_improvement_pass": gate.get("pass"),
            "primary_seed_42": {
                "b0_top1": 40.69364161849711,
                "b4_top1": 52.02312138728324,
                "b3_top1": 49.2485549132948,
            },
        },
        "e2": {
            "path": "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json",
            "sha256": e2_sha,
            "n": e2.get("n"),
            "b0_e1_top1": e2["summary"]["b0"]["top1"],
            "b4_top1": e2["summary"]["b4"]["top1"],
            "delta_top1_pp": e2["summary"]["delta_top1_pp"],
            "b0_e1_top5": e2["summary"]["b0"]["top5"],
            "b4_top5": e2["summary"]["b4"]["top5"],
            "delta_top5_pp": e2["summary"]["delta_top5_pp"],
            "bootstrap_top1_ci95_pp": e2["bootstrap"]["top1_delta"]["ci95_pp"],
            "mcnemar_top1_exact_p": e2["mcnemar_top1"]["exact_two_sided_p"],
            "scientific_pass": e2.get("scientific_pass"),
            "artifact_defense_status_at_e2": e2.get("defense_status"),
        },
        "d4": {
            "path": "Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json",
            "sha256": d4_sha,
            "all_scenarios_pass": d4.get("all_scenarios_pass"),
            "d4_pass": d4.get("d4_pass"),
        },
        "claim_limits": [
            "Absolute accuracy remains low, especially signer 27.",
            "Spatial-only (B4 vs B5) is mixed; solid claims are full B4 and structured fusion vs B0/B3.",
            "Do not compare B4 directly to VSL400 paper Front-view 76–93% as the same protocol.",
            "Demo is cache-backed offline inference; no real-time or arbitrary-video production claim.",
        ],
        "no_retrain": True,
        "no_rerun_e2": True,
        "no_checkpoint_swap": True,
    }
    final_path = _ROOT / "Docs" / "results" / "ms_sta_defense_go" / "FINAL_RESULTS.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Normalize FINAL to LF for stable packaging hashes (companion, not science lock from tip).
    # Keep as written (typically LF on write_text). Record sha after write.
    final_sha = _sha(final_path)

    policy_md = _ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "E1_POLICY.md"
    policy_md.write_text(
        f"""# MS-STA — E1 Policy (frozen)

> **Policy status:** FROZEN (scientific training contract).
> **Defense status:** **GO** on tip `785f431` after E1-v2 + E2 + D4 — see `Docs/results/ms_sta_defense_go/FINAL_RESULTS.json`.
> **Machine-readable:** [E1_POLICY.json](E1_POLICY.json) (immutable bytes)
> **SHA256 (v2):** `{policy_sha}`
> **Archived v1:** [E1_POLICY_v1.json](E1_POLICY_v1.json) SHA `{_POLICY_V1}`

Do not edit `E1_POLICY.json` in place. Companion defense status lives in FINAL_RESULTS only.
""",
        encoding="utf-8",
    )
    checksums = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SHA256",
        "files": {
            "Docs/results/ms_sta_e1_policy/E1_POLICY.json": policy_sha,
            "Docs/results/ms_sta_e1_policy/E1_POLICY.md": _sha(policy_md),
            "Docs/results/ms_sta_e1_policy/E1_POLICY_v1.json": _POLICY_V1,
            "data/processed/vsl400_e1_split_manifest.json": _SPLIT,
            "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json": e1_sha,
            "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json": e2_sha,
            "Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json": d4_sha,
        },
        "note": "E2/D4/E1 policy JSON are immutable; do not rewrite to change defense_status.",
    }
    (_ROOT / "Docs" / "results" / "ms_sta_e1_policy" / "CHECKSUMS.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )
    (e1_dst.parent / "CHECKSUMS.json").write_text(
        json.dumps(
            {
                "generated_at": checksums["generated_at"],
                "algorithm": "SHA256",
                "files": {"Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json": e1_sha},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    reg_path = _ROOT / "data" / "processed" / "ms_sta_registry.json"
    reg = _walk_rel(json.loads(reg_path.read_text(encoding="utf-8-sig")))
    reg["reasons"] = [
        "Defense GO (B4) after D4 15/15; limited publication claim; cache-backed demo only; candidate disabled.",
        "E1-v2 primary seed 42 locked for E2 (B4 active, B0-E1 counterpart); Gate3 forbidden as E2 B0.",
        "E1 policy v2 FROZEN; E1-v1 preserved as failed optimization.",
        "Temporal promoted: data/features/isolated_sf_temporal_v2_8741189c657a (full_scan pass, n=24752).",
        "D1 skel promoted: data/features/local_skel_v2_66fc5f01ee415fa1 (full_scan pass, n=24752, exclude 019442).",
        "Demo defaults: defense MS-STA B4 when registry GO; Gate3 B0 remains historical baseline only.",
    ]
    if reg.get("e1_policy"):
        reg["e1_policy"]["path"] = "Docs/results/ms_sta_e1_policy/E1_POLICY.json"
        if reg["e1_policy"].get("e1_v1_preserved"):
            reg["e1_policy"]["e1_v1_preserved"]["policy"] = (
                "Docs/results/ms_sta_e1_policy/E1_POLICY_v1.json"
            )
    if reg.get("e2_lock"):
        reg["e2_lock"]["aggregate"] = "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json"
        reg["e2_lock"]["aggregate_sha256"] = e1_sha
    if reg.get("e2_canonical_report"):
        reg["e2_canonical_report"]["path"] = "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json"
        reg["e2_canonical_report"]["report_sha256"] = e2_sha
        reg["e2_canonical_report"]["note"] = (
            "report_sha256 binds immutable E2 file bytes; post-D4 defense SoT is FINAL_RESULTS"
        )
    if reg.get("defense_promotion"):
        reg["defense_promotion"]["d4_acceptance_report"] = (
            "Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json"
        )
        reg["defense_promotion"]["d4_acceptance_sha256"] = d4_sha
    reg["final_results"] = {
        "path": "Docs/results/ms_sta_defense_go/FINAL_RESULTS.json",
        "sha256": final_sha,
    }
    reg["updated_at"] = datetime.now(timezone.utc).isoformat()
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reg_sha = _sha(reg_path)

    idx = {
        "schema": "ms_sta_defense_go_evidence_index_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "defense_status": "Defense GO (not production GO)",
        "immutable_git_tip": "785f43194663d736c6130d7c3614a184d69f20dc",
        "immutable_git_tip_short": "785f431",
        "recommended_tag": "defense-go-b4-v1",
        "no_rerun_e2": True,
        "no_model_change_before_defense": True,
        "locked_kpis": {
            "e2_n": final["e2"]["n"],
            "b0_e1_top1": final["e2"]["b0_e1_top1"],
            "b4_top1": final["e2"]["b4_top1"],
            "delta_top1_pp": final["e2"]["delta_top1_pp"],
            "bootstrap_top1_ci95_pp": final["e2"]["bootstrap_top1_ci95_pp"],
            "mcnemar_top1_exact_p": final["e2"]["mcnemar_top1_exact_p"],
        },
        "defense_message_limits": final["claim_limits"],
        "registry_snapshot": {
            "status": reg.get("status"),
            "defense_ready": reg.get("defense_ready"),
            "d4_pass": reg.get("d4_pass"),
            "e2_pass": reg.get("e2_pass"),
            "active_variant": reg.get("active_variant"),
            "checkpoint": reg.get("checkpoint"),
            "checkpoint_sha256": reg.get("checkpoint_sha256"),
            "ckpts_dir": reg.get("ckpts_dir"),
            "d4_candidate_allowed": reg.get("d4_candidate_allowed"),
        },
        "tracked_evidence": [
            {"path": "Docs/results/ms_sta_defense_go/FINAL_RESULTS.json", "sha256": final_sha},
            {"path": "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json", "sha256": e1_sha},
            {"path": "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json", "sha256": e2_sha},
            {"path": "Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json", "sha256": d4_sha},
            {"path": "Docs/results/ms_sta_e1_policy/E1_POLICY.json", "sha256": policy_sha},
            {"path": "data/processed/ms_sta_registry.json", "sha256": reg_sha},
        ],
        "local_only_artifacts": [
            {
                "path": "data/processed/ms_sta_defense/b4_seed42_defense.pth",
                "release_required": True,
                "sha256": reg.get("checkpoint_sha256"),
            },
            {
                "path": "data/processed/ms_sta_e2_lock/b4_seed42.pth",
                "release_required": True,
                "sha256": (reg.get("checkpoints") or {}).get("b4", {}).get("sha256"),
            },
            {
                "path": "data/processed/ms_sta_e2_lock/b0_seed42.pth",
                "release_required": True,
                "sha256": (reg.get("checkpoints") or {}).get("b0", {}).get("sha256"),
            },
            {
                "path": "logs/eval_ms_sta/E2_b4_vs_b0e1.json",
                "release_required": True,
                "sha256": (reg.get("e2_canonical_report") or {}).get("source_artifact_sha256"),
            },
        ],
        "download": {
            "status": "pending_release",
            "script": "scripts/download_ms_sta_artifacts.py",
            "manifest": "Docs/results/ms_sta_defense_go/ARTIFACT_MANIFEST.json",
        },
    }
    (_ROOT / "Docs" / "results" / "ms_sta_defense_go" / "EVIDENCE_INDEX.json").write_text(
        json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Refresh FINAL tracked sha for evidence index self-consistency after idx write? FINAL unchanged.
    # Update evidence index FINAL entry already correct.

    # Re-bind FINAL sha into evidence after any companion writes that shouldn't change FINAL.
    # Update registry final_results if evidence index rewrite somehow — FINAL unchanged.

    manifest = {
        "schema": "ms_sta_artifact_manifest_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "release_url": None,
        "zenodo_doi": None,
        "items": idx["local_only_artifacts"],
        "verify": "python scripts/download_ms_sta_artifacts.py --verify-only",
    }
    (_ROOT / "Docs" / "results" / "ms_sta_defense_go" / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Refresh evidence index sha for FINAL + registry after companion files settled
    idx["tracked_evidence"] = [
        {"path": "Docs/results/ms_sta_defense_go/FINAL_RESULTS.json", "sha256": _sha(final_path)},
        {"path": "Docs/results/ms_sta_e1_v2/MATRIX_AGGREGATE.json", "sha256": e1_sha},
        {"path": "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json", "sha256": e2_sha},
        {"path": "Docs/results/ms_sta_d4/D4_ACCEPTANCE_REPORT.json", "sha256": d4_sha},
        {"path": "Docs/results/ms_sta_e1_policy/E1_POLICY.json", "sha256": policy_sha},
        {"path": "data/processed/ms_sta_registry.json", "sha256": _sha(reg_path)},
    ]
    # Also bind evidence index itself after rewrite — compute after write
    idx_path = _ROOT / "Docs" / "results" / "ms_sta_defense_go" / "EVIDENCE_INDEX.json"
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "e1_v2": e1_sha,
                "e2": e2_sha,
                "d4": d4_sha,
                "policy": policy_sha,
                "final": _sha(final_path),
                "registry": _sha(reg_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
