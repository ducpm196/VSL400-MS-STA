#!/usr/bin/env python3
"""Write E4-R3 feasibility provenance companion (does not mutate the report)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.training.retrain_v4 import executable_tree_sha256  # noqa: E402

RESEARCH_COMMIT = "4709a9453be963584867b037cb83dde0df2e54ec"
MERGE_COMMIT = "30919fc13071d1d9d7bb41cf7cbbce7f9138c1c6"
FEAS_REL = "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json"
FEAS_SHA_EXPECTED = "5eb0b2f5d97ac6a95c1809563f6976448a3d30880901746c1231bab89e3a649e"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, obj: dict) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    feas = _ROOT / FEAS_REL
    feas_sha = sha_file(feas)
    if feas_sha != FEAS_SHA_EXPECTED:
        raise SystemExit(f"feasibility_sha_mismatch:{feas_sha}")
    feas_obj = json.loads(feas.read_bytes().decode("utf-8"))

    pol = _ROOT / "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_POLICY.json"
    pol_bytes = pol.read_bytes()
    if b"\r\n" in pol_bytes:
        raise SystemExit("policy_must_be_lf_on_disk")
    pol_sha_lf = hashlib.sha256(pol_bytes).hexdigest()
    pol_sha_crlf = hashlib.sha256(pol_bytes.replace(b"\n", b"\r\n")).hexdigest()
    embedded_pol = feas_obj.get("policy_sha256")
    if embedded_pol != pol_sha_crlf:
        raise SystemExit(f"policy_embedded_sha_not_crlf_of_disk:{embedded_pol}!={pol_sha_crlf}")

    pre_path = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_PREFLIGHT.json"
    pre_bytes = pre_path.read_bytes()
    if b"\r\n" in pre_bytes:
        raise SystemExit("preflight_must_be_lf_on_disk")
    pre_sha_lf = hashlib.sha256(pre_bytes).hexdigest()
    pre_sha_crlf = hashlib.sha256(pre_bytes.replace(b"\n", b"\r\n")).hexdigest()
    pre = json.loads(pre_bytes.decode("utf-8"))

    nogo = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json"
    close = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json"
    ckpt = _ROOT / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"

    for rev in (RESEARCH_COMMIT, MERGE_COMMIT):
        subprocess.check_call(["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=_ROOT)

    tree_now = executable_tree_sha256(_ROOT)
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT, text=True).strip()
    )

    speedups = {k: float(v["speedup_vs_b4"]) for k, v in feas_obj["arms"].items()}
    med_ms = {
        "B4": float(feas_obj["b4"]["median_sec"]) * 1000.0,
        **{k: float(v["median_sec"]) * 1000.0 for k, v in feas_obj["arms"].items()},
    }

    companion = {
        "schema": "ms_sta_v6_e4_r3_feasibility_provenance_companion_v1",
        "campaign": "E4-R3",
        "role": "post_hoc_bind_of_immutable_feasibility_report",
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parent_report_rel": FEAS_REL,
        "parent_report_sha256": feas_sha,
        "parent_report_mutated": False,
        "canonical_clean_execution_identity": False,
        "canonical_clean_execution_identity_note": (
            "Original E4_R3_FEASIBILITY.json did not embed git revision / executable-tree. "
            "This companion binds post-hoc commit SHAs and a packaging-time tree hash; "
            "it must not be cited as hermetic clean-run identity for the microbench itself."
        ),
        "line_ending_bind": {
            "note": (
                "policy_sha256 / historical EVIDENCE_INDEX preflight hashes recorded at "
                "Windows CRLF write-time. On-disk tracked files are LF (-text). "
                "CRLF(LF_bytes) reconstitutes the embedded policy hash."
            ),
            "policy_sha256_lf_on_disk": pol_sha_lf,
            "policy_sha256_crlf_matches_feasibility_embed": pol_sha_crlf,
            "preflight_sha256_lf_on_disk": pre_sha_lf,
            "preflight_sha256_crlf": pre_sha_crlf,
        },
        "commits": {
            "research_closeout": RESEARCH_COMMIT,
            "main_nogo_closeout_merge": MERGE_COMMIT,
            "merge_kind": "E4-R3_NO-GO_closeout_merge",
            "not_d4_final": True,
        },
        "bindings": {
            "policy_rel": "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_POLICY.json",
            "policy_sha256_lf": pol_sha_lf,
            "policy_sha256_embedded_in_feasibility": embedded_pol,
            "preflight_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_PREFLIGHT.json",
            "preflight_sha256_lf": pre_sha_lf,
            "feasibility_no_go_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json",
            "feasibility_no_go_sha256": sha_file(nogo),
            "closeout_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json",
            "closeout_sha256": sha_file(close),
            "teacher_checkpoint_rel": "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
            "teacher_checkpoint_sha256": sha_file(ckpt),
            "split_manifest_sha256": pre["bindings"]["split_manifest"]["sha256"],
            "eligible_manifest_sha256": pre["bindings"]["eligible_manifest"]["sha256"],
            "registry_sha256": pre["bindings"]["registry"]["sha256"],
            "pip_freeze_sha256": pre.get("pip_freeze_sha256"),
            "preflight_executable_tree_sha256": pre.get("executable_tree_sha256"),
            "preflight_git_revision": pre.get("git_revision"),
        },
        "environment_from_preflight": {
            "gpu": (pre.get("gpu") or {}).get("name"),
            "torch": pre.get("torch"),
            "cuda_version": pre.get("cuda_version"),
            "cuda_available": pre.get("cuda_available"),
        },
        "protocol_from_report": {
            "tier": feas_obj.get("tier"),
            "dtype": feas_obj.get("dtype"),
            "batch_size": feas_obj.get("batch_size"),
            "warmup": feas_obj.get("warmup"),
            "trials_per_model_approx": feas_obj.get("trials_per_model_approx"),
            "interleaved": feas_obj.get("interleaved"),
            "randomized": feas_obj.get("randomized"),
            "speedup_min_required": feas_obj.get("speedup_min_required"),
            "device": feas_obj.get("device"),
        },
        "metrics_bound": {
            "median_ms": med_ms,
            "speedup_vs_b4": speedups,
            "params": {
                "B4": int(feas_obj["b4"]["params"]),
                **{k: int(v["params"]) for k, v in feas_obj["arms"].items()},
            },
            "display_round_ms": {"B4": 12.47, "A": 12.24, "B": 13.23, "C": 13.01},
            "display_speedup_pct": {"A": 1.9, "B": -5.7, "C": -4.1},
        },
        "packaging_time": {
            "executable_tree_sha256": tree_now,
            "git_dirty_at_packaging": dirty,
            "note": "Tree hash at companion write; not proof of microbench-time identity.",
        },
        "verdict": "E4_R3_FEASIBILITY_NO_GO",
        "e4_ready": False,
        "b4_remains_primary": True,
    }
    out = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_PROVENANCE_COMPANION.json"
    sha = atomic(out, companion)

    idx_path = _ROOT / "Docs/results/ms_sta_v6_e4_r3/EVIDENCE_INDEX.json"
    idx = {
        "schema": "ms_sta_v6_e4_r3_evidence_index_v1",
        "campaign": "E4-R3",
        "status": "CLOSED_FEASIBILITY_NO_GO",
        "line_ending_policy": "tracked_json_is_lf_via_gitattributes_-text",
        "provenance_companion_note": (
            "Companion binds report SHA + commits; not a substitute for clean-run identity. "
            "Embedded policy_sha in feasibility matches CRLF(LF) serialization."
        ),
        "entries": [
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_PREFLIGHT.json",
                "sha256": pre_sha_lf,
                "sha256_crlf_historical": pre_sha_crlf,
                "role": "phase0_preflight",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_POLICY.json",
                "sha256": pol_sha_lf,
                "sha256_crlf_matches_feasibility_embed": pol_sha_crlf,
                "role": "immutable_policy",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_TRAIN_POLICY.json",
                "sha256": sha_file(
                    _ROOT / "Docs/results/ms_sta_v6_e4_r3_policy/E4_R3_TRAIN_POLICY.json"
                ),
                "role": "train_policy_cancelled",
            },
            {
                "rel": FEAS_REL,
                "sha256": feas_sha,
                "role": "feasibility_microbench",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json",
                "sha256": sha_file(nogo),
                "role": "feasibility_no_go",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json",
                "sha256": sha_file(close),
                "role": "closeout",
            },
            {
                "rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_PROVENANCE_COMPANION.json",
                "sha256": sha,
                "role": "feasibility_provenance_companion",
            },
        ],
    }
    atomic(idx_path, idx)
    print(json.dumps({"companion_sha256": sha, "feasibility_sha256": feas_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
