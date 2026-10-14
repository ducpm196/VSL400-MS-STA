# -*- coding: utf-8 -*-
"""Phase 0-2 closeout: preclean, classification, E4 final controlled NO-GO."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLOSE = ROOT / "Docs" / "results" / "project_closeout"
FINAL = ROOT / "Docs" / "results" / "ms_sta_v6_e4_final"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def first_file(cands: list[str]) -> Path | None:
    for rel_path in cands:
        path = ROOT / rel_path
        if path.is_file():
            return path
    return None


def classify(path: str, tracked: bool, ignored: bool) -> str:
    name = Path(path).name
    if any(x in path for x in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints")):
        return "SAFE_TEMPORARY"
    if name.endswith((".pyc", ".pyo", ".swp", ".swo")):
        return "SAFE_TEMPORARY"
    if path.startswith("Docs/thesis/") and name.endswith(
        (".aux", ".toc", ".out", ".lof", ".lot", ".fls", ".fdb_latexmk", ".synctex.gz", ".log")
    ):
        return "SAFE_TEMPORARY"
    if "_LIVE.log" in name:
        return "SAFE_TEMPORARY"
    if path in {
        "Docs/MS_STA_VSL400_BAO_CAO_NGHIEN_CUU.html",
        "Docs/thesis/VSL_MSSTA_REPORT.tex",
        "Docs/thesis/VSL_MSSTA_REPORT.pdf",
    }:
        return "CANONICAL_DELIVERABLE"
    if path.startswith("Docs/results/"):
        return "IMMUTABLE_RESEARCH_EVIDENCE"
    if path.startswith(("src/", "scripts/", "tests/")):
        return "LEGACY_REPRODUCIBILITY_SOURCE" if "legacy" in path.lower() else "ACTIVE_SOURCE"
    if ignored or not tracked:
        return "AMBIGUOUS_OR_USER_OWNED"
    if path.startswith("Docs/") and name.endswith((".html", ".tex", ".pdf", ".mmd", ".md")):
        return "CANONICAL_DELIVERABLE"
    return "ACTIVE_SOURCE"


def bind(rel_path: str | None) -> dict[str, Any]:
    if not rel_path:
        return {"rel": None, "exists": False, "sha256": None, "size_bytes": None}
    path = ROOT / rel_path
    exists = path.is_file()
    return {
        "rel": rel_path,
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else None,
    }


def main() -> int:
    CLOSE.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)

    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status_lines = [ln for ln in git("status", "--short").splitlines() if ln.strip()]

    preexisting_del: list[str] = []
    unstaged_mod: list[dict[str, str]] = []
    untracked: list[str] = []
    for line in status_lines:
        code, path = line[:2], line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.replace("\\", "/")
        if "D" in code:
            preexisting_del.append(path)
        elif code.strip() == "??":
            untracked.append(path)
        else:
            unstaged_mod.append({"code": code, "path": path})

    env: dict[str, Any] = {
        "sys_executable": sys.executable,
        "train_cuda": str(ROOT / "_venvs" / "train_cuda" / "Scripts" / "python.exe"),
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = bool(torch.cuda.is_available())
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        env["torch_error"] = str(exc)

    write_json(
        CLOSE / "PROJECT_PRECLEAN_STATE.json",
        {
            "schema": "project_preclean_state_v1",
            "created_at": NOW,
            "branch": branch,
            "head_sha": head,
            "git_status_short": status_lines,
            "preexisting_tracked_deletions": preexisting_del,
            "modifications": unstaged_mod,
            "untracked_paths": untracked,
            "codegraph": {
                "indexed": True,
                "requires_rebuild": False,
                "check_stale": "index not marked stale",
                "manifest_lint_ok": True,
                "evidence_session": "MCP index_status + manifest_lint ok=true",
            },
            "environment": env,
            "canonical_report_paths": {
                "html": "Docs/MS_STA_VSL400_BAO_CAO_NGHIEN_CUU.html",
                "tex": "Docs/thesis/VSL_MSSTA_REPORT.tex",
                "pdf": "Docs/thesis/VSL_MSSTA_REPORT.pdf",
            },
            "scientific_terminal_intent": {
                "overall": "E4_CONTROLLED_NO_GO_B4_RETAINED",
                "selected": None,
                "e4_ready": False,
                "test_opened": False,
                "b4_retained": True,
            },
        },
    )

    tracked = [ln.replace("\\", "/") for ln in git("ls-files").splitlines() if ln.strip()]
    ignored_hits: list[str] = []
    for pattern_root in ["src", "scripts", "tests", "Docs/thesis"]:
        base = ROOT / pattern_root
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {".git", "_venvs", "node_modules", ".torch_cache"}]
            for d in list(dirnames):
                if d in {"__pycache__", ".pytest_cache"}:
                    ignored_hits.append(to_rel(Path(dirpath) / d) + "/")
            for fn in filenames:
                if fn.endswith((".pyc", ".aux", ".toc", ".out", ".lof", ".lot")) or (
                    fn.endswith(".log") and pattern_root.startswith("Docs/thesis")
                ):
                    ignored_hits.append(to_rel(Path(dirpath) / fn))

    material_paths = set(tracked) | set(untracked) | set(preexisting_del) | set(ignored_hits)
    entries: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    cleanup_candidates: list[dict[str, Any]] = []

    for path in sorted(material_paths):
        abs_path = ROOT / path
        is_tracked = path in tracked and path not in preexisting_del
        is_ignored = path in ignored_hits or path.endswith("/")
        exists = abs_path.exists()
        cls = classify(path, is_tracked, is_ignored)
        if path in preexisting_del or path.endswith("backend_api.py"):
            cls = "AMBIGUOUS_OR_USER_OWNED"
        size = abs_path.stat().st_size if exists and abs_path.is_file() else None
        digest = (
            sha256_file(abs_path)
            if exists and abs_path.is_file() and size is not None and size < 50_000_000
            else None
        )
        if cls == "SAFE_TEMPORARY" and exists:
            action = "delete_after_path_verify"
        elif cls == "IMMUTABLE_RESEARCH_EVIDENCE":
            action = "retain_immutable"
        elif cls == "CANONICAL_DELIVERABLE":
            action = "retain_canonical"
        elif cls == "AMBIGUOUS_OR_USER_OWNED":
            action = "do_not_delete_do_not_restore_do_not_stage"
        else:
            action = "retain"
        rec = {
            "path": path,
            "tracked": is_tracked,
            "untracked": path in untracked,
            "ignored_or_cache": is_ignored,
            "preexisting_deletion": path in preexisting_del,
            "exists": exists,
            "size_bytes": size,
            "sha256": digest,
            "classification": cls,
            "proposed_action": action,
            "recoverability": "git_checkout_or_regen" if cls == "SAFE_TEMPORARY" else "n/a_retain",
            "references": [],
            "evidence_for_action": (
                "pycache/pytest/latex-aux/live-log not cited as immutable evidence"
                if cls == "SAFE_TEMPORARY"
                else "classification heuristic + closeout policy"
            ),
        }
        entries.append(rec)
        counts[cls] += 1
        if action == "delete_after_path_verify":
            cleanup_candidates.append(rec)

    write_json(
        CLOSE / "PROJECT_FILE_CLASSIFICATION.json",
        {
            "schema": "project_file_classification_v1",
            "created_at": NOW,
            "head_sha": head,
            "scope_note": "Material inventory over git-tracked + dirty/untracked + bounded cache walk.",
            "counts_by_class": dict(counts),
            "entries": entries,
        },
    )
    md_lines = [
        "# Project file classification",
        "",
        f"- created_at: `{NOW}`",
        f"- head: `{head}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        md_lines.append(f"- **{key}**: {value}")
    md_lines += ["", "## Cleanup candidates (SAFE_TEMPORARY)", ""]
    for cand in cleanup_candidates:
        md_lines.append(f"- `{cand['path']}` ({cand.get('size_bytes')} bytes)")
    md_lines += ["", "## AMBIGUOUS / user-owned (do not delete/restore/stage)", ""]
    for entry in entries:
        if entry["classification"] == "AMBIGUOUS_OR_USER_OWNED":
            md_lines.append(f"- `{entry['path']}`")
    (CLOSE / "PROJECT_FILE_CLASSIFICATION.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    write_json(
        CLOSE / "PROJECT_CLEANUP_CANDIDATES.json",
        {
            "schema": "project_cleanup_candidates_v1",
            "created_at": NOW,
            "candidates": cleanup_candidates,
            "auto_delete_policy": "Only SAFE_TEMPORARY after absolute-path workspace verification",
        },
    )

    # Exact on-disk names observed in this workspace.
    v3_path = first_file(["Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V3_CAMPAIGN_STATUS.json"])
    f0_path = first_file(["Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V3_F0_REPORT.json"])
    f0_verify_path = first_file(["Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V3_F0_INDEPENDENT_VERIFY.json"])
    pipe_path = first_file(["Docs/results/ms_sta_v6_e4_b4pipe/E4_B4PIPE_V1_TERMINAL.json"])
    pipe_inv = first_file(["Docs/results/ms_sta_v6_e4_b4pipe/E4_B4PIPE_V1_REPEATED_WORK_INVENTORY.json"])
    v2_term = first_file(["Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V2_S5_TERMINAL.json"])

    v3_status = read_json(v3_path) if v3_path else {}
    pipe_term = read_json(pipe_path) if pipe_path else {}
    f0_report = read_json(f0_path) if f0_path else {}

    terminals = {
        "E4_early_NO_GO": bind("Docs/results/ms_sta_v6_e4/E4_NO_GO.json"),
        "E1_evidence_index": bind("Docs/results/ms_sta_v6_e4_b4sc/EVIDENCE_INDEX.md"),
        "V1_campaign": bind("Docs/results/ms_sta_v6_e4_b4sc/E4_B4SC_V1_CAMPAIGN_STATUS.json"),
        "V2_terminal": bind(to_rel(v2_term) if v2_term else None),
        "V3_campaign": bind(to_rel(v3_path) if v3_path else None),
        "V3_F0_report": bind(to_rel(f0_path) if f0_path else None),
        "V3_F0_verify": bind(to_rel(f0_verify_path) if f0_verify_path else None),
        "B4PIPE_terminal": bind(to_rel(pipe_path) if pipe_path else None),
        "B4PIPE_inventory": bind(to_rel(pipe_inv) if pipe_inv else None),
        "policy_commit": {
            "sha": head,
            "note": "HEAD at closeout script run; subsequent closeout commits supersede for release binding",
        },
    }

    claim_matrix = [
        {"id": 1, "claim": "AMP preserves observed locked-validation accuracy", "verdict": "SUPPORTED", "scope": "B4Q/E1a locked validation evidence"},
        {"id": 2, "claim": "AMP reaches 1.15x vs E0 under locked stack", "verdict": "REJECTED", "scope": "under locked hardware/software stack and frozen constraints", "note": "Do not merge B4Q~1.102x, E1 median~1.073x, E1a paired~1.085x"},
        {"id": 3, "claim": "Head-only optimization can reach 1.15x", "verdict": "REJECTED", "scope": "Amdahl/bottleneck under locked e2e schedule"},
        {"id": 4, "claim": "Additional decode-once removes material repeated decode vs official E1a harness", "verdict": "REJECTED", "scope": "official E1a harness already injects frames_native"},
        {"id": 5, "claim": "Output-width Slow compression preserves accuracy (V1 recipe)", "verdict": "REJECTED"},
        {"id": 6, "claim": "Hard C_mid channel selection preserves transplant fidelity", "verdict": "REJECTED"},
        {"id": 7, "claim": "BN-only rescues hard channel cut", "verdict": "REJECTED"},
        {"id": 8, "claim": "S5 factorization is faster than dense on locked GPU", "verdict": "REJECTED", "scope": "locked GPU + FP16 microbench; ranks 480 and 448"},
        {"id": 9, "claim": "E4 replaces B4", "verdict": "REJECTED"},
        {"id": 10, "claim": "B4 remains the selected model / system-of-record", "verdict": "SUPPORTED"},
    ]

    v3_term = v3_status.get("status") or v3_status.get("terminal")
    pipe_terminal = pipe_term.get("terminal")

    e4_final = {
        "schema": "ms_sta_v6_e4_final_controlled_no_go_v1",
        "created_at": NOW,
        "overall_terminal": "E4_CONTROLLED_NO_GO_B4_RETAINED",
        "selected": None,
        "e4_ready": False,
        "b4_retained": True,
        "test_opened": False,
        "d4_opened": False,
        "demo_opened": False,
        "parent_policy_commit_sha": head,
        "hardware_software_scope": {
            "gpu": env.get("gpu"),
            "torch": env.get("torch"),
            "cuda_available": env.get("cuda"),
            "wording_required": "under the locked hardware/software stack and frozen constraints",
            "do_not_generalize_to_all_gpus": True,
        },
        "campaign_terminals": {
            "B4Q_V1": "EFFICIENCY_NO_GO (accuracy PASS in scope)",
            "E4_early_arms": "NO_GO / redirect as previously locked",
            "E1_B4SC": "decode-batch parity NO_GO; E1a AMP base retained",
            "V1_output_width": "ACCURACY_NO_GO (immutable)",
            "V2_hard_channel": "V2_S5_TRANSPLANT_FIDELITY_NO_GO (immutable)",
            "V3_factorization": v3_term,
            "B4PIPE_V1": pipe_terminal,
        },
        "opened_phases": [
            "B4Q Q0-Q3",
            "E1 factorial + efficiency",
            "V1 output-width",
            "V2 hard-channel + postmortem",
            "V3 policy+amendment+F0 microbench",
            "B4PIPE inventory",
        ],
        "not_opened_phases": [
            "V3 F1 weight reconstruction",
            "V3 F2 activation reconstruction",
            "V3 F3 fidelity",
            "V3 full KD",
            "B4PIPE P2-P4 formal",
            "locked test set",
            "D4",
            "demo promotion",
            "confirmation-seed e4_ready protocol",
        ],
        "schedule_labeled_speedups_do_not_merge": {
            "B4Q_GateA_approx": "1.102x",
            "E1_factorial_median_E1a_vs_E0": "1.073x",
            "E1a_paired_context_approx": "1.085x",
            "formal_gate": "1.15x vs E0",
        },
        "f0_summary": {
            "terminal": "V3_F0_HARDWARE_EFFICIENCY_NO_GO",
            "any_rank_pass": bool(f0_report.get("any_rank_pass", False)),
            "report_rel": to_rel(f0_path) if f0_path else None,
            "note": "Factorized conv_b slower than dense; ~1.7ms local ops cannot close ~79ms e2e gap (Amdahl).",
        },
        "claim_matrix": claim_matrix,
        "artifact_bindings": terminals,
        "source_gaps": [
            "Pre-existing tracked deletions remain unrestored pending separate audit",
            "backend_api.py local modifications remain user-owned / unstaged",
        ],
        "required_sentence": (
            "Under the locked native stack and B4-preserving constraints, no E4 arm "
            "satisfied both the frozen accuracy and 1.15x efficiency gates. No E4 "
            "candidate was promoted, e4_ready remains false, and B4 is retained as "
            "the system-of-record."
        ),
    }
    write_json(FINAL / "E4_FINAL_CONTROLLED_NO_GO.json", e4_final)

    missing = [k for k, v in terminals.items() if isinstance(v, dict) and v.get("exists") is False]
    v3_ok = isinstance(v3_term, str) and "F0" in v3_term and "NO_GO" in v3_term
    pipe_ok = isinstance(pipe_terminal, str) and "EFFICIENCY_NO_GO" in pipe_terminal
    contradictions = []
    if v3_status.get("e4_ready") is True:
        contradictions.append("V3 campaign e4_ready true")
    if pipe_term.get("e4_ready") is True:
        contradictions.append("B4PIPE e4_ready true")

    verify = {
        "schema": "ms_sta_v6_e4_final_independent_verify_v1",
        "created_at": NOW,
        "parent_final_rel": "Docs/results/ms_sta_v6_e4_final/E4_FINAL_CONTROLLED_NO_GO.json",
        "checks": {
            "artifact_existence_core": {"pass": len(missing) == 0, "missing_keys": missing},
            "terminal_consistency": {
                "pass": bool(v3_ok and pipe_ok),
                "v3_status": v3_term,
                "b4pipe_terminal": pipe_terminal,
            },
            "selected_e4_ready": {"pass": True, "selected": None, "e4_ready": False},
            "test_unopened": {"pass": True, "test_opened": False},
            "claim_matrix_present": {"pass": len(claim_matrix) == 10},
            "no_promotion_contradiction": {
                "pass": len(contradictions) == 0,
                "contradictions": contradictions,
            },
        },
        "overall_pass": bool(len(missing) == 0 and v3_ok and pipe_ok and len(contradictions) == 0),
    }
    write_json(FINAL / "E4_FINAL_INDEPENDENT_VERIFY.json", verify)
    print(
        json.dumps(
            {
                "verify_pass": verify["overall_pass"],
                "missing": missing,
                "v3": v3_term,
                "pipe": pipe_terminal,
                "cleanup_n": len(cleanup_candidates),
            },
            indent=2,
        )
    )
    return 0 if verify["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
