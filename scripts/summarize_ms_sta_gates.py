"""Aggregate active MS-STA status with fail-closed evidence verification.

The active registry pointer is authoritative. V6 DEFENSE_GO is returned only
when registry, checkpoint, final results, one-shot completion, and D4 hashes
match. V5 DEFENSE_GO remains rollback/provenance-only when not active.
Historical campaign trees remain provenance-only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_GATE3_SHA = "26d866c33676f921198c9ffc61710ff1c4671f6debe782c81c828124977cfe11"


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _resolve_active_registry() -> tuple[Path | None, dict, str]:
    """Return (path, reg, source) where source is active|env|none."""
    import os

    env = (os.environ.get("MS_STA_REGISTRY_PATH") or "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = _ROOT / p
        if p.is_file():
            return p, _load(p) or {}, "env"

    ptr_path = _ROOT / "data" / "processed" / "ms_sta_registry_active.json"
    if ptr_path.is_file():
        ptr = _load(ptr_path) or {}
        rel = ptr.get("registry_path")
        if not rel:
            return None, {}, "active_pointer_invalid"
        p = _ROOT / str(rel)
        if not p.is_file():
            return None, {}, "active_registry_missing"
        expect = str(ptr.get("registry_sha256") or "")
        got = _sha(p)
        if expect and got != expect:
            return None, {}, "active_registry_sha_mismatch"
        return p, _load(p) or {}, "active"

    return None, {}, "none"


def _write_status(status: dict) -> Path:
    out = _ROOT / "logs" / "ms_sta_gates" / "STATUS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    legacy = _ROOT / "logs" / "option_b_gates" / "STATUS.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def _status_validation_no_go(
    *,
    reg_path: Path | None,
    reg: dict,
    source: str,
    contradictions: list[str],
) -> dict:
    # Prefer v4 closeout when present; keep v3 paths as historical fallback only.
    closeout_v4 = _ROOT / "Docs" / "results" / "ms_sta_retrain_v4" / "CONFIRM_VALIDATION_REPORT.json"
    metrics_v4 = _ROOT / "Docs" / "results" / "ms_sta_retrain_v4" / "VALIDATION_METRICS.json"
    marker_v4 = _ROOT / "Docs" / "results" / "ms_sta_retrain_v4" / "VALIDATION_NO_GO.json"
    closeout_v3 = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3" / "CONFIRM_VALIDATION_REPORT.json"
    metrics_v3 = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3" / "VALIDATION_METRICS.json"
    marker_v3 = _ROOT / "Docs" / "results" / "ms_sta_retrain_v3" / "VALIDATION_NO_GO.json"
    use_v4 = closeout_v4.is_file() and marker_v4.is_file() and (
        reg.get("retrain_v4") is True or "retrain_v4" in str(reg_path or "")
    )
    closeout = closeout_v4 if use_v4 else closeout_v3
    metrics = metrics_v4 if use_v4 else metrics_v3
    marker = marker_v4 if use_v4 else marker_v3
    report = _load(closeout) or {}
    metrics_doc = _load(metrics) or {}
    top1 = None
    if use_v4:
        canon = metrics_doc.get("canonical_reference") or {}
        top1 = canon.get("top1_mean_pct")
        if top1 is None:
            top1 = (reg.get("current_best_metrics") or {}).get("top1_mean_pct")
        arch_note = "B4R did not pass registered outperform gates vs B3 (v4)"
        eff_note = "v4e not opened; efficiency deferred until accuracy PASS"
    else:
        variants = report.get("variants") or {}
        b3 = variants.get("b3") or {}
        top1 = b3.get("top1_mean")
        arch_note = "B4 did not beat B3 on all confirmation seeds (v3 historical)"
        eff_note = "B4-E mean Top-1 regression exceeds -1.0 pp (v3 historical)"
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "VALIDATION_NO_GO",
        "defense_ready": False,
        "demo_allowed": False,
        "campaign_status": "VALIDATION_NO_GO",
        "registry_source": source,
        "registry_path": None if reg_path is None else reg_path.relative_to(_ROOT).as_posix(),
        "contradictions": contradictions,
        "current_best_variant": reg.get("current_best_variant") or "b3",
        "current_best_top1_mean_pct": top1,
        "closeout_campaign": "retrain_v4" if use_v4 else "retrain_v3",
        "evidence_hashes": {
            "confirm_validation_report": _sha(closeout),
            "validation_metrics": _sha(metrics),
            "validation_no_go_marker": _sha(marker),
            "active_registry": _sha(reg_path) if reg_path else None,
        },
        "gates": {
            "architecture": {
                "pass": False,
                "note": arch_note,
            },
            "efficiency_accuracy": {
                "pass": False,
                "note": eff_note,
            },
            "defense": {"pass": False},
            "demo": {"pass": False},
        },
        "historical_not_primary": {
            "registry": "data/processed/ms_sta_registry.json",
            "defense_go": "Docs/results/ms_sta_defense_go/",
            "retrain_v3": "Docs/results/ms_sta_retrain_v3/",
            "note": "Raw historical evidence retained; not live verdict",
        },
        "registry_status": reg.get("status"),
        "active_variant": reg.get("active_variant"),
        "checkpoint": reg.get("checkpoint"),
    }


def _status_defense_go_v6(
    *, reg_path: Path, reg: dict, source: str, contradictions: list[str]
) -> dict:
    final_path = _ROOT / str(reg.get("final_results_rel") or "")
    d4_path = _ROOT / "Docs/results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json"
    completion_path = _ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_COMPLETION.json"
    checkpoint_path = _ROOT / str(reg.get("checkpoint") or "")

    expected = {
        "schema": "ms_sta_registry_v6_defense_go_v1",
        "status": "GO",
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "d4_pass": True,
        "evaluation_locked": True,
        "ms_sta_v6": True,
        "ms_sta_v5": False,
        "active_variant": "b4_v6_ctr",
        "primary_seed": 42,
        "e4_ready": False,
    }
    for key, value in expected.items():
        if reg.get(key) != value:
            contradictions.append(f"registry {key}={reg.get(key)!r} expected {value!r}")

    hashes = {
        "active_registry": _sha(reg_path),
        "final_results": _sha(final_path),
        "d4_report": _sha(d4_path),
        "test_completion": _sha(completion_path),
        "checkpoint": _sha(checkpoint_path),
    }
    hash_contract = {
        "final_results": str(reg.get("final_results_sha256") or ""),
        "d4_report": str(reg.get("d4_report_sha256") or ""),
        "test_completion": str(reg.get("test_completion_sha256") or ""),
        "checkpoint": str(reg.get("checkpoint_sha256") or ""),
    }
    for key, expected_sha in hash_contract.items():
        if not expected_sha or hashes[key] != expected_sha:
            contradictions.append(
                f"{key} sha mismatch: got={hashes[key]!r} expected={expected_sha!r}"
            )

    final = _load(final_path) or {}
    d4 = _load(d4_path) or {}
    if not (
        final.get("defense_status") == "GO"
        and final.get("defense_ready") is True
        and final.get("proposed_final_model") == "b4_v6_ctr"
        and final.get("primary_seed") == 42
        and final.get("checkpoint_sha256") == reg.get("checkpoint_sha256")
        and final.get("test_kind") == "retrospective_signer_disjoint_confirmation"
        and final.get("pristine_blind_claim") is False
        and final.get("e4_status") == "E4_ACCURACY_NO_GO"
        and (final.get("test") or {}).get("n") == 6165
    ):
        contradictions.append("FINAL_RESULTS identity/claim contract mismatch")
    if not (
        d4.get("status") == "PASS"
        and d4.get("d4_pass") is True
        and d4.get("promotion_allowed") is True
        and d4.get("proposed_final_model") == "b4_v6_ctr"
        and d4.get("checkpoint_sha256") == reg.get("checkpoint_sha256")
        and d4.get("n_scenarios") == d4.get("n_pass") == 15
        and d4.get("e4_ready") is False
    ):
        contradictions.append("D4 identity/acceptance contract mismatch")

    passed = not contradictions
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "DEFENSE_GO" if passed else "NO-GO",
        "defense_ready": passed,
        "demo_allowed": passed,
        "campaign_status": "DEFENSE_GO" if passed else "NO-GO",
        "registry_source": source,
        "registry_path": reg_path.relative_to(_ROOT).as_posix(),
        "contradictions": contradictions,
        "current_best_variant": "b4_v6_ctr",
        "primary_seed": 42,
        "test_kind": "retrospective_signer_disjoint_confirmation",
        "pristine_blind_claim": False,
        "e4_ready": False,
        "e4_status": "E4_ACCURACY_NO_GO",
        "evidence_hashes": hashes,
        "gates": {
            "D0": {"pass": passed},
            "M0_validation": {"pass": passed},
            "one_shot_test": {"pass": passed},
            "D4": {"pass": passed},
            "defense": {"pass": passed},
            "demo": {"pass": passed},
            "e4_efficiency": {"pass": False, "note": "E4_ACCURACY_NO_GO appendix-only"},
        },
        "registry_status": reg.get("status"),
        "active_variant": reg.get("active_variant"),
        "checkpoint": reg.get("checkpoint"),
    }


def _status_defense_go_v5(
    *, reg_path: Path, reg: dict, source: str, contradictions: list[str]
) -> dict:
    final_path = _ROOT / str(reg.get("final_results_rel") or "")
    d4_path = _ROOT / "Docs/results/ms_sta_v5_d4/D4_ACCEPTANCE_REPORT.json"
    completion_path = _ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_COMPLETION.json"
    checkpoint_path = _ROOT / str(reg.get("checkpoint") or "")

    expected = {
        "schema": "ms_sta_registry_v5_defense_go_v1",
        "status": "GO",
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "d4_pass": True,
        "evaluation_locked": True,
        "ms_sta_v5": True,
        "active_variant": "b4_star_fixed",
        "primary_seed": 42,
    }
    for key, value in expected.items():
        if reg.get(key) != value:
            contradictions.append(f"registry {key}={reg.get(key)!r} expected {value!r}")

    hashes = {
        "active_registry": _sha(reg_path),
        "final_results": _sha(final_path),
        "d4_report": _sha(d4_path),
        "test_completion": _sha(completion_path),
        "checkpoint": _sha(checkpoint_path),
    }
    hash_contract = {
        "final_results": str(reg.get("final_results_sha256") or ""),
        "d4_report": str(reg.get("d4_report_sha256") or ""),
        "test_completion": str(reg.get("test_completion_sha256") or ""),
        "checkpoint": str(reg.get("checkpoint_sha256") or ""),
    }
    for key, expected_sha in hash_contract.items():
        if not expected_sha or hashes[key] != expected_sha:
            contradictions.append(
                f"{key} sha mismatch: got={hashes[key]!r} expected={expected_sha!r}"
            )

    final = _load(final_path) or {}
    d4 = _load(d4_path) or {}
    if not (
        final.get("defense_status") == "GO"
        and final.get("defense_ready") is True
        and final.get("proposed_final_model") == "b4_star_fixed"
        and final.get("primary_seed") == 42
        and final.get("checkpoint_sha256") == reg.get("checkpoint_sha256")
        and final.get("test_kind") == "retrospective_signer_disjoint_confirmation"
        and final.get("pristine_blind_claim") is False
        and (final.get("test") or {}).get("n") == 6165
    ):
        contradictions.append("FINAL_RESULTS identity/claim contract mismatch")
    if not (
        d4.get("status") == "PASS"
        and d4.get("d4_pass") is True
        and d4.get("promotion_allowed") is True
        and d4.get("proposed_final_model") == "b4_star_fixed"
        and d4.get("checkpoint_sha256") == reg.get("checkpoint_sha256")
        and d4.get("n_scenarios") == d4.get("n_pass") == 11
    ):
        contradictions.append("D4 identity/acceptance contract mismatch")

    passed = not contradictions
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "DEFENSE_GO" if passed else "NO-GO",
        "defense_ready": passed,
        "demo_allowed": passed,
        "campaign_status": "DEFENSE_GO" if passed else "NO-GO",
        "registry_source": source,
        "registry_path": reg_path.relative_to(_ROOT).as_posix(),
        "contradictions": contradictions,
        "current_best_variant": "b4_star_fixed",
        "primary_seed": 42,
        "test_kind": "retrospective_signer_disjoint_confirmation",
        "pristine_blind_claim": False,
        "evidence_hashes": hashes,
        "gates": {
            "D0": {"pass": passed},
            "M0_validation": {"pass": passed},
            "one_shot_test": {"pass": passed},
            "D4": {"pass": passed},
            "defense": {"pass": passed},
            "demo": {"pass": passed},
        },
        "registry_status": reg.get("status"),
        "active_variant": reg.get("active_variant"),
        "checkpoint": reg.get("checkpoint"),
    }

def main() -> int:
    contradictions: list[str] = []
    reg_path, reg, source = _resolve_active_registry()

    status_token = str(reg.get("campaign_status") or reg.get("status") or "").upper()
    if source in ("active", "env") and reg.get("ms_sta_v6") is True:
        status = _status_defense_go_v6(
            reg_path=reg_path,
            reg=reg,
            source=source,
            contradictions=contradictions,
        )
        out = _write_status(status)
        print(
            json.dumps(
                {
                    "verdict": status["verdict"],
                    "registry_source": source,
                    "defense_ready": status["defense_ready"],
                    "current_best_variant": status["current_best_variant"],
                    "e4_ready": status.get("e4_ready"),
                    "contradictions": contradictions,
                    "gates": {k: v.get("pass") for k, v in status["gates"].items()},
                },
                indent=2,
            )
        )
        print(f"STATUS -> {out}")
        return 0 if status["verdict"] == "DEFENSE_GO" else 2
    if source in ("active", "env") and reg.get("ms_sta_v5") is True:
        status = _status_defense_go_v5(
            reg_path=reg_path,
            reg=reg,
            source=source,
            contradictions=contradictions,
        )
        out = _write_status(status)
        print(
            json.dumps(
                {
                    "verdict": status["verdict"],
                    "registry_source": source,
                    "defense_ready": status["defense_ready"],
                    "current_best_variant": status["current_best_variant"],
                    "contradictions": contradictions,
                    "gates": {k: v.get("pass") for k, v in status["gates"].items()},
                },
                indent=2,
            )
        )
        print(f"STATUS -> {out}")
        return 0 if status["verdict"] == "DEFENSE_GO" else 2
    if source in ("active", "env") and (
        status_token == "VALIDATION_NO_GO"
        or reg.get("demo_allowed") is False
        or (
            reg.get("defense_ready") is False
            and (reg.get("retrain_v4") is True or reg.get("retrain_v3") is True)
        )
    ):
        if status_token != "VALIDATION_NO_GO":
            contradictions.append(
                f"active registry status={status_token!r} expected VALIDATION_NO_GO"
            )
        status = _status_validation_no_go(
            reg_path=reg_path,
            reg=reg,
            source=source,
            contradictions=contradictions,
        )
        out = _write_status(status)
        print(
            json.dumps(
                {
                    "verdict": status["verdict"],
                    "registry_source": source,
                    "defense_ready": False,
                    "current_best_variant": status.get("current_best_variant"),
                    "contradictions": contradictions,
                    "gates": {k: v.get("pass") for k, v in status["gates"].items()},
                },
                indent=2,
            )
        )
        print(f"STATUS -> {out}")
        return 2

    if source.startswith("active_"):
        contradictions.append(f"active registry resolution failed: {source}")
        status = _status_validation_no_go(
            reg_path=None, reg={}, source=source, contradictions=contradictions
        )
        status["verdict"] = "NO-GO"
        out = _write_status(status)
        print(json.dumps({"verdict": "NO-GO", "contradictions": contradictions}, indent=2))
        print(f"STATUS -> {out}")
        return 2

    # No active cutover: refuse to promote historical GO as live SoT.
    hist = _ROOT / "data" / "processed" / "ms_sta_registry.json"
    contradictions.append(
        "ms_sta_registry_active.json missing — refuse historical GO as live verdict "
        f"(historical file present={hist.is_file()})"
    )
    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "NO-GO",
        "defense_ready": False,
        "demo_allowed": False,
        "registry_source": source,
        "contradictions": contradictions,
        "historical_not_primary": {
            "registry": "data/processed/ms_sta_registry.json",
            "note": "Use active pointer for live status; historical GO is provenance only",
        },
        "gates": {
            "D0": {"pass": False, "errors": ["active_registry_required"]},
            "defense": {"pass": False},
        },
        "gate3_checksum_expected": _GATE3_SHA,
    }
    out = _write_status(status)
    print(json.dumps({"verdict": "NO-GO", "contradictions": contradictions}, indent=2))
    print(f"STATUS -> {out}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
