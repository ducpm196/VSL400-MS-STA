"""VSL400 word-level inference API (isolated + gloss→LLM). Continuous_News removed."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_prep.raw_video_resolve import RawVideoResolver
from src.deployment.gloss_llm import compose_vietnamese_from_glosses
from src.load_env import load_project_env
from src.models.isolated_sign_classifier import DEFAULT_CKPT, DEFAULT_LABELS, IsolatedSignClassifier
from src.models.ms_sta_features import resolve_pooled_dir, resolve_skel_dir, resolve_temporal_dir

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_project_env()
logger = logging.getLogger("vsl400.api")
app = FastAPI(
    title="VSL400 Word-Level API",
    description=(
        "VSL400 signer-independent word-level recognition with human-assisted output. "
        "Not a complete continuous VSL translation system."
    ),
)

# Word-level only. Continuous free-gen / retrieval were removed (fail-closed).
_ALLOWED_MODES = frozenset({"isolated", "gloss_llm"})
MSSTA_CKPT_DIR_DEFAULT = _PROJECT_ROOT / "data" / "processed" / "ms_sta_e2_lock"
RGB_POOLED_DIR = resolve_pooled_dir(_PROJECT_ROOT)
RGB_TEMPORAL_DIR = resolve_temporal_dir(_PROJECT_ROOT)
SKEL_FEAT_DIR = resolve_skel_dir(_PROJECT_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
isolated_clf: IsolatedSignClassifier | None = None
_ms_sta_model = None
_ms_sta_variant: str | None = None
_ms_sta_load_mode: str | None = None  # "defense" | "candidate" | None
_ms_sta_experiment_variant: str | None = None
_ms_sta_skel_temporal_stride: int | None = None
_ms_sta_v5 = False
_ms_sta_v6 = False
_ms_sta_v5_rgb_root: Path | None = None
_ms_sta_v5_skel_root: Path | None = None
_metadata_df: pd.DataFrame | None = None
_resolver: RawVideoResolver | None = None
_video_index: dict[str, int] | None = None


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_truthy(name: str) -> bool:
    import os

    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _primary_campaign_is_no_go() -> bool:
    """True when active cutover registry is VALIDATION_NO_GO (no historical KPI surface)."""
    reg_path = _resolve_ms_sta_registry_path()
    if reg_path is None or not reg_path.is_file():
        return True
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return True
    status = str(reg.get("campaign_status") or reg.get("status") or "")
    return status == "VALIDATION_NO_GO" or reg.get("demo_allowed") is False


def _primary_campaign_is_v5() -> bool:
    """True only when the active registry is the sealed v5 Defense GO campaign."""
    reg_path = _resolve_ms_sta_registry_path()
    if reg_path is None or not reg_path.is_file():
        return False
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return (
        reg.get("ms_sta_v5") is True
        and str(reg.get("campaign_status") or "") == "DEFENSE_GO"
        and reg.get("defense_ready") is True
        and reg.get("demo_allowed") is True
    )


def _primary_campaign_is_v6() -> bool:
    """True only when the active registry is the sealed v6 Defense GO campaign."""
    reg_path = _resolve_ms_sta_registry_path()
    if reg_path is None or not reg_path.is_file():
        return False
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return (
        reg.get("ms_sta_v6") is True
        and str(reg.get("campaign_status") or "") == "DEFENSE_GO"
        and reg.get("defense_ready") is True
        and reg.get("demo_allowed") is True
    )


def _suppress_historical_kpi() -> bool:
    return (
        _primary_campaign_is_no_go()
        or _primary_campaign_is_v5()
        or _primary_campaign_is_v6()
    ) and not _env_truthy("MS_STA_ALLOW_HISTORICAL_KPI")


def _load_v3_closeout_kpi() -> dict:
    """Primary Path-B confirmation metrics after VALIDATION_NO_GO (not E1/E2).

    Prefer retrain-v4 closeout when present; v3 remains historical fallback.
    """
    metrics_v4 = (
        _PROJECT_ROOT / "Docs" / "results" / "ms_sta_retrain_v4" / "VALIDATION_METRICS.json"
    )
    report_v4 = (
        _PROJECT_ROOT
        / "Docs"
        / "results"
        / "ms_sta_retrain_v4"
        / "CONFIRM_VALIDATION_REPORT.json"
    )
    if metrics_v4.is_file() and report_v4.is_file():
        try:
            metrics = json.loads(metrics_v4.read_text(encoding="utf-8-sig"))
            report = json.loads(report_v4.read_text(encoding="utf-8-sig"))
            canon = metrics.get("canonical_reference") or {}
            proposal = metrics.get("primary_proposal") or {}
            variants = metrics.get("variants") or {}
            b3 = variants.get("b3") or {}
            b4r = variants.get("b4r") or {}
            return {
                "ready": True,
                "scope": "path_b_confirmation_validation_only",
                "campaign": "retrain_v4",
                "campaign_status": "VALIDATION_NO_GO",
                "architecture_passed": False,
                "defense_ready": False,
                "demo_allowed": False,
                "current_best_variant": "b3",
                "b3_top1_mean_pct": canon.get("top1_mean_pct", b3.get("top1_mean_pct")),
                "b3_top1_std_pct": canon.get(
                    "top1_sample_std_pct", b3.get("top1_sample_std_pct")
                ),
                "b3_top5_mean_pct": canon.get("top5_mean_pct", b3.get("top5_mean_pct")),
                "b4r_top1_mean_pct": b4r.get("top1_mean_pct"),
                "b4r_top1_std_pct": b4r.get("top1_sample_std_pct"),
                "mean_b4r_minus_b3_pp": proposal.get("mean_delta_top1_pp"),
                "claim_language": proposal.get("claim_language") or "improve_not_outperform",
                "report_sha256": _sha256_file(report_v4),
                "metrics_sha256": _sha256_file(metrics_v4),
                "forbid_historical_e1e2_primary": True,
                "historical_v3_not_current_best": True,
            }
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            pass

    report_path = (
        _PROJECT_ROOT / "Docs" / "results" / "ms_sta_retrain_v3" / "CONFIRM_VALIDATION_REPORT.json"
    )
    unavailable = {"ready": False, "scope": "retrain_v3_validation_no_go"}
    if not report_path.is_file():
        return unavailable
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        variants = report.get("variants") or {}
        b3 = variants.get("b3") or {}
        b4 = variants.get("b4") or {}
        return {
            "ready": True,
            "scope": "path_b_confirmation_validation_only",
            "campaign": "retrain_v3",
            "campaign_status": report.get("campaign_verdict") or "VALIDATION_NO_GO",
            "architecture_passed": bool(report.get("architecture_passed")),
            "defense_ready": False,
            "demo_allowed": False,
            "current_best_variant": "b3",
            "b3_top1_mean_pct": b3.get("top1_mean"),
            "b3_top1_std_pct": b3.get("top1_sample_std"),
            "b3_top5_mean_pct": b3.get("top5_mean"),
            "b4_top1_mean_pct": b4.get("top1_mean"),
            "b4_top1_std_pct": b4.get("top1_sample_std"),
            "mean_b4_minus_b3_pp": (report.get("architecture_gates") or {}).get(
                "mean_b4_minus_b3_pp"
            ),
            "report_sha256": _sha256_file(report_path),
            "forbid_historical_e1e2_primary": True,
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return unavailable


def _load_efficiency_ablation(path: Path | None = None) -> dict:
    """Load locked validation-only efficiency evidence for the demo surface."""
    unavailable = {"ready": False, "scope": "validation_only_auxiliary"}
    # Default demo surface suppresses historical KPI under VALIDATION_NO_GO.
    # Explicit path still parses locked historical reports (provenance tests).
    if path is None and _suppress_historical_kpi():
        return {
            **unavailable,
            "suppressed": True,
            "reason": "Historical efficiency KPI is not part of the active campaign",
        }
    report_path = path or (
        _PROJECT_ROOT / "Docs" / "results" / "ms_sta_efficiency" / "FINAL_REPORT.json"
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if report.get("status") != "PASS" or report.get("claim_allowed") is not True:
            return unavailable
        architecture = report["architecture"]
        theory = report["theoretical_result"]
        validation = report["validation_result"]
        benchmark = report["benchmark_result"]
        per_seed = validation["per_seed"]
        base_top1 = [float(row["base_top1"]) for row in per_seed]
        base_top5 = [float(row["base_top5"]) for row in per_seed]
        if len(base_top1) != 3 or validation["decision_checks"].get("pass") is not True:
            return unavailable
        return {
            "ready": True,
            "scope": "validation_only_auxiliary",
            "report_sha256": _sha256_file(report_path),
            "stream": architecture["stream"],
            "operation": architecture["operation"],
            "stride": int(architecture["stride"]),
            "tokens_before": int(architecture["tokens_before"]),
            "tokens_after": int(architecture["tokens_after"]),
            "score_elements_before": int(theory["before_per_head_per_sample"]),
            "score_elements_after": int(theory["after_per_head_per_sample"]),
            "score_reduction_pct": float(theory["reduction_pct"]),
            "base_top1_mean_pct": float(np.mean(base_top1)),
            "base_top5_mean_pct": float(np.mean(base_top5)),
            "stride4_top1_mean_pct": float(validation["stride4_top1_mean_pct"]),
            "stride4_top1_std_pct": float(validation["stride4_top1_std_pct"]),
            "stride4_top5_mean_pct": float(validation["stride4_top5_mean_pct"]),
            "paired_delta_top1_mean_pp": float(validation["paired_delta_top1_mean_pp"]),
            "accuracy_interpretation": validation["interpretation"],
            "latency_before_ms": float(benchmark["baseline_stride1"]["median_latency_ms"]),
            "latency_after_ms": float(benchmark["stride4"]["median_latency_ms"]),
            "latency_reduction_pct": float(
                benchmark["median_model_forward_latency_reduction_pct"]
            ),
            "peak_vram_reduction_pct": float(
                benchmark["peak_incremental_vram_reduction_pct"]
            ),
            "benchmark_device": benchmark["device"]["name"],
            "benchmark_scope": benchmark["protocol"]["scope"],
            "demo_engine_uses_stride4": False,
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return unavailable


def _load_accuracy_study(path: Path | None = None) -> dict:
    """Load the locked post-defense validation optimization study."""
    unavailable = {"ready": False, "scope": "validation_only_auxiliary"}
    if path is None and _suppress_historical_kpi():
        return {
            **unavailable,
            "suppressed": True,
            "reason": "Historical B4-only accuracy study is not part of the active campaign",
        }
    report_path = path or (
        _PROJECT_ROOT / "Docs" / "results" / "ms_sta_accuracy" / "FINAL_REPORT.json"
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        result = report["validation_result"]
        checks = result["decision_checks"]
        if (
            report.get("status") != "PASS"
            or report.get("claim_allowed") is not True
            or checks.get("pass") is not True
            or len(result["per_seed"]) != 3
        ):
            return unavailable
        return {
            "ready": True,
            "scope": "validation_only_auxiliary",
            "report_sha256": _sha256_file(report_path),
            "base_top1_mean_pct": float(
                np.mean([float(row["base_top1"]) for row in result["per_seed"]])
            ),
            "optimized_top1_mean_pct": float(result["top1"]["mean"]),
            "optimized_top1_std_pct": float(result["top1"]["std"]),
            "optimized_top5_mean_pct": float(result["top5"]["mean"]),
            "paired_delta_top1_mean_pp": float(result["paired_delta_top1_pp"]["mean"]),
            "paired_delta_top1_std_pp": float(result["paired_delta_top1_pp"]["std"]),
            "seeds": [int(seed) for seed in result["seeds"]],
            "learning_rate_before": float(report["change"]["learning_rate_before"]),
            "learning_rate_after": float(report["change"]["learning_rate_after"]),
            "demo_engine_uses_optimized_checkpoint": False,
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return unavailable


def _resolve_ms_sta_ckpt_dir(reg: dict) -> Path:
    rel = reg.get("ckpts_dir")
    if rel:
        return _PROJECT_ROOT / str(rel)
    return MSSTA_CKPT_DIR_DEFAULT


def _resolve_ms_sta_registry_path() -> Path | None:
    """Fail-closed registry resolution: env → v3 cutover pointer → historical."""
    import os

    env = (os.environ.get("MS_STA_REGISTRY_PATH") or "").strip()
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p if p.is_file() else None

    active_ptr = _PROJECT_ROOT / "data" / "processed" / "ms_sta_registry_active.json"
    if active_ptr.is_file():
        try:
            ptr = json.loads(active_ptr.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("MS-STA active registry pointer unreadable: %s", exc)
            return None
        rel = ptr.get("registry_path")
        if not rel:
            logger.warning("MS-STA active pointer missing registry_path")
            return None
        p = _PROJECT_ROOT / str(rel)
        if not p.is_file():
            logger.warning("MS-STA active registry missing: %s", p)
            return None
        expect = str(ptr.get("registry_sha256") or "")
        if expect and _sha256_file(p) != expect:
            logger.warning("MS-STA active registry SHA mismatch — refuse cutover load")
            return None
        return p

    # Historical E1/E2 registry remains on disk for provenance only.
    # Never auto-load it for demo/defense after retrain-v3 VALIDATION_NO_GO cutover.
    hist = _PROJECT_ROOT / "data" / "processed" / "ms_sta_registry.json"
    if hist.is_file():
        logger.warning(
            "MS-STA historical registry present but fallback disabled "
            "(require ms_sta_registry_active.json or MS_STA_REGISTRY_PATH)"
        )
    return None


def _ms_sta_v5_identity_errors(reg: dict, payload: dict) -> list[str]:
    errors: list[str] = []
    checks = {
        "schema": reg.get("schema") == "ms_sta_registry_v5_defense_go_v1",
        "campaign_status": reg.get("campaign_status") == "DEFENSE_GO",
        "defense_ready": reg.get("defense_ready") is True,
        "demo_allowed": reg.get("demo_allowed") is True,
        "d4_pass": reg.get("d4_pass") is True,
        "evaluation_locked": reg.get("evaluation_locked") is True,
        "variant": reg.get("active_variant") == payload.get("variant") == "b4_star_fixed",
        "seed": int(payload.get("seed", -1)) == int(reg.get("primary_seed", -2)) == 42,
        "b4_e_forbidden": "b4_star_e_stride4" in set(reg.get("forbidden_variants_on_demo") or []),
    }
    for name, ok in checks.items():
        if not ok:
            errors.append(f"v5_identity_{name}")
    return errors


def _ms_sta_v6_identity_errors(reg: dict, payload: dict) -> list[str]:
    errors: list[str] = []
    checks = {
        "schema": reg.get("schema") == "ms_sta_registry_v6_defense_go_v1",
        "campaign_status": reg.get("campaign_status") == "DEFENSE_GO",
        "defense_ready": reg.get("defense_ready") is True,
        "demo_allowed": reg.get("demo_allowed") is True,
        "d4_pass": reg.get("d4_pass") is True,
        "evaluation_locked": reg.get("evaluation_locked") is True,
        "variant": reg.get("active_variant") == payload.get("variant") == "b4_v6_ctr",
        "seed": int(payload.get("seed", -1)) == int(reg.get("primary_seed", -2)) == 42,
        "e4_forbidden": all(
            v in set(reg.get("forbidden_variants_on_demo") or [])
            for v in ("e4_w192", "e4_w192_t1", "e4_w160_t1")
        ),
        "e4_not_ready": reg.get("e4_ready") is False,
        "b4_ready": reg.get("b4_ready") is True,
    }
    for name, ok in checks.items():
        if not ok:
            errors.append(f"v6_identity_{name}")
    return errors


def _load_ms_sta_from_registry(reg: dict) -> tuple[object | None, str | None, str | None]:
    """Load the exact active defense/candidate model or refuse on identity drift.

    V5 requires the Defense GO registry, exact B4* checkpoint and sealed D0 feature
    roots. Historical E1/E2 and retrain-v3 paths retain their original contracts.
    """
    global _ms_sta_experiment_variant, _ms_sta_skel_temporal_stride
    global _ms_sta_v5, _ms_sta_v6, _ms_sta_v5_rgb_root, _ms_sta_v5_skel_root
    ckpt_dir = _resolve_ms_sta_ckpt_dir(reg)
    ckpt_name = reg.get("checkpoint")
    if not ckpt_name:
        return None, None, None
    ckpt_p = ckpt_dir / Path(str(ckpt_name)).name
    if not ckpt_p.is_file():
        # Allow absolute/relative checkpoint paths written by v3 promote.
        alt = _PROJECT_ROOT / str(ckpt_name)
        if alt.is_file():
            ckpt_p = alt
        else:
            logger.warning("MS-STA checkpoint missing under %s: %s", ckpt_dir, ckpt_name)
            return None, None, None
    expect_sha = str(reg.get("checkpoint_sha256") or "")
    if not expect_sha:
        logger.warning("MS-STA registry missing checkpoint_sha256 — refuse load")
        return None, None, None
    digest = _sha256_file(ckpt_p)
    if digest != expect_sha:
        logger.warning("MS-STA checkpoint SHA mismatch: %s != %s", digest[:12], expect_sha[:12])
        return None, None, None

    payload = torch.load(ckpt_p, map_location=device, weights_only=False)

    if reg.get("ms_sta_v6") is True:
        variant = str(reg.get("active_variant") or "")
        payload_variant = str(payload.get("variant") or "")
        feature_roots = reg.get("feature_roots") or {}
        final_path = _PROJECT_ROOT / str(reg.get("final_results_rel") or "")
        d4_path = _PROJECT_ROOT / "Docs/results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json"
        evidence_ok = (
            not _ms_sta_v6_identity_errors(reg, payload)
            and variant == "b4_v6_ctr"
            and payload_variant == variant
            and final_path.is_file()
            and _sha256_file(final_path) == str(reg.get("final_results_sha256") or "")
            and d4_path.is_file()
            and _sha256_file(d4_path) == str(reg.get("d4_report_sha256") or "")
        )
        if not evidence_ok:
            logger.warning("MS-STA v6 registry/payload evidence contract failed - refuse load")
            return None, None, None

        rgb_root = _PROJECT_ROOT / str(feature_roots.get("rgb_root_rel") or "")
        skel_root = _PROJECT_ROOT / str(feature_roots.get("skeleton_root_rel") or "")
        if not (rgb_root / "rgb_token_v3").is_dir() or not (skel_root / "skeleton_v3").is_dir():
            logger.warning("MS-STA v6 feature roots missing - refuse load")
            return None, None, None

        from src.models.ms_sta_v6 import build_ms_sta_v6

        freeze_path = _PROJECT_ROOT / "Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json"
        if not freeze_path.is_file():
            logger.warning("MS-STA v6 pretest freeze missing - refuse load")
            return None, None, None
        freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
        model_cfg = freeze.get("model") or {}
        model = build_ms_sta_v6(
            "b4_v6_ctr",
            num_classes=int(model_cfg.get("num_classes", 400)),
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            temporal_layers=int(model_cfg["temporal_layers"]),
            dropout=float(payload["dropout"]),
            alpha_max=float(model_cfg["alpha_max"]),
        )
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).eval()
        _ms_sta_experiment_variant = variant
        _ms_sta_skel_temporal_stride = 1
        _ms_sta_v5 = False
        _ms_sta_v6 = True
        _ms_sta_v5_rgb_root = rgb_root
        _ms_sta_v5_skel_root = skel_root
        return model, variant, "defense"

    if reg.get("ms_sta_v5") is True:
        variant = str(reg.get("active_variant") or "")
        payload_variant = str(payload.get("variant") or "")
        feature_roots = reg.get("feature_roots") or {}
        final_path = _PROJECT_ROOT / str(reg.get("final_results_rel") or "")
        d4_path = _PROJECT_ROOT / "Docs/results/ms_sta_v5_d4/D4_ACCEPTANCE_REPORT.json"
        evidence_ok = (
            not _ms_sta_v5_identity_errors(reg, payload)
            and variant == "b4_star_fixed"
            and payload_variant == variant
            and final_path.is_file()
            and _sha256_file(final_path) == str(reg.get("final_results_sha256") or "")
            and d4_path.is_file()
            and _sha256_file(d4_path) == str(reg.get("d4_report_sha256") or "")
        )
        if not evidence_ok:
            logger.warning("MS-STA v5 registry/payload evidence contract failed - refuse load")
            return None, None, None

        rgb_root = _PROJECT_ROOT / str(feature_roots.get("rgb_root_rel") or "")
        skel_root = _PROJECT_ROOT / str(feature_roots.get("skeleton_root_rel") or "")
        if not (rgb_root / "rgb_token_v3").is_dir() or not (skel_root / "skeleton_v3").is_dir():
            logger.warning("MS-STA v5 feature roots missing - refuse load")
            return None, None, None

        from src.models.ms_sta_v5 import build_ms_sta_v5

        policy_path = _PROJECT_ROOT / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json"
        if not policy_path.is_file():
            logger.warning("MS-STA v5 policy missing - refuse load")
            return None, None, None
        policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
        model_cfg = policy.get("model") or {}
        model = build_ms_sta_v5(
            "b4_star_fixed",
            num_classes=400,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            temporal_layers=int(model_cfg["temporal_layers"]),
            dropout=float(payload["dropout"]),
            alpha_max=float(model_cfg["alpha_max"]),
        )
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).eval()
        _ms_sta_experiment_variant = variant
        _ms_sta_skel_temporal_stride = 1
        _ms_sta_v5 = True
        _ms_sta_v6 = False
        _ms_sta_v5_rgb_root = rgb_root
        _ms_sta_v5_skel_root = skel_root
        return model, variant, "defense"

    from src.models.ms_sta_wordlevel import build_ms_sta
    from src.training.retrain_v3 import model_variant_name

    v3 = reg.get("retrain_v3") is True or payload.get("retrain_v3") is True
    experiment_variant = str(
        reg.get("experiment_variant") or payload.get("experiment_variant") or ""
    ).strip()
    model_variant = model_variant_name(experiment_variant) if experiment_variant else str(
        reg.get("active_variant") or payload.get("variant") or "b4"
    )
    payload_variant = str(payload.get("variant") or "")
    if payload_variant != model_variant:
        logger.warning(
            "MS-STA payload.variant=%s != model_variant=%s (experiment=%s)",
            payload_variant,
            model_variant,
            experiment_variant or None,
        )
        return None, None, None
    reg_active = str(reg.get("active_variant") or model_variant)
    if reg_active != model_variant:
        logger.warning("MS-STA registry.active_variant=%s != model_variant=%s", reg_active, model_variant)
        return None, None, None

    if v3:
        if payload.get("retrain_v3") is not True:
            logger.warning("MS-STA v3 registry requires payload.retrain_v3=true")
            return None, None, None
        reg_policy = reg.get("policy_sha256")
        payload_policy = payload.get("retrain_v3_policy_sha256")
        if reg_policy:
            if not payload_policy or str(payload_policy) != str(reg_policy):
                logger.warning("MS-STA v3 policy SHA mismatch — refuse load")
                return None, None, None
        status = str(payload.get("checkpoint_status") or "")
        # V3 defense lifecycle is registry/pointer owned; exact-byte final checkpoints
        # are not rewritten with defense_ready after test.
        defense_ok = (
            reg.get("defense_ready") is True
            and reg.get("d4_pass") is True
            and reg.get("evaluation_locked") is True
            and status == "final_selected_checkpoint"
            and payload.get("retrain_v3") is True
        )
        candidate_ok = (
            not defense_ok
            and reg.get("defense_ready") is not True
            and reg.get("evaluation_locked") is True
            and status == "final_selected_checkpoint"
            and (
                _env_truthy("ALLOW_MS_STA_CANDIDATE")
                or reg.get("d4_candidate_allowed") is True
            )
        )
        if not defense_ok and not candidate_ok:
            return None, None, None
        if "skel_temporal_stride" not in payload:
            logger.warning("MS-STA v3 payload missing skel_temporal_stride — refuse load")
            return None, None, None
        stride = int(payload["skel_temporal_stride"])
        if experiment_variant == "b4e" and stride != 4:
            logger.warning("MS-STA b4e requires skel_temporal_stride=4, got %s", stride)
            return None, None, None
        if experiment_variant == "b4" and stride != 1:
            logger.warning("MS-STA b4 requires skel_temporal_stride=1, got %s", stride)
            return None, None, None
    else:
        defense_ok = (
            reg.get("defense_ready") is True
            and reg.get("e1_pass") is True
            and reg.get("e2_pass") is True
            and reg.get("d4_pass") is True
            and reg.get("d1_pass") is True
            and payload.get("defense_ready") is True
        )
        candidate_ok = (
            not defense_ok
            and reg.get("defense_ready") is not True
            and reg.get("e1_selected") is True
            and reg.get("evaluation_locked") is True
            and reg.get("e2_pass") is True
            and reg.get("d1_pass") is True
            and payload.get("e1_selected") is True
            and payload.get("evaluation_locked") is True
            and (
                _env_truthy("ALLOW_MS_STA_CANDIDATE")
                or reg.get("d4_candidate_allowed") is True
            )
        )
        if not defense_ok and not candidate_ok:
            return None, None, None
        stride = int(payload.get("skel_temporal_stride") or 1)

    n_cls = int(payload.get("num_classes") or (isolated_clf.num_classes if isolated_clf else 400))
    model = build_ms_sta(
        model_variant,  # type: ignore[arg-type]
        num_classes=n_cls,
        disable_spatial=bool(payload.get("disable_spatial", False)),
        skel_temporal_stride=stride,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    mode = "defense" if defense_ok else "candidate"
    _ms_sta_experiment_variant = experiment_variant or model_variant
    _ms_sta_skel_temporal_stride = int(stride)
    return model, model_variant, mode


class ClipPredictRequest(BaseModel):
    video_id: str
    mode: str = Field(
        default="isolated",
        description="isolated = VSL400 top-k | gloss_llm = windowed gloss → optional Gemini (experimental heuristic)",
    )
    top_k: int = 3
    use_llm: bool = Field(default=True, description="For gloss_llm: compose Vietnamese via Gemini")
    return_attention: bool = Field(
        default=False,
        description="Gate D4: include MS-STA attention maps when ms_sta ckpt is loaded",
    )
    use_ms_sta: bool = Field(
        default=False,
        description="If true and MS-STA ready/candidate, prefer MS-STA logits over B0 SlowFast head",
    )


def _public_error(exc: Exception) -> str:
    """Never leak raw internals to HTTP clients."""
    name = type(exc).__name__
    return f"Inference failed ({name}). Check server logs."


@app.on_event("startup")
def load_production_models() -> None:
    global isolated_clf, _metadata_df, _resolver, _video_index, _ms_sta_model, _ms_sta_variant
    global _ms_sta_load_mode, _ms_sta_experiment_variant, _ms_sta_skel_temporal_stride
    global _ms_sta_v5, _ms_sta_v6, _ms_sta_v5_rgb_root, _ms_sta_v5_skel_root
    global RGB_POOLED_DIR, RGB_TEMPORAL_DIR, SKEL_FEAT_DIR
    RGB_POOLED_DIR = resolve_pooled_dir(_PROJECT_ROOT)
    RGB_TEMPORAL_DIR = resolve_temporal_dir(_PROJECT_ROOT)
    SKEL_FEAT_DIR = resolve_skel_dir(_PROJECT_ROOT)
    print(f"[OK] Feature dirs pooled={RGB_POOLED_DIR.name} temporal={RGB_TEMPORAL_DIR.name} skel={SKEL_FEAT_DIR.name}")
    ckpt = _PROJECT_ROOT / DEFAULT_CKPT
    labels = _PROJECT_ROOT / DEFAULT_LABELS
    meta_path = _PROJECT_ROOT / "metadata.csv"
    _resolver = RawVideoResolver(_PROJECT_ROOT)
    # Warm filesystem indexes once (avoid ~1.5s rebuild per request).
    _ = _resolver.vsl400_index()
    _ = _resolver.vsl_index()
    if meta_path.is_file():
        _metadata_df = pd.read_csv(meta_path)
        _video_index = {
            str(vid): i for i, vid in enumerate(_metadata_df["video_id"].astype(str).tolist())
        }
    else:
        _metadata_df = None
        _video_index = None
        logger.warning("metadata.csv missing — /predict_clip will 503 until present")

    if ckpt.is_file() and labels.is_file():
        isolated_clf = IsolatedSignClassifier(ckpt, labels, device=device)
        print(f"[OK] Isolated classifier classes={isolated_clf.num_classes} on {device}")
    else:
        isolated_clf = None
        print(f"[WARN] Missing {ckpt.name}/{labels.name} — isolated/gloss_llm unavailable.")

    # Optional MS-STA: defense-approved cutover, or operator-only D4 candidate (never silent promote).
    # Registry path: MS_STA_REGISTRY_PATH → ms_sta_registry_active.json cutover → historical.
    _ms_sta_model = None
    _ms_sta_variant = None
    _ms_sta_load_mode = None
    _ms_sta_experiment_variant = None
    _ms_sta_skel_temporal_stride = None
    _ms_sta_v5 = False
    _ms_sta_v6 = False
    _ms_sta_v5_rgb_root = None
    _ms_sta_v5_skel_root = None
    registry_path = _resolve_ms_sta_registry_path()
    try:
        if registry_path is not None and registry_path.is_file():
            reg = json.loads(registry_path.read_text(encoding="utf-8"))
            model, variant, mode = _load_ms_sta_from_registry(reg)
            if model is not None and variant and mode:
                _ms_sta_model = model
                _ms_sta_variant = variant
                _ms_sta_load_mode = mode
                print(
                    f"[OK] MS-STA load_mode={mode} variant={variant} "
                    f"experiment={_ms_sta_experiment_variant} stride={_ms_sta_skel_temporal_stride} "
                    f"registry={registry_path.relative_to(_PROJECT_ROOT).as_posix()} "
                    f"retrain_v3={reg.get('retrain_v3')} defense_ready={reg.get('defense_ready')}"
                )
            else:
                print(
                    f"[OK] MS-STA not loaded — demo defaults to B0. "
                    f"registry={registry_path.name} "
                    f"retrain_v3={reg.get('retrain_v3')} "
                    f"d4_candidate_allowed={reg.get('d4_candidate_allowed')} "
                    f"ALLOW_MS_STA_CANDIDATE={_env_truthy('ALLOW_MS_STA_CANDIDATE')}"
                )
        else:
            print("[OK] No MS-STA registry resolved — demo defaults to B0 Gate3 head only")
    except Exception as exc:
        logger.warning("MS-STA optional load skipped: %s", exc)
        _ms_sta_model = None
        _ms_sta_variant = None
        _ms_sta_load_mode = None
        _ms_sta_experiment_variant = None
        _ms_sta_skel_temporal_stride = None

    print("[OK] Startup complete (VSL400 word-level only; Continuous_News removed from API).")


def _ready_payload() -> dict:
    iso_acc = None
    iso_path = _PROJECT_ROOT / "logs" / "train_vsl400" / "status.json"
    if not iso_path.is_file():
        iso_path = _PROJECT_ROOT / "logs" / "train_stage1_isolated" / "status.json"
    if iso_path.is_file():
        try:
            st = json.loads(iso_path.read_text(encoding="utf-8"))
            iso_acc = st.get("test_acc") if st.get("test_acc") is not None else st.get("best_acc")
        except Exception:
            iso_acc = None
    vsl400_metrics: dict = {}
    vsl400_eval = _PROJECT_ROOT / "logs" / "eval_vsl400" / "eval.json"
    if vsl400_eval.is_file():
        try:
            vsl400_metrics = json.loads(vsl400_eval.read_text(encoding="utf-8")).get("summary") or {}
        except Exception:
            vsl400_metrics = {}
    top1 = vsl400_metrics.get("top1_acc")
    if top1 is None:
        top1 = iso_acc
    ready = isolated_clf is not None and _metadata_df is not None and _resolver is not None
    ms_sta_defense = _ms_sta_load_mode == "defense" and _ms_sta_model is not None
    ms_sta_candidate = _ms_sta_load_mode == "candidate" and _ms_sta_model is not None
    reg_path = _resolve_ms_sta_registry_path()
    reg: dict = {}
    ckpt_sha = None
    if reg_path is not None and reg_path.is_file():
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8-sig"))
            ckpt_sha = reg.get("checkpoint_sha256")
        except Exception:
            reg = {}
            ckpt_sha = None
    if ms_sta_defense and _ms_sta_variant:
        if _ms_sta_v6:
            engine = "ms_sta_v6_b4_v6_ctr"
        elif _ms_sta_v5:
            engine = "ms_sta_v5_b4_star_fixed"  # rollback compatibility only
        else:
            engine = f"ms_sta_{_ms_sta_variant}"
    elif isolated_clf is not None:
        engine = "b0_gate3"
    else:
        engine = None

    # Historical E2 KPIs: retain on disk but do not surface as primary after VALIDATION_NO_GO.
    e2_kpi: dict = {}
    if _env_truthy("MS_STA_ALLOW_HISTORICAL_KPI") and not _primary_campaign_is_no_go():
        final_path = _PROJECT_ROOT / "Docs" / "results" / "ms_sta_defense_go" / "FINAL_RESULTS.json"
        if final_path.is_file():
            try:
                final = json.loads(final_path.read_text(encoding="utf-8-sig"))
                e2 = final.get("e2") or {}
                e2_kpi = {
                    "e2_b0_e1_top1": e2.get("b0_e1_top1"),
                    "e2_b0_e1_top5": e2.get("b0_e1_top5"),
                    "e2_b4_top1": e2.get("b4_top1"),
                    "e2_b4_top5": e2.get("b4_top5"),
                    "e2_delta_top1_pp": e2.get("delta_top1_pp"),
                    "e2_n": e2.get("n"),
                    "defense_status": final.get("defense_status"),
                    "active_variant": final.get("active_variant"),
                }
            except Exception:
                e2_kpi = {}

    v5_kpi: dict = {}
    v6_kpi: dict = {}
    if reg.get("ms_sta_v6") is True and reg.get("defense_ready") is True:
        final_path = _PROJECT_ROOT / str(reg.get("final_results_rel") or "")
        try:
            if _sha256_file(final_path) != str(reg.get("final_results_sha256") or ""):
                raise ValueError("v6_final_results_sha_mismatch")
            final = json.loads(final_path.read_text(encoding="utf-8-sig"))
            validation = final.get("validation") or {}
            test = final.get("test") or {}
            v6_kpi = {
                "scope": "v6_defense_go",
                "validation_confirm_top1": validation.get("b4_confirm_top1"),
                "test_kind": final.get("test_kind"),
                "test_n": test.get("n"),
                "test_b4_top1": test.get("b4_top1"),
                "test_mlp_top1": test.get("mlp_top1"),
                "test_delta_top1_pp": test.get("delta_top1_pp"),
                "pristine_blind_claim": final.get("pristine_blind_claim"),
                "e4_ready": False,
                "b4_ready": True,
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            v6_kpi = {}
    elif reg.get("ms_sta_v5") is True and reg.get("defense_ready") is True:
        final_path = _PROJECT_ROOT / str(reg.get("final_results_rel") or "")
        try:
            if _sha256_file(final_path) != str(reg.get("final_results_sha256") or ""):
                raise ValueError("v5_final_results_sha_mismatch")
            final = json.loads(final_path.read_text(encoding="utf-8-sig"))
            validation = final.get("validation") or {}
            test = final.get("test") or {}
            v5_kpi = {
                "scope": "v5_defense_go",
                "validation_b4_top1_pct": validation.get("b4_mean_top1_pct"),
                "validation_mlp_top1_pct": validation.get("mlp_mean_top1_pct"),
                "validation_delta_top1_pp": validation.get("mean_delta_top1_pp"),
                "test_kind": final.get("test_kind"),
                "test_n": test.get("n"),
                "test_b4_top1": test.get("b4_top1"),
                "test_mlp_top1": test.get("mlp_top1"),
                "test_delta_top1_pp": test.get("delta_top1_pp"),
                "pristine_blind_claim": final.get("pristine_blind_claim"),
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            v5_kpi = {}

    if _ms_sta_v5 or _ms_sta_v6:
        efficiency_ablation = {"ready": False, "scope": "historical_not_primary"}
        accuracy_study = {"ready": False, "scope": "historical_not_primary"}
    else:
        efficiency_ablation = _load_efficiency_ablation()
        accuracy_study = _load_accuracy_study()
    v3_kpi = _load_v3_closeout_kpi()
    no_go = _primary_campaign_is_no_go()

    disclaimer = (
        "Retrain-v4 = VALIDATION_NO_GO. Canonical reference on Path-B confirmation validation is B3 concat (64.49%). "
        "Historical E1-v2/E2/D4 and B4-only studies are not primary. "
        "No MS-STA checkpoint is defense/demo approved. Not a certified assistive translator."
        if no_go
        else (
            "Primary=Unified Attention on VSL400 Front-view word-level recognition. "
            "Test is retrospective signer-disjoint confirmation; demo uses sealed D0 caches. "
            "Compression/student studies are NO-GO appendix-only. Continuous_News API removed. "
            "Not a certified assistive translator."
            if _ms_sta_v6
            else (
                "Primary=Unified Attention (rollback v5) on VSL400 Front-view word-level recognition. "
                "Test is retrospective signer-disjoint confirmation; demo uses sealed D0 caches. "
                "Continuous_News API removed. Not a certified assistive translator."
            )
        )
    )

    return {
        "status": "ok" if ready else "not_ready",
        "ready": ready,
        "device": str(device),
        "isolated_classifier_ready": isolated_clf is not None,
        "metadata_ready": _metadata_df is not None,
        "isolated_num_classes": None if isolated_clf is None else isolated_clf.num_classes,
        "isolated_best_acc": top1,
        "vsl400_top1": top1,
        "vsl400_top5": vsl400_metrics.get("top5_acc"),
        "vsl400_fusion_top1": vsl400_metrics.get("fusion_top1_acc"),
        "fusion_scope": vsl400_metrics.get("fusion_scope"),
        "b0_gate3_top1": top1,
        "b0_gate3_top5": vsl400_metrics.get("top5_acc"),
        "primary_dataset": "VSL400",
        "modes": sorted(_ALLOWED_MODES),
        "continuous_news_removed": True,
        "engine": engine,
        "ms_sta_ready": ms_sta_defense,
        "ms_sta_candidate_ready": ms_sta_candidate,
        "ms_sta_load_mode": _ms_sta_load_mode,
        "ms_sta_variant": _ms_sta_variant,
        "ms_sta_experiment_variant": _ms_sta_experiment_variant,
        "ms_sta_skel_temporal_stride": _ms_sta_skel_temporal_stride,
        "ms_sta_checkpoint_sha256": ckpt_sha if ms_sta_defense else None,
        "ms_sta_v5": bool(_ms_sta_v5),
        "ms_sta_v6": bool(_ms_sta_v6),
        "experiment_variant": _ms_sta_experiment_variant or _ms_sta_variant,
        "feature_root_identities": (
            {
                "rgb_root_rel": str(reg.get("feature_roots", {}).get("rgb_root_rel") or ""),
                "skeleton_root_rel": str(reg.get("feature_roots", {}).get("skeleton_root_rel") or ""),
            }
            if ms_sta_defense
            else None
        ),
        "b4_ready": bool(reg.get("b4_ready")) if ms_sta_defense and _ms_sta_v6 else bool(ms_sta_defense and _ms_sta_v5),
        "e4_ready": bool(reg.get("e4_ready")) if ms_sta_defense and _ms_sta_v6 else False,
        "ms_sta_input_mode": "cache_backed_offline" if (_ms_sta_v5 or _ms_sta_v6) else None,
        "campaign_status": (
            "VALIDATION_NO_GO" if no_go else str(reg.get("campaign_status") or "unknown")
        ),
        "defense_ready": False if no_go else bool(ms_sta_defense),
        "demo_allowed": False if no_go else bool(ms_sta_defense),
        "retrain_v3_kpi": v3_kpi,
        "efficiency_ablation": efficiency_ablation,
        "accuracy_study": accuracy_study,
        "v5_kpi": v5_kpi,
        "v6_kpi": v6_kpi,
        **e2_kpi,
        "product": "VSL400 signer-independent word-level recognition with human-assisted output",
        "disclaimer": disclaimer,
    }


@app.get("/health")
async def health(response: Response) -> dict:
    payload = _ready_payload()
    if not payload["ready"]:
        response.status_code = 503
    return payload


def _resolve_raw_video(video_id: str) -> Path:
    if _metadata_df is None or _resolver is None or _video_index is None:
        raise HTTPException(status_code=503, detail="Metadata/resolver chưa sẵn sàng")
    idx = _video_index.get(str(video_id))
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Không thấy video_id={video_id} trong metadata")
    row = _metadata_df.iloc[idx]
    ds = str(row.get("dataset", "")).strip()
    if ds == "Continuous_News":
        raise HTTPException(
            status_code=410,
            detail="Continuous_News đã gỡ khỏi API — dùng video_id VSL400 / VSL-Cropped.",
        )
    path = _resolver.resolve(row)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file video cho {video_id}")
    return path


def _predict_ms_sta_v5(
    video_id: str, *, return_diagnostics: bool
) -> tuple[torch.Tensor, dict | None, dict]:
    if _ms_sta_model is None or _ms_sta_v5_rgb_root is None or _ms_sta_v5_skel_root is None:
        raise HTTPException(status_code=409, detail="MS-STA v5 defense cache/model not ready")

    from src.models.ms_sta_v5 import RGB_REGIONS
    from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5

    rgb_path = _ms_sta_v5_rgb_root / "rgb_token_v3" / f"{video_id}.npz"
    skel_path = _ms_sta_v5_skel_root / "skeleton_v3" / f"{video_id}.npz"
    missing = [name for name, path in (("rgb_token_v3", rgb_path), ("skeleton_v3", skel_path)) if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Missing MS-STA v5 cache for {video_id}: {', '.join(missing)} (no fallback)",
        )
    try:
        ds = VSL400MSSTAv5Dataset(
            [video_id],
            rgb_root=_ms_sta_v5_rgb_root,
            skel_root=_ms_sta_v5_skel_root,
            labels={video_id: 0},
            metadata_by_id={video_id: {"signer": "demo"}},
            forbid_ids=set(),
            require_all=True,
        )
        batch = collate_v5([ds[0]])
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid MS-STA v5 cache: {exc}") from exc

    x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        result = _ms_sta_model(x)
    logits = result["logits"]
    diagnostics = None
    if return_diagnostics:
        gate = result.get("gate_weights")
        alpha = result.get("alpha")
        diagnostics = {
            "scope": "model_diagnostics_not_attention_map",
            "stream_gate_rgb_skeleton": (
                gate.detach().float().cpu().squeeze(0).tolist() if torch.is_tensor(gate) else None
            ),
            "residual_alpha": float(alpha.detach().float().cpu()) if torch.is_tensor(alpha) else None,
        }

    with np.load(rgb_path, allow_pickle=False) as rgb_npz, np.load(skel_path, allow_pickle=False) as skel_npz:
        rgb_windows = int(rgb_npz["token_global"].shape[0])
        skel_frames = int(skel_npz["coordinates"].shape[0])
        validity = np.asarray(skel_npz["validity"], dtype=np.float32)
        missing_by_region = {
            region: int(np.asarray(rgb_npz[f"missing_{region}"], dtype=bool).sum())
            for region in RGB_REGIONS
            if f"missing_{region}" in rgb_npz.files
        }
    evidence = {
        "mode": "cache_backed_offline",
        "rgb_cache": "rgb_token_v3",
        "skeleton_cache": "skeleton_v3",
        "rgb_regions": list(RGB_REGIONS),
        "rgb_windows": rgb_windows,
        "skeleton_frames": skel_frames,
        "skeleton_joints": int(validity.shape[1]),
        "valid_joint_fraction": float(validity.mean()),
        "missing_rgb_windows_by_region": missing_by_region,
    }
    return logits, diagnostics, evidence


def _predict_isolated(
    video_id: str,
    top_k: int,
    *,
    return_attention: bool = False,
    use_ms_sta: bool = False,
) -> dict:
    if isolated_clf is None:
        raise HTTPException(status_code=503, detail="Isolated classifier chưa sẵn sàng")
    path = _resolve_raw_video(video_id)
    attention = None
    input_evidence = None
    engine = "b0_slowfast_head"
    tops = None

    if use_ms_sta and (_ms_sta_v5 or _ms_sta_v6):
        if isolated_clf is None:
            raise HTTPException(status_code=503, detail="VSL400 label map not ready")
        logits, attention, input_evidence = _predict_ms_sta_v5(
            video_id, return_diagnostics=return_attention
        )
        prob = torch.softmax(logits, dim=-1).squeeze(0)
        k = min(top_k, isolated_clf.num_classes)
        vals, idxs = torch.topk(prob, k=k)
        tops = [
            {
                "gloss": isolated_clf.id_to_gloss[int(i)],
                "vietnamese": isolated_clf.id_to_vietnamese[int(i)],
                "probability": float(score),
                "class_id": int(i),
            }
            for score, i in zip(vals.tolist(), idxs.tolist())
        ]
        if _ms_sta_v6:
            engine = "ms_sta_v6_b4_v6_ctr"
        elif _ms_sta_v5:
            engine = "ms_sta_v5_b4_star_fixed"  # rollback compatibility only
        else:
            engine = f"ms_sta_{_ms_sta_variant}"
    elif use_ms_sta:
        if _ms_sta_model is None or _ms_sta_variant is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "MS-STA requested but not loaded. "
                    "Defense requires status=GO; D4 candidate requires e2_pass + "
                    "d4_candidate_allowed or ALLOW_MS_STA_CANDIDATE=1. Use B0 / use_ms_sta=false."
                ),
            )
        from src.models.ms_sta_features import (
            FIXED_RGB_L,
            load_rgb_array,
            to_rgb_tensor,
            valid_mask_to_key_padding_mask,
        )

        needs_temporal = _ms_sta_variant in ("b1", "b3", "b4", "b5")
        rgb_p = (RGB_TEMPORAL_DIR if needs_temporal else RGB_POOLED_DIR) / f"{video_id}.npy"
        if not rgb_p.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Thiếu RGB {'temporal' if needs_temporal else 'pooled'} cache cho {video_id}",
            )
        try:
            rgb_t, valid = to_rgb_tensor(
                load_rgb_array(rgb_p),
                fixed_l=FIXED_RGB_L if needs_temporal else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        rgb = rgb_t.unsqueeze(0).to(device)
        rgb_kpm = None
        if valid is not None:
            rgb_kpm = valid_mask_to_key_padding_mask(valid.unsqueeze(0)).to(device)
        skel_t = None
        skel_p = SKEL_FEAT_DIR / f"{video_id}.npy"
        needs_skel = _ms_sta_variant in ("b2", "b3", "b4", "b5")
        if needs_skel and not skel_p.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Thiếu skeleton cache cho {video_id} (MS-STA {_ms_sta_variant}; no B0 fallback)",
            )
        if skel_p.is_file() and needs_skel:
            sk = np.load(skel_p).astype(np.float32)
            if sk.shape[0] != 64:
                ix = np.linspace(0, sk.shape[0] - 1, 64).astype(int)
                sk = sk[ix]
            skel_t = torch.from_numpy(sk).unsqueeze(0).to(device)
        with torch.no_grad():
            if _ms_sta_variant in ("b0", "b1"):
                out = _ms_sta_model(
                    rgb=rgb, rgb_key_padding_mask=rgb_kpm, return_attention=return_attention
                )
            elif _ms_sta_variant == "b2":
                if skel_t is None:
                    raise HTTPException(status_code=404, detail="MS-STA B2 cần skeleton cache")
                out = _ms_sta_model(skel=skel_t, return_attention=return_attention)
            else:
                if skel_t is None:
                    raise HTTPException(status_code=404, detail="MS-STA fusion cần skeleton cache")
                out = _ms_sta_model(
                    rgb=rgb,
                    skel=skel_t,
                    rgb_key_padding_mask=rgb_kpm,
                    return_attention=return_attention,
                )
            if return_attention:
                logits, attn_pack = out
                attention = {
                    k: (v.detach().float().cpu().tolist() if torch.is_tensor(v) else v)
                    for k, v in attn_pack.items()
                }
            else:
                logits = out
            prob = torch.softmax(logits, dim=-1).squeeze(0)
            k = min(top_k, isolated_clf.num_classes)
            vals, idxs = torch.topk(prob, k=k)
            tops = []
            for score, i in zip(vals.tolist(), idxs.tolist()):
                tops.append(
                    {
                        "gloss": isolated_clf.id_to_gloss[int(i)],
                        "vietnamese": isolated_clf.id_to_vietnamese[int(i)],
                        "probability": float(score),
                        "class_id": int(i),
                    }
                )
        engine = f"ms_sta_{_ms_sta_variant}"

    if tops is None:
        tops = isolated_clf.predict(path, top_k=max(1, int(top_k)))

    out = {
        "video_id": video_id,
        "mode": "isolated",
        "engine": engine,
        "predictions": tops,
        "attention": attention,
        "input_evidence": input_evidence,
        "suggestion_vietnamese_text": tops[0]["vietnamese"] if tops else None,
        "final_vietnamese_text": None,
        "requires_human_confirmation": True,
        "safe_for_end_user": False,
        "status": "NeedsHumanInterpreter",
        "user_message": "Top-k từ đơn — dùng cho từ điển / kiểm tra; cần người xác nhận.",
    }
    if use_ms_sta and _ms_sta_model is not None:
        out["ms_sta_variant"] = _ms_sta_variant
        out["ms_sta_experiment_variant"] = _ms_sta_experiment_variant
        out["ms_sta_skel_temporal_stride"] = _ms_sta_skel_temporal_stride
        out["ms_sta_load_mode"] = _ms_sta_load_mode
    return out


def _gloss_llm_assist(video_id: str, *, use_llm: bool, top_k: int) -> dict:
    if isolated_clf is None:
        raise HTTPException(
            status_code=503,
            detail="Chưa có VSL400 classifier — chạy src/training/train_vsl400_wordlevel.py --device cuda",
        )
    path = _resolve_raw_video(video_id)
    windows = isolated_clf.predict_windows(path, stride=16, conf_min=0.12)
    if len(windows) < 1:
        top = isolated_clf.predict(path, top_k=max(1, top_k))
        windows = [
            {
                "gloss": t["gloss"],
                "vietnamese": t["vietnamese"],
                "probability": t["probability"],
                "window_start": 0,
            }
            for t in top[:1]
        ]
    glosses = [w["gloss"] for w in windows]
    words = [w["vietnamese"] for w in windows]
    suggestion = None
    llm_meta: dict = {}
    if use_llm and glosses:
        llm_meta = compose_vietnamese_from_glosses(glosses, vietnamese_words=words)
        if llm_meta.get("ok"):
            suggestion = llm_meta.get("suggestion_vietnamese_text")
        else:
            llm_meta = {
                **llm_meta,
                "fail_closed": True,
                "note": "Gemini compose failed — only gloss sequence returned (no fake sentence).",
            }
    elif not use_llm and words:
        suggestion = " · ".join(words)
    return {
        "mode": "gloss_llm",
        "accepted": False,
        "requires_human_confirmation": True,
        "safe_for_end_user": False,
        "experimental_heuristic": True,
        "gloss_sequence": windows,
        "suggestion_vietnamese_text": suggestion,
        "final_vietnamese_text": None,
        "llm": llm_meta,
        "candidates": [
            {
                "text_vietnamese": suggestion or " · ".join(words),
                "similarity": float(np.mean([w["probability"] for w in windows])) if windows else 0.0,
                "video_id": video_id,
                "source": "gloss_llm",
            }
        ],
        "user_message": (
            "Chuỗi gloss nhận dạng từ video"
            + (" + Gemini biên câu (experimental heuristic)." if suggestion and use_llm else " (không có câu LLM — chỉ gloss).")
            + " CHƯA phải bản dịch chính thức — phiên dịch phải xác nhận/sửa trước khi gửi người dùng."
        ),
        "status": "NeedsHumanInterpreter",
    }


@app.post("/predict_clip")
async def predict_clip(payload: ClipPredictRequest) -> dict:
    """Word-level clip inference: mode=isolated|gloss_llm (sync work off event loop)."""
    vid = payload.video_id.strip()
    mode = (payload.mode or "isolated").strip().lower()
    if mode not in _ALLOWED_MODES:
        raise HTTPException(
            status_code=400,
            detail="mode phải là isolated|gloss_llm (Continuous_News assist/research đã gỡ).",
        )
    try:
        if mode == "isolated":
            return await asyncio.to_thread(
                _predict_isolated,
                vid,
                max(1, int(payload.top_k)),
                return_attention=bool(payload.return_attention),
                use_ms_sta=bool(payload.use_ms_sta),
            )
        out = await asyncio.to_thread(
            _gloss_llm_assist, vid, use_llm=bool(payload.use_llm), top_k=max(1, int(payload.top_k))
        )
        out["video_id"] = vid
        return out
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("predict_clip failed video_id=%s mode=%s", vid, mode)
        raise HTTPException(status_code=500, detail=_public_error(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
