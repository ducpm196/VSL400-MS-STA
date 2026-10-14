"""D4 acceptance matrix (temp fixtures) + optional final defense GO promotion.

Does NOT mutate E1-v2 / E2 lock source checkpoints. Promotion creates a new
defense copy and atomically updates registry only after all scenarios PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
E2_LOCK_DIR = _ROOT / "data" / "processed" / "ms_sta_e2_lock"
DEFENSE_DIR = _ROOT / "data" / "processed" / "ms_sta_defense"
DEMO = _ROOT / "data" / "processed" / "demo_set_ms_sta.json"
if not DEMO.is_file():
    DEMO = _ROOT / "data" / "processed" / "demo_set_option_b.json"
REPORT = _ROOT / "Docs" / "results" / "ms_sta_d4" / "D4_ACCEPTANCE_REPORT.json"
E2_CANON = _ROOT / "Docs" / "results" / "ms_sta_e2" / "E2_CANONICAL_REPORT.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_state_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode("utf-8"))
        t = state[k]
        if torch.is_tensor(t):
            h.update(t.detach().cpu().numpy().tobytes())
        else:
            h.update(repr(t).encode("utf-8"))
    return h.hexdigest()


def _scenario(name: str, ok: bool, **detail) -> dict:
    return {"scenario": name, "pass": bool(ok), **detail}


def _run_load_matrix() -> list[dict]:
    """Temp fixture scenarios for candidate load gates (no canonical mutation)."""
    import src.deployment.backend_api as api

    results = []
    src = E2_LOCK_DIR / "b4_seed42.pth"
    if not src.is_file():
        return [_scenario("fixtures_available", False, error="missing E2 lock b4_seed42.pth")]

    orig_resolve = api._resolve_ms_sta_ckpt_dir
    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ckpt_dir = td_path / "ckpts"
            ckpt_dir.mkdir()
            dest = ckpt_dir / "b4_seed42.pth"
            shutil.copy2(src, dest)
            good_sha = _sha256(dest)
            payload = torch.load(dest, map_location="cpu", weights_only=False)

            base = {
                "defense_ready": False,
                "e1_pass": True,
                "e1_selected": True,
                "evaluation_locked": True,
                "e2_pass": True,
                "d4_pass": False,
                "d1_pass": True,
                "active_variant": "b4",
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": good_sha,
                "d4_candidate_allowed": False,
            }

            os.environ.pop("ALLOW_MS_STA_CANDIDATE", None)
            api._resolve_ms_sta_ckpt_dir = lambda r: ckpt_dir  # type: ignore[method-assign]
            m, v, mode = api._load_ms_sta_from_registry(dict(base))
            results.append(
                _scenario(
                    "candidate_flag_off_no_load",
                    m is None and mode is None,
                    ms_sta_candidate_ready=False,
                    ms_sta_ready=False,
                )
            )

            reg2 = dict(base)
            reg2["d4_candidate_allowed"] = True
            m, v, mode = api._load_ms_sta_from_registry(reg2)
            results.append(
                _scenario(
                    "candidate_valid_load",
                    m is not None and mode == "candidate" and v == "b4",
                    ms_sta_candidate_ready=True,
                    ms_sta_ready=False,
                    load_mode=mode,
                )
            )

            reg3 = dict(reg2)
            reg3["checkpoint_sha256"] = "0" * 64
            m, v, mode = api._load_ms_sta_from_registry(reg3)
            results.append(_scenario("checkpoint_sha_mismatch_no_load", m is None and mode is None))

            reg4 = dict(reg2)
            reg4["checkpoint"] = "missing.pth"
            m, v, mode = api._load_ms_sta_from_registry(reg4)
            results.append(_scenario("checkpoint_missing_no_load", m is None and mode is None))

            reg5 = dict(reg2)
            reg5["active_variant"] = "b3"
            m, v, mode = api._load_ms_sta_from_registry(reg5)
            results.append(
                _scenario("registry_payload_variant_mismatch_no_load", m is None and mode is None)
            )

            reg6 = {
                "defense_ready": True,
                "e1_pass": True,
                "e2_pass": True,
                "d4_pass": True,
                "d1_pass": True,
                "active_variant": "b4",
                "checkpoint": "b4_seed42.pth",
                "checkpoint_sha256": good_sha,
            }
            m, v, mode = api._load_ms_sta_from_registry(reg6)
            results.append(
                _scenario(
                    "defense_refuses_non_defense_payload",
                    m is None and mode is None,
                    payload_defense_ready=payload.get("defense_ready"),
                )
            )

            m1, _, mode1 = api._load_ms_sta_from_registry(reg2)
            m2, _, mode2 = api._load_ms_sta_from_registry(reg2)
            results.append(
                _scenario(
                    "restart_reload_same_sha",
                    m1 is not None and m2 is not None and mode1 == mode2 == "candidate",
                    note="Two independent loads from disk/SHA; no shared RAM state required",
                )
            )
    finally:
        api._resolve_ms_sta_ckpt_dir = orig_resolve  # type: ignore[method-assign]

    return results


def _run_http_matrix() -> list[dict]:
    """HTTP scenarios via FastAPI TestClient with in-process candidate load."""
    from fastapi.testclient import TestClient

    import src.deployment.backend_api as api

    results = []
    demo = json.loads(DEMO.read_text(encoding="utf-8"))
    ok_vid = str(demo["scenarios"]["top1_correct"])  # 017681

    # Ensure models loaded for candidate from real registry (read-only)
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    # Force candidate path for this process
    os.environ.pop("ALLOW_MS_STA_CANDIDATE", None)
    # Use registry d4_candidate_allowed
    model, variant, mode = api._load_ms_sta_from_registry(reg)
    api._ms_sta_model = model
    api._ms_sta_variant = variant
    api._ms_sta_load_mode = mode

    # Need isolated_clf + metadata — call startup if needed
    if api.isolated_clf is None or api._metadata_df is None:
        api.load_production_models()
        # Re-apply candidate after startup (startup may have loaded already)
        model, variant, mode = api._load_ms_sta_from_registry(
            json.loads(REGISTRY.read_text(encoding="utf-8"))
        )
        api._ms_sta_model = model
        api._ms_sta_variant = variant
        api._ms_sta_load_mode = mode

    client = TestClient(api.app)

    # Health candidate
    h = client.get("/health").json()
    results.append(
        _scenario(
            "health_candidate_vs_ready",
            h.get("ms_sta_candidate_ready") is True and h.get("ms_sta_ready") is False,
            health={
                "ms_sta_candidate_ready": h.get("ms_sta_candidate_ready"),
                "ms_sta_ready": h.get("ms_sta_ready"),
                "ms_sta_load_mode": h.get("ms_sta_load_mode"),
                "ms_sta_variant": h.get("ms_sta_variant"),
            },
        )
    )

    # Valid cache-backed clip + attention schema
    r = client.post(
        "/predict_clip",
        json={
            "video_id": ok_vid,
            "mode": "isolated",
            "top_k": 5,
            "use_ms_sta": True,
            "return_attention": True,
        },
    )
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    attn_ok = isinstance(body.get("attention"), dict) and len(body.get("attention") or {}) > 0
    preds = body.get("predictions") or []
    schema_ok = (
        r.status_code == 200
        and body.get("engine") == "ms_sta_b4"
        and isinstance(preds, list)
        and len(preds) == 5
        and all({"gloss", "vietnamese", "probability", "class_id"} <= set(p) for p in preds)
        and attn_ok
    )
    results.append(
        _scenario(
            "cache_backed_clip_ms_sta_b4_attention",
            schema_ok,
            http_status=r.status_code,
            engine=body.get("engine"),
            attention_keys=list((body.get("attention") or {}).keys()),
            wording=(
                "offline/cache-backed inference only; not arbitrary-video or real-time deployment"
            ),
        )
    )

    # Missing temporal — fail-closed (no B0 fallback)
    # Temporarily point temporal dir to empty temp
    empty = Path(tempfile.mkdtemp())
    old_t = api.RGB_TEMPORAL_DIR
    api.RGB_TEMPORAL_DIR = empty
    r2 = client.post(
        "/predict_clip",
        json={"video_id": ok_vid, "mode": "isolated", "top_k": 3, "use_ms_sta": True},
    )
    api.RGB_TEMPORAL_DIR = old_t
    shutil.rmtree(empty, ignore_errors=True)
    no_fallback = r2.status_code in {404, 409, 422} and "b0_slowfast" not in (r2.text or "").lower()
    results.append(
        _scenario(
            "missing_temporal_fail_closed",
            no_fallback,
            http_status=r2.status_code,
            detail=(r2.text or "")[:240],
        )
    )

    # Missing skeleton — fail-closed
    empty_s = Path(tempfile.mkdtemp())
    old_s = api.SKEL_FEAT_DIR
    api.SKEL_FEAT_DIR = empty_s
    r3 = client.post(
        "/predict_clip",
        json={"video_id": ok_vid, "mode": "isolated", "top_k": 3, "use_ms_sta": True},
    )
    api.SKEL_FEAT_DIR = old_s
    shutil.rmtree(empty_s, ignore_errors=True)
    results.append(
        _scenario(
            "missing_skeleton_fail_closed",
            r3.status_code == 404 and "no b0 fallback" in (r3.text or "").lower(),
            http_status=r3.status_code,
            detail=(r3.text or "")[:240],
        )
    )

    # Broken / unknown video
    r4 = client.post(
        "/predict_clip",
        json={"video_id": "999999_DOES_NOT_EXIST", "mode": "isolated", "top_k": 3, "use_ms_sta": True},
    )
    results.append(
        _scenario(
            "broken_or_missing_video_4xx",
            r4.status_code in {404, 422, 400},
            http_status=r4.status_code,
            detail=(r4.text or "")[:240],
        )
    )

    # Explicit MS-STA request when model unloaded → 409
    saved = (api._ms_sta_model, api._ms_sta_variant, api._ms_sta_load_mode)
    api._ms_sta_model = None
    api._ms_sta_variant = None
    api._ms_sta_load_mode = None
    r5 = client.post(
        "/predict_clip",
        json={"video_id": ok_vid, "mode": "isolated", "top_k": 3, "use_ms_sta": True},
    )
    api._ms_sta_model, api._ms_sta_variant, api._ms_sta_load_mode = saved
    results.append(
        _scenario(
            "explicit_ms_sta_rejected_when_unloaded",
            r5.status_code == 409,
            http_status=r5.status_code,
        )
    )

    # Gemini failure keeps gloss sequence; LLM reports error separately
    def _boom(*_a, **_k):
        return {"ok": False, "error": "simulated_gemini_failure", "suggestion_vietnamese_text": None}

    old_fn = api.compose_vietnamese_from_glosses
    api.compose_vietnamese_from_glosses = _boom  # type: ignore[assignment]
    r6 = client.post(
        "/predict_clip",
        json={"video_id": ok_vid, "mode": "gloss_llm", "top_k": 3, "use_llm": True, "use_ms_sta": False},
    )
    api.compose_vietnamese_from_glosses = old_fn  # type: ignore[assignment]
    b6 = r6.json() if r6.status_code == 200 else {}
    gemini_ok = (
        r6.status_code == 200
        and bool(b6.get("gloss_sequence") or b6.get("candidates"))
        and (b6.get("llm") or {}).get("ok") is False
        and (b6.get("llm") or {}).get("error")
    )
    results.append(
        _scenario(
            "gemini_error_isolated_preserved",
            gemini_ok,
            http_status=r6.status_code,
            llm=b6.get("llm"),
        )
    )

    # UI contract (static): session key + health fields
    app_txt = (_ROOT / "src" / "deployment" / "app.py").read_text(encoding="utf-8")
    ui_ok = (
        "ms_sta_candidate_ready" in app_txt
        and "ms_sta_ready" in app_txt
        and "_clip_key" in app_txt
        and "st.session_state.result_payload" in app_txt
    )
    results.append(
        _scenario(
            "ui_candidate_ready_and_no_stale_result_key",
            ui_ok,
            note="Streamlit clears/binds results via _clip_key; health distinguishes candidate vs ready",
        )
    )

    return results


def _write_report(scenarios: list[dict], *, promote: bool) -> dict:
    all_pass = all(s.get("pass") for s in scenarios)
    report = {
        "schema": "option_b_d4_acceptance_report_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "D4",
        "all_scenarios_pass": all_pass,
        "d4_pass": all_pass,
        "defense_ready": False,
        "status": "NO-GO",
        "demo_smoke_wording": {
            "demo_set_top1_top5": "43.3% / 80% is functional smoke only",
            "not_kpi": True,
            "not_e2_substitute": True,
            "not_generalization_claim": True,
            "inference_mode": "offline/cache-backed feature clips only",
            "no_realtime_arbitrary_video_claim": True,
        },
        "e2_binding": {
            "canonical_report": "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json",
            "exists": E2_CANON.is_file(),
        },
        "scenarios": scenarios,
        "environment": {
            "python": sys.executable,
            "cuda_available": bool(torch.cuda.is_available()),
            "git_head": _git_rev(),
        },
        "promote_requested": promote,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _git_rev() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return ""


def _promote_defense(report: dict) -> dict:
    """Create defense checkpoint copy + atomic registry GO. Rollback helper returns prior reg."""
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    prior = json.loads(json.dumps(reg))
    src = E2_LOCK_DIR / Path(str(reg["checkpoint"])).name
    if not src.is_file():
        raise FileNotFoundError(src)
    src_sha = _sha256(src)
    if src_sha != reg.get("checkpoint_sha256"):
        raise RuntimeError("E2 lock SHA drift before promotion")

    payload = torch.load(src, map_location="cpu", weights_only=False)
    model_hash = _model_state_sha256(payload["model"])
    DEFENSE_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEFENSE_DIR / "b4_seed42_defense.pth"
    if dest.is_file():
        dest.unlink()

    payload = dict(payload)
    payload["defense_ready"] = True
    payload["d4_pass"] = True
    payload["e1_selected"] = True
    payload["evaluation_locked"] = True
    payload["source_checkpoint"] = str(src.relative_to(_ROOT)).replace("\\", "/")
    payload["source_checkpoint_sha256"] = src_sha
    payload["model_state_sha256"] = model_hash
    payload["d4_acceptance_report"] = str(REPORT.relative_to(_ROOT)).replace("\\", "/")
    payload["promoted_at"] = datetime.now(timezone.utc).isoformat()
    payload["defense_claim_limits"] = {
        "absolute_accuracy_limited": True,
        "signer_27_weak": True,
        "spatial_only_mixed": True,
        "no_efficiency_claim": True,
        "no_triview_superiority_claim": True,
        "no_realtime_arbitrary_video_claim": True,
        "cache_backed_inference_only": True,
    }
    torch.save(payload, dest)
    defense_sha = _sha256(dest)

    # Verify weights unchanged vs source model state
    src_payload = torch.load(src, map_location="cpu", weights_only=False)
    if _model_state_sha256(src_payload["model"]) != model_hash:
        raise RuntimeError("model_state hash changed during promotion")

    reg["d4_pass"] = True
    reg["defense_ready"] = True
    reg["status"] = "GO"
    reg["d4_candidate_allowed"] = False
    reg["ckpts_dir"] = str(DEFENSE_DIR.relative_to(_ROOT)).replace("\\", "/")
    reg["checkpoint"] = dest.name
    reg["checkpoint_sha256"] = defense_sha
    reg["active_variant"] = "b4"
    reg["e2_pass"] = True
    reg["e1_pass"] = True
    reg["e1_selected"] = True
    reg["evaluation_locked"] = True
    reg["defense_promotion"] = {
        "promoted_at": payload["promoted_at"],
        "source_checkpoint": payload["source_checkpoint"],
        "source_checkpoint_sha256": src_sha,
        "defense_checkpoint": str(dest.relative_to(_ROOT)).replace("\\", "/"),
        "defense_checkpoint_sha256": defense_sha,
        "model_state_sha256": model_hash,
        "d4_acceptance_report": payload["d4_acceptance_report"],
        "weights_unchanged_from_e2_lock": True,
    }
    note = (
        "D4 PASS → defense GO (B4); limited publication claim; "
        "cache-backed demo only; candidate disabled"
    )
    reg["reasons"] = [note] + [r for r in (reg.get("reasons") or []) if "D4 PASS" not in r]
    reg["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Atomic-ish write
    tmp = REGISTRY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(REGISTRY)

    report["defense_ready"] = True
    report["d4_pass"] = True
    report["status"] = "GO"
    report["promotion"] = reg["defense_promotion"]
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"prior": prior, "reg": reg, "defense_sha": defense_sha, "model_state_sha256": model_hash}


def _post_promotion_smoke() -> dict:
    import src.deployment.backend_api as api

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    model, variant, mode = api._load_ms_sta_from_registry(reg)
    ok = model is not None and mode == "defense" and variant == "b4"
    # Health payload simulation
    api._ms_sta_model = model
    api._ms_sta_variant = variant
    api._ms_sta_load_mode = mode
    from fastapi.testclient import TestClient

    if api.isolated_clf is None:
        api.load_production_models()
        model, variant, mode = api._load_ms_sta_from_registry(reg)
        api._ms_sta_model = model
        api._ms_sta_variant = variant
        api._ms_sta_load_mode = mode
    client = TestClient(api.app)
    h = client.get("/health").json()
    demo = json.loads(DEMO.read_text(encoding="utf-8"))
    vid = str(demo["scenarios"]["top1_correct"])
    r = client.post(
        "/predict_clip",
        json={"video_id": vid, "mode": "isolated", "top_k": 5, "use_ms_sta": True, "return_attention": True},
    )
    body = r.json() if r.status_code == 200 else {}
    smoke = {
        "pass": bool(
            ok
            and h.get("ms_sta_ready") is True
            and h.get("ms_sta_candidate_ready") is False
            and body.get("engine") == "ms_sta_b4"
            and r.status_code == 200
        ),
        "health": {
            "ms_sta_ready": h.get("ms_sta_ready"),
            "ms_sta_candidate_ready": h.get("ms_sta_candidate_ready"),
            "ms_sta_load_mode": h.get("ms_sta_load_mode"),
        },
        "http_status": r.status_code,
        "engine": body.get("engine"),
        "checkpoint": reg.get("checkpoint"),
        "checkpoint_sha256": reg.get("checkpoint_sha256"),
    }
    return smoke


def _rollback(prior: dict) -> None:
    REGISTRY.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true", help="Promote to defense GO if acceptance PASS")
    args = ap.parse_args()

    scenarios = []
    scenarios.extend(_run_load_matrix())
    scenarios.extend(_run_http_matrix())
    report = _write_report(scenarios, promote=bool(args.promote))
    print(json.dumps({"all_pass": report["all_scenarios_pass"], "n": len(scenarios), "report": str(REPORT)}, indent=2))
    failed = [s["scenario"] for s in scenarios if not s.get("pass")]
    if failed:
        print("[FAIL] scenarios:", failed)
        return 2

    if not args.promote:
        print("[OK] acceptance PASS; promote not requested")
        return 0

    print("[PROMOTE] creating defense checkpoint + atomic GO registry…")
    try:
        promo = _promote_defense(report)
    except Exception as exc:
        print(f"[FAIL] promotion aborted: {exc}")
        return 2

    smoke = _post_promotion_smoke()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["post_promotion_smoke"] = smoke
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not smoke.get("pass"):
        print("[FAIL] post-promotion smoke failed — rolling back registry to NO-GO")
        _rollback(promo["prior"])
        # leave defense file for audit but registry not GO
        return 3

    print(json.dumps({"status": "GO", "smoke": smoke, "defense_sha": promo["defense_sha"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
