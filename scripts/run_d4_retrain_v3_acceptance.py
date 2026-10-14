"""Retrain-v3 D4 acceptance: exact-byte defense + post-promotion smoke before cutover."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v3 import load_policy, model_variant_name, require_frozen_v3_policy, sha256_file

DEFAULT_POLICY = _ROOT / "Docs/results/ms_sta_retrain_v3_policy/RETRAIN_POLICY.json"
HISTORICAL_REGISTRY = _ROOT / "data/processed/ms_sta_registry.json"
DEMO_V3 = _ROOT / "data/processed/demo_set_ms_sta_v3.json"
ACTIVE_POINTER = _ROOT / "data/processed/ms_sta_registry_active.json"


def _scenario(name: str, passed: bool, **extra) -> dict:
    out = {"name": name, "pass": bool(passed)}
    out.update(extra)
    return out


def _expected_stride(active: str) -> int:
    return 4 if active == "b4e" else 1


def _v3_candidate_reg(
    *,
    active: str,
    ckpt_name: str,
    sha: str,
    policy_sha: str,
    allowed: bool,
) -> dict:
    return {
        "schema": "ms_sta_registry_v3_candidate",
        "retrain_v3": True,
        "defense_ready": False,
        "d4_pass": False,
        "d4_candidate_allowed": bool(allowed),
        "evaluation_locked": True,
        "active_variant": model_variant_name(active),
        "experiment_variant": active,
        "checkpoint": ckpt_name,
        "checkpoint_sha256": sha,
        "policy_sha256": policy_sha,
    }


def _v3_defense_reg(
    *,
    active: str,
    ckpt_rel: str,
    sha: str,
    policy_sha: str,
    ckpts_dir: str,
    lock: dict,
) -> dict:
    return {
        "schema": "ms_sta_registry_v3",
        "retrain_v3": True,
        "defense_ready": True,
        "d4_pass": True,
        "d4_candidate_allowed": False,
        "evaluation_locked": True,
        "active_variant": model_variant_name(active),
        "experiment_variant": active,
        "checkpoint": ckpt_rel,
        "checkpoint_sha256": sha,
        "ckpts_dir": ckpts_dir,
        "test_claim_label": lock.get("test_claim_label"),
        "policy_sha256": policy_sha,
        "historical_registry_immutable": HISTORICAL_REGISTRY.relative_to(_ROOT).as_posix(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _bind_runtime(api, model, variant, mode) -> None:
    api._ms_sta_model = model
    api._ms_sta_variant = variant
    api._ms_sta_load_mode = mode
    # Loader already sets experiment/stride globals; keep them if model present.


def _clear_runtime(api) -> None:
    api._ms_sta_model = None
    api._ms_sta_variant = None
    api._ms_sta_load_mode = None
    api._ms_sta_experiment_variant = None
    api._ms_sta_skel_temporal_stride = None


def _check_predict_identity(active: str, demo: dict, client) -> list[dict]:
    results = []
    top1 = demo["scenario_rows"]["top1_correct"]
    assist = demo["scenario_rows"]["top5_assist"]
    exp_stride = _expected_stride(active)
    model_var = model_variant_name(active)

    r = client.post(
        "/predict_clip",
        json={
            "video_id": str(top1["video_id"]),
            "mode": "isolated",
            "top_k": 5,
            "use_ms_sta": True,
            "return_attention": True,
        },
    )
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    preds = body.get("predictions") or []
    engine = str(body.get("engine") or "")
    schema_ok = (
        r.status_code == 200
        and engine == f"ms_sta_{model_var}"
        and isinstance(preds, list)
        and len(preds) == 5
        and all({"gloss", "vietnamese", "probability", "class_id"} <= set(p) for p in preds)
    )
    gold_id = int(top1["gold_class_id"])
    pred_id = int(top1["pred_active_class_id"])
    top1_ok = (
        schema_ok
        and int(preds[0]["class_id"]) == pred_id
        and int(preds[0]["class_id"]) == gold_id
        and body.get("ms_sta_experiment_variant") == active
        and int(body.get("ms_sta_skel_temporal_stride") or -1) == exp_stride
        and body.get("ms_sta_variant") == model_var
    )
    results.append(
        _scenario(
            "predict_identity_top1_correct",
            top1_ok,
            http_status=r.status_code,
            engine=engine,
            expected_class_id=gold_id,
            got_class_id=preds[0]["class_id"] if preds else None,
            experiment_variant=body.get("ms_sta_experiment_variant"),
            stride=body.get("ms_sta_skel_temporal_stride"),
        )
    )

    r2 = client.post(
        "/predict_clip",
        json={
            "video_id": str(assist["video_id"]),
            "mode": "isolated",
            "top_k": 5,
            "use_ms_sta": True,
        },
    )
    body2 = r2.json() if r2.headers.get("content-type", "").startswith("application/json") else {}
    preds2 = body2.get("predictions") or []
    assist_gold = int(assist["gold_class_id"])
    ids = [int(p["class_id"]) for p in preds2] if isinstance(preds2, list) else []
    assist_ok = (
        r2.status_code == 200
        and len(preds2) == 5
        and assist_gold in ids
        and body2.get("ms_sta_experiment_variant") == active
        and int(body2.get("ms_sta_skel_temporal_stride") or -1) == exp_stride
    )
    results.append(
        _scenario(
            "predict_identity_top5_assist",
            assist_ok,
            http_status=r2.status_code,
            gold_class_id=assist_gold,
            returned_class_ids=ids,
        )
    )
    return results


def _run_load_matrix(active: str, src: Path, expected_sha: str, policy_sha: str) -> list[dict]:
    import src.deployment.backend_api as api

    results = []
    orig_resolve = api._resolve_ms_sta_ckpt_dir
    try:
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "ckpts"
            ckpt_dir.mkdir()
            dest = ckpt_dir / f"{active}_seed42.pth"
            shutil.copy2(src, dest)
            good_sha = sha256_file(dest)
            if good_sha != expected_sha:
                return [_scenario("fixture_sha_match_source", False, error="temp copy sha mismatch")]

            base = _v3_candidate_reg(
                active=active,
                ckpt_name=dest.name,
                sha=good_sha,
                policy_sha=policy_sha,
                allowed=False,
            )
            os.environ.pop("ALLOW_MS_STA_CANDIDATE", None)
            api._resolve_ms_sta_ckpt_dir = lambda r: ckpt_dir  # type: ignore[method-assign]

            m, v, mode = api._load_ms_sta_from_registry(dict(base))
            results.append(
                _scenario(
                    "candidate_flag_off_no_load",
                    m is None and mode is None,
                )
            )

            reg2 = dict(base)
            reg2["d4_candidate_allowed"] = True
            m, v, mode = api._load_ms_sta_from_registry(reg2)
            stride_ok = m is not None and int(getattr(m, "skel_temporal_stride", -1)) == _expected_stride(
                active
            )
            results.append(
                _scenario(
                    "candidate_valid_load",
                    m is not None
                    and mode == "candidate"
                    and v == model_variant_name(active)
                    and stride_ok,
                    skel_temporal_stride=getattr(m, "skel_temporal_stride", None) if m else None,
                )
            )
            results.append(
                _scenario(
                    "active_stride_identity",
                    stride_ok,
                    expected_stride=_expected_stride(active),
                    actual_stride=getattr(m, "skel_temporal_stride", None) if m else None,
                )
            )
            if active == "b4e":
                results.append(
                    _scenario(
                        "b4e_stride_identity",
                        m is not None and int(getattr(m, "skel_temporal_stride", -1)) == 4,
                    )
                )

            reg3 = dict(reg2)
            reg3["checkpoint_sha256"] = "0" * 64
            m, _, _ = api._load_ms_sta_from_registry(reg3)
            results.append(_scenario("checkpoint_sha_mismatch_no_load", m is None))

            reg4 = dict(reg2)
            reg4["checkpoint"] = "missing.pth"
            m, _, _ = api._load_ms_sta_from_registry(reg4)
            results.append(_scenario("checkpoint_missing_no_load", m is None))

            reg5 = dict(reg2)
            reg5["active_variant"] = "b3" if model_variant_name(active) != "b3" else "b0"
            m, _, _ = api._load_ms_sta_from_registry(reg5)
            results.append(_scenario("registry_payload_variant_mismatch_no_load", m is None))

            payload = torch.load(dest, map_location="cpu", weights_only=False)
            results.append(
                _scenario(
                    "payload_engine_identity_retrain_v3_final",
                    payload.get("retrain_v3") is True
                    and payload.get("checkpoint_status") == "final_selected_checkpoint"
                    and int(payload.get("seed") or -1) == 42
                    and payload.get("e1_selected") is False
                    and "skel_temporal_stride" in payload
                    and str(payload.get("retrain_v3_policy_sha256") or "") == policy_sha,
                )
            )
    finally:
        api._resolve_ms_sta_ckpt_dir = orig_resolve  # type: ignore[method-assign]
    return results


def _run_http_candidate_matrix(active: str, src: Path, expected_sha: str, policy_sha: str) -> list[dict]:
    from fastapi.testclient import TestClient

    import src.deployment.backend_api as api

    results = []
    if not DEMO_V3.is_file():
        return [_scenario("demo_set_v3_lineage", False, error="missing demo_set_ms_sta_v3.json")]
    demo = json.loads(DEMO_V3.read_text(encoding="utf-8"))
    lineage_ok = (
        demo.get("schema") == "ms_sta_demo_set_v3"
        and demo.get("retrain_v3") is True
        and demo.get("reuse_e2_demo") is False
        and demo.get("reuse_metrics") is False
        and "gold_class_id" in (demo.get("scenario_rows") or {}).get("top1_correct", {})
        and "pred_active_class_id" in (demo.get("scenario_rows") or {}).get("top1_correct", {})
    )
    results.append(_scenario("demo_set_v3_lineage", lineage_ok, demo_schema=demo.get("schema")))
    if not lineage_ok:
        return results

    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td) / "ckpts"
        ckpt_dir.mkdir()
        dest = ckpt_dir / f"{active}_seed42.pth"
        shutil.copy2(src, dest)
        good_sha = sha256_file(dest)
        assert good_sha == expected_sha

        reg = _v3_candidate_reg(
            active=active,
            ckpt_name=dest.name,
            sha=good_sha,
            policy_sha=policy_sha,
            allowed=True,
        )
        reg["ckpts_dir"] = str(ckpt_dir)
        orig_resolve = api._resolve_ms_sta_ckpt_dir
        try:
            api._resolve_ms_sta_ckpt_dir = lambda r: ckpt_dir  # type: ignore[method-assign]
            if api.isolated_clf is None or api._metadata_df is None:
                api.load_production_models()
            model, variant, mode = api._load_ms_sta_from_registry(reg)
            _bind_runtime(api, model, variant, mode)
            client = TestClient(api.app)

            h = client.get("/health").json()
            health_ok = (
                h.get("ms_sta_candidate_ready") is True
                and h.get("ms_sta_ready") is False
                and h.get("ms_sta_variant") == model_variant_name(active)
                and h.get("ms_sta_experiment_variant") == active
                and int(h.get("ms_sta_skel_temporal_stride") or -1) == _expected_stride(active)
            )
            results.append(
                _scenario(
                    "health_candidate_vs_ready",
                    health_ok,
                    health={
                        "ms_sta_candidate_ready": h.get("ms_sta_candidate_ready"),
                        "ms_sta_ready": h.get("ms_sta_ready"),
                        "ms_sta_variant": h.get("ms_sta_variant"),
                        "ms_sta_experiment_variant": h.get("ms_sta_experiment_variant"),
                        "ms_sta_skel_temporal_stride": h.get("ms_sta_skel_temporal_stride"),
                    },
                )
            )

            results.extend(_check_predict_identity(active, demo, client))
            # Keep legacy schema smoke name for verify_d4 compatibility.
            results.append(
                _scenario(
                    "cache_backed_predict_clip_ms_sta",
                    any(s["name"] == "predict_identity_top1_correct" and s["pass"] for s in results),
                )
            )

            ok_vid = str(demo["scenarios"]["top1_correct"])
            empty = Path(tempfile.mkdtemp())
            old_t = api.RGB_TEMPORAL_DIR
            api.RGB_TEMPORAL_DIR = empty
            try:
                r2 = client.post(
                    "/predict_clip",
                    json={"video_id": ok_vid, "mode": "isolated", "top_k": 5, "use_ms_sta": True},
                )
                results.append(
                    _scenario("missing_temporal_fail_closed", r2.status_code >= 400, http_status=r2.status_code)
                )
            finally:
                api.RGB_TEMPORAL_DIR = old_t

            r3 = client.post(
                "/predict_clip",
                json={"video_id": "999999", "mode": "isolated", "top_k": 5, "use_ms_sta": True},
            )
            results.append(
                _scenario("unknown_video_4xx", 400 <= r3.status_code < 500, http_status=r3.status_code)
            )

            _clear_runtime(api)
            r4 = client.post(
                "/predict_clip",
                json={"video_id": ok_vid, "mode": "isolated", "top_k": 5, "use_ms_sta": True},
            )
            results.append(_scenario("unloaded_ms_sta_409", r4.status_code == 409, http_status=r4.status_code))
        finally:
            api._resolve_ms_sta_ckpt_dir = orig_resolve  # type: ignore[method-assign]
            _clear_runtime(api)
    return results


def _rel_or_abs(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _stage_exact_defense(
    *,
    active: str,
    src: Path,
    src_sha: str,
    ckpt_dir: Path,
    policy_sha: str,
    lock: dict,
) -> dict:
    """Copy exact final-test bytes into defense path; write staged v3 registry (no cutover)."""
    defense_dir = ckpt_dir / "defense"
    defense_dir.mkdir(parents=True, exist_ok=True)
    defense_ckpt = defense_dir / f"{active}_seed42_defense.pth"
    tmp_ckpt = defense_ckpt.with_suffix(".pth.tmp")
    shutil.copy2(src, tmp_ckpt)
    defense_sha = sha256_file(tmp_ckpt)
    if defense_sha != src_sha:
        tmp_ckpt.unlink(missing_ok=True)
        raise RuntimeError(f"defense copy sha mismatch: {defense_sha} != {src_sha}")
    os.replace(tmp_ckpt, defense_ckpt)
    if sha256_file(defense_ckpt) != src_sha:
        raise RuntimeError("defense checkpoint sha drifted after replace")

    v3_reg_path = ckpt_dir / "ms_sta_registry_v3.json"
    v3_reg = _v3_defense_reg(
        active=active,
        ckpt_rel=_rel_or_abs(defense_ckpt),
        sha=defense_sha,
        policy_sha=policy_sha,
        ckpts_dir=_rel_or_abs(defense_dir),
        lock=lock,
    )
    tmp_reg = v3_reg_path.with_suffix(".json.tmp")
    tmp_reg.write_text(json.dumps(v3_reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_reg, v3_reg_path)
    return {
        "defense_checkpoint": _rel_or_abs(defense_ckpt),
        "defense_sha256": defense_sha,
        "source_sha256": src_sha,
        "exact_byte_copy": True,
        "v3_registry": _rel_or_abs(v3_reg_path),
        "v3_registry_sha256": sha256_file(v3_reg_path),
        "v3_reg": v3_reg,
        "defense_ckpt_abs": defense_ckpt,
        "v3_reg_path": v3_reg_path,
    }


def _run_defense_post_promotion_smoke(
    *,
    active: str,
    staged: dict,
    policy_sha: str,
) -> list[dict]:
    from fastapi.testclient import TestClient

    import src.deployment.backend_api as api

    results = []
    demo = json.loads(DEMO_V3.read_text(encoding="utf-8"))
    defense_ckpt = Path(staged["defense_ckpt_abs"])
    defense_dir = defense_ckpt.parent
    reg = dict(staged["v3_reg"])
    # Load via basename under defense dir (same as production path resolution).
    reg["checkpoint"] = defense_ckpt.name
    reg["ckpts_dir"] = str(defense_dir)
    reg["checkpoint_sha256"] = staged["defense_sha256"]
    reg["policy_sha256"] = policy_sha

    if staged["defense_sha256"] != staged["source_sha256"]:
        return [_scenario("defense_exact_sha_equals_source", False)]

    results.append(
        _scenario(
            "defense_exact_sha_equals_source",
            True,
            defense_sha256=staged["defense_sha256"],
            source_sha256=staged["source_sha256"],
        )
    )

    orig_resolve = api._resolve_ms_sta_ckpt_dir
    try:
        api._resolve_ms_sta_ckpt_dir = lambda r: defense_dir  # type: ignore[method-assign]
        if api.isolated_clf is None or api._metadata_df is None:
            api.load_production_models()
        model, variant, mode = api._load_ms_sta_from_registry(reg)
        _bind_runtime(api, model, variant, mode)
        client = TestClient(api.app)
        h = client.get("/health").json()
        health_ok = (
            model is not None
            and mode == "defense"
            and h.get("ms_sta_ready") is True
            and h.get("ms_sta_candidate_ready") is False
            and h.get("ms_sta_variant") == model_variant_name(active)
            and h.get("ms_sta_experiment_variant") == active
            and int(h.get("ms_sta_skel_temporal_stride") or -1) == _expected_stride(active)
            and int(getattr(model, "skel_temporal_stride", -1)) == _expected_stride(active)
        )
        results.append(
            _scenario(
                "health_defense_post_promotion",
                health_ok,
                health={
                    "ms_sta_ready": h.get("ms_sta_ready"),
                    "ms_sta_candidate_ready": h.get("ms_sta_candidate_ready"),
                    "ms_sta_variant": h.get("ms_sta_variant"),
                    "ms_sta_experiment_variant": h.get("ms_sta_experiment_variant"),
                    "ms_sta_skel_temporal_stride": h.get("ms_sta_skel_temporal_stride"),
                    "load_mode": mode,
                },
            )
        )
        if active == "b4e":
            results.append(
                _scenario(
                    "b4e_stride_identity_defense",
                    health_ok and int(h.get("ms_sta_skel_temporal_stride") or -1) == 4,
                )
            )
        identity = _check_predict_identity(active, demo, client)
        # Rename for defense phase clarity while keeping identity semantics.
        for s in identity:
            s = dict(s)
            s["name"] = f"defense_{s['name']}"
            results.append(s)
    finally:
        api._resolve_ms_sta_ckpt_dir = orig_resolve  # type: ignore[method-assign]
        _clear_runtime(api)
    return results


def _cutover_active_pointer(*, staged: dict, status_dir: Path, policy_sha: str) -> dict:
    """Atomically publish active pointer only after post-promotion smoke PASS."""
    v3_reg_path = Path(staged["v3_reg_path"])
    v3_reg_sha = staged["v3_registry_sha256"]
    ACTIVE_POINTER.parent.mkdir(parents=True, exist_ok=True)
    cutover = {
        "schema": "ms_sta_registry_active_pointer_v1",
        "lineage": "retrain_v3",
        "registry_path": v3_reg_path.relative_to(_ROOT).as_posix(),
        "registry_sha256": v3_reg_sha,
        "historical_registry_immutable": HISTORICAL_REGISTRY.relative_to(_ROOT).as_posix(),
        "historical_registry_mutated": False,
        "cutover_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_sha,
        "defense_sha256": staged["defense_sha256"],
        "source_sha256": staged["source_sha256"],
        "exact_byte_copy": True,
    }
    tmp_ptr = ACTIVE_POINTER.with_suffix(".json.tmp")
    tmp_ptr.write_text(json.dumps(cutover, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_ptr, ACTIVE_POINTER)

    marker = status_dir / "REGISTRY_V3_PROMOTED.json"
    marker.write_text(
        json.dumps(
            {
                "v3_registry": staged["v3_registry"],
                "v3_registry_sha256": v3_reg_sha,
                "defense_checkpoint": staged["defense_checkpoint"],
                "defense_sha256": staged["defense_sha256"],
                "source_sha256": staged["source_sha256"],
                "exact_byte_copy": True,
                "registry_cutover_path": ACTIVE_POINTER.relative_to(_ROOT).as_posix(),
                "historical_registry_mutated": False,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "registry_updated": True,
        "historical_registry_mutated": False,
        "registry_cutover_path": ACTIVE_POINTER.relative_to(_ROOT).as_posix(),
        "promotion_marker": marker.relative_to(_ROOT).as_posix(),
        "defense_checkpoint": staged["defense_checkpoint"],
        "defense_sha256": staged["defense_sha256"],
        "source_sha256": staged["source_sha256"],
        "exact_byte_copy": True,
        "v3_registry": staged["v3_registry"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--eval-lock", type=Path, default=None)
    ap.add_argument("--confirm-d4", action="store_true")
    args = ap.parse_args()
    if not args.confirm_d4:
        print("[FAIL] refuse D4 without --confirm-d4")
        return 2

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    require_frozen_v3_policy(policy)
    policy_sha = sha256_file(policy_path)
    status_dir = _ROOT / str((policy.get("artifact_dirs") or {}).get("status") or "logs/ms_sta_retrain_v3")
    ckpt_dir = _ROOT / str(
        (policy.get("artifact_dirs") or {}).get("checkpoints") or "data/processed/ms_sta_retrain_v3"
    )
    lock_path = args.eval_lock or (status_dir / "EVAL_LOCK.json")
    if not Path(lock_path).is_absolute():
        lock_path = _ROOT / lock_path
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not lock.get("evaluation_locked"):
        print("[FAIL] evaluation not locked")
        return 2
    test_matrix = status_dir / "TEST_MATRIX.json"
    if not test_matrix.is_file():
        print("[FAIL] TEST_MATRIX.json missing; run test phase first")
        return 2
    tm = json.loads(test_matrix.read_text(encoding="utf-8"))
    if set(tm.get("variants_present") or []) != {"b0", "b3", "b4", "b4e"}:
        print("[FAIL] TEST_MATRIX missing required variants")
        return 2
    if not DEMO_V3.is_file():
        print("[FAIL] demo_set_ms_sta_v3.json missing; evaluator must emit it")
        return 2

    active = lock.get("active_variant")
    meta = (lock.get("checkpoints") or {}).get(active)
    if not meta:
        print(f"[FAIL] active variant {active} missing from EVAL_LOCK")
        return 2
    src = _ROOT / meta["path"]
    if sha256_file(src) != meta["sha256"]:
        print("[FAIL] active checkpoint sha mismatch")
        return 2
    if meta.get("checkpoint_status") != "final_selected_checkpoint":
        print("[FAIL] active checkpoint not final_selected_checkpoint")
        return 2

    scenarios: list[dict] = []
    scenarios.extend(_run_load_matrix(str(active), src, meta["sha256"], policy_sha))
    scenarios.extend(_run_http_candidate_matrix(str(active), src, meta["sha256"], policy_sha))
    candidate_pass = all(bool(s.get("pass")) for s in scenarios)

    promote_info = {
        "registry_updated": False,
        "historical_registry_mutated": False,
        "defense_checkpoint": None,
        "defense_sha256": None,
        "source_sha256": meta["sha256"],
        "exact_byte_copy": False,
        "registry_cutover_path": None,
        "post_promotion_pass": False,
    }

    if candidate_pass:
        try:
            staged = _stage_exact_defense(
                active=str(active),
                src=src,
                src_sha=meta["sha256"],
                ckpt_dir=ckpt_dir,
                policy_sha=policy_sha,
                lock=lock,
            )
            post = _run_defense_post_promotion_smoke(
                active=str(active),
                staged=staged,
                policy_sha=policy_sha,
            )
            scenarios.extend(post)
            post_pass = all(bool(s.get("pass")) for s in post)
            promote_info["post_promotion_pass"] = post_pass
            promote_info["defense_checkpoint"] = staged["defense_checkpoint"]
            promote_info["defense_sha256"] = staged["defense_sha256"]
            promote_info["exact_byte_copy"] = True
            if post_pass:
                promote_info.update(
                    _cutover_active_pointer(staged=staged, status_dir=status_dir, policy_sha=policy_sha)
                )
            else:
                print("[FAIL] post-promotion smoke failed; active pointer not cut over")
        except Exception as exc:
            scenarios.append(_scenario("defense_stage_or_smoke", False, error=str(exc)))
            print(f"[FAIL] defense stage/smoke: {exc}")

    all_pass = all(bool(s.get("pass")) for s in scenarios) and bool(promote_info.get("registry_updated"))
    report = {
        "schema": "ms_sta_retrain_v3_d4_acceptance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_variant": active,
        "active_experiment_variant": active,
        "source_checkpoint": meta["path"],
        "source_sha256": meta["sha256"],
        "scenarios": scenarios,
        "pass": all_pass,
        "test_matrix_sha256": sha256_file(test_matrix),
        "eval_lock_sha256": sha256_file(lock_path),
        "demo_set_v3": DEMO_V3.relative_to(_ROOT).as_posix(),
        "reuse_historical_d4": False,
        "reuse_e2_demo": False,
        **promote_info,
        "note": (
            "Exact-byte defense copy (defense_sha256 == source_sha256). "
            "Active cutover only after defense post-promotion health/predict smoke PASS."
        ),
    }
    out = status_dir / "D4_ACCEPTANCE_REPORT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": all_pass,
                "out": str(out),
                "active_variant": active,
                "n_scenarios": len(scenarios),
                "exact_byte_copy": promote_info.get("exact_byte_copy"),
                "registry_updated": promote_info.get("registry_updated"),
            },
            indent=2,
        )
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
