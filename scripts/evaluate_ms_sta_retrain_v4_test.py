"""Retrain-v4 retrospective *test* evaluation after EVALUATION_LOCKED.json.

Explicit launch only (orchestrator ``--phase test``). Does not auto-chain from
confirm/lock. Requires ``--confirm-retrospective-test``.

One-shot fail-closed: refuses overwrite of TEST_REPORT / TEST_PER_SAMPLE /
RUN_STATE.phases.test. Resume verifies bit-exact artifacts, finishes staged
promote, or rebuilds report+completion from validated rows (no re-inference).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import load_json, sha256_file
from src.models.ms_sta_features import resolve_pooled_dir, resolve_skel_dir, resolve_temporal_dir
from src.models.ms_sta_wordlevel import build_ms_sta
from src.training.retrain_v3 import final_stem_from_candidate_stem
from src.training.retrain_v3_stats import macro_f1, mcnemar, signer_metrics
from src.training.retrain_v4 import (
    CONFIRM_VARIANTS,
    PRIMARY_SEED,
    exact_id_list_sha256,
    lineage_block,
    load_policy,
    primary_seed_final_sha_by_variant,
    require_exact_str,
    require_freeze_binding,
    require_frozen_v4_policy,
    require_hex64,
    run_key,
    verify_checkpoint_payload_identity,
)
from src.training.retrain_v4_stats import hierarchical_paired_bootstrap_delta_ci
from src.training.train_ms_sta_vsl400 import VSL400MSSTADataset, _collate

REGISTRY_PATH = _ROOT / "data/processed/ms_sta_registry.json"
TEST_VARIANTS = CONFIRM_VARIANTS  # b0,b3,b4,b4r
BINDING_REL = "Docs/results/ms_sta_retrain_v4_policy/POLICY_FREEZE_BINDING.json"


def _norm_signer(value: object) -> str:
    s = str(value).strip()
    if s.startswith("signer_"):
        return s
    if s.isdigit() or s.lstrip("0").isdigit() or s == "0":
        return f"signer_{int(s)}"
    return s


def _rel(path: Path) -> str:
    return path.resolve().relative_to(_ROOT.resolve()).as_posix()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _verify_runtime_feature_dirs(
    policy: dict, *, rgb_pooled_dir: Path, rgb_temporal_dir: Path, skel_dir: Path
) -> None:
    runtime = policy.get("runtime_locks") or {}
    active_skel = str(runtime.get("active_skel_dir") or "").replace("\\", "/")
    active_temp = str(runtime.get("active_temporal_dir") or "").replace("\\", "/")
    active_pooled = str(runtime.get("active_pooled_dir") or "").replace("\\", "/")
    if not active_skel or not active_temp:
        raise ValueError("runtime_locks.active_skel_dir/active_temporal_dir required for test")
    require_exact_str(_rel(skel_dir), active_skel, label="active_skel_dir")
    require_exact_str(_rel(rgb_temporal_dir), active_temp, label="active_temporal_dir")
    if active_pooled:
        require_exact_str(_rel(rgb_pooled_dir), active_pooled, label="active_pooled_dir")


@torch.no_grad()
def _predict_ordered(model, loader, device, variant: str, expected_ids: list[str]) -> dict:
    model.eval()
    by_id: dict[str, dict] = {}
    for rgb, skel, y, batch_vids, rgb_kpm in loader:
        rgb = rgb.to(device)
        y = y.to(device)
        skel_t = skel.to(device) if skel is not None else None
        kpm = rgb_kpm.to(device) if rgb_kpm is not None else None
        if variant in ("b0", "b1"):
            logits = model(rgb=rgb, rgb_key_padding_mask=kpm)
        elif variant == "b2":
            logits = model(skel=skel_t)
        else:
            logits = model(rgb=rgb, skel=skel_t, rgb_key_padding_mask=kpm)
        prob = torch.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)
        t5 = prob.topk(5, dim=-1).indices
        for i, vid in enumerate(batch_vids):
            yi = int(y[i].item())
            pi = int(pred[i].item())
            top5 = [int(x) for x in t5[i].tolist()]
            by_id[str(vid)] = {
                "y_true": yi,
                "y_pred": pi,
                "correct_top1": int(pi == yi),
                "correct_top5": int(yi in top5),
                "top5": top5,
            }
    missing = [vid for vid in expected_ids if vid not in by_id]
    if missing:
        raise RuntimeError(f"missing predictions for {len(missing)} IDs e.g. {missing[:5]}")
    y = np.array([by_id[v]["y_true"] for v in expected_ids], dtype=np.int64)
    p = np.array([by_id[v]["y_pred"] for v in expected_ids], dtype=np.int64)
    t5 = np.array([by_id[v]["top5"] for v in expected_ids], dtype=np.int64)
    c1 = np.array([by_id[v]["correct_top1"] for v in expected_ids], dtype=float)
    c5 = np.array([by_id[v]["correct_top5"] for v in expected_ids], dtype=float)
    return {
        "by_id": by_id,
        "y": y,
        "pred": p,
        "top5": t5,
        "correct_top1": c1,
        "correct_top5": c5,
    }


def _safe_rel(path: Path) -> str:
    try:
        return _rel(path)
    except ValueError:
        return str(path)


def _checkpoints_rel(policy: dict) -> str:
    art = policy.get("artifact_dirs") or {}
    return str(art.get("checkpoints") or "data/processed/ms_sta_retrain_v4").replace(
        "\\", "/"
    ).rstrip("/")


def _expected_ckpt_by_variant_from_lock(
    *,
    status_dir: Path,
    lock: dict,
    policy: dict,
) -> dict[str, str]:
    """Exact {variant → primary-seed final SHA} from SELECTED_CONFIGS + lock (not from rows)."""
    selected_path = status_dir / "SELECTED_CONFIGS.json"
    if not selected_path.is_file():
        raise ValueError("SELECTED_CONFIGS.json missing for rows-only recovery")
    require_exact_str(
        sha256_file(selected_path),
        lock.get("selected_configs_sha256"),
        label="SELECTED_CONFIGS vs lock.selected_configs_sha256",
    )
    if int(lock.get("primary_seed") or -1) != int(PRIMARY_SEED):
        raise ValueError(f"lock.primary_seed must be {PRIMARY_SEED}")
    selected = json.loads(selected_path.read_text(encoding="utf-8")).get("selected") or {}
    finals = dict(lock.get("final_checkpoint_sha256_by_path") or {})
    if len(finals) != 12:
        raise ValueError(f"lock finals n={len(finals)} != 12")
    return primary_seed_final_sha_by_variant(
        selected={str(k): str(v) for k, v in selected.items()},
        final_sha_by_path=finals,
        checkpoints_rel=_checkpoints_rel(policy),
        variants=TEST_VARIANTS,
        primary_seed=PRIMARY_SEED,
    )


def _canonical_signer_by_id(
    *,
    policy: dict,
    test_ids: list[str],
    metadata_path: Path | None = None,
) -> dict[str, str]:
    """Load signer_id for each test sample from SHA-locked metadata."""
    lineage = lineage_block(policy)
    data_locks = policy.get("data_locks") or {}
    meta_rel = lineage.get("metadata") or data_locks.get("metadata") or "metadata.csv"
    meta_sha = lineage.get("metadata_sha256") or data_locks.get("metadata_sha256")
    path = metadata_path if metadata_path is not None else _ROOT / str(meta_rel)
    if not path.is_file():
        raise ValueError(f"metadata missing: {path}")
    if meta_sha:
        require_exact_str(sha256_file(path), meta_sha, label="metadata_sha256")
    meta = pd.read_csv(path)
    df = meta[meta["dataset"] == "VSL400"].copy() if "dataset" in meta.columns else meta.copy()
    df["video_id"] = df["video_id"].astype(str)
    by_id = {
        str(r.video_id): _norm_signer(r.signer_id) for r in df.itertuples(index=False)
    }
    out: dict[str, str] = {}
    for sid in test_ids:
        if sid not in by_id:
            raise ValueError(f"test sample_id {sid} missing from metadata")
        out[sid] = by_id[sid]
    return out


def _clear_staging_ready(status_dir: Path) -> None:
    staging = status_dir / ".test_staging"
    ready = staging / "READY.json"
    try:
        ready.unlink(missing_ok=True)
        if staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()
    except OSError:
        pass


def _finish_staging_promote(
    *,
    status_dir: Path,
    rows_path: Path,
    report_path: Path,
) -> None:
    """Validate READY marker hashes, then complete interrupted staged promote.

    Does **not** delete READY — caller must clear only after completion is verified.
    """
    staging = status_dir / ".test_staging"
    ready = staging / "READY.json"
    if not ready.is_file():
        return
    ready_doc = json.loads(ready.read_text(encoding="utf-8"))
    if ready_doc.get("schema") != "ms_sta_retrain_v4_test_staging_ready_v1":
        raise ValueError(f"bad READY schema: {ready_doc.get('schema')!r}")
    expect_rows = require_hex64(ready_doc.get("rows_sha256"), label="READY.rows_sha256")
    expect_report = require_hex64(ready_doc.get("report_sha256"), label="READY.report_sha256")
    require_hex64(ready_doc.get("payload_sha256"), label="READY.payload_sha256")

    staged_rows = staging / "TEST_PER_SAMPLE.jsonl"
    staged_report = staging / "TEST_REPORT.json"

    if rows_path.is_file():
        require_exact_str(sha256_file(rows_path), expect_rows, label="published rows vs READY")
    elif staged_rows.is_file():
        require_exact_str(sha256_file(staged_rows), expect_rows, label="staged rows vs READY")
        os.replace(staged_rows, rows_path)
        require_exact_str(sha256_file(rows_path), expect_rows, label="promoted rows vs READY")
    else:
        raise ValueError("READY present but rows missing from publish and staging")

    if report_path.is_file():
        require_exact_str(
            sha256_file(report_path), expect_report, label="published report vs READY"
        )
    elif staged_report.is_file():
        require_exact_str(
            sha256_file(staged_report), expect_report, label="staged report vs READY"
        )
        os.replace(staged_report, report_path)
        require_exact_str(
            sha256_file(report_path), expect_report, label="promoted report vs READY"
        )
    # else: rows-only crash after promote of rows — report rebuild path handles it.


def _verify_existing_artifacts(
    *,
    report_path: Path,
    rows_path: Path,
    completion_path: Path,
    policy_sha: str,
    lock_sha: str,
    binding_sha: str,
    state: dict,
    test_ids: list[str],
    id_sha: str,
    split_sha: str,
    lock: dict,
    policy: dict,
    num_classes: int,
    n_boot: int,
    bootstrap_seed: int,
    status_dir: Path,
    expected_signer_by_id: dict[str, str],
) -> int:
    """Resume path: bit-exact verify / rows-only rebuild; never re-run inference."""
    try:
        _finish_staging_promote(
            status_dir=status_dir, rows_path=rows_path, report_path=report_path
        )
    except ValueError as exc:
        print(f"[FAIL] staging READY verify/promote failed: {exc}")
        return 2

    if rows_path.is_file() and not report_path.is_file():
        rc = _rebuild_report_from_rows_only(
            report_path=report_path,
            rows_path=rows_path,
            completion_path=completion_path,
            policy_sha=policy_sha,
            lock_sha=lock_sha,
            binding_sha=binding_sha,
            test_ids=test_ids,
            id_sha=id_sha,
            split_sha=split_sha,
            lock=lock,
            policy=policy,
            num_classes=num_classes,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed,
            status_dir=status_dir,
            expected_signer_by_id=expected_signer_by_id,
        )
        if rc == 0:
            _clear_staging_ready(status_dir)
        return rc

    if not report_path.is_file() or not rows_path.is_file():
        print("[FAIL] --resume requires TEST_PER_SAMPLE.jsonl (and report, or rebuildable rows)")
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_sha = sha256_file(report_path)
    rows_sha = sha256_file(rows_path)
    try:
        bindings = report.get("bindings") or {}
        require_exact_str(bindings.get("policy_sha256"), policy_sha, label="report.policy_sha256")
        require_exact_str(
            bindings.get("evaluation_locked_sha256"),
            lock_sha,
            label="report.evaluation_locked_sha256",
        )
        require_exact_str(
            bindings.get("freeze_binding_sha256"),
            binding_sha,
            label="report.freeze_binding_sha256",
        )
        art = report.get("artifacts") or {}
        require_exact_str(
            art.get("test_per_sample_jsonl_sha256"),
            rows_sha,
            label="report.artifacts.test_per_sample_jsonl_sha256",
        )
        payload = dict(report)
        art2 = dict(payload.get("artifacts") or {})
        payload_sha_expected = art2.pop("test_report_payload_sha256", None)
        payload["artifacts"] = art2
        import hashlib

        payload_sha = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        require_exact_str(
            payload_sha_expected, payload_sha, label="report.artifacts.test_report_payload_sha256"
        )

        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("schema") != "ms_sta_retrain_v4_test_completion_v1":
                raise ValueError(f"bad TEST_COMPLETION schema: {completion.get('schema')!r}")
            require_exact_str(
                completion.get("policy_sha256"), policy_sha, label="completion.policy_sha256"
            )
            require_exact_str(
                completion.get("evaluation_locked_sha256"),
                lock_sha,
                label="completion.evaluation_locked_sha256",
            )
            require_exact_str(
                completion.get("freeze_binding_sha256"),
                binding_sha,
                label="completion.freeze_binding_sha256",
            )
            require_exact_str(
                completion.get("test_report_sha256"),
                report_sha,
                label="completion.test_report_sha256",
            )
            require_exact_str(
                completion.get("test_per_sample_jsonl_sha256"),
                rows_sha,
                label="completion.test_per_sample_jsonl_sha256",
            )
            require_exact_str(
                completion.get("test_report_payload_sha256"),
                payload_sha,
                label="completion.test_report_payload_sha256",
            )
        else:
            _write_test_completion(
                completion_path,
                policy_sha=policy_sha,
                lock_sha=lock_sha,
                binding_sha=binding_sha,
                report_sha=report_sha,
                rows_sha=rows_sha,
                payload_sha=payload_sha,
            )

        phase = (state.get("phases") or {}).get("test") or {}
        if phase.get("test_report_sha256"):
            require_exact_str(
                phase.get("test_report_sha256"),
                report_sha,
                label="phases.test.test_report_sha256",
            )
    except ValueError as exc:
        print(f"[FAIL] resume bit-exact verify failed: {exc}")
        return 2
    _clear_staging_ready(status_dir)
    print(
        json.dumps(
            {
                "ok": True,
                "resume_verified": True,
                "test_report_sha256": report_sha,
                "test_per_sample_jsonl_sha256": rows_sha,
                "test_completion": str(completion_path),
                "note": "No re-inference; artifacts (+completion) bit-exact.",
            },
            indent=2,
        )
    )
    return 0


def _rebuild_report_from_rows_only(
    *,
    report_path: Path,
    rows_path: Path,
    completion_path: Path,
    policy_sha: str,
    lock_sha: str,
    binding_sha: str,
    test_ids: list[str],
    id_sha: str,
    split_sha: str,
    lock: dict,
    policy: dict,
    num_classes: int,
    n_boot: int,
    bootstrap_seed: int,
    status_dir: Path,
    expected_signer_by_id: dict[str, str],
) -> int:
    """Recover crash after rows promote / before report: rebuild without inference."""
    try:
        ckpt_by_variant = _expected_ckpt_by_variant_from_lock(
            status_dir=status_dir, lock=lock, policy=policy
        )
        preds, signers = _load_preds_from_test_rows(
            rows_path,
            test_ids=test_ids,
            policy_sha=policy_sha,
            expected_ckpt_by_variant=ckpt_by_variant,
            expected_signer_by_id=expected_signer_by_id,
            num_classes=num_classes,
        )
        used_ckpts = {
            v: {"sha256": ckpt_by_variant[v], "seed": PRIMARY_SEED, "rebuilt_from_rows": True}
            for v in TEST_VARIANTS
        }
        rows_sha = sha256_file(rows_path)
        report, payload_sha = _build_test_report_dict(
            preds=preds,
            test_ids=test_ids,
            signers=signers,
            num_classes=num_classes,
            id_sha=id_sha,
            claim_label=(policy.get("test_protocol") or {}).get("label"),
            used_ckpts=used_ckpts,
            policy_sha=policy_sha,
            split_sha=split_sha,
            git_revision=lock.get("git_revision"),
            lock_sha=lock_sha,
            binding_sha=binding_sha,
            rows_rel=_safe_rel(rows_path),
            rows_sha=rows_sha,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed,
        )
        report["artifacts"]["rebuilt_from_rows_only"] = True
        import hashlib

        tmp = dict(report)
        art = dict(tmp.get("artifacts") or {})
        art.pop("test_report_payload_sha256", None)
        tmp["artifacts"] = art
        payload_sha = hashlib.sha256(
            json.dumps(tmp, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        report["artifacts"]["test_report_payload_sha256"] = payload_sha
        _atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        report_sha = sha256_file(report_path)
        _write_test_completion(
            completion_path,
            policy_sha=policy_sha,
            lock_sha=lock_sha,
            binding_sha=binding_sha,
            report_sha=report_sha,
            rows_sha=rows_sha,
            payload_sha=payload_sha,
        )
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        print(f"[FAIL] rows-only resume rebuild failed: {exc}")
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "resume_verified": True,
                "rebuilt_from_rows_only": True,
                "test_report_sha256": report_sha,
                "test_per_sample_jsonl_sha256": rows_sha,
                "note": "No re-inference; report+completion rebuilt from validated rows.",
            },
            indent=2,
        )
    )
    return 0


def _write_test_completion(
    path: Path,
    *,
    policy_sha: str,
    lock_sha: str,
    binding_sha: str,
    report_sha: str,
    rows_sha: str,
    payload_sha: str,
) -> None:
    payload = {
        "schema": "ms_sta_retrain_v4_test_completion_v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": policy_sha,
        "evaluation_locked_sha256": lock_sha,
        "freeze_binding_sha256": binding_sha,
        "test_report_sha256": report_sha,
        "test_report_payload_sha256": payload_sha,
        "test_per_sample_jsonl_sha256": rows_sha,
        "immutable": True,
        "forbid_reinference_on_resume": True,
    }
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _require_class_id(value: object, *, label: str, num_classes: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} not int: {value!r}") from exc
    if v < 0 or v >= int(num_classes):
        raise ValueError(f"{label}={v} out of range [0,{num_classes - 1}]")
    return v


def _load_preds_from_test_rows(
    rows_path: Path,
    *,
    test_ids: list[str],
    policy_sha: str,
    expected_ckpt_by_variant: dict[str, str],
    expected_signer_by_id: dict[str, str],
    num_classes: int,
) -> tuple[dict[str, dict], list[str]]:
    """Validate TEST_PER_SAMPLE.jsonl and rebuild in-memory preds (no inference)."""
    raw_rows: list[dict] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw_rows.append(json.loads(line))
    expected_n = len(test_ids) * len(TEST_VARIANTS)
    if len(raw_rows) != expected_n:
        raise ValueError(f"rows n={len(raw_rows)} != expected {expected_n}")

    by_var: dict[str, dict[str, dict]] = {v: {} for v in TEST_VARIANTS}
    for r in raw_rows:
        variant = str(r.get("variant") or "")
        if variant not in TEST_VARIANTS:
            raise ValueError(f"unexpected variant in rows: {variant!r}")
        require_exact_str(r.get("split"), "test", label="row.split")
        if int(r.get("seed", -1)) != int(PRIMARY_SEED):
            raise ValueError(f"row seed != PRIMARY_SEED for {r.get('sample_id')}")
        require_exact_str(r.get("policy_sha256"), policy_sha, label="row.policy_sha256")
        require_exact_str(
            r.get("final_checkpoint_sha256"),
            expected_ckpt_by_variant[variant],
            label=f"row.final_checkpoint_sha256[{variant}]",
        )
        sid = str(r.get("sample_id") or "")
        if not sid:
            raise ValueError("row missing sample_id")
        if sid not in expected_signer_by_id:
            raise ValueError(f"row sample_id {sid} not in canonical metadata signer map")
        require_exact_str(
            _norm_signer(r.get("signer_id")),
            expected_signer_by_id[sid],
            label=f"row.signer_id[{sid}]",
        )
        if "top5" not in r or not isinstance(r.get("top5"), list) or len(r["top5"]) != 5:
            raise ValueError(f"row.top5 invalid for {sid}")
        y_true = _require_class_id(r.get("y_true"), label="y_true", num_classes=num_classes)
        y_pred = _require_class_id(r.get("y_pred"), label="y_pred", num_classes=num_classes)
        top5 = [
            _require_class_id(x, label="top5[i]", num_classes=num_classes) for x in r["top5"]
        ]
        if len(set(top5)) != 5:
            raise ValueError(f"row.top5 must be 5 distinct class IDs for {sid}: {top5}")
        expect_t1 = int(y_pred == y_true)
        expect_t5 = int(y_true in top5)
        if int(r.get("correct_top1", -1)) != expect_t1:
            raise ValueError(f"correct_top1 mismatch for {sid}")
        if int(r.get("correct_top5", -1)) != expect_t5:
            raise ValueError(f"correct_top5 mismatch for {sid}")
        if sid in by_var[variant]:
            raise ValueError(f"duplicate sample_id {sid} for {variant}")
        by_var[variant][sid] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "correct_top1": expect_t1,
            "correct_top5": expect_t5,
            "top5": top5,
            "signer_id": expected_signer_by_id[sid],
        }

    preds: dict[str, dict] = {}
    signers_ref: list[str] | None = None
    for variant in TEST_VARIANTS:
        if set(by_var[variant]) != set(test_ids):
            raise ValueError(f"{variant} ID set mismatch vs test_ids")
        signers = [by_var[variant][sid]["signer_id"] for sid in test_ids]
        if signers_ref is None:
            signers_ref = signers
        elif signers != signers_ref:
            raise ValueError(f"signer vector mismatch for {variant}")
        y = np.array([by_var[variant][sid]["y_true"] for sid in test_ids], dtype=np.int64)
        p = np.array([by_var[variant][sid]["y_pred"] for sid in test_ids], dtype=np.int64)
        t5 = np.array([by_var[variant][sid]["top5"] for sid in test_ids], dtype=np.int64)
        c1 = np.array(
            [by_var[variant][sid]["correct_top1"] for sid in test_ids], dtype=float
        )
        c5 = np.array(
            [by_var[variant][sid]["correct_top5"] for sid in test_ids], dtype=float
        )
        preds[variant] = {
            "by_id": {sid: by_var[variant][sid] for sid in test_ids},
            "y": y,
            "pred": p,
            "top5": t5,
            "correct_top1": c1,
            "correct_top5": c5,
        }
    assert signers_ref is not None
    y_ref = preds["b0"]["y"]
    for variant in TEST_VARIANTS:
        if not np.array_equal(preds[variant]["y"], y_ref):
            raise ValueError(f"y_true mismatch for {variant}")
    return preds, signers_ref


def _build_test_report_dict(
    *,
    preds: dict[str, dict],
    test_ids: list[str],
    signers: list[str],
    num_classes: int,
    id_sha: str,
    claim_label: object,
    used_ckpts: dict[str, dict],
    policy_sha: str,
    split_sha: str,
    git_revision: object,
    lock_sha: str,
    binding_sha: str,
    rows_rel: str,
    rows_sha: str,
    n_boot: int,
    bootstrap_seed: int,
) -> tuple[dict, str]:
    """Deterministic report assembly from preds (shared by fresh + rows-only rebuild)."""
    import hashlib

    y_ref = preds["b0"]["y"]
    summary: dict[str, dict] = {}
    for variant in TEST_VARIANTS:
        summary[variant] = {
            "top1": float(preds[variant]["correct_top1"].mean() * 100.0),
            "top5": float(preds[variant]["correct_top5"].mean() * 100.0),
            "macro_f1": float(macro_f1(y_ref, preds[variant]["pred"], num_classes)),
            "n": len(test_ids),
        }
    c1_b3 = preds["b3"]["correct_top1"]
    c1_b4r = preds["b4r"]["correct_top1"]
    c5_b3 = preds["b3"]["correct_top5"]
    c5_b4r = preds["b4r"]["correct_top5"]
    paired_top1 = hierarchical_paired_bootstrap_delta_ci(
        c1_b3,
        c1_b4r,
        signers,
        n_boot=int(n_boot),
        seed=int(bootstrap_seed),
        label_a="b3",
        label_b="b4r",
    )
    paired_top1["unit_of_analysis"] = "primary_seed_per_sample_correctness"
    paired_top5 = hierarchical_paired_bootstrap_delta_ci(
        c5_b3,
        c5_b4r,
        signers,
        n_boot=int(n_boot),
        seed=int(bootstrap_seed),
        label_a="b3",
        label_b="b4r",
    )
    paired_top5["unit_of_analysis"] = "primary_seed_per_sample_top5_hit"
    per_signer = signer_metrics(y_ref, preds["b4r"]["pred"], signers)
    per_signer_b3 = signer_metrics(y_ref, preds["b3"]["pred"], signers)
    signer_delta = {}
    for s, row in (per_signer.get("per_signer") or {}).items():
        b3 = float((per_signer_b3.get("per_signer") or {}).get(s, {}).get("top1") or 0.0)
        b4r = float(row.get("top1") or 0.0)
        signer_delta[s] = {
            "n": int(row.get("n") or 0),
            "b3_top1": b3,
            "b4r_top1": b4r,
            "delta_top1_pp": float(b4r - b3),
        }
    report: dict = {
        "schema": "ms_sta_retrain_v4_test_report_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_label": claim_label,
        "n_test": len(test_ids),
        "exact_test_id_sha256": id_sha,
        "primary_seed": PRIMARY_SEED,
        "variants": list(TEST_VARIANTS),
        "summary": summary,
        "b4r_vs_b3": {
            "delta_top1_pp": float(summary["b4r"]["top1"] - summary["b3"]["top1"]),
            "delta_top5_pp": float(summary["b4r"]["top5"] - summary["b3"]["top5"]),
            "delta_macro_f1_pp": float(summary["b4r"]["macro_f1"] - summary["b3"]["macro_f1"]),
            "paired_hierarchical_bootstrap_top1": paired_top1,
            "paired_hierarchical_bootstrap_top5": paired_top5,
            "mcnemar_exact_top1": mcnemar(c1_b3.astype(bool), c1_b4r.astype(bool)),
            "mcnemar_exact_top5": mcnemar(c5_b3.astype(bool), c5_b4r.astype(bool)),
            "note": "Retrospective confirmation only; does not reopen architecture selection.",
        },
        "per_signer_b3_b4r_delta": signer_delta,
        "checkpoints": used_ckpts,
        "bindings": {
            "policy_sha256": policy_sha,
            "split_manifest_sha256": split_sha,
            "git_revision": git_revision,
            "evaluation_locked_sha256": lock_sha,
            "freeze_binding_sha256": binding_sha,
            "evaluator": "scripts/evaluate_ms_sta_retrain_v4_test.py",
            "canonical_producer": True,
            "one_shot": True,
        },
        "artifacts": {
            "test_per_sample_jsonl": rows_rel,
            "test_per_sample_jsonl_sha256": rows_sha,
        },
        "note": "Retrospective Path-B test; does not reopen validation gates.",
    }
    payload_sha = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    report["artifacts"]["test_report_payload_sha256"] = payload_sha
    return report, payload_sha


def _publish_staged_test_artifacts(
    *,
    status_dir: Path,
    rows_path: Path,
    report_path: Path,
    completion_path: Path,
    rows_text: str,
    report: dict,
    policy_sha: str,
    lock_sha: str,
    binding_sha: str,
    payload_sha: str,
) -> tuple[str, str]:
    """Stage rows+report, promote both, then write completion (crash-safe)."""
    staging = status_dir / ".test_staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged_rows = staging / "TEST_PER_SAMPLE.jsonl"
    staged_report = staging / "TEST_REPORT.json"
    ready = staging / "READY.json"
    _atomic_write_text(staged_rows, rows_text)
    _atomic_write_text(staged_report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    ready_payload = {
        "schema": "ms_sta_retrain_v4_test_staging_ready_v1",
        "rows_sha256": sha256_file(staged_rows),
        "report_sha256": sha256_file(staged_report),
        "payload_sha256": payload_sha,
    }
    _atomic_write_text(ready, json.dumps(ready_payload, ensure_ascii=False, indent=2) + "\n")
    # Promote both only after READY exists.
    os.replace(staged_rows, rows_path)
    os.replace(staged_report, report_path)
    rows_sha = sha256_file(rows_path)
    report_sha = sha256_file(report_path)
    _write_test_completion(
        completion_path,
        policy_sha=policy_sha,
        lock_sha=lock_sha,
        binding_sha=binding_sha,
        report_sha=report_sha,
        rows_sha=rows_sha,
        payload_sha=payload_sha,
    )
    # Verify completion against promoted artifacts before clearing READY.
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    require_exact_str(
        completion.get("test_report_sha256"), report_sha, label="completion.test_report_sha256"
    )
    require_exact_str(
        completion.get("test_per_sample_jsonl_sha256"),
        rows_sha,
        label="completion.test_per_sample_jsonl_sha256",
    )
    require_exact_str(
        completion.get("test_report_payload_sha256"),
        payload_sha,
        label="completion.test_report_payload_sha256",
    )
    _clear_staging_ready(status_dir)
    return rows_sha, report_sha


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--eval-lock", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--confirm-retrospective-test", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Verify existing test artifacts bit-exact; refuse re-inference",
    )
    args = ap.parse_args()

    if not args.confirm_retrospective_test:
        print("[FAIL] refuse test without --confirm-retrospective-test")
        return 2

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    lock_path = args.eval_lock if args.eval_lock.is_absolute() else _ROOT / args.eval_lock
    policy = load_policy(policy_path)
    try:
        require_frozen_v4_policy(policy, root=_ROOT)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    if not lock_path.is_file():
        print(f"[FAIL] missing EVALUATION_LOCKED: {lock_path}")
        return 2

    art = policy.get("artifact_dirs") or {}
    status_dir = _ROOT / str(art.get("status") or "logs/ms_sta_retrain_v4")
    state_path = status_dir / "RUN_STATE.json"
    report_path = status_dir / "TEST_REPORT.json"
    rows_path = status_dir / "TEST_PER_SAMPLE.jsonl"
    completion_path = status_dir / "TEST_COMPLETION.json"
    if not state_path.is_file():
        print("[FAIL] missing RUN_STATE.json")
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))

    # Identity checks run for both fresh and --resume paths (fail-closed CLI).
    policy_sha = sha256_file(policy_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha = sha256_file(lock_path)
    lineage = lineage_block(policy)
    split_sha = require_hex64(
        lineage.get("split_manifest_sha256"), label="lineage.split_manifest_sha256"
    )

    binding_path = _ROOT / BINDING_REL
    if not binding_path.is_file():
        print(f"[FAIL] missing required freeze binding: {BINDING_REL}")
        return 2
    binding_sha = sha256_file(binding_path)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()

    try:
        require_exact_str(lock.get("policy_sha256"), policy_sha, label="lock.policy_sha256")
        require_exact_str(
            lock.get("split_manifest_sha256"), split_sha, label="lock.split_manifest_sha256"
        )
        if int(lock.get("primary_seed") or -1) != int(PRIMARY_SEED):
            raise ValueError(f"lock.primary_seed must be {PRIMARY_SEED}")
        if not lock.get("architecture_gate_pass"):
            raise ValueError("lock.architecture_gate_pass is not true")
        if not state.get("evaluation_locked"):
            raise ValueError("RUN_STATE.evaluation_locked is not true")
        require_exact_str(
            state.get("evaluation_locked_sha256"),
            lock_sha,
            label="RUN_STATE.evaluation_locked_sha256",
        )
        require_exact_str(
            lock.get("freeze_binding_sha256"),
            binding_sha,
            label="lock.freeze_binding_sha256",
        )
        require_freeze_binding(
            json.loads(binding_path.read_text(encoding="utf-8")),
            root=_ROOT,
            policy_path=policy_path,
            policy_sha=policy_sha,
            head_rev=head,
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    # Split / labels needed for both --resume (rows-only rebuild) and fresh inference.
    split_path = _ROOT / str(lineage["split_manifest"])
    if sha256_file(split_path) != split_sha:
        print("[FAIL] split sha mismatch")
        return 2
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = [str(x) for x in split["test_ids"]]
    n_test = int(lineage.get("n_test") or split.get("n_test") or len(test_ids))
    if len(test_ids) != n_test:
        print(f"[FAIL] test_ids n={len(test_ids)} != n_test={n_test}")
        return 2
    id_sha = exact_id_list_sha256(sorted(test_ids))
    labels = load_json(_ROOT / "data/processed/vsl400_labels.json")
    num_classes = int(labels["num_classes"])
    try:
        expected_signer_by_id = _canonical_signer_by_id(policy=policy, test_ids=test_ids)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    if args.resume:
        return _verify_existing_artifacts(
            report_path=report_path,
            rows_path=rows_path,
            completion_path=completion_path,
            policy_sha=policy_sha,
            lock_sha=lock_sha,
            binding_sha=binding_sha,
            state=state,
            test_ids=test_ids,
            id_sha=id_sha,
            split_sha=split_sha,
            lock=lock,
            policy=policy,
            num_classes=num_classes,
            n_boot=int(args.bootstrap),
            bootstrap_seed=int(args.bootstrap_seed),
            status_dir=status_dir,
            expected_signer_by_id=expected_signer_by_id,
        )

    # One-shot: refuse any prior test artifacts / phase / completion / incomplete staging.
    staging_dir = status_dir / ".test_staging"
    if (
        report_path.is_file()
        or rows_path.is_file()
        or completion_path.is_file()
        or (state.get("phases") or {}).get("test")
        or (staging_dir / "READY.json").is_file()
    ):
        print(
            "[FAIL] test artifacts or RUN_STATE.phases.test already exist; "
            "refuse overwrite (use --resume to verify bit-exact only)"
        )
        return 2

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[FAIL] CUDA unavailable for retrospective test")
        return 2

    gloss_to_id = labels["gloss_to_id"]
    signer_by_id = expected_signer_by_id
    signers = [signer_by_id[v] for v in test_ids]
    # Keep a DataFrame for the dataset loader (features keyed by video_id).
    meta = pd.read_csv(_ROOT / "metadata.csv")
    df = meta[meta["dataset"] == "VSL400"].copy()
    df["video_id"] = df["video_id"].astype(str)

    ckpt_dir = _ROOT / str(art.get("checkpoints") or "data/processed/ms_sta_retrain_v4")
    selected = json.loads((status_dir / "SELECTED_CONFIGS.json").read_text(encoding="utf-8"))
    configs = selected["selected"]
    if set(configs) != set(TEST_VARIANTS):
        print(f"[FAIL] selected variants {sorted(configs)} != {list(TEST_VARIANTS)}")
        return 2

    locked_finals = lock.get("final_checkpoint_sha256_by_path") or {}
    try:
        require_exact_str(
            sha256_file(status_dir / "SELECTED_CONFIGS.json"),
            lock.get("selected_configs_sha256"),
            label="SELECTED_CONFIGS vs lock",
        )
        expected_primary = primary_seed_final_sha_by_variant(
            selected={str(k): str(v) for k, v in configs.items()},
            final_sha_by_path=dict(locked_finals),
            checkpoints_rel=_checkpoints_rel(policy),
            variants=TEST_VARIANTS,
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    reg = load_json(REGISTRY_PATH) if REGISTRY_PATH.is_file() else {}
    rgb_pooled_dir = resolve_pooled_dir(_ROOT, reg)
    rgb_temporal_dir = resolve_temporal_dir(_ROOT, reg)
    skel_dir = resolve_skel_dir(_ROOT, reg)
    try:
        _verify_runtime_feature_dirs(
            policy,
            rgb_pooled_dir=rgb_pooled_dir,
            rgb_temporal_dir=rgb_temporal_dir,
            skel_dir=skel_dir,
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    device = torch.device(args.device)
    preds: dict[str, dict] = {}
    used_ckpts: dict[str, dict] = {}

    for variant in TEST_VARIANTS:
        cid = configs[variant]
        stem = run_key(
            {
                "phase": "confirm",
                "variant": variant,
                "config_id": cid,
                "seed": PRIMARY_SEED,
                "skel_temporal_stride": 1,
            }
        )
        final_path = ckpt_dir / f"{final_stem_from_candidate_stem(stem)}.pth"
        if not final_path.is_file():
            print(f"[FAIL] missing final {final_path}")
            return 2
        final_sha = sha256_file(final_path)
        rel = _rel(final_path)
        try:
            require_exact_str(
                expected_primary[variant], final_sha, label=f"primary final sha {variant}"
            )
            require_exact_str(
                locked_finals.get(rel), final_sha, label=f"lock final sha {variant}"
            )
            payload = torch.load(final_path, map_location="cpu", weights_only=False)
            verify_checkpoint_payload_identity(
                payload,
                phase="confirm",
                variant=variant,
                config_id=str(cid),
                seed=int(PRIMARY_SEED),
                expected_policy_sha=policy_sha,
                expected_split_sha=split_sha,
                expected_git_rev=str(lock.get("git_revision") or ""),
                require_final=True,
            )
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 2

        residual_init = payload.get("residual_init") or (
            "zero_delta_head" if variant == "b4r" else "none"
        )
        model = build_ms_sta(
            variant,
            num_classes=num_classes,
            residual_init=residual_init if variant == "b4r" else "zero_delta_head",
        ).to(device)
        model.load_state_dict(payload["model"], strict=True)
        require_skel = variant in ("b2", "b3", "b4", "b4r", "b5")
        require_temporal = variant in ("b1", "b3", "b4", "b4r", "b5")
        if variant == "b0":
            require_skel = False
            require_temporal = False
        ds = VSL400MSSTADataset(
            df,
            gloss_to_id,
            require_skel=require_skel,
            require_temporal_rgb=require_temporal,
            rgb_pooled_dir=rgb_pooled_dir,
            rgb_temporal_dir=rgb_temporal_dir,
            skel_dir=skel_dir,
            video_ids=test_ids,
        )
        if len(ds) != len(test_ids):
            print(f"[FAIL] missing features for test IDs: got {len(ds)} / {len(test_ids)}")
            return 2
        loader = DataLoader(
            ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=_collate
        )
        try:
            preds[variant] = _predict_ordered(model, loader, device, variant, test_ids)
        except Exception as exc:
            print(f"[FAIL] predict {variant}: {exc}")
            return 2
        used_ckpts[variant] = {"path": rel, "sha256": final_sha, "seed": PRIMARY_SEED}

    y_ref = preds["b0"]["y"]
    for variant in TEST_VARIANTS:
        if not np.array_equal(preds[variant]["y"], y_ref):
            print(f"[FAIL] y_true mismatch for {variant}")
            return 2

    lines: list[str] = []
    for sid in test_ids:
        for variant in TEST_VARIANTS:
            row = preds[variant]["by_id"][sid]
            lines.append(
                json.dumps(
                    {
                        "sample_id": sid,
                        "signer_id": signer_by_id[sid],
                        "seed": int(PRIMARY_SEED),
                        "variant": variant,
                        "split": "test",
                        "y_true": row["y_true"],
                        "y_pred": row["y_pred"],
                        "correct_top1": row["correct_top1"],
                        "correct_top5": row["correct_top5"],
                        "top5": row["top5"],
                        "final_checkpoint_sha256": used_ckpts[variant]["sha256"],
                        "policy_sha256": policy_sha,
                    },
                    ensure_ascii=False,
                )
            )
    rows_text = "\n".join(lines) + "\n"
    # Precompute rows sha from staged bytes so report can embed it before promote.
    import hashlib

    rows_sha = hashlib.sha256(rows_text.encode("utf-8")).hexdigest()
    report, payload_sha = _build_test_report_dict(
        preds=preds,
        test_ids=test_ids,
        signers=signers,
        num_classes=num_classes,
        id_sha=id_sha,
        claim_label=(policy.get("test_protocol") or {}).get("label"),
        used_ckpts=used_ckpts,
        policy_sha=policy_sha,
        split_sha=split_sha,
        git_revision=lock.get("git_revision"),
        lock_sha=lock_sha,
        binding_sha=binding_sha,
        rows_rel=_safe_rel(rows_path),
        rows_sha=rows_sha,
        n_boot=int(args.bootstrap),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    rows_sha, report_file_sha = _publish_staged_test_artifacts(
        status_dir=status_dir,
        rows_path=rows_path,
        report_path=report_path,
        completion_path=completion_path,
        rows_text=rows_text,
        report=report,
        policy_sha=policy_sha,
        lock_sha=lock_sha,
        binding_sha=binding_sha,
        payload_sha=payload_sha,
    )
    summary = report["summary"]
    delta_top1 = float(summary["b4r"]["top1"] - summary["b3"]["top1"])
    delta_top5 = float(summary["b4r"]["top5"] - summary["b3"]["top5"])
    delta_f1 = float(summary["b4r"]["macro_f1"] - summary["b3"]["macro_f1"])

    print(
        json.dumps(
            {
                "ok": True,
                "out": _rel(report_path),
                "completion": _rel(completion_path),
                "n_test": len(test_ids),
                "b3_top1": summary["b3"]["top1"],
                "b4r_top1": summary["b4r"]["top1"],
                "delta_b4r_minus_b3_top1_pp": delta_top1,
                "delta_top5_pp": delta_top5,
                "delta_macro_f1_pp": delta_f1,
                "test_report_payload_sha256": payload_sha,
                "test_report_file_sha256": report_file_sha,
                "test_per_sample_jsonl_sha256": rows_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
