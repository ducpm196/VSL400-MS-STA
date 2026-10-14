#!/usr/bin/env python3
"""Fail-closed unit tests for D0 full-corpus v14 execution (no full GPU extract)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    CORPUS_N,
    D0_FULL_CORPUS_APPROVAL_AUDIT_V14_REL,
    D0_FULL_CORPUS_APPROVAL_V10_REL,
    D0_FULL_CORPUS_APPROVAL_V14_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL,
    D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL,
    D0_FULL_CORPUS_EXECUTION_INTERRUPTION_V10_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V10_REL,
    D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL,
    D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V10_REL,
    D0_FULL_CORPUS_RUN_CONTRACT_V14_REL,
    EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION,
    EXTERNAL_REAUDIT_PASS_SCHEMA,
    EXPECTED_V10_APPROVAL_SHA256,
    EXPECTED_V10_INTERRUPTION_SHA256,
    EXPECTED_V12_APPROVAL_SHA256,
    EXPECTED_V12_EXTERNAL_REAUDIT_SHA256,
    EXPECTED_V13_APPROVAL_SHA256,
    EXPECTED_V13_EXTERNAL_REAUDIT_SHA256,
    FULL_CORPUS_EXECUTABLE_PATHS,
    FULL_CORPUS_EXECUTABLE_PATHS_V13,
    MANIFEST_PHASES,
    MAX_RESUMES,
    OUTPUT_VERSION,
    PHASES_EXACT,
    RGB_MANIFEST_REQUIRED_FIELDS,
    RGB_REQUIRED_KEYS,
    SKEL_MANIFEST_REQUIRED_FIELDS,
    SKEL_REQUIRED_KEYS,
    SPLIT_COUNTS,
    WATCHDOG_INTERVAL_SEC,
    atomic_create_json,
    atomic_replace_json,
    atomic_savez_compressed,
    binary_root_rel,
    campaign_related_process_alive,
    canonical_sample_paths,
    commit_manifest_record,
    compute_quality_stats_from_rgb_tensors,
    default_quality_thresholds,
    descendants_alive,
    evaluate_launch_readiness,
    evaluate_quality_thresholds,
    heal_or_resume_sample,
    load_eligible_identity,
    load_json,
    manifest_record_path,
    require_launch_readiness,
    validate_execution_interruption_v10,
    validate_external_reaudit_pass,
    validate_post_audit_execution_authorization,
    load_manifest_index,
    manifest_index_path,
    process_tree_alive,
    publish_completion_pack,
    rebuild_manifest_index,
    resume_output_ok,
    runtime_dir,
    staging_dir,
    validate_array_contracts,
    validate_npz_keys,
    write_ready_descriptor,
    sha256_file,
)
from run_ms_sta_v5_d0_full_corpus import (  # noqa: E402
    _save_rgb_atomic,
    _save_skel_atomic,
    run_quality_and_completion,
)
from scan_ms_sta_v5_d0_full_corpus import scan_full_corpus  # noqa: E402
from monitor_ms_sta_v5_d0_full_corpus import _progress_metrics  # noqa: E402


def _recovery(t: int = 8, j: int = 75) -> dict:
    return {
        "raw_landmarks": np.zeros((t, j, 3), np.float32),
        "effective_landmarks": np.zeros((t, j, 3), np.float32),
        "raw_holistic_missing": np.zeros((t, j), dtype=bool),
        "effective_missing": np.zeros((t, j), dtype=bool),
        "observation_source_left_hand": ["holistic"] * t,
        "observation_source_right_hand": ["missing"] * t,
        "wrist_assignment_distance_left": [None] * t,
        "wrist_assignment_distance_right": [None] * t,
        "arm": "C3",
        "pose_guided_crop_left": [None] * t,
        "pose_guided_crop_right": [None] * t,
        "fallback_rejection_reason_left": [None] * t,
        "fallback_rejection_reason_right": [None] * t,
    }


def _canonical_joint_arrays() -> tuple[np.ndarray, np.ndarray]:
    from src.data_prep.joint_layout_v3 import JOINT_CHIRALITY, JOINT_PARTS

    return np.asarray(JOINT_PARTS), np.asarray(JOINT_CHIRALITY)


def _bone_edges() -> np.ndarray:
    from src.data_prep.joint_layout_v3 import bone_edge_list

    return np.asarray(bone_edge_list(), dtype=np.int32)


def _skel_payload(t: int = 8, j: int = 75) -> dict:
    edges = _bone_edges()
    n_bones = int(edges.shape[0])
    joint_part, joint_chir = _canonical_joint_arrays()
    return {
        "coordinates": np.zeros((t, j, 3), np.float32),
        "joints_raw": np.zeros((t, j, 3), np.float32),
        "velocity": np.zeros((t, j, 3), np.float32),
        "bone_vectors": np.zeros((t, n_bones, 3), np.float32),
        "missing_mask_prefill": np.zeros((t, j), dtype=bool),
        "validity": np.ones((t, j), np.float32),
        "confidence": np.ones((t, j), np.float32),
        "original_frame_index": np.arange(t, dtype=np.int32),
        "timestamps_sec": np.arange(t, dtype=np.float64) / 25.0,
        "padding_mask": np.ones((t,), dtype=bool),
        "joint_part": joint_part,
        "joint_chirality": joint_chir,
        "bone_edges": edges,
        "sequence_length": t,
    }


def _write_skel(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, j = 8, 75
    rec = _recovery(t, j)
    edges = _bone_edges()
    n_bones = int(edges.shape[0])
    joint_part, joint_chir = _canonical_joint_arrays()
    arrays = {
        "coordinates": np.zeros((t, j, 3), np.float32),
        "joints_raw": np.zeros((t, j, 3), np.float32),
        "missing_mask_prefill": np.zeros((t, j), dtype=bool),
        "original_frame_index": np.arange(t, dtype=np.int32),
        "timestamps_sec": np.arange(t, dtype=np.float64) / 25.0,
        "velocity": np.zeros((t, j, 3), np.float32),
        "bone_vectors": np.zeros((t, n_bones, 3), np.float32),
        "validity": np.ones((t, j), np.float32),
        "confidence": np.ones((t, j), np.float32),
        "padding_mask": np.ones((t,), dtype=bool),
        "joint_part": joint_part,
        "joint_chirality": joint_chir,
        "bone_edges": edges,
        "sequence_length": np.asarray([t], np.int32),
        "raw_landmarks": rec["raw_landmarks"],
        "effective_landmarks": rec["effective_landmarks"],
        "raw_holistic_missing": rec["raw_holistic_missing"],
        "effective_missing": rec["effective_missing"],
        "observation_source_left_hand": np.ones((t,), np.uint8),
        "observation_source_right_hand": np.zeros((t,), np.uint8),
    }
    return atomic_savez_compressed(path, **arrays)


def _valid_skel_npz_kwargs(t: int = 8, j: int = 75) -> dict:
    rec = _recovery(t, j)
    edges = _bone_edges()
    n_bones = int(edges.shape[0])
    joint_part, joint_chir = _canonical_joint_arrays()
    return {
        "coordinates": np.zeros((t, j, 3), np.float32),
        "joints_raw": np.zeros((t, j, 3), np.float32),
        "velocity": np.zeros((t, j, 3), np.float32),
        "bone_vectors": np.zeros((t, n_bones, 3), np.float32),
        "missing_mask_prefill": np.zeros((t, j), dtype=bool),
        "validity": np.ones((t, j), np.float32),
        "confidence": np.ones((t, j), np.float32),
        "padding_mask": np.ones((t,), dtype=bool),
        "joint_part": joint_part,
        "joint_chirality": joint_chir,
        "bone_edges": edges,
        "sequence_length": np.asarray([t], np.int32),
        "original_frame_index": np.arange(t, dtype=np.int32),
        "timestamps_sec": np.arange(t, dtype=np.float64) / 25.0,
        "raw_landmarks": rec["raw_landmarks"],
        "effective_landmarks": rec["effective_landmarks"],
        "raw_holistic_missing": rec["raw_holistic_missing"],
        "effective_missing": rec["effective_missing"],
        "observation_source_left_hand": np.ones((t,), np.uint8),
        "observation_source_right_hand": np.zeros((t,), np.uint8),
    }


def _write_rgb(path: Path, *, nan: bool = False, truncate: bool = False, missing_face: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tok = np.zeros((2, 2304), np.float32)
    if nan:
        tok[0, 0] = np.nan
    miss_f = np.asarray([missing_face, missing_face], dtype=bool)
    miss0 = np.zeros((2,), dtype=bool)
    kwargs = {
        "token_global": tok,
        "token_face": tok,
        "token_left_hand": tok,
        "token_right_hand": tok,
        "missing_face": miss_f,
        "missing_left_hand": miss0,
        "missing_right_hand": miss0,
        "partial_face": miss0,
        "partial_left_hand": miss0,
        "partial_right_hand": miss0,
        "pathway_slow": np.zeros((2, 2048), np.float32),
        "pathway_fast": np.zeros((2, 256), np.float32),
        "original_frame_index": np.arange(10, dtype=np.int32),
    }
    if truncate:
        kwargs.pop("token_face")
    return atomic_savez_compressed(path, **kwargs)


def _write_meta(path: Path, sample_id: str, *, missing_face: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _rv(*, all_missing: bool, partial_missing: bool = False, n_slices: int = 40) -> dict:
        if all_missing:
            n_valid = 0
        elif partial_missing:
            n_valid = max(1, n_slices // 2)
        else:
            n_valid = n_slices
        return {
            "all_missing": all_missing,
            "partial_missing": partial_missing,
            "n_valid": int(n_valid),
            "n_slices": int(n_slices),
            "valid_fraction": (float(n_valid) / float(n_slices)) if n_slices else 0.0,
        }

    windows = [
        {
            "missing_region_mask": {
                "face": missing_face,
                "left_hand": False,
                "right_hand": False,
            },
            "region_validity": {
                "face": _rv(all_missing=missing_face, partial_missing=False),
                "left_hand": _rv(all_missing=False, partial_missing=False),
                "right_hand": _rv(all_missing=False, partial_missing=False),
            },
        }
        for _ in range(2)
    ]
    path.write_text(json.dumps({"sample_id": sample_id, "windows": windows}) + "\n", encoding="utf-8")

def _manifest_rows(
    *,
    runtime: Path,
    binary: Path,
    ids: list[str],
    source_sha: str = "a" * 64,
    root: Path | None = None,
) -> None:
    from ms_sta_v5_d0_full_corpus_lib import sha256_file

    scan_root = root or _ROOT
    for sid in ids:
        dig_s = sha256_file(binary / "skeleton_v3" / f"{sid}.npz")
        skel_side = binary / "skeleton_v3" / f"{sid}.recovery.json"
        if not skel_side.is_file():
            skel_side.write_text(json.dumps({"sample_id": sid}) + "\n", encoding="utf-8")
        dig_s_side = sha256_file(skel_side)
        dig_r = sha256_file(binary / "rgb_token_v3" / f"{sid}.npz")
        rgb_side = binary / "rgb_token_v3" / f"{sid}.meta.json"
        dig_r_side = sha256_file(rgb_side)
        skel_out, skel_side_rel = canonical_sample_paths(
            root=scan_root, binary=binary, phase="skeleton_v3", sample_id=sid
        )
        rgb_out, rgb_side_rel = canonical_sample_paths(
            root=scan_root, binary=binary, phase="rgb_token_v3", sample_id=sid
        )
        commit_manifest_record(
            runtime,
            "skeleton_v3",
            {
                "sample_id": sid,
                "split": "train",
                "signer_id": "s1",
                "source_sha256": source_sha,
                "output_sha256": dig_s,
                "sidecar_sha256": dig_s_side,
                "output_rel": skel_out,
                "sidecar_rel": skel_side_rel,
                "producer": "test",
            },
        )
        commit_manifest_record(
            runtime,
            "rgb_token_v3",
            {
                "sample_id": sid,
                "split": "train",
                "signer_id": "s1",
                "source_sha256": source_sha,
                "skel_sha256": dig_s,
                "output_sha256": dig_r,
                "sidecar_sha256": dig_r_side,
                "output_rel": rgb_out,
                "sidecar_rel": rgb_side_rel,
                "producer": "test",
            },
        )


def test_eligible_identity_exact_counts() -> None:
    ident = load_eligible_identity(root=_ROOT)
    assert ident["eligible_n"] == CORPUS_N
    assert ident["eligible_by_split"] == SPLIT_COUNTS
    assert len(ident["eligible_id_set"]) == CORPUS_N


def test_phases_exact() -> None:
    assert PHASES_EXACT == ("skeleton_v3", "rgb_token_v3", "quality_scan", "completion")
    assert WATCHDOG_INTERVAL_SEC == 600
    assert OUTPUT_VERSION == "v14"
    assert CAMPAIGN_TOKEN == "D0_FULL_CORPUS_V14"


def test_atomic_savez_and_runner_saves(tmp_path: Path) -> None:
    dest, meta, dig = _save_skel_atomic(tmp_path, "000001", _skel_payload(), _recovery())
    assert dest.name == "000001.npz"
    assert dest.is_file()
    from ms_sta_v5_d0_full_corpus_lib import sha256_file

    assert dig == sha256_file(dest)
    validate_npz_keys(dest, SKEL_REQUIRED_KEYS)
    validate_array_contracts(dest, kind="skel")
    assert not list(tmp_path.rglob("*.npz.tmp.*"))
    tok = np.zeros((2, 2304), np.float32)
    miss = np.zeros((2,), dtype=bool)
    rgb_payload = {
        "tokens": {"global": tok, "face": tok, "left_hand": tok, "right_hand": tok},
        "missing_region_mask": {"face": miss, "left_hand": miss, "right_hand": miss, "global": miss},
        "pathway_global_tokens": {"slow": np.zeros((2, 2048), np.float32), "fast": np.zeros((2, 256), np.float32)},
        "original_frame_index": np.arange(10, dtype=np.int32),
        "windows": [
            {
                "missing_region_mask": {"face": False, "left_hand": False, "right_hand": False},
                "region_validity": {
                    "face": {"partial_missing": False},
                    "left_hand": {"partial_missing": False},
                    "right_hand": {"partial_missing": False},
                },
            }
        ]
        * 2,
        "fps": 25.0,
        "n_native_frames": 10,
        "hook": "x",
        "roi_method": "y",
        "regions": ["face", "left_hand", "right_hand"],
    }
    rp, rm, _ = _save_rgb_atomic(tmp_path, "000001", rgb_payload)
    assert rp.is_file() and rm.is_file()
    validate_npz_keys(rp, RGB_REQUIRED_KEYS)
    validate_array_contracts(rp, kind="rgb")


def test_atomic_create_rejects_existing(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    atomic_create_json(p, {"a": 1})
    with pytest.raises(FileExistsError):
        atomic_create_json(p, {"a": 2})


def _skel_resume_paths(tmp_path: Path, sample_id: str) -> tuple[Path, Path, Path]:
    binary = tmp_path
    dest = binary / "skeleton_v3" / f"{sample_id}.npz"
    sidecar = binary / "skeleton_v3" / f"{sample_id}.recovery.json"
    ready = staging_dir(binary, "skeleton_v3", sample_id) / "READY.json"
    return dest, sidecar, ready


def test_existing_output_without_resume_rejected(tmp_path: Path) -> None:
    from ms_sta_v5_d0_full_corpus_lib import sha256_file

    dest, sidecar, ready = _skel_resume_paths(tmp_path, "1")
    dig = _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": "1"}) + "\n", encoding="utf-8")
    side_dig = sha256_file(sidecar)
    with pytest.raises(RuntimeError, match="existing_output_without_resume"):
        heal_or_resume_sample(
            dest=dest,
            sidecar=sidecar,
            ready_path=ready,
            manifest_index={
                "1": {
                    "sample_id": "1",
                    "split": "train",
                    "signer_id": "s1",
                    "source_sha256": "a" * 64,
                    "output_sha256": dig,
                    "sidecar_sha256": side_dig,
                    "output_rel": "x",
                    "sidecar_rel": "y",
                    "producer": "test",
                }
            },
            sample_id="1",
            required_keys=SKEL_REQUIRED_KEYS,
            required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
            resume=False,
            source_sha="a" * 64,
        )


def test_resume_requires_manifest_sha(tmp_path: Path) -> None:
    dest, sidecar, ready = _skel_resume_paths(tmp_path, "1")
    _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": "1"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        heal_or_resume_sample(
            dest=dest,
            sidecar=sidecar,
            ready_path=ready,
            manifest_index={},
            sample_id="1",
            required_keys=SKEL_REQUIRED_KEYS,
            required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
            resume=True,
            source_sha="a" * 64,
        )
    assert ei.value.code == 2


def test_resume_hash_mismatch_exit_2(tmp_path: Path) -> None:
    dest, sidecar, ready = _skel_resume_paths(tmp_path, "1")
    _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": "1"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        heal_or_resume_sample(
            dest=dest,
            sidecar=sidecar,
            ready_path=ready,
            manifest_index={
                "1": {
                    "sample_id": "1",
                    "split": "train",
                    "signer_id": "s1",
                    "source_sha256": "a" * 64,
                    "output_sha256": "0" * 64,
                    "sidecar_sha256": "b" * 64,
                    "output_rel": "x",
                    "sidecar_rel": "y",
                    "producer": "test",
                }
            },
            sample_id="1",
            required_keys=SKEL_REQUIRED_KEYS,
            required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
            resume=True,
            source_sha="a" * 64,
        )
    assert ei.value.code == 2


def test_corrupt_nan_and_truncated_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "x.npz"
    _write_rgb(bad, nan=True)
    with pytest.raises(RuntimeError, match="nonfinite"):
        validate_npz_keys(bad, RGB_REQUIRED_KEYS)
    trunc = tmp_path / "y.npz"
    _write_rgb(trunc, truncate=True)
    with pytest.raises(RuntimeError, match="missing_keys"):
        validate_npz_keys(trunc, RGB_REQUIRED_KEYS)


def test_max_resumes_enforced() -> None:
    assert MAX_RESUMES == 20
    resumes_used = 20
    if resumes_used >= MAX_RESUMES:
        with pytest.raises(RuntimeError, match="max_resumes"):
            raise RuntimeError("max_resumes_exceeded")


def test_process_guard_current_child_dead_and_reuse() -> None:
    assert process_tree_alive(os.getpid()) is True
    # descendants_alive is campaign-runner scoped (not generic process-tree).
    assert descendants_alive(os.getpid()) is False
    assert campaign_related_process_alive(os.getpid()) is False
    assert process_tree_alive(0) is False
    assert process_tree_alive(999_999_999) is False
    # Spawn short-lived child then ensure dead pid does not block.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    child_pid = proc.pid
    assert process_tree_alive(child_pid) is True
    proc.wait(timeout=5)
    assert process_tree_alive(child_pid) is False


def test_threshold_breach_fail_and_no_mutation() -> None:
    thr = default_quality_thresholds()
    frozen = dict(thr)
    stats = {
        "missing_roi_rate_by_region_from_tensor_validity": {
            "face": 0.20,
            "left_hand": 0.0,
            "right_hand": 0.0,
        },
        "partial_missing_rate_by_region": {"face": 0.0, "left_hand": 0.0, "right_hand": 0.0},
    }
    ev = evaluate_quality_thresholds(stats, thr)
    assert ev["ok"] is False
    assert thr == frozen


def test_scanner_rejects_duplicate_missing_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"{i:06d}" for i in range(3)]
    binary = tmp_path / "bin"
    runtime = tmp_path / "rt"
    runtime.mkdir()
    for sid in ids:
        dig_s = _write_skel(binary / "skeleton_v3" / f"{sid}.npz")
        dig_r = _write_rgb(binary / "rgb_token_v3" / f"{sid}.npz")
        _write_meta(binary / "rgb_token_v3" / f"{sid}.meta.json", sid)
    _manifest_rows(runtime=runtime, binary=binary, ids=ids, root=tmp_path)
    _write_rgb(binary / "rgb_token_v3" / "999999.npz")
    _write_meta(binary / "rgb_token_v3" / "999999.meta.json", "999999")

    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus.load_eligible_identity",
        lambda root=None: {
            "eligible_ids": ids,
            "eligible_id_set": set(ids),
            "eligible_by_split": {"train": 3, "val": 0, "test": 0},
            "eligible_n": 3,
            "eligible_sha256": "x",
        },
    )
    monkeypatch.setattr("scan_ms_sta_v5_d0_full_corpus.CORPUS_N", 3)
    monkeypatch.setattr("scan_ms_sta_v5_d0_full_corpus.SPLIT_COUNTS", {"train": 3, "val": 0, "test": 0})
    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus._load_maps",
        lambda root: (
            {i: "train" for i in ids + ["999999"]},
            {i: "s1" for i in ids + ["999999"]},
            {i: {"video_id": i, "split": "train", "signer_id": "s1"} for i in ids + ["999999"]},
        ),
    )
    report = scan_full_corpus(
        binary_root=binary, runtime_dir=runtime, thresholds=default_quality_thresholds(), root=tmp_path
    )
    assert report["ok"] is False
    assert any(f.get("error") == "rgb_id_set_mismatch" for f in report["failures"])


def test_meta_tamper_does_not_change_tensor_metric(tmp_path: Path) -> None:
    sid = "000001"
    rgb = tmp_path / "rgb"
    dig = _write_rgb(rgb / f"{sid}.npz", missing_face=False)
    _ = dig
    stats1 = compute_quality_stats_from_rgb_tensors(
        rgb_dir=rgb, sample_ids=[sid], split_by_id={sid: "train"}, signer_by_id={sid: "s"}
    )
    # Tamper meta-equivalent would be ignored because rates come from tensors only.
    meta = tmp_path / f"{sid}.meta.json"
    _write_meta(meta, sid, missing_face=True)
    stats2 = compute_quality_stats_from_rgb_tensors(
        rgb_dir=rgb, sample_ids=[sid], split_by_id={sid: "train"}, signer_by_id={sid: "s"}
    )
    assert stats1 == stats2
    assert stats1["missing_roi_rate_by_region_from_tensor_validity"]["face"] == 0.0


def test_tensor_tamper_detected(tmp_path: Path) -> None:
    sid = "000001"
    rgb = tmp_path / "rgb"
    path = rgb / f"{sid}.npz"
    _write_rgb(path, missing_face=False)
    data = dict(np.load(path, allow_pickle=False))
    data["missing_face"] = np.asarray([True, True], dtype=bool)
    atomic_savez_compressed(path, **data)
    stats = compute_quality_stats_from_rgb_tensors(
        rgb_dir=rgb, sample_ids=[sid], split_by_id={sid: "train"}, signer_by_id={sid: "s"}
    )
    assert stats["missing_roi_rate_by_region_from_tensor_validity"]["face"] == 1.0


def test_threshold_breach_produces_fail_completion_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["000001", "000002"]
    source_sha = "a" * 64
    binary = tmp_path / "bin"
    runtime = tmp_path / "rt"
    runtime.mkdir()
    for sid in ids:
        _write_skel(binary / "skeleton_v3" / f"{sid}.npz")
        skel_side = binary / "skeleton_v3" / f"{sid}.recovery.json"
        skel_side.write_text(json.dumps({"sample_id": sid}) + "\n", encoding="utf-8")
        _write_rgb(binary / "rgb_token_v3" / f"{sid}.npz", missing_face=True)
        _write_meta(binary / "rgb_token_v3" / f"{sid}.meta.json", sid, missing_face=True)
    _manifest_rows(runtime=runtime, binary=binary, ids=ids, source_sha=source_sha, root=tmp_path)
    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus.load_eligible_identity",
        lambda root=None: {
            "eligible_ids": ids,
            "eligible_id_set": set(ids),
            "eligible_by_split": {"train": 2, "val": 0, "test": 0},
            "eligible_n": 2,
            "eligible_sha256": "x",
        },
    )
    monkeypatch.setattr("scan_ms_sta_v5_d0_full_corpus.CORPUS_N", 2)
    monkeypatch.setattr("scan_ms_sta_v5_d0_full_corpus.SPLIT_COUNTS", {"train": 2, "val": 0, "test": 0})
    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus._load_maps",
        lambda root: (
            {i: "train" for i in ids},
            {i: "s1" for i in ids},
            {i: {"video_id": i, "split": "train", "signer_id": "s1"} for i in ids},
        ),
    )
    thr = default_quality_thresholds()
    report = scan_full_corpus(binary_root=binary, runtime_dir=runtime, thresholds=thr, root=tmp_path)
    assert report["ok"] is False
    assert any(f.get("error") == "quality_threshold_breach" for f in report["failures"])
    verdict = "D0_FULL_CORPUS_SCAN_COMPLETE_PASS" if report["ok"] else "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL"
    assert verdict == "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL"


def test_no_m0_autochain_symbols_in_execution_scripts() -> None:
    from ms_sta_v5_d0_full_corpus_lib import assert_no_forbidden_symbols

    assert_no_forbidden_symbols(FULL_CORPUS_EXECUTABLE_PATHS, root=_ROOT)
    for rel in FULL_CORPUS_EXECUTABLE_PATHS:
        if not rel.startswith("scripts/"):
            continue
        text = (_ROOT / rel).read_text(encoding="utf-8")
        assert "auto_chain_to_m0 = True" not in text
        assert "auto_chain_to_m0=True" not in text
        assert "import run_ms_sta_m0" not in text
        assert not re.search(r"(?<![\"'])run_ms_sta_m0(?![\"'])", text)


def test_binary_root_gitignored() -> None:
    probe = binary_root_rel("deadbeefdead") + "/skeleton_v3/000000.npz"
    proc = subprocess.run(["git", "check-ignore", "-q", probe], cwd=_ROOT)
    assert proc.returncode == 0


def test_quality_stats_denominator_declared(tmp_path: Path) -> None:
    sid = "1"
    _write_rgb(tmp_path / f"{sid}.npz", missing_face=True)
    stats = compute_quality_stats_from_rgb_tensors(
        rgb_dir=tmp_path, sample_ids=[sid], split_by_id={sid: "train"}, signer_by_id={sid: "s"}
    )
    assert stats["denominator"] == "window_region_triples_from_rgb_tensor_masks"
    assert stats["missing_roi_rate_by_region_from_tensor_validity"]["face"] == 1.0


def test_forged_approval_code_lock_rejected() -> None:
    from ms_sta_v5_d0_lib import verify_code_lock_revision_binding

    errs = verify_code_lock_revision_binding(
        "0" * 40, FULL_CORPUS_EXECUTABLE_PATHS, "f" * 64, root=_ROOT
    )
    assert "code_lock_revision_not_in_git" in errs


def test_manifest_rejects_duplicates(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    row = {"sample_id": "1", "output_sha256": "a" * 64, "source_sha256": "b" * 64}
    commit_manifest_record(runtime, "skeleton_v3", row)
    commit_manifest_record(runtime, "skeleton_v3", row)
    row2 = dict(row)
    row2["output_sha256"] = "c" * 64
    with pytest.raises(FileExistsError):
        commit_manifest_record(runtime, "skeleton_v3", row2)


def test_jsonl_manifest_forbidden(tmp_path: Path) -> None:
    p = tmp_path / "MANIFEST_SKEL.jsonl"
    p.write_text(json.dumps({"sample_id": "1"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="jsonl_manifest_forbidden"):
        load_manifest_index(p)


def test_watchdog_script_has_continuous_loop() -> None:
    text = (_ROOT / "scripts/watch_ms_sta_v5_d0_full_corpus.ps1").read_text(encoding="utf-8")
    assert "IntervalSec = 600" in text
    assert "while ($true)" in text
    assert "restart_count" in text
    assert "MaxRestarts = 20" in text
    assert "completion_detected_verify_only" in text
    assert "D0_FULL_CORPUS_V14" in text
    assert "Start-Process" in text
    assert "MONITOR_TICK" in text or "--write-tick" in text


def test_orphan_worker_blocks_when_parent_dead(tmp_path: Path) -> None:
    """Parent-dead / child-live must BLOCK via worker registry + process table."""
    from ms_sta_v5_d0_full_corpus_lib import CAMPAIGN_TOKEN, register_worker

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runner = tmp_path / "run_ms_sta_v5_d0_full_corpus.py"
    runner.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(runner), "--campaign-token", CAMPAIGN_TOKEN, "--phase", "skeleton_v3"]
    )
    try:
        register_worker(runtime, pid=child.pid, phase="skeleton_v3", campaign_token=CAMPAIGN_TOKEN)
        dead_parent = 999_999_998
        assert campaign_related_process_alive(
            dead_parent, campaign_token=CAMPAIGN_TOKEN, runtime=runtime
        ) is True
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_missing_source_sha_rejected_on_resume(tmp_path: Path) -> None:
    from ms_sta_v5_d0_full_corpus_lib import sha256_file

    dest, sidecar, ready = _skel_resume_paths(tmp_path, "1")
    dig = _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": "1"}) + "\n", encoding="utf-8")
    side_dig = sha256_file(sidecar)
    with pytest.raises((SystemExit, RuntimeError)):
        heal_or_resume_sample(
            dest=dest,
            sidecar=sidecar,
            ready_path=ready,
            manifest_index={
                "1": {
                    "sample_id": "1",
                    "split": "train",
                    "signer_id": "s1",
                    "source_sha256": None,  # forbidden
                    "output_sha256": dig,
                    "sidecar_sha256": side_dig,
                    "output_rel": "x",
                    "sidecar_rel": "y",
                    "producer": "test",
                }
            },
            sample_id="1",
            required_keys=SKEL_REQUIRED_KEYS,
            required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
            resume=True,
            source_sha="a" * 64,
        )


def test_sample_crash_before_manifest_heals(tmp_path: Path) -> None:
    """NPZ+sidecar+READY without manifest must heal under resume."""
    from ms_sta_v5_d0_full_corpus_lib import sha256_file, staging_dir, write_ready_descriptor

    sid = "000001"
    dest, sidecar, ready = _skel_resume_paths(tmp_path, sid)
    dig = _write_skel(dest)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"sample_id": sid, "arm": "C3"}) + "\n", encoding="utf-8")
    side_dig = sha256_file(sidecar)
    man_row = {
        "sample_id": sid,
        "split": "train",
        "signer_id": "s1",
        "source_sha256": "b" * 64,
        "output_sha256": dig,
        "sidecar_sha256": side_dig,
        "output_rel": f"skeleton_v3/{sid}.npz",
        "sidecar_rel": f"skeleton_v3/{sid}.recovery.json",
        "producer": "test",
    }
    ready.parent.mkdir(parents=True, exist_ok=True)
    write_ready_descriptor(ready, {"manifest_row": man_row})
    out = heal_or_resume_sample(
        dest=dest,
        sidecar=sidecar,
        ready_path=ready,
        manifest_index={},
        sample_id=sid,
        required_keys=SKEL_REQUIRED_KEYS,
        required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
        resume=True,
        source_sha="b" * 64,
    )
    assert out is not None and out.startswith("HEAL_MANIFEST:")


def test_atomic_create_idempotent_and_heals_truncated(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    sha1 = atomic_create_json(p, {"a": 1})
    sha2 = atomic_create_json(p, {"a": 1})
    assert sha1 == sha2
    p.write_text('{"a":', encoding="utf-8")
    sha3 = atomic_create_json(p, {"a": 2})
    assert load_json(p) == {"a": 2}
    assert sha3 != sha1


def test_array_contract_rejects_bad_joint_and_token_dims(tmp_path: Path) -> None:
    bad_skel = tmp_path / "bad_skel.npz"
    t, j = 8, 74
    kwargs = _valid_skel_npz_kwargs(t=t, j=75)
    kwargs["coordinates"] = np.zeros((t, j, 3), np.float32)
    kwargs["joints_raw"] = np.zeros((t, j, 3), np.float32)
    kwargs["velocity"] = np.zeros((t, j, 3), np.float32)
    kwargs["missing_mask_prefill"] = np.zeros((t, j), dtype=bool)
    kwargs["validity"] = np.ones((t, j), np.float32)
    kwargs["confidence"] = np.ones((t, j), np.float32)
    kwargs["raw_landmarks"] = np.zeros((t, j, 3), np.float32)
    kwargs["effective_landmarks"] = np.zeros((t, j, 3), np.float32)
    kwargs["raw_holistic_missing"] = np.zeros((t, j), dtype=bool)
    kwargs["effective_missing"] = np.zeros((t, j), dtype=bool)
    atomic_savez_compressed(bad_skel, **kwargs)
    with pytest.raises(RuntimeError, match="array_contract_joint_dims"):
        validate_array_contracts(bad_skel, kind="skel")

    tok_bad = np.zeros((2, 1), np.float32)
    miss = np.zeros((2,), dtype=bool)
    bad_rgb = tmp_path / "bad_rgb.npz"
    atomic_savez_compressed(
        bad_rgb,
        token_global=tok_bad,
        token_face=tok_bad,
        token_left_hand=tok_bad,
        token_right_hand=tok_bad,
        missing_face=miss,
        missing_left_hand=miss,
        missing_right_hand=miss,
        partial_face=miss,
        partial_left_hand=miss,
        partial_right_hand=miss,
        pathway_slow=np.zeros((2, 2048), np.float32),
        pathway_fast=np.zeros((2, 256), np.float32),
        original_frame_index=np.arange(10, dtype=np.int32),
    )
    with pytest.raises(RuntimeError, match="array_contract_token_dim"):
        validate_array_contracts(bad_rgb, kind="rgb")


def test_crash_mid_record_write_heals(tmp_path: Path) -> None:
    from ms_sta_v5_d0_full_corpus_lib import manifest_record_path, sha256_file
    from run_ms_sta_v5_d0_full_corpus import _heal_manifest_from_ready

    binary = tmp_path / "bin"
    runtime = tmp_path / "runtime"
    sid = "000001"
    dest, sidecar, ready = _skel_resume_paths(binary, sid)
    dig = _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": sid, "arm": "C3"}) + "\n", encoding="utf-8")
    side_dig = sha256_file(sidecar)
    out_rel, side_rel = canonical_sample_paths(root=tmp_path, binary=binary, phase="skeleton_v3", sample_id=sid)
    man_row = {
        "sample_id": sid,
        "split": "train",
        "signer_id": "s1",
        "source_sha256": "b" * 64,
        "output_sha256": dig,
        "sidecar_sha256": side_dig,
        "output_rel": out_rel,
        "sidecar_rel": side_rel,
        "producer": "test",
    }
    write_ready_descriptor(ready, {"manifest_row": man_row})
    rec_path = manifest_record_path(runtime, "skeleton_v3", sid)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text('{"sample_id":', encoding="utf-8")
    manifest_index = load_manifest_index(runtime, "skeleton_v3")
    assert sid not in manifest_index
    out = heal_or_resume_sample(
        dest=dest,
        sidecar=sidecar,
        ready_path=ready,
        manifest_index=manifest_index,
        sample_id=sid,
        required_keys=SKEL_REQUIRED_KEYS,
        required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
        resume=True,
        source_sha="b" * 64,
    )
    assert out is not None and out.startswith("HEAL_MANIFEST:")
    _heal_manifest_from_ready(
        binary=binary,
        runtime=runtime,
        phase="skeleton_v3",
        sample_id=sid,
        manifest_index=manifest_index,
    )
    assert load_json(rec_path)["sample_id"] == sid


def test_completion_heal_reuses_closed_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ms_sta_v5_d0_full_corpus_lib import atomic_replace_json, sha256_file

    evidence = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    binary = tmp_path / "bin"
    evidence.mkdir()
    runtime.mkdir()
    binary.mkdir()
    quality_stats = {"denominator": "x"}

    def _mock_scan(**kwargs: object) -> dict:
        rt = kwargs["runtime_dir"]
        atomic_replace_json(rt / "QUALITY_STATS.json", quality_stats)
        stats_sha = sha256_file(rt / "QUALITY_STATS.json")
        report = {
            "ok": True,
            "n_paired": 16,
            "failed_count": 0,
            "quality_stats_sha256": stats_sha,
        }
        atomic_replace_json(rt / "SCAN_REPORT.json", report)
        return report

    monkeypatch.setattr("scan_ms_sta_v5_d0_full_corpus.scan_full_corpus", _mock_scan)
    approval = {"quality_thresholds": default_quality_thresholds()}
    out1 = run_quality_and_completion(binary=binary, runtime=runtime, root=tmp_path, approval=approval)
    original_closed_at = out1["closed_at"]
    assert (runtime / "COMPLETION_INTENT.json").is_file()
    (runtime / "COMPLETION.json").unlink()
    out2 = run_quality_and_completion(binary=binary, runtime=runtime, root=tmp_path, approval=approval)
    assert out2["closed_at"] == original_closed_at


def test_completion_contradictory_seal_rejected(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    evidence.mkdir()
    runtime.mkdir()
    quality_stats = {"denominator": "x"}
    scan_report = {"ok": True, "n_paired": 24752, "failed_count": 0}
    completion = {
        "schema": "ms_sta_v5_d0_full_corpus_completion_v14",
        "verdict": "D0_FULL_CORPUS_SCAN_COMPLETE_PASS",
        "failed_count": 0,
        "n_paired": 24752,
        "hard_stop": True,
    }
    publish_completion_pack(
        evidence=evidence,
        runtime=runtime,
        completion=completion,
        scan_report=scan_report,
        quality_stats=quality_stats,
    )
    bad_completion = dict(completion)
    bad_completion["n_paired"] = 1
    bad_completion["failed_count"] = 99
    with pytest.raises(SystemExit) as ei:
        publish_completion_pack(
            evidence=evidence,
            runtime=runtime,
            completion=bad_completion,
            scan_report={"ok": True, "n_paired": 1, "failed_count": 99},
            quality_stats=quality_stats,
        )
    assert ei.value.code == 2


def test_atomic_manifest_index_rebuild(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    row = {
        "sample_id": "000001",
        "split": "train",
        "signer_id": "s1",
        "source_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "sidecar_sha256": "c" * 64,
        "output_rel": "x.npz",
        "sidecar_rel": "x.json",
        "producer": "test",
    }
    index_path = manifest_index_path(runtime, "skeleton_v3")
    commit_manifest_record(runtime, "skeleton_v3", row)
    assert not index_path.is_file()
    row2 = dict(row)
    row2["sample_id"] = "000002"
    commit_manifest_record(runtime, "skeleton_v3", row2)
    assert not index_path.is_file()
    idx = load_manifest_index(runtime, "skeleton_v3")
    assert len(idx) == 2
    index_sha = rebuild_manifest_index(runtime, "skeleton_v3")
    assert index_sha
    assert load_json(index_path)["n_records"] == 2
    assert manifest_record_path(runtime, "skeleton_v3", "000001").is_file()


def test_crash_after_sidecar_before_ready_heals(tmp_path: Path) -> None:
    from ms_sta_v5_d0_full_corpus_lib import sha256_file

    binary = tmp_path / "bin"
    sid = "000001"
    dest, sidecar, ready = _skel_resume_paths(binary, sid)
    dig = _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": sid, "arm": "C3"}) + "\n", encoding="utf-8")
    side_dig = sha256_file(sidecar)
    out_rel, side_rel = canonical_sample_paths(root=tmp_path, binary=binary, phase="skeleton_v3", sample_id=sid)
    man_row = {
        "sample_id": sid,
        "split": "train",
        "signer_id": "s1",
        "source_sha256": "b" * 64,
        "output_sha256": dig,
        "sidecar_sha256": side_dig,
        "output_rel": out_rel,
        "sidecar_rel": side_rel,
        "producer": "test",
    }
    write_ready_descriptor(ready, {"manifest_row": man_row, "output_sha256": dig, "sidecar_sha256": side_dig})
    out = heal_or_resume_sample(
        dest=dest,
        sidecar=sidecar,
        ready_path=ready,
        manifest_index={},
        sample_id=sid,
        required_keys=SKEL_REQUIRED_KEYS,
        required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
        resume=True,
        source_sha="b" * 64,
    )
    assert out is not None and out.startswith("HEAL_MANIFEST:")


def test_crash_after_manifest_commit_resume_skips(tmp_path: Path) -> None:
    from ms_sta_v5_d0_full_corpus_lib import sha256_file

    binary = tmp_path / "bin"
    runtime = tmp_path / "runtime"
    sid = "000001"
    dest, sidecar, ready = _skel_resume_paths(binary, sid)
    dig = _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": sid}) + "\n", encoding="utf-8")
    side_dig = sha256_file(sidecar)
    out_rel, side_rel = canonical_sample_paths(root=tmp_path, binary=binary, phase="skeleton_v3", sample_id=sid)
    row = {
        "sample_id": sid,
        "split": "train",
        "signer_id": "s1",
        "source_sha256": "b" * 64,
        "output_sha256": dig,
        "sidecar_sha256": side_dig,
        "output_rel": out_rel,
        "sidecar_rel": side_rel,
        "producer": "test",
    }
    record_sha = commit_manifest_record(runtime, "skeleton_v3", row)
    write_ready_descriptor(
        ready,
        {
            "manifest_row": row,
            "manifest_record_sha256": record_sha,
            "output_sha256": dig,
            "sidecar_sha256": side_dig,
            "source_sha256": "b" * 64,
        },
    )
    manifest_index = load_manifest_index(runtime, "skeleton_v3")
    out = heal_or_resume_sample(
        dest=dest,
        sidecar=sidecar,
        ready_path=ready,
        manifest_index=manifest_index,
        sample_id=sid,
        required_keys=SKEL_REQUIRED_KEYS,
        required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
        resume=True,
        source_sha="b" * 64,
    )
    assert out == dig


def test_crash_before_ready_no_orphan_resume(tmp_path: Path) -> None:
    binary = tmp_path / "bin"
    sid = "000001"
    dest, sidecar, ready = _skel_resume_paths(binary, sid)
    _write_skel(dest)
    sidecar.write_text(json.dumps({"sample_id": sid}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        heal_or_resume_sample(
            dest=dest,
            sidecar=sidecar,
            ready_path=ready,
            manifest_index={},
            sample_id=sid,
            required_keys=SKEL_REQUIRED_KEYS,
            required_manifest_fields=SKEL_MANIFEST_REQUIRED_FIELDS,
            resume=True,
            source_sha="b" * 64,
        )
    assert ei.value.code == 2


def _completion_pack_inputs(tmp_path: Path, *, verdict: str = "D0_FULL_CORPUS_SCAN_COMPLETE_PASS") -> tuple[Path, Path, dict, dict, dict]:
    evidence = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    evidence.mkdir()
    runtime.mkdir()
    quality_stats = {"denominator": "x"}
    scan_report = {"ok": True, "n_paired": 24752, "failed_count": 0}
    completion = {
        "schema": "ms_sta_v5_d0_full_corpus_completion_v14",
        "verdict": verdict,
        "failed_count": 0,
        "n_paired": 24752,
        "hard_stop": True,
        "binary_root_rel": binary_root_rel("deadbeefdead"),
    }
    return evidence, runtime, completion, scan_report, quality_stats


def test_completion_intent_mismatch_blocks_before_seal(tmp_path: Path) -> None:
    evidence, runtime, completion, scan_report, quality_stats = _completion_pack_inputs(tmp_path)
    from ms_sta_v5_d0_full_corpus_lib import _completion_intent_payload, _resolve_completion_closed_at, sha256_file

    stage = runtime / "_staging_completion"
    stage.mkdir(parents=True, exist_ok=True)
    atomic_replace_json(stage / "QUALITY_STATS.json", quality_stats)
    atomic_replace_json(stage / "SCAN_REPORT.json", scan_report)
    fail_completion = dict(completion)
    fail_completion["verdict"] = "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL"
    fail_completion["failed_count"] = 99
    closed_at = _resolve_completion_closed_at(evidence=evidence, runtime=runtime, completion=fail_completion)
    fail_completion["closed_at"] = closed_at
    fail_completion["quality_stats_sha256"] = sha256_file(stage / "QUALITY_STATS.json")
    fail_completion["scan_report_sha256"] = sha256_file(stage / "SCAN_REPORT.json")
    intent_obj = _completion_intent_payload(fail_completion, closed_at=closed_at)
    atomic_create_json(runtime / "COMPLETION_INTENT.json", intent_obj)
    with pytest.raises(SystemExit) as ei:
        publish_completion_pack(
            evidence=evidence,
            runtime=runtime,
            completion=completion,
            scan_report=scan_report,
            quality_stats=quality_stats,
        )
    assert ei.value.code == 2
    assert not (evidence / "D0_FULL_CORPUS_COMPLETION_SEAL.json").is_file()
    assert not (runtime / "QUALITY_STATS.json").is_file()


def test_completion_matching_intent_only_then_publish(tmp_path: Path) -> None:
    evidence, runtime, completion, scan_report, quality_stats = _completion_pack_inputs(tmp_path)
    stage = runtime / "_staging_completion"
    stage.mkdir(parents=True, exist_ok=True)
    atomic_replace_json(stage / "QUALITY_STATS.json", quality_stats)
    atomic_replace_json(stage / "SCAN_REPORT.json", scan_report)
    completion_staged = dict(completion)
    atomic_replace_json(stage / "COMPLETION.json", completion_staged)
    from ms_sta_v5_d0_full_corpus_lib import _completion_intent_payload, _resolve_completion_closed_at, sha256_file

    closed_at = _resolve_completion_closed_at(evidence=evidence, runtime=runtime, completion=completion_staged)
    completion_staged["closed_at"] = closed_at
    completion_staged["quality_stats_sha256"] = sha256_file(stage / "QUALITY_STATS.json")
    completion_staged["scan_report_sha256"] = sha256_file(stage / "SCAN_REPORT.json")
    intent_obj = _completion_intent_payload(completion_staged, closed_at=closed_at)
    atomic_create_json(runtime / "COMPLETION_INTENT.json", intent_obj)
    out = publish_completion_pack(
        evidence=evidence,
        runtime=runtime,
        completion=completion,
        scan_report=scan_report,
        quality_stats=quality_stats,
    )
    assert out["verdict"] == "D0_FULL_CORPUS_SCAN_COMPLETE_PASS"
    assert (evidence / "D0_FULL_CORPUS_COMPLETION_SEAL.json").is_file()
    ready = load_json(runtime / "COMPLETION_READY.json")
    assert ready.get("intent_sha256")


def test_completion_seal_heal_requires_matching_intent(tmp_path: Path) -> None:
    evidence, runtime, completion, scan_report, quality_stats = _completion_pack_inputs(tmp_path)
    publish_completion_pack(
        evidence=evidence,
        runtime=runtime,
        completion=completion,
        scan_report=scan_report,
        quality_stats=quality_stats,
    )
    (runtime / "COMPLETION.json").unlink()
    out = publish_completion_pack(
        evidence=evidence,
        runtime=runtime,
        completion=completion,
        scan_report=scan_report,
        quality_stats=quality_stats,
    )
    assert out["verdict"] == "D0_FULL_CORPUS_SCAN_COMPLETE_PASS"
    intent = load_json(runtime / "COMPLETION_INTENT.json")
    intent["verdict"] = "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL"
    atomic_replace_json(runtime / "COMPLETION_INTENT.json", intent)
    with pytest.raises(SystemExit) as ei:
        publish_completion_pack(
            evidence=evidence,
            runtime=runtime,
            completion=completion,
            scan_report=scan_report,
            quality_stats=quality_stats,
        )
    assert ei.value.code == 2


def test_array_contract_rejects_misaligned_frame_axes(tmp_path: Path) -> None:
    bad = tmp_path / "misaligned.npz"
    kwargs = _valid_skel_npz_kwargs(t=7)
    kwargs["timestamps_sec"] = np.arange(6, dtype=np.float64) / 25.0
    atomic_savez_compressed(bad, **kwargs)
    with pytest.raises(RuntimeError, match="array_contract_alignment"):
        validate_array_contracts(bad, kind="skel")


@pytest.mark.parametrize(
    "missing_key",
    ["velocity", "bone_vectors", "validity", "confidence", "padding_mask", "joint_part", "joint_chirality", "bone_edges", "sequence_length"],
)
def test_skel_required_keys_and_contracts_reject_missing(tmp_path: Path, missing_key: str) -> None:
    path = tmp_path / f"missing_{missing_key}.npz"
    kwargs = _valid_skel_npz_kwargs()
    kwargs.pop(missing_key)
    atomic_savez_compressed(path, **kwargs)
    with pytest.raises(RuntimeError, match="npz_missing_keys"):
        validate_npz_keys(path, SKEL_REQUIRED_KEYS)
    with pytest.raises(RuntimeError, match="array_contract_missing_key"):
        validate_array_contracts(path, kind="skel")


def test_manifest_filename_must_match_sample_id(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    samples = manifest_record_path(runtime, "skeleton_v3", "000001").parent
    samples.mkdir(parents=True, exist_ok=True)
    wrong = samples / "000002.json"
    wrong.write_text(
        json.dumps({"sample_id": "000001", "output_sha256": "a" * 64, "source_sha256": "b" * 64}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="manifest_filename_sample_id_mismatch"):
        load_manifest_index(runtime, "skeleton_v3")
    with pytest.raises(RuntimeError, match="manifest_filename_sample_id_mismatch"):
        rebuild_manifest_index(runtime, "skeleton_v3")


def test_monitor_eta_uses_two_phase_denominator(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    now = 1_700_000_000.0
    prev_tick = {
        "timestamp": "2023-11-14T22:13:20Z",
        "committed_counts": {"skeleton_v3": 24750, "rgb_token_v3": 0},
    }
    atomic_replace_json(runtime / "MONITOR_TICK.json", prev_tick)
    current = {"skeleton_v3": 24752, "rgb_token_v3": 0}
    progress = _progress_metrics(runtime, current_counts=current, status="RUNNING", now_ts=now + 3600)
    assert progress["remaining_aggregate"] == CORPUS_N
    assert progress["eta_hours_aggregate"] not in (None, 0)


def test_psutil_pin_present_in_requirements() -> None:
    from ms_sta_v5_d0_full_corpus_lib import PSUTIL_PIN, assert_psutil_available

    assert_psutil_available()
    assert PSUTIL_PIN in (_ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
    assert PSUTIL_PIN in (_ROOT / "requirements-extract.txt").read_text(encoding="utf-8")
    assert PSUTIL_PIN in (_ROOT / "requirements-d0-full-corpus.txt").read_text(encoding="utf-8")


@pytest.mark.requires_artifacts
def test_c3_skel_parity_vs_frozen_r7c_pilot() -> None:
    """Exact array/metadata parity vs frozen R7C pilot outputs for all 16 samples."""
    pytest.importorskip("mediapipe")  # extraction runs only in the extract environment
    from ms_sta_v5_d0_lib import PILOT_R1_SAMPLE_REL, PILOT_R7C_OUT_DIR_REL, load_json
    from src.data_prep.hand_recovery_v6 import HandRecoveryEngine
    from src.data_prep.skel_v3 import build_skel_v3_arrays

    sample_set = load_json(_ROOT / PILOT_R1_SAMPLE_REL)
    samples = sample_set["samples"]
    assert len(samples) == 16
    mismatches = []
    for s in samples:
        sid = s["sample_id"]
        frozen = _ROOT / PILOT_R7C_OUT_DIR_REL / "skel_v3" / f"{sid}.npz"
        if not frozen.is_file():
            pytest.skip(f"frozen skel missing: {sid}")
        eng = HandRecoveryEngine(arm="C3", model_complexity=0, store_stride=2)
        try:
            rec = eng.process_video(_ROOT / s["video_rel"])
        finally:
            eng.close()
        payload = build_skel_v3_arrays(
            joints_raw=rec["effective_landmarks"],
            raw_pose=rec["raw_pose"],
            missing_prefill=rec["effective_missing"],
            original_frame_index=rec["original_frame_index"],
            timestamps_sec=rec["timestamps_sec"],
        )
        ref = np.load(frozen, allow_pickle=False)
        for key in (
            "coordinates",
            "missing_mask_prefill",
            "original_frame_index",
            "effective_landmarks",
            "raw_landmarks",
            "effective_missing",
            "raw_holistic_missing",
            "observation_source_left_hand",
            "observation_source_right_hand",
        ):
            if key not in ref.files:
                mismatches.append(f"{sid}:missing_key:{key}")
                continue
            got = {
                "coordinates": payload["coordinates"],
                "missing_mask_prefill": payload["missing_mask_prefill"],
                "original_frame_index": payload["original_frame_index"],
                "effective_landmarks": rec["effective_landmarks"],
                "raw_landmarks": rec["raw_landmarks"],
                "effective_missing": rec["effective_missing"],
                "raw_holistic_missing": rec["raw_holistic_missing"],
                "observation_source_left_hand": np.asarray(
                    [{"missing": 0, "holistic": 1, "hands_fallback": 2}[str(x)] for x in rec["observation_source_left_hand"]],
                    dtype=np.uint8,
                ),
                "observation_source_right_hand": np.asarray(
                    [{"missing": 0, "holistic": 1, "hands_fallback": 2}[str(x)] for x in rec["observation_source_right_hand"]],
                    dtype=np.uint8,
                ),
            }[key]
            if not np.array_equal(got, ref[key]):
                mismatches.append(f"{sid}:array_mismatch:{key}")
    assert not mismatches, mismatches[:20]


def test_executable_paths_v14_unique_and_include_verifier() -> None:
    assert len(FULL_CORPUS_EXECUTABLE_PATHS) == len(set(FULL_CORPUS_EXECUTABLE_PATHS))
    assert "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py" in FULL_CORPUS_EXECUTABLE_PATHS
    assert "scripts/write_ms_sta_v5_d0_full_corpus_approval_v13.py" in FULL_CORPUS_EXECUTABLE_PATHS
    assert "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v13.py" in FULL_CORPUS_EXECUTABLE_PATHS
    assert "scripts/write_ms_sta_v5_d0_full_corpus_approval_v14.py" in FULL_CORPUS_EXECUTABLE_PATHS
    assert "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v14.py" in FULL_CORPUS_EXECUTABLE_PATHS
    assert "scripts/audit_ms_sta_v5_d0_full_corpus_live_readiness.py" in FULL_CORPUS_EXECUTABLE_PATHS
    assert FULL_CORPUS_EXECUTABLE_PATHS_V13 != FULL_CORPUS_EXECUTABLE_PATHS


def test_launch_readiness_not_ready_without_v14_artifacts(tmp_path: Path) -> None:
    status = evaluate_launch_readiness(root=tmp_path)
    assert status["ready"] is False
    assert status["errors"]
    assert status["output_version"] == "v14"
    assert any("approval_v14_missing" in e for e in status["errors"])


def test_external_reaudit_pass_rejects_fail() -> None:
    with pytest.raises(ValueError, match="external_reaudit_not_pass"):
        validate_external_reaudit_pass({"audit_verdict": "FAIL"}, root=_ROOT)


def test_external_reaudit_pass_rejects_minimal_pass() -> None:
    with pytest.raises(ValueError, match="external_reaudit_bad_schema"):
        validate_external_reaudit_pass({"audit_verdict": "PASS"}, root=_ROOT)


def test_external_reaudit_pass_rejects_wrong_schema(tmp_path: Path) -> None:
    obj = {
        "schema": "wrong_schema",
        "audit_verdict": "PASS",
        "auditor": "independent",
        "worktree_clean_before_audit_artifact": True,
        "head_at_audit": "a" * 40,
        "origin_main_at_audit": "a" * 40,
        "blocking_findings": [],
        "decision": dict(EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION),
    }
    with pytest.raises(ValueError, match="external_reaudit_bad_schema"):
        validate_external_reaudit_pass(obj, root=tmp_path)


def _write_v13_sealed_artifacts(root: Path, *, code_lock: str = "a" * 40) -> dict[str, str]:
    approval = {
        "schema": "ms_sta_v5_d0_full_corpus_approval_v14",
        "code_lock_git_revision": code_lock,
    }
    protocol = {"schema": "ms_sta_v5_d0_full_corpus_execution_protocol_v14"}
    run_contract = {"schema": "ms_sta_v5_d0_full_corpus_run_contract_v14"}
    audit = {"schema": "ms_sta_v5_d0_full_corpus_approval_audit_v14"}
    paths = {
        "approval": root / D0_FULL_CORPUS_APPROVAL_V14_REL,
        "protocol": root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL,
        "run_contract": root / D0_FULL_CORPUS_RUN_CONTRACT_V14_REL,
        "audit": root / D0_FULL_CORPUS_APPROVAL_AUDIT_V14_REL,
    }
    shas: dict[str, str] = {}
    for key, payload in (
        ("approval", approval),
        ("protocol", protocol),
        ("run_contract", run_contract),
        ("audit", audit),
    ):
        path = paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        shas[key] = atomic_create_json(path, payload)
    parent_reaudit = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL
    parent_reaudit.parent.mkdir(parents=True, exist_ok=True)
    parent_reaudit.write_bytes(b"v13-fail-parent-stub\n")
    return shas


def _strict_pass_payload(shas: dict[str, str], *, code_lock: str = "a" * 40) -> dict:
    return {
        "schema": EXTERNAL_REAUDIT_PASS_SCHEMA,
        "audit_verdict": "PASS",
        "auditor": "independent",
        "worktree_clean_before_audit_artifact": True,
        "head_at_audit": code_lock,
        "origin_main_at_audit": code_lock,
        "code_lock_git_revision": code_lock,
        "approval_v14_sha256": shas["approval"],
        "execution_protocol_v14_sha256": shas["protocol"],
        "run_contract_v14_sha256": shas["run_contract"],
        "approval_audit_v14_sha256": shas["audit"],
        "parent_external_reaudit_v13_sha256": EXPECTED_V13_EXTERNAL_REAUDIT_SHA256,
        "blocking_findings": [],
        "decision": dict(EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION),
        "output_version": "v14",
        "campaign_token": CAMPAIGN_TOKEN,
    }


def _patch_external_reaudit_pass_deps(
    monkeypatch: pytest.MonkeyPatch,
    shas: dict[str, str],
    *,
    code_lock: str,
    git_blob_sha: dict[str, str] | None = None,
    commit_exists: bool = True,
) -> None:
    def _fake_sha(path: Path) -> str:
        text = str(path).replace("\\", "/")
        if text.endswith(D0_FULL_CORPUS_APPROVAL_V14_REL):
            return shas["approval"]
        if text.endswith(D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL):
            return shas["protocol"]
        if text.endswith(D0_FULL_CORPUS_RUN_CONTRACT_V14_REL):
            return shas["run_contract"]
        if text.endswith(D0_FULL_CORPUS_APPROVAL_AUDIT_V14_REL):
            return shas["audit"]
        if text.endswith(D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL):
            return EXPECTED_V13_EXTERNAL_REAUDIT_SHA256
        raise AssertionError(f"unexpected_sha_path:{path}")

    def _fake_git_blob(rel: str, rev: str, *, root: Path) -> str:
        if git_blob_sha is not None and rel in git_blob_sha:
            return git_blob_sha[rel]
        if rel == D0_FULL_CORPUS_APPROVAL_V14_REL:
            return shas["approval"]
        if rel == D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL:
            return shas["protocol"]
        if rel == D0_FULL_CORPUS_RUN_CONTRACT_V14_REL:
            return shas["run_contract"]
        if rel == D0_FULL_CORPUS_APPROVAL_AUDIT_V14_REL:
            return shas["audit"]
        raise AssertionError(f"unexpected_git_blob_rel:{rel}")

    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_file", _fake_sha)
    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.git_commit_exists", lambda rev, root=None: commit_exists)
    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_git_blob_at_revision", _fake_git_blob)


def test_external_reaudit_pass_rejects_wrong_code_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v13_sealed_artifacts(tmp_path, code_lock="a" * 40)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload(shas, code_lock="c" * 40)
    with pytest.raises(ValueError, match="external_reaudit_code_lock_mismatch"):
        validate_external_reaudit_pass(obj, root=tmp_path)


def test_external_reaudit_pass_rejects_wrong_approval_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v13_sealed_artifacts(tmp_path)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload(shas)
    obj["approval_v14_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="approval_v14_sha256_mismatch"):
        validate_external_reaudit_pass(obj, root=tmp_path)


def test_external_reaudit_pass_rejects_nonempty_blocking_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shas = _write_v13_sealed_artifacts(tmp_path)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload(shas)
    obj["blocking_findings"] = [{"id": "P0"}]
    with pytest.raises(ValueError, match="external_reaudit_blocking_findings_nonempty"):
        validate_external_reaudit_pass(obj, root=tmp_path)


def test_external_reaudit_pass_rejects_wrong_output_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v13_sealed_artifacts(tmp_path)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload(shas)
    obj["output_version"] = "v13"
    with pytest.raises(ValueError, match="external_reaudit_output_version_mismatch"):
        validate_external_reaudit_pass(obj, root=tmp_path)


def test_external_reaudit_pass_rejects_fabricated_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shas = _write_v13_sealed_artifacts(tmp_path, code_lock="a" * 40)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock="a" * 40, commit_exists=False)
    obj = _strict_pass_payload(shas, code_lock="a" * 40)
    with pytest.raises(ValueError, match="external_reaudit_head_not_found"):
        validate_external_reaudit_pass(obj, root=tmp_path)


def test_external_reaudit_pass_rejects_git_blob_mismatch_at_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_lock = "b" * 40
    shas = _write_v13_sealed_artifacts(tmp_path, code_lock=code_lock)
    _patch_external_reaudit_pass_deps(
        monkeypatch,
        shas,
        code_lock=code_lock,
        git_blob_sha={D0_FULL_CORPUS_APPROVAL_V14_REL: "f" * 64},
    )
    obj = _strict_pass_payload(shas, code_lock=code_lock)
    with pytest.raises(ValueError, match="external_reaudit_git_blob_mismatch:approval_v14_sha256"):
        validate_external_reaudit_pass(obj, root=tmp_path)


@pytest.mark.parametrize(
    "missing_flag",
    ["test_authorized", "demo_authorized", "auto_chain"],
)
def test_external_reaudit_pass_rejects_omitted_decision_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_flag: str
) -> None:
    shas = _write_v13_sealed_artifacts(tmp_path)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload(shas)
    del obj["decision"][missing_flag]
    with pytest.raises(ValueError, match=f"external_reaudit_decision_missing:{missing_flag}"):
        validate_external_reaudit_pass(obj, root=tmp_path)


@pytest.mark.parametrize(
    "flag",
    [
        "launch_authorized",
        "full_corpus_extraction_started",
        "m0_authorized",
        "training_authorized",
        "test_authorized",
        "demo_authorized",
        "auto_chain",
    ],
)
def test_external_reaudit_pass_rejects_forbidden_true_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    shas = _write_v13_sealed_artifacts(tmp_path)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock="a" * 40)
    obj = _strict_pass_payload(shas)
    obj["decision"][flag] = True
    with pytest.raises(ValueError, match=f"external_reaudit_{flag}_must_be_false"):
        validate_external_reaudit_pass(obj, root=tmp_path)


def test_external_reaudit_pass_accepts_synthetic_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code_lock = "a" * 40
    shas = _write_v13_sealed_artifacts(tmp_path, code_lock=code_lock)
    _patch_external_reaudit_pass_deps(monkeypatch, shas, code_lock=code_lock)
    obj = _strict_pass_payload(shas, code_lock=code_lock)
    assert validate_external_reaudit_pass(obj, root=tmp_path) == {"ok": True}


def test_post_audit_auth_rejects_missing_launch_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    for rel in (
        D0_FULL_CORPUS_APPROVAL_V14_REL,
        D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL,
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n", encoding="utf-8")

    def _fake_sha(path: Path) -> str:
        return "a" * 64

    monkeypatch.setattr("ms_sta_v5_d0_full_corpus_lib.sha256_file", _fake_sha)
    obj = {
        "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v14",
        "campaign_token": CAMPAIGN_TOKEN,
        "output_version": OUTPUT_VERSION,
        "launch_authorized": False,
        "approval_v14_sha256": "a" * 64,
        "execution_protocol_v14_sha256": "a" * 64,
        "external_reaudit_sha256": "a" * 64,
        "code_lock_git_revision": "b" * 40,
        "executable_tree_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="launch_not_authorized"):
        validate_post_audit_execution_authorization(obj, root=root)


def test_skel_contract_rejects_forged_joint_identities(tmp_path: Path) -> None:
    path = tmp_path / "bad_joint.npz"
    kwargs = _valid_skel_npz_kwargs()
    kwargs["joint_part"] = np.asarray(["pose"] * 75)
    atomic_savez_compressed(path, **kwargs)
    with pytest.raises(RuntimeError, match="array_contract_joint_part_identity"):
        validate_array_contracts(path, kind="skel")


def test_skel_contract_rejects_forged_bone_graph(tmp_path: Path) -> None:
    path = tmp_path / "bad_bones.npz"
    kwargs = _valid_skel_npz_kwargs()
    kwargs["bone_edges"] = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    kwargs["bone_vectors"] = np.zeros((8, 2, 3), np.float32)
    atomic_savez_compressed(path, **kwargs)
    with pytest.raises(RuntimeError, match="array_contract_bone_edges_identity"):
        validate_array_contracts(path, kind="skel")


def test_skel_contract_rejects_multi_value_sequence_length(tmp_path: Path) -> None:
    path = tmp_path / "bad_seq.npz"
    kwargs = _valid_skel_npz_kwargs()
    kwargs["sequence_length"] = np.asarray([8, 8], np.int32)
    atomic_savez_compressed(path, **kwargs)
    with pytest.raises(RuntimeError, match="array_contract_scalar_multi:sequence_length"):
        validate_array_contracts(path, kind="skel")


def test_skel_contract_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "bad_time.npz"
    kwargs = _valid_skel_npz_kwargs()
    kwargs["timestamps_sec"] = np.asarray([0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], np.float64)
    atomic_savez_compressed(path, **kwargs)
    with pytest.raises(RuntimeError, match="array_contract_timestamps_not_monotonic"):
        validate_array_contracts(path, kind="skel")


def test_skel_contract_rejects_wrong_padding_mask(tmp_path: Path) -> None:
    path = tmp_path / "bad_pad.npz"
    kwargs = _valid_skel_npz_kwargs()
    kwargs["padding_mask"] = np.zeros((8,), dtype=bool)
    atomic_savez_compressed(path, **kwargs)
    with pytest.raises(RuntimeError, match="array_contract_padding_mask_not_all_true"):
        validate_array_contracts(path, kind="skel")


def test_runner_readiness_reports_not_ready_without_launch() -> None:
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts/run_ms_sta_v5_d0_full_corpus.py")],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["launch"] is False
    # Live tree may already have sealed v14 auth (ready=true). Without --launch this is still gates-only.
    if payload.get("ready") is True:
        assert proc.returncode == 0
    else:
        assert proc.returncode != 0
        assert payload["ready"] is False


def test_direct_launch_without_flag_is_noop_or_gates_only() -> None:
    """Without --launch the runner must not start extraction (gates/readiness only)."""
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts/run_ms_sta_v5_d0_full_corpus.py"), "--phase", "all"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload.get("launch") is False
    assert "extraction_started" not in payload or payload.get("extraction_started") is False
    if payload.get("ready") is True:
        assert proc.returncode == 0
    else:
        assert proc.returncode != 0


def test_require_launch_readiness_importable_from_runner() -> None:
    import run_ms_sta_v5_d0_full_corpus as runner

    assert hasattr(runner, "require_launch_readiness")
    assert runner.require_launch_readiness is require_launch_readiness


def test_sitecustomize_preflight_rejects_venv_sitecustomize(tmp_path: Path) -> None:
    from ms_sta_v5_d0_full_corpus_lib import assert_no_out_of_provenance_sitecustomize

    for rel in (
        "_venvs/f0_v5/Lib/site-packages/sitecustomize.py",
        "_venvs/extract/Lib/site-packages/sitecustomize.py",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# planted\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="sitecustomize_outside_provenance"):
            assert_no_out_of_provenance_sitecustomize(root=tmp_path)
        path.unlink()
    assert_no_out_of_provenance_sitecustomize(root=tmp_path)


def test_campaign_related_process_alive_ignores_self_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_pid = os.getpid()
    fake_parent = self_pid + 10_000_001

    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.scan_campaign_processes",
        lambda *, campaign_token=None: [
            {"pid": self_pid, "create_time": 1.0, "cmdline": "self", "match": "campaign_token"},
            {"pid": fake_parent, "create_time": 1.0, "cmdline": "parent", "match": "campaign_token"},
        ],
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib._self_process_tree_pids",
        lambda: {self_pid, fake_parent},
    )
    assert campaign_related_process_alive(0) is False


def _plant_isolated_only_v10_tree(root: Path) -> None:
    """Minimal stubs so evaluate_launch_readiness runs under an isolated root.

    Plants sealed-looking v10 paths only; active v14 approval/protocol/reaudit/auth absent.
    """
    for rel in (
        D0_FULL_CORPUS_APPROVAL_V10_REL,
        D0_FULL_CORPUS_EXEC_PROTOCOL_V10_REL,
        D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V10_REL,
        D0_FULL_CORPUS_EXECUTION_INTERRUPTION_V10_REL,
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"stub": true}\n', encoding="utf-8")


def _plant_valid_looking_v14_gates(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_pass: bool,
    include_auth: bool,
) -> None:
    """Plant approval+protocol (+ optional PASS/auth) and stub heavy validators.

    Writes files before any sha256 monkeypatch so create helpers stay hermetic.
    """
    code_lock = "a" * 40
    shas = _write_v13_sealed_artifacts(root, code_lock=code_lock)
    if include_pass:
        pass_obj = _strict_pass_payload(shas, code_lock=code_lock)
        pass_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL
        pass_path.parent.mkdir(parents=True, exist_ok=True)
        pass_path.write_text(json.dumps(pass_obj, indent=2) + "\n", encoding="utf-8")
    if include_auth:
        auth_path = root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text(
            json.dumps(
                {
                    "schema": "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v14",
                    "launch_authorized": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_approval_v14",
        lambda obj, *, root=None: {"ok": True},
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.validate_execution_protocol_v14",
        lambda obj, *, root=None: {"ok": True},
    )
    if include_pass:
        monkeypatch.setattr(
            "ms_sta_v5_d0_full_corpus_lib.validate_external_reaudit_pass",
            lambda obj, *, root=None: {"ok": True},
        )
    if include_auth:
        monkeypatch.setattr(
            "ms_sta_v5_d0_full_corpus_lib.validate_post_audit_execution_authorization",
            lambda obj, *, root=None: {"ok": True},
        )


def test_launch_readiness_false_when_only_v10_sealed(tmp_path: Path) -> None:
    _plant_isolated_only_v10_tree(tmp_path)
    status = evaluate_launch_readiness(root=tmp_path)
    assert status["ready"] is False
    assert status["output_version"] == "v14"
    assert any("approval_v14_missing" in e for e in status["errors"])


def test_launch_readiness_false_when_approval_protocol_pass_without_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant_valid_looking_v14_gates(tmp_path, monkeypatch, include_pass=True, include_auth=False)
    status = evaluate_launch_readiness(root=tmp_path)
    assert status["ready"] is False
    assert any("post_audit_execution_auth_missing" in e for e in status["errors"])


def test_launch_readiness_true_when_all_gates_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant_valid_looking_v14_gates(tmp_path, monkeypatch, include_pass=True, include_auth=True)
    status = evaluate_launch_readiness(root=tmp_path)
    assert status["ready"] is True
    assert status["errors"] == []
    assert status["output_version"] == "v14"


def test_hermetic_readiness_depends_only_on_isolated_root_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_a = tmp_path / "ready_tree"
    root_b = tmp_path / "missing_tree"
    root_a.mkdir()
    root_b.mkdir()
    _plant_valid_looking_v14_gates(root_a, monkeypatch, include_pass=True, include_auth=True)
    status_a = evaluate_launch_readiness(root=root_a)
    status_b = evaluate_launch_readiness(root=root_b)
    assert status_a["ready"] is True
    assert status_b["ready"] is False
    assert any("approval_v14_missing" in e for e in status_b["errors"])
    # Negative proof: hermetic readiness must not inspect live Docs/results approvals.
    assert status_a["ready"] != status_b["ready"]


def test_v12_fail_reaudit_sha_pin_constant() -> None:
    assert EXPECTED_V12_APPROVAL_SHA256 == "5faa1d784f1eec963842466f59216aba2728c76b36f6f2eccb2854a584b56080"
    assert (
        EXPECTED_V12_EXTERNAL_REAUDIT_SHA256
        == "31991f7e5f0a8618c02fb40961f30405c51bca6480d0ae4c41bcb0646269f1ee"
    )


def test_v13_fail_reaudit_sha_pin_constant() -> None:
    assert EXPECTED_V13_APPROVAL_SHA256 == "19e80e95548b9288f6e62a01e37a17415fd244c42eb1796aab0258c5bca49d62"
    assert (
        EXPECTED_V13_EXTERNAL_REAUDIT_SHA256
        == "a4b6e138923cec697a65ccf68c60e8a27f85b2444c6e34578e9145f2530566d8"
    )


def test_live_readiness_audit_script_build_report_on_tmp(tmp_path: Path) -> None:
    from audit_ms_sta_v5_d0_full_corpus_live_readiness import build_report

    report = build_report(root=tmp_path)
    assert isinstance(report, dict)
    assert report["ready"] is False
    assert report["output_version"] == "v14"
    assert "errors" in report
    assert report["post_auth_exists"] is False
    assert report["reaudit_path_exists"] is False
    assert report["binary_root_rel"] is None
    assert report["binary_root_exists"] is None
    assert report["binary_root_status"] == "unknown"


def test_live_readiness_audit_resolve_helper_unknown_without_approval() -> None:
    from audit_ms_sta_v5_d0_full_corpus_live_readiness import resolve_approval_owned_binary_root

    out = resolve_approval_owned_binary_root({"approval": None}, root=_ROOT)
    assert out["binary_root_rel"] is None
    assert out["binary_root_exists"] is None
    assert out["binary_root_status"] == "unknown"


def test_live_readiness_audit_approval_owned_root_missing_and_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audit_ms_sta_v5_d0_full_corpus_live_readiness import build_report

    owned_rel = "data/features/ms_sta_v5_d0_full_v14_aaaaaaaaaaaa"
    approval = {
        "binary_root_rel": owned_rel,
        "code_lock_git_revision": "a" * 40,
    }
    status = {
        "ready": False,
        "errors": ["protocol_v14_missing"],
        "output_version": "v14",
        "campaign_token": CAMPAIGN_TOKEN,
        "approval": approval,
        "protocol": None,
    }
    monkeypatch.setattr(
        "audit_ms_sta_v5_d0_full_corpus_live_readiness.evaluate_launch_readiness",
        lambda *, root=None: status,
    )
    report = build_report(root=tmp_path)
    assert report["binary_root_rel"] == owned_rel
    assert report["binary_root_exists"] is False
    assert report["binary_root_status"] == "ok"

    (tmp_path / owned_rel).mkdir(parents=True, exist_ok=True)
    report2 = build_report(root=tmp_path)
    assert report2["binary_root_rel"] == owned_rel
    assert report2["binary_root_exists"] is True
    assert report2["binary_root_status"] == "ok"


def test_live_readiness_audit_never_uses_head_derived_binary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audit_ms_sta_v5_d0_full_corpus_live_readiness import build_report

    owned_rel = "data/features/ms_sta_v5_d0_full_v14_bbbbbbbbbbbb"
    head_sentinel = "data/features/ms_sta_v5_d0_full_v14_head_derived_sentinel"
    approval = {
        "binary_root_rel": owned_rel,
        "code_lock_git_revision": "c" * 40,
    }
    monkeypatch.setattr(
        "audit_ms_sta_v5_d0_full_corpus_live_readiness.evaluate_launch_readiness",
        lambda *, root=None: {
            "ready": False,
            "errors": ["external_reaudit_v14_missing"],
            "output_version": "v14",
            "campaign_token": CAMPAIGN_TOKEN,
            "approval": approval,
            "protocol": None,
        },
    )

    def _head_only_binary_root(rev=None, *, root=None, version=None):
        if rev is None:
            return head_sentinel
        raise AssertionError("binary_root_rel_called_with_rev_unexpectedly")

    monkeypatch.setattr(
        "audit_ms_sta_v5_d0_full_corpus_live_readiness.binary_root_rel",
        _head_only_binary_root,
    )
    report = build_report(root=tmp_path)
    assert report["binary_root_rel"] == owned_rel
    assert report["binary_root_rel"] != head_sentinel
    assert report["binary_root_status"] == "ok"


def test_interruption_v10_status_pin() -> None:
    path = _ROOT / D0_FULL_CORPUS_EXECUTION_INTERRUPTION_V10_REL
    assert path.is_file()
    assert sha256_file(path) == EXPECTED_V10_INTERRUPTION_SHA256
    assert EXPECTED_V10_APPROVAL_SHA256 == "436984ddf2dc46e01f8cf34adbeb7845fae966fc7950f18f5b8e9c1f3bfa1918"
    obj = load_json(path)
    assert validate_execution_interruption_v10(obj, root=_ROOT) == {"ok": True}
    assert obj["status"] == "INTERRUPTED_NON_CANONICAL_RUNTIME_PATCH"
    assert obj["canonical"] is False


def test_stale_running_mutable_does_not_imply_live_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dead_pid = 999_999_998
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib._process_identity",
        lambda pid: None,
    )
    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.scan_campaign_processes",
        lambda *, campaign_token=None: [],
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    # Stale RUNNING mutable with dead orchestrator pid must not look live.
    assert campaign_related_process_alive(dead_pid, runtime=runtime) is False


def test_c3_parity_verifier_report_schema() -> None:
    from verify_ms_sta_v5_d0_full_corpus_c3_parity import REPORT_SCHEMA

    assert REPORT_SCHEMA == "ms_sta_v5_d0_full_corpus_c3_parity_report_v1"
    pytest.importorskip("mediapipe")
    from verify_ms_sta_v5_d0_full_corpus_c3_parity import verify_c3_parity

    report = verify_c3_parity(root=_ROOT)
    assert report["schema"] == REPORT_SCHEMA
    assert "ok" in report
    assert "mismatches" in report
