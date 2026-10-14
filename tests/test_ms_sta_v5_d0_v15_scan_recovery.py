"""Hermetic v15 scan-recovery tests (no extraction / no live seal)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tests"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION,
    OUTPUT_VERSION_V15,
    canonical_sample_paths,
    commit_manifest_record,
    default_quality_thresholds,
    runtime_dir,
    sha256_file,
)
from scan_ms_sta_v5_d0_full_corpus import (  # noqa: E402
    SCAN_REPORT_SCHEMA_V15_RECOVERY,
    scan_full_corpus,
)
from test_ms_sta_v5_d0_full_corpus import _write_meta, _write_rgb, _write_skel  # noqa: E402


def test_executable_paths_include_scan_recovery() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_scan_recovery.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15
    assert (
        "scripts/run_ms_sta_v5_d0_full_corpus_scan_recovery_v15.py"
        in FULL_CORPUS_EXECUTABLE_PATHS_V15
    )


def _plant_dual_root(
    tmp_path: Path,
    *,
    ids: list[str],
    source_sha: str | None = None,
) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    """v14-like skel + v15-like rgb with independent manifests + real metadata/videos."""
    root = Path(tmp_path).resolve()
    skel_bin = root / "data" / "features" / "skel_v14"
    rgb_bin = root / "data" / "features" / "rgb_v15"
    skel_rt = runtime_dir(skel_bin)
    rgb_rt = runtime_dir(rgb_bin)
    skel_rt.mkdir(parents=True)
    rgb_rt.mkdir(parents=True)
    videos = root / "data" / "raw" / "VSL-Cropped" / "raw" / "raw" / "online_sourced"
    videos.mkdir(parents=True)
    source_sha_by: dict[str, str] = {}
    meta_lines = ["video_id,split,signer_id,dataset,view\n"]
    for sid in ids:
        vp = videos / f"{sid}.mp4"
        vp.write_bytes(b"fake-video-" + sid.encode("ascii"))
        file_dig = sha256_file(vp)
        declared = source_sha if source_sha is not None else file_dig
        source_sha_by[sid] = declared
        meta_lines.append(f"{sid},train,s1,VSL-Cropped,front\n")
        _write_skel(skel_bin / "skeleton_v3" / f"{sid}.npz")
        skel_side = skel_bin / "skeleton_v3" / f"{sid}.recovery.json"
        skel_side.write_text(json.dumps({"sample_id": sid}) + "\n", encoding="utf-8")
        dig_s = sha256_file(skel_bin / "skeleton_v3" / f"{sid}.npz")
        dig_s_side = sha256_file(skel_side)
        _write_rgb(rgb_bin / "rgb_token_v3" / f"{sid}.npz")
        _write_meta(rgb_bin / "rgb_token_v3" / f"{sid}.meta.json", sid)
        dig_r = sha256_file(rgb_bin / "rgb_token_v3" / f"{sid}.npz")
        dig_r_side = sha256_file(rgb_bin / "rgb_token_v3" / f"{sid}.meta.json")
        skel_out, skel_side_rel = canonical_sample_paths(
            root=root, binary=skel_bin, phase="skeleton_v3", sample_id=sid
        )
        rgb_out, rgb_side_rel = canonical_sample_paths(
            root=root, binary=rgb_bin, phase="rgb_token_v3", sample_id=sid
        )
        commit_manifest_record(
            skel_rt,
            "skeleton_v3",
            {
                "sample_id": sid,
                "split": "train",
                "signer_id": "s1",
                "source_sha256": declared,
                "output_sha256": dig_s,
                "sidecar_sha256": dig_s_side,
                "output_rel": skel_out,
                "sidecar_rel": skel_side_rel,
                "producer": "test",
            },
        )
        commit_manifest_record(
            rgb_rt,
            "rgb_token_v3",
            {
                "sample_id": sid,
                "split": "train",
                "signer_id": "s1",
                "source_sha256": declared,
                "skel_sha256": dig_s,
                "output_sha256": dig_r,
                "sidecar_sha256": dig_r_side,
                "output_rel": rgb_out,
                "sidecar_rel": rgb_side_rel,
                "producer": "test",
            },
        )
    (root / "metadata.csv").write_text("".join(meta_lines), encoding="utf-8")
    return skel_bin, rgb_bin, skel_rt, rgb_rt, source_sha_by


def _patch_identity(monkeypatch: pytest.MonkeyPatch, ids: list[str]) -> None:
    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus.load_eligible_identity",
        lambda root=None: {
            "eligible_ids": ids,
            "eligible_id_set": set(ids),
            "eligible_by_split": {"train": len(ids), "val": 0, "test": 0},
            "eligible_n": len(ids),
            "eligible_sha256": "x",
        },
    )
    monkeypatch.setattr("scan_ms_sta_v5_d0_full_corpus.CORPUS_N", len(ids))
    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus.SPLIT_COUNTS",
        {"train": len(ids), "val": 0, "test": 0},
    )
    monkeypatch.setattr(
        "scan_ms_sta_v5_d0_full_corpus._load_maps",
        lambda root: (
            {i: "train" for i in ids},
            {i: "s1" for i in ids},
            {
                i: {
                    "video_id": i,
                    "split": "train",
                    "signer_id": "s1",
                    "dataset": "VSL-Cropped",
                    "view": "front",
                }
                for i in ids
            },
        ),
    )


def test_positive_carry_forward_dual_root_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["000001", "000002", "000003"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)
    # Relax thresholds so tiny fixture is not judged on ROI rates.
    thr = default_quality_thresholds()
    thr.update(
        {
            "max_missing_roi_rate_face": 1.0,
            "max_missing_roi_rate_left_hand": 1.0,
            "max_missing_roi_rate_right_hand": 1.0,
        }
    )
    report = scan_full_corpus(
        binary_root=rgb_bin,
        runtime_dir=rgb_rt,
        thresholds=thr,
        root=tmp_path,
        skeleton_tensor_root=skel_bin,
        skeleton_manifest_runtime=skel_rt,
        rgb_tensor_root=rgb_bin,
        rgb_manifest_runtime=rgb_rt,
        output_version=OUTPUT_VERSION_V15,
        write_artifacts=True,
        output_runtime_dir=tmp_path / "recovery_rt",
    )
    assert report["ok"] is True
    assert report["output_version"] == OUTPUT_VERSION_V15
    assert report["schema"] == SCAN_REPORT_SCHEMA_V15_RECOVERY
    assert report["n_paired"] == 3
    assert report["failed_count"] == 0
    assert report["global_failure_count"] == 0
    assert report["unique_failed_sample_count"] == 0
    assert report["dual_root"] is True
    assert report["binds"]["skeleton_manifest"]["runtime_rel"].endswith("skel_v14/runtime")
    assert report["binds"]["rgb_manifest"]["runtime_rel"].endswith("rgb_v15/runtime")


def test_refuse_shared_manifest_runtime_when_dual_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["000001"]
    skel_bin, rgb_bin, _skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)
    with pytest.raises(RuntimeError, match="scan_manifest_runtimes_not_independently_bound"):
        scan_full_corpus(
            binary_root=rgb_bin,
            runtime_dir=rgb_rt,
            root=tmp_path,
            skeleton_tensor_root=skel_bin,
            skeleton_manifest_runtime=rgb_rt,  # wrong: same as RGB
            rgb_manifest_runtime=rgb_rt,
            output_version=OUTPUT_VERSION_V15,
            write_artifacts=False,
        )


def test_missing_skel_manifest_counts_as_global_not_24753_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduce prior defect class: empty skel manifest under RGB runtime."""
    ids = ["000001", "000002"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)
    thr = default_quality_thresholds()
    thr.update(
        {
            "max_missing_roi_rate_face": 1.0,
            "max_missing_roi_rate_left_hand": 1.0,
            "max_missing_roi_rate_right_hand": 1.0,
        }
    )
    # Bug-class call: skel tensors from skel_bin but manifests both from rgb_rt (empty skel man).
    # Explicit independent runtimes required — using skel_rt empty by wiping samples.
    import shutil

    shutil.rmtree(skel_rt / "manifests" / "skeleton_v3", ignore_errors=True)
    report = scan_full_corpus(
        binary_root=rgb_bin,
        runtime_dir=rgb_rt,
        thresholds=thr,
        root=tmp_path,
        skeleton_tensor_root=skel_bin,
        skeleton_manifest_runtime=skel_rt,
        rgb_tensor_root=rgb_bin,
        rgb_manifest_runtime=rgb_rt,
        output_version=OUTPUT_VERSION_V15,
        write_artifacts=False,
    )
    assert report["ok"] is False
    assert report["global_failure_count"] >= 1
    assert any(f.get("error") == "skel_manifest_id_set_mismatch" for f in report["failures"])
    # Unique sample failures for missing rows; not global+sample double-count into absurd totals.
    assert report["failed_count"] == report["global_failure_count"] + report["unique_failed_sample_count"]


def test_swapped_manifest_runtimes_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["000001", "000002"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)
    thr = default_quality_thresholds()
    thr.update(
        {
            "max_missing_roi_rate_face": 1.0,
            "max_missing_roi_rate_left_hand": 1.0,
            "max_missing_roi_rate_right_hand": 1.0,
        }
    )
    report = scan_full_corpus(
        binary_root=rgb_bin,
        runtime_dir=rgb_rt,
        thresholds=thr,
        root=tmp_path,
        skeleton_tensor_root=skel_bin,
        skeleton_manifest_runtime=rgb_rt,  # swapped
        rgb_tensor_root=rgb_bin,
        rgb_manifest_runtime=skel_rt,  # swapped
        output_version=OUTPUT_VERSION_V15,
        write_artifacts=False,
    )
    assert report["ok"] is False


def test_stale_output_version_rejected_for_dual_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["000001"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)
    with pytest.raises(RuntimeError, match="scan_dual_root_requires_output_version_v15"):
        scan_full_corpus(
            binary_root=rgb_bin,
            runtime_dir=rgb_rt,
            root=tmp_path,
            skeleton_tensor_root=skel_bin,
            skeleton_manifest_runtime=skel_rt,
            rgb_manifest_runtime=rgb_rt,
            output_version=OUTPUT_VERSION,  # stale v14
            write_artifacts=False,
        )


def test_wrong_sha_fails_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["000001"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)
    # Tamper RGB tensor bytes after manifest bind; keep missing_* flags so meta still agrees.
    rgb_path = rgb_bin / "rgb_token_v3" / "000001.npz"
    data = dict(np.load(rgb_path, allow_pickle=False))
    data["token_face"] = np.asarray(data["token_face"], dtype=np.float32).copy()
    data["token_face"][0, 0] = 1.0
    from ms_sta_v5_d0_full_corpus_lib import atomic_savez_compressed

    atomic_savez_compressed(rgb_path, **data)
    thr = default_quality_thresholds()
    thr.update(
        {
            "max_missing_roi_rate_face": 1.0,
            "max_missing_roi_rate_left_hand": 1.0,
            "max_missing_roi_rate_right_hand": 1.0,
        }
    )
    report = scan_full_corpus(
        binary_root=rgb_bin,
        runtime_dir=rgb_rt,
        thresholds=thr,
        root=tmp_path,
        skeleton_tensor_root=skel_bin,
        skeleton_manifest_runtime=skel_rt,
        rgb_manifest_runtime=rgb_rt,
        output_version=OUTPUT_VERSION_V15,
        write_artifacts=False,
    )
    assert report["ok"] is False
    assert report["unique_failed_sample_count"] == 1
    assert report["sample_failure_count"] == 1
    assert any(f.get("error") == "rgb_manifest_sha_mismatch" for f in report["failures"])
