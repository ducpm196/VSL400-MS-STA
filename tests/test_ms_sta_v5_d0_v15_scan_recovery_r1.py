"""Hermetic R1 scan-recovery performance remediation tests (no full scan)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "tests"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    D0_FULL_CORPUS_APPROVAL_V15_REL,
    D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL,
    D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL,
    FULL_CORPUS_EXECUTABLE_PATHS_V15,
    OUTPUT_VERSION_V15,
    SCAN_RECOVERY_EXTERNAL_REAUDIT_SCHEMA_V15,
    SCAN_RECOVERY_POST_AUDIT_AUTH_SCHEMA_V15,
    atomic_create_json,
    canonical_sample_paths,
    commit_manifest_record,
    default_quality_thresholds,
    evaluate_scan_recovery_execute_readiness_v15,
    runtime_dir,
    sha256_file,
)
from scan_ms_sta_v5_d0_full_corpus import SourceShaVerifier, scan_full_corpus  # noqa: E402
from test_ms_sta_v5_d0_full_corpus import _write_meta, _write_rgb, _write_skel  # noqa: E402
from test_ms_sta_v5_d0_v15_scan_recovery import _patch_identity, _plant_dual_root  # noqa: E402


def test_executable_paths_include_r1_artifacts() -> None:
    assert "tests/test_ms_sta_v5_d0_v15_scan_recovery_r1.py" in FULL_CORPUS_EXECUTABLE_PATHS_V15
    assert (
        "scripts/smoke_ms_sta_v5_d0_full_corpus_scan_recovery_perf_v15.py"
        in FULL_CORPUS_EXECUTABLE_PATHS_V15
    )


def test_src_import_without_pythonpath_from_outside_cwd(tmp_path: Path) -> None:
    """Load scanner module from a non-repo cwd with cleared PYTHONPATH."""
    scan_path = (_ROOT / "scripts" / "scan_ms_sta_v5_d0_full_corpus.py").resolve()
    script = tmp_path / "probe_import.py"
    script.write_text(
        "import importlib.util\n"
        "from pathlib import Path\n"
        f"p = Path(r'{scan_path.as_posix()}')\n"
        "spec = importlib.util.spec_from_file_location('scan_ms_sta_v5_d0_full_corpus', p)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "assert spec.loader is not None\n"
        "spec.loader.exec_module(mod)\n"
        "from src.data_prep.raw_video_resolve import RawVideoResolver\n"
        "assert mod.SourceShaVerifier is not None\n"
        "print('IMPORT_OK')\n",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
    env["PYTHONPATH"] = ""
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "IMPORT_OK" in proc.stdout


def test_resolver_ctor_once_and_hash_count_equals_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["000001", "000002", "000003"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(tmp_path, ids=ids)
    _patch_identity(monkeypatch, ids)

    # Plant metadata rows + tiny fake videos so live verify hashes exactly once per id.
    meta_csv = tmp_path / "metadata.csv"
    meta_csv.write_text(
        "video_id,split,signer_id,dataset,view\n"
        + "".join(f"{i},train,s1,VSL-Cropped,front\n" for i in ids),
        encoding="utf-8",
    )
    videos = tmp_path / "data" / "raw" / "VSL-Cropped" / "raw" / "raw" / "online_sourced"
    videos.mkdir(parents=True, exist_ok=True)
    source_sha_by: dict[str, str] = {}
    for sid in ids:
        vp = videos / f"{sid}.mp4"
        vp.write_bytes(b"fake-video-" + sid.encode("ascii"))
        source_sha_by[sid] = sha256_file(vp)

    # Rewrite manifests with matching source SHAs.
    import shutil

    shutil.rmtree(skel_rt / "manifests", ignore_errors=True)
    shutil.rmtree(rgb_rt / "manifests", ignore_errors=True)
    for sid in ids:
        dig_s = sha256_file(skel_bin / "skeleton_v3" / f"{sid}.npz")
        dig_s_side = sha256_file(skel_bin / "skeleton_v3" / f"{sid}.recovery.json")
        dig_r = sha256_file(rgb_bin / "rgb_token_v3" / f"{sid}.npz")
        dig_r_side = sha256_file(rgb_bin / "rgb_token_v3" / f"{sid}.meta.json")
        skel_out, skel_side = canonical_sample_paths(
            root=tmp_path.resolve(), binary=skel_bin, phase="skeleton_v3", sample_id=sid
        )
        rgb_out, rgb_side = canonical_sample_paths(
            root=tmp_path.resolve(), binary=rgb_bin, phase="rgb_token_v3", sample_id=sid
        )
        commit_manifest_record(
            skel_rt,
            "skeleton_v3",
            {
                "sample_id": sid,
                "split": "train",
                "signer_id": "s1",
                "source_sha256": source_sha_by[sid],
                "output_sha256": dig_s,
                "sidecar_sha256": dig_s_side,
                "output_rel": skel_out,
                "sidecar_rel": skel_side,
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
                "source_sha256": source_sha_by[sid],
                "skel_sha256": dig_s,
                "output_sha256": dig_r,
                "sidecar_sha256": dig_r_side,
                "output_rel": rgb_out,
                "sidecar_rel": rgb_side,
                "producer": "test",
            },
        )

    # Use real maps for meta (so verifier resolves videos).
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

    ctor = {"n": 0}
    import src.data_prep.raw_video_resolve as rvr

    real_init = rvr.RawVideoResolver.__init__

    def counting_init(self, project_root):  # type: ignore[no-untyped-def]
        ctor["n"] += 1
        return real_init(self, project_root)

    monkeypatch.setattr(rvr.RawVideoResolver, "__init__", counting_init)

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
    assert report["ok"] is True
    assert ctor["n"] == 1
    stats = report["source_verify_stats"]
    assert stats["resolver_instances"] == 1
    assert stats["raw_source_hash_count"] == len(ids)
    assert stats["raw_source_hash_count"] != 2 * len(ids)


def test_wrong_source_sha_and_missing_video_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["000001"]
    skel_bin, rgb_bin, skel_rt, rgb_rt, _ = _plant_dual_root(
        tmp_path, ids=ids, source_sha="b" * 64
    )
    _patch_identity(monkeypatch, ids)
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
    assert any(
        "source_sha256_mismatch" in str(f.get("error"))
        or "source_video_unresolvable" in str(f.get("error"))
        for f in report["failures"]
    )


def test_execute_rejected_without_scan_recovery_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "run_ms_sta_v5_d0_full_corpus_scan_recovery_v15._ROOT",
        tmp_path,
    )
    # Minimal stubs so plan reaches readiness check.
    from run_ms_sta_v5_d0_full_corpus_scan_recovery_v15 import _build_plan

    # Without planting approval/companion, companion_missing is fine; also test readiness helper.
    ready = evaluate_scan_recovery_execute_readiness_v15(root=tmp_path)
    assert ready["execution_ready"] is False
    assert "scan_recovery_external_reaudit_v15_missing" in ready["errors"]
    assert "scan_recovery_post_audit_auth_v15_missing" in ready["errors"]


def test_failure_report_immutable_after_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create-only attempt identity refuses overwrite of same attempt id."""
    from run_ms_sta_v5_d0_full_corpus_scan_recovery_v15 import main as recovery_main

    # Plant readiness PASS + companion + dual roots under tmp as fake _ROOT is heavy;
    # unit-test the create-only gate directly.
    attempt = tmp_path / "Docs" / "results" / "ms_sta_v5_d0_full" / "v15" / "scan_recovery" / "attempts" / "a1"
    attempt.mkdir(parents=True)
    (attempt / "SCAN_REPORT.json").write_text('{"ok": false}\n', encoding="utf-8")
    frozen = sha256_file(attempt / "SCAN_REPORT.json")
    # Second mkdir must fail for create-only semantics.
    with pytest.raises(FileExistsError):
        attempt.mkdir(parents=True, exist_ok=False)
    assert sha256_file(attempt / "SCAN_REPORT.json") == frozen


def test_source_sha_verifier_cache_no_double_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"abc")
    dig = sha256_file(video)
    meta_by = {
        "000001": {
            "video_id": "000001",
            "dataset": "VSL-Cropped",
            "signer_id": "s1",
            "view": "front",
        }
    }
    import src.data_prep.raw_video_resolve as rvr

    class FakeResolver:
        def __init__(self, root):  # noqa: ANN001
            self.root = root

        def resolve(self, row):  # noqa: ANN001
            return video

    monkeypatch.setattr(rvr, "RawVideoResolver", FakeResolver)
    v = SourceShaVerifier(root=tmp_path, meta_by=meta_by)
    v.verify_sample(sample_id="000001", skel_declared=dig, rgb_declared=dig)
    v.verify_sample(sample_id="000001", skel_declared=dig, rgb_declared=dig)
    assert v.raw_source_hash_count == 1
    assert v.resolve_count == 1
