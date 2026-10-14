"""Guards: empty dirs, run contracts, eligible manifest, validated promotion."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import (
    ELIGIBLE_MANIFEST_REL,
    assert_fresh_or_empty_dir,
    build_run_contract,
    count_npy,
    load_eligible_sample_manifest,
    require_resume_contract,
    sha256_file,
    write_run_contract,
)


def test_assert_fresh_rejects_nonempty(tmp_path: Path) -> None:
    d = tmp_path / "skel_v2"
    d.mkdir()
    (d / "a.npy").write_bytes(b"x")
    with pytest.raises(SystemExit, match="already has"):
        assert_fresh_or_empty_dir(d, resume=False, label="clean-d1")


def test_assert_fresh_allows_empty_and_resume(tmp_path: Path) -> None:
    d = tmp_path / "skel_v2"
    d.mkdir()
    assert_fresh_or_empty_dir(d, resume=False, label="clean-d1")
    (d / "a.npy").write_bytes(b"x")
    assert_fresh_or_empty_dir(d, resume=True, label="clean-d1")
    assert count_npy(d) == 1


def test_resume_requires_run_contract(tmp_path: Path) -> None:
    d = tmp_path / "out"
    d.mkdir()
    expected = build_run_contract(
        kind="skel_d1",
        sample_ids=["a", "b"],
        metadata_sha256="m",
        splits=["val"],
        config={"frame_stride": 2},
        root=_ROOT,
        sample_manifest_sha256="s",
    )
    with pytest.raises(SystemExit, match="_run_contract"):
        require_resume_contract(d, expected)


def test_resume_contract_match_and_mismatch(tmp_path: Path) -> None:
    d = tmp_path / "out"
    d.mkdir()
    expected = build_run_contract(
        kind="rgb_temporal",
        sample_ids=["a", "b"],
        metadata_sha256="m",
        splits=["val"],
        config={"max_windows": 8},
        root=_ROOT,
        sample_manifest_sha256="s",
    )
    write_run_contract(d, expected)
    require_resume_contract(d, expected)
    bad = dict(expected)
    bad["sample_id_sha256"] = "nope"
    with pytest.raises(SystemExit, match="mismatch"):
        require_resume_contract(d, bad)


def test_eligible_manifest_excludes_019442() -> None:
    path = _ROOT / ELIGIBLE_MANIFEST_REL
    pack = load_eligible_sample_manifest(path, root=_ROOT)
    assert "019442" not in pack["eligible_set"]
    assert pack["raw"]["excluded_n"] == 1
    assert pack["raw"]["eligible_by_split"]["test"] == 6165
    assert len(pack["eligible_ids"]) == 24752
    assert pack["raw"]["metadata"] == "metadata.csv"


def test_eligible_manifest_rejects_absolute_metadata(tmp_path: Path) -> None:
    src = _ROOT / ELIGIBLE_MANIFEST_REL
    man = json.loads(src.read_text(encoding="utf-8"))
    man["metadata"] = str((_ROOT / "metadata.csv").resolve())
    bad = tmp_path / "bad_eligible.json"
    bad.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="repo-relative"):
        load_eligible_sample_manifest(bad, root=_ROOT, require_metadata_match=False)


def _load_promote():
    path = _ROOT / "scripts" / "promote_feature_dir.py"
    spec = importlib.util.spec_from_file_location("promote_feature_dir", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_eligible(tmp_root: Path) -> str:
    dest = tmp_root / ELIGIBLE_MANIFEST_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_ROOT / ELIGIBLE_MANIFEST_REL, dest)
    # Point metadata to real repo metadata so validation can pass if needed
    return sha256_file(dest)


def test_promote_skel_requires_quality_and_manifest(tmp_path, monkeypatch) -> None:
    mod = _load_promote()
    d = tmp_path / "local_skel_v2_x"
    d.mkdir()
    (d / "1.npy").write_bytes(b"x")
    monkeypatch.setattr(mod, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.setattr(
        sys,
        "argv",
        ["promote_feature_dir.py", "--kind", "skel", "--dir", str(d)],
    )
    assert mod.main() == 2
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"pass": True, "full_scan": True}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_feature_dir.py",
            "--kind",
            "skel",
            "--dir",
            str(d),
            "--quality-report",
            str(q),
        ],
    )
    assert mod.main() == 2


def test_promote_temporal_refuses_dirty_without_flag(tmp_path, monkeypatch) -> None:
    mod = _load_promote()
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY", tmp_path / "data" / "processed" / "ms_sta_registry.json")
    elig_sha = _seed_eligible(tmp_path)
    d = tmp_path / "data" / "features" / "isolated_sf_temporal_v2_x"
    d.mkdir(parents=True)
    (d / "1.npy").write_bytes(b"x")
    write_run_contract(
        d,
        {
            "sample_id_sha256": "abc",
            "sample_manifest_sha256": elig_sha,
            "metadata_sha256": "meta",
        },
    )
    man = d / "_manifest.json"
    man.write_text(
        json.dumps(
            {
                "out_dir": "data/features/isolated_sf_temporal_v2_x",
                "failed": 0,
                "saved": 1,
                "skipped": 0,
                "n_sample_ids": 1,
                "sample_id_sha256": "abc",
                "sample_manifest_sha256": elig_sha,
                "metadata_sha256": "meta",
                "min_temporal_l": 2,
                "shape_contract": "(L, 2304)",
                "git_revision": "deadbeef",
                "git_dirty": True,
                "backbone": {"hub": "pytorchvideo.models.hub.slowfast_r50"},
            }
        ),
        encoding="utf-8",
    )
    q = tmp_path / "tq.json"
    q.write_text(
        json.dumps(
            {
                "pass": True,
                "full_scan": True,
                "temporal_dir": str(d.resolve()),
                "present": 1,
                "bad_shape": 0,
                "short_l": 0,
                "nan_files": 0,
                "inf_files": 0,
                "all_zero_files": 0,
                "dtype_mismatch": 0,
                "dtype_unified": True,
                "L_min": 2,
                "L_max": 3,
                "L_mean": 2.5,
            }
        ),
        encoding="utf-8",
    )
    cuda = tmp_path / "cuda_prov.json"
    cuda.write_text(
        json.dumps(
            {
                "pass": True,
                "git_revision": "deadbeef",
                "temporal_output_dir": "data/features/isolated_sf_temporal_v2_x",
                "pip_freeze_sha256": "pipsha",
                "backbone_checkpoint_identity": {"sha256": "bbsha"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_feature_dir.py",
            "--kind",
            "temporal",
            "--dir",
            "data/features/isolated_sf_temporal_v2_x",
            "--manifest",
            str(man),
            "--quality-report",
            str(q),
            "--cuda-provenance",
            str(cuda),
        ],
    )
    assert mod.main() == 2


def test_promote_temporal_ok(tmp_path, monkeypatch) -> None:
    mod = _load_promote()
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY", tmp_path / "data" / "processed" / "ms_sta_registry.json")
    elig_sha = _seed_eligible(tmp_path)
    d = tmp_path / "data" / "features" / "isolated_sf_temporal_v2_ok"
    d.mkdir(parents=True)
    (d / "1.npy").write_bytes(b"x")
    write_run_contract(
        d,
        {
            "sample_id_sha256": "abc",
            "sample_manifest_sha256": elig_sha,
            "metadata_sha256": "meta",
        },
    )
    man = d / "_manifest.json"
    man.write_text(
        json.dumps(
            {
                "out_dir": "data/features/isolated_sf_temporal_v2_ok",
                "failed": 0,
                "saved": 1,
                "skipped": 0,
                "n_sample_ids": 1,
                "sample_id_sha256": "abc",
                "sample_manifest_sha256": elig_sha,
                "metadata_sha256": "meta",
                "min_temporal_l": 2,
                "shape_contract": "(L, 2304)",
                "git_revision": "deadbeef",
                "git_dirty": False,
                "backbone": {"hub": "pytorchvideo.models.hub.slowfast_r50"},
            }
        ),
        encoding="utf-8",
    )
    q = tmp_path / "tq.json"
    q.write_text(
        json.dumps(
            {
                "pass": True,
                "full_scan": True,
                "temporal_dir": str(d.resolve()),
                "present": 1,
                "bad_shape": 0,
                "short_l": 0,
                "nan_files": 0,
                "inf_files": 0,
                "all_zero_files": 0,
                "dtype_mismatch": 0,
                "dtype_unified": True,
                "L_min": 2,
                "L_max": 3,
                "L_mean": 2.9,
            }
        ),
        encoding="utf-8",
    )
    cuda = tmp_path / "cuda_prov.json"
    cuda.write_text(
        json.dumps(
            {
                "pass": True,
                "git_revision": "deadbeef",
                "temporal_output_dir": "data/features/isolated_sf_temporal_v2_ok",
                "pip_freeze_sha256": "pipsha",
                "backbone_checkpoint_identity": {"sha256": "bbsha"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_feature_dir.py",
            "--kind",
            "temporal",
            "--dir",
            "data/features/isolated_sf_temporal_v2_ok",
            "--manifest",
            str(man),
            "--quality-report",
            str(q),
            "--cuda-provenance",
            str(cuda),
        ],
    )
    assert mod.main() == 0
    reg = json.loads((tmp_path / "data" / "processed" / "ms_sta_registry.json").read_text(encoding="utf-8"))
    assert reg["active_temporal_dir"] == "data/features/isolated_sf_temporal_v2_ok"
    ev = reg["features"]["temporal_promotion_evidence"]
    assert ev["pass"] is True
    assert ev["full_scan"] is True
    assert ev["backbone_checkpoint_sha256"] == "bbsha"
    assert ev["pip_freeze_sha256"] == "pipsha"
    assert ev["cuda_provenance_sha256"]
    assert ev["quality_report_sha256"]
    assert ev["run_contract_sha256"]


def test_promote_skel_ok(tmp_path, monkeypatch) -> None:
    mod = _load_promote()
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY", tmp_path / "data" / "processed" / "ms_sta_registry.json")
    elig_sha = _seed_eligible(tmp_path)
    # Registry empty so require_canonical only checks disk file under tmp_root
    d = tmp_path / "data" / "features" / "local_skel_v2_ok"
    d.mkdir(parents=True)
    (d / "1.npy").write_bytes(b"x")
    (d / "2.npy").write_bytes(b"y")
    write_run_contract(
        d,
        {
            "sample_id_sha256": "ids",
            "sample_manifest_sha256": elig_sha,
            "metadata_sha256": "meta",
        },
    )
    man = d / "_manifest_VSL400_all.json"
    man.write_text(
        json.dumps(
            {
                "gate": "D1",
                "output_dir": "data/features/local_skel_v2_ok",
                "sample_id_sha256": "ids",
                "sample_manifest_sha256": elig_sha,
                "metadata_sha256": "meta",
                "git_revision": "deadbeef",
                "git_dirty": False,
                "shape_contract": "(T, 75, 3)",
                "postprocess": {
                    "shoulder_fill": True,
                    "leading_joint_backfill": True,
                    "never_detected_to_origin": True,
                },
            }
        ),
        encoding="utf-8",
    )
    q = tmp_path / "q.json"
    q.write_text(
        json.dumps(
            {
                "pass": True,
                "full_scan": True,
                "skel_dir": str(d.resolve()),
                "coverage": 0.96,
                "present": 2,
                "bad_shape": 0,
                "nan_files": 0,
                "inf_files": 0,
                "zero_frame_files": 0,
                "outlier_scale_files": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_feature_dir.py",
            "--kind",
            "skel",
            "--dir",
            "data/features/local_skel_v2_ok",
            "--quality-report",
            str(q),
            "--manifest",
            str(man),
            "--min-coverage",
            "0.95",
        ],
    )
    assert mod.main() == 0
    reg = json.loads((tmp_path / "data" / "processed" / "ms_sta_registry.json").read_text(encoding="utf-8"))
    assert reg["active_skel_dir"] == "data/features/local_skel_v2_ok"
    assert reg["d1_pass"] is True
    assert reg["eligible_sample_manifest_sha256"] == elig_sha


def test_promote_skel_refuses_dirty_without_flag(tmp_path, monkeypatch) -> None:
    mod = _load_promote()
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    monkeypatch.setattr(mod, "REGISTRY", tmp_path / "data" / "processed" / "ms_sta_registry.json")
    elig_sha = _seed_eligible(tmp_path)
    d = tmp_path / "data" / "features" / "local_skel_v2_dirty"
    d.mkdir(parents=True)
    (d / "1.npy").write_bytes(b"x")
    write_run_contract(
        d,
        {
            "sample_id_sha256": "ids",
            "sample_manifest_sha256": elig_sha,
            "metadata_sha256": "meta",
        },
    )
    man = d / "_manifest.json"
    man.write_text(
        json.dumps(
            {
                "output_dir": "data/features/local_skel_v2_dirty",
                "sample_id_sha256": "ids",
                "sample_manifest_sha256": elig_sha,
                "metadata_sha256": "meta",
                "git_revision": "deadbeef",
                "git_dirty": True,
                "shape_contract": "(T, 75, 3)",
                "postprocess": {"shoulder_fill": True},
            }
        ),
        encoding="utf-8",
    )
    q = tmp_path / "q.json"
    q.write_text(
        json.dumps(
            {
                "pass": True,
                "full_scan": True,
                "skel_dir": str(d.resolve()),
                "coverage": 1.0,
                "present": 1,
                "bad_shape": 0,
                "nan_files": 0,
                "inf_files": 0,
                "zero_frame_files": 0,
                "outlier_scale_files": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_feature_dir.py",
            "--kind",
            "skel",
            "--dir",
            "data/features/local_skel_v2_dirty",
            "--quality-report",
            str(q),
            "--manifest",
            str(man),
        ],
    )
    assert mod.main() == 2
