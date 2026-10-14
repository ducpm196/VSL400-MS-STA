"""v15 remediation tests: pandas-free RGB path, dict resolve, fail-fast HALT."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CORPUS_N,
    OUTPUT_VERSION,
    RGB_REQUIRED_KEYS,
    RGB_TOKEN_DIM,
    publish_deterministic_phase_failure_create_only,
    verify_skeleton_carry_forward_from_v14,
)


def test_hermetic_default_output_version_remains_v14():
    assert OUTPUT_VERSION == "v14"


def test_raw_video_resolve_has_no_pandas_import():
    src = (_ROOT / "src" / "data_prep" / "raw_video_resolve.py").read_text(encoding="utf-8")
    assert "import pandas" not in src
    assert "from pandas" not in src
    pandas_before = "pandas" in sys.modules
    importlib.import_module("src.data_prep.raw_video_resolve")
    if not pandas_before:
        assert "pandas" not in sys.modules


def test_resolve_video_from_dict(tmp_path):
    from src.data_prep.raw_video_resolve import RawVideoResolver

    vsl = tmp_path / "data" / "raw" / "VSL-Cropped" / "raw" / "raw" / "VSL400" / "Part_1" / "split_1"
    front = vsl / "front_view"
    front.mkdir(parents=True)
    (front / "abc.mp4").write_bytes(b"x")
    resolver = RawVideoResolver(tmp_path)
    row = {"dataset": "VSL400", "video_id": "abc", "signer_id": "signer_1"}
    path = resolver.resolve(row, view="front")
    assert path is not None and path.is_file()
    assert resolver.resolve(row, view="left") is None


def test_preflight_rgb_f0_runtime_ok():
    from run_ms_sta_v5_d0_full_corpus import preflight_rgb_f0_runtime

    preflight_rgb_f0_runtime()


def test_deterministic_dependency_failure_halts_without_fanout(tmp_path, monkeypatch):
    from run_ms_sta_v5_d0_full_corpus import (
        DeterministicDependencyError,
        run_rgb_phase,
    )

    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    binary = tmp_path / "binary"
    binary.mkdir(parents=True)
    (runtime / "RUNTIME_MUTABLE.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "resumes_used": 0,
                "restart_count": 0,
                "orchestrator_pid": None,
                "campaign_token": "D0_FULL_CORPUS_V14",
                "phase": "rgb_token_v3",
                "updated_at": "2026-08-16T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    def boom() -> None:
        raise DeterministicDependencyError("rgb_f0_preflight_missing_deps:pandas")

    monkeypatch.setattr("run_ms_sta_v5_d0_full_corpus.preflight_rgb_f0_runtime", boom)
    out = run_rgb_phase(binary=binary, runtime=runtime, root=_ROOT, resume=False, device="cpu")
    assert out.get("deterministic_dependency_failure") is True
    assert out.get("n_fail") == 0  # no per-sample fan-out
    det = json.loads((runtime / "DETERMINISTIC_PHASE_FAILURE.json").read_text(encoding="utf-8"))
    assert det["watchdog_resume_forbidden"] is True
    ledger = (runtime / "FAILURE_LEDGER.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger) == 1
    mutable = json.loads((runtime / "RUNTIME_MUTABLE.json").read_text(encoding="utf-8"))
    assert str(mutable["status"]).startswith("HALTED_")


def test_create_only_deterministic_failure_preserves_n_ok(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "RUNTIME_MUTABLE.json").write_text(
        json.dumps({"status": "RUNNING", "phase": "rgb_token_v3"}),
        encoding="utf-8",
    )
    summary = publish_deterministic_phase_failure_create_only(
        runtime,
        "rgb_token_v3",
        RuntimeError("boom"),
        n_ok=17,
        n_fail=0,
        n_resume=3,
        sample_id="019484",
    )
    assert summary["n_ok"] == 17
    assert summary["n_resume"] == 3
    assert summary["n_fail"] == 0
    rgb = json.loads((runtime / "RGB_PHASE_SUMMARY.json").read_text(encoding="utf-8"))
    assert rgb["n_ok"] == 17
    assert rgb["n_resume"] == 3
    # Idempotent same payload
    again = publish_deterministic_phase_failure_create_only(
        runtime,
        "rgb_token_v3",
        RuntimeError("boom"),
        n_ok=17,
        n_fail=0,
        n_resume=3,
        sample_id="019484",
    )
    assert again["n_ok"] == 17
    assert again["deterministic_failure_created"] is False
    # Different content must refuse overwrite
    with pytest.raises(FileExistsError):
        publish_deterministic_phase_failure_create_only(
            runtime,
            "rgb_token_v3",
            RuntimeError("different"),
            n_ok=17,
            n_fail=0,
            n_resume=3,
            sample_id="019484",
        )


def test_carry_forward_helper_rejects_mutation_paths(tmp_path, monkeypatch):
    """Unit-level fake tree: verifier is read-only and never copies/hardlinks."""
    root = tmp_path / "repo"
    binary = root / "data" / "features" / "ms_sta_v5_d0_full_v14_fake"
    runtime = binary / "runtime"
    skel = binary / "skeleton_v3"
    man = runtime / "manifests" / "skeleton_v3"
    man.mkdir(parents=True)
    skel.mkdir(parents=True)

    (runtime / "SKEL_PHASE_SUMMARY.json").write_text(
        json.dumps({"ok": True, "n_ok": CORPUS_N, "n_fail": 0, "phase": "skeleton_v3"}),
        encoding="utf-8",
        newline="\n",
    )
    (man / "INDEX.json").write_text(
        json.dumps({"n_records": CORPUS_N, "records": {}}),
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(
        "ms_sta_v5_d0_full_corpus_lib.load_eligible_identity",
        lambda *, root: {"eligible_ids": [f"{i:06d}" for i in range(CORPUS_N)], "eligible_id_set": set()},
    )

    before_files = {p.relative_to(binary) for p in binary.rglob("*") if p.is_file()}
    report = verify_skeleton_carry_forward_from_v14(
        root=root,
        sealed_binary_root_rel=str(binary.relative_to(root)).replace("\\", "/"),
        verify_all_shas=False,
    )
    after_files = {p.relative_to(binary) for p in binary.rglob("*") if p.is_file()}
    assert before_files == after_files
    assert report["mutated"] is False
    assert report["copied"] is False
    assert report["hardlinked"] is False
    assert "copy_skeleton_tree" in report["forbid"]
    assert report["ok"] is False  # incomplete fake tree (missing npz / index rows)


def test_rgb_required_keys_and_dim_contract():
    assert "token_global" in RGB_REQUIRED_KEYS
    assert RGB_TOKEN_DIM == 2304


@pytest.mark.requires_artifacts
def test_frozen_pilot_samples_resolve_from_dict():
    from src.data_prep.raw_video_resolve import RawVideoResolver

    sample_set = json.loads(
        (_ROOT / "Docs" / "results" / "ms_sta_v5_d0" / "D0_PILOT_SAMPLE_SET.json").read_text(encoding="utf-8")
    )
    resolver = RawVideoResolver(_ROOT)
    meta_by = {}
    import csv

    with (_ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meta_by[row["video_id"]] = row
    resolved = 0
    for sid in sample_set["sample_ids"]:
        row = meta_by.get(sid)
        assert row is not None
        path = resolver.resolve(row)
        if path is not None and path.is_file():
            resolved += 1
    assert resolved == len(sample_set["sample_ids"])


def test_rgb_smoke_one_frozen_sample_schema(tmp_path):
    """Optional GPU smoke: one frozen pilot sample → RGB schema/SHA fields."""
    import os

    if os.environ.get("D0_V15_RGB_SMOKE") != "1":
        pytest.skip("set D0_V15_RGB_SMOKE=1 to enable GPU RGB smoke")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuda_required_for_rgb_smoke")

    from run_ms_sta_v5_d0_full_corpus import _resolve_video, _rgb_arrays, _save_rgb_atomic
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video
    from ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK
    import csv

    sample_set = json.loads(
        (_ROOT / "Docs" / "results" / "ms_sta_v5_d0" / "D0_PILOT_SAMPLE_SET.json").read_text(encoding="utf-8")
    )
    sid = sample_set["sample_ids"][0]
    meta_by = {}
    with (_ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meta_by[row["video_id"]] = row
    row = meta_by[sid]
    video = _resolve_video(_ROOT, row)
    skel_path = _ROOT / "data" / "features" / "ms_sta_v5_d0_full_v14_2adbdf3d2bb5" / "skeleton_v3" / f"{sid}.npz"
    if not skel_path.is_file():
        pytest.skip("v14_skeleton_missing_for_smoke")
    skel = np.load(skel_path, allow_pickle=False)
    landmarks_by_frame: dict[int, np.ndarray] = {}
    missing_by_frame: dict[int, np.ndarray] = {}
    raw = skel["effective_landmarks"] if "effective_landmarks" in skel.files else skel["joints_raw"]
    miss = skel["effective_missing"] if "effective_missing" in skel.files else skel["missing_mask_prefill"]
    for i, fi in enumerate(skel["original_frame_index"].tolist()):
        landmarks_by_frame[int(fi)] = raw[i]
        missing_by_frame[int(fi)] = miss[i]
    extractor = FrozenSlowFastBlock4(device="cuda")
    payload = extract_rgb_token_v3_for_video(
        video_path=video,
        extractor=extractor,
        landmarks_by_frame=landmarks_by_frame,
        missing_by_frame=missing_by_frame,
        enable_pose_wrist_prior=False,
        enable_ambiguity_aware_prior=True,
        ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
    )
    arrays = _rgb_arrays(payload)
    for key in RGB_REQUIRED_KEYS:
        assert key in arrays
    assert arrays["token_global"].shape[1] == RGB_TOKEN_DIM
    binary = tmp_path / "bin"
    binary.mkdir()
    man_row = {
        "sample_id": sid,
        "split": "test",
        "signer_id": row.get("signer_id"),
        "source_sha256": "0" * 64,
        "skel_sha256": "0" * 64,
        "output_rel": f"rgb_token_v3/{sid}.npz",
        "sidecar_rel": f"rgb_token_v3/{sid}.meta.json",
        "output_sha256": None,
        "sidecar_sha256": None,
        "producer": "tests:rgb_smoke",
    }
    path, meta_path, dig = _save_rgb_atomic(binary, sid, payload, manifest_row=man_row, cleanup_staging=True)
    assert path.is_file() and meta_path.is_file()
    assert len(dig) == 64
