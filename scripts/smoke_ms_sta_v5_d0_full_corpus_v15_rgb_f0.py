#!/usr/bin/env python3
"""v15 RGB F0 smoke: one frozen pilot sample under _venvs/f0_v5 only (no pytest).

Writes create-only smoke report schema v2 under v15 evidence.
Does not launch full-corpus RGB. Exit non-zero on failure.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

EXPECTED_F0_MARKER = "_venvs/f0_v5"


def _die(msg: str, code: int = 2) -> int:
    print(json.dumps({"ok": False, "error": msg}, indent=2))
    return code


def _pip_freeze_sha256(python_exe: Path) -> str:
    out = subprocess.check_output([str(python_exe), "-m", "pip", "freeze"], cwd=_ROOT)
    return hashlib.sha256(out).hexdigest()


def main() -> int:
    exe = Path(sys.executable).resolve()
    if EXPECTED_F0_MARKER.replace("/", "\\") not in str(exe) and EXPECTED_F0_MARKER not in str(exe).replace("\\", "/"):
        return _die(f"f0_python_required_got:{exe}")

    try:
        import importlib

        for mod in (
            "numpy",
            "torch",
            "cv2",
            "src.data_prep.raw_video_resolve",
            "src.models.rgb_token_v3",
        ):
            importlib.import_module(mod)
    except ModuleNotFoundError as exc:
        return _die(f"preflight_import_failed:{exc}")

    import numpy as np
    import torch

    from ms_sta_v5_d0_full_corpus_lib import (
        D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER_REL,
        D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL,
        FULL_CORPUS_EXECUTABLE_PATHS_V15,
        RGB_REQUIRED_KEYS,
        RGB_TOKEN_DIM,
        V14_SEALED_SKELETON_BINARY_ROOT_REL,
        atomic_create_json,
        validate_rgb_f0_smoke_report_v2,
    )
    from ms_sta_v5_d0_lib import (
        EXPECTED_SLOWFAST_CKPT_REL,
        EXPECTED_SLOWFAST_CKPT_SHA256,
        R7C_AMBIGUITY_ROI_LOCK,
        executable_tree_sha256,
        executable_tree_sha256_at_revision,
        git_porcelain,
        git_rev,
        sha256_file,
    )
    from run_ms_sta_v5_d0_full_corpus import _resolve_video, _rgb_arrays, preflight_rgb_f0_runtime
    from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video

    try:
        preflight_rgb_f0_runtime()
    except Exception as exc:  # noqa: BLE001
        return _die(f"preflight_rgb_f0_runtime:{exc}")

    backbone_path = _ROOT / EXPECTED_SLOWFAST_CKPT_REL
    if not backbone_path.is_file():
        return _die("backbone_missing")
    backbone_sha = sha256_file(backbone_path)
    if backbone_sha != EXPECTED_SLOWFAST_CKPT_SHA256:
        return _die("backbone_sha_drift")

    pilot_path = _ROOT / "Docs" / "results" / "ms_sta_v5_d0" / "D0_PILOT_SAMPLE_SET.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    sample_ids = list(pilot.get("sample_ids") or [])
    if not sample_ids:
        return _die("pilot_sample_ids_empty")
    sid = str(sample_ids[0])

    import csv

    meta_by: dict[str, dict[str, str]] = {}
    with (_ROOT / "metadata.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            meta_by[row["video_id"]] = row
    row = meta_by.get(sid)
    if row is None:
        return _die(f"metadata_missing:{sid}")

    skel_path = _ROOT / V14_SEALED_SKELETON_BINARY_ROOT_REL / "skeleton_v3" / f"{sid}.npz"
    if not skel_path.is_file():
        return _die(f"v14_skeleton_missing:{skel_path}")
    skeleton_sha = sha256_file(skel_path)

    if not torch.cuda.is_available():
        return _die("cuda_required_for_rgb_smoke")

    try:
        video = _resolve_video(_ROOT, row)
        source_video_sha = sha256_file(Path(video))
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
    except Exception as exc:  # noqa: BLE001
        return _die(f"rgb_extract_failed:{type(exc).__name__}:{exc}", code=1)

    errors: list[str] = []
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    for key in RGB_REQUIRED_KEYS:
        if key not in arrays:
            errors.append(f"missing_key:{key}")
            continue
        arr = np.asarray(arrays[key])
        shapes[key] = list(arr.shape)
        dtypes[key] = str(arr.dtype)
        if arr.size == 0:
            errors.append(f"empty:{key}")
            continue
        if np.issubdtype(arr.dtype, np.floating) and (not np.isfinite(arr).all()):
            errors.append(f"non_finite:{key}")
        if key == "token_global" and (arr.ndim < 2 or int(arr.shape[-1]) != RGB_TOKEN_DIM):
            errors.append(f"token_global_dim:{arr.shape}")

    required_keys_ok = not any(e.startswith("missing_key:") for e in errors)
    finite_ok = not any(e.startswith("non_finite:") for e in errors) and "token_global" in arrays
    token_content_sha256 = None
    if "token_global" in arrays:
        token = np.ascontiguousarray(np.asarray(arrays["token_global"]))
        token_content_sha256 = hashlib.sha256(token.tobytes()).hexdigest()

    rev = git_rev(_ROOT)
    dirty = bool(git_porcelain(_ROOT).strip())
    if dirty:
        # Worktree hash even when dirty; allow untracked remediation scripts during iteration.
        tree_sha = executable_tree_sha256(
            FULL_CORPUS_EXECUTABLE_PATHS_V15, root=_ROOT, require_git_tracked=False
        )
    else:
        tree_sha = executable_tree_sha256_at_revision(FULL_CORPUS_EXECUTABLE_PATHS_V15, rev, root=_ROOT)

    try:
        pip_sha = _pip_freeze_sha256(exe)
    except subprocess.CalledProcessError as exc:
        return _die(f"pip_freeze_failed:{exc}")

    report = {
        "schema": "ms_sta_v5_d0_full_corpus_v15_rgb_f0_smoke_report_v2",
        "status": "PASS" if not errors else "FAIL",
        "ok": not errors,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": "scripts/smoke_ms_sta_v5_d0_full_corpus_v15_rgb_f0.py",
        "code_lock_git_revision": rev,
        "executable_tree_sha256": tree_sha,
        "python_exe": str(exe),
        "pip_freeze_sha256": pip_sha,
        "backbone_sha256": backbone_sha,
        "source_video_sha256": source_video_sha,
        "skeleton_sha256": skeleton_sha,
        "sample_id": sid,
        "skeleton_source_rel": str(skel_path.relative_to(_ROOT)).replace("\\", "/"),
        "required_keys_ok": required_keys_ok,
        "shapes": shapes,
        "dtypes": dtypes,
        "finite_ok": finite_ok and not errors,
        "token_content_sha256": token_content_sha256,
        "errors": errors,
        "notes": [
            "Smoke report v2 — single-sample only; does not authorize full-corpus RGB launch.",
            "Do not install pandas into f0_v5.",
        ],
    }
    if report["ok"]:
        try:
            validate_rgb_f0_smoke_report_v2(report, root=_ROOT)
        except ValueError as exc:
            return _die(f"smoke_report_v2_self_check:{exc}")

    out = _ROOT / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL
    if out.is_file():
        prev = json.loads(out.read_text(encoding="utf-8"))
        prev_schema = str(prev.get("schema") or "")
        allow_replace = (
            prev_schema.endswith("_smoke_report_v1")
            or prev.get("status") == "FAIL"
            or prev.get("ok") is not True
        )
        if not allow_replace:
            for companion_rel in (
                D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER_REL,
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_INDEPENDENT_AUDIT_FAIL_ORCHESTRATION.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_WATCHDOG_COLD_START_FAIL.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_INTERRUPTED_RGB_WORKER_EXIT_WITHOUT_PROGRESS.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_INTERRUPTED_BOUND_WORKER_RECHECKED_COLD_START_GATE.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R2_AUDIT_FAIL_REMEDIATION.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R3_AUDIT_FAIL_REMEDIATION.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R4_EXECUTION_PROVENANCE_REMEDIATION.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R5_META_CONTRACT_REMEDIATION.json",
                "Docs/results/ms_sta_v5_d0_full/v15/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R6_FAIL_CLOSED_REMEDIATION.json",
            ):
                companion = _ROOT / companion_rel
                if companion.is_file():
                    prev_lock = str(prev.get("code_lock_git_revision") or "")
                    prev_tree = str(prev.get("executable_tree_sha256") or "")
                    if prev_lock != rev or prev_tree != tree_sha:
                        allow_replace = True
                        break
        if not allow_replace:
            return _die("smoke_report_pass_exists_create_only")
        out.unlink()
    try:
        sha = atomic_create_json(out, report)
    except FileExistsError as exc:
        return _die(f"smoke_report_exists_different:{exc}")

    print(json.dumps({"ok": report["ok"], "sample_id": sid, "report_sha256": sha, "errors": errors}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
