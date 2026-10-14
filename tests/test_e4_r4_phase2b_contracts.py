"""E4-R4 Phase-2b contract tests — semantic IDs, RGB budget, fail-closed, -O safe."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.data_prep.e4_r4_joint_layouts import (
    FACE_MESH_ID_BY_LOCAL,
    LAYOUT_S_A,
    LAYOUT_S_B,
    MEDIAPIPE_FACE_SELECTED,
    MEDIAPIPE_POSE_SELECTED,
    MOUTH_ANCHOR_FACE_NAMES,
    layout_indices,
    semantic_group_indices,
    validate_layout_invariants,
)
from src.data_prep.e4_r4_rgb_contract import (
    is_non_compact,
    pixels_per_frame,
    theoretical_work_units,
    validate_rgb_budget_table,
)
from src.data_prep.e4_r4_skel_compact import compact_skeleton_arrays
from src.data_prep.joint_layout_v3 import LEFT_HAND, N_JOINTS, RIGHT_HAND


def test_pose9_not_mouth_center():
    assert MEDIAPIPE_POSE_SELECTED[3][1] == 9
    assert MEDIAPIPE_POSE_SELECTED[3][2] == "mouth_left"
    assert MEDIAPIPE_POSE_SELECTED[3][2] != "mouth_center"


def test_face_original_ids_exact():
    expected = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    ]
    got = [mp for _l, mp, _n in MEDIAPIPE_FACE_SELECTED]
    assert got == expected
    assert FACE_MESH_ID_BY_LOCAL[0] == 61
    assert FACE_MESH_ID_BY_LOCAL[10] == 291
    assert FACE_MESH_ID_BY_LOCAL[16] == 14


def test_mouth_anchors_bind_corners_and_vertical():
    names = set(MOUTH_ANCHOR_FACE_NAMES)
    assert "lip_outer_left" in names
    assert "lip_outer_right" in names
    assert "lip_outer_bottom" in names
    assert "lip_inner_lower" in names
    idxs = semantic_group_indices("mouth_anchors")
    assert len(idxs) == 4
    assert set(idxs).issubset(LAYOUT_S_A)


def test_uniqueness_subset_chirality():
    validate_layout_invariants()
    assert len(LAYOUT_S_A) == len(set(LAYOUT_S_A))
    assert set(LAYOUT_S_A).issubset(set(LAYOUT_S_B))
    assert all(0 <= i < N_JOINTS for i in LAYOUT_S_B)
    li = semantic_group_indices("hand_mcp")[0]
    ri = semantic_group_indices("hand_mcp")[5]
    assert LEFT_HAND.start <= li < LEFT_HAND.stop
    assert RIGHT_HAND.start <= ri < RIGHT_HAND.stop


def test_required_presence_groups():
    for g in ("shoulders", "elbows", "wrists", "hand_mcp", "hand_fingertips", "mouth_anchors"):
        idxs = semantic_group_indices(g)
        assert idxs
        assert set(idxs).issubset(LAYOUT_S_A)


def test_rgb_budget_compact_after_redesign():
    rows = validate_rgb_budget_table()
    assert rows["R0"]["vs_r0"] == 1.0
    assert rows["R1"]["non_compact"] is False
    assert rows["R2"]["non_compact"] is False
    assert rows["R1"]["pixels_per_frame"] <= rows["R0"]["pixels_per_frame"]
    assert rows["R2"]["pixels_per_frame"] <= rows["R0"]["pixels_per_frame"]
    assert pixels_per_frame("R1") == 160 * 160 * 3
    assert theoretical_work_units("R1") <= theoretical_work_units("R0")
    assert is_non_compact("R3") is False  # ablation flag separate
    assert rows["R3"]["promote_forbidden"] is True


def test_skel_compact_from_synthetic_sealed():
    t, j = 5, 75
    sealed = {
        "coordinates": np.zeros((t, j, 3), dtype=np.float32),
        "velocity": np.zeros((t, j, 3), dtype=np.float32),
        "validity": np.ones((t, j), dtype=np.float32),
        "confidence": np.ones((t, j), dtype=np.float32),
        "missing_mask_prefill": np.zeros((t, j), dtype=bool),
        "original_frame_index": np.arange(t, dtype=np.int32),
        "timestamps_sec": np.arange(t, dtype=np.float64) * 0.04,
        "padding_mask": np.zeros((t,), dtype=bool),
        "sequence_length": np.asarray([t], dtype=np.int32),
        "joint_part": np.array(["pose"] * 12 + ["left_hand"] * 21 + ["right_hand"] * 21 + ["face"] * 21),
        "joint_chirality": np.array(["none"] * 12 + ["left"] * 21 + ["right"] * 21 + ["none"] * 21),
    }
    out = compact_skeleton_arrays(sealed, "S_A")
    assert out["coordinates"].shape == (t, len(LAYOUT_S_A), 3)
    assert str(out["source_mode"]) == "sealed_75joint_read_only_compact_transform"


def _assert_no_assert_stmts(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            raise AssertionError(f"assert_forbidden_in_producer:{path}:{node.lineno}")


def test_producers_have_no_assert_statements():
    files = [
        ROOT / "src/data_prep/e4_r4_joint_layouts.py",
        ROOT / "src/data_prep/e4_r4_rgb_contract.py",
        ROOT / "src/data_prep/e4_r4_skel_compact.py",
        ROOT / "src/data_prep/e4_r4_rgb_pack.py",
        ROOT / "scripts/e4_r4_freeze_pilot_samples.py",
        ROOT / "scripts/e4_r4_extract_pilot.py",
        ROOT / "scripts/e4_r4_validate_pilot.py",
    ]
    for p in files:
        assert p.is_file(), p
        _assert_no_assert_stmts(p)


def test_import_contracts_under_python_dash_O():
    """Negative/positive: contracts must raise even when python -O strips assert."""
    code = (
        "from src.data_prep.e4_r4_joint_layouts import validate_layout_invariants, "
        "MEDIAPIPE_POSE_SELECTED\n"
        "validate_layout_invariants()\n"
        "assert MEDIAPIPE_POSE_SELECTED[3][2] != 'mouth_center'\n"
        "from src.data_prep.e4_r4_rgb_contract import validate_rgb_budget_table\n"
        "validate_rgb_budget_table()\n"
        "print('OK')\n"
    )
    for opt in ([], ["-O"]):
        r = subprocess.run(
            [sys.executable, *opt, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        assert "OK" in r.stdout


def test_negative_runtimeerror_survives_dash_O():
    code = (
        "from src.data_prep.e4_r4_joint_layouts import layout_indices\n"
        "try:\n"
        "    layout_indices('S_Z')\n"
        "except KeyError as e:\n"
        "    print('CAUGHT', type(e).__name__)\n"
        "else:\n"
        "    raise SystemExit('expected_keyerror')\n"
    )
    for opt in ([], ["-O"]):
        r = subprocess.run(
            [sys.executable, *opt, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        assert "CAUGHT KeyError" in r.stdout


def test_freezer_rejects_wrong_split_sha(tmp_path, monkeypatch):
    # Smoke: freezer module exits 2 when EXPECTED_SPLIT_SHA mismatched via env not available —
    # instead call die path by running script after monkeypatching constant.
    import scripts.e4_r4_freeze_pilot_samples as fr

    monkeypatch.setattr(fr, "EXPECTED_SPLIT_SHA", "deadbeef" * 8)
    with pytest.raises(SystemExit) as ei:
        fr.main()
    assert ei.value.code == 2


def test_layout_id_roundtrip():
    assert layout_indices("S_A") == LAYOUT_S_A
    assert layout_indices("S_B") == LAYOUT_S_B
