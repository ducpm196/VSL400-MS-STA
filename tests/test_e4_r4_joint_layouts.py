from src.data_prep.e4_r4_joint_layouts import (
    LAYOUT_S_A,
    LAYOUT_S_B,
    MEDIAPIPE_POSE_SELECTED,
    layout_indices,
    semantic_group_indices,
)
from src.data_prep.joint_layout_v3 import N_JOINTS


def test_layouts_subset_of_75():
    assert all(0 <= i < N_JOINTS for i in LAYOUT_S_A)
    assert all(0 <= i < N_JOINTS for i in LAYOUT_S_B)
    assert len(LAYOUT_S_A) == len(set(LAYOUT_S_A))
    assert len(LAYOUT_S_B) == len(set(LAYOUT_S_B))
    assert set(LAYOUT_S_A).issubset(set(LAYOUT_S_B))


def test_required_semantic_groups_present():
    for g in (
        "shoulders",
        "elbows",
        "wrists",
        "hand_mcp",
        "hand_fingertips",
        "mouth_anchors",
    ):
        idxs = semantic_group_indices(g)
        assert idxs
        assert set(idxs).issubset(LAYOUT_S_A)


def test_layout_id_roundtrip():
    assert layout_indices("S_A") == LAYOUT_S_A
    assert layout_indices("S_B") == LAYOUT_S_B


def test_pose9_semantic():
    assert MEDIAPIPE_POSE_SELECTED[3] == (3, 9, "mouth_left")
