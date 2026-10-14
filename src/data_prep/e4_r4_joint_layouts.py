"""E4-R4 compact Skeleton layouts — semantic names bound to verified MediaPipe IDs.

Source of truth for the 21 face points and 12 pose points:
  src/data_prep/extract_and_smooth.py (SignLanguageDataEngine.pose_indices / face_indices)

Rules:
  - Never invent a semantic name without a verified original MediaPipe ID.
  - Pose landmark 9 is mouth_left (not mouth_center).
  - Production contracts raise RuntimeError/ValueError (no assert).
"""

from __future__ import annotations

from typing import Final

from src.data_prep.joint_layout_v3 import FACE, LEFT_HAND, N_JOINTS, POSE, RIGHT_HAND

# (local_index_in_12pose_block, mediapipe_pose_id, semantic_name)
MEDIAPIPE_POSE_SELECTED: Final[tuple[tuple[int, int, str], ...]] = (
    (0, 0, "nose"),
    (1, 2, "left_eye"),
    (2, 5, "right_eye"),
    (3, 9, "mouth_left"),  # MediaPipe Pose 9 = mouth left — NOT midpoint
    (4, 11, "left_shoulder"),
    (5, 12, "right_shoulder"),
    (6, 13, "left_elbow"),
    (7, 14, "right_elbow"),
    (8, 15, "left_wrist"),
    (9, 16, "right_wrist"),
    (10, 23, "left_hip"),
    (11, 24, "right_hip"),
)

# (local_index_in_21face_block, mediapipe_face_mesh_id, semantic_role_or_id_label)
# Topology copied exactly from extract_and_smooth.face_indices.
# Semantic lip roles assigned only for landmarks with established Face Mesh mouth IDs.
MEDIAPIPE_FACE_SELECTED: Final[tuple[tuple[int, int, str], ...]] = (
    (0, 61, "lip_outer_left"),
    (1, 146, "face_mesh_146"),
    (2, 91, "face_mesh_91"),
    (3, 181, "face_mesh_181"),
    (4, 84, "face_mesh_84"),
    (5, 17, "lip_outer_bottom"),
    (6, 314, "face_mesh_314"),
    (7, 405, "face_mesh_405"),
    (8, 321, "face_mesh_321"),
    (9, 375, "face_mesh_375"),
    (10, 291, "lip_outer_right"),
    (11, 78, "lip_inner_left"),
    (12, 95, "face_mesh_95"),
    (13, 88, "face_mesh_88"),
    (14, 178, "face_mesh_178"),
    (15, 87, "face_mesh_87"),
    (16, 14, "lip_inner_lower"),
    (17, 317, "face_mesh_317"),
    (18, 402, "face_mesh_402"),
    (19, 318, "face_mesh_318"),
    (20, 324, "face_mesh_324"),
)

POSE_LOCAL_BY_NAME: Final[dict[str, int]] = {name: loc for loc, _mp, name in MEDIAPIPE_POSE_SELECTED}
FACE_LOCAL_BY_NAME: Final[dict[str, int]] = {name: loc for loc, _mp, name in MEDIAPIPE_FACE_SELECTED}
FACE_MESH_ID_BY_LOCAL: Final[dict[int, int]] = {loc: mp for loc, mp, _ in MEDIAPIPE_FACE_SELECTED}
POSE_MP_ID_BY_LOCAL: Final[dict[int, int]] = {loc: mp for loc, mp, _ in MEDIAPIPE_POSE_SELECTED}

HAND_LOCAL_BY_NAME: Final[dict[str, int]] = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}

# Mouth anchors required by policy: L/R lip corners + vertical lip anchors present in 21-pt source.
MOUTH_ANCHOR_FACE_NAMES: Final[tuple[str, ...]] = (
    "lip_outer_left",
    "lip_outer_right",
    "lip_outer_bottom",
    "lip_inner_lower",
)


def _validate_tables() -> None:
    if len(MEDIAPIPE_POSE_SELECTED) != 12:
        raise RuntimeError("pose_selected_len_not_12")
    if len(MEDIAPIPE_FACE_SELECTED) != 21:
        raise RuntimeError("face_selected_len_not_21")
    pose_locs = [t[0] for t in MEDIAPIPE_POSE_SELECTED]
    face_locs = [t[0] for t in MEDIAPIPE_FACE_SELECTED]
    if pose_locs != list(range(12)):
        raise RuntimeError("pose_local_indices_not_contiguous")
    if face_locs != list(range(21)):
        raise RuntimeError("face_local_indices_not_contiguous")
    if len(set(POSE_LOCAL_BY_NAME)) != 12:
        raise RuntimeError("pose_semantic_names_not_unique")
    if len(set(FACE_LOCAL_BY_NAME)) != 21:
        raise RuntimeError("face_semantic_names_not_unique")
    # Exact MediaPipe IDs must match extract_and_smooth.py
    expected_pose_mp = [0, 2, 5, 9, 11, 12, 13, 14, 15, 16, 23, 24]
    expected_face_mp = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    ]
    got_pose = [mp for _l, mp, _n in MEDIAPIPE_POSE_SELECTED]
    got_face = [mp for _l, mp, _n in MEDIAPIPE_FACE_SELECTED]
    if got_pose != expected_pose_mp:
        raise RuntimeError(f"pose_mp_ids_mismatch:{got_pose}")
    if got_face != expected_face_mp:
        raise RuntimeError(f"face_mp_ids_mismatch:{got_face}")
    if MEDIAPIPE_POSE_SELECTED[3][2] == "mouth_center":
        raise RuntimeError("pose9_must_not_be_named_mouth_center")
    for name in MOUTH_ANCHOR_FACE_NAMES:
        if name not in FACE_LOCAL_BY_NAME:
            raise RuntimeError(f"missing_mouth_anchor:{name}")


_validate_tables()


def _pose(name: str) -> int:
    if name not in POSE_LOCAL_BY_NAME:
        raise ValueError(f"unknown_pose_semantic:{name}")
    return POSE.start + POSE_LOCAL_BY_NAME[name]


def _face(name: str) -> int:
    if name not in FACE_LOCAL_BY_NAME:
        raise ValueError(f"unknown_face_semantic:{name}")
    return FACE.start + FACE_LOCAL_BY_NAME[name]


def _hand(which: str, name: str) -> int:
    if which not in {"left", "right"}:
        raise ValueError(f"bad_hand_side:{which}")
    if name not in HAND_LOCAL_BY_NAME:
        raise ValueError(f"unknown_hand_semantic:{name}")
    base = LEFT_HAND.start if which == "left" else RIGHT_HAND.start
    return base + HAND_LOCAL_BY_NAME[name]


def semantic_group_indices(group: str) -> tuple[int, ...]:
    if group == "shoulders":
        return (_pose("left_shoulder"), _pose("right_shoulder"))
    if group == "elbows":
        return (_pose("left_elbow"), _pose("right_elbow"))
    if group == "wrists":
        return (_pose("left_wrist"), _pose("right_wrist"))
    if group == "hips":
        return (_pose("left_hip"), _pose("right_hip"))
    if group == "mouth_anchors":
        return tuple(_face(n) for n in MOUTH_ANCHOR_FACE_NAMES)
    if group == "hand_mcp":
        names = ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp", "thumb_mcp")
        return tuple(_hand(side, n) for side in ("left", "right") for n in names)
    if group == "hand_fingertips":
        names = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
        return tuple(_hand(side, n) for side in ("left", "right") for n in names)
    if group == "face_sparse_contour":
        # Sparse: keep named lip anchors + a few unlabeled mesh IDs for contour continuity.
        names = (
            "lip_outer_left",
            "face_mesh_146",
            "lip_outer_bottom",
            "face_mesh_321",
            "lip_outer_right",
            "lip_inner_left",
            "lip_inner_lower",
        )
        return tuple(_face(n) for n in names)
    raise KeyError(f"unknown_semantic_group:{group}")


def layout_indices(layout_id: str) -> tuple[int, ...]:
    if layout_id == "S_A":
        groups = (
            "shoulders",
            "elbows",
            "wrists",
            "hand_mcp",
            "hand_fingertips",
            "mouth_anchors",
        )
    elif layout_id == "S_B":
        groups = (
            "shoulders",
            "elbows",
            "wrists",
            "hips",
            "hand_mcp",
            "hand_fingertips",
            "mouth_anchors",
            "face_sparse_contour",
        )
    else:
        raise KeyError(f"unknown_layout:{layout_id}")
    out: list[int] = []
    seen: set[int] = set()
    for g in groups:
        for i in semantic_group_indices(g):
            if i in seen:
                continue
            if not (0 <= i < N_JOINTS):
                raise RuntimeError(f"joint_out_of_range:{i}")
            seen.add(i)
            out.append(i)
    return tuple(out)


def validate_layout_invariants() -> None:
    """Fail-closed invariants for producers (safe under python -O)."""
    _validate_tables()
    s_a = layout_indices("S_A")
    s_b = layout_indices("S_B")
    if len(s_a) != len(set(s_a)) or len(s_b) != len(set(s_b)):
        raise RuntimeError("layout_duplicate_indices")
    if not set(s_a).issubset(set(s_b)):
        raise RuntimeError("layout_s_a_not_subset_s_b")
    for g in (
        "shoulders",
        "elbows",
        "wrists",
        "hand_mcp",
        "hand_fingertips",
        "mouth_anchors",
    ):
        idxs = semantic_group_indices(g)
        if not idxs:
            raise RuntimeError(f"empty_group:{g}")
        if not set(idxs).issubset(set(s_a)):
            raise RuntimeError(f"required_group_missing_from_s_a:{g}")
    # Chirality: left/right hand MCP/tips must land in correct halves.
    for name in ("index_mcp", "index_tip"):
        li, ri = _hand("left", name), _hand("right", name)
        if not (LEFT_HAND.start <= li < LEFT_HAND.stop):
            raise RuntimeError("left_hand_chirality_fail")
        if not (RIGHT_HAND.start <= ri < RIGHT_HAND.stop):
            raise RuntimeError("right_hand_chirality_fail")


LAYOUT_S_A: Final[tuple[int, ...]] = layout_indices("S_A")
LAYOUT_S_B: Final[tuple[int, ...]] = layout_indices("S_B")
validate_layout_invariants()
