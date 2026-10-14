"""Canonical 75-joint layout for MS-STA v5 Skeleton/RGB token v3.

Must match SignLanguageDataEngine concatenation:
  pose(12) + face(21) + left_hand(21) + right_hand(21) = 75
"""

from __future__ import annotations

from typing import Final

N_JOINTS: Final[int] = 75

POSE: Final[slice] = slice(0, 12)
FACE: Final[slice] = slice(12, 33)
LEFT_HAND: Final[slice] = slice(33, 54)
RIGHT_HAND: Final[slice] = slice(54, 75)

POSE_N: Final[int] = 12
FACE_N: Final[int] = 21
HAND_N: Final[int] = 21

JOINT_PARTS: Final[tuple[str, ...]] = tuple(
    (["pose"] * POSE_N)
    + (["face"] * FACE_N)
    + (["left_hand"] * HAND_N)
    + (["right_hand"] * HAND_N)
)
JOINT_CHIRALITY: Final[tuple[str, ...]] = tuple(
    (["none"] * POSE_N)
    + (["none"] * FACE_N)
    + (["left"] * HAND_N)
    + (["right"] * HAND_N)
)

# Local bone graphs (relative indices within each part).
POSE_BONES: Final[list[tuple[int, int]]] = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]
# Face contour chain covering all 21 face joints (0..20).
FACE_BONES: Final[list[tuple[int, int]]] = [(i, i + 1) for i in range(FACE_N - 1)]
HAND_BONES: Final[list[tuple[int, int]]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def bone_edge_list() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for a, b in POSE_BONES:
        edges.append((POSE.start + a, POSE.start + b))
    for a, b in FACE_BONES:
        edges.append((FACE.start + a, FACE.start + b))
    for a, b in HAND_BONES:
        edges.append((LEFT_HAND.start + a, LEFT_HAND.start + b))
        edges.append((RIGHT_HAND.start + a, RIGHT_HAND.start + b))
    return edges


def assert_layout_invariants() -> None:
    if len(JOINT_PARTS) != N_JOINTS or len(JOINT_CHIRALITY) != N_JOINTS:
        raise AssertionError("joint_identity_len_not_75")
    if FACE.stop - FACE.start != FACE_N:
        raise AssertionError("face_slice_mismatch")
    if LEFT_HAND.stop - LEFT_HAND.start != HAND_N:
        raise AssertionError("lh_slice_mismatch")
    if RIGHT_HAND.stop - RIGHT_HAND.start != HAND_N:
        raise AssertionError("rh_slice_mismatch")
    edges = bone_edge_list()
    covered = {i for e in edges for i in e}
    if (N_JOINTS - 1) not in covered:
        raise AssertionError("joint_74_not_covered_by_bones")
    if max(covered) != N_JOINTS - 1:
        raise AssertionError(f"max_bone_index:{max(covered)}")
    if min(covered) < 0:
        raise AssertionError("negative_bone_index")


assert_layout_invariants()
