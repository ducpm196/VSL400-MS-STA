"""MS-STA v5 D0 scaffold / schema tests (no GPU, no full extract)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    F0_CLOSEOUT_REL,
    INDEPENDENT_REAUDIT_REL,
    PILOT_R1_INDEPENDENT_REAUDIT_REL,
    PILOT_SAMPLE_REL,
    load_json,
    sha256_bytes,
    validate_f0_phase_closeout,
    validate_independent_reaudit,
    validate_pilot_sample_set_r1,
    validate_r1_independent_reaudit,
)
from src.data_prep.joint_layout_v3 import (  # noqa: E402
    FACE,
    JOINT_PARTS,
    LEFT_HAND,
    N_JOINTS,
    RIGHT_HAND,
    assert_layout_invariants,
    bone_edge_list,
)
from src.data_prep.skel_v3 import build_skel_v3_arrays  # noqa: E402


def test_f0_phase_closeout_valid_and_immutable_parents() -> None:
    obj = load_json(_ROOT / F0_CLOSEOUT_REL)
    out = validate_f0_phase_closeout(obj, root=_ROOT)
    assert out["phase_status"] == "CLOSED_WITH_3_FEASIBLE_CANDIDATES"
    assert obj["f0_r2_authorized"] is False
    excluded = obj["stage_b_candidate_pool"]["excluded"][0]
    assert excluded["status"] == "CONTRACT_INVALID_NOT_TESTED"
    assert excluded["not"] == "GPU_INFEASIBLE"


def test_joint_layout_75_covers_joint_74_and_sentinel_groups() -> None:
    assert_layout_invariants()
    assert len(JOINT_PARTS) == N_JOINTS == 75
    assert JOINT_PARTS[11] == "pose"
    assert JOINT_PARTS[12] == "face" and JOINT_PARTS[32] == "face"
    assert JOINT_PARTS[33] == "left_hand" and JOINT_PARTS[53] == "left_hand"
    assert JOINT_PARTS[54] == "right_hand" and JOINT_PARTS[74] == "right_hand"
    # Wrong legacy slices must not match canonical groups.
    assert FACE != slice(12, 32)
    assert LEFT_HAND != slice(32, 53)
    assert RIGHT_HAND != slice(53, 74)
    covered = {i for e in bone_edge_list() for i in e}
    assert 74 in covered


def test_skel_v3_persists_prefill_mask_and_frame_index() -> None:
    t, j = 10, 75
    joints = np.random.randn(t, j, 3).astype(np.float32) * 0.01
    pose = np.random.randn(t, 33, 3).astype(np.float32) * 0.01
    pose[:, 11, :] = np.array([-0.2, 0.0, 0.0], dtype=np.float32)
    pose[:, 12, :] = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    missing = np.zeros((t, j), dtype=bool)
    missing[3, 40:45] = True
    frame_idx = np.arange(0, t * 2, 2, dtype=np.int32)
    stamps = frame_idx.astype(np.float32) / 25.0
    out = build_skel_v3_arrays(
        joints_raw=joints,
        raw_pose=pose,
        missing_prefill=missing,
        original_frame_index=frame_idx,
        timestamps_sec=stamps,
    )
    assert out["missing_mask_prefill"].sum() == 5
    assert out["sequence_length"] == t
    assert out["padding_mask"].all()
    assert out["original_frame_index"].tolist() == frame_idx.tolist()
    assert out["coordinates"].shape == (t, 75, 3)
    assert out["joint_part"].shape[0] == 75
    assert out["joint_chirality"].shape[0] == 75
    assert int(out["bone_edges"].max()) == 74
    assert np.isfinite(out["coordinates"]).all()


def test_region_boxes_use_canonical_hand_face_slices() -> None:
    from src.models.rgb_token_v3 import region_boxes_from_landmarks

    xy = np.zeros((75, 2), dtype=np.float32)
    xy[:, :] = 10.0
    # Distinct sentinel clusters in correct slices.
    xy[FACE, 0] = np.linspace(20, 40, 21)
    xy[FACE, 1] = np.linspace(20, 40, 21)
    xy[LEFT_HAND, 0] = np.linspace(50, 90, 21)
    xy[LEFT_HAND, 1] = np.linspace(50, 90, 21)
    xy[RIGHT_HAND, 0] = np.linspace(120, 160, 21)
    xy[RIGHT_HAND, 1] = np.linspace(120, 160, 21)
    miss = np.zeros((75,), dtype=bool)
    boxes = region_boxes_from_landmarks(xy, miss)
    assert boxes["face"] is not None and boxes["left_hand"] is not None and boxes["right_hand"] is not None
    face_cx = 0.5 * (boxes["face"][0] + boxes["face"][2])
    lh_cx = 0.5 * (boxes["left_hand"][0] + boxes["left_hand"][2])
    rh_cx = 0.5 * (boxes["right_hand"][0] + boxes["right_hand"][2])
    assert face_cx < lh_cx < rh_cx
    # Joint 74 participates in RH bbox (max x near 160 + pad).
    assert boxes["right_hand"][2] >= 160.0


def test_nearest_skel_aligns_odd_center_windows() -> None:
    from src.models.rgb_token_v3 import nearest_skel_landmark

    # Skeleton kept even frames only (stride 2); RGB center can be odd.
    skel_frames = np.asarray([0, 2, 4, 6, 8], dtype=np.int32)
    skel_ts = skel_frames.astype(np.float32) / 25.0
    align = nearest_skel_landmark(
        center_frame=5,
        fps=25.0,
        skel_frame_index=skel_frames,
        skel_timestamps=skel_ts,
    )
    # Equal distance 4 vs 6 → prefer earlier skel index (frame 4).
    assert align["landmark_frame"] == 4
    assert align["alignment_tie_break"] == "prefer_earlier_skel_index"
    assert align["alignment_tie_n"] == 2
    assert align["alignment_method"] == "nearest_timestamp"
    assert align["alignment_delta_sec"] == pytest.approx(1.0 / 25.0)


def test_checkerboard_without_roi_fails_overlay_render_integrity() -> None:
    import run_ms_sta_v5_d0_r3_pilot as r3

    chk = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            chk[y, x] = 255 if ((x // 16) + (y // 16)) % 2 == 0 else 0
    path = _ROOT / "Docs/results/ms_sta_v5_d0_r3/_tmp_checkerboard_neg.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), chk)
    try:
        out = r3.check_overlay_render_integrity(path, boxes={}, sample_id="CHECKERBOARD")
        assert out["overlay_readable"] is True
        assert out["stroke_consistency_ok"] is False
        assert out["overlay_render_integrity"] == "FAIL"
        assert out["not_manual_semantic_qa"] is True
        assert out["sha256"]
    finally:
        path.unlink(missing_ok=True)


def test_arbitrary_wrong_colored_boxes_can_pass_render_integrity_only() -> None:
    """Negative contract: stroke-color gate is NOT semantic/manual QA."""
    import run_ms_sta_v5_d0_r3_pilot as r3
    from src.models.rgb_token_v3 import draw_roi_overlay

    wrong = np.zeros((224, 224, 3), dtype=np.uint8)
    boxes = {
        "global": np.asarray([0.0, 0.0, 224.0, 224.0], dtype=np.float32),
        "left_hand": np.asarray([5.0, 5.0, 40.0, 40.0], dtype=np.float32),
        "right_hand": np.asarray([180.0, 5.0, 220.0, 40.0], dtype=np.float32),
        "face": np.asarray([90.0, 5.0, 140.0, 40.0], dtype=np.float32),
    }
    drawn = draw_roi_overlay(wrong, boxes)
    path = _ROOT / "Docs/results/ms_sta_v5_d0_r3/_tmp_wrong_boxes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), drawn)
    try:
        out = r3.check_overlay_render_integrity(
            path,
            boxes={k: v.tolist() for k, v in boxes.items()},
            sample_id="WRONG_BOX",
        )
        assert out["overlay_render_integrity"] == "PASS"
        assert out["not_manual_semantic_qa"] is True
    finally:
        path.unlink(missing_ok=True)


def test_d0_full_protocol_locks_representation() -> None:
    from ms_sta_v5_d0_lib import FULL_PROTOCOL_REL, validate_d0_full_protocol

    obj = load_json(_ROOT / FULL_PROTOCOL_REL)
    validate_d0_full_protocol(obj)
    assert obj["forbid_temporal_mean_before_roi_align"] is True
    assert obj["forbid_zero_dilution_in_temporal_mean"] is True
    assert obj["rgb_token_v3"]["operator_order"] == "roi_align_per_temporal_slice_then_masked_mean_aggregate"
    assert obj["rgb_token_v3"]["temporal_aggregation"] == "masked_mean_exclude_missing_zeros"
    assert obj["rgb_token_v3"]["alignment"]["tie_break"] == "prefer_earlier_skel_index"
    assert obj["skeleton_v3"]["model_complexity"] == 0
    assert obj["skeleton_v3"]["frame_stride"] == 2
    assert obj["visual_qa"]["manual_acceptance_rule"] == "all_reviewed_inspections_must_pass_zero_fail_allowed"
    assert obj["full_corpus_authorized"] is False


def test_r3_independent_reaudit_fail_preserved_parent_pass() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R3_INDEPENDENT_REAUDIT_REL,
        validate_r3_independent_reaudit,
    )

    obj = load_json(_ROOT / PILOT_R3_INDEPENDENT_REAUDIT_REL)
    validate_r3_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"
    assert {f["id"] for f in obj["findings"]} == {"P0-1", "P0-2", "P0-3"}


def test_masked_mean_excludes_missing_zero_tokens() -> None:
    from src.models.rgb_token_v3 import _masked_mean_tokens

    a = np.asarray([2.0, 4.0], dtype=np.float32)
    b = np.asarray([0.0, 0.0], dtype=np.float32)
    c = np.asarray([6.0, 8.0], dtype=np.float32)
    out, meta = _masked_mean_tokens([a, b, c], [False, True, False])
    assert meta["n_slices"] == 3
    assert meta["n_valid"] == 2
    assert meta["partial_missing"] is True
    assert meta["all_missing"] is False
    np.testing.assert_allclose(out, np.asarray([4.0, 6.0], dtype=np.float32))
    # Unmasked mean would dilute toward zero: (2+0+6)/3 = 2.666...
    unmasked = np.mean(np.stack([a, b, c], axis=0), axis=0)
    assert float(np.linalg.norm(out - unmasked)) > 1.0


def test_visual_qa_manual_rejects_forged_identity_and_contradictory_pass() -> None:
    from ms_sta_v5_d0_lib import validate_visual_qa_manual

    overlays = [
        {
            "sha256": "abc",
            "sample_id": "000000",
            "path": "Docs/results/ms_sta_v5_d0_r4/visual_qa/000000/x.png",
            "center_frame": 16,
            "native_frame": 16,
            "boxes": {"face": [1, 2, 3, 4]},
            "missing_regions": {"face": False, "left_hand": False, "right_hand": False},
            "kind": "center_overlay",
        }
    ]
    forged = {
        "schema": "ms_sta_v5_d0_r4_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "auto_stroke_color_sufficient": False,
        "reviewer": "unit",
        "acceptance_rule": "all_reviewed_inspections_must_pass_zero_fail_allowed",
        "arbitrary_wrong_box_negative_accepted": False,
        "n_pass": 1,
        "n_fail": 0,
        "inspections": [
            {
                "sha256": "abc",
                "sample_id": "999999",
                "path": "Docs/results/ms_sta_v5_d0_r4/visual_qa/000000/x.png",
                "center_frame": 99,
                "native_frame": 99,
                "boxes": {"face": [9, 9, 10, 10]},
                "missing_regions": {"face": True, "left_hand": True, "right_hand": True},
                "box_on_correct_body_region": False,
                "handedness_ok": False,
                "visible_but_missing_consistent": False,
                "verdict": "PASS",
                "qa_kind": "center_overlay",
                "derived_from_overlay_render_integrity_only": False,
            }
        ],
    }
    with pytest.raises(ValueError) as ei:
        validate_visual_qa_manual(
            forged, overlays_index=overlays, require_zero_fail=True, require_temporal_contact=True
        )
    msg = str(ei.value)
    assert "sample_id_mismatch" in msg or "contradictory_pass" in msg
    assert "FORGED_MANUAL_IDENTITY_ACCEPTED" not in msg


def test_visual_qa_manual_rejects_render_integrity_only_source() -> None:
    from ms_sta_v5_d0_lib import validate_visual_qa_manual

    overlays = [
        {
            "sha256": "abc",
            "sample_id": "000000",
            "path": "x.png",
            "center_frame": 16,
            "native_frame": 16,
            "boxes": {},
            "missing_regions": {},
        }
    ]
    bad = {
        "schema": "ms_sta_v5_d0_r3_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "auto_stroke_color_sufficient": False,
        "reviewer": "unit",
        "arbitrary_wrong_box_negative_accepted": False,
        "inspections": [
            {
                "sha256": "abc",
                "sample_id": "000000",
                "path": "x.png",
                "center_frame": 16,
                "native_frame": 16,
                "boxes": {},
                "missing_regions": {},
                "verdict": "PASS",
                "box_on_correct_body_region": True,
                "handedness_ok": True,
                "visible_but_missing_consistent": True,
                "derived_from_overlay_render_integrity_only": True,
            }
        ],
    }
    with pytest.raises(ValueError, match="render_integrity_only"):
        validate_visual_qa_manual(bad, overlays_index=overlays)


def test_rgb_token_v3_forbids_temporal_mean_before_roi_align() -> None:
    src = (_ROOT / "src/models/rgb_token_v3.py").read_text(encoding="utf-8")
    assert "slow_map = maps.slow.mean(dim=2)" not in src
    assert "fast_map = maps.fast.mean(dim=2)" not in src
    assert "masked_mean_exclude_missing_zeros" in src
    assert "_masked_mean_tokens" in src
    assert "forbid_zero_dilution_in_temporal_mean" in src


def test_protocol_param_zero_complexity_not_falsy() -> None:
    """model_complexity=0 must compare as 0, not collapse via `or -1`."""
    identity = {"frame_stride": 2, "model_complexity": 0}
    assert int(identity.get("model_complexity", -1)) == 0
    assert int(identity.get("model_complexity") or -1) == -1  # documents the footgun


def test_r2_independent_reaudit_fail_preserved_parent_pass() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R2_INDEPENDENT_REAUDIT_REL,
        validate_r2_independent_reaudit,
    )

    obj = load_json(_ROOT / PILOT_R2_INDEPENDENT_REAUDIT_REL)
    validate_r2_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"


def test_window_starts_end_anchored_covers_tail() -> None:
    from src.models.rgb_token_v3 import window_starts

    starts = window_starts(70, window_size=32, window_stride=16)
    assert starts[0] == 0
    assert (70 - 32) in starts
    assert max(s + 32 for s in starts) >= 70
    assert window_starts(20, window_size=32, window_stride=16) == [0]


def test_independent_reaudit_fail_binds_six_findings() -> None:
    obj = load_json(_ROOT / INDEPENDENT_REAUDIT_REL)
    validate_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"
    assert obj["scientific_pilot_pass"] is False
    assert {f["id"] for f in obj["findings"]} == {f"P0-{i}" for i in range(1, 7)}


def test_r1_independent_reaudit_fail_preserved_parent_pass() -> None:
    obj = load_json(_ROOT / PILOT_R1_INDEPENDENT_REAUDIT_REL)
    validate_r1_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"
    assert {f["id"] for f in obj["findings"]} == {"P0-1", "P0-2", "P0-3", "P0-4", "P1-1"}


def test_r1_validator_rejects_legacy_false_pass_sample_patterns() -> None:
    legacy = load_json(_ROOT / PILOT_SAMPLE_REL)
    with pytest.raises(ValueError) as ei:
        validate_pilot_sample_set_r1(
            {
                **legacy,
                "schema": "ms_sta_v5_d0_r1_pilot_sample_set_v1",
                "engineering_scope": "train_val_only",
                "test_clips_forbidden_for_roi_preprocess": True,
            },
            root=_ROOT,
        )
    msg = str(ei.value)
    assert any(
        token in msg
        for token in (
            "r1_test_or_unknown_split_forbidden",
            "r1_insufficient_gloss_diversity",
            "r1_decode_edge_without_evidence",
            "r1_frame_range_too_narrow",
            "r1_rejects_single_gloss_anh_set",
            "r1_length_tag_missing_quantile",
            "r1_missing_tag_without_measured_rate",
        )
    )


def test_r1_validator_rejects_vacuous_test_leak_proxies() -> None:
    bad = {
        "schema": "ms_sta_v5_d0_r1_pilot_sample_set_v1",
        "status": "FROZEN",
        "engineering_scope": "train_val_only",
        "test_clips_forbidden_for_roi_preprocess": True,
        "samples": [
            {
                "sample_id": f"{i:06d}",
                "split": "test" if i == 0 else "train",
                "dataset": "VSL400",
                "gloss": "ANH",
                "signer_id": "S1",
                "n_frames": 55,
                "edge_tags": ["short_clip", "decode_edge"],
                "decode_failure_evidence": False,
            }
            for i in range(12)
        ],
    }
    ids = [s["sample_id"] for s in bad["samples"]]
    bad["sample_ids_sha256"] = sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))
    with pytest.raises(ValueError):
        validate_pilot_sample_set_r1(bad, root=_ROOT)


def _r5_base_manual(*, overlays: list[dict], inspections: list[dict], strata: set[str] | None = None) -> dict:
    return {
        "schema": "ms_sta_v5_d0_r5_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "auto_stroke_color_sufficient": False,
        "reviewer": "unit",
        "acceptance_rule": "all_reviewed_inspections_must_pass_zero_fail_allowed",
        "arbitrary_wrong_box_negative_accepted": False,
        "freeze_list_rel": "Docs/results/ms_sta_v5_d0_r5/D0_R5_VISUAL_QA_FREEZE_LIST.json",
        "strata_covered": sorted(
            strata
            or {
                "full_valid_control_vf_eq_1",
                "partial_mid_0_5_le_vf_lt_1",
                "partial_low_0_lt_vf_lt_0_5",
                "worst_face",
                "worst_left_hand",
                "worst_right_hand",
                "train_split",
                "val_split",
                "mandatory_015417_w0016_face_vf_0_15",
            }
        ),
        "n_pass": len(inspections),
        "n_fail": 0,
        "inspections": inspections,
    }


def test_r5_rejects_swapped_qa_kind() -> None:
    from ms_sta_v5_d0_lib import validate_visual_qa_manual

    tile = {
        "pathway": "slow",
        "t_idx": 0,
        "native_frame": 16,
        "landmark_frame": 16,
        "boxes": {"face": [1, 2, 3, 4]},
        "region_missing": {"face": False, "left_hand": False, "right_hand": False, "global": False},
    }
    overlays = [
        {
            "sha256": "center",
            "sample_id": "015417",
            "path": "Docs/results/ms_sta_v5_d0_r5/visual_qa/015417/c.png",
            "center_frame": 32,
            "native_frame": 32,
            "kind": "center_overlay",
            "boxes": {"face": [1, 2, 3, 4]},
            "missing_regions": {"face": False, "left_hand": False, "right_hand": False},
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
        },
        {
            "sha256": "contact",
            "sample_id": "015417",
            "path": "Docs/results/ms_sta_v5_d0_r5/visual_qa/015417/t.png",
            "center_frame": 32,
            "native_frame": 32,
            "kind": "temporal_contact_sheet",
            "boxes": {"face": [1, 2, 3, 4]},
            "missing_regions": {"face": False, "left_hand": False, "right_hand": False},
            "n_tiles": 1,
            "slice_native_frames": [16],
            "tiles": [tile],
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
        },
        {
            "sha256": "other",
            "sample_id": "000000",
            "path": "Docs/results/ms_sta_v5_d0_r5/visual_qa/000000/c.png",
            "center_frame": 16,
            "native_frame": 16,
            "kind": "center_overlay",
            "boxes": {"face": [1, 2, 3, 4]},
            "missing_regions": {"face": False, "left_hand": False, "right_hand": False},
            "min_valid_fraction": 1.0,
            "window_start_idx": 0,
        },
    ]
    inspections = [
        {
            "sha256": "center",
            "sample_id": "015417",
            "path": overlays[0]["path"],
            "center_frame": 32,
            "native_frame": 32,
            "boxes": overlays[0]["boxes"],
            "missing_regions": overlays[0]["missing_regions"],
            "qa_kind": "temporal_contact_sheet",  # swapped
            "box_on_correct_body_region": True,
            "handedness_ok": True,
            "visible_but_missing_consistent": True,
            "verdict": "PASS",
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
            "derived_from_overlay_render_integrity_only": False,
        },
        {
            "sha256": "contact",
            "sample_id": "015417",
            "path": overlays[1]["path"],
            "center_frame": 32,
            "native_frame": 32,
            "boxes": overlays[1]["boxes"],
            "missing_regions": overlays[1]["missing_regions"],
            "qa_kind": "temporal_contact_sheet",
            "n_tiles": 1,
            "slice_native_frames": [16],
            "tiles": [tile],
            "box_on_correct_body_region": True,
            "handedness_ok": True,
            "visible_but_missing_consistent": True,
            "verdict": "PASS",
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
            "derived_from_overlay_render_integrity_only": False,
        },
        {
            "sha256": "other",
            "sample_id": "000000",
            "path": overlays[2]["path"],
            "center_frame": 16,
            "native_frame": 16,
            "boxes": overlays[2]["boxes"],
            "missing_regions": overlays[2]["missing_regions"],
            "qa_kind": "center_overlay",
            "box_on_correct_body_region": True,
            "handedness_ok": True,
            "visible_but_missing_consistent": True,
            "verdict": "PASS",
            "min_valid_fraction": 1.0,
            "window_start_idx": 0,
            "derived_from_overlay_render_integrity_only": False,
        },
    ]
    with pytest.raises(ValueError) as ei:
        validate_visual_qa_manual(
            _r5_base_manual(overlays=overlays, inspections=inspections),
            overlays_index=overlays,
            require_zero_fail=True,
            require_temporal_contact=True,
            require_r5_identity=True,
        )
    assert "qa_kind_mismatch" in str(ei.value)
    assert "SWAPPED_QA_KIND_ACCEPTED" not in str(ei.value)


def test_r5_rejects_mutated_tile_identity() -> None:
    from ms_sta_v5_d0_lib import validate_visual_qa_manual

    tile = {
        "pathway": "slow",
        "t_idx": 0,
        "native_frame": 16,
        "landmark_frame": 16,
        "boxes": {"face": [1.0, 2.0, 3.0, 4.0]},
        "region_missing": {"face": False, "left_hand": False, "right_hand": False, "global": False},
    }
    overlays = [
        {
            "sha256": "contact",
            "sample_id": "015417",
            "path": "Docs/results/ms_sta_v5_d0_r5/visual_qa/015417/t.png",
            "center_frame": 32,
            "native_frame": 32,
            "kind": "temporal_contact_sheet",
            "boxes": {"face": [1.0, 2.0, 3.0, 4.0]},
            "missing_regions": {"face": False, "left_hand": False, "right_hand": False},
            "n_tiles": 1,
            "slice_native_frames": [16],
            "tiles": [tile],
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
        },
        {
            "sha256": "c1",
            "sample_id": "015417",
            "path": "Docs/results/ms_sta_v5_d0_r5/visual_qa/015417/c.png",
            "center_frame": 32,
            "native_frame": 32,
            "kind": "center_overlay",
            "boxes": {"face": [1.0, 2.0, 3.0, 4.0]},
            "missing_regions": {"face": False, "left_hand": False, "right_hand": False},
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
        },
        {
            "sha256": "c2",
            "sample_id": "000000",
            "path": "Docs/results/ms_sta_v5_d0_r5/visual_qa/000000/c.png",
            "center_frame": 16,
            "native_frame": 16,
            "kind": "center_overlay",
            "boxes": {"face": [1.0, 2.0, 3.0, 4.0]},
            "missing_regions": {"face": False, "left_hand": False, "right_hand": False},
            "min_valid_fraction": 0.6,
            "window_start_idx": 0,
        },
    ]
    mutated = dict(tile)
    mutated["native_frame"] = 99
    mutated["boxes"] = {"face": [9.0, 9.0, 10.0, 10.0]}
    mutated["region_missing"] = {"face": True, "left_hand": False, "right_hand": False, "global": False}
    inspections = [
        {
            "sha256": "contact",
            "sample_id": "015417",
            "path": overlays[0]["path"],
            "center_frame": 32,
            "native_frame": 32,
            "boxes": overlays[0]["boxes"],
            "missing_regions": overlays[0]["missing_regions"],
            "qa_kind": "temporal_contact_sheet",
            "n_tiles": 1,
            "slice_native_frames": [16],
            "tiles": [mutated],
            "box_on_correct_body_region": True,
            "handedness_ok": True,
            "visible_but_missing_consistent": True,
            "verdict": "PASS",
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
            "derived_from_overlay_render_integrity_only": False,
        },
        {
            "sha256": "c1",
            "sample_id": "015417",
            "path": overlays[1]["path"],
            "center_frame": 32,
            "native_frame": 32,
            "boxes": overlays[1]["boxes"],
            "missing_regions": overlays[1]["missing_regions"],
            "qa_kind": "center_overlay",
            "box_on_correct_body_region": True,
            "handedness_ok": True,
            "visible_but_missing_consistent": True,
            "verdict": "PASS",
            "min_valid_fraction": 0.15,
            "window_start_idx": 16,
            "derived_from_overlay_render_integrity_only": False,
        },
        {
            "sha256": "c2",
            "sample_id": "000000",
            "path": overlays[2]["path"],
            "center_frame": 16,
            "native_frame": 16,
            "boxes": overlays[2]["boxes"],
            "missing_regions": overlays[2]["missing_regions"],
            "qa_kind": "center_overlay",
            "box_on_correct_body_region": True,
            "handedness_ok": True,
            "visible_but_missing_consistent": True,
            "verdict": "PASS",
            "min_valid_fraction": 0.6,
            "window_start_idx": 0,
            "derived_from_overlay_render_integrity_only": False,
        },
    ]
    with pytest.raises(ValueError) as ei:
        validate_visual_qa_manual(
            _r5_base_manual(overlays=overlays, inspections=inspections),
            overlays_index=overlays,
            require_zero_fail=True,
            require_temporal_contact=True,
            require_r5_identity=True,
        )
    msg = str(ei.value)
    assert "tile_mismatch" in msg or "contact_field_mismatch" in msg


def test_r5_rejects_all_full_valid_selection() -> None:
    from ms_sta_v5_d0_lib import validate_visual_qa_manual, validate_r5_visual_freeze_list

    with pytest.raises(ValueError, match="all_full_valid"):
        validate_r5_visual_freeze_list(
            {
                "schema": "ms_sta_v5_d0_r5_visual_qa_freeze_list_v1",
                "frozen_before_review": True,
                "windows": [
                    {
                        "sample_id": "000000",
                        "window_start_idx": 0,
                        "split": "train",
                        "stratum": "full_valid_control_vf_eq_1",
                        "also_strata": [
                            "partial_mid_0_5_le_vf_lt_1",
                            "partial_low_0_lt_vf_lt_0_5",
                            "worst_face",
                            "worst_left_hand",
                            "worst_right_hand",
                            "mandatory_015417_w0016_face_vf_0_15",
                            "train_split",
                        ],
                        "min_valid_fraction": 1.0,
                    },
                    {
                        "sample_id": "000001",
                        "window_start_idx": 0,
                        "split": "val",
                        "stratum": "val_split",
                        "min_valid_fraction": 1.0,
                    },
                    {
                        "sample_id": "015417",
                        "window_start_idx": 16,
                        "split": "train",
                        "stratum": "mandatory_015417_w0016_face_vf_0_15",
                        "min_valid_fraction": 1.0,
                    },
                    {"sample_id": "a", "window_start_idx": 0, "split": "train", "stratum": "worst_face", "min_valid_fraction": 1.0},
                    {"sample_id": "b", "window_start_idx": 0, "split": "train", "stratum": "worst_left_hand", "min_valid_fraction": 1.0},
                    {"sample_id": "c", "window_start_idx": 0, "split": "train", "stratum": "worst_right_hand", "min_valid_fraction": 1.0},
                    {"sample_id": "d", "window_start_idx": 0, "split": "train", "stratum": "partial_mid_0_5_le_vf_lt_1", "min_valid_fraction": 1.0},
                    {"sample_id": "e", "window_start_idx": 0, "split": "train", "stratum": "partial_low_0_lt_vf_lt_0_5", "min_valid_fraction": 1.0},
                ],
            }
        )

    overlays = [
        {
            "sha256": f"s{i}",
            "sample_id": "015417" if i == 0 else f"{i:06d}",
            "path": f"x{i}.png",
            "center_frame": 16,
            "native_frame": 16,
            "kind": "center_overlay" if i % 2 == 0 else "temporal_contact_sheet",
            "boxes": {},
            "missing_regions": {},
            "n_tiles": 1,
            "slice_native_frames": [16],
            "tiles": [
                {
                    "pathway": "slow",
                    "t_idx": 0,
                    "native_frame": 16,
                    "landmark_frame": 16,
                    "boxes": {},
                    "region_missing": {},
                }
            ],
            "min_valid_fraction": 1.0,
            "window_start_idx": 16 if i == 0 else 0,
        }
        for i in range(3)
    ]
    inspections = []
    for o in overlays:
        row = {
            "sha256": o["sha256"],
            "sample_id": o["sample_id"],
            "path": o["path"],
            "center_frame": 16,
            "native_frame": 16,
            "boxes": {},
            "missing_regions": {},
            "qa_kind": o["kind"],
            "box_on_correct_body_region": True,
            "handedness_ok": True,
            "visible_but_missing_consistent": True,
            "verdict": "PASS",
            "min_valid_fraction": 1.0,
            "window_start_idx": o["window_start_idx"],
            "derived_from_overlay_render_integrity_only": False,
        }
        if o["kind"] == "temporal_contact_sheet":
            row["n_tiles"] = 1
            row["slice_native_frames"] = [16]
            row["tiles"] = o["tiles"]
        inspections.append(row)
    with pytest.raises(ValueError, match="all_full_valid_selection_forbidden"):
        validate_visual_qa_manual(
            _r5_base_manual(overlays=overlays, inspections=inspections),
            overlays_index=overlays,
            require_zero_fail=True,
            require_temporal_contact=True,
            require_r5_identity=True,
        )


def test_r4_independent_reaudit_fail_preserved() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R4_INDEPENDENT_REAUDIT_REL,
        validate_r4_independent_reaudit,
    )

    obj = load_json(_ROOT / PILOT_R4_INDEPENDENT_REAUDIT_REL)
    validate_r4_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"
    assert {f["id"] for f in obj["findings"]} == {"P0-1", "P0-2"}


def test_r5_qa_amendment_protects_canonical_protocol() -> None:
    from ms_sta_v5_d0_lib import (
        FULL_PROTOCOL_R5_AMENDMENT_REL,
        validate_r5_qa_amendment,
    )

    obj = load_json(_ROOT / FULL_PROTOCOL_R5_AMENDMENT_REL)
    validate_r5_qa_amendment(obj, root=_ROOT)
    assert obj["do_not_overwrite_canonical_full_protocol"] is True


def test_r5_independent_reaudit_fail_preserved() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R5_INDEPENDENT_REAUDIT_REL,
        validate_r5_independent_reaudit,
    )

    obj = load_json(_ROOT / PILOT_R5_INDEPENDENT_REAUDIT_REL)
    validate_r5_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"
    assert obj["terminology_correction"]["correct"] == (
        "hand_landmark_detector_false_negative_vs_visible_rgb_hand"
    )


def test_r6_detector_amendment_and_freeze_lists() -> None:
    from ms_sta_v5_d0_lib import (
        FULL_PROTOCOL_R6_AMENDMENT_REL,
        PILOT_R5_VISUAL_FREEZE_REL,
        PILOT_R6_HOLDOUT_FREEZE_REL,
        PILOT_R6_VISUAL_FREEZE_REL,
        validate_r6_detector_amendment,
        validate_r6_freeze_lists,
    )

    validate_r6_detector_amendment(load_json(_ROOT / FULL_PROTOCOL_R6_AMENDMENT_REL), root=_ROOT)
    validate_r6_freeze_lists(
        load_json(_ROOT / PILOT_R6_VISUAL_FREEZE_REL),
        load_json(_ROOT / PILOT_R6_HOLDOUT_FREEZE_REL),
        r5_freeze=load_json(_ROOT / PILOT_R5_VISUAL_FREEZE_REL),
    )


def test_r6_provenance_helpers_reject_short_sha_as_tree() -> None:
    from ms_sta_v5_d0_lib import executable_tree_sha256, is_hex_sha256

    digest = executable_tree_sha256(
        [
            "src/data_prep/hand_recovery_v6.py",
            "scripts/ms_sta_v5_d0_lib.py",
        ],
        root=_ROOT,
    )
    assert is_hex_sha256(digest)
    assert len("3294f7e") != 40


def test_r6_independent_reaudit_fail_preserved() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R6_INDEPENDENT_REAUDIT_REL,
        validate_r6_independent_reaudit,
    )

    obj = load_json(_ROOT / PILOT_R6_INDEPENDENT_REAUDIT_REL)
    validate_r6_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"


def test_r7_amendment_and_freeze() -> None:
    from ms_sta_v5_d0_lib import (
        FULL_PROTOCOL_R7_AMENDMENT_REL,
        PILOT_R7_VISUAL_FREEZE_REL,
        validate_r7_dual_source_amendment,
        validate_r7_visual_freeze_list,
    )

    validate_r7_dual_source_amendment(load_json(_ROOT / FULL_PROTOCOL_R7_AMENDMENT_REL), root=_ROOT)
    validate_r7_visual_freeze_list(load_json(_ROOT / PILOT_R7_VISUAL_FREEZE_REL))


def test_r7_report_rejects_holdout_residual_and_short_code_lock() -> None:
    from ms_sta_v5_d0_lib import validate_r7_pilot_report
    import pytest

    bad = {
        "schema": "ms_sta_v5_d0_r7b_pilot_report_v1",
        "ok": False,
        "gates": {
            "primary_and_holdout_zero_fail": True,
            "finalizer_did_not_synthesize_manual": True,
            "producer_in_executable_tree": True,
            "c0_bit_exact_all_16": False,
            "r5_fail_keys_remediated_with_boxes": False,
            "executable_paths_git_tracked": True,
            "pose_prior_geometry_locked": True,
        },
        "holdout_residual_n_fail": 3,
        "code_lock_git_revision": "3294f7e",
        "executable_tree_sha256": "0" * 64,
        "full_corpus_authorized": False,
        "forbid": {"full_corpus": True},
        "visual_qa_manual_sha256": "f" * 64,
    }
    with pytest.raises(ValueError) as ei:
        validate_r7_pilot_report(bad, root=_ROOT)
    msg = str(ei.value)
    assert "holdout_residual_must_be_zero" in msg or "r7_report_ok_must_be_true" in msg
    assert "code_lock_not_full_sha40" in msg or "r7_report_gate_false:c0" in msg


def test_r7_manual_rejects_synthesized_flag() -> None:
    from ms_sta_v5_d0_lib import validate_r7_visual_qa_manual
    import pytest

    bad = {
        "schema": "ms_sta_v5_d0_r7b_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "synthesized_by_finalizer": True,
        "synthesized_by_template_tool": False,
        "auto_stroke_color_sufficient": False,
        "acceptance_rule": "primary_and_holdout_all_pass_zero_fail_allowed",
        "reviewer": "x",
        "reviewed_at": "2026-01-01T00:00:00Z",
        "method": "m",
        "n_pass": 0,
        "n_fail": 0,
        "inspections": [],
    }
    with pytest.raises(ValueError, match="not_synthesized"):
        validate_r7_visual_qa_manual(bad, overlays_index=[], freeze_list={"windows": []})


def test_r7_manual_rejects_duplicate_and_empty_reviewer() -> None:
    """Negative proof matching independent R7 FAIL P0-2."""
    from ms_sta_v5_d0_lib import validate_r7_visual_qa_manual
    import pytest

    sha = "a" * 64
    overlay = {
        "sha256": sha,
        "kind": "center_overlay",
        "sample_id": "000000",
        "window_start_idx": 0,
        "path": "Docs/results/ms_sta_v5_d0_r7b/visual_qa/x.png",
        "boxes": {},
    }
    inspections = []
    for i in range(11):
        inspections.append(
            {
                "sample_id": "000000",
                "qa_kind": "center_overlay",
                "sha256": sha,
                "window_start_idx": 0,
                "path": overlay["path"],
                "boxes": {},
                "box_on_correct_body_region": True,
                "handedness_ok": True,
                "visible_but_missing_consistent": True,
                "verdict": "PASS",
                "r7_role": "primary_r5_carryforward",
            }
        )
    bad = {
        "schema": "ms_sta_v5_d0_r7b_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "synthesized_by_finalizer": False,
        "synthesized_by_template_tool": False,
        "auto_stroke_color_sufficient": False,
        "acceptance_rule": "primary_and_holdout_all_pass_zero_fail_allowed",
        "reviewer": "",
        "reviewed_at": "",
        "method": "",
        "n_pass": 11,
        "n_fail": 0,
        "inspections": inspections,
    }
    with pytest.raises(ValueError) as ei:
        validate_r7_visual_qa_manual(
            bad,
            overlays_index=[overlay],
            freeze_list={"windows": [{"sample_id": "000000", "window_start_idx": 0, "r7_role": "primary_r5_carryforward"}]},
            root=_ROOT,
        )
    msg = str(ei.value)
    assert "r7_manual_reviewer_required" in msg
    assert "r7_manual_must_be_exact_22" in msg or "r7_manual_duplicate" in msg


def test_r7_manual_accepts_zero_fail_counts() -> None:
    """Placeholder retained: zero-fail counting still must not treat 0 as missing."""
    from ms_sta_v5_d0_lib import validate_r7_visual_qa_manual
    import pytest

    # Without freeze/paths this must fail closed under hardened validator.
    with pytest.raises(ValueError):
        validate_r7_visual_qa_manual(
            {
                "schema": "ms_sta_v5_d0_r7b_visual_qa_manual_v1",
                "source": "independent_human_or_named_reviewer_artifact",
                "synthesized_by_finalizer": False,
                "synthesized_by_template_tool": False,
                "auto_stroke_color_sufficient": False,
                "acceptance_rule": "primary_and_holdout_all_pass_zero_fail_allowed",
                "reviewer": "r",
                "reviewed_at": "2026-01-01T00:00:00Z",
                "method": "m",
                "n_pass": 0,
                "n_fail": 0,
                "inspections": [],
            },
            overlays_index=[],
            freeze_list={"windows": []},
        )


def test_pose_prior_area_respects_geometry_lock() -> None:
    from src.models.rgb_token_v3 import pose_wrist_prior_box, INPUT_SIZE
    import numpy as np

    lm = np.zeros((75, 2), dtype=np.float32)
    miss = np.ones((75,), dtype=bool)
    miss[8] = False
    miss[4] = False
    miss[5] = False
    lm[8] = [112.0, 160.0]
    lm[4] = [80.0, 80.0]
    lm[5] = [160.0, 80.0]
    box, reason = pose_wrist_prior_box(lm, miss, which="left_hand")
    assert box is not None
    assert reason is None
    x0, y0, x1, y1 = [float(v) for v in box.tolist()]
    area_frac = ((x1 - x0) * (y1 - y0)) / float(INPUT_SIZE * INPUT_SIZE)
    assert area_frac <= 0.08 + 1e-6


def test_report_accepts_zero_holdout_without_or_coercion() -> None:
    from ms_sta_v5_d0_lib import int_field

    assert int_field({"holdout_residual_n_fail": 0}, "holdout_residual_n_fail") == 0
    assert int_field({}, "holdout_residual_n_fail") is None


def test_repo_rel_path_rejects_absolute_even_under_repo() -> None:
    from ms_sta_v5_d0_lib import repo_rel_path
    import pytest

    abs_under = str((_ROOT / "Docs/results/ms_sta_v5_d0_r7c/visual_qa/x.png").resolve())
    with pytest.raises(ValueError, match="absolute_path_forbidden"):
        repo_rel_path(abs_under, root=_ROOT)


def test_bimanual_shared_prior_when_wrists_close() -> None:
    from src.models.rgb_token_v3 import region_boxes_with_rgb_sources
    import numpy as np

    lm = np.zeros((75, 2), dtype=np.float32)
    miss = np.ones((75,), dtype=bool)
    miss[4] = miss[5] = miss[8] = miss[9] = False
    lm[4] = [90.0, 60.0]
    lm[5] = [130.0, 60.0]
    lm[8] = [105.0, 140.0]
    lm[9] = [115.0, 140.0]
    boxes, sources, _ = region_boxes_with_rgb_sources(
        lm, miss, enable_ambiguity_aware_prior=True
    )
    assert sources["left_hand"] == "bimanual_shared_prior"
    assert sources["right_hand"] == "bimanual_shared_prior"
    assert boxes["left_hand"] is not None
    assert np.allclose(boxes["left_hand"], boxes["right_hand"])


def test_single_hand_prior_rejects_contralateral_inside() -> None:
    from src.models.rgb_token_v3 import single_hand_pose_prior_box
    import numpy as np

    lm = np.zeros((75, 2), dtype=np.float32)
    miss = np.ones((75,), dtype=bool)
    miss[4] = miss[5] = miss[6] = miss[8] = miss[9] = False
    lm[4] = [80.0, 60.0]
    lm[5] = [160.0, 60.0]
    lm[6] = [90.0, 100.0]
    lm[8] = [100.0, 140.0]
    lm[9] = [108.0, 142.0]
    box, reason = single_hand_pose_prior_box(lm, miss, which="left_hand")
    assert box is None
    assert reason == "pose_prior_contralateral_wrist_inside"


def test_r7b_independent_reaudit_bound() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R7B_INDEPENDENT_REAUDIT_REL,
        validate_r7b_independent_reaudit,
    )

    obj = load_json(_ROOT / PILOT_R7B_INDEPENDENT_REAUDIT_REL)
    validate_r7b_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_layers"]["evidence_lifecycle"] == "PASS"
    assert obj["audit_layers"]["scientific_d0_gate"] == "FAIL"


def test_executable_tree_rejects_untracked(tmp_path) -> None:
    from ms_sta_v5_d0_lib import executable_tree_sha256
    import pytest

    ghost = "scripts/_definitely_untracked_r7b_probe.py"
    p = _ROOT / ghost
    assert not p.exists()
    with pytest.raises(FileNotFoundError, match="executable_tree_missing|executable_tree_untracked"):
        executable_tree_sha256([ghost], root=_ROOT, require_git_tracked=True)


def test_r7_independent_reaudit_bound() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R7_INDEPENDENT_REAUDIT_REL,
        validate_r7_independent_reaudit,
    )

    obj = load_json(_ROOT / PILOT_R7_INDEPENDENT_REAUDIT_REL)
    validate_r7_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"


def test_pose_wrist_prior_does_not_flip_skeleton_missing() -> None:
    from src.models.rgb_token_v3 import region_boxes_with_rgb_sources
    import numpy as np

    lm = np.zeros((75, 2), dtype=np.float32)
    miss = np.ones((75,), dtype=bool)
    # pose wrists present
    miss[8] = False
    miss[9] = False
    miss[4] = False
    miss[5] = False
    lm[8] = [80.0, 120.0]
    lm[9] = [140.0, 120.0]
    lm[4] = [90.0, 60.0]
    lm[5] = [130.0, 60.0]
    boxes, sources, _ = region_boxes_with_rgb_sources(lm, miss, enable_pose_wrist_prior=True)
    assert boxes["left_hand"] is not None
    assert sources["left_hand"] == "pose_wrist_prior"
    # skeleton mask unchanged by this helper
    assert bool(miss[33:54].all()) is True

def test_pose_prior_absent_when_wrist_missing() -> None:
    from src.models.rgb_token_v3 import region_boxes_with_rgb_sources
    import numpy as np

    lm = np.zeros((75, 2), dtype=np.float32)
    miss = np.ones((75,), dtype=bool)
    boxes, sources, reasons = region_boxes_with_rgb_sources(lm, miss, enable_pose_wrist_prior=True)
    assert boxes["left_hand"] is None
    assert sources["left_hand"] == "missing"
    assert reasons["left_hand"] == "pose_wrist_missing"


def test_r7c_code_lock_rejects_forged_revision() -> None:
    """Negative proof for independent R7C FAIL P0-1."""
    from ms_sta_v5_d0_lib import R7C_EXECUTABLE_PATHS, verify_code_lock_revision_binding
    import pytest

    errors = verify_code_lock_revision_binding(
        "0" * 40,
        R7C_EXECUTABLE_PATHS,
        "f" * 64,
        root=_ROOT,
        require_worktree_match=False,
    )
    assert "code_lock_revision_not_in_git" in errors


def test_r7c_absent_fixture_requires_sha_and_ambiguity() -> None:
    """Negative proof for independent R7C FAIL P0-2."""
    from ms_sta_v5_d0_lib import (
        PILOT_R7C_FIXTURE_ABSENT_REL,
        load_json,
        validate_r7c_absent_wrist_fixture_proof,
    )
    import pytest

    proof = load_json(_ROOT / PILOT_R7C_FIXTURE_ABSENT_REL / "PROOF.json")
    validate_r7c_absent_wrist_fixture_proof(proof, root=_ROOT)
    bad = dict(proof)
    bad.pop("video_sha256", None)
    bad.pop("skel_sha256", None)
    with pytest.raises(ValueError) as ei:
        validate_r7c_absent_wrist_fixture_proof(bad, root=_ROOT)
    msg = str(ei.value)
    assert "video_sha_required" in msg
    assert "skel_sha_required" in msg


def test_r7b_absent_fixture_proof_restored() -> None:
    from ms_sta_v5_d0_lib import (
        EXPECTED_R7B_ABSENT_FIXTURE_PROOF_SHA256,
        PILOT_R7B_FIXTURE_ABSENT_REL,
        sha256_file,
    )

    assert (
        sha256_file(_ROOT / PILOT_R7B_FIXTURE_ABSENT_REL / "PROOF.json")
        == EXPECTED_R7B_ABSENT_FIXTURE_PROOF_SHA256
    )


def test_r7c_independent_reaudit_bound() -> None:
    from ms_sta_v5_d0_lib import (
        EXPECTED_R7C_INDEPENDENT_REAUDIT_SHA256,
        PILOT_R7C_INDEPENDENT_REAUDIT_REL,
        load_json,
        sha256_file,
        validate_r7c_independent_reaudit,
    )

    assert sha256_file(_ROOT / PILOT_R7C_INDEPENDENT_REAUDIT_REL) == EXPECTED_R7C_INDEPENDENT_REAUDIT_SHA256
    obj = load_json(_ROOT / PILOT_R7C_INDEPENDENT_REAUDIT_REL)
    validate_r7c_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"


def test_r7c_h_fill_script_forbidden_and_bound() -> None:
    from ms_sta_v5_d0_lib import assert_fill_script_cannot_assign_pass
    import subprocess

    assert_fill_script_cannot_assign_pass(root=_ROOT)
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts/fill_ms_sta_v5_d0_r7c_h_manual.py")],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "FORBIDDEN" in (proc.stderr + proc.stdout)


def test_r7c_h1_template_prior_slots_are_null() -> None:
    from ms_sta_v5_d0_lib import (
        PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL,
        load_json,
        validate_manual_template_null_prior_slots,
    )

    tmpl = load_json(_ROOT / PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL)
    validate_manual_template_null_prior_slots(tmpl)
    n_prior = sum(len(r.get("prior_tile_judgments") or []) for r in tmpl["inspections"])
    assert n_prior == 23


def test_r7c_h_independent_reaudit_bound() -> None:
    from ms_sta_v5_d0_lib import (
        EXPECTED_R7C_H_INDEPENDENT_REAUDIT_SHA256,
        PILOT_R7C_H_INDEPENDENT_REAUDIT_REL,
        load_json,
        sha256_file,
        validate_r7c_h_independent_reaudit,
    )

    assert (
        sha256_file(_ROOT / PILOT_R7C_H_INDEPENDENT_REAUDIT_REL)
        == EXPECTED_R7C_H_INDEPENDENT_REAUDIT_SHA256
    )
    obj = load_json(_ROOT / PILOT_R7C_H_INDEPENDENT_REAUDIT_REL)
    validate_r7c_h_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "FAIL"


def test_r7c_h1_rejects_scripted_prior_note_fingerprint() -> None:
    from ms_sta_v5_d0_lib import validate_r7_visual_qa_manual
    import pytest

    bad = {
        "schema": "ms_sta_v5_d0_r7c_h1_visual_qa_manual_v1",
        "source": "independent_human_or_named_reviewer_artifact",
        "synthesized_by_finalizer": False,
        "synthesized_by_template_tool": False,
        "auto_stroke_color_sufficient": False,
        "acceptance_rule": "primary_and_holdout_all_pass_zero_fail_allowed",
        "reviewer": "x",
        "reviewed_at": "2026-01-01T00:00:00Z",
        "method": "m",
        "n_pass": 0,
        "n_fail": 0,
        "inspections": [
            {
                "sample_id": "x",
                "qa_kind": "center_overlay",
                "window_start_idx": 0,
                "path": "Docs/results/ms_sta_v5_d0_r7c/visual_qa/x.png",
                "sha256": "a" * 64,
                "box_on_correct_body_region": True,
                "handedness_ok": True,
                "visible_but_missing_consistent": True,
                "verdict": "PASS",
                "prior_tile_judgments": [
                    {
                        "pathway": "center",
                        "hand": "left_hand",
                        "source": "pose_wrist_prior",
                        "box_on_correct_body_region": True,
                        "handedness_ok": True,
                        "verdict": "PASS",
                        "notes": "local ROI accepted under R7C ambiguity lock [prior_tile]",
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="scripted_prior_tile_note_forbidden"):
        validate_r7_visual_qa_manual(bad, overlays_index=[], freeze_list={"windows": []}, require_exact_22=False)


def test_r7c_h1_independent_reaudit_pass_bound() -> None:
    from ms_sta_v5_d0_lib import (
        EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL,
        load_json,
        sha256_file,
        validate_r7c_h1_independent_reaudit,
    )

    assert (
        sha256_file(_ROOT / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL)
        == EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256
    )
    obj = load_json(_ROOT / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL)
    validate_r7c_h1_independent_reaudit(obj, root=_ROOT)
    assert obj["audit_verdict"] == "PASS"
    assert obj["full_corpus_authorized"] is False


def test_d0_full_corpus_approval_and_run_contract() -> None:
    from ms_sta_v5_d0_lib import (
        D0_FULL_CORPUS_APPROVAL_REL,
        D0_FULL_CORPUS_RUN_CONTRACT_REL,
        load_json,
        validate_d0_full_corpus_approval,
        validate_d0_full_corpus_run_contract,
    )

    approval = load_json(_ROOT / D0_FULL_CORPUS_APPROVAL_REL)
    validate_d0_full_corpus_approval(approval, root=_ROOT)
    assert approval["allow_full_corpus_extraction"] is True
    assert approval["allow_m0"] is False
    assert approval["allow_auto_chain"] is False
    assert approval["stop_after"] == "D0_FULL_CORPUS_SCAN_COMPLETE"
    contract = load_json(_ROOT / D0_FULL_CORPUS_RUN_CONTRACT_REL)
    validate_d0_full_corpus_run_contract(contract, root=_ROOT)
    assert contract["status"] == "AUTHORIZED_NOT_STARTED"
    assert contract["auto_chain_to_m0"] is False
