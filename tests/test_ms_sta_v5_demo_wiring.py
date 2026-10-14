from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.deployment.backend_api import _ms_sta_v5_identity_errors, _ms_sta_v6_identity_errors
from src.deployment.skeleton_visualization import (
    render_edge_list,
    render_skeleton_gif,
    shoulder_center_for_display,
    skeleton_summary,
    stabilize_non_hand_display,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_identity() -> tuple[dict, dict]:
    reg = {
        "schema": "ms_sta_registry_v5_defense_go_v1",
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "d4_pass": True,
        "evaluation_locked": True,
        "active_variant": "b4_star_fixed",
        "primary_seed": 42,
        "forbidden_variants_on_demo": ["b4_star_e_stride4"],
    }
    return reg, {"variant": "b4_star_fixed", "seed": 42}


def test_v5_identity_accepts_only_defense_go_b4_star() -> None:
    reg, payload = _valid_identity()
    assert _ms_sta_v5_identity_errors(reg, payload) == []
    mutations = (
        ("campaign_status", "VALIDATION_NO_GO"),
        ("defense_ready", False),
        ("demo_allowed", False),
        ("d4_pass", False),
        ("evaluation_locked", False),
        ("active_variant", "b4_star_e_stride4"),
        ("primary_seed", 123),
        ("forbidden_variants_on_demo", []),
    )
    for key, value in mutations:
        bad = dict(reg)
        bad[key] = value
        assert _ms_sta_v5_identity_errors(bad, payload), key
    assert _ms_sta_v5_identity_errors(reg, {"variant": "b4_star_e_stride4", "seed": 42})


def test_v6_identity_accepts_only_defense_go_b4_v6_ctr() -> None:
    reg = {
        "schema": "ms_sta_registry_v6_defense_go_v1",
        "campaign_status": "DEFENSE_GO",
        "defense_ready": True,
        "demo_allowed": True,
        "d4_pass": True,
        "evaluation_locked": True,
        "active_variant": "b4_v6_ctr",
        "primary_seed": 42,
        "b4_ready": True,
        "e4_ready": False,
        "forbidden_variants_on_demo": ["e4_w192", "e4_w192_t1", "e4_w160_t1", "b4_star_e_stride4"],
    }
    payload = {"variant": "b4_v6_ctr", "seed": 42}
    assert _ms_sta_v6_identity_errors(reg, payload) == []
    bad = dict(reg)
    bad["e4_ready"] = True
    assert _ms_sta_v6_identity_errors(bad, payload)
    bad2 = dict(reg)
    bad2["active_variant"] = "e4_w192"
    assert _ms_sta_v6_identity_errors(bad2, payload)


def test_v5_demo_set_is_bound_to_oneshot_rows() -> None:
    demo = json.loads((ROOT / "data/processed/demo_set_ms_sta_v5.json").read_text(encoding="utf-8"))
    report = json.loads((ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_REPORT.json").read_text(encoding="utf-8"))
    preds = json.loads(
        (ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_PREDS_b4_star_fixed_seed42.json").read_text(
            encoding="utf-8"
        )
    )
    rows = {str(row["sample_id"]): row for row in preds["rows"]}
    assert demo["checkpoint_sha256"] == report["variants"]["b4_star_fixed"]["checkpoint_sha256"]
    assert set(demo["video_ids"]) == set(demo["scenarios"].values())
    improve = rows[demo["scenarios"]["top1_improvement"]]
    assist = rows[demo["scenarios"]["top5_assist"]]
    assert improve["correct"] is True
    assert assist["correct"] is False and assist["top5_hit"] is True
    assert demo["not_a_kpi"] is True


def test_skeleton_renderer_validates_and_renders(tmp_path: Path) -> None:
    frames = 4
    coords = np.zeros((frames, 75, 3), dtype=np.float32)
    coords[..., 0] = np.linspace(-1.0, 1.0, 75)[None, :]
    coords[..., 1] = np.linspace(-0.8, 0.8, 75)[None, :]
    coords[:, :, 0] += np.arange(frames, dtype=np.float32)[:, None] * 0.03
    validity = np.ones((frames, 75), dtype=np.float32)
    path = tmp_path / "sample.npz"
    np.savez(path, coordinates=coords, validity=validity, timestamps_sec=np.arange(frames) * 0.08)
    summary = skeleton_summary(path)
    assert summary["frames"] == 4
    assert summary["joints"] == 75
    assert summary["duration_sec"] == 0.24
    assert summary["valid_joint_fraction"] == 1.0
    assert summary["shoulder_center_max_abs"] > 0.0
    pytest = __import__("pytest")
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    gif = render_skeleton_gif(path, max_frames=4, fps=8)
    assert gif.startswith(b"GIF") and len(gif) > 1000

def test_skeleton_display_uses_anatomical_pose_edges_and_two_shoulder_center() -> None:
    edges = set(render_edge_list())
    assert {(4, 5), (4, 6), (6, 8), (5, 7), (7, 9)} <= edges
    assert (8, 10) not in edges
    assert (5, 6) not in edges

    coords = np.zeros((3, 75, 3), dtype=np.float32)
    coords[:, 4, :2] = np.asarray([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    coords[:, 5, :2] = np.asarray([[4.0, 3.0], [5.0, 4.0], [6.0, 5.0]])
    original = coords.copy()
    valid = np.ones((3, 75), dtype=bool)
    centered, centers = shoulder_center_for_display(coords, valid)
    assert np.array_equal(coords, original)
    assert np.allclose(centers, [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    assert np.allclose((centered[:, 4, :2] + centered[:, 5, :2]) / 2.0, 0.0)


def test_display_smoothing_does_not_modify_hands_or_input() -> None:
    coords = np.zeros((3, 75, 3), dtype=np.float32)
    coords[1, 4, 0] = 1.0
    coords[1, 33, 0] = 1.0
    original = coords.copy()
    valid = np.ones((3, 75), dtype=bool)
    shown = stabilize_non_hand_display(coords, valid, alpha=0.5)
    assert np.array_equal(coords, original)
    assert shown[1, 4, 0] == 0.5
    assert shown[1, 33, 0] == 1.0
