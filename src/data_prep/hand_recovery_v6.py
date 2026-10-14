"""D0-PILOT-R6 hand detector recovery (pose-guided Hands fallback).

Does NOT falsify Holistic missing flags. Fallback only writes effective
observations when standalone Hands returns a valid hand that can be
unambiguously assigned to LH/RH via pose wrist distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

ObservationSource = Literal["holistic", "hands_fallback", "missing"]

# MediaPipe Pose landmark indices
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

LH_SLICE = slice(33, 54)
RH_SLICE = slice(54, 75)


@dataclass
class HandObservation:
    landmarks_21x3: np.ndarray | None
    source: ObservationSource
    pose_guided_crop: list[int] | None = None  # x0,y0,x1,y1 pixels
    wrist_assignment_distance: float | None = None
    fallback_rejection_reason: str | None = None


@dataclass
class FrameRecoveryResult:
    joints_raw_holistic: np.ndarray
    raw_pose: np.ndarray
    raw_holistic_missing: np.ndarray
    effective_joints: np.ndarray
    effective_missing: np.ndarray
    observation_source: dict[str, ObservationSource] = field(
        default_factory=lambda: {"left_hand": "missing", "right_hand": "missing", "face": "missing", "pose": "missing"}
    )
    pose_guided_crop: dict[str, list[int] | None] = field(default_factory=dict)
    wrist_assignment_distance: dict[str, float | None] = field(default_factory=dict)
    fallback_rejection_reason: dict[str, str | None] = field(default_factory=dict)


class HandRecoveryEngine:
    """Holistic primary + optional Hands fallback arms C0–C3."""

    def __init__(
        self,
        *,
        arm: Literal["C0", "C1", "C2", "C3"] = "C3",
        model_complexity: int = 0,
        store_stride: int = 2,
        crop_scale: float = 1.8,
        min_crop_px: int = 96,
        max_assign_dist: float = 0.18,
        ambiguous_margin: float = 0.03,
    ) -> None:
        import mediapipe as mp

        if arm not in {"C0", "C1", "C2", "C3"}:
            raise ValueError(f"bad_arm:{arm}")
        self.arm = arm
        self.store_stride = max(1, int(store_stride))
        self.model_complexity = int(model_complexity)
        self.crop_scale = float(crop_scale)
        self.min_crop_px = int(min_crop_px)
        self.max_assign_dist = float(max_assign_dist)
        self.ambiguous_margin = float(ambiguous_margin)

        self.mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            refine_face_landmarks=True,
        )
        # Hands only for C2/C3; static mode on crops/full frames for recovery.
        self.hands = None
        if arm in {"C2", "C3"}:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        self.pose_indices = [0, 2, 5, 9, 11, 12, 13, 14, 15, 16, 23, 24]
        self.face_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        ]

    def close(self) -> None:
        self.holistic.close()
        if self.hands is not None:
            self.hands.close()

    def _holistic_parse(self, frame_bgr: np.ndarray) -> FrameRecoveryResult:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        pose_arr = np.zeros((33, 3), dtype=np.float32)
        face_arr = np.zeros((478, 3), dtype=np.float32)
        lh_arr = np.zeros((21, 3), dtype=np.float32)
        rh_arr = np.zeros((21, 3), dtype=np.float32)
        pose_ok = face_ok = lh_ok = rh_ok = False

        if results.pose_landmarks:
            pose_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
            pose_ok = True
        if results.face_landmarks:
            face_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark],
                dtype=np.float32,
            )
            face_ok = True
        if results.left_hand_landmarks:
            lh_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            )
            lh_ok = True
        if results.right_hand_landmarks:
            rh_arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            )
            rh_ok = True

        selected_pose = pose_arr[self.pose_indices]
        selected_face = face_arr[self.face_indices]
        joints = np.concatenate([selected_pose, selected_face, lh_arr, rh_arr], axis=0)
        missing = np.zeros((75,), dtype=bool)
        if not pose_ok:
            missing[0:12] = True
        if not face_ok:
            missing[12:33] = True
        if not lh_ok:
            missing[33:54] = True
        if not rh_ok:
            missing[54:75] = True

        sources: dict[str, ObservationSource] = {
            "pose": "holistic" if pose_ok else "missing",
            "face": "holistic" if face_ok else "missing",
            "left_hand": "holistic" if lh_ok else "missing",
            "right_hand": "holistic" if rh_ok else "missing",
        }
        return FrameRecoveryResult(
            joints_raw_holistic=joints.copy(),
            raw_pose=pose_arr.copy(),
            raw_holistic_missing=missing.copy(),
            effective_joints=joints.copy(),
            effective_missing=missing.copy(),
            observation_source=sources,
            pose_guided_crop={"left_hand": None, "right_hand": None},
            wrist_assignment_distance={"left_hand": None, "right_hand": None},
            fallback_rejection_reason={"left_hand": None, "right_hand": None},
        )

    def _shoulder_width_px(self, pose: np.ndarray, w: int, h: int) -> float:
        if bool(np.all(pose[POSE_LEFT_SHOULDER] == 0) and np.all(pose[POSE_RIGHT_SHOULDER] == 0)):
            return float(max(w, h) * 0.25)
        lx, ly = float(pose[POSE_LEFT_SHOULDER, 0] * w), float(pose[POSE_LEFT_SHOULDER, 1] * h)
        rx, ry = float(pose[POSE_RIGHT_SHOULDER, 0] * w), float(pose[POSE_RIGHT_SHOULDER, 1] * h)
        return float(max(np.hypot(lx - rx, ly - ry), max(w, h) * 0.12))

    def _crop_around_wrist(
        self, frame_bgr: np.ndarray, pose: np.ndarray, which: Literal["left", "right"]
    ) -> tuple[np.ndarray | None, list[int] | None, str | None]:
        h, w = frame_bgr.shape[:2]
        wi = POSE_LEFT_WRIST if which == "left" else POSE_RIGHT_WRIST
        ei = POSE_LEFT_ELBOW if which == "left" else POSE_RIGHT_ELBOW
        wrist = pose[wi]
        if float(np.linalg.norm(wrist[:2])) < 1e-8:
            return None, None, "pose_wrist_missing"
        cx = float(wrist[0] * w)
        cy = float(wrist[1] * h)
        # Bias crop slightly toward elbow→wrist direction when elbow exists.
        elbow = pose[ei]
        if float(np.linalg.norm(elbow[:2])) > 1e-8:
            ex, ey = float(elbow[0] * w), float(elbow[1] * h)
            vx, vy = cx - ex, cy - ey
            n = max(float(np.hypot(vx, vy)), 1.0)
            cx = cx + 0.15 * (vx / n) * self._shoulder_width_px(pose, w, h)
            cy = cy + 0.15 * (vy / n) * self._shoulder_width_px(pose, w, h)
        half = max(self.min_crop_px / 2.0, self.crop_scale * 0.5 * self._shoulder_width_px(pose, w, h))
        x0 = int(max(0, np.floor(cx - half)))
        y0 = int(max(0, np.floor(cy - half)))
        x1 = int(min(w, np.ceil(cx + half)))
        y1 = int(min(h, np.ceil(cy + half)))
        if (x1 - x0) < self.min_crop_px or (y1 - y0) < self.min_crop_px:
            return None, None, "crop_too_small"
        crop = frame_bgr[y0:y1, x0:x1]
        return crop, [x0, y0, x1, y1], None

    def _hands_detections(
        self, image_bgr: np.ndarray
    ) -> list[dict[str, Any]]:
        if self.hands is None:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        out: list[dict[str, Any]] = []
        if not res.multi_hand_landmarks:
            return out
        handedness = res.multi_handedness or []
        for i, hlm in enumerate(res.multi_hand_landmarks):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in hlm.landmark], dtype=np.float32)
            label = None
            score = None
            if i < len(handedness):
                cats = handedness[i].classification
                if cats:
                    label = str(cats[0].label).lower()  # "left"/"right" in image coords
                    score = float(cats[0].score)
            out.append({"landmarks": arr, "handedness_label": label, "handedness_score": score})
        return out

    def _map_crop_landmarks_to_frame(
        self, landmarks_crop_norm: np.ndarray, crop_box: list[int], frame_w: int, frame_h: int
    ) -> np.ndarray:
        x0, y0, x1, y1 = crop_box
        cw = max(1, x1 - x0)
        ch = max(1, y1 - y0)
        out = landmarks_crop_norm.copy()
        out[:, 0] = (x0 + landmarks_crop_norm[:, 0] * cw) / float(frame_w)
        out[:, 1] = (y0 + landmarks_crop_norm[:, 1] * ch) / float(frame_h)
        return out

    def _assign_hand(
        self,
        *,
        detections: list[dict[str, Any]],
        pose: np.ndarray,
        target: Literal["left", "right"],
    ) -> HandObservation:
        if not detections:
            return HandObservation(None, "missing", fallback_rejection_reason="hands_no_detection")
        tw = pose[POSE_LEFT_WRIST if target == "left" else POSE_RIGHT_WRIST, :2]
        ow = pose[POSE_RIGHT_WRIST if target == "left" else POSE_LEFT_WRIST, :2]
        if float(np.linalg.norm(tw)) < 1e-8:
            return HandObservation(None, "missing", fallback_rejection_reason="pose_wrist_missing")

        scored: list[tuple[float, float, dict[str, Any]]] = []
        for det in detections:
            wrist = det["landmarks"][0, :2]
            d_t = float(np.linalg.norm(wrist - tw))
            d_o = float(np.linalg.norm(wrist - ow)) if float(np.linalg.norm(ow)) > 1e-8 else 1e9
            scored.append((d_t, d_o, det))
        scored.sort(key=lambda x: x[0])
        best_d, best_other, best = scored[0]
        if best_d > self.max_assign_dist:
            return HandObservation(
                None,
                "missing",
                wrist_assignment_distance=best_d,
                fallback_rejection_reason="wrist_distance_too_large",
            )
        # Ambiguous: nearly equidistant to both wrists or second candidate close.
        if abs(best_d - best_other) < self.ambiguous_margin:
            return HandObservation(
                None,
                "missing",
                wrist_assignment_distance=best_d,
                fallback_rejection_reason="ambiguous_wrist_assignment",
            )
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < self.ambiguous_margin:
            return HandObservation(
                None,
                "missing",
                wrist_assignment_distance=best_d,
                fallback_rejection_reason="overlapping_hands_unresolved",
            )
        # Handedness is NOT a hard reject for non-mirrored world-facing video.
        # MediaPipe Hands labels assume selfie/mirrored input; wrist distance is primary.
        _ = best.get("handedness_label")
        _ = best.get("handedness_score")
        return HandObservation(
            best["landmarks"].astype(np.float32),
            "hands_fallback",
            wrist_assignment_distance=best_d,
        )

    def _apply_fallback(self, frame_bgr: np.ndarray, base: FrameRecoveryResult) -> FrameRecoveryResult:
        if self.arm not in {"C2", "C3"} or self.hands is None:
            return base
        h, w = frame_bgr.shape[:2]
        pose = base.raw_pose
        need_lh = bool(base.raw_holistic_missing[LH_SLICE].all())
        need_rh = bool(base.raw_holistic_missing[RH_SLICE].all())
        if not need_lh and not need_rh:
            return base

        if self.arm == "C2":
            dets_frame = self._hands_detections(frame_bgr)
            if need_lh:
                obs = self._assign_hand(detections=dets_frame, pose=pose, target="left")
                base.fallback_rejection_reason["left_hand"] = obs.fallback_rejection_reason
                base.wrist_assignment_distance["left_hand"] = obs.wrist_assignment_distance
                if obs.source == "hands_fallback" and obs.landmarks_21x3 is not None:
                    base.effective_joints[LH_SLICE] = obs.landmarks_21x3
                    base.effective_missing[LH_SLICE] = False
                    base.observation_source["left_hand"] = "hands_fallback"
                else:
                    base.observation_source["left_hand"] = "missing"
            if need_rh:
                obs = self._assign_hand(detections=dets_frame, pose=pose, target="right")
                base.fallback_rejection_reason["right_hand"] = obs.fallback_rejection_reason
                base.wrist_assignment_distance["right_hand"] = obs.wrist_assignment_distance
                if obs.source == "hands_fallback" and obs.landmarks_21x3 is not None:
                    base.effective_joints[RH_SLICE] = obs.landmarks_21x3
                    base.effective_missing[RH_SLICE] = False
                    base.observation_source["right_hand"] = "hands_fallback"
                else:
                    base.observation_source["right_hand"] = "missing"
            return base

        # C3: pose-guided crops, one per missing hand; assign within that crop's detections.
        for which, sl, key in (
            ("left", LH_SLICE, "left_hand"),
            ("right", RH_SLICE, "right_hand"),
        ):
            if not (need_lh if which == "left" else need_rh):
                continue
            crop, box, reason = self._crop_around_wrist(frame_bgr, pose, which)  # type: ignore[arg-type]
            base.pose_guided_crop[key] = box
            if crop is None:
                base.fallback_rejection_reason[key] = reason
                base.observation_source[key] = "missing"
                continue
            dets = self._hands_detections(crop)
            # Map crop-normalized landmarks to full-frame normalized.
            mapped: list[dict[str, Any]] = []
            for d in dets:
                md = dict(d)
                md["landmarks"] = self._map_crop_landmarks_to_frame(d["landmarks"], box, w, h)
                mapped.append(md)
            obs = self._assign_hand(detections=mapped, pose=pose, target=which)  # type: ignore[arg-type]
            base.fallback_rejection_reason[key] = obs.fallback_rejection_reason
            base.wrist_assignment_distance[key] = obs.wrist_assignment_distance
            if obs.source == "hands_fallback" and obs.landmarks_21x3 is not None:
                base.effective_joints[sl] = obs.landmarks_21x3
                base.effective_missing[sl] = False
                base.observation_source[key] = "hands_fallback"
            else:
                base.observation_source[key] = "missing"
        return base

    def process_video(
        self, video_path: Any
    ) -> dict[str, Any]:
        """Run recovery arm; return arrays at store_stride indices (native frame index)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot_open:{video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0

        stored: list[FrameRecoveryResult] = []
        frame_idx: list[int] = []
        stamps: list[float] = []
        native_i = 0
        process_every = self.arm != "C0"  # C1/C2/C3: process all frames for tracker

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            should_store = (native_i % self.store_stride) == 0
            if process_every or should_store:
                base = self._holistic_parse(frame)
                if should_store:
                    if self.arm in {"C2", "C3"}:
                        base = self._apply_fallback(frame, base)
                    stored.append(base)
                    frame_idx.append(native_i)
                    stamps.append(native_i / fps)
            native_i += 1
        cap.release()
        if len(stored) < 5:
            raise RuntimeError(f"too_few_frames:{len(stored)}")

        def stack_attr(getter):
            return np.stack([getter(s) for s in stored], axis=0)

        obs_left = np.asarray([s.observation_source["left_hand"] for s in stored])
        obs_right = np.asarray([s.observation_source["right_hand"] for s in stored])
        return {
            "arm": self.arm,
            "n_native_frames": native_i,
            "fps": fps,
            "original_frame_index": np.asarray(frame_idx, dtype=np.int32),
            "timestamps_sec": np.asarray(stamps, dtype=np.float64),
            "raw_landmarks": stack_attr(lambda s: s.joints_raw_holistic).astype(np.float32),
            "effective_landmarks": stack_attr(lambda s: s.effective_joints).astype(np.float32),
            "raw_pose": stack_attr(lambda s: s.raw_pose).astype(np.float32),
            "raw_holistic_missing": stack_attr(lambda s: s.raw_holistic_missing).astype(bool),
            "effective_missing": stack_attr(lambda s: s.effective_missing).astype(bool),
            "observation_source_left_hand": obs_left,
            "observation_source_right_hand": obs_right,
            "pose_guided_crop_left": [s.pose_guided_crop.get("left_hand") for s in stored],
            "pose_guided_crop_right": [s.pose_guided_crop.get("right_hand") for s in stored],
            "wrist_assignment_distance_left": [
                s.wrist_assignment_distance.get("left_hand") for s in stored
            ],
            "wrist_assignment_distance_right": [
                s.wrist_assignment_distance.get("right_hand") for s in stored
            ],
            "fallback_rejection_reason_left": [
                s.fallback_rejection_reason.get("left_hand") for s in stored
            ],
            "fallback_rejection_reason_right": [
                s.fallback_rejection_reason.get("right_hand") for s in stored
            ],
        }
