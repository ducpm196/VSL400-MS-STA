#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build thesis figures from the locked VSL400 caches (real data, faces blurred).

Sample 018922 (gloss ĐIỆN_THOẠI, signer 24, test split). Sources:
  video  data/raw/VSL-Cropped/raw/raw/VSL400/Part_6/split_6/front_view/018922.mp4
  skel   data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3/018922.npz
  rgb    data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70/rgb_token_v3/018922.{npz,meta.json}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/thesis/figures"
OUT.mkdir(parents=True, exist_ok=True)
SID = "018922"
GLOSS = "ĐIỆN_THOẠI"
VIDEO = ROOT / f"data/raw/VSL-Cropped/raw/raw/VSL400/Part_6/split_6/front_view/{SID}.mp4"
SKEL = ROOT / f"data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/skeleton_v3/{SID}.npz"
RGBZ = ROOT / f"data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70/rgb_token_v3/{SID}.npz"
META = ROOT / f"data/features/ms_sta_v5_d0_full_v15_ca3a7b75fe70/rgb_token_v3/{SID}.meta.json"

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
COL = {"pose": "#ff8c00", "face": "#b04fa0", "left_hand": "#2ca02c", "right_hand": "#1f6fd0"}
# Local pose layout (MediaPipe [0,2,5,9,11,12,13,14,15,16,23,24]):
# 4/5 shoulders, 6/7 elbows, 8/9 wrists, 10/11 hips. Anatomical lines for display only.
POSE_DRAW = [(4, 5), (4, 6), (6, 8), (5, 7), (7, 9), (4, 10), (5, 11), (10, 11)]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
              (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9),
              (9, 13), (13, 17)]


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def blur_box(img: np.ndarray, x0, y0, x1, y1) -> None:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
    roi = img[y0:y1, x0:x1]
    if roi.size:
        k = max(41, (min(roi.shape[:2]) // 2) | 1)
        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)


def blur_face(img: np.ndarray, pts: np.ndarray, pad: float = 0.45) -> None:
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    bw, bh = x1 - x0, y1 - y0
    blur_box(img, x0 - pad * bw, y0 - 1.3 * bh, x1 + pad * bw, y1 + 0.5 * bh)


z = np.load(SKEL, allow_pickle=True)
coords = z["effective_landmarks"].astype(float)
missing = z["effective_missing"]
validity = z["validity"]
parts = z["joint_part"]
orig_idx = z["original_frame_index"]
ts = z["timestamps_sec"]
frames = read_frames(VIDEO)
H, W = frames[0].shape[:2]
FACE = np.arange(12, 33)
LH = np.arange(33, 54)
RH = np.arange(54, 75)


def pix(k: int) -> np.ndarray:
    p = coords[k, :, :2].copy()
    p[missing[k]] = np.nan
    p[:, 0] *= W
    p[:, 1] *= H
    return p


steps = [int(round(x)) for x in np.linspace(2, len(orig_idx) - 3, 4)]
CROP = (slice(int(0.05 * H), int(0.95 * H)), slice(int(0.22 * W), int(0.78 * W)))  # keep signer, drop empty backdrop


def crop(img):
    return img[CROP]


def shift(p):
    q = p.copy()
    q[:, 0] -= CROP[1].start
    q[:, 1] -= CROP[0].start
    return q


# ---- figure: raw frames + skeleton (2 rows x 4 frames)
fig, axes = plt.subplots(2, 4, figsize=(6.3, 4.7))
for c, k in enumerate(steps):
    f = frames[int(orig_idx[k])].copy()
    p = pix(k)
    blur_face(f, p[FACE])
    fc = crop(f)
    ps = shift(p)
    axes[0, c].imshow(fc)
    axes[0, c].set_title(f"Khung {int(orig_idx[k])} ({ts[k]:.2f} s)".replace(".", ","), fontsize=8)
    axes[1, c].imshow(fc)
    for a, b in POSE_DRAW:
        if not (np.isnan(ps[a]).any() or np.isnan(ps[b]).any()):
            axes[1, c].plot(*zip(ps[a], ps[b]), color=COL["pose"], lw=1.1)
    for base, name in ((33, "left_hand"), (54, "right_hand")):
        for a, b in HAND_BONES:
            A, B = ps[base + a], ps[base + b]
            if not (np.isnan(A).any() or np.isnan(B).any()):
                axes[1, c].plot(*zip(A, B), color=COL[name], lw=0.6)
    for name, idx in (("pose", np.arange(0, 12)), ("face", FACE), ("left_hand", LH), ("right_hand", RH)):
        q = ps[idx]
        q = q[~np.isnan(q).any(axis=1)]
        axes[1, c].scatter(q[:, 0], q[:, 1], s=2.5 if name != "pose" else 6, color=COL[name], zorder=3)
    n_ok = int((~missing[k]).sum())
    axes[1, c].set_title(f"{n_ok}/75 điểm hợp lệ", fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
handles = [plt.Line2D([], [], marker="o", ls="", color=COL[k], label=l) for k, l in
           (("pose", "Thân (12 điểm)"), ("face", "Mặt (21 điểm)"), ("left_hand", "Tay trái (21 điểm)"),
            ("right_hand", "Tay phải (21 điểm)"))]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.text(0.005, 0.74, "(a) Ảnh RGB", rotation=90, va="center", fontsize=8)
fig.text(0.005, 0.30, "(b) 75 điểm mốc", rotation=90, va="center", fontsize=8)
fig.tight_layout(rect=(0.02, 0.05, 1, 1))
fig.savefig(OUT / f"fig_frames_skeleton_{SID}.png", dpi=300)
plt.close(fig)

# ---- figure: ROI boxes at window centres (coordinates from the locked meta, canvas 224 -> 1080)
meta = json.loads(META.read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0))
for ax, w in zip(axes, meta["windows"]):
    c = int(w["center_frame"])
    s = W / float(w["transform"]["dst"])
    f = frames[c].copy()
    boxes = {k: [v * s for v in b] for k, b in w["roi_boxes_canvas224"].items() if b is not None}
    if "face" in boxes:
        blur_box(f, *boxes["face"])
    fc = crop(f)
    ax.imshow(fc)
    for name, lab in (("face", "Mặt"), ("left_hand", "Tay trái"), ("right_hand", "Tay phải")):
        if name not in boxes:
            continue
        x0, y0, x1, y1 = boxes[name]
        x0 -= CROP[1].start
        x1 -= CROP[1].start
        y0 -= CROP[0].start
        y1 -= CROP[0].start
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.8, color=COL[name]))
    ax.set_title(f"Cửa sổ khung {w['start_frame']}–{w['end_frame']} (tâm {c})", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
handles = [plt.Rectangle((0, 0), 1, 1, fill=False, color=COL[k], lw=1.8, label=l) for k, l in
           (("face", "Vùng mặt"), ("left_hand", "Vùng tay trái"), ("right_hand", "Vùng tay phải"))]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / f"fig_roi_{SID}.png", dpi=300)
plt.close(fig)

# ---- figure: skeleton validity mask + hand trajectories
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8), gridspec_kw={"width_ratios": [1.25, 1]})
im = ax[0].imshow(validity.T, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
for b in np.cumsum([12, 21, 21]):
    ax[0].axhline(b - 0.5, color="k", lw=0.6)
ax[0].set_yticks([5.5, 22, 43, 64])
ax[0].set_yticklabels(["Thân (12)", "Mặt (21)", "Tay trái (21)", "Tay phải (21)"])
ax[0].set_xlabel(f"Bước thời gian của chuỗi khung xương ({len(ts)} bước)")
ax[0].set_title("(a) Mặt nạ hợp lệ của 75 điểm mốc", fontsize=8)
fig.colorbar(im, ax=ax[0], fraction=0.035, pad=0.02)
for name, idx, col, lab in (("left_hand", LH, COL["left_hand"], "Tay trái"),
                            ("right_hand", RH, COL["right_hand"], "Tay phải")):
    cy = []
    for k in range(len(ts)):
        v = coords[k, idx, 1][~missing[k, idx]]
        cy.append(np.mean(v) if len(v) else np.nan)
    ax[1].plot(ts, 1 - np.array(cy), marker="o", ms=3, color=col, label=lab)
ax[1].set_xlabel("Thời gian (giây)")
ax[1].set_ylabel("Độ cao tâm bàn tay (1 = đỉnh khung)")
ax[1].set_title("(b) Quỹ đạo theo chiều dọc\ncủa tâm hai bàn tay", fontsize=8)
ax[1].legend(frameon=False)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / f"fig_skeleton_signals_{SID}.png", dpi=300)
plt.close(fig)

# ---- figure: RGB region tokens that enter the fusion block
r = np.load(RGBZ, allow_pickle=True)
regions = ["global", "left_hand", "right_hand", "face"]
names_vi = ["Toàn cảnh", "Tay trái", "Tay phải", "Mặt"]
mat = np.vstack([r[f"token_{g}"] for g in regions])
fig, ax = plt.subplots(figsize=(6.3, 2.6))
show = mat[:, :256]
im = ax.imshow(show, aspect="auto", cmap="magma", interpolation="nearest", vmin=0, vmax=np.percentile(show, 99))
yt, yl = [], []
for gi, nm in enumerate(names_vi):
    for wi in range(mat.shape[0] // len(regions)):
        yt.append(gi * (mat.shape[0] // len(regions)) + wi)
        yl.append(f"{nm} · cửa sổ {wi + 1}")
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=6.5)
for gi in range(1, 4):
    ax.axhline(gi * (mat.shape[0] // len(regions)) - 0.5, color="w", lw=1)
ax.set_xlabel("Chỉ số chiều đặc trưng (hiển thị 256 trong 2.304 chiều)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
fig.tight_layout()
fig.savefig(OUT / f"fig_rgb_tokens_{SID}.png", dpi=300)
plt.close(fig)

# fig_ui_rgb_skeleton.png and fig_ui_top5.png are manual screenshots of the demo; they are not regenerated here.

info = {
    "sample_id": SID, "gloss": GLOSS, "signer": 24, "split": "test",
    "video": str(VIDEO.relative_to(ROOT)).replace("\\", "/"), "video_size": [W, H], "fps": meta["fps"],
    "n_frames": len(frames), "n_skeleton_steps": int(len(orig_idx)),
    "skeleton_original_frames_shown": [int(orig_idx[k]) for k in steps],
    "valid_points_shown": [int((~missing[k]).sum()) for k in steps],
    "validity_mean": round(float(validity.mean()), 4),
    "windows": [[w["start_frame"], w["end_frame"], w["center_frame"]] for w in meta["windows"]],
    "window_size": meta["window_size"], "window_stride": meta["window_stride"],
    "faces_blurred": True,
    "pose_lines_are_display_only": True,
    "ui_screenshots": ["fig_ui_rgb_skeleton.png", "fig_ui_top5.png"],
}
(OUT / "FIGURE_PROVENANCE.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(info, ensure_ascii=False))
