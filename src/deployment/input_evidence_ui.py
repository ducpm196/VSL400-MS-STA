"""Streamlit-only render helpers for sealed input evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.deployment.input_evidence import (
    _ROOT,
    make_browser_playable_bytes,
    rgb_token_summary,
    skeleton_tensor_summary,
)
from src.deployment.skeleton_visualization import render_skeleton_gif


@st.cache_data(show_spinner="Transcoding preview to H.264 for browser…")
def browser_playable_bytes(src_path: str, mtime_ns: int) -> tuple[bytes, str]:
    del mtime_ns
    return make_browser_playable_bytes(Path(src_path), root=_ROOT)


@st.cache_data(show_spinner="Rendering Skeleton v3 motion…")
def skeleton_gif_bytes(path: str, mtime_ns: int, stabilize_display: bool) -> bytes:
    del mtime_ns
    return render_skeleton_gif(Path(path), stabilize_display=stabilize_display)


def show_front_video(video_path: Path, *, root: Path | None = None) -> None:
    root = root or _ROOT
    try:
        data, note = browser_playable_bytes(
            str(video_path.resolve()), video_path.stat().st_mtime_ns
        )
        st.video(data, format="video/mp4")
        st.caption(note)
    except Exception as exc:
        st.error(
            f"Browser playback failed: {exc}\n\n"
            "FFmpeg on PATH is required to transcode mpeg4 → H.264 preview."
        )
        return
    try:
        rel = video_path.resolve().relative_to(root)
    except ValueError:
        rel = video_path
    st.caption(f"Source: `{rel}`")


def render_novice_glossary() -> None:
    with st.expander("Giải thích nhanh (cho người mới)", expanded=False):
        st.markdown(
            "- **RGB video:** dữ liệu camera gốc (Front-view).\n"
            "- **RGB token:** vector đặc trưng đã trích từ các cửa sổ thời gian "
            "(4 vùng × dim 2304), không phải ảnh thô đưa thẳng vào classifier.\n"
            "- **Skeleton raw:** tọa độ detector trong cache sealed; "
            "là tensor mô hình thực sự dùng.\n"
            "- **Stabilized (VIZ-ONLY):** chỉ làm mượt torso/face để nhìn; "
            "**không** phải tensor train/inference.\n"
            "- **Validity mask:** vị trí khớp có quan sát hợp lệ.\n"
            "- **Temporal attention:** so khớp các thời điểm hợp lệ để học quan hệ chuyển động.\n"
            "- **Cross-attention:** một luồng dùng thông tin luồng kia để chọn đặc trưng.\n"
            "- **Stream gate:** học trọng số kết hợp hai luồng.\n"
            "- **Chiều biểu diễn:** Unified Attention dùng 256 chiều. Thử nghiệm nén mạng dùng 192 chiều và không phải mô hình bảo vệ. "
            "Đây không phải số khung hình hay số khớp."
        )


def render_two_stream_input(
    *,
    sample_id: str,
    video_path: Path,
    rgb_path: Path,
    skel_path: Path,
    widget_key: str,
    model_label: str = "Unified Attention",
) -> dict[str, Any]:
    """Render Front RGB + Skeleton evidence. Does not mutate tensors."""
    evidence: dict[str, Any] = {"sample_id": sample_id}
    col_video, col_skel = st.columns(2, gap="large")
    with col_video:
        st.markdown("**RGB · Front camera (dữ liệu gốc)**")
        if video_path.is_file():
            show_front_video(video_path)
        else:
            st.error(f"Video missing for `{sample_id}`")
        if rgb_path.is_file():
            rgb_info = rgb_token_summary(rgb_path)
            evidence["rgb"] = rgb_info
            st.caption(
                f"RGB token v3 · {rgb_info['windows']} temporal windows · "
                f"4 regions · feature dim **{rgb_info['feature_dim']}** · "
                f"SHA `{str(rgb_info['cache_sha256'])[:12]}…`"
            )
            if rgb_info.get("missing"):
                st.caption(f"Missing counts by region: {rgb_info['missing']}")
        else:
            st.error("RGB token v3 cache missing — fail closed.")
    with col_skel:
        st.markdown("**Skeleton v3 · 75 khớp (cache sealed)**")
        if skel_path.is_file():
            skeleton_view = st.radio(
                "Skeleton visualization",
                ("Raw sealed motion (model tensor)", "Stabilized display (VIZ-ONLY)"),
                horizontal=True,
                key=f"skel_view_{widget_key}_{sample_id}",
            )
            stabilize = skeleton_view.startswith("Stabilized")
            try:
                gif = skeleton_gif_bytes(
                    str(skel_path.resolve()),
                    skel_path.stat().st_mtime_ns,
                    stabilize,
                )
                st.image(gif, width="stretch")
                skel_info = skeleton_tensor_summary(skel_path)
                evidence["skeleton"] = skel_info
                st.caption(
                    f"Skeleton v3 · {skel_info['frames']} frames · {skel_info['joints']} joints · "
                    f"valid {100.0 * float(skel_info['valid_joint_fraction']):.1f}% "
                    f"({skel_info['valid_joint_count']}/{skel_info['total_joint_slots']}) · "
                    f"SHA `{str(skel_info['cache_sha256'])[:12]}…`"
                )
                if stabilize:
                    st.warning(
                        "VIZ-ONLY: torso/face smoothing for display. "
                        f"{model_label} always reads the unchanged sealed cache."
                    )
                else:
                    st.caption("Raw sealed motion — matches the tensor consumed by the model.")
            except (OSError, ValueError) as exc:
                st.error(f"Skeleton visualization failed closed: {exc}")
        else:
            st.error("Skeleton v3 cache missing — fail closed.")
    return evidence


def render_pipeline_diagram(*, width_b4: int = 256, width_e4: int | None = None) -> None:
    if width_e4 is None:
        st.markdown(
            f"""
```text
Front video
  → RGB token v3 (4 windows × 4 regions × 2304) + Skeleton v3 (T × 75)
  → projection d={width_b4}
  → spatial / temporal / cross-attention
  → stream gate + residual
  → classifier 400 lớp → Top-k
```
"""
        )
    else:
        st.markdown(
            f"""
```text
Same external input (RGB token v3 + Skeleton v3)
  ├─ Unified Attention:  projection d={width_b4} → MS-STA blocks → 400-class
  └─ E4:     projection d={width_e4} → MS-STA-lite    → 400-class
E4 does NOT grayscale RGB, drop joints, stride time, or change caches.
```
"""
        )
