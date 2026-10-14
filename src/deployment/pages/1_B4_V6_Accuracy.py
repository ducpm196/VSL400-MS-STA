"""Streamlit page: Unified Attention defense result, visual-first and evidence-bound."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests
import streamlit as st

from src.deployment.campaign_evidence import load_b4_canonical_test, load_scientific_claim_lock
from src.deployment.demo_visuals import render_flow, render_test_comparison_chart
from src.deployment.input_evidence import (
    load_e4_input_demo_manifest,
    validate_e4_input_demo_manifest,
)
from src.deployment.input_evidence_ui import render_novice_glossary, render_two_stream_input

_API = "http://127.0.0.1:8000"
_CKPT_PREFIX = "0b73f8f231525f23"

_VAL_ROWS = [
    {"Model": "Concat RGB+Skeleton", "Top-1 (%)": 82.54},
    {"Model": "Simple Combined", "Top-1 (%)": 80.54},
    {"Model": "Unified Attention", "Top-1 (%)": 84.81},
]

st.set_page_config(page_title="Unified Attention · Accuracy", layout="wide")
st.title("Luận điểm 1 — Unified Attention: hợp nhất RGB–Skeleton")
st.info("Unified Attention là mô hình nhận dạng ký hiệu đơn lẻ đã được kiểm chứng trên tập 400 lớp.")
st.caption(
    "Đây là hệ thống nhận dạng ký hiệu đơn lẻ, chưa phải hệ thống dịch câu ngôn ngữ ký hiệu liên tục."
)

try:
    health = requests.get(f"{_API}/health", timeout=3).json()
    claim = load_scientific_claim_lock(_ROOT)
    b4m = claim["B4"]
    b4_can = load_b4_canonical_test(_ROOT)
except Exception as exc:
    st.error(f"API/evidence unavailable: {exc}")
    st.stop()
if health.get("ms_sta_ready") is not True or health.get("engine") != "ms_sta_v6_b4_v6_ctr":
    st.error("Hệ thống dừng vì engine đang chạy không phải Unified Attention của bản bảo vệ.")
    st.stop()
if health.get("e4_ready") is True:
    st.error(
        "Hệ thống dừng vì cờ Optimized inference đang bật. "
        "Giao diện bảo vệ chỉ được đọc đặc trưng đã tính sẵn của Unified Attention."
    )
    st.stop()

ckpt = str(health.get("ms_sta_checkpoint_sha256") or "")
_TEST_ROWS = [
    {"Model": "RGB-only", "Top-1 (%)": 29.80, "Top-5 (%)": 56.06},
    {"Model": "Concat RGB+Skeleton", "Top-1 (%)": 46.62, "Top-5 (%)": 71.22},
    {"Model": "Simple Combined", "Top-1 (%)": 42.22, "Top-5 (%)": 67.02},
    {
        "Model": "Unified Attention",
        "Top-1 (%)": round(100.0 * float(b4m["top1"]), 2),
        "Top-5 (%)": round(100.0 * float(b4m["top5"]), 2),
    },
]
st.success(
    "**Defense GO** · Protocol A cache-backed · signer-ID disjoint · "
    f"checkpoint `{ckpt[:16]}…`"
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Validation Top-1", "84.81 ± 0.51%")
k2.metric("Test Top-1", f"{100.0 * float(b4m['top1']):.2f}%")
k3.metric("Test Top-5", f"{100.0 * float(b4m['top5']):.2f}%")
k4.metric("Δ vs Simple Combined", f"+{float(b4m['delta_top1_pp_vs_b3_mlp_ctr']):.2f} pp")
st.caption(
    f"Canonical cache-backed n={b4m['test_n']} · Macro-F1={100.0 * float(b4m['macro_f1']):.2f}% · "
    f"source={b4_can.get('source')}"
)

tab_result, tab_input, tab_arch, tab_evidence = st.tabs(
    ["Kết quả", "Đầu vào", "Kiến trúc", "Bằng chứng"]
)

with tab_result:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Validation — mean ba seed**")
        render_test_comparison_chart(
            _VAL_ROWS, x_domain=(75.0, 87.0), x_title="Validation Top-1 mean (%)"
        )
        st.caption("Cùng CTR-matched finite policy; Unified Attention vượt hai đối chứng có cấu trúc.")
    with right:
        st.markdown("**Test — n=6.165, seed 42**")
        render_test_comparison_chart(_TEST_ROWS)
        st.caption("Retrospective signer-disjoint confirmation; không phải pristine blind.")
    st.info(
        "Unified Attention vượt Simple Combined trên validation với CI95 [+2,94; +5,69] pp và trên test "
        "với Δ +9,33 pp, CI95 [+6,37; +11,34] pp."
    )

with tab_input:
    try:
        manifest = load_e4_input_demo_manifest(root=_ROOT)
        errors = validate_e4_input_demo_manifest(manifest, root=_ROOT)
    except Exception as exc:
        st.error(f"Input manifest unavailable: {exc}")
        st.stop()
    if errors:
        st.error("Fail closed — input manifest invalid:\n- " + "\n- ".join(errors[:12]))
        st.stop()
    samples = manifest["samples"]
    labels = {
        str(s["sample_id"]): f"{s['sample_id']} · {s['signer_id']} · {s['gloss']}"
        for s in samples
    }
    sample_id = st.selectbox(
        "Mẫu validation", options=list(labels), format_func=lambda key: labels[key], key="b4_sample_visual"
    )
    sample = next(s for s in samples if str(s["sample_id"]) == sample_id)
    evidence = render_two_stream_input(
        sample_id=sample_id,
        video_path=_ROOT / sample["video_path_rel"],
        rgb_path=_ROOT / sample["rgb_cache_rel"],
        skel_path=_ROOT / sample["skeleton_cache_rel"],
        widget_key="b4_visual",
        model_label="Unified Attention",
    )

with tab_arch:
    render_flow(
        [
            ("Front video", "Một camera; RGB token v3"),
            ("Skeleton v3", "75 khớp + validity mask"),
            ("Spatial + temporal", "Học vùng và thời điểm quan trọng"),
            ("Cross-attention", "Skeleton định hướng RGB"),
            ("Stream gate", "Kết hợp thích nghi hai luồng"),
            ("400 lớp", "Top-1 / Top-5"),
        ]
    )
    st.caption(
        "Unified Attention gồm spatial + temporal + cross-attention + gate. "
        "Không tách claim riêng spatial attention."
    )

with tab_evidence:
    v6 = health.get("v6_kpi") or {}
    st.code(
        "\n".join(
            [
                "engine=Unified Attention (Defense GO)",
                f"checkpoint={ckpt or _CKPT_PREFIX}",
                "optimized_inference_ready=false",
                f"scope={v6.get('scope')}",
                "test_n=6165",
                "mcnemar=1047_vs_472",
                "acceptance=đủ_17_kịch_bản",
            ]
        ),
        language="text",
    )
    if "sample" in locals():
        with st.expander("Input provenance", expanded=False):
            st.code(
                "\n".join(
                    [
                        f"sample_id={sample_id}",
                        f"rgb_sha={sample['rgb_cache_sha256']}",
                        f"skel_sha={sample['skeleton_cache_sha256']}",
                        f"video_sha={sample['video_sha256']}",
                        f"rgb_windows={evidence.get('rgb', {}).get('windows')}",
                        f"skel_frames={evidence.get('skeleton', {}).get('frames')}",
                    ]
                ),
                language="text",
            )
    render_novice_glossary()

st.caption(
    "Số trên trang này đo theo protocol cục bộ, trên đặc trưng đã tính sẵn. "
    "Không xếp hạng SOTA, không tuyên bố thời gian thực hay sản phẩm vận hành, "
    "và không so trực tiếp với bảng độ đúng của bài báo VSL400."
)
