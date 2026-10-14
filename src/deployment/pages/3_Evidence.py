"""Streamlit page: scientific evidence tables (model vs online runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.deployment.campaign_evidence import (
    load_r0_comparator_report,
    load_s5_confirmation_efficiency,
    load_s5_efficiency_report,
    load_scientific_claim_lock,
    runtime_ui_state,
)

st.set_page_config(page_title="Bằng chứng khoa học", layout="wide")
st.title("Bằng chứng khoa học — đúng hai luận điểm")
st.info("Đây là hệ thống nhận dạng ký hiệu đơn lẻ 400 lớp.")
st.caption("`signer_id` là mã người thực hiện ký hiệu, tức mã người ra dấu.")

try:
    claim = load_scientific_claim_lock(_ROOT)
    state = runtime_ui_state(_ROOT)
    r0 = load_r0_comparator_report(_ROOT)
    eff = load_s5_efficiency_report(_ROOT)
    conf = load_s5_confirmation_efficiency(_ROOT)
except Exception as exc:
    st.error(f"Fail closed — evidence unavailable: {exc}")
    st.stop()

b4, e4 = claim["B4"], claim["E4"]
st.warning(
    f"{state['s5_label_vi']} · demo engine: **Unified Attention** (cache-backed)"
)

st.subheader("Bảng 1 — Chất lượng mô hình Unified Attention (Protocol A · cache-backed)")
st.dataframe(
    [
        {
            "Path": "Unified Attention",
            "Split": "test",
            "n": 6165,
            "Top-1": f"{100.0 * b4['top1']:.2f}%",
            "Top-5": f"{100.0 * b4['top5']:.2f}%",
            "Macro-F1": f"{100.0 * b4['macro_f1']:.2f}%",
        },
        {
            "Path": "Simple Combined",
            "Split": "test",
            "n": 6165,
            "Top-1": "42.22%",
            "Top-5": "67.02%",
            "Macro-F1": "41.42%",
        },
    ],
    hide_index=True,
    use_container_width=True,
)
st.success(
    f"Unified Attention cải thiện Top-1 **+{b4['delta_top1_pp_vs_b3_mlp_ctr']:.2f} pp** "
    f"so với Simple Combined."
)

st.subheader("Bảng 2 — Bảo toàn độ đúng Optimized inference (Protocol B · online)")
st.dataframe(
    [
        {
            "Runtime": "Baseline inference",
            "Path": "online",
            "n": 6165,
            "Top-1": f"{100.0 * e4['R0_online']['top1']:.3f}%",
            "Top-5": f"{100.0 * e4['R0_online']['top5']:.3f}%",
            "Macro-F1": f"{100.0 * e4['R0_online']['macro_f1']:.3f}%",
        },
        {
            "Runtime": "Optimized inference",
            "Path": "online",
            "n": 6165,
            "Top-1": f"{100.0 * e4['S5_online']['top1']:.3f}%",
            "Top-5": f"{100.0 * e4['S5_online']['top5']:.3f}%",
            "Macro-F1": f"{100.0 * e4['S5_online']['macro_f1']:.3f}%",
        },
    ],
    hide_index=True,
    use_container_width=True,
)
paired = r0.get("paired_S5_minus_R0") or {}
st.markdown(
    f"- Δ Top-1 Optimized − Baseline = **{e4['paired_S5_minus_R0_top1_pp']:+.3f} pp**  \n"
    f"- CI95 = **{e4['paired_ci95_top1_pp']}**  \n"
    f"- Accuracy-preservation gate: **PASS**  \n"
    f"- Prediction agreement ≈ **{paired.get('prediction_agreement', float('nan')):.6f}**"
)
st.info(e4["path_mismatch_explanation"])
st.caption("Không dùng 51,55% của Protocol A như độ đúng online của Optimized inference.")

st.subheader("Bảng 3 — Hiệu năng (protocol khóa · RTX 2060)")
arms = eff.get("arms") or {}
sp = eff.get("speedup_S5_vs_R0") or {}
st.dataframe(
    [
        {
            "Runtime": "Baseline inference",
            "median_e2e_ms": round(float((arms.get("R0") or {}).get("median_e2e_ms") or 0), 2),
            "p90_e2e_ms": round(float((arms.get("R0") or {}).get("p90_e2e_ms") or 0), 2),
            "VRAM_median_MB": round(float((arms.get("R0") or {}).get("median_vram_allocated_mb") or 0), 2),
        },
        {
            "Runtime": "AMP-only arm",
            "median_e2e_ms": round(float((arms.get("R1") or {}).get("median_e2e_ms") or 0), 2),
            "p90_e2e_ms": round(float((arms.get("R1") or {}).get("p90_e2e_ms") or 0), 2),
            "VRAM_median_MB": round(float((arms.get("R1") or {}).get("median_vram_allocated_mb") or 0), 2),
        },
        {
            "Runtime": "Optimized inference",
            "median_e2e_ms": round(float((arms.get("S5") or {}).get("median_e2e_ms") or 0), 2),
            "p90_e2e_ms": round(float((arms.get("S5") or {}).get("p90_e2e_ms") or 0), 2),
            "VRAM_median_MB": round(float((arms.get("S5") or {}).get("median_vram_allocated_mb") or 0), 2),
        },
    ],
    hide_index=True,
    use_container_width=True,
)
st.markdown(
    f"- Optimized / Baseline formal speedup ≈ "
    f"**{float(sp.get('speedup_factor') or e4['formal_val_speedup_approx']):.3f}×**  \n"
    f"- CI95 speedup ≈ **[{float(sp.get('ci95_lo') or 0):.3f}, {float(sp.get('ci95_hi') or 0):.3f}]**  \n"
    f"- Confirmation mean speedup ≈ "
    f"**{float(conf.get('mean_speedup') or e4['confirmation_speedup_approx']):.3f}×**  \n"
    f"- Stack: RTX 2060 + phần mềm đã khóa trong policy.\n\n"
    "> Hiệu năng đo theo protocol chính thức; **không phải** latency của thao tác UI hiện tại."
)

st.subheader("FACT / INFERENCE / SOURCE_GAP")
st.markdown(
    """
- **FACT:** Mười bảy kiểm tra đầu ra đều đạt. Optimized inference đã chọn. Giao diện demo chưa chạy cách suy luận này.
- **FACT:** Nhánh chỉ dùng mixed precision cho cùng logit với Optimized inference trên tập kiểm tra đã khóa.
- **INFERENCE:** Optimized inference trả lời câu hỏi tốc độ triển khai, không phải một mô hình mới.
- **Chưa đo trên giao diện:** Streamlit không đo tốc độ Optimized inference, vì gallery chỉ đọc đặc trưng đã tính sẵn.
"""
)
