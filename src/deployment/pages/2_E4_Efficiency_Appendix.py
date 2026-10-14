"""Streamlit page: Optimized inference (Protocol B) — evidence-only until live switch."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.deployment.campaign_evidence import (
    load_locked_test_report,
    load_r0_comparator_report,
    load_scientific_claim_lock,
    runtime_ui_state,
)
from src.deployment.demo_visuals import render_flow

st.set_page_config(page_title="Optimized inference", layout="wide")
st.title("Luận điểm 2 — Optimized inference: gom ROI + mixed precision")

st.info(
    "Đây là hệ thống nhận dạng ký hiệu đơn lẻ, chưa phải hệ thống dịch câu ngôn ngữ ký hiệu liên tục."
)

try:
    claim = load_scientific_claim_lock(_ROOT)
    state = runtime_ui_state(_ROOT)
    r0rep = load_r0_comparator_report(_ROOT)
    lt = load_locked_test_report(_ROOT)
except Exception as exc:
    st.error(f"Fail closed — không tải được bằng chứng Optimized inference: {exc}")
    st.stop()

st.warning(
    f"**{state['s5_label_vi']}** · engine UI đang hoạt động: **Unified Attention** (cache-backed)"
)
st.caption(
    "Capability: `SEALED_DEMO_SAMPLE_ONLY`. Gallery cache-backed **không** đo ROI batching live; "
    "speedup chính thức lấy từ evidence RTX 2060 đã khóa."
)

e4 = claim["E4"]
b4 = claim["B4"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Optimized · online Top-1", f"{100.0 * e4['S5_online']['top1']:.3f}%")
k2.metric("Baseline · online Top-1", f"{100.0 * e4['R0_online']['top1']:.3f}%")
k3.metric("Optimized − Baseline Δ", f"{e4['paired_S5_minus_R0_top1_pp']:+.3f} pp")
k4.metric("Acceptance", "PASS" if state.get("d4_pass") else "xem evidence")

tab_story, tab_acc, tab_eff, tab_neg = st.tabs(
    ["Câu chuyện hệ thống", "Accuracy online", "Hiệu năng", "Kết quả âm"]
)

with tab_story:
    st.markdown(
        """
**Optimized inference không phải mô hình phân loại mới.** Cùng checkpoint Unified Attention; chỉ đổi đường thực thi:

1. Baseline inference — SlowFast FP32 online
2. AMP-only — mixed precision
3. Optimized inference — ROIAlign batching (`global, left_hand, right_hand, face`)
4. Cùng head → logits Optimized ≡ AMP-only (parity đã khóa)
"""
    )
    render_flow(
        [
            ("Canonical windows", "Online decode / features"),
            ("Per-region ROIAlign", "~480 calls / clip"),
            ("ROI batching", "Order: global→L→R→face"),
            ("Same token order", "No semantic change"),
            ("Same checkpoint", "Exact SHA load"),
            ("Same predictions", "Lower latency · lower VRAM"),
        ]
    )
    st.info(e4["path_mismatch_explanation"])

with tab_acc:
    st.markdown("### Bảng tách path (bắt buộc)")
    rows = [
        {
            "Path": "Unified Attention · cache-backed",
            "Role": "model evaluation",
            "n": b4["test_n"],
            "Top-1 %": round(100.0 * b4["top1"], 3),
            "Top-5 %": round(100.0 * b4["top5"], 3),
            "Macro-F1 %": round(100.0 * b4["macro_f1"], 3),
        },
        {
            "Path": "Baseline inference · online",
            "Role": "system baseline",
            "n": 6165,
            "Top-1 %": round(100.0 * e4["R0_online"]["top1"], 3),
            "Top-5 %": round(100.0 * e4["R0_online"]["top5"], 3),
            "Macro-F1 %": round(100.0 * e4["R0_online"]["macro_f1"], 3),
        },
        {
            "Path": "Optimized inference · online",
            "Role": "optimized runtime",
            "n": 6165,
            "Top-1 %": round(100.0 * e4["S5_online"]['top1'], 3),
            "Top-5 %": round(100.0 * e4["S5_online"]["top5"], 3),
            "Macro-F1 %": round(100.0 * e4["S5_online"]["macro_f1"], 3),
        },
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    paired = r0rep.get("paired_S5_minus_R0") or {}
    st.markdown(
        f"- Paired Top-1 Δ: **{paired.get('paired_top1_delta_pp', e4['paired_S5_minus_R0_top1_pp']):+.4f} pp**  \n"
        f"- CI95: **{paired.get('hierarchical_ci95_top1_pp', e4['paired_ci95_top1_pp'])}**  \n"
        f"- Prediction agreement: **{paired.get('prediction_agreement', float('nan')):.6f}**  \n"
        f"- Accuracy gate: **{'PASS' if r0rep.get('accuracy_gate_pass') else 'FAIL'}**  \n"
        f"- AMP ≡ Optimized locked-test logit parity: **{lt.get('EXACT_LOGIT_PARITY')}**"
    )
    st.caption("Không xếp 51,55% của Protocol A cùng hàng với khoảng 49,94% khi suy luận trực tiếp từ video.")

with tab_eff:
    st.markdown(
        f"""
### Hiệu năng chính thức (Protocol B · RTX 2060)

- Validation formal speedup Optimized / Baseline ≈ **{e4['formal_val_speedup_approx']:.3f}×**
- Confirmation mean speedup ≈ **{e4['confirmation_speedup_approx']:.3f}×**
- VRAM Optimized ≤ Baseline (gate đã khóa)

> Hiệu năng đo theo protocol chính thức trên RTX 2060; **không phải** latency của thao tác UI hiện tại.
"""
    )

with tab_neg:
    st.markdown(
        """
| Arm | Kết quả |
|-----|---------|
| Ghim bộ nhớ khi chuyển dữ liệu | Không đạt cổng tốc độ |
| Chồng thời gian giải mã | Không an toàn về nghĩa của khung |
| Chạy song song hai luồng | Không có việc song song đáng kể |
| Gom ROI | Đạt, đưa vào Optimized inference |
| Mạng học sinh nén đặc | Không đủ phần cứng, không huấn luyện |
"""
    )
    k1 = claim.get("K1") or {}
    st.caption(f"Negative-study status: `{k1.get('status')}` · trained={k1.get('trained')}")

st.markdown("---")
st.caption(
    f"{state['s5_label_vi']} · mode={state['s5_mode']} · "
    f"demo_live_authorized={state['demo_cutover_authorized']}"
)
