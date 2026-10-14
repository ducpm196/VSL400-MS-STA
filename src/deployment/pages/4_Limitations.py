"""Streamlit page: limitations and future continuous recognition/translation."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.deployment.campaign_evidence import runtime_ui_state

st.set_page_config(page_title="Giới hạn và hướng phát triển", layout="wide")
st.title("Giới hạn và hướng phát triển")

st.error("Hệ thống hiện tại chưa dịch câu ngôn ngữ ký hiệu liên tục.")
st.info("Đây là hệ thống nhận dạng ký hiệu đơn lẻ 400 lớp.")

state = runtime_ui_state(_ROOT)
st.markdown(
    f"""
### Phạm vi đã khóa
- 400 lớp ký hiệu đơn lẻ (isolated recognition).
- Đánh giá tách biệt theo mã người thực hiện ký hiệu (`signer_id`).
- Demo chỉ minh họa Protocol A (đặc trưng đã tính sẵn). Engine trên giao diện là **Unified Attention**.
- Optimized inference đã có bằng chứng riêng, nhưng giao diện **không** chạy Optimized inference.
- Phần cứng evidence: RTX 2060 + stack đã khóa — không khái quát mọi GPU.

### Không thuộc đồ án hiện tại
- Continuous Sign Language Recognition (video câu → chuỗi gloss).
- Sign Language Translation (video câu → câu tiếng Việt).
- Production real-time serving công cộng.

### Hướng tương lai (không tuyên bố đã làm)
1. Giữ embedding temporal Unified Attention làm encoder nền tảng.
2. Thu thập/chuẩn hóa dữ liệu câu.
3. CSLR → SLT trên encoder Unified Attention.
4. Chỉ khi có đường suy luận online thật (ROI trên video) mới xét chạy Optimized inference trên giao diện.
"""
)
st.caption(
    f"{state['s5_label_vi']}. "
    + (
        "Giao diện đang chạy Optimized inference."
        if state["demo_cutover_authorized"]
        else "Giao diện chưa chạy Optimized inference. Demo vẫn đọc đặc trưng đã tính sẵn."
    )
)
