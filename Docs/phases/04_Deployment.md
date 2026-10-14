# Giai đoạn 4: Đóng gói suy diễn & demo — Unified Attention

> **Trạng thái bảo vệ:** đã khóa.  
> **Mô hình đề xuất:** **Unified Attention** (Mô hình Chú ý Hợp nhất).  
> Registry active: `data/processed/ms_sta_registry_active.json`.  
> *Tái lập kỹ thuật:* checkpoint seed 42 · SHA pretest freeze trong registry — chi tiết [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).  
> Thử nghiệm nén mạng **không** load trên demo.

**Unified Attention là mô hình đề xuất khi bảo vệ.** Nghiên cứu nén/thu nhỏ mạng là phụ lục âm — không triển khai.

## Tư thế demo (hội đồng)

- Cache-backed offline (Protocol A) trên đặc trưng đã tính sẵn.
- Fail-closed: sai SHA, thiếu NPZ, clip hỏng, backend lỗi.
- Test evidence = xác nhận signer-ID disjoint, **retrospective** (không pristine blind).
- Streamlit minh họa Protocol A; Optimized inference chỉ bằng chứng (không cutover live).

Acceptance mô hình: `../results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json` ·  
Gói bảo vệ: `../results/ms_sta_v6_defense_go/`.

Registry lịch sử / rollback chỉ phục vụ provenance — không phải nguồn khóa số liệu hội đồng.
