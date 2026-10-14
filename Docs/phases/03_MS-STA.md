# Giai đoạn 3: Mô hình MS-STA — Unified Attention

> **Trạng thái bảo vệ:** đã khóa (`DEFENSE_GO`).  
> **Mô hình đề xuất:** **Unified Attention** (Mô hình Chú ý Hợp nhất).  
> **Baseline đối chứng:** **Simple Combined Baseline** (Mốc Kết hợp Đơn giản).  
> **Optimized inference:** cùng checkpoint, AMP + gom ROI (Protocol B).  
> *Tái lập kỹ thuật:* xem [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) (checkpoint / registry / SHA).  
> Phụ lục âm: thử nghiệm nén mạng lịch sử — **không** triển khai.

## Kiến trúc đề xuất

1. **RGB-token:** video Front chia cửa sổ; SlowFast trích đặc trưng; ROIAlign tạo token toàn cảnh, tay trái, tay phải và mặt.
2. **Skeleton:** 75 khớp (pose, face, hai tay); validity mask, vận tốc.
3. **Spatial attention:** chấm điểm khớp trong mỗi khung (Softmax + validity).
4. **Temporal attention:** quan hệ giữa các bước thời gian.
5. **Cross-attention:** Skeleton = Query; RGB = Key/Value — một chiều.
6. **Stream gate:** trọng số kết hợp hai luồng theo mẫu.
7. **Bounded residual (ReZero):** nhánh mới khởi đầu trung tính.
8. **Classifier:** 400 logits → Softmax → một nhãn.

Sơ đồ: [`../MSSTA_END_TO_END_PIPELINE.md`](../MSSTA_END_TO_END_PIPELINE.md) · [`../MSSTA_DEFENSE_FULL_FLOW_MERMAID.md`](../MSSTA_DEFENSE_FULL_FLOW_MERMAID.md).

## Kết quả khóa

| Scope | Simple Combined | Unified Attention | Δ Top-1 |
|---|---:|---:|---:|
| Validation, ba seed | 80,54 ± 1,57% | **84,81 ± 0,51%** | **+4,27 điểm phần trăm** |
| Test Protocol A, n=6.165 | 42,22% | **51,55%** | **+9,33 điểm phần trăm** |

Validation seeds Unified Attention: 85,32% / 84,82% / 84,29%. CI95 validation **[+2,94; +5,69]**; test **[+6,37; +11,34]**; McNemar 1047 vs 472.

## Kỷ luật claim

- Được claim: Unified Attention vượt Simple Combined trên validation/test đã khóa (cùng encoder).
- Không claim: test pristine blind, SOTA VSL400, dịch câu, thời gian thực.
- Không dùng biến thể nén mạng làm mô hình bảo vệ.
- Optimized inference không thay trọng số Unified Attention.

SoT hội đồng: báo cáo HTML/TeX · Evidence: `../results/ms_sta_v6_defense_go/`.
