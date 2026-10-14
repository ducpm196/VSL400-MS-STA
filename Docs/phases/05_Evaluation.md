# Giai đoạn 5: Kiểm thử và đóng gói — nguồn khóa số liệu

> **Trạng thái bảo vệ:** đã khóa (`DEFENSE_GO`).  
> **Mô hình chính:** **Unified Attention** (Mô hình Chú ý Hợp nhất).  
> **Baseline:** **Simple Combined Baseline**.  
> **Hệ thống:** **Optimized inference** trên cùng checkpoint (Protocol B).  
> **Validation:** CTR-matched, nhiều seed.  
> **Test Protocol A:** signer-ID disjoint, retrospective (n=6.165) — **không** phải pristine blind.  
> *Evidence đã khóa:* `../results/e4_system_kd_v1/` · `../results/ms_sta_v6_oneshot/` · [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

**Unified Attention là mô hình đề xuất khi bảo vệ.** Thử nghiệm nén mạng lịch sử là phụ lục âm — không nhầm với Optimized inference (cùng checkpoint).

## Cách đọc bằng chứng

1. Bảng validation và bảng test **tách riêng** — không trộn số.
2. Đối chứng khóa kiến trúc: **Simple Combined**, không phải Concat thuần.
3. Nghiên cứu nén mạng lịch sử chỉ phụ lục NO-GO — không deploy.
4. Bản MS-STA cũ / retrain lịch sử = `historical_not_primary` hoặc rollback.

## Bảng A — Validation (3 seed)

| Cấu hình | Top-1 | Ghi chú |
|---|---:|---|
| Simple Combined | 80,54 ± 1,57% | Đối chứng mạnh |
| **Unified Attention** | **84,81 ± 0,51%** | Δ **+4,27 điểm phần trăm** · CI95 **[+2,94; +5,69]** |
| Seed Unified Attention | 85,32 / 84,82 / 84,29% | 42 / 123 / 456 |

## Bảng B — Test Protocol A (n=6.165, seed 42)

| Cấu hình | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| RGB-only | 29,80% | — | — |
| Concat RGB+Skeleton | 46,62% | — | — |
| Simple Combined | 42,22% | — | — |
| **Unified Attention** | **51,55%** | **73,88%** | **51,37%** |

Δ Unified Attention − Simple Combined: Top-1 **+9,33 điểm phần trăm** · CI95 **[+6,37; +11,34]** · McNemar **1047 vs 472**.

## Phụ lục — nghiên cứu nén mạng (NO-GO)

Các chiến dịch thu nhỏ/nén không đạt cổng độ đúng đồng thời với tăng tốc dưới ràng buộc giữ mạng đã khóa → **không** dùng làm kết quả chính (xem JSON `*_NO_GO` trong `Docs/results/`). Không nhập số phụ lục vào kết luận hội đồng.
