# Chiến lược triển khai MS-STA v6

> **SoT:** `DEFENSE_GO`. Mô hình đề xuất là **Unified Attention** (seed 42). Đối chứng khóa là **Simple Combined** trên cùng đặc trưng RGB và skeleton, cùng encoder. Test là xác nhận signer-ID disjoint, retrospective, không phải pristine blind.

**Unified Attention là mô hình đề xuất khi bảo vệ.** Thử nghiệm nén mạng lịch sử (`E4_ACCURACY_NO_GO`) là phụ lục âm, không triển khai. Đóng góp hệ thống là Optimized inference trên cùng checkpoint.

## Câu hỏi và đối chứng

Với một camera Front-view và người ra dấu không xuất hiện trong tập huấn luyện, Unified Attention có cải thiện so với RGB một luồng và các đối chứng không có chú ý chéo, trên cùng protocol, hay không?

| Cấu hình | Vai trò | Validation Top-1, mean ± sample std |
|---|---|---:|
| Simple Combined | đối chứng khóa, cùng ROI RGB + encoder skeleton; không temporal RGB, không cross-attention, không gate, không residual | **80,54 ± 1,57%** |
| **Unified Attention** | thêm temporal RGB + skeleton hỏi RGB + stream gate + residual có chặn | **84,81 ± 0,51%** |

Seeds 42/123/456: Unified Attention **85,32% / 84,82% / 84,29%**. Hơn Simple Combined **+4,27 điểm phần trăm** trên validation; CI95 **[+2,94; +5,69]**. Trên test signer-ID disjoint, retrospective (n=6.165): **51,55%** so với **42,22%**, chênh **+9,33 điểm phần trăm**, CI95 **[+6,37; +11,34]**, McNemar 1047 vs 472.

## Quy trình khóa

1. D0 tạo cache RGB-token v3 và Skeleton v3, kiểm chất lượng đủ 24.752 mẫu.
2. HPO hữu hạn trên development split; không dùng test để chọn cấu hình.
3. CTR-matched architecture gate; confirmation ba seed; khóa teacher `V6_TEACHER_LOCK.json`.
4. Chỉ sau validation PASS mới chạy test seed 42 một lần trên 6.165 mẫu.
5. D4 kiểm checkpoint, cache, lỗi thiếu dữ liệu và forward smoke; **15/15 PASS**.
6. Active registry chỉ trỏ đúng checkpoint SHA `0b73f8f2...`; sai SHA hoặc thiếu feature thì demo từ chối phục vụ. Rollback: `ms_sta_registry_v5_defense_go.json` (historical).

## Luận điểm hiệu suất tách biệt

E4 (`e4_w192`) là ablation phụ lục, không phải mô hình đề xuất. E4 thất bại gate accuracy (`E4_ACCURACY_NO_GO`): mean Δ Top-1 **−1,90 pp**, CI95 **[−3,05; −0,65] pp**. State-efficiency gate **không chạy**. Không claim nhanh hơn, tương đương accuracy, hay triển khai E4.

## Giới hạn

- Không so trực tiếp accuracy tuyệt đối với Table 3 của paper VSL400 do khác protocol, backbone, split và signer khả dụng.
- Không claim mọi metric tăng trên mọi signer.
- Demo là cache-backed offline, Front-view word-level, có xác nhận của người dùng; không real-time, production, arbitrary-video hay dịch câu liên tục.

Evidence: `results/ms_sta_v6/` · `results/ms_sta_v6_oneshot/` · `results/ms_sta_v6_d4/` · `results/ms_sta_v6_defense_go/FINAL_RESULTS.json`.
