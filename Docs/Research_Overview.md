# Tổng quan nghiên cứu — MS-STA trên VSL400

> **Trạng thái:** **Unified Attention** (Mô hình Chú ý Hợp nhất) = **`DEFENSE_GO`**. Validation đối chứng cùng encoder với **Simple Combined**; test = xác nhận signer-ID disjoint, retrospective (n=6.165).

> **Hai đóng góp đang dùng:** (1) Unified Attention; (2) Optimized inference trên **cùng checkpoint** (mixed precision + gom ROI). Bằng chứng hệ thống đã đủ; demo chưa chuyển live.

**Tên trong mã nguồn:** Unified Attention = `b4_v6_ctr` · Simple Combined Baseline = `b3_mlp_ctr` (B3+MLP) · Ghép đặc trưng CTR = `b3_ctr_concat` · Chỉ RGB = `rgb_strong`.

> **Phụ lục âm:** thử nghiệm nén mạng = `E4_ACCURACY_NO_GO` — không triển khai làm mô hình bảo vệ.

> Nguồn khóa: [`phases/05_Evaluation.md`](phases/05_Evaluation.md) · Bằng chứng mô hình: `results/ms_sta_v6_defense_go/` · Bằng chứng hệ thống: `results/e4_system_kd_v1/` · ID checkpoint: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
> Sơ đồ bảo vệ: [`MSSTA_DEFENSE_FULL_FLOW_MERMAID.md`](MSSTA_DEFENSE_FULL_FLOW_MERMAID.md).

**Câu hỏi nghiên cứu:** với video chính diện và protocol signer-ID disjoint cục bộ, Unified Attention (RGB theo thời gian + skeleton có cấu trúc, cùng encoder) có tốt hơn RGB một luồng, nối đặc trưng và Simple Combined cùng đầu vào không?

**Kết quả:** Unified Attention thắng ổn định trên validation nhiều seed (84,81 ± 0,51% so với Simple Combined 80,54 ± 1,57%) và được xác nhận trên test retrospective (51,55% Top-1, Δ +9,33 điểm phần trăm). Không gọi test là pristine blind.

**Unified Attention là mô hình đề xuất khi bảo vệ.** Thí nghiệm nén mạng lịch sử là nghiên cứu hiệu năng âm riêng (`E4_ACCURACY_NO_GO`) và không được triển khai. **Đóng góp hệ thống** là Optimized inference trên cùng checkpoint.

**Phụ lục nén:** trung bình Δ Top-1 **−1,90 điểm phần trăm**, CI95 **[−3,05; −0,65]** — `E4_ACCURACY_NO_GO`. Đợt sau chỉnh siêu tham số: trung bình Δ **−3,69 điểm phần trăm**, CI95 **[−5,02; −2,32]** — không mở làm mô hình bảo vệ.

**Bản MS-STA cũ / retrain lịch sử:** rollback hoặc `historical_not_primary`.

**Số liệu paper VSL400 76–93%:** protocol khác — không đối chứng trực tiếp. Xem [`references/VSL400_SOURCE_AUDIT.md`](references/VSL400_SOURCE_AUDIT.md).
