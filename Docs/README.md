# Tài liệu kỹ thuật

Bản công khai không kèm báo cáo đồ án, slide bảo vệ và thư mục bằng chứng `results/` vì các tệp này chứa dữ liệu hoặc hình ảnh người ký thuộc bộ VSL400 (xem mục "Dữ liệu" trong [README](../README.md)). Một số tài liệu dưới đây vẫn nhắc tới đường dẫn trong `results/`; đó là vị trí bằng chứng trong kho private gốc.

| Tài liệu | Nội dung |
|---|---|
| [`TONG_QUAN_DO_AN.md`](TONG_QUAN_DO_AN.md) | Tổng quan đồ án: bài toán, kiến trúc, kết quả, vận hành |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Các bước tái lập từ dữ liệu đến kết quả |
| [`MSSTA_END_TO_END_PIPELINE.md`](MSSTA_END_TO_END_PIPELINE.md) | Luồng xử lý từ video đến Top-k |
| [`MSSTA_DEFENSE_FULL_FLOW_MERMAID.md`](MSSTA_DEFENSE_FULL_FLOW_MERMAID.md) | Sơ đồ luồng đầy đủ |
| [`B4_CURRENT_REPORT.md`](B4_CURRENT_REPORT.md) | Bảng số liệu đã khóa |
| [`Research_Overview.md`](Research_Overview.md) | Câu hỏi nghiên cứu và kết quả |
| [`Strategy_Implementation.md`](Strategy_Implementation.md) | Chiến lược triển khai |
| [`phases/`](phases/) | Ghi chú theo giai đoạn: dữ liệu, mạng nền, mô hình, triển khai, đánh giá |
| [`reviews/`](reviews/) | Rà soát kiến trúc và các quyết định nghiên cứu |
| [`references/`](references/) | Nguồn dữ liệu VSL400 và tài liệu tham khảo |
| [`NHAT_KY_DO_AN.md`](NHAT_KY_DO_AN.md) | Nhật ký quá trình làm đồ án |

Quy ước tên gọi:

| Tên trong báo cáo | Tên trong mã | Ý nghĩa |
|---|---|---|
| Mô hình Chú ý Hợp nhất (Unified Attention) | `b4_v6_ctr` | Mô hình đề xuất |
| Mốc Kết hợp Đơn giản (Simple Combined Baseline) | `b3_mlp_ctr` | Đối chứng chính, cùng bộ mã hóa |
| Ghép đặc trưng CTR | `b3_ctr_concat` | Đường nền của mô hình đề xuất |
| Chỉ RGB | `rgb_strong` | Đối chứng chỉ dùng ảnh |
| Suy luận cơ sở / Suy luận tối ưu | R0 / S5 | Hai cách chạy trên cùng bộ trọng số |
