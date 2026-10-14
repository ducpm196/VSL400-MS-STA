# Rà soát bản bàn giao — 29/09/2026

## Kết quả kiểm tra

- CI trên checkout sau sửa: **744 passed, 5 skipped, 10 deselected**, dùng `_venvs/ci` và `-m "not requires_artifacts"`. Hai thư viện kiểm tra PPTX/PDF trước đây thiếu đã được thêm vào `requirements-ci.txt`; `pip check` đạt.
- Bộ test tập trung cho UI, manifest đầu vào, nội dung đã khóa, split và đóng gói: **62 passed** ở lần chạy sau sửa. Số test đóng gói có thể tăng khi bổ sung trường hợp kiểm tra.
- Gói demo thử nghiệm đã giải nén: đúng SHA registry, checkpoint, metadata và sáu mẫu validation; suy luận CPU thành công cho `023292`, `018613`; Top-5 giữ yêu cầu người xác nhận; clip không tồn tại trả 404. AppTest chạy được trang chính và bốn trang phụ.
- Kiểm tra độc lập lặp lại hash và suy luận, chặn kết nối ngoài máy: không có yêu cầu mạng ngoài. Đây là kiểm tra runtime dùng môi trường demo hiện có, không phải cài môi trường mới từ Internet.
- Launcher thật trên Windows PowerShell: API `/health` và Streamlit `/_stcore/health` đạt; quá trình kiểm tra đã dừng các tiến trình do nó tạo. Kiểm tra này không thay thế buổi diễn tập bằng trình duyệt với người trình bày.

Log và hash của **gói cuối cùng** nằm cạnh ZIP trong thư mục bàn giao `verification/` và `SHA256SUMS.txt`; kiểm tra sau đóng gói quyết định trạng thái của từng ZIP.

## Thay đổi có căn cứ

1. `app.py` chỉ gọi `main()` khi là entry point: trước đây import hàm tiện ích cũng chạy UI và gọi API, gây lỗi khi server tắt.
2. Launcher kiểm tra môi trường, FFmpeg, tính toàn vẹn gói, registry và checkpoint; dùng backbone đóng gói cục bộ; kiểm tra cổng bận và quản lý đúng PID API của phiên hiện tại. Bỏ việc dừng mọi Python có tên lệnh tương tự. SHA tính bằng .NET để không phụ thuộc khả năng nạp cmdlet `Get-FileHash`.
3. `.gitattributes` giữ nguyên byte cho metadata và JSON đã khóa. Đã tái hiện `git archive` đổi LF thành CRLF làm sai SHA split; test nay kiểm tra các thuộc tính làm việc hiện hành. Không sửa nội dung split/checkpoint hoặc tính lại số liệu khoa học.
4. Các test giao diện cũ yêu cầu trang E4-R2/R3 dù nguồn khóa khoa học hiện hành chọn Optimized inference S5. Cập nhật kiểm tra theo hợp đồng hiện hành: cùng mô hình, B4 vẫn chạy cache, S5 chỉ là bằng chứng. Dữ liệu nghiên cứu âm vẫn giữ nguyên.
5. Builder tạo ZIP mới, kiểm tra SHA từng tệp sau nén; verifier kiểm tra tệp giải nén và có chế độ chạy suy luận/UI thật. Gói source giữ mã nghiên cứu và hồ sơ nguồn số liệu; gói demo chỉ chọn phụ thuộc chạy cần thiết. Không kèm `.env`, virtualenv, toolkit agent hoặc thư mục tạm.
6. `.env.example` bỏ trạng thái NO-GO cũ và token giả. Hướng dẫn bàn giao phân biệt rõ baseline, Unified Attention, Protocol A và Protocol B.

## Giới hạn và các tài sản được giữ

Không huấn luyện lại, không mở lại tập test khoa học, không sửa registry/claim lock và không thay checkpoint. Môi trường kiểm tra demo là CPU; số liệu GPU trong đồ án vẫn là bằng chứng lịch sử, chưa đo lại trong lần này.

Các test GPU, MediaPipe, công cụ PDF→DOCX tùy chọn hoặc cần artifact riêng không được tính là đã chạy. Hồ sơ nghiên cứu, checkpoint, corpus và bản sao có thể chứa dữ liệu duy nhất được giữ ở repo gốc. Chỉ xóa cache có thể sinh lại và bản ZIP toolkit không thuộc đồ án; danh sách xóa được ghi trong báo cáo cleanup. Không dùng `git clean -fdx`.

Manifest SHA-256 phát hiện thất lạc/hỏng tệp; không phải chữ ký số. Kiểm tra tên tệp và mẫu credential thông dụng không chứng minh mọi nội dung đều không thể chứa bí mật. Không công bố gói video ra Internet; sử dụng theo điều kiện truy cập VSL400.

Gói demo chứa tám clip, không chứa toàn bộ corpus. Chọn clip thủ công ngoài tám mẫu sẽ bị từ chối vì thiếu tài sản. Cài mới thư viện và thử trên phần cứng hội đồng vẫn cần làm trước buổi bảo vệ. Xem [hướng dẫn bàn giao](SUBMISSION.md).
