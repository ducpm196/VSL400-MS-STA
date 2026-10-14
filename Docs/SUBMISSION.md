# Bàn giao mã nguồn và demo bảo vệ

## Hai gói độc lập

- **VSL_SOURCE.zip**: mã nguồn, test, script nghiên cứu, cấu hình, hồ sơ bằng chứng và báo cáo tổng quan `Docs/TONG_QUAN_DO_AN.md`. Không chứa môi trường Python, khóa API, toàn bộ dữ liệu hay checkpoint lớn. `Docs/results/` và các nghiên cứu NO-GO được giữ để kiểm tra nguồn số liệu; không phải các mô hình chạy demo.
- **VSL_DEMO.zip**: mã chạy, checkpoint B4-V6 và B0, backbone SlowFast, hai clip demo cùng sáu clip validation minh họa, feature cache tương ứng, bằng chứng cho các trang giao diện và báo cáo tổng quan `Docs/TONG_QUAN_DO_AN.md`. Không cần tải trọng số hay gọi Gemini khi demo.

Mỗi ZIP có `SUBMISSION_MANIFEST.json`: đường dẫn, kích thước và SHA-256 từng tệp. Giữ nguyên cấu trúc khi giải nén. Không ghi đè gói phát hành cũ; tạo tên hoặc thư mục mới khi đóng gói lại.

## Chạy demo trên Windows

Máy đã kiểm tra: Python 3.11, CPU. `requirements-demo-lock.txt` ghi chính xác các phiên bản đang cài trên môi trường này; không phải cam kết hiệu năng RTX 2060 hay bản cài offline hoàn toàn. Cài thư viện lần đầu cần Internet/PyPI; sau khi cài xong demo dùng tài sản cục bộ. Cần FFmpeg và FFprobe trên PATH để phát video trong trình duyệt.

Tại thư mục giải nén **VSL_DEMO.zip**:

```powershell
py -3.11 -m venv env_demo
.\env_demo\Scripts\python.exe -m pip install -r requirements-demo-lock.txt
.\env_demo\Scripts\python.exe -m pip check
.\env_demo\Scripts\python.exe scripts/verify_submission.py --smoke --ui --output logs/verification.json
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\demo_vsl400.ps1
```

Mở `http://localhost:8501`. Launcher kiểm tra hash, registry, checkpoint, công cụ video và health của API. Nếu cổng 8000/8501 đang bận, đóng phiên demo trước rồi chạy lại. Launcher chỉ dừng API do chính phiên đó tạo khi kết thúc.

Chọn hai kịch bản có sẵn `023292` và `018613`; nhấn **Run recognition**, xem Top-5 và xác nhận/sửa kết quả. Nhãn kịch bản là minh họa, không phải KPI. Tab đầu vào của trang Accuracy có sáu mẫu validation đã khóa SHA. Gói gọn chỉ mang tám clip này: lựa chọn thủ công những clip khác trong metadata sẽ thiếu video/cache và bị từ chối. Không sửa `metadata.csv` để thu gọn vì manifest khóa hash toàn bộ tệp.

Demo thể hiện **Protocol A: Unified Attention đọc feature cache**. Các trang Optimized inference trình bày bằng chứng Protocol B đã lưu; không đo tốc độ online hay chạy mô hình nén. Hệ thống nhận dạng ký hiệu đơn lẻ 400 lớp, luôn cần người xác nhận; chưa dịch câu liên tục.

## Mã nguồn và tái lập

Các phần chính: `src/data_prep/` tiền xử lý; `src/models/` kiến trúc; `src/training/` huấn luyện; `src/optimization/` các nghiên cứu tối ưu; `src/deployment/` API/giao diện; `tests/` kiểm thử; `scripts/` vận hành và thí nghiệm. Xem `README.md`, `scripts/OPS.md` và `Docs/REPRODUCIBILITY.md` trong gói source.

Môi trường CI, extract, train và demo tách biệt. Không chạy toàn bộ test bằng môi trường demo rồi coi thư viện extract thiếu là lỗi ứng dụng:

```powershell
py -3.11 -m venv env_ci
.\env_ci\Scripts\python.exe -m pip install -r requirements-ci.txt
.\env_ci\Scripts\python.exe -m pip check
.\env_ci\Scripts\python.exe -m pytest -m "not requires_artifacts"
```

Test gắn `requires_artifacts` cần dữ liệu/checkpoint/thư viện riêng được mô tả trong test; không được coi là đã chạy chỉ vì CI đạt. Gói source không chứa đầy đủ corpus hoặc mọi checkpoint thí nghiệm. Giữ bản nghiên cứu gốc để tái lập toàn bộ; dữ liệu VSL400 sử dụng theo điều kiện truy cập/DUA của bộ dữ liệu, không đăng công khai gói demo có video.

## Đóng gói lại từ repo gốc

```powershell
python scripts/package_submission.py --profile source --output dist/submission/VSL_SOURCE.zip
python scripts/package_submission.py --profile demo --output dist/submission/VSL_DEMO.zip
```

Builder lấy tệp source đang được Git theo dõi và danh sách tệp bàn giao mới được khai báo; giữ nguyên byte trên ổ đĩa. Demo lần đầu lấy backbone từ cache Torch hiện có, hoặc truyền `--backbone PATH`. Không tải dữ liệu/trọng số trong quá trình đóng gói. Luôn giải nén sang thư mục mới, chạy `verify_submission.py`, rồi thử demo trước buổi bảo vệ. Không coi `git archive HEAD` là thay thế cho gói demo hoặc bản làm việc chưa commit.
