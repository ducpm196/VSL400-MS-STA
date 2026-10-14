# VSL400 MS-STA: nhận dạng ngôn ngữ ký hiệu tiếng Việt cấp từ đơn

Mã nguồn đồ án tốt nghiệp *Nghiên cứu cải tiến mô-đun chú ý không gian - thời gian đa luồng trong nhận diện ngôn ngữ ký hiệu tiếng Việt từ video* (Trường Đại học Công nghệ Thông tin, ĐHQG-HCM, 2026). Sinh viên thực hiện: Phạm Minh Đức, Nguyễn Thị Kiều Thanh. Giảng viên hướng dẫn: ThS. Trần Hoàng Lộc.

*Source code of a bachelor thesis on isolated Vietnamese Sign Language recognition on VSL400. The dataset and every file derived from it are not included; see "Data" below.*

## Nội dung đồ án

Đầu vào là một đoạn video quay chính diện chứa một ký hiệu; đầu ra là một trong 400 từ kèm 5 gợi ý. Mô hình đề xuất, **Mô hình Chú ý Hợp nhất** (Unified Attention), dùng chuỗi khung xương 75 điểm làm truy vấn chú ý chéo lên đặc trưng của bốn vùng ảnh (toàn cảnh, hai tay, mặt) lấy từ mạng SlowFast giữ nguyên trọng số, rồi hợp nhất qua cổng và nhánh dư ReZero. Đồ án còn tối ưu cách chạy suy luận (độ chính xác hỗn hợp và gom vùng ảnh) mà không huấn luyện lại.

Kết quả trên bản chia theo người của đồ án (tập kiểm tra 5 người, 6.165 clip, đánh giá hồi cứu):

| | Top-1 | Ghi chú |
|---|---|---|
| Mô hình Chú ý Hợp nhất | 51,55% | Top-5 73,88%, F1 vĩ mô 51,37% |
| Mốc Kết hợp Đơn giản (đối chứng chính) | 42,22% | chênh +9,33 điểm, CI95 [+6,37; +11,34] |
| Tập kiểm định (3 lần khởi tạo) | 84,81 ± 0,51% | đối chứng 80,54 ± 1,57% |
| Suy luận tối ưu | nhanh hơn 1,220 lần | RTX 2060, độ chính xác giữ nguyên trong ngưỡng |

Các số này không so trực tiếp được với bài báo VSL400 vì khác bản chia, số người và đường tiền xử lý.

## Dữ liệu

Kho này **không** chứa dữ liệu VSL400 hay bất kỳ tệp nào dẫn xuất từ dữ liệu: không video, không metadata, không bản chia tập, không đặc trưng đã trích, không bộ trọng số, không dự đoán theo từng clip, không khung hình người ký. VSL400 là bộ dữ liệu truy cập có kiểm soát; Data Usage Agreement của bộ dữ liệu ([Zenodo 21366957](https://zenodo.org/records/21366957)) không cho phép đưa các thành phần này lên kho mã công khai. Báo cáo đồ án có hình ảnh người ký nên cũng không đăng ở đây.

Muốn chạy lại thí nghiệm, cần tự xin quyền truy cập VSL400 qua [bản ghi Zenodo 17943574](https://zenodo.org/records/17943574) và ký Data Usage Agreement, sau đó trích đặc trưng và huấn luyện lại theo [`Docs/REPRODUCIBILITY.md`](Docs/REPRODUCIBILITY.md). Một số đường dẫn trong tài liệu trỏ tới thư mục bằng chứng `Docs/results/` của kho private gốc; thư mục này không có trong bản công khai.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/data_prep/` | Trích khung xương (MediaPipe), chuẩn hóa, dựng metadata |
| `src/models/` | Mô hình đề xuất và các cấu hình đối chứng |
| `src/training/` | Huấn luyện và đánh giá trên kho đặc trưng |
| `src/optimization/` | Các nghiên cứu tối ưu suy luận |
| `src/eval/` | Thống kê: bootstrap phân cấp, McNemar |
| `src/deployment/` | API FastAPI và giao diện Streamlit minh họa |
| `scripts/` | Lệnh vận hành và thí nghiệm (xem `scripts/README.md`, `scripts/OPS.md`) |
| `tests/` | Kiểm thử |
| `Docs/` | Tài liệu kỹ thuật (xem [`Docs/README.md`](Docs/README.md)) |
| `pdfdocx/` | Công cụ chuyển PDF sang Word giữ bố cục |

Tên trong báo cáo và tên trong mã: Mô hình Chú ý Hợp nhất = `b4_v6_ctr`, Mốc Kết hợp Đơn giản = `b3_mlp_ctr`, Ghép đặc trưng CTR = `b3_ctr_concat`, Chỉ RGB = `rgb_strong`.

## Cài đặt và kiểm thử

Python 3.11. Các môi trường tách riêng vì MediaPipe cần phiên bản protobuf khác:

| Tệp | Dùng cho |
|---|---|
| `requirements-ci.txt` | Kiểm thử, không cần GPU |
| `requirements-extract.txt` | Trích khung xương (MediaPipe) |
| `requirements-train.txt` | Huấn luyện và đánh giá (PyTorch, pytorchvideo) |
| `requirements-demo.txt` | API và giao diện minh họa |

```bash
python -m venv env_ci
env_ci/Scripts/pip install -r requirements-ci.txt   # Linux/macOS: env_ci/bin/pip
env_ci/Scripts/python -m pytest -m "not requires_artifacts"
```

Trong bản công khai, các bài kiểm thử cần dữ liệu hoặc bằng chứng niêm phong được bỏ qua tự động (danh sách ở `tests/private_assets_tests.txt`); các bài còn lại kiểm tra mã nguồn và không cần dữ liệu.

## Trích dẫn

Khi dùng mã này cùng VSL400, cần trích dẫn bộ dữ liệu theo yêu cầu của Data Usage Agreement:

- Nguyen Quoc et al., *VSL400: A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition*, preprint, DOI [10.22541/au.177075292.21810740/v1](https://doi.org/10.22541/au.177075292.21810740/v1).
- VSL400 dataset, Zenodo, DOI [10.5281/zenodo.17943574](https://doi.org/10.5281/zenodo.17943574).
