# pdfdocx — MCP chuyển PDF sang DOCX

Bộ này gọi đúng engine đã kiểm chứng trong `scripts/convert_datn_pdf_to_docx.py`. Không có bản layout thứ hai. Mỗi máy cài một lần bằng `python pdfdocx/install.py`.

## Cài cho một máy mới

Cần Python 3.11 trở lên. Lệnh sau tạo môi trường riêng tại `_venvs/pdfdocx` (không dùng gói cài sẵn của hệ thống), cài đúng phiên bản đã dùng khi dựng báo cáo, rồi ghi đường dẫn interpreter của máy đó vào `.cursor/mcp.json`.

```text
python pdfdocx/install.py
```

Sau đó: **Developer: Reload Window**.

Trên Windows, so ảnh cần WPS hoặc Word. `Word.Application` được đọc từ registry: máy này đang trỏ khóa đó tới `wps.exe`. Chuyển PDF không cần Word. Ba font phải có trong `C:\Windows\Fonts`: `times.ttf`, `cambria.ttc` (chứa Cambria Math), `cour.ttf`.

Phiên bản khóa trong `pdfdocx/requirements.txt`: `mcp==1.27.2`, `pymupdf==1.28.2`, `python-docx==1.2.0`, `fonttools==4.61.1`, `lxml==6.1.3`, và `pywin32==311` trên Windows.

## Contract

- Một trang PDF thành một trang DOCX, giữ chỗ xuống dòng.
- Baseline Word/WPS = đỉnh dòng + 0,8 × khoảng cách dòng chính xác.
- Bảng booktabs thành bảng Word, chỉ giữ đường kẻ ngang của PDF.
- Công thức một dòng lấy đúng lớp chữ của PDF và thành chữ trong DOCX. Công thức có gạch phân số, sơ đồ vector và ảnh raster vẫn được cắt từ PDF vì đó là bản đúng nét. Không dùng OCR: PDF đã có glyph và nét vector, nhận dạng lại từ ảnh sẽ sai ký tự.
- Chữ: Times New Roman. Chữ toán: Cambria Math. Mã: Courier New. Ngôn ngữ `vi-VN`, tắt kiểm tra chính tả.
- Lề chữ được căn theo PDF nộp: khoảng 99,21–538,58 pt trên khổ A4.
- `convert_pdf_to_docx` từ chối PDF có ô chữ lệch lề này và không ghi file, trừ khi `allow_uncalibrated=true`. Với PDF khổ A4, lựa chọn đó áp ô chữ đo được cho riêng lần chạy đó, rồi trả lại lề đã căn.
- `compare_docx_pdf` mở server mà `Word.Application` đang trỏ tới. Nếu đó là WPS thì chạy `wps.exe`; nếu đó là Microsoft Word thì chạy `winword.exe`. Khi khóa này không có, thử `kwps.Application`.
- Hồ sơ mặc định của `verify_docx` là `layout` (font và thân bài). Báo cáo đã nộp dùng `profile=datn`: 17 mục, gồm Top-1 51,55%, n=6165, speedup 1,220, khoảng tin cậy chứa 0.

## Tool MCP

| Tool | Việc làm | Ghi đĩa |
|---|---|---|
| `engine_contract` | Cam kết, giới hạn, font, lề đã căn, chính sách từ chối | Không |
| `runtime_status` | Đối chiếu phiên bản, font, file engine, đăng ký Word/WPS | Không |
| `inspect_pdf` | Đếm trang, khổ giấy, xem ô chữ có khớp lề đã căn | Không |
| `convert_pdf_to_docx` | Chuyển PDF khi ô chữ khớp. `docx_path` trống thì ghi cạnh PDF | Có |
| `verify_docx` | Mặc định `layout`. `datn` là 17 mục của báo cáo | Không |
| `compare_docx_pdf` | Xuất qua Word/WPS, so dòng đầu từng trang, ghi PNG cạnh nhau | Có |

Bốn tool chỉ đọc được auto-approve. `convert_pdf_to_docx` và `compare_docx_pdf` ghi file nên cần xác nhận.

`runtime_status.ok` là cổng trước khi chuyển: sai phiên bản, thiếu engine, hoặc thiếu font thì `ok` là false. `compare_ready` đứng riêng vì so ảnh cần Word/WPS còn chuyển PDF thì không.

## Kiểm tra

Cài `pytest` vào môi trường pdfdocx rồi chạy riêng file này. `tests/conftest.py` của dự án cần `torch`, nên phải có `--noconftest`.

```text
_venvs\pdfdocx\Scripts\python.exe -m pip install pytest
_venvs\pdfdocx\Scripts\python.exe -m pytest --noconftest tests/test_pdfdocx_mcp.py
```

Trên máy không phải Windows, thay `Scripts\python.exe` bằng `bin/python`.

## Dòng lệnh, không qua MCP

```text
python scripts/convert_datn_pdf_to_docx.py
python scripts/compare_datn_docx_pdf.py --pages 1,21,33
```

`--pdf`, `--docx` và `--work` của script so sánh nhận đường dẫn khác khi không dùng file mặc định của báo cáo.
