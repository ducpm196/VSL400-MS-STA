# VSL400 Source Audit and Comparison Rules

> Mục đích: khóa nguồn, số liệu và cách diễn đạt khi viết báo cáo, slide hoặc bài báo.
> Tài liệu này không thay thế SoT kết quả cục bộ tại
> [`../B4_CURRENT_REPORT.md`](../B4_CURRENT_REPORT.md).
> **Kết quả đang bảo vệ:** Unified Attention, trạng thái `DEFENSE_GO`. Validation **84,81 ± 0,51%**, test hồi cứu **51,55%** so với Simple Combined. Bản mô hình cũ (64,75% / 36,06%) chỉ là lịch sử, không phải số công bố.

## 1. Nhận dạng nguồn

| Nguồn | Định danh dùng trong báo cáo | Trạng thái |
|---|---|---|
| VSL400 manuscript | Nguyen Quoc et al., *VSL400: A Multi-view Dataset for Vietnamese Word-Level Sign Language Recognition*, DOI [`10.22541/au.177075292.21810740/v1`](https://doi.org/10.22541/au.177075292.21810740/v1) | Preprint đăng ngày 10/02/2026, CC BY 4.0, **chưa peer review** |
| Bản PDF đã audit | `Docs/au.177075292.21810740_v1.pdf` | SHA256 `19278a92a43597f89942e8a618da5f48ffe399e0204f6dcaafd90908e8b700c0` |
| Dataset record | [Zenodo record 17943574](https://zenodo.org/records/17943574), DOI [`10.5281/zenodo.17943574`](https://doi.org/10.5281/zenodo.17943574) | Bản ghi dữ liệu cụ thể, công bố 09/03/2026 |

**Lưu ý DOI:** trang Zenodo hiển thị DOI của bản ghi là `10.5281/zenodo.17943574`, nhưng phần mô tả do tác giả nhập lại ghi `10.5281/zenodo.17943573`. Báo cáo dùng `17943574` để định danh đúng bản ghi được truy cập; có thể ghi thêm `17943573` là “DOI được nêu trong mô tả Zenodo”, không tự gán vai trò concept/version khi chưa có metadata xác nhận rõ hơn.

## 2. Sự kiện được nguồn hỗ trợ

| Claim | Bằng chứng trong preprint | Cách ghi an toàn |
|---|---|---|
| 400 gloss, 74.259 clip, 28 signer, 3 RGB view đồng bộ | Abstract; §2.5, Table 2 (trang in 10) | “Theo Nguyen Quoc et al. (2026), VSL400 gồm…” |
| Trung bình 2,61 giây/clip, tổng 53,99 giờ | §2.5, Table 2 | Ghi rõ là thống kê của toàn bộ ba view |
| 23 người điếc dùng VSL bản ngữ + 5 người nghe đã được huấn luyện | §2.1 (trang in 6–7) | “Nhóm tác giả báo cáo…”; không suy rộng đại diện dân số |
| Metadata có ID video, signer, gloss, frame count và runtime | §2.7 (trang in 17) | Chỉ mô tả schema do paper công bố |
| Khuôn mặt được làm mờ; video được chuẩn hóa resolution/frame rate | §2.7 (trang in 17) | “Paper cho biết…”; không xem là kiểm toán quyền riêng tư độc lập |
| Baseline dùng official implementations, sửa đổi tối thiểu, protocol thống nhất và signer-disjoint | §3.2 (trang in 19–20) | Có thể nói paper **tuyên bố signer-disjoint**; không nói paper trộn signer |
| Split 80/10/10 | §3.3 (trang in 20) | Paper nêu tỷ lệ nhưng không công bố đủ signer-ID assignment để tái dựng split từ nội dung PDF |

`74.259 / 3 = 24.753` là phép suy ra từ ba view đồng bộ, không phải một hàng riêng trong Table 2. Con số `24.753` cục bộ còn được kiểm tra trực tiếp từ `metadata.csv`.

## 3. Audit Table 3 của VSL400

Đơn vị: accuracy (%). Các số dưới đây được chép từ Table 3 (trang in 20), không phải kết quả tái lập trong repository này.

| Model | Front Top-1 | Front Top-5 | Tri-view Top-1 | Tri-view Top-5 | Delta tính lại |
|---|---:|---:|---:|---:|---:|
| VideoMAE | 76,28 | 90,18 | 80,62 | 94,26 | +4,34 / +4,08 |
| Swin3D | 90,03 | 95,68 | 93,79 | 96,35 | +3,76 / +0,67 |
| DSTA-SLR | 90,74 | 95,72 | 92,85 | 96,13 | +2,11 / +0,41 |
| SAM-SLR | 91,05 | 97,42 | 94,84 | 96,62 | **+3,79 / -0,80** |
| SPOTER | 93,58 | 96,62 | 96,19 | 97,58 | +2,61 / +0,96 |

**Sai lệch nội tại của nguồn:** Table 3 in gain Top-5 của SAM-SLR là `+1,78`, nhưng `96,62 - 97,42 = -0,80`.

**Cách dùng Table 3 trong báo cáo:**

- Claim hợp lệ: “Tri-view tăng Top-1 ở cả năm baseline được paper báo cáo.”
- Không suy rộng: “Tri-view tăng mọi metric” hoặc “tăng Top-5 ở cả năm baseline.”
- Nếu nêu SAM-SLR Top-5: ghi hai giá trị gốc và chú thích phép tính lại; không âm thầm sửa bảng của paper.

## 4. Paper protocol và local protocol không đồng nhất

| Trục so sánh | VSL400 preprint | Repository này |
|---|---|---|
| Signer | 28 | 26 signer khả dụng; thiếu ID 5 và 17 |
| View cho KPI chính | Front-only và tri-view báo cáo **riêng** | Front only (v5 validation và retrospective signer-disjoint test) |
| Split được nêu | 80/10/10, signer-disjoint | Path-B v4 split (val n=2629); test 24–28 retrospective only |
| Test signer assignment | Không đủ chi tiết trong PDF để tái dựng | signer 24–28, khóa trong `vsl400_split_signers.json` |
| Backbone/baseline | VideoMAE, Swin3D, DSTA-SLR, SAM-SLR, SPOTER | Đặc trưng RGB và skeleton đã khóa; Unified Attention và Simple Combined |
| Mẫu hợp lệ | Paper không nêu clip decode-fail cục bộ | 24.752 eligible; loại `019442` |

Hai bên đều dùng mô tả cấp cao “signer-disjoint”, nhưng khác tập signer khả dụng, tỷ lệ/assignment split, backbone, preprocessing và training policy.

Paper báo cáo hai thiết lập riêng: Front-only và tri-view. Khoảng **76,28–93,58%** là kết quả **Front-only**; không sử dụng đồng thời ba camera trong khoảng đó. Đối chứng hợp lệ của đồ án là **Unified Attention và Simple Combined**. Unified Attention đạt **84,81 ± 0,51%** trên validation ba seed và **51,55%** trên test signer-ID disjoint hồi cứu; không gọi test là kiểm thử mù nguyên trạng. Bản mô hình cũ (64,75% / 36,06%) chỉ lưu lịch sử.

**Đối chứng hợp lệ (live):**

| So sánh | Gate | Ý nghĩa claim |
|---|---|---|
| Unified Attention so với Simple Combined | Validation và test signer-ID disjoint hồi cứu | Validation khoảng 84,81%; test +9,33 điểm phần trăm; `DEFENSE_GO` |
| Bản mô hình cũ so với Simple Combined | Giao thức cũ | Chỉ lịch sử, không phải số công bố |
| Các mốc nội bộ cũ | Cùng chiến dịch lịch sử | Thang đối chứng nội bộ, không phải số công bố |
| Bảng 3 của paper so với Unified Attention | **Không** | Khác giao thức, chỉ tham khảo |

**Lịch sử, không phải số công bố:** các thí nghiệm nén và bản đối chứng cũ (Top-1 test 23,84% so với mốc ảnh) giữ để truy vết; không đưa vào kết quả chính.

## 5. Vai trò của các tài liệu liên quan

| Nguồn | Điều nguồn đó hỗ trợ | Không được suy diễn |
|---|---|---|
| [SlowFast](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html) (Feichtenhofer et al., 2019) | Thiết kế hai pathway video: Slow cho ngữ nghĩa không gian, Fast cho chuyển động thời gian | Không chứng minh trực tiếp accuracy VSL400 của implementation cục bộ |
| [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) (Vaswani et al., 2017) | Scaled dot-product/multi-head attention và positional encoding | Không tự chứng minh MS-STA tốt hơn trên VSL |
| [MediaPipe Holistic](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/) (Google Research, 2020) | Pipeline kết hợp pose, face và hai hand landmark | Tensor 75 khớp là **subset do project chọn**, không phải số landmark chuẩn của Holistic |
| [SPOTER](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html) (Boháček & Hrúz, 2022) | Transformer trên chuỗi pose cho word-level SLR | Không phải kiến trúc được tái lập trong project |
| [SAM-SLR](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html) (Jiang et al., 2021) | Multimodal RGB/depth/skeleton cho isolated SLR | Không đồng nghĩa với RGB–Skeleton MS-STA của project |
| [DSTA-SLR](https://aclanthology.org/2024.lrec-main.484/) (Hu et al., 2024) | Tổng hợp quan hệ spatial-temporal và temporal đa tỷ lệ cho skeleton-aware SLR | Không phải bằng chứng cho hiệu quả của cross-attention cục bộ |
| [VideoMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html) (Tong et al., 2022) | Masked video pretraining | Không phải backbone RGB của đồ án |

**Source gap cần công khai:** reference [45] của preprint gắn tên “Swin3D” với bài [*Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding*](https://doi.org/10.26599/CVM.2025.9450383) (point-cloud/indoor-scene), trong khi §3.2 mô tả một transformer video. Không tự thay thành Video Swin Transformer; cần tác giả VSL400 xác nhận implementation/reference trước khi dùng baseline này cho phân tích kiến trúc sâu.

## 6. Mẫu diễn đạt dùng trong báo cáo

**Khi hội đồng hỏi về khoảng Front-view 76–93%:**

> Paper báo cáo hai thiết lập riêng: Front-only và tri-view. Khoảng 76,28–93,58% là kết quả Front-only theo protocol của tác giả (VideoMAE, Swin3D, DSTA-SLR, SAM-SLR, SPOTER), không phải đối chứng trực tiếp của đồ án. Đối chứng hợp lệ là Unified Attention và Simple Combined: Unified Attention đạt 84,81 ± 0,51% trên validation và 51,55% trên test signer-ID disjoint hồi cứu. Bản mô hình cũ chỉ là lịch sử. Không xếp hạng tốt nhất trên VSL400 và không tái lập Bảng 3 của paper.

**Bảng lệch protocol** đặt ở phụ lục / slide Q&A dự phòng.

**Cách nói khi so sánh:** Bảng 3 của paper là kết quả theo giao thức của tác giả. Đối chứng hợp lệ của đồ án là Unified Attention so với Simple Combined. Bản cũ và các thí nghiệm trước không phải số công bố. Muốn so một baseline của paper thì phải chạy lại trên bản chia tập của đồ án; đó là reproduction cục bộ, không phải tái lập Bảng 3.

## 7. Quy tắc trích dẫn

- Dùng file [`references.bib`](references.bib) cho báo cáo/bài báo.
- Nêu “preprint, chưa peer review” ở lần đầu giới thiệu VSL400 manuscript.
- Nêu ngày truy cập nếu trường yêu cầu cho Zenodo/web resources.
- Khi tái sử dụng bảng/hình, tuân thủ CC BY 4.0 và ghi nguồn ngay dưới bảng/hình.
- Không trích PDF cục bộ như một publication khác; trích DOI của preprint và ghi SHA PDF trong phụ lục tái lập.
