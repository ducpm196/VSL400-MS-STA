# MS-STA v5 B4-E2 - Architecture and evidence audit

## Verdict

**CONDITIONAL GO cho một campaign mới `B4-E2`; không hồi sinh hay ghi đè B4-E cũ.**

- B4* (`b4_star_fixed`) vẫn là mô hình đề xuất và checkpoint bảo vệ.
- B4-E (`b4_star_e_stride4`) vẫn là ablation âm bất biến: validation Top-1 giảm `1,42 pp`, CI95 `[-3,70; +0,65] pp`, latency median tăng `12,43%`, VRAM không giảm.
- B4-E2 chỉ là nghiên cứu phụ về trade-off accuracy/latency. Không chạy final test, không auto-load demo và không thay registry B4*.

## Vì sao B4-E hiện tại không nhanh

1. Average pooling stride 4 nằm **sau** `JointGraphEncoder`; 75 joint của mọi frame vẫn được encode đầy đủ.
2. Chuỗi Skeleton chỉ khoảng 32 frame, nên self-attention chưa chắc là bottleneck. Pooling, mask và CUDA kernel launch có thể lớn hơn phần tính toán tiết kiệm.
3. B4-E có cùng `6.169.171` tham số với B4*; RGB temporal branch giữ nguyên.
4. B4-E downsample cả đường base Skeleton và cross-attention query, làm mất thông tin chuyển động trước khi classifier sử dụng.
5. Benchmark hiện tại là model-forward batch 8 FP32, chưa có component profile, batch 1, AMP hoặc cold/warm cache end-to-end.

## Sửa claim 93,75%

Với chiều dài hợp lệ `T` và stride `s=4`, số phần tử score thực tế phải so:

`reduction(T) = 1 - ceil(T/4)^2 / T^2`.

Chỉ khi `T` chia hết cho 4, reduction bằng đúng `93,75%`. Không được dùng `T_pad^2` làm baseline nếu B4* thực tế chạy `T^2`. Báo cáo mới phải xuất distribution theo `T`, tổng score thực tế trên corpus và điều kiện của công thức; không gọi đây là FLOPs, latency hoặc VRAM toàn model.

## Kiến trúc B4-E2 đề xuất

Tên khoa học: **Efficient MS-STA student with early motion-aware temporal reduction**.

### Khác biệt bắt buộc so với B4-E

1. Nén Skeleton **trước** phần encode joint đắt tiền, không sau encoder.
2. Pool theo validity từng joint; không đưa tọa độ missing vào trung bình.
3. Dùng motion energy từ `velocity` để giữ frame giàu chuyển động thay vì average đều.
4. Giữ một cheap full-rate summary theo bốn part (pose/face/LH/RH) để không mất hoàn toàn thông tin toàn chuỗi.
5. Dùng B4* làm teacher trên train IDs: CE nhãn thật + KL logits teacher + feature consistency. Teacher tuyệt đối không sinh loss từ validation/test.

### Candidate hữu hạn

| ID | Early reducer | Width/layers | Vai trò |
|---|---|---|---|
| `e2_s2` | motion-aware stride 2 | d=256, 2 layer | accuracy-first |
| `e2_lite_s2` | motion-aware stride 2 | d=192, 1 layer | trade-off chính |
| `e2_lite_s4` | motion-aware stride 4 | d=192, 1 layer | speed-first |
| `e1_late_s4` | average pool sau encoder | d=256, 2 layer | control âm hiện tại, không retrain/test |

Không thêm candidate sau khi đã xem kết quả pilot. Không đổi margin sau khi có metric.

## Gate khoa học đề xuất

### Accuracy non-inferiority trên validation

- Primary: hierarchical signer-bootstrap CI95 lower của `B4-E2 - B4*` Top-1 `>= -2,0 pp`.
- Mean Top-1 delta `>= -2,0 pp`.
- Mean Top-5 delta `>= -1,0 pp`.
- Macro-F1 regression `<= 2,0 pp`.
- Không signer validation nào giảm Top-1 quá `5,0 pp`.
- Confirm đủ seed `42/123/456`, cùng train/val IDs và epoch budget; báo mean ± sample std.

Margin `2 pp` là yêu cầu engineering đăng ký trước: chấp nhận tối đa thêm hai lỗi trên 100 mẫu để đổi lấy speed đo được. Đây không phải accuracy-equivalence tuyệt đối.

### Efficiency trên cùng RTX 2060

- Batch 1 model-forward median latency giảm ít nhất `15%`; paired bootstrap CI95 xác nhận delta latency `< 0`.
- Batch 8 median latency giảm ít nhất `10%`; p90 không chậm hơn B4*.
- Throughput không giảm; peak allocated VRAM không tăng quá `5%`.
- Đo FP32 và AMP riêng, warmup >= 30, interleaved randomized/ABBA >= 200 lần/model.
- Báo thêm cache-load + collate + H2D + forward batch 1; tách khỏi model-forward.
- Profiler phải chỉ ra phần giảm đến từ module nào; không suy từ score matrix.

Chỉ khi cả accuracy và efficiency cùng PASS mới dùng cụm **“B4-E2 đạt trade-off hiệu suất bổ sung”**. Nếu chỉ score count giảm hoặc latency không giảm, verdict vẫn supplementary NO-GO.

## Cơ sở nghiên cứu

- [Token Merging (ToMe)](https://arxiv.org/abs/2210.09461): merge token tương tự thay vì loại đều; nhấn mạnh throughput phải được đo thực tế.
- [DynamicViT](https://arxiv.org/abs/2106.02034): token importance và pruning động có thể tạo speed-up khi thiết kế phần cứng phù hợp.
- [vid-TLDR](https://openaccess.thecvf.com/content/CVPR2024/html/Choi_vid-TLDR_Training_Free_Token_Merging_for_Light-weight_Video_Transformer_CVPR_2024_paper.html): token merging cho video dựa trên saliency.
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531): teacher truyền phân bố dự đoán cho student nhỏ hơn.

Các paper là cơ sở thiết kế, không phải bằng chứng B4-E2 local đã nhanh hoặc giữ accuracy.

## Claim hierarchy

1. **Main claim:** B4* vượt RGB/B3*/B3+MLP theo evidence validation và retrospective signer-disjoint test đã khóa.
2. **Supplementary claim nếu E2 PASS:** trên validation và RTX 2060 protocol đã khóa, B4-E2 giảm latency đo được với accuracy nằm trong margin đã đăng ký.
3. **Không claim:** real-time, production, final-test equivalence, giảm 93,75% FLOPs/latency/VRAM hoặc nhanh trên mọi phần cứng.

