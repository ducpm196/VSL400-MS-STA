# MS-STA v5 - Rà soát độ ổn định Skeleton

## Kết luận

Skeleton v3 **không lấy một vai làm gốc**. Code chuẩn hóa dùng trung điểm của hai vai và chia cho khoảng cách hai vai ở từng frame. Quét read-only toàn bộ 24.752 tensor (799.815 frame) cho sai số tâm hai vai cực đại `8,12e-7`, với `0` frame pose invalid.

Ảnh demo cũ có một lỗi topology thật: các cạnh pose persisted nối sai local joint index. Lỗi này làm hình vẽ méo và tạo cảm giác khung xương lệch tâm. Renderer đã chuyển sang danh sách cạnh giải phẫu đúng, cố định canvas tại trung điểm hai vai, dùng timestamp thật và cung cấp hai chế độ **Raw sealed motion** / **Stabilized display**.

## Phạm vi ảnh hưởng

| Thành phần | Ảnh hưởng | Bằng chứng |
|---|---|---|
| Hình nối xương trong demo | Có | `POSE_BONES` persisted không khớp thứ tự 12 pose joint |
| Tâm chuẩn hóa Skeleton v3 | Không có lỗi một-vai | `max_abs(mid(L-shoulder,R-shoulder)) = 8,12e-7` trên toàn corpus |
| Checkpoint B4* hiện tại | Không bị ảnh hưởng bởi cạnh vẽ sai | `JointGraphEncoder` chỉ đọc coordinates, velocity, validity, part/chirality IDs |
| Nhiễu landmark theo thời gian | Có thể ảnh hưởng | B4* đọc trực tiếp coordinates và velocity; validity chỉ loại landmark invalid |

`bone_vectors` và `bone_edges` vẫn tồn tại trong cache để bảo toàn contract D0, nhưng không đi vào forward của B4*. Vì vậy không được diễn giải lỗi renderer là lỗi làm sai kết quả B4* đã khóa.

## Định lượng độ ổn định

Độ dịch chuyển dùng đơn vị bề rộng vai và chỉ tính cặp landmark hợp lệ liên tiếp.

| Tín hiệu | Median | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| Vai/hông | 0,0140 | 0,1037 | 0,2079 | 13,5618 |
| Gốc bàn tay | 0,0574 | 0,3711 | 0,6338 | 7,4111 |
| Face | 0,0134 | 0,0566 | 0,0949 | 2,0700 |

Tỷ lệ landmark invalid: pose `0%`, face `10,46%`, tay trái `4,68%`, tay phải `7,67%`. Có 5.240/24.752 chuỗi có ít nhất một bước vai/hông lớn hơn `0,25`, và 1.346/24.752 chuỗi có ít nhất một bước gốc bàn tay lớn hơn `1,0`.

Các ngưỡng này **không tự chứng minh detector sai**, vì chúng có thể chứa chuyển động thật. Tuy nhiên, các ngoại lệ lớn cùng hiện tượng mất/bắt lại bàn tay xác nhận representation có nhiễu temporal. Validity mask bảo vệ landmark bị mất, nhưng không tự loại landmark được đánh dấu valid mà định vị sai; velocity tại điểm bắt lại cũng có thể lớn. Đây là giới hạn hợp lệ của pipeline và là một nguyên nhân tiềm năng làm giảm trần accuracy, không phải quan hệ nhân quả đã được thí nghiệm chứng minh.

## Quyết định kỹ thuật

1. Giữ nguyên cache, checkpoint và metric v5 đã khóa; không làm lại số liệu sau khi xem test.
2. Sửa topology và ổn định canvas chỉ trong renderer; không dùng kết quả làm mượt cho inference.
3. Demo hiển thị RGB cạnh Skeleton và cho phép chuyển raw/stabilized để hội đồng kiểm tra trực tiếp.
4. Công bố B4* theo evidence hiện có; ghi temporal landmark noise là giới hạn representation.
5. Chỉ mở Skeleton v6 như nghiên cứu tiếp theo theo protocol đăng ký trước: confidence gating, reset velocity sau missing/reacquisition, robust derivative/outlier clipping và ablation công bằng B3+MLP/B4*. Chỉ claim tốt hơn nếu retrain cùng split, budget, seed và validation gate mới.

## Claim được phép

> Skeleton v3 được chuẩn hóa theo trung điểm hai vai; kiểm tra toàn corpus cho sai số tâm cực nhỏ. Demo cũ bị sai cạnh nối pose và đã được sửa ở lớp hiển thị. Landmark temporal vẫn có nhiễu, đặc biệt ở bàn tay; B4* dùng validity mask nhưng không loại hoàn toàn valid-but-noisy observations. Hạn chế này không làm mất hiệu lực so sánh đã khóa vì mọi baseline dùng cùng representation, nhưng có thể giới hạn accuracy tuyệt đối.

Không được claim rằng smoothing làm tăng accuracy, lỗi cạnh vẽ đã làm sai checkpoint, hoặc sửa preprocessing chắc chắn cải thiện model khi chưa retrain đối chứng.

Nguồn số liệu máy đọc: `Docs/results/ms_sta_v5_skeleton_stability/SKELETON_STABILITY_AUDIT.json`.
