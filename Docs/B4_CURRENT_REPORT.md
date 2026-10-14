# Nguồn số liệu đã khóa — Unified Attention (Mô hình Chú ý Hợp nhất)

**Tên hiển thị hội đồng:** Unified Attention · Simple Combined Baseline · Optimized inference · Protocol A (cache-backed) · Protocol B (online).  

**Tên trong mã nguồn:** Unified Attention = `b4_v6_ctr` · Simple Combined Baseline = `b3_mlp_ctr` (B3+MLP) · Ghép đặc trưng CTR = `b3_ctr_concat` · Chỉ RGB = `rgb_strong`.

**Trạng thái:** chiến dịch bảo vệ đã khóa (`DEFENSE_GO`).  
**Tái lập kỹ thuật:** checkpoint seed 42 · registry `data/processed/ms_sta_registry_active.json` · chi tiết ID trong [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — không dùng làm ngôn ngữ kể chuyện.

**Đóng góp hệ thống trên cùng checkpoint:** Optimized inference (AMP + gom ROI). Online Baseline inference 49,895% vs Optimized 49,943%; Δ +0,049 điểm phần trăm; CI95 [−0,015; +0,147] chứa 0; acceptance **đủ 17 kịch bản**; demo không chuyển live Optimized inference. Bằng chứng: `results/e4_system_kd_v1/`.

## Bằng chứng chuẩn

Tập validation và tập test là hai protocol khác nhau; không gộp thành một trung bình.

| Phạm vi | Simple Combined | Unified Attention | Δ Top-1 |
|---|---:|---:|---:|
| Validation, ba seed | 80,54 ± 1,57% | **84,81 ± 0,51%** | **+4,27 điểm phần trăm** |
| Test Protocol A, n=6.165 | 42,22% | **51,55%** | **+9,33 điểm phần trăm** |

- Validation: Unified Attention thắng cả ba seed so với Simple Combined cùng encoder; CI95 **[+2,94; +5,69]** điểm phần trăm.
- Test: xác nhận signer-ID disjoint, **retrospective**, không phải pristine blind; CI95 **[+6,37; +11,34]**; McNemar *p* ≈ 2,8×10⁻⁵⁰.
- Không khẳng định vượt paper / bảng VSL400 công bố (khác giao thức).
- SHA256 checkpoint (tái lập): `0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1`.

Bằng chứng: `results/ms_sta_v6_defense_go/` · `results/ms_sta_v6_oneshot/` · `results/ms_sta_v6_d4/`.

## Phụ lục — nghiên cứu nén mạng (NO-GO)

Các chiến dịch thu nhỏ/nén (`E4_ACCURACY_NO_GO` và các gói liên quan) không đạt cổng độ đúng → **không** triển khai, **không** thay Unified Attention. E4 is a separate negative efficiency appendix — không phải engine bảo vệ. Chi tiết JSON trong `results/ms_sta_v6_e4*/` và `results/ms_sta_v6_defense_go/E4_EFFICIENCY_APPENDIX.html`. **Không nhầm** phụ lục này với Optimized inference (cùng checkpoint).

Cần token khóa cho đối chiếu: `cache-backed offline`.

## Giới hạn phát biểu

- Test là xác nhận signer-ID disjoint, retrospective, không phải pristine blind.
- Không khẳng định thời gian thực, sản phẩm vận hành, dịch câu liên tục, hay video tùy ý.
- Demo là **cache-backed offline** (Protocol A), fail-closed khi sai SHA/cache/API.
- Bản lịch sử / rollback chỉ provenance — **không phải nguồn chính**.

## Ranh giới demo

Streamlit minh họa Protocol A; Optimized inference chỉ bằng chứng (không cutover live). Xem [`phases/04_Deployment.md`](phases/04_Deployment.md).
