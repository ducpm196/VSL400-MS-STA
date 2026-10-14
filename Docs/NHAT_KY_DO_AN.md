# Nhật ký đồ án — VSL400 & MS-STA

> **Số đang bảo vệ:** Unified Attention, trạng thái `DEFENSE_GO`. Validation **84,81 ± 0,51%** (hơn Simple Combined **+4,27 điểm phần trăm**); test signer-ID disjoint hồi cứu **51,55%** Top-1 (hơn **+9,33 điểm phần trăm**). Bản mô hình cũ (64,75% / 36,06%) chỉ là lịch sử. Nguồn số: [`B4_CURRENT_REPORT.md`](B4_CURRENT_REPORT.md). Sơ đồ: [`MSSTA_DEFENSE_FULL_FLOW_MERMAID.md`](MSSTA_DEFENSE_FULL_FLOW_MERMAID.md).
> Viết như người bạn đồng hành: kể chuyện từng bước, giảng kiến thức từ gốc, chỉ rõ đang ở đâu và sẽ đi đâu.
> **Không thay** nguồn sự thật khoa học (SoT). Thân nhật ký bên dưới có thể còn mô tả mốc v4/v5 tại thời điểm viết — khi lệch số, luôn ưu tiên SoT.

| Mục | Giá trị |
|---|---|
| **Cập nhật** | 24 Aug 2026 |
| **Số công bố** | Unified Attention · `DEFENSE_GO` · validation **84,81 ± 0,51%** · test **51,55%** · hơn Simple Combined **+9,33 điểm phần trăm** |
| **Vị trí vận hành** | Active registry `ms_sta_registry_v6_defense_go.json` · D4 **15/15** · demo fail-closed |
| **Không claim** | Pristine blind · SOTA paper 76–93% · E4 như engine đề xuất |

---

## 0. Cách đọc nhật ký này

Bạn không cần đã học deep learning. Đọc theo thứ tự:

1. **Phần 1** — nền tảng (tech + toán thô sơ).
2. **Phần 2** — câu chuyện đã đi qua (tự sự theo timeline).
3. **Phần 3** — đang đứng đâu (D0 v14) và flow đang chạy.
4. **Phần 4** — kịch bản sắp tới đến khi đồ án “chuẩn mực / có thể bảo vệ”.
5. **Phần 5** — bài học mang đi + checklist.

Gặp từ lạ → xem **bản đồ từ** ở §1.4. Muốn số cứng → mở SoT / evidence, đừng tin nhật ký hơn JSON.

---

## 1. Nền tảng: hiểu bài trước khi hiểu model

### 1.1 Bài toán bằng lời thường

Có khoảng **25 nghìn** đoạn video ngắn. Mỗi đoạn một người **ký một từ** bằng tay (ngôn ngữ ký hiệu Việt). Máy phải đoán: đó là từ nào trong **400 từ**.

- Đúng từ đứng đầu danh sách → **Top-1**.
- Từ đúng nằm trong 5 ứng viên đầu → **Top-5**.
- Không dịch cả câu — chỉ **nhận diện từ đơn** (*word-level*).

**Ví dụ trực giác:** giống nhìn ảnh chữ viết tay và đoán chữ, nhưng ở đây là **video cử chỉ**.

### 1.2 Vì sao khó? (signer-independent)

Nếu train bằng người A, B, C rồi test cũng A, B, C → dễ “nhớ mặt người” hơn là “học từ”.
Đồ án chọn kiểu khó hơn: **người lúc test không xuất hiện lúc train** (*signer-independent*).

*Chú thích:* Giống thi với đề mới và giám thị mới — chứng tỏ bạn hiểu bài, không thuộc máy.

### 1.3 Hai cách “nhìn” cùng một video

| Luồng | Là gì? | Hình dung |
|---|---|---|
| **RGB** | Ảnh màu theo thời gian | Camera nhìn hình dáng, áo, nền |
| **Skeleton** | Tọa độ khớp (cổ tay, ngón, mặt…) | Bộ xương số — bỏ màu áo, giữ cử chỉ |

Ý tưởng đề tài: **trộn hai luồng** có thể tốt hơn chỉ xem ảnh. Câu hỏi khoa học:

> Trộn bằng **Attention phức (MS-STA)** có **thắng** cách **ghép đơn giản (concat)** trên **cùng luật chơi** không?

### 1.4 Bản đồ từ (mang theo suốt hành trình)

| Mã / từ | Nghĩa đời thường |
|---|---|
| **VSL400** | Bộ dữ liệu ~400 từ ký hiệu (nguồn Zenodo) |
| **B0** | Chỉ RGB |
| **B3 / B3\*** | Nối thẳng RGB + Skeleton rồi phân loại — **baseline mạnh** |
| **B4 / B4R** | Attention / residual kiểu v4 — **lịch sử, chưa outperform** |
| **B4\*** | Họ kiến trúc v5 (ROI tay/mặt + residual feature có chặn) — **chưa train** |
| **Backbone** | Mạng “đọc video” sẵn (SlowFast, VideoMAE…) |
| **Token** | Một “mẩu thông tin” đã nén (như một câu tóm tắt ngắn) |
| **ROI** | Vùng quan tâm (ô quanh tay / mặt) |
| **Attention** | Cơ chế học “nên nhìn chỗ nào / lúc nào” |
| **Residual** | Không bỏ baseline; chỉ học **phần sửa thêm** |
| **Train / Val / Test** | Học / chỉnh & chọn / thi cuối (test chỉ mở khi đủ điều kiện) |
| **Top-1 %** | Tỷ lệ đoán đúng từ đứng đầu |
| **Seed** | Số khởi tạo ngẫu nhiên — chạy nhiều seed để khỏi “ăn may” |
| **Gate** | Luật phải đạt hết mới gọi “thắng” |
| **SoT** | Nguồn sự thật đang dùng |
| **NO-GO** | Chưa đủ chứng cứ — *không* nghĩa là đồ án hỏng |
| **Fail-closed** | Thiếu bằng chứng → **dừng**, không chạy tiếp |
| **R0 / F0 / D0 / M0 / T0 / P0** | Các pha v5 theo thứ tự (xem §3) |
| **Code-lock** | Commit khóa: chỉ code đó được phép chạy |
| **Canonical** | Kết quả được tin / promote — bản “vá tay giữa đường” không tính |
| **Promote** | Đưa cache lên làm chuẩn chính thức (có SHA) |

### 1.5 Toán thô sơ đủ dùng

**Phần trăm Top-1.**
Đúng 645 / 1000 mẫu → 64,5%. Trong nhật ký, **~64,49%** của B3 là chuẩn đối chứng hiện tại.

**Δ (delta) điểm phần trăm.**
65,16% − 64,49% ≈ **+0,67 điểm**. Nhỉnh nhẹ ≠ “vượt trội” nếu luật (gate) đòi ≥ +1 điểm và thắng mọi lần thử.

**Trung bình ± độ lệch.**
`64,49 ± 0,58%` ≈ chạy vài seed, trung bình 64,49; dao động khoảng 0,58. Số ổn định hơn một lần chạy may mắn.

**Vector / tensor (trực giác).**
Danh sách số xếp hàng. Video → hàng nghìn số. Model học cách biến hàng số đó thành “điểm cho 400 từ”.

**Softmax (trực giác).**
Đưa 400 điểm thô → 400 xác suất cộng lại = 1. Từ có xác suất cao nhất = Top-1.

**Concat.**
Ghép hai vector đầu–đuôi: `[RGB | Skeleton]` → một vector dài hơn → đưa vào lớp phân loại.

**Attention (metafora).**
Trong phòng họp, không nghe đều mọi người: học **ai đáng nghe hơn**. Spatial = khớp nào; temporal = lúc nào; cross = xương hỏi ảnh chỗ nào liên quan.

**Residual (metafora).**
Giữ bài làm gốc (B3), chỉ viết thêm phần sửa. Nếu phần sửa = 0 → đúng bằng bài gốc. v5 đòi phần sửa **có chặn** (không phóng đại vô hạn).

### 1.6 Pipeline tổng quát (một hình trong đầu)

```text
Video Front
   ├─► RGB backbone ──► token ảnh (v2 cũ: quá thô · v3 mới: có ROI tay/mặt)
   └─► Skeleton extract ──► token xương (v3: giàu hơn, có mask…)
              │
              ▼
     Ghép đơn giản (B3*)  ── hoặc ──  Attention / residual (B4*)
              │
              ▼
        400 điểm → Top-1 / Top-5
```

**Chuẩn mực đồ án:** không nhảy từ ý tưởng → train → slide. Đi theo **pha khóa** (R0→F0→D0→M0→…), mỗi pha có giấy phép hẹp.

---

## 2. Timeline tổng thể

```text
[1] Ý tưởng & phạm vi
        ↓
[2] Dữ liệu nền (cache v2) PASS
        ↓
[3] MS-STA vòng đầu (E1/E2/D4)          ← lịch sử
        ↓
[4] Retrain-v3 → NO-GO
        ↓
[5] Retrain-v4 → NO-GO                  ← ★ SoT accuracy (B3 chuẩn)
        ↓
[6] V5-R0 Research Lock                 ← khóa hướng B4*
        ↓
[7] V5-F0 feasibility CLOSED            ← 3 ứng viên Stage-B khả thi
        ↓
[8] V5-D0 pilot R7C-H1 PASS             ← ROI có QA mắt người
        ↓
[9] D0 full-corpus v2…v14               ← ★ đang ở v14: ready, chờ --launch
        ↓
[10] --launch → skel→rgb→scan→completion → PASS
        ↓
[11] Promote cache v3 · STOP · xin M0
        ↓
[12] M0 → T0 → P0 → Train Stage-A → Gate
        ↓
[13] Gate PASS → Test 1 lần → (Stage-B) → Demo GO
```

---

## 3. Câu chuyện tự sự — từng bước xây dựng

### Chương 1 — “Ta chọn một bài vừa đủ lớn”

Chúng mình chọn VSL400: có dữ liệu công bố, có paper tham chiếu, phạm vi **từ đơn** rõ ràng cho đồ án.

**Khó khăn sớm:** Dễ bị hút bởi số paper ~76–93% và tưởng “phải đạt bằng đó mới được”.
**Cách vượt:** Ghi audit nguồn — paper dùng protocol khác (split/backbone…). Số đó là **tham chiếu**, không phải đề thi bắt buộc.
→ [`references/VSL400_SOURCE_AUDIT.md`](references/VSL400_SOURCE_AUDIT.md)

**Học được:** Đặt **câu hỏi nghiên cứu** trước khi đặt **mục tiêu phần trăm**.

---

### Chương 2 — “Trước thông minh, phải sạch”

Xây metadata, lọc clip lỗi, split **không giao signer**, trích skeleton + RGB, promote có checksum.

Con số sống với đồ án:

| Tập | Khoảng số mẫu | Ý nghĩa |
|---|---:|---|
| Train | 15.992 | Học |
| Val | 2.595 | Chỉnh / so mô hình |
| Test eligible | 6.165 | Thi cuối (chỉ khi được mở) |
| Tổng eligible | **24.752** | Mọi pha extraction full-corpus |

**Bài học:** *Đừng train khi dữ liệu chưa PASS.*
**Giới hạn chỉ thấy sau này:** cache RGB v2 chỉ còn vài vector toàn cục (~2–3 token/clip) → Attention **không có chỗ nhìn tay/mặt** → phải có **D0 rebuild** ở v5.

---

### Chương 3 — “Ra đời MS-STA” (lịch sử)

Thiết kế hai luồng + spatial / temporal / cross-attention + gate. Có vòng E1/E2/D4; từng có Defense GO với checkpoint cũ.

**Intuition giữ mãi:**

- Spatial — khớp nào quan trọng?
- Temporal — lúc nào quan trọng?
- Cross — xương hỏi ảnh chỗ nào?
- Gate — trộn mềm hai luồng.

**Cách xử sự hôm nay:** Giữ evidence trong archive / thư mục lịch sử; **không** gọi đó là SoT Aug 2026.

---

### Chương 4–5 — “Đo lại cho thật — và nhận ra sự thật”

Retrain có kiểm soát (v3 rồi **v4**). Cùng luật chơi, nhiều seed, gate viết trước.

| Variant | Top-1 (v4) | Đọc cùng nhau |
|---|---:|---|
| B0 | 37,11 ± 0,41% | Chỉ RGB: yếu |
| **B3** | **64,49 ± 0,58%** | **Chuẩn ổn định** |
| B4 | 53,34 ± 1,02% | Attention thuần: kém B3 |
| B4R | 65,16 ± 1,26% | Nhỉnh +0,67 điểm — **gate thắng FAIL** |

Cảm giác lúc đó dễ là: “MS-STA thua thì sao?”. Câu trả lời trung thực:

> Không phải ý tưởng Attention sai.
> Mà **mắt–tai (representation) quá nghèo**, và **chỗ gắn residual chưa sạch** đủ để chứng minh vượt B3.

B4R được phép nói *có tín hiệu cải thiện nhẹ* — **không** được nói *outperform*. Demo MS-STA / mở test: **NO-GO**.
SoT: [`B4_CURRENT_REPORT.md`](B4_CURRENT_REPORT.md).

**Học được:** Thành công khoa học = **trung thực với gate**, không phải số cao nhất trên slide.

---

### Chương 6 — R0: “Khóa hướng trước khi sửa lung tung”

Viết research decision: họ **B4\***, tách hai mục tiêu.

| Stage | Hỏi gì? | Ghi điểm riêng |
|---|---|---|
| **A** | Trên cùng protocol cục bộ + cùng token v3, B4\* có thắng B0 / B3\* / control? | Claim khoa học chính |
| **B** | Cùng recipe lên backbone mạnh hơn (trainable) | Trần tuyệt đối / engineering |

Blocker đã liệt kê (giả thuyết thiết kế, không phải kết án nhân quả một câu): RGB thô · backbone đóng băng · skeleton méo · cross một chiều · residual logit không chặn.
→ [`reviews/MS_STA_V5_RESEARCH_DECISION.md`](reviews/MS_STA_V5_RESEARCH_DECISION.md)

**Học được:** *Một campaign = một câu hỏi.* Sửa hết một lần = không đo được gì.

---

### Chương 7 — F0: “Chạy nổi đã, đo hay tính sau”

F0 **không** chọn backbone bằng Top-1. Chỉ hỏi: ứng viên nào **chạy nổi** trên GPU (VRAM/time) trong env tách.

| Ứng viên | Kết luận |
|---|---|
| VideoMAE ViT-S / ViT-B · Video Swin Tiny | **FEASIBLE** → vào pool Stage-B |
| SlowFast partial-unfreeze | Loại khỏi pool Stage-B (contract trainable lệch) |

Pha **CLOSED**. SlowFast frozen vẫn dùng để **trích** RGB token v3 (Stage-A lineage) — khác chuyện “trainable Stage-B”.
→ `results/ms_sta_v5_f0/F0_PHASE_CLOSEOUT.json`

**Học được:** Feasibility ≠ accuracy. Trộn hai thứ là tự dối.

---

### Chương 8 — Pilot R7C-H1: “Máy bảo ổn chưa đủ — mắt người phải thấy”

Trước khi trích 24.752 mẫu: chứng minh ROI tay/mặt **không mơ hồ** trên pilot, có **visual QA thủ công**.

Gate script xanh ≠ khung tay đã đúng trên video. Independent re-audit **PASS** — phạm vi **chỉ pilot**.
→ `results/ms_sta_v5_d0_r7c/`

**Học được:** Với bài cử chỉ, **nhìn thử vài mẫu** rẻ hơn train cả tuần trên representation sai.

---

### Chương 9 — Full-corpus D0: “Con đường kỷ luật dài nhất” ★

Mục tiêu pha: trích **Skeleton v3 + RGB token v3** cho đủ 24.752 mẫu → quality scan → completion.
**Chỉ extraction.** Không train.

#### Flow vận hành chuẩn (mỗi phiên bản vN)

```text
Sửa code / remediation
    → Seal: approval + protocol + run-contract (vN)
    → External re-audit độc lập
         ├─ FAIL → giữ bằng chứng FAIL · sửa · lên v(N+1)
         └─ PASS → post-audit authorization
                     → evaluate_launch_readiness (ready?)
                     → vẫn CHƯA chạy: cần --launch tường minh
                     → watchdog + extraction
                          skeleton_v3 → rgb_token_v3 → quality_scan → completion
                     → SCAN_COMPLETE_PASS / FAIL
                     → PASS thì promote · STOP xin M0
                        (cấm auto-chain train/demo)
```

#### v2…v9 — mỗi FAIL một lớp giáp

Runner, scanner, watchdog, manifest atomic, SHA, heal, intent identity, schema Exact, chống CRLF / worktree giả…

**Tinh thần:** Không “sửa tay rồi claim PASS”. FAIL cũ **giữ immutable**.

#### v10 — bài học đau: chạy được ≠ chạy đúng

Đã PASS + auth + launch. Giữa đường: reboot, PID stale, rồi **runtime patch / `sitecustomize` ngoài provenance** (bỏ SHA verify, fast-resume).
Hàng nghìn NPZ **không canonical** → **cấm promote / cấm resume v10**.

Quyết định: bảo toàn evidence interruption · sang root mới · không đem bản “vá ngoài sổ” vào SoT.
→ `results/ms_sta_v5_d0_full/v10/`

#### v11…v13 — FAIL có chủ đích (test & audit identity)

| Vòng | Bài học ngắn |
|---|---|
| v11–v12 | Test readiness phải **hermetic** (fixture), không phụ thuộc “repo đang auth hay chưa” |
| v13 | Live audit phải lấy **binary root từ approval**, không bám HEAD ngẫu nhiên |

#### v14 — chỗ ta đứng hôm nay (15 Aug 2026)

| Việc | Trạng thái |
|---|---|
| Approval / protocol / run-contract | Sealed (v14) |
| External re-audit | **PASS** |
| Post-audit auth | **Sealed** (`launch_authorized=true`) |
| `evaluate_launch_readiness` | **`ready=true`** |
| `--launch` | **Chưa** |
| Binary root | `data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5/` — **trống** |
| Scope | `d0_full_corpus_extraction_only_v14` |
| Phases khi launch | skeleton_v3 → rgb_token_v3 → quality_scan → completion |
| Skeleton arm | C3 / HandRecovery |
| Eligible | 24.752 (train 15.992 / val 2.595 / test 6.165) |
| max_resumes | 20 |
| Code-lock | `2adbdf3d2bb5…` |
| Evidence | `results/ms_sta_v5_d0_full/v14/` |

**Một câu đồng hành:** Giấy tờ đã đủ để *được phép bấm launch* — **chưa** có nghĩa là đã trích xong, càng chưa được train.

---

## 4. Flow đang & sẽ phát triển (đến chuẩn mực đồ án)

### 4.1 Bản đồ pha v5 — không nhảy cóc

```text
R0 ──► F0 ──► D0 ──► M0 ──► T0 ──► P0 ──► CV/Gate ──► Test ──► E/Demo
 ✓      ✓     ★v14    │      │      │        │          │       │
              chờ      khóa   unit   policy  Stage-A    1 lần   sau
              launch   arch   test   HPO     accuracy   sau PASS accuracy
```

| ID | Tên dễ hiểu | Được làm | Chưa được |
|---|---|---|---|
| R0 | Khóa hướng | Doc + lock | Code/train |
| F0 | Backbone chạy nổi? | Isolated env | Chọn backbone bằng Top-1 |
| **D0** | Xây lại mắt–tai | Skeleton v3 + RGB token v3 | Train model |
| M0 | Khóa instance B4\* | Cross-attn / graph / residual arm | Sửa B4 cũ ad-hoc |
| T0 | Unit test identity | α=0 ≈ B3\* · mask · timestamp | Train đầy đủ |
| P0 | Policy + HPO hữu hạn | Equal-budget theo stage | Nhồi loss lung tung |
| Gate | Accuracy Stage-A | B4\* vs B3\* / controls | Trộn bảng Stage-B |
| Test | Signer 24–28 | **Một lần** sau gate PASS | Claim “blind” sớm |
| E / Demo | Hiệu năng · hội đồng | Sau accuracy PASS + **GO mới** | Demo khi NO-GO |

### 4.2 Kịch bản ngay trước mắt — D0 v14

**Kịch bản A — Happy path (đích gần)**

1. Worktree sạch · đúng code-lock / auth v14 · `ready=true`.
2. Gõ **`--launch` tường minh** (không auto từ auth).
3. Watchdog + 4 phase trên 24.752 mẫu; root bắt đầu trống.
4. Quality → **SCAN_COMPLETE_PASS**.
5. Promote / registry cache v3 (SHA + provenance).
6. **STOP** — xin approval riêng cho **M0**.

**Kịch bản B — Fail-closed giữa đường**

Watchdog / worker / SHA / env / patch ngoài lock lệch → dừng · ghi FAIL/interruption immutable · **không** promote partial · remediation → **v15** root mới nếu cần.

**Kịch bản C — Nhầm phạm vi (cấm)**

Đang extract / extract xong → nhảy train, demo, mở test; reuse NPZ v10; auto-chain M0.
→ Vi phạm chuẩn mực.

**Kịch bản D — Reboot / mất PID (nhớ v10)**

Không tin JSON `RUNNING` stale. Không plant `sitecustomize` / wrapper ngoài executable tree. Resume chỉ khi đủ SHA source+output+sidecar.

### 4.3 Sau D0 PASS — đường tới bảo vệ chuẩn mực

```text
[D0 SCAN_COMPLETE_PASS + promote cache v3]
        ↓
[M0] Freeze instance B4*
     (chọn đúng MỘT residual arm: R-ReZero XOR R-PosEps)
        ↓
[T0] Identity tests xanh
        ↓
[P0] Freeze policy Stage-A · HPO hữu hạn
     Đối chứng: B0 · B3* · control khớp input–param · B4*
        ↓
[Train + Confirm đủ seed]
        ↓
   ┌─ Gate FAIL → improve_not_outperform / đóng / đăng ký campaign mới
   └─ Gate PASS → Lock · Test 24–28 đúng một lần
                        ↓
                   (tuỳ chọn) Stage-B: VideoMAE/Swin từ pool F0
                        ↓
                   Efficiency · cập nhật SoT · Demo GO
                   · slide / nhật ký / report đồng bộ
```

**Tinh thần gate outperform (freeze chính thức ở P0):**

- B4\* > B3\* trên mọi seed/fold trong cùng stage
- Mean ΔTop-1 ≥ +1.0 pp · CI95 lower > 0
- Top-5 / Macro-F1 theo gate đã viết
- Thiếu **một** điều → chưa được nói “vượt trội”

### 4.4 Hai bảng điểm — đừng trộn bao giờ

| Bảng | Nội dung | Dùng để |
|---|---|---|
| **Stage-A** | Token v3 cố định (SlowFast frozen + ROI) | Claim chính: attention có thắng concat khi mắt–tai đủ giàu? |
| **Stage-B** | Backbone mạnh trainable (pool F0) | Trần tuyệt đối |

Paper 76–93% không thay Stage-A.

### 4.5 Metafora xuyên suốt (để nhớ flow)

| Giai đoạn | Metafora |
|---|---|
| B3 | Hai bạn đọc tờ tóm tắt rồi bỏ phiếu |
| B4 / B4R | Họp phức / thêm thư ký — chưa đủ “vượt trội” |
| D0 | **In lại tài liệu đủ chi tiết** (tay/mặt/mask) trước khi họp lại |
| v10 | Photocopy ngoài sổ sách → hủy phiên, mở sổ mới |
| v14 | Sổ mới đóng dấu sẵn — **chưa** bắt đầu viết cho đến khi `--launch` |
| M0–Gate | Soạn luật họp + họp thật + bỏ phiếu công bằng |
| Demo GO | Chỉ mời hội đồng khi có biên bản thắng đủ gate |

---

## 5. Checklist hoàn thiện chuẩn mực đồ án

### Đã có / đang giữ

- [x] SoT accuracy rõ (v4 NO-GO / B3)
- [x] R0 lock · F0 closeout · pilot R7C-H1 PASS
- [x] Full-corpus lifecycle fail-closed (v2…v14) · FAIL/interrupt immutable
- [x] v14 `ready=true` · auth sealed · chưa nhảy cóc train

### Sắp tới (theo thứ tự — đánh dấu khi xong)

- [ ] `--launch` v14 → **SCAN_COMPLETE_PASS**
- [ ] Promote cache v3 + registry provenance
- [ ] M0 freeze instance B4\* + residual arm
- [ ] T0 identity suite xanh
- [ ] P0 policy freeze · HPO hữu hạn
- [ ] B3\* train lại trên cùng contract (không lấy warm-start lịch sử làm primary)
- [ ] Confirmation đủ seed · mọi gate trong evidence
- [ ] Claim language đúng: improve / outperform / paper-ref tách bạch
- [ ] (Sau gate PASS) Test **một lần** · cập nhật SoT
- [ ] Demo chỉ khi **Defense GO** mới
- [ ] Slide / report / nhật ký **không** mâu thuẫn SoT

---

## 6. Playbook & xử sự khi khó

### 10 điều đồ án dạy (mang ra chỗ khác vẫn dùng)

1. Khóa dữ liệu / representation trước, mô hình sau.
2. Đối chứng đơn giản mạnh trước khi khoe mô hình phức.
3. Viết gate / protocol trước confirmation.
4. Fail-closed từng lớp: approval → audit → auth → readiness → **launch tường minh**.
5. Một campaign = một câu hỏi; PASS hẹp phạm vi.
6. Identity git-bound (SHA); không tin byte worktree / CRLF.
7. Test hermetic — không phụ thuộc “tip đang auth”.
8. Live audit bám **approval-owned root**, không bám HEAD ngẫu nhiên.
9. Không plant runtime ngoài executable tree — dù “chỉ để nhanh”.
10. Archive FAIL / interruption; không xóa để làm đẹp lịch sử.

### Khi cảm xúc nổi

| Khó khăn | Cảm giác | Việc nên làm |
|---|---|---|
| Nhiều vòng FAIL | Mệt | Đọc finding · sửa đúng lớp · seal vN+1 · giữ FAIL cũ |
| v10 interrupt | Tức | Không resume canonical · root mới |
| `ready=true` chưa launch | Nóng | Chỉ `--launch` khi đúng lock; theo dõi watchdog |
| Muốn train sớm | Nôn | Nhắc: D0 → M0 → T0 → P0 |
| Số thấp hơn kỳ vọng | Nản | Hỏi: đã so cùng stage / cùng mắt–tai chưa? |
| Paper cao hơn local | Tự ti | Mở source audit |
| Demo khi NO-GO | Áp lực | Tắt MS-STA defense; trung thực với hội đồng |

---

## 7. Bản đồ đọc tiếp

| Muốn hiểu | Đọc |
|---|---|
| **Báo cáo đồ án (PDF)** | `thesis/VSL_MSSTA_REPORT.pdf` |
| SoT accuracy | [`B4_CURRENT_REPORT.md`](B4_CURRENT_REPORT.md) |
| Tổng quan nghiên cứu | [`Research_Overview.md`](Research_Overview.md) |
| Hướng B4\* | [`reviews/MS_STA_V5_RESEARCH_DECISION.md`](reviews/MS_STA_V5_RESEARCH_DECISION.md) |
| F0 đóng pha | `results/ms_sta_v5_f0/F0_PHASE_CLOSEOUT.json` |
| Pilot ROI + QA người | `results/ms_sta_v5_d0_r7c/` |
| **Full-corpus v14 (đang đứng)** | `results/ms_sta_v5_d0_full/v14/` |
| v10 interruption | `results/ms_sta_v5_d0_full/v10/` |
| Paper ≠ local | [`references/VSL400_SOURCE_AUDIT.md`](references/VSL400_SOURCE_AUDIT.md) |
| Index docs | [`README.md`](README.md) |

---

## 8. Lời kết — đứng ở đâu, đi tiếp thế nào

Chúng ta đã làm được những việc “ít hoa nhưng nhiều xương sống”:

1. **Đo trung thực** (v4): attention cũ chưa outperform concat.
2. **Khóa hướng** (R0) và **loại backbone không khả thi** (F0).
3. **Chứng minh ROI pilot bằng mắt người** (R7C-H1).
4. **Trải launch hỏng có kỷ luật** (v10) — biết thế nào là non-canonical.
5. **Siết identity test/audit** (v11–v13) rồi **seal v14 sẵn sàng launch sạch**.

Chúng ta **chưa** được phép nói “MS-STA v5 đã thắng” — và đó là dấu hiệu đang làm **đúng chuẩn mực**, không phải chậm.

**Bước đồng hành kế tiếp:**

```text
--launch D0 full-corpus v14
    → skeleton_v3 → rgb_token_v3 → quality_scan → completion
    → SCAN_COMPLETE_PASS
    → promote cache v3
    → STOP xin M0
```

Mỗi PASS sau (M0 → Gate → Test → Demo GO) là một dấu mốc cùng nhau vượt — **không nhảy cóc**.

Nếu bạn mới vào repo: đọc §1 cho vững chân, §3 để biết câu chuyện, §4 để biết mình đang đứng trước cánh cửa nào. Rồi mở evidence JSON khi cần số — nhật ký này là **bạn đồng hành**, SoT mới là **trọng tài**.

---

*Cập nhật nhật ký khi: D0 scan PASS/FAIL · interruption mới · M0 freeze · campaign accuracy đổi SoT · Defense GO mới. Giữ giọng đồng hành; không ghi đè evidence JSON.*
