# Giai đoạn 2: Backbone và Feature Engineering

> **Trạng thái (Aug 2026): PASS.** Canonical RGB temporal và Skeleton cache đã full-scan, promote và khóa provenance.
> Tài liệu scaffold Stage-1/VOYA trước đây không phải implementation khoa học và đã được loại khỏi luồng đọc chính.
> SoT thiết kế: [`../MSSTA_VSL400_Lock.md`](../MSSTA_VSL400_Lock.md) · SoT kết quả: `../results/ms_sta_defense_go/FINAL_RESULTS.json`.

## 1. Phạm vi đã hiện thực

| Nhánh | Tensor/contract | Vai trò |
|---|---|---|
| RGB pooled | `(1, 2304)` từ `isolated_sf2304` | B0-E1 baseline |
| RGB temporal | raw `(L, 2304)`, `L >= 2`; canonical corpus có `L in [2,3]` | B1/B3/B4/B5 |
| RGB model input | `(8, 2304)` sau zero-pad/sample + validity mask | Temporal attention; padding không được xem là token thật |
| Skeleton | `(T, 75, 3)` float32 | Spatial/temporal skeleton stream |

`FIXED_RGB_L=8` là chiều dài tensor cho batching, **không** có nghĩa mỗi clip có tám segment quan sát. Canonical corpus chỉ có 2–3 SlowFast windows hợp lệ; vị trí padding bị chặn bằng `rgb_key_padding_mask`.

## 2. RGB stream

Backbone là SlowFast-R50 pretrained trên Kinetics và được đóng băng trong pipeline cục bộ. SlowFast gốc sử dụng một pathway tốc độ thấp để nắm ngữ nghĩa không gian và một pathway tốc độ cao để nắm chuyển động; paper gốc là bằng chứng cho kiến trúc video, **không** phải bằng chứng accuracy VSL400 của project.

Pipeline canonical:

```text
Front-view clip
  -> SlowFast feature extractor
  -> temporal windows [L, 2304], L >= 2
  -> deterministic pad/sample to [8, 2304]
  -> validity mask
  -> RGB TemporalTransformer
```

Code trace:

- Backbone/extractor: `src/models/rgb_backbone.py`, `scripts/build_rgb_temporal_cache.py`.
- Temporal corpus builder: `scripts/build_rgb_temporal_cache.py`.
- Shared length/mask contract: `src/models/ms_sta_features.py`.
- Active cache resolver: `resolve_temporal_dir()` đọc `data/processed/ms_sta_registry.json`.

## 3. Skeleton stream

MediaPipe Holistic cung cấp pose, face và two-hand landmarks. Project chọn một subset riêng gồm:

- 12 pose landmarks;
- 21 lip/face landmarks;
- 21 left-hand landmarks;
- 21 right-hand landmarks.

Tổng là 75 joint, mỗi joint lưu `(x, y, z)`. Đây là **thiết kế của project**, không phải topology 75 điểm tiêu chuẩn của MediaPipe.

Pipeline canonical:

```text
Front-view frames (stride 2)
  -> MediaPipe Holistic
  -> [12 pose + 21 face + 21 left hand + 21 right hand, xyz]
  -> missing-mask aware fill
  -> shoulder-reference normalization
  -> reject invalid scale / NaN / Inf / zero-frame
  -> smooth
  -> [T, 75, 3]
```

Code trace:

- Landmark selection/extract: `src/data_prep/extract_and_smooth.py`.
- Fill, normalization và quality contract: `src/data_prep/skel_postprocess.py`.
- Full-scan: `scripts/report_vsl400_skel_quality.py`.
- Tests: `tests/test_skel_postprocess.py`, `tests/test_feature_versioning.py`.

## 4. Provenance khóa

| Artifact | Giá trị canonical |
|---|---|
| Eligible manifest | `data/processed/vsl400_eligible_sample_manifest.json`, SHA256 `f6b04713…`, n=24.752 |
| Skeleton cache | `data/features/local_skel_v2_66fc5f01ee415fa1` |
| Skeleton code-lock | `66fc5f01ee415fa1dbd68846c38b7254898aae90`, clean worktree |
| Temporal cache | `data/features/isolated_sf_temporal_v2_8741189c657a` |
| Temporal code-lock | `8741189c657a4a8941bd09e3f2a4687ecf8f499e`, clean worktree |
| Resolver SoT | `data/processed/ms_sta_registry.json` |

Promotion chỉ hợp lệ sau full-scan PASS, count khớp eligible manifest, `failed=0`, path/hash/provenance khớp và `git_dirty=false`. Legacy dirs không được dùng cho canonical training.

## 5. Cơ sở học thuật và giới hạn claim

- SlowFast: Feichtenhofer et al. (ICCV 2019) hỗ trợ lựa chọn backbone video hai tốc độ.
- MediaPipe Holistic: tài liệu Google Research hỗ trợ pipeline landmark kết hợp; không cung cấp bảo đảm riêng cho VSL400.
- Transformer: Vaswani et al. (NeurIPS 2017) là cơ sở cho self-/cross-attention và positional encoding.
- SPOTER, SAM-SLR và DSTA-SLR là related work cho pose/multimodal/spatial-temporal SLR; project không tuyên bố tái lập các kiến trúc đó.

Nguồn và BibTeX đã kiểm: [`../references/VSL400_SOURCE_AUDIT.md`](../references/VSL400_SOURCE_AUDIT.md) · [`../references/references.bib`](../references/references.bib).

Các claim về latency, VRAM hoặc FLOPs chỉ được mở sau benchmark trên cùng hardware, batch size và precision. Phạm vi hiện tại là suy luận cache-backed offline.

## 6. Acceptance

- D1 skeleton / RGB temporal promotes remain lineage prerequisites for Path-B features.
- Live SoT hội đồng: **Unified Attention** (Mô hình Chú ý Hợp nhất) — Val **84,81 ± 0,51%**; test Protocol A **51,55%**. Feature lineage: D0 RGB-token + Skeleton.
- Bản MS-STA cũ / retrain lịch sử = rollback / `historical_not_primary` — **không** phải SoT bảo vệ.
- Full-flow Mermaid: [`../MSSTA_DEFENSE_FULL_FLOW_MERMAID.md`](../MSSTA_DEFENSE_FULL_FLOW_MERMAID.md).

