# Giai đoạn 1: Kỹ nghệ dữ liệu VSL400

> **Trạng thái (Aug 2026): D0/D1 PASS.**
> Canonical: [`../MSSTA_VSL400_Lock.md`](../MSSTA_VSL400_Lock.md) · Strategy: [`../Strategy_Implementation.md`](../Strategy_Implementation.md).

## 1. Phạm vi dữ liệu

| Mục | Giá trị canonical |
|---|---|
| Dataset | VSL400 word-level |
| Nhãn | 400 gloss |
| Signer local | 26; thiếu ID 5 và 17 so với mô tả 28 signer của nguồn |
| Góc dùng cho mô hình bảo vệ | Front |
| Split | Signer-disjoint |
| Eligible IDs | 24.752; loại `019442` do lỗi decode |

VSL-Cropped online-sourced và VOYA_VSL không được ghép vào closed-set KPI VSL400 vì ontology và cấp nhãn không đồng nhất với protocol đã khóa.

## 2. Split signer-independent

| Split | Số mẫu | Signer |
|---|---:|---|
| Train | 15.992 | 18 signer |
| Validation | 2.595 | 21, 22, 23 |
| Test eligible | 6.165 | 24, 25, 26, 27, 28 |

Các tập signer không giao nhau. Manifest đủ điều kiện:
`data/processed/vsl400_eligible_sample_manifest.json`.

## 3. Skeleton D1

```text
Front video
  → MediaPipe landmark extraction
  → missing-mask-aware fill
  → shoulder-reference normalization
  → reject invalid scale / NaN / Inf / zero-frame
  → smoothing
  → tensor [T,75,3]
```

Canonical cache: `data/features/local_skel_v2_66fc5f01ee415fa1`.

Kết quả full-scan và validated promote:

- `24.752/24.752` eligible tensor hiện diện.
- `full_scan=true`, `pass=true`, `failed=0`.
- Manifest, quality report, run contract, metadata SHA, sample-manifest SHA và git revision được đối chiếu trước promote.
- Canonical provenance dùng clean code-lock `66fc5f01ee415fa1dbd68846c38b7254898aae90`.

## 4. RGB temporal features

SlowFast-R50 đóng băng trích chuỗi đặc trưng `[L,2304]` từ video Front. Corpus canonical có `L∈[2,3]`, sau đó pad tới chiều dài cố định bằng zero padding và validity mask; padding không được xem là token quan sát.

Canonical cache: `data/features/isolated_sf_temporal_v2_8741189c657a`.

Kết quả full-scan và validated promote:

- `24.752/24.752` tensor, dtype `float32`.
- Split train/validation/test khớp eligible manifest.
- Bad shape, short sequence, NaN, Inf và all-zero đều bằng 0.
- Provenance khóa CUDA environment, SlowFast backbone, run contract và git revision `8741189c657a4a8941bd09e3f2a4687ecf8f499e`.

## 5. Code trace

| Vai trò | Path |
|---|---|
| Metadata | `src/data_prep/build_metadata.py` |
| Skeleton extract | `src/data_prep/extract_and_smooth.py` |
| Skeleton postprocess | `src/data_prep/skel_postprocess.py` |
| Temporal build | `scripts/build_rgb_temporal_cache.py` |
| Quality scan | `scripts/report_vsl400_skel_quality.py` |
| Validated promote | `scripts/promote_feature_dir.py` |
| Active dirs / provenance | `data/processed/ms_sta_registry.json` |

## 6. Acceptance

- D0 data contract: **PASS**.
- D1 skeleton full-scan/promote: **PASS**.
- Temporal full-scan/promote: **PASS**.
- Trainer, evaluator và backend resolve cùng `active_skel_dir` / `active_temporal_dir` từ registry.
- Thiếu feature trên eligible ID hoặc sai provenance làm pipeline dừng, không bỏ qua âm thầm.
