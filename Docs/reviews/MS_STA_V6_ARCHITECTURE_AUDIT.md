# MS-STA V6 — Architecture and campaign audit

## Verdict

**CONDITIONAL GO** cho campaign mới `B4-v6` → (nếu teacher PASS) `B4-E3`.

- Không tiếp tục tinh chỉnh E2 trên B4* như tối ưu cuối.
- E2 `EFFICIENCY_NO_GO` bất biến; Defense GO B4* giữ nguyên cho đến authorization riêng.
- “B4 tốt nhất” = tốt nhất trong **policy hữu hạn đã khóa**, không tối ưu toàn cục.

## Vì sao không sửa E2 tiếp

E2 giảm chủ yếu Skeleton temporal path; RGB gần như nguyên. Locked efficiency protocol cho thấy `e2_lite_s4` vẫn chậm hơn B4* dù accuracy non-inferiority PASS. Vòng tiếp theo cần teacher mạnh hơn và student nén **cả hai** luồng.

## Hai stage representation (tách bảng)

| Track | Input | Ghi chú |
|---|---|---|
| **Stage-A-fixed** | Skeleton v14 + RGB token v15 (SlowFast ROI, sealed) | Có thể pilot fusion/graph/cross ngay trên cache |
| **Stage-B-strong** | Skeleton v14 + RGB mới (VideoMAE-S hoặc Video-Swin-T) | Cần extractor mới / subset rồi full corpus; F0 weights đã có local |

Không ghép Top-1 Stage-A với Stage-B trong một mean.

## Ma trận đối chứng bắt buộc (mỗi stage)

1. `rgb_strong` — RGB only  
2. `b3_strong` — RGB + simple Skeleton fusion  
3. `b3_mlp_strong` — near-matched params, no cross-stream  
4. `b4_v6` — full spatial/temporal/cross/gate/residual  

## Cải tiến hữu hạn đề xuất

1. **RGB (Stage-B):** VideoMAE-Small hoặc Video-Swin-Tiny; partial FT block cuối; không Base trên RTX 2060 primary path.  
2. **Skeleton:** CTR-GC-lite / adjacency 75-joint riêng (pose/face/LH/RH); dùng velocity + bone; không copy NTU.  
3. **Cross:** predeclare one-way vs bidirectional trước metric.  
4. **Fusion:** bounded residual + regularization chống gate collapse.  

## Gate tóm tắt

**Teacher B4-v6:** thắng RGB-strong và B3+MLP-strong mọi seed; mean Δ≥+1 pp vs B3+MLP-strong; CI95 lower >0; Top-5/F1 không regress; không signer catastrophe.

**Student B4-E3 (chỉ sau teacher PASS):** ΔTop-1 ≥ −2 pp vs teacher; batch1 ≥15%↓; batch8 ≥10%↓; paired CI faster; p90/VRAM OK; báo tách forward/extract/E2E.

## Test note

Signers 24–28 đã quan sát → mọi test tương lai = **retrospective signer-disjoint confirmation**. Dev chỉ train/tune/val. Chỉ B4-v6 có thể xin test riêng để thay defense; E3 không cần test.

## References (design support only)

- VideoMAE V2: https://arxiv.org/abs/2303.16727  
- Video Swin: https://arxiv.org/abs/2106.13230  
- CTR-GCN: ICCV 2021 Chen et al.  
- vid-TLDR: CVPR 2024 Choi et al.  
