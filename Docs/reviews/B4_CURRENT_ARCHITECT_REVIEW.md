# Architect review - current B4 evidence contract

## Verdict

The project has three distinct B4 identities. They must not be merged in reports, slides, the API, or the demo:

| Identity | Purpose | Valid evidence | Status |
|---|---|---|---|
| **Current optimized B4** | Best current training result | Validation Top-1 **60.01 ± 0.96%**, Top-5 **86.28 ± 0.81%**, three locked seeds | **Current validation candidate**; no E2 or D4 |
| **Locked defense B4** | Historical blind-test and demo evidence | E2 Top-1 **23.84%**, Top-5 **50.67%**, `n=6165`; D4 15/15 | **Defense checkpoint already locked**; not the optimized checkpoint |
| **B4 stride-4 auxiliary** | Temporal-efficiency ablation | Skeleton score elements `4096 -> 256` (**-93.75%**); model-forward median `12.14 -> 8.76 ms` (**-27.80%**) | Validation/benchmark only; not the current or defense checkpoint |

Primary evidence:

- Current optimized B4: `Docs/results/ms_sta_accuracy/FINAL_REPORT.json`.
- Locked defense result: `Docs/results/ms_sta_defense_go/FINAL_RESULTS.json`.
- Efficiency ablation: `Docs/results/ms_sta_efficiency/FINAL_REPORT.json`.
- Active demo identity: `data/processed/ms_sta_registry.json`.

## Narrative decision

The defense narrative has three ordered claims. **Best-vs-best means equal-policy mean ± sample std**, not best seed and not mixing training budgets.

1. **Fair architecture comparison first (primary table).** Report B0 `40.50 ± 1.03%`, B3 `49.38 ± 1.43%`, and B4 `51.96 ± 0.83%` under the equal-policy E1-v2 matrix. This is the only valid absolute B0/B3/B4 comparison and the number set used to open the defense.
2. **Efficiency before/after second.** Report stride 1 versus stride 4 using the locked auxiliary study: accuracy, score elements, measured model-forward latency, and VRAM are separate quantities.
3. **Traceable pipeline third.** Block diagram from Front video through preprocessing, attentions, gate, and Top-k.

`60.01 ± 0.96%` remains a **B4-only appendix** (training-budget sensitivity). It must not appear on the cover KPI strip as a peer of the fair B0/B3/B4 table, and must not be inserted into that table because B0/B3 did not receive the new learning rate and epoch budget. E2/D4 stay identity hygiene for the earlier locked checkpoint.

The `51.96%` B4 result is retained only where scientifically required: the equal-policy B0/B3/B4 table and the stride-1 side of the efficiency before/after table. It must never be relabeled as current-best B4. E2/D4 remain evidence of the earlier locked checkpoint and are not assigned to current-best or stride-4 B4.

Immutable JSON evidence is never deleted, rewritten, or relabeled. Cleanup applies to narrative hierarchy, stale explanations, and presentation layout, not scientific provenance.
## Claims allowed now

- Current optimized B4 reaches validation Top-1 `60.01 ± 0.96%` and Top-5 `86.28 ± 0.81%` across seeds `42, 123, 456`.
- All three optimized runs improve over their paired earlier B4 runs; paired Top-1 delta is `+8.05 ± 1.60 pp`.
- An earlier equal-policy matrix supports the architectural claim that B4 improves over B0 and B3 on validation by `+11.46 pp` and `+2.58 pp` on average, respectively.
- The earlier locked B4 checkpoint improves over B0-E1 by `+7.28 pp` Top-1 on the signer-disjoint E2 test.
- The stride-4 auxiliary reduces Skeleton temporal-attention score elements by `93.75%` and measured model-forward median latency by `27.80%` under its locked benchmark.

## Claims not allowed now

- Current optimized B4 achieves `23.84%` on test or has passed D4.
- Current optimized B4 is fairly superior to old B0/B3 at the new training budget; only B4 received that optimization study.
- Stride-4 is part of current optimized B4 or the locked defense checkpoint.
- The whole model, end-to-end pipeline, FLOPs, VRAM, or latency is reduced by `93.75%`.
- The demo is real-time, production-ready, or supports arbitrary uploaded video.
- Results are directly comparable to VSL400 Table 3.

## Promotion requirement

To make current optimized B4 the sole defense model, the project needs a new predeclared evidence lifecycle:

1. Freeze a common training policy and rerun the required comparison variants under the same budget.
2. Select the primary seed on validation without consulting the existing test results.
3. Obtain a genuinely unused signer-disjoint holdout. Reusing E2 is descriptive evaluation, not a new blind confirmation.
4. Lock the selected checkpoint and hashes before holdout evaluation.
5. Run a new demo acceptance gate against that exact checkpoint.
6. Promote the registry only after all gates pass.

Until then, documentation must show both identities explicitly: **current validation candidate** and **earlier locked E2/D4 checkpoint**.
