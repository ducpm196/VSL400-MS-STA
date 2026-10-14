# MS-STA VSL400 Lock

> **Live SoT:** MS-STA v6 = **`DEFENSE_GO`**. Proposed model = **B4-V6** (`b4_v6_ctr`).
> Evidence: `results/ms_sta_v6_defense_go/` · Active registry: `data/processed/ms_sta_registry_active.json` → `ms_sta_registry_v6_defense_go.json`. Rollback (historical): `ms_sta_registry_v5_defense_go.json`.

## Locked gates

| Gate | Status |
|---|---|
| D0 RGB-token v3 + Skeleton v3 corpus | PASS |
| Validation vs B3+MLP CTR, three seeds | PASS — 84,81% vs 80,54% |
| Retrospective signer-disjoint test, n=6165 | PASS — 51,55%, Δ +9,33 pp |
| D4 checkpoint/cache/fail-closed acceptance | PASS — 15/15 |
| E4 efficiency ablation | `E4_ACCURACY_NO_GO`, appendix only; does not override B4-V6 |
| Defense/demo B4-V6 | **DEFENSE_GO** |

Checkpoint: `Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt`
SHA256: `0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1`

## Claim limits

- Test is retrospective signer-disjoint confirmation, not pristine blind.
- Do not claim every metric improves on every signer.
- Do not describe E4 as deployed, faster, or accuracy-equivalent.
- Demo is cache-backed offline and requires human confirmation.
- MS-STA v5 B4* remains in rollback registry and historical evidence only — **not primary**.

Retrain-v3/v4 and v5 evidence remain immutable historical provenance; they are not the live model or demo identity.
