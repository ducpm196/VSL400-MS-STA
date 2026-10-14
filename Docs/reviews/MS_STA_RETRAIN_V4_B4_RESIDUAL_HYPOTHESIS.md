# MS-STA retrain-v4 hypothesis: B4R residual on B3 (zero-init `cls` last Linear)

Status: **CLOSED · VALIDATION_NO_GO**
Live result SoT is **B3 v4 / `VALIDATION_NO_GO`** (**64,49 ± 0,58%** Top-1). B4R did not pass registered outperform gates (mean Δ +0,672 pp). v3 **62,29%** is historical only. Do not train v5 until a new campaign is registered.

## Blocker of the first B4R sketch (`zero_scale`)

```text
logits = B3 + alpha * attention_delta
alpha_init = 0
```

First-step grads: B3 head and `alpha` nonzero; **delta classifier / cross / gate = 0**. Initialization matches B3, but the improvement branch does not learn at step 0. Keep as **blocker ablation only**.

## Recommended default (`zero_delta_head`)

```text
base_logits  = concat_cls([rgb_pool, skel_pool])          # B3 path
delta_logits = cls(StreamGate(CrossAttention(...)))      # last Linear zero-init
final_logits = base_logits + residual_scale * delta_logits  # residual_scale init = 1
```

- Reuses B4’s `cls` as the delta head (no duplicate `delta_cls`).
- Param count: `n(b4r) == n(b4) + 1` (scalar `residual_scale` only).
- Init identity to B3 (delta output is exactly 0).
- `cls` last Linear receives first-step gradients; cross/gate receive gradients after that layer leaves zero.
- Legacy B4 state dict keys unchanged (no `delta_cls.*`).

Code: `src/models/ms_sta_wordlevel.py` · `residual_init=zero_delta_head|zero_scale`.

## Primary vs warm-start (do not mix)

| Track | What | In primary accuracy table? |
|---|---|---|
| **Primary** | B0/B3/B4/B4R from scratch; shared finite HPO; confirm full 16 epochs | Yes |
| **Warm-start diagnostic** | Fork a fresh B3-v4 checkpoint → `b3_continued` vs `b4r_warm` × seeds 42/123/456 (6 runs) | No |

## Finite accuracy HPO (draft)

```text
Variants: B0, B3, B4, B4R-zero-delta
LR:       1e-4, 3e-4, 6e-4
WD:       0.01, 0.05
Seeds:    11, 29 on tune
counts:   configs_per_variant=6; variant_config_pairs=24; runs=48
Selection: mean Top-1 → mean Top-5 → config_id (per variant)
```

Confirmation: full 16 epochs (no early stop); checkpoint = best val Top-1 within those 16.
`zero_scale` is ablation-only. Auxiliary delta loss opens only if mean tune Top-1 of B4R ≤ B3 after finite HPO (grad norms are diagnostic only).

Confirmation matrix: `B0 / B3 / B4 / B4R × seeds 42, 123, 456 = 12` with aligned per-sample validation rows.

Split: new deterministic Path-B development partition (`path_b_dev_pool_shuffle_v2`; seed = first 8 hex of SHA-256 of locked protocol string). Val ⊆ v3 hpo_train and disjoint from v3 tune∪val. Test 24–28 fixed, retrospective. Manifest is bit-exact (`created_at=protocol_locked`) and git-tracked.

## Accuracy gates (registered intent)

- B4R > B3 on all three confirm seeds (42, 123, 456)
- Mean ΔTop-1 ≥ **+1.0 pp** required for `architecture_gate_pass` / “outperform”
- Intermediate band `0 < mean ΔTop-1 < 1.0 pp` ⇒ claim language `improve_not_outperform` only (`architecture_gate_pass = false`); do **not** call this “vượt trội/outperform”
- Paired 95% CI of ΔTop-1 lower bound > 0 via seed-averaged per-sample correctness + hierarchical signer→sample bootstrap (10k, seed 42); never pool `3×n` as i.i.d.
- Mean ΔTop-5 ≥ −1.0 pp
- Macro-F1 B4R > B3
- No test if any gate fails

Efficiency is campaign **`v4e`**, only after accuracy PASS.

## Scaffold / freeze contract

- Policy draft: `Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY_DRAFT.json` (promote via `scripts/promote_ms_sta_retrain_v4_policy_candidate.py` to an external TEMP candidate)
- Immutable policy: `Docs/results/ms_sta_retrain_v4_policy/RETRAIN_POLICY.json` + `POLICY_FREEZE_BINDING.json` (two-step freeze; HPO only after post-freeze audit PASS)
- Orchestrator: `scripts/run_ms_sta_retrain_v4.py` (dry-plan reports `HPO_READY` when frozen; does not auto-start)
- Trainer: `src/training/train_ms_sta_vsl400.py` — B4R residual init + per-epoch residual/gate/grad diagnostics under retrain-v4

Identity:
- **B4** = historical attention-fusion control
- **B4R** = residual MS-STA proposal (`B3 concat + residual_scale × attention delta`)
- Live SoT is VALIDATION_NO_GO / B3 v4 ~64.49% Top-1. B4R is a non-outperforming hypothesis pending a registered v5 campaign.

Do not auto-chain confirm/lock/test/D4. After HPO 48, STOP for review.
