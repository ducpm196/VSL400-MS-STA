# MS-STA v5 Research Decision (B4*) — Research Lock Candidate

Status: **V5-R0 COMMITTED CANDIDATE · NO CODE · NO TRAIN · F0 NOT APPROVED**

Parent SoT (immutable until v5 gates PASS): retrain-v4 **`VALIDATION_NO_GO`** · canonical **B3 64,49 ± 0,58% Top-1**

Evidence: `Docs/results/ms_sta_retrain_v4/` · closeout HEAD: `805307a`

This document freezes the **research direction** for a future registered campaign. It does **not** authorize training, policy freeze, HPO, lock, test, F0 GPU work, or v5e.

**Provenance:** document SHA256 and binding commit are recorded in the external companion
`Docs/results/ms_sta_v5_r0/R0_LOCK.json` (schema `ms_sta_v5_r0_lock_v1`).
This markdown file **must not** embed its own content hash.

---

## 1. Architecture verdict

A small change to `residual_scale` alone cannot be expected to lift B4R (~65%) into the paper Front-only band (76,28–93,58%). Two objectives must be separated **and scored in separate tables** (see §8):

| Stage | Objective | Backbone / representation contract | Success criterion |
|---|---|---|---|
| **A — Controlled superiority** | **B4\*-fixed** beats Stage-A controls on the **same local protocol** | Fixed/frozen SlowFast feature lineage **v3** (compact region tokens; see §6.1) | Registered Stage-A gates PASS |
| **B — Absolute ceiling** | Port the **same** B4\* recipe onto a stronger **trainable** video backbone | Trainable strong backbone; feature maps may stay online | Stage-B gates PASS; absolute Top-1 only here |

Paper Front-only 76,28–93,58% is an **author-protocol reference**, not a mandatory local KPI. See `Docs/references/VSL400_SOURCE_AUDIT.md`.

**Scientific claim language (mandatory):**
On the same local protocol and the same stage backbone/representation contract (never mixed), **B4\*** stably outperforms RGB-only, concat (**B3\***), and an **input-matched + parameter-matched** control.
“Approach paper numbers” is an engineering aspiration only — never the primary thesis.

---

## 2. Current evidence (v4 — reference only)

| Model | Top-1 (confirm val, mean ± sample std) |
|---|---:|
| RGB frozen B0 | 37,11 ± 0,41% |
| **B3 concat** | **64,49 ± 0,58%** |
| B4 attention-only | 53,34 ± 1,02% |
| B4R residual (logit) | 65,16 ± 1,26% |

Interpretation locked for planning:

- Skeleton carries strong signal (B3 ≫ B0).
- Current attention path does **not** beat concat (B4 ≪ B3; B4R only marginal vs B3, gates FAIL).
- **Residual-on-B3 is a candidate family supported for further controlled testing; v4 does not establish causal superiority.**
- v4 used an **unbounded logit residual** — a design to replace, not a proven mechanism of gain.
- Current RGB cache `(L,2304)` with `L≈2–3` is **global-pooled only** and cannot support hand/face-guided ROI pooling over `F[t,h,w,c]`.

---

## 3. Direct blockers in current code (inventory)

These motivate v5 contracts; they are **not** a license to edit identity of B4/B4R in place.

| Blocker | Location / behavior |
|---|---|
| RGB backbone fully frozen | `src/models/rgb_backbone.py` — SlowFast-R50 `requires_grad=False` |
| Global-only RGB cache | `(L,2304)` tokens; **no spatial feature map** for ROIAlign |
| Degenerate RGB temporal length | Full-scan RGB cache typically **L≈2–3**, then padded to 8 |
| Skeleton forced to T=64 | `linspace(...).astype(int)` repeats short sequences; **no true pad mask** |
| Joint scores from XYZ only | No joint identity / body-part / chirality / bone graph |
| Cross-attn one-way | Skeleton-query → RGB-key/value; no timestamp alignment |
| B4R residual at 400-way logits | Single unbounded scalar × delta logits |
| Cache v2 insufficient for D0 | Filled tensors **cannot** recover pre-fill missing masks or native frame indices |

---

## 4. Architecture family lock: B4\* (not a fully frozen instance)

§4 locks the **family**. Exact cross-attention variant, graph encoder instance, residual arm (§4.1), and gate MLP must be frozen at **V5-M0 / V5-P0** before trial 1.

```text
Front video
  ├─ Stage A: frozen SlowFast → F[t,h,w,c] → landmark-guided ROIAlign
  │            → compact global / left-hand / right-hand / face tokens (RGB cache v3)
  ├─ Stage B: trainable strong backbone → F online → same ROI/token recipe
  └─ Skeleton extraction (Skeleton v3 / D0)
       → coordinates + velocity + bone vectors
       → joint/part/chirality embeddings + validity mask
       → graph-aware spatial encoder  [exact encoder frozen at M0/P0]
                 │
                 ├─ hand/face-guided pooling over F (ROIAlign; torchvision.ops)
                 └─ timestamp-aligned RGB–Skeleton tokens
                              │
                    temporal encoders with masks
                              │
                    cross-attention family (choice frozen at M0/P0):
                      { unidirectional skel→RGB , bidirectional , local/windowed }
                              │
                  bounded feature-level residual on B3*  [arm frozen at M0/P0]
                              │
                         shared 400-class head
```

### 4.1 Residual family (replaces v4 logit residual)

**Forbidden as primary:**
`logits = base_logits + residual_scale × delta_logits` (v4 B4R).

**Shared outer form:**

```text
h_base = B3* feature
u      = MS-STA update feature
h      = h_base + alpha * LN(u)
logits = shared_classifier(h)
```

`alpha` must be **bounded**. The following two arms are **diagnostic options**; **exactly one** is registered as **primary** at M0/P0 before trial 1. Do **not** claim both “sigmoid-only” and “zero-init exact identity” for the same primary arm.

| Arm ID | Formula / init | Exact B3\* identity | Sign of alpha |
|---|---|---|---|
| **R-ReZero** (zero-init bounded) | `alpha = alpha_max * tanh(g)` with last gate layer **zero-init** ⇒ `alpha=0` at init | Yes, at initialization and via `alpha_override=0` | May be negative |
| **R-PosEps** (positive small-init) | `alpha = alpha_max * sigmoid(g + b0)` init so `alpha≈ε>0` | **Not** at init; identity tests use explicit **`alpha_override=0` only** | Positive |

Additional requirements (all arms):

1. `alpha_max`, arm ID, and init **registered before train** (P0).
2. Auxiliary CE on the **base branch** so B3\* is not destroyed.
3. Shared classifier with B3\* under the same stage contract.

### 4.2 Identity rules

| Symbol | Meaning | Mutability |
|---|---|---|
| **B4 / B4R (v4)** | Historical attention / logit-residual | **Immutable** |
| **B4\*** | New architecture family in planned `ms_sta_v5.py` | New module; instance frozen at M0/P0 |
| **B3\*** | Concat baseline; stage-suffixed | Fresh train |
| **B3+MLP/ROI** | Input-matched + param-matched non-attention control | Fresh train; see §8 |
| **RGB-fixed-v5 / RGB-strong** | RGB-only controls | Never mix stages |

---

## 5. Literature role (claim discipline)

| Source | May support | Must not claim |
|---|---|---|
| VideoMAE (NeurIPS 2022) | Domain-specific video pretrain/FT | Local Top-1 without local run |
| Hand Pose Guided Pooling (WACV 2021) | Hand-guided RGB pooling vs global pool | Automatic B4\* win |
| SAM-SLR (CVPRW 2021) | Skeleton graph + RGB complementarity | Equivalence to local MS-STA |
| DSTA-SLR (LREC-COLING 2024) | Input-dependent joints / multi-scale time | Copy = our B4\* |
| SignBERT (ICCV 2021) | Hand identity / chirality / SSL hands | Mandatory full SignBERT |
| ReZero / LayerScale | Small/bounded residual init precedent | Causal proof B4\* rises |

AttnRes / Review Residuals (2026 preprints): **secondary reference only**.

---

## 6. Campaign stages (ordered; no skipping)

| ID | Name | Allowed work | Forbidden |
|---|---|---|---|
| **V5-R0** | Research lock candidate | This doc + `R0_LOCK.json` | Code, train, HPO, F0 GPU |
| **V5-F0** | Backbone feasibility | §9; isolated env; **no accuracy** | Backbone pick by val Top-1 |
| **V5-D0** | Data / representation rebuild | §6.1 Skeleton v3 + RGB token cache v3 | Infer masks from filled v2; store full F for all videos without size study |
| **V5-M0** | Architecture | `ms_sta_v5.py`; freeze cross/graph/residual arm | Edit B4/B4R; ad-hoc ROI math |
| **V5-T0** | Tests | Identity via arm rules; bounded α; grads; pad; timestamps; adjacency | Train |
| **V5-P0** | Policy | Equal-budget HPO **per stage table** | Silent aux; kitchen-sink |
| **V5-CV** | Eval design | Signer-grouped repeated/nested CV = **repeated validation** | “Untouched validation” |
| **V5-Gate** | Accuracy gates | §7 within one stage | Mixing Stage A/B metrics |
| **V5-Test** | Retrospective test | Test 24–28 once after PASS | Blind-test claim |
| **V5-E** | Efficiency | After accuracy PASS | Reuse 93,75% claim |

**Critical sequencing rule:**
Do **not** start with full VideoMAE training. Prove **B4\*-fixed** vs Stage-A controls first. Then port the recipe to Stage B. Do not change backbone + skeleton encoder + alignment + residual simultaneously without registered ablations.

### 6.1 D0 representation rebuild (mandatory)

**Rebuild:**

1. **Skeleton v3** from raw video: native timestamps, **`original_frame_index`**, **missing-detection mask before fill**, validity mask; native length + real pad masks; **no** linspace-repeat-to-64 as primary axis.
2. **RGB feature lineage / token cache v3** (Stage A) — global `(L,2304)` v2 is **insufficient** for ROI:

```text
raw video
  → frame/timestamp index
  → frozen SlowFast pre-pooling feature maps F[t,h,w,c]
  → landmark-guided ROIAlign
  → compact tokens: global / left-hand / right-hand / face
  → versioned RGB token cache v3
```

3. Do **not** store full feature maps for all ~24,752 videos until a **disk/VRAM size study** passes. Stage A default = **compact region tokens after ROIAlign**, with metadata:
   - backbone layer name;
   - backbone/checkpoint SHA256;
   - feature-map shape before ROI;
   - ROIAlign parameters;
   - spatial transform matrix (crop/resize → landmark space);
   - source frame/window indices (start/end/center);
   - missing-region mask.
4. Stage B may keep `F` **online** (trainable backbone); still report Stage B in its **own** table.
5. Crop/resize must transform landmarks into the **same** coordinate system as RGB windows.
6. **Forbid horizontal flip** in the primary campaign until label/chirality invariance is proven.
7. **Forbid** deriving missing masks from already-filled v2 tensors.

**Do not rebuild (if inventory/checksums remain valid):**

- raw VSL400 bytes;
- metadata / eligible manifest / test signer assignment 24–28;
- evidence trees v3/v4;
- historical checkpoints (**no warm-start** into primary v5).

**Must train fresh under v5 policy:** B3\*, B3+MLP/ROI, B4\* (both stages as applicable); Stage B arms from the **same** pinned pretrained backbone checkpoint.

---

## 7. Accuracy gates (proposed; freeze at V5-P0 per stage)

1. B4\* Top-1 > B3\* on every registered fold/seed **within the same stage**.
2. Mean ΔTop-1 ≥ **+1.0 pp**.
3. Paired 95% CI lower bound of ΔTop-1 **> 0** (signer-aware hierarchical bootstrap).
4. Mean ΔTop-5 ≥ **−1.0 pp**.
5. Macro-F1(B4\*) > Macro-F1(B3\*).
6. Primary unit: seed-/fold-averaged per-sample correctness.
7. No lock / test / v5e if any gate fails.
8. If `0 < mean Δ < 1.0` or CI includes 0: `improve_not_outperform` only.

Interpret contrasts:

- B4\* vs B3\*: full new system vs simple concat on the stage contract.
- B4\* vs **B3+MLP/ROI**: structured attention benefit under **near-equivalent inputs**.
- RGB-fixed-v5 / RGB-strong: contribution of skeleton / multistream.

---

## 8. Baselines — two primary tables (never merge)

### Stage A — fixed RGB lineage v3

| Arm | Inputs | Head |
|---|---|---|
| `RGB-fixed-v5` | Stage-A RGB tokens only | classifier |
| `B3*-fixed` | **global RGB + graph Skeleton** → concat | classifier |
| `B3+MLP/ROI-fixed` | **same global/ROI/graph tokens as B4\*** → concat/MLP **without** temporal/cross-attention | param-matched classifier |
| `B4*-fixed` | same tokens + temporal/cross + bounded residual | shared classifier family |

### Stage B — trainable strong backbone

| Arm | Inputs | Head |
|---|---|---|
| `RGB-strong` | Stage-B RGB only | classifier |
| `B3*-strong` | global RGB + graph Skeleton → concat | classifier |
| `B3+MLP/ROI-strong` | same tokens as B4\*-strong; no temporal/cross-attn; param-matched | classifier |
| `B4*-strong` | full B4\* recipe | shared classifier family |

**Forbidden:** mixing Stage A/B metrics; calling `B3+MLP` “matched” if it lacks ROI/graph tokens that B4\* receives.

### Excluded from primary claims

- Paper Table 3 as local KPI
- v3/E1/E2/D4 as live SoT
- v4 B4R as outperform / causal residual proof
- 93,75% as efficiency PASS
- Convex mix `(1−α)base + α·attn` as primary residual
- Kitchen-sink before M0/P0 freeze

---

## 9. F0 feasibility contract (draft — **not approved to run**)

F0 only after **re-audit of this amended R0** and a **separate F0 approval**. F0 does not choose the backbone by accuracy; D0 starts the representation rebuild.

1. Isolated env — do not mutate train/demo envs.
2. Pin implementation, pretrained checkpoint SHA256, license.
3. Matched input shape / clip length / resolution when feasible; document mismatches.
4. Record trainable parameter sets (VideoMAE / Video Swin / partial SlowFast stages).
5. Protocol: one optimizer step; warm-up; **multiple timed steps**; batch=1 first; AMP and grad-checkpoint as separate conditions; accumulation ≠ lower per-step VRAM.
6. PASS iff: no OOM; finite loss/grads; exact ckpt save/load; peak allocated/reserved VRAM + min free VRAM logged; correct shapes; **no accuracy**.
7. F0 only eliminates infeasible candidates.

Do **not** create `F0_PROTOCOL.json` or feasibility scripts under this R0 amendment.

---

## 10. Rebuild scope summary

| Rebuild | Keep |
|---|---|
| Skeleton v3 + masks/timestamps | Raw VSL400 if checksums OK |
| RGB token cache v3 (compact ROI tokens) | metadata, eligible manifest, test signers 24–28 |
| Fresh B3\* / B3+MLP/ROI / B4\* trains | v3/v4 evidence trees |
| Fresh Stage-B trains from one pinned pretrained ckpt | Historical ckpts (provenance only; **no warm-start**) |
| New v5 evaluator/policy/orchestrator namespace | — |

---

## 11. Authorization matrix

| Action | Authorized by this amended R0? |
|---|---|
| Amend/commit this doc + `R0_LOCK.json` | Yes |
| Create F0 protocol / download weights / GPU | **No** |
| Implement `ms_sta_v5.py` / D0 extractors | **No** until sequenced approvals |
| Train / HPO / lock / test / v5e | **No** |
| Change live SoT away from v4 B3 | **No** |

---

## 12. Next allowed step

1. Commit/push this amendment + `Docs/results/ms_sta_v5_r0/R0_LOCK.json`.
2. **STOP** for independent R0 re-audit.
3. Only after re-audit PASS: separate approval may authorize F0 artifacts.
4. No model download / GPU before that F0 approval.

Live SoT remains: **v4 VALIDATION_NO_GO · B3 64,49 ± 0,58% · B4R not outperform · no lock/test/v4e · F0 not approved**.
