# Roadmap — Continuous sentence recognition (future)

**Status:** deferred. Current thesis/package ships **VSL400 word-level MS-STA only**.

Continuous sentence recognition remains a valid **post-defense** direction once fresh continuous data are collected. Do not treat this note as shipped evidence.

## Why deferred

- Word-level Front SI protocol is locked and documented.
- Prior continuous data-prep and Seq2Seq training paths were removed for packaging cleanliness.
- New continuous clips, labels, and signer-disjoint splits must be rebuilt before any continuous claim.

## Suggested future stack (when data return)

1. Collect continuous signing videos with verified Vietnamese sentence labels.
2. Reuse MS-STA visual encoding (`ms_sta_wordlevel` / multi-stream features) as a frozen or fine-tuned encoder.
3. Add a sequence decoder (gloss sequence or direct text) under a new versioned results directory.
4. Freeze protocol (split, seeds, selection metric) before training; never reuse historical continuous artifacts.
5. Keep demo fail-closed and human-confirmed; do not claim real-time assistive translation without new gates.

## Explicit non-goals for now

- No continuous dataset under `data/raw/` is required for packaging.
- No continuous data-prep modules in `src/data_prep/`.
- Do not restore deleted Seq2Seq training code without a new design review.
