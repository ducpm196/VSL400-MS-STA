# Reproducibility — MS-STA v6 Defense GO

## Canonical identity

- Campaign: **`DEFENSE_GO`**
- Variant: `b4_v6_ctr`, primary seed 42
- Checkpoint: `Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt`
- Checkpoint SHA256: `0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1`
- Active registry: `data/processed/ms_sta_registry_active.json` → `ms_sta_registry_v6_defense_go.json`
- Rollback registry (historical): `data/processed/ms_sta_registry_v5_defense_go.json`
- Validation lock: `Docs/results/ms_sta_v6/V6_TEACHER_LOCK.json`
- Test completion SHA256: `8d2394bc830364230c70240ce6584fde1fad5f8b399c336f24f7c9a2d5218acb`

## Verification

```powershell
python scripts/summarize_ms_sta_gates.py
_venvs\ci\Scripts\python.exe -m pytest -m "not requires_artifacts" -q
_venvs\ci\Scripts\python.exe scripts/claim_lint_primary_sot.py
_venvs\demo\Scripts\python.exe -m pip check
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\demo_vsl400.ps1
```

Expected summarizer verdict: `DEFENSE_GO`, exit 0. The demo preflight must report engine `ms_sta_v6_b4_v6_ctr`, exact checkpoint SHA and `cache_backed_offline` input mode.

## Scientific boundaries

- Validation numbers are mean +/- sample std across seeds 42/123/456.
- Test is a one-shot retrospective signer-disjoint confirmation on 6.165 eligible IDs, signers 24–28.
- Demo reuses sealed feature caches; it is not a benchmark of raw-video real-time latency.
- E4 is appendix-only (`E4_ACCURACY_NO_GO`); not deployed.
- Historical v3/v4/v5 and historical E1/E2/D4 remain provenance only.
