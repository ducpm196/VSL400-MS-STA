# VSL400 MS-STA v6 — active operations

Canonical dataset: **VSL400 Front-view**. Product surface: signer-independent word-level recognition with human confirmation.

## Status

- Campaign: **`DEFENSE_GO`**
- Model: **B4-V6** (`b4_v6_ctr`, seed 42)
- Validation Top-1: **84,81 ± 0,51%**; B3+MLP CTR: 80,54 ± 1,57%
- Test (retrospective signer-disjoint): **51,55%**; Δ vs B3+MLP **+9,33 pp**, n=6165
- D4: **15/15 PASS**
- Inference: cache-backed offline; no arbitrary-video or real-time claim

**B4-V6 is the proposed defense model. E4 / E4-R2 are separate negative efficiency studies and are not deployed (`e4_ready=false`).**

## Defense demo

```powershell
git status --short
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\demo_vsl400.ps1
```

The launcher requires:

- active registry schema `ms_sta_registry_v6_defense_go_v1`;
- variant `b4_v6_ctr` and exact checkpoint SHA;
- `demo_set_ms_sta_v6.json`;
- `/health` engine `ms_sta_v6_b4_v6_ctr` with `cache_backed_offline` input mode.

Open `http://localhost:8501`. Present scenario 1 (B4-V6 Top-1 correct), then scenario 2 (Top-5 requires human confirmation). The UI shows Front RGB and aligned 75-joint Skeleton motion side by side.

Báo cáo, slide và tài liệu bảo vệ: `Docs/thesis/README.md`.

Logs: `logs/demo_api.log`, `logs/demo_api.err.log`. Stop only the API/Streamlit processes by their verified command line; do not modify checkpoint, cache or evidence during the defense.

## Environments

Use only `env_demo` or `_venvs/demo`; `pip check` is a hard preflight. `GEMINI_API_KEY` is optional and not required for the isolated B4-V6 defense path. Never commit `.env`.

Historical demo: `demo_vsl400_historical.ps1 -AllowHistoricalDefense` is provenance-only and must not be used for the current defense. Rollback registry `ms_sta_registry_v5_defense_go.json` is historical only.
