# Scripts surface — VSL400 MS-STA

## Active (primary)

| Script | Role |
|--------|------|
| `evaluate_vsl400.py` | SI Top-1/Top-5 (+ `--fusion`, `--cache-mode read_only`) |
| `demo_vsl400.ps1` | Preflight + API + health + Streamlit (`powershell.exe -NoProfile -ExecutionPolicy Bypass -File`) |
| `demo_health_probe.py` | Exit 0 iff `/health` is current schema |
| `OPS.md` | Canonical commands + demo-day |
| `init_workspace.py` | Workspace dirs for packaging |
| `run_e1_matrix.py` / `evaluate_ms_sta_matrix.py` | Fair matrix + evaluation |
| `run_d4_acceptance_and_promote.py` | Demo acceptance gate |

Continuous Seq2Seq training/eval scripts have been removed from the repository.
