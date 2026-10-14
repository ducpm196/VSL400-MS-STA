#!/usr/bin/env python3
"""Refresh RELEASE_AUDIT after E4-R3 packaging (no registry/ckpt mutation)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

DOC_RELS = [
    "README.md",
    "Docs/README.md",
    "Docs/B4_CURRENT_REPORT.md",
    "Docs/Research_Overview.md",
    "Docs/Strategy_Implementation.md",
    "Docs/MSSTA_END_TO_END_PIPELINE.md",
    "Docs/MSSTA_VSL400_Lock.md",
    "Docs/REPRODUCIBILITY.md",
    "Docs/phases/03_MS-STA.md",
    "Docs/phases/04_Deployment.md",
    "Docs/phases/05_Evaluation.md",
    "Docs/references/VSL400_SOURCE_AUDIT.md",
    "scripts/OPS.md",
    "Docs/MS-STA_VSL400_Presentation.html",
    "Docs/MS-STA_VSL400_Presentation_offline.pdf",
    "Docs/MS-STA_VSL400_Presentation_offline.provenance.json",
    "Docs/results/ms_sta_v6_defense_go/B4_ACCURACY_PAGE.html",
    "Docs/results/ms_sta_v6_defense_go/E4_EFFICIENCY_APPENDIX.html",
    "src/deployment/app.py",
    "src/deployment/backend_api.py",
    "src/deployment/pages/1_B4_V6_Accuracy.py",
    "src/deployment/pages/2_E4_Efficiency_Appendix.py",
    "src/deployment/input_evidence.py",
    "src/deployment/input_evidence_ui.py",
    "src/deployment/demo_visuals.py",
]


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    audit_path = _ROOT / "Docs/results/ms_sta_v6_defense_go/RELEASE_AUDIT.json"
    a = json.loads(audit_path.read_text(encoding="utf-8"))
    a["recorded_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    a["narrative"] = (
        "B4-V6 remains the proposed defense model. Historical e4_w192 remains E4_ACCURACY_NO_GO. "
        "E4-R2 remains E4_R2_ACCURACY_NO_GO. E4-R3 closed as E4_R3_FEASIBILITY_NO_GO "
        "(token/param reduction; no arm ≥8% model-forward speedup vs B4); "
        "HPO/confirm/efficiency/oneshot/D4 not opened; e4_ready=false. "
        "E4-R3 packaging is an NO-GO closeout (not D4/FINAL)."
    )

    feas = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json"
    companion = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_PROVENANCE_COMPANION.json"
    idx = _ROOT / "Docs/results/ms_sta_v6_e4_r3/EVIDENCE_INDEX.json"
    nogo = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json"
    close = _ROOT / "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json"
    status = _ROOT / "Docs/results/ms_sta_v6_e4_r3/CAMPAIGN_STATUS.json"

    a["bindings"]["e4_r3_feasibility_rel"] = "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json"
    a["bindings"]["e4_r3_feasibility_sha256"] = sha(feas)
    a["bindings"]["e4_r3_feasibility_provenance_companion_rel"] = (
        "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_PROVENANCE_COMPANION.json"
    )
    a["bindings"]["e4_r3_feasibility_provenance_companion_sha256"] = sha(companion)
    a["bindings"]["e4_r3_evidence_index_rel"] = "Docs/results/ms_sta_v6_e4_r3/EVIDENCE_INDEX.json"
    a["bindings"]["e4_r3_evidence_index_sha256"] = sha(idx)
    a["bindings"]["e4_r3_feasibility_no_go_rel"] = "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json"
    a["bindings"]["e4_r3_feasibility_no_go_sha256"] = sha(nogo)
    a["bindings"]["e4_r3_closeout_rel"] = (
        "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json"
    )
    a["bindings"]["e4_r3_closeout_sha256"] = sha(close)
    a["bindings"]["e4_r3_campaign_status_rel"] = "Docs/results/ms_sta_v6_e4_r3/CAMPAIGN_STATUS.json"
    a["bindings"]["e4_r3_campaign_status_sha256"] = sha(status)

    a["identity_checks"]["e4_ready"] = False
    a["identity_checks"]["e4_r3_ready"] = False
    a["metrics_sot"]["e4_r3"] = "E4_R3_FEASIBILITY_NO_GO"
    a["metrics_sot"]["e4_r3_best_arm_speedup"] = 0.019394073264405787
    a["metrics_sot"]["e4_r3_speedup_floor"] = 0.08

    docs = {}
    for rel in DOC_RELS:
        docs[rel] = sha(_ROOT / rel)
    a["document_html_pdf_shas"] = docs

    qa = _ROOT / "Docs/results/ms_sta_v6_defense_go/qa"
    a["visual_qa_screenshots"] = {
        x.name: sha(x) for x in sorted(qa.glob("*.png")) if x.is_file()
    }

    a["gates"]["e4_r3_feasibility_nogo_packaging"] = (
        "PASS (FEASIBILITY_NO_GO closeout + provenance companion; e4_ready=false; B4 primary)"
    )
    a["gates"]["e4_r3_demo_sync"] = (
        "PASS (E4 page leads with R3 latency/param charts; R2 under prior research)"
    )
    a["gates"]["streamlit_apptest_e4"] = "PASS (AppTest + live HTTP 200)"
    a["gates"]["visual_qa_desktop_mobile"] = "PASS (E4 desktop+mobile refreshed for R3)"

    claims = list(a.get("claim_limits") or [])
    extra = [
        "E4-R3 is FEASIBILITY_NO_GO; not an optional efficiency engine; no faster/real-time/production claim.",
        "E4-R3 NO-GO closeout merge is not D4/FINAL.",
        "Feasibility microbench is screening-only; companion is not clean-run identity.",
    ]
    for c in extra:
        if c not in claims:
            claims.append(c)
    a["claim_limits"] = claims

    a["commits"] = {
        "e4_r3_research_closeout": "4709a9453be963584867b037cb83dde0df2e54ec",
        "e4_r3_nogo_closeout_merge": "30919fc13071d1d9d7bb41cf7cbbce7f9138c1c6",
        "merge_kind": "E4-R3_NO-GO_closeout_merge",
        "not_d4_final": True,
        "packaging_tip": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip(),
    }

    a["e4_r3_appendix"] = {
        "status": "E4_R3_FEASIBILITY_NO_GO",
        "feasibility_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY.json",
        "provenance_companion_rel": (
            "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_PROVENANCE_COMPANION.json"
        ),
        "no_go_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_FEASIBILITY_NO_GO.json",
        "closeout_rel": "Docs/results/ms_sta_v6_e4_r3/E4_R3_CLOSEOUT_FEASIBILITY_NO_GO.json",
        "campaign_status_rel": "Docs/results/ms_sta_v6_e4_r3/CAMPAIGN_STATUS.json",
        "canonical_clean_execution_identity": False,
        "not_registered": True,
        "b4_remains_primary": True,
        "hpo_opened": False,
        "confirm_opened": False,
        "efficiency_opened": False,
        "oneshot_opened": False,
        "d4_opened": False,
        "display_table_ms": {"B4": 12.47, "A": 12.24, "B": 13.23, "C": 13.01},
        "display_speedup_pct": {"A": 1.9, "B": -5.7, "C": -4.1},
    }

    a["comparison_tables"]["e4_page"] = "src/deployment/pages/2_E4_Efficiency_Appendix.py"
    a["comparison_tables"]["e4_page_sha256"] = sha(
        _ROOT / "src/deployment/pages/2_E4_Efficiency_Appendix.py"
    )
    a["comparison_tables"]["demo_visuals"] = "src/deployment/demo_visuals.py"
    a["comparison_tables"]["demo_visuals_sha256"] = sha(
        _ROOT / "src/deployment/demo_visuals.py"
    )
    a["comparison_tables"]["e4_latency_claim"] = (
        "E4-R3 feasibility screening: B4 12.47ms; A +1.9%; B -5.7%; C -4.1%; "
        "floor +8% FAIL; not canonical efficiency"
    )
    a["comparison_tables"]["e4_no_canonical_efficiency"] = True

    # Keep e4_r2_appendix intact.
    raw = (json.dumps(a, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = audit_path.with_suffix(".json.tmp")
    tmp.write_bytes(raw)
    tmp.replace(audit_path)
    print(json.dumps({"release_audit_sha256": hashlib.sha256(raw).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
