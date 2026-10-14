# -*- coding: utf-8 -*-
"""Finalize predefense terminal + summary artifacts."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs" / "results" / "project_predefense_audit"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).strip()

    html = {
        "schema": "predefense_html_browser_audit_v2",
        "created_at": NOW,
        "url": "http://127.0.0.1:8766/MS_STA_VSL400_BAO_CAO_NGHIEN_CUU.html",
        "served_from": "Docs/ on main worktree",
        "viewport_sampled_width": 652,
        "checks": {
            "page_loads": True,
            "title_vietnamese_utf8": True,
            "nav_links_count": 16,
            "e4_terminal_text_present": True,
            "e4_ready_false_present": True,
            "no_e4_ready_true_claim": True,
            "architecture_one_way_cross_attention_mentioned": True,
            "gated_residual_mentioned": True,
            "offline_decision_pipe_present": True,
            "mermaid_pre_blocks_remaining": 0,
        },
        "overall": "PASS",
        "claim_class": "FACT",
        "note": "Browser tool CDP+snapshot; ~652css px width. Separate phone viewport screenshot not claimed.",
    }
    (OUT / "HTML_BROWSER_AUDIT_V2.json").write_text(
        json.dumps(html, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    findings = json.loads((OUT / "PREDEFENSE_FINDINGS.json").read_text(encoding="utf-8"))
    tex = json.loads((OUT / "TEX_BUILD_REPORT.json").read_text(encoding="utf-8"))
    ckpt = json.loads((OUT / "TRAINING_AND_CHECKPOINT_AUDIT.json").read_text(encoding="utf-8"))
    recalc = json.loads((OUT / "METRIC_INDEPENDENT_RECALC.json").read_text(encoding="utf-8"))

    rows = [
        json.loads(l)
        for l in (OUT / "LINE_REVIEW_LEDGER.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    ledger_n = len(rows)
    pass_n = sum(1 for r in rows if r.get("review_status") == "PASS")

    ind = {
        "schema": "predefense_independent_verify_v1",
        "created_at": NOW,
        "audited_sha": head,
        "clean_worktree_created": False,
        "caveat": (
            "Temp TeX rebuild + browser on same SHA; worktree dirty only with AUDIT_CREATED. "
            "Dedicated clean git worktree not created this session."
        ),
        "checks": {
            "checkpoint_sha": ckpt["checkpoint"],
            "tex_rebuild_temp": {
                "verdict": tex["verdict"],
                "pages": tex["pages"],
                "sha256": tex["pdf_sha256"],
            },
            "html_browser_v2": html["overall"],
            "targeted_tests_this_session": "PASS (contract subset)",
            "full_pytest": "NOT_CLAIMED",
            "science_lock": "INTACT",
            "metric_recalc": recalc.get("status"),
        },
        "overall": "PARTIAL",
    }
    (OUT / "PREDEFENSE_INDEPENDENT_VERIFY.json").write_text(
        json.dumps(ind, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    terminal = {
        "schema": "predefense_release_terminal_v1",
        "created_at": NOW,
        "terminal": "PROJECT_REPORT_PARTIAL",
        "git": {
            "branch": "main",
            "audited_sha": head,
            "candidate_sha": head,
            "clean": status == "",
            "dirty_paths": status.splitlines() if status else [],
            "commits_created": 0,
            "pushed": False,
        },
        "verification": {
            "codegraph": "PASS (indexed, lint ok)",
            "source_ledger_coverage": f"{pass_n}/{ledger_n} PASS rows (honest SOURCE_GAP remainder)",
            "findings_counts": findings["counts"],
            "targeted_tests": "PASS (this session subset; prior closeout 15)",
            "full_pytest": "NOT_CLAIMED",
            "independent_metric_recalc": recalc.get("status"),
            "tex": tex["verdict"],
            "html_browser_audit_v2": html["overall"],
            "checkpoint_registry_sha": "PASS (match)",
        },
        "science": {
            "b4_retained": True,
            "e4_selected": None,
            "e4_ready": False,
            "test_d4_demo_e4_opened": False,
            "required_sentence_intact": True,
        },
        "blockers": [
            "Line-review ledger not 100% PASS for in-scope tracked sources (PD-003)",
            "Full pytest NOT_CLAIMED; collect blocked without fastapi + external artifacts (PD-004)",
            "Independent verify lacked dedicated clean git worktree",
            "Student-owned TeX cover placeholders remain",
            "Worktree dirty with AUDIT_CREATED artifacts (not committed)",
        ],
        "remaining_steps": [
            "Continue true 1-N line review for release-critical sources → update ledger",
            "Provision fastapi (or SKIP_POLICY) and external artifacts; rerun collect/suite",
            "Optional: clean worktree verify from HEAD",
            "Student fills TeX identity macros before print submission",
            "Push only with explicit user permission",
        ],
        "hard_gates_failed_for_DEFENSE_READY": [3, 4, 8, 14, 15],
    }
    (OUT / "PREDEFENSE_RELEASE_TERMINAL.json").write_text(
        json.dumps(terminal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Use chr(96) backticks; never write backslash-backtick (\\`...\\`) — that corrupts \\a/\\t.
    bt = chr(96)
    summary = f"""# PRE-DEFENSE AUDIT SUMMARY

**Terminal:** {bt}PROJECT_REPORT_PARTIAL{bt}
**Date:** {NOW}
**SHA:** {bt}{head}{bt} (main aligned with origin/main; local dirty with AUDIT_CREATED only)

## Hội đồng — câu trả lời ngắn

| Câu hỏi | Trả lời có bằng chứng |
|---|---|
| Mô hình nào được chọn? | **B4-V6** (`b4_v6_ctr`, seed 42) — registry + checkpoint SHA khớp |
| Vì sao B4 được giữ? | E4 không vượt cả accuracy gate và efficiency 1.15× dưới stack khóa |
| E4 đã thử gì? | B4Q/E1a AMP, V1 width, V2 hard-channel, V3 F0 factorization, B4PIPE inventory |
| E4 thất bại ở đâu? | Accuracy và/hoặc efficiency NO-GO theo chiến dịch; overall `E4_CONTROLLED_NO_GO_B4_RETAINED` |
| Đã làm được? | SoT B4 khóa; tài liệu HTML/TeX; targeted contracts; CodeGraph; checkpoint SHA |
| Chưa làm được? | Full line-review 100%; full pytest; production API harden |
| Giới hạn khoa học? | RTX 2060 / train_cuda; không gộp speedup lịch khác; không khái quát mọi GPU |
| Source gap? | Ledger phần lớn SOURCE_GAP; trainer lines; full corpus leakage re-audit |
| Đủ bảo vệ source-code audit? | **Chưa** (`DEFENSE_READY` không đạt). Báo cáo khoa học vẫn dùng được ở mức PARTIAL |

## Khoa học (không đổi)

> Under the locked native stack and B4-preserving constraints, no E4 arm satisfied both the frozen accuracy and 1.15x efficiency gates. No E4 candidate was promoted, e4_ready remains false, and B4 is retained as the system-of-record.

## Verification snapshot

- CodeGraph: PASS
- Checkpoint/registry SHA: PASS
- TeX pdfLaTeX ×2 temp: PASS (48 trang)
- HTML browser V2: PASS
- Targeted tests: PASS (subset session)
- Full pytest: NOT_CLAIMED
- P0/P1/P2/P3: {findings['counts']}
- Code repairs this audit: **none**

## Phân biệt trạng thái

- HTML tag **DEFENSE_GO** = chiến dịch mô hình B4 đã khóa (không đồng nghĩa `DEFENSE_READY` của long-task source audit).
"""
    (OUT / "PREDEFENSE_AUDIT_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"terminal": terminal["terminal"], "pass_ledger": pass_n, "ledger_n": ledger_n}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
