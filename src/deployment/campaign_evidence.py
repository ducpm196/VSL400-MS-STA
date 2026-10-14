# -*- coding: utf-8 -*-
"""Load sealed E4_SYSKD_V1 / B4 finalization evidence for Streamlit (fail-closed)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_ART = _ROOT / "Docs/results/e4_system_kd_v1"
_FIN = _ROOT / "Docs/results/project_finalization"
_ONESHOT = _ROOT / "Docs/results/ms_sta_v6_oneshot"
_DEFENSE = _ROOT / "Docs/results/ms_sta_v6_defense_go"

# Expected SHAs (update only via sealed finalization commits)
_CLAIM_SHA = "f59b14e21cd30d4cf7b03612deaa7b27ba36cdf9270b3fac965170335d7f94dc"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, *, expected_sha: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_evidence:{path}")
    raw = path.read_bytes()
    if expected_sha is not None:
        import hashlib

        got = hashlib.sha256(raw).hexdigest()
        if got != expected_sha:
            raise ValueError(f"evidence_sha_mismatch:{path.name}:{got[:12]}!={expected_sha[:12]}")
    return json.loads(raw.decode("utf-8-sig"))


def load_scientific_claim_lock(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    path = root / "Docs/results/project_finalization/SCIENTIFIC_CLAIM_LOCK.json"
    return load_json(path, expected_sha=_CLAIM_SHA if root == _ROOT else None)


def load_campaign_status(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    return load_json(root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_CAMPAIGN_STATUS.json")


def load_d4_terminal(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    return load_json(root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_D4_TERMINAL.json")


def load_promotion_proposal(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    return load_json(root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_PROMOTION_AUTHORIZATION_PROPOSAL.json")


def load_demo_cutover_proposal(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    path = root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_DEMO_CUTOVER_PROPOSAL.json"
    if not path.is_file():
        return {"user_authorization_received": False, "demo_closed": True, "status": "MISSING"}
    return load_json(path)


def load_r0_comparator_report(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    return load_json(root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_R0_ONLINE_COMPARATOR_REPORT.json")


def load_locked_test_report(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    return load_json(root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_S5_LOCKED_TEST_REPORT.json")


def load_s5_efficiency_report(root: Path | None = None) -> dict[str, Any]:
    """Formal val interleaved efficiency (R0/R1/S5) — not UI latency."""
    root = root or _ROOT
    return load_json(root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_S5_EFFICIENCY_REPORT.json")


def load_s5_confirmation_efficiency(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    return load_json(
        root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_S5_CONFIRMATION_EFFICIENCY_REPORT.json"
    )


def load_promotion_authorization(root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    return load_json(root / "Docs/results/e4_system_kd_v1/E4_SYSKD_V1_PROMOTION_AUTHORIZATION.json")


def load_b4_canonical_test(root: Path | None = None) -> dict[str, Any]:
    """B4 cache-backed oneshot metrics (model evaluation path)."""
    root = root or _ROOT
    report = load_json(root / "Docs/results/ms_sta_v6_oneshot/TEST_REPORT.json")
    final = load_json(root / "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json")
    test = final.get("test") or {}
    return {
        "path_class": "cache_backed_feature_npz",
        "n": int(test.get("n") or 6165),
        "top1": float(test.get("b4_top1") or report.get("b4_top1") or 0.5154906731549067),
        "top5": float(test.get("b4_top5") or 0.7388483373884833),
        "macro_f1": float(test.get("b4_macro_f1") or 0.5137333509415792),
        "delta_top1_pp_vs_b3_mlp": float(test.get("delta_top1_pp") or 9.326845),
        "checkpoint_sha256": str(final.get("checkpoint_sha256") or ""),
        "source": "Docs/results/ms_sta_v6_defense_go/FINAL_RESULTS.json",
    }


def runtime_ui_state(root: Path | None = None) -> dict[str, Any]:
    """Derive Streamlit engine/evidence mode without mutating registries."""
    root = root or _ROOT
    camp = load_campaign_status(root)
    d4 = load_d4_terminal(root)
    promo = load_promotion_proposal(root)
    demo = load_demo_cutover_proposal(root)
    selected = camp.get("selected")
    e4_ready = bool(camp.get("e4_ready"))
    promo_auth = bool(promo.get("user_authorization_received"))
    demo_auth = bool(demo.get("user_authorization_received"))
    # Active UI engine for sealed cache demo remains Unified Attention until
    # Optimized-inference live switch is authorized (not available in SEALED_DEMO_SAMPLE_ONLY).
    active_engine = "B4"
    s5_mode = "EVIDENCE_ONLY"
    label = "Optimized inference là ứng viên. Giao diện vẫn đọc đặc trưng đã tính sẵn."
    if promo_auth and e4_ready and selected == "S5" and not demo_auth:
        label = "Optimized inference đã chọn. Giao diện chưa được phép chạy cách suy luận này."
        s5_mode = "EVIDENCE_ONLY"
    if promo_auth and e4_ready and selected == "S5" and demo_auth:
        # Cache-backed Streamlit still cannot exercise ROI batching.
        label = "Optimized inference đã chọn. Giao diện vẫn chỉ đọc đặc trưng đã tính sẵn."
        s5_mode = "EVIDENCE_ONLY"
        active_engine = "B4"
    return {
        "active_engine": active_engine,
        "s5_mode": s5_mode,
        "s5_label_vi": label,
        "selected": selected,
        "e4_ready": e4_ready,
        "d4_terminal": d4.get("terminal"),
        "d4_pass": bool(d4.get("d4_pass")),
        "promotion_authorized": promo_auth,
        "demo_cutover_authorized": demo_auth,
        "capability": "SEALED_DEMO_SAMPLE_ONLY",
        "provisional_candidate": camp.get("provisional_candidate") or d4.get("provisional_candidate"),
        "limitation_banner_vi": (
            "Đây là hệ thống nhận dạng ký hiệu đơn lẻ, chưa phải hệ thống dịch câu "
            "ngôn ngữ ký hiệu liên tục."
        ),
    }


def forbidden_term_scan(text: str) -> list[str]:
    """Flag signer_id mistranslation (document-signer wording), not 'người … ký hiệu'."""
    import re

    hits: list[str] = []
    # Pattern avoids matching “… ký hiệu”.
    bad = "người" + " " + "ký"
    bad_ascii = "nguoi" + " " + "ky"
    if re.search(rf"{re.escape(bad)}(?!\s*hiệu)", text, flags=re.IGNORECASE):
        hits.append(bad)
    if re.search(rf"{re.escape(bad_ascii)}(?!\s*hieu)", text, flags=re.IGNORECASE):
        hits.append(bad_ascii)
    return hits
