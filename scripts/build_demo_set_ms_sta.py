"""Rebuild demo_set_ms_sta.json scenarios from locked E2 rows (B4 vs B0-E1).

Fail-closed: refuses wrong E2 source SHA, missing/ambiguous ARTIFACT_MANIFEST E2 entry,
and broken scenario story. Does not retrain or re-run E2.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_E2 = _ROOT / "logs" / "eval_ms_sta" / "E2_b4_vs_b0e1.json"
_E2_CANON = _ROOT / "Docs" / "results" / "ms_sta_e2" / "E2_CANONICAL_REPORT.json"
_ARTIFACT_MANIFEST = _ROOT / "Docs" / "results" / "ms_sta_defense_go" / "ARTIFACT_MANIFEST.json"
_OUT = _ROOT / "data" / "processed" / "demo_set_ms_sta.json"
_LEGACY = _ROOT / "data" / "processed" / "demo_set_option_b.json"

# Locked E2 rows SHA (must match E2_CANONICAL_REPORT.source_artifact_sha256).
_EXPECTED_E2_SHA = "0e21abb940f80a49ed528c98d4c26e6c235dfde855f03128a28a214c2871d554"

# Defense-day scenarios locked from E2 tip evidence.
# Keep original Top-5 assist (018613); controlled_fail stays off the UI.
_SCENARIOS = {
    "top1_correct": "023292",  # signer 27 · CÁI_ĐÈN · B4 Top-1, B0 wrong
    "top5_assist": "018613",  # signer 24 · CHỊ · B4 Top-1=CÔ, GT in Top-5
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 2


def _expect(cond: bool, msg: str) -> str | None:
    return None if cond else msg


def main() -> int:
    if not _E2.is_file():
        return _fail(f"missing E2 rows: {_E2}")
    if not _E2_CANON.is_file():
        return _fail(f"missing E2 canonical report: {_E2_CANON}")

    got = _sha(_E2)
    canon = json.loads(_E2_CANON.read_text(encoding="utf-8-sig"))
    expect = str(canon.get("source_artifact_sha256") or _EXPECTED_E2_SHA)
    if got != expect:
        return _fail(f"E2 rows sha mismatch disk={got[:12]} expect={expect[:12]} (canonical report)")
    if expect != _EXPECTED_E2_SHA:
        return _fail(
            f"canonical source_artifact_sha256 drifted: {expect[:12]} != locked {_EXPECTED_E2_SHA[:12]}"
        )

    if not _ARTIFACT_MANIFEST.is_file():
        return _fail(f"missing ARTIFACT_MANIFEST: {_ARTIFACT_MANIFEST}")
    man = json.loads(_ARTIFACT_MANIFEST.read_text(encoding="utf-8-sig"))
    e2_entries = [
        item
        for item in (man.get("items") or [])
        if isinstance(item, dict)
        and str(item.get("path") or "").replace("\\", "/").endswith("E2_b4_vs_b0e1.json")
    ]
    if len(e2_entries) != 1:
        return _fail(
            f"ARTIFACT_MANIFEST must contain exactly one E2_b4_vs_b0e1.json entry, found {len(e2_entries)}"
        )
    man_sha = str(e2_entries[0].get("sha256") or "")
    if man_sha != _EXPECTED_E2_SHA:
        return _fail(
            f"ARTIFACT_MANIFEST E2 sha mismatch manifest={man_sha[:12] or 'missing'} "
            f"expect={_EXPECTED_E2_SHA[:12]}"
        )
    if man_sha != got:
        return _fail(
            f"ARTIFACT_MANIFEST E2 sha mismatch manifest={man_sha[:12]} disk={got[:12]}"
        )

    data = json.loads(_E2.read_text(encoding="utf-8-sig"))
    rows = data.get("rows")
    if not isinstance(rows, list):
        return _fail("E2 JSON missing rows[]")
    by_id = {str(r["video_id"]): r for r in rows if isinstance(r, dict) and "video_id" in r}

    scenarios: dict = {}
    for role, vid in _SCENARIOS.items():
        row = by_id.get(vid)
        if row is None:
            return _fail(f"scenario {role} video_id={vid} missing from E2")
        scenarios[role] = {
            "video_id": vid,
            "signer_id": row.get("signer_id"),
            "gold": row.get("gold"),
            "b0_pred": row.get("b0_pred"),
            "b0_top1": row.get("b0_top1"),
            "b0_top5_hit": row.get("b0_top5_hit"),
            "b4_pred": row.get("b4_pred"),
            "b4_top1": row.get("b4_top1"),
            "b4_top5_hit": row.get("b4_top5_hit"),
            "role": role,
        }

    checks = [
        _expect(scenarios["top1_correct"]["b4_top1"] is True, "top1_correct requires b4_top1=True"),
        _expect(scenarios["top1_correct"]["b0_top1"] is False, "top1_correct requires b0_top1=False"),
        _expect(scenarios["top5_assist"]["b4_top1"] is False, "top5_assist requires b4_top1=False"),
        _expect(scenarios["top5_assist"]["b4_top5_hit"] is True, "top5_assist requires b4_top5_hit=True"),
    ]
    for err in checks:
        if err:
            return _fail(err)

    pack = {
        "schema": "ms_sta_demo_set_v2",
        "gate": "D4",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "Defense demo scenarios locked from E2 B4 vs B0-E1 rows (test split). "
            "Not a KPI; live Top-5 lists come from cache-backed inference."
        ),
        "e2_source": "logs/eval_ms_sta/E2_b4_vs_b0e1.json",
        "e2_source_sha256": got,
        "e2_canonical_report": "Docs/results/ms_sta_e2/E2_CANONICAL_REPORT.json",
        "n": len(_SCENARIOS),
        "video_ids": [scenarios[k]["video_id"] for k in _SCENARIOS],
        "scenarios": {k: v["video_id"] for k, v in scenarios.items()},
        "scenario_rows": scenarios,
        "talking_points": {
            "top1_correct": "B4 Top-1 recovers CÁI_ĐÈN where B0-E1 predicted MŨ_BẢO_HIỂM.",
            "top5_assist": "Top-1 is CÔ while GT is CHỊ; GT remains in Top-5 — shortlist for human confirm.",
        },
        "note": "Original top5_assist 018613 restored. controlled_fail stays off defense UI.",
    }
    text = json.dumps(pack, ensure_ascii=False, indent=2) + "\n"
    _OUT.write_text(text, encoding="utf-8")
    _LEGACY.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(_OUT.relative_to(_ROOT)),
                "sha256": _sha(_OUT),
                "e2_source_sha256": got,
                "scenarios": pack["scenarios"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
