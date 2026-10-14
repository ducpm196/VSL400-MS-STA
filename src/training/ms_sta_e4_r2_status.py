"""E4-R2 campaign STATUS lifecycle (top-level vs confirm).

Runtime SoT for live jobs is confirm/STATUS.json + process tree.
Top-level STATUS.json under runs_canonical is campaign lifecycle only.
Stale HPO_NO_SELECT closeout must not remain authoritative once remediation
confirm or accuracy closeout supersedes it — without deleting historical files.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

STALE_HPO_CLOSEOUT = "E4_R2_HPO_NO_SELECT"
LIVE_CONFIRM = frozenset({"RUNNING"})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_write_json(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def assert_no_live_confirm_with_stale_closeout(
    *,
    top_level: dict[str, Any] | None,
    confirm: dict[str, Any] | None,
) -> None:
    """Negative invariant: live confirm must not coexist with stale HPO closeout.

    Stale = top-level ``status`` is still ``E4_R2_HPO_NO_SELECT`` (authoritative).
    Live = confirm ``status`` is ``RUNNING``. Historical evidence files may remain.
    """
    if not confirm or confirm.get("status") not in LIVE_CONFIRM:
        return
    if not top_level:
        return
    if str(top_level.get("status") or "") == STALE_HPO_CLOSEOUT:
        raise RuntimeError("live_confirm_coexists_with_stale_hpo_closeout_status")


def historical_evidence_block(prev: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preserve pointers to invalid-comparator packaging; never delete those files."""
    base = {
        "invalid_comparator_no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_NO_GO.json",
        "invalid_comparator_closeout_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_CLOSEOUT.json",
        "invalid_comparator_companion_rel": (
            "Docs/results/ms_sta_v6_e4_r2/E4_R2_NO_GO_COMPANION_INVALID_COMPARATOR.json"
        ),
        "note": "Historical HPO_NO_SELECT packaging retained; superseded by remediation audit.",
    }
    if prev and isinstance(prev.get("historical_evidence"), dict):
        merged = dict(prev["historical_evidence"])
        merged.update(base)
        return merged
    return base


def build_campaign_status(
    *,
    status: str,
    phase: str,
    git_revision: str | None = None,
    executable_tree_sha256: str | None = None,
    prev: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "ms_sta_v6_e4_r2_campaign_status_v1",
        "campaign": "E4-R2",
        "phase": phase,
        "status": status,
        "canonical": True,
        "e4_ready": False,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "historical_evidence": historical_evidence_block(prev),
    }
    if status != STALE_HPO_CLOSEOUT:
        payload["supersedes"] = {
            "status": STALE_HPO_CLOSEOUT,
            "reason": "REMEDIATION_AFTER_INVALID_HPO_COMPARATOR",
            "historical_no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_NO_GO.json",
            "historical_closeout_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_CLOSEOUT.json",
        }
    if git_revision:
        payload["git_revision"] = git_revision
    if executable_tree_sha256:
        payload["executable_tree_sha256"] = executable_tree_sha256
    if extra:
        payload.update(extra)
    return payload


def write_campaign_status(
    *,
    runs_root: Path,
    sealed_rel_root: Path | None,
    payload: dict[str, Any],
    confirm: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write runtime STATUS under runs_root and optional sealed copy for git."""
    assert_no_live_confirm_with_stale_closeout(top_level=payload, confirm=confirm)
    shas: dict[str, str] = {}
    runtime_path = runs_root / "STATUS.json"
    shas["runtime"] = atomic_write_json(runtime_path, payload)
    if sealed_rel_root is not None:
        sealed_path = sealed_rel_root / "CAMPAIGN_STATUS.json"
        shas["sealed"] = atomic_write_json(sealed_path, payload)
    return shas
