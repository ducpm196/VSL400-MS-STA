"""Negative + lifecycle tests for E4-R2 campaign STATUS."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.ms_sta_e4_r2_status import (
    STALE_HPO_CLOSEOUT,
    assert_no_live_confirm_with_stale_closeout,
    build_campaign_status,
    write_campaign_status,
)


def test_live_confirm_cannot_coexist_with_stale_hpo_closeout() -> None:
    with pytest.raises(RuntimeError, match="live_confirm_coexists_with_stale_hpo_closeout"):
        assert_no_live_confirm_with_stale_closeout(
            top_level={"status": STALE_HPO_CLOSEOUT, "phase": "closeout"},
            confirm={"status": "RUNNING", "phase": "confirm_remediated"},
        )


def test_live_confirm_ok_when_top_level_superseded() -> None:
    assert_no_live_confirm_with_stale_closeout(
        top_level={
            "status": "REMEDIATION_CONFIRM_RUNNING",
            "supersedes": {"status": STALE_HPO_CLOSEOUT},
        },
        confirm={"status": "RUNNING"},
    )


def test_stale_closeout_ok_when_confirm_idle() -> None:
    assert_no_live_confirm_with_stale_closeout(
        top_level={"status": STALE_HPO_CLOSEOUT},
        confirm={"status": "CONFIRM_COMPLETE"},
    )


def test_write_campaign_status_preserves_historical_and_supersedes(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    sealed = tmp_path / "sealed"
    prev = {"status": STALE_HPO_CLOSEOUT, "no_go_rel": "old.json"}
    payload = build_campaign_status(
        status="E4_R2_ACCURACY_NO_GO",
        phase="accuracy_closeout",
        git_revision="abc",
        prev=prev,
        extra={"accuracy_no_go_rel": "Docs/results/ms_sta_v6_e4_r2/E4_R2_ACCURACY_NO_GO.json"},
    )
    shas = write_campaign_status(
        runs_root=runs,
        sealed_rel_root=sealed,
        payload=payload,
        confirm={"status": "CONFIRM_COMPLETE"},
    )
    assert "runtime" in shas and "sealed" in shas
    written = (runs / "STATUS.json").read_text(encoding="utf-8")
    assert "E4_R2_ACCURACY_NO_GO" in written
    assert "E4_R2_NO_GO.json" in written
    assert "supersedes" in written
    assert STALE_HPO_CLOSEOUT in written


def test_refuse_write_stale_status_while_confirm_running(tmp_path: Path) -> None:
    payload = build_campaign_status(
        status=STALE_HPO_CLOSEOUT,
        phase="closeout",
    )
    # build_campaign_status with STALE does not add supersedes; status stays stale
    payload["status"] = STALE_HPO_CLOSEOUT
    payload.pop("supersedes", None)
    with pytest.raises(RuntimeError, match="live_confirm_coexists_with_stale_hpo_closeout"):
        write_campaign_status(
            runs_root=tmp_path / "runs",
            sealed_rel_root=None,
            payload=payload,
            confirm={"status": "RUNNING"},
        )
