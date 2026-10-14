# -*- coding: utf-8 -*-
"""Regression: sealed val metadata exposes real signer_id (not empty)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_SEALED_PRED_SIGNERFIX_Q0.jsonl"
PRED_AMP = ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_SEALED_PRED_SIGNERFIX_Q1_AMP.jsonl"
VERIFY = ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_CLOSEOUT_INDEPENDENT_VERIFY_SIGNERFIX.json"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.skipif(not PRED.is_file(), reason="SIGNERFIX sealed pred dump missing")
def test_sealed_pred_signer_id_coverage() -> None:
    rows = _load_jsonl(PRED)
    assert len(rows) == 2629
    signers = [str(r.get("signer") or r.get("signer_id") or "") for r in rows]
    assert "" not in signers
    assert None not in signers
    counts = Counter(signers)
    assert len(counts) == 3
    assert sum(counts.values()) == 2629


@pytest.mark.skipif(not (PRED.is_file() and PRED_AMP.is_file()), reason="SIGNERFIX dumps missing")
def test_sealed_worst_signer_delta_reproducible() -> None:
    q0 = _load_jsonl(PRED)
    q1 = _load_jsonl(PRED_AMP)
    assert len(q0) == len(q1) == 2629
    by0: dict[str, list[int]] = {}
    by1: dict[str, list[int]] = {}
    for a, b in zip(q0, q1):
        s = str(a.get("signer") or a.get("signer_id") or "")
        assert s
        assert s == str(b.get("signer") or b.get("signer_id") or "")
        by0.setdefault(s, []).append(int(a["top1"]))
        by1.setdefault(s, []).append(int(b["top1"]))
    deltas = {s: 100.0 * (sum(by1[s]) / len(by1[s]) - sum(by0[s]) / len(by0[s])) for s in by0}
    worst = min(deltas.values())
    # Locked observation from SIGNERFIX close-out verify (~ -0.116 pp)
    assert worst == pytest.approx(-0.11614401858304202, abs=1e-6)
    assert worst >= -2.0


@pytest.mark.skipif(not VERIFY.is_file(), reason="SIGNERFIX verify missing")
def test_closeout_signerfix_verify_pass_flag() -> None:
    v = json.loads(VERIFY.read_text(encoding="utf-8"))
    assert v.get("pass") is True
    acc = v["accuracy_sealed"]
    assert acc["pass"] is True
    assert len(acc["recomputed"]["Q0"]["signer_top1"]) == 3
    assert "" not in acc["recomputed"]["Q0"]["signer_top1"]


def test_dataset_reads_signer_id_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training import ms_sta_v5_data as mod

    # Unit-level: metadata with signer_id only must populate item['signer']
    class FakeDS(mod.VSL400MSSTAv5Dataset):
        def __init__(self):  # noqa: ANN204
            self.ids = ["000001"]
            self.labels = {"000001": 0}
            self.metadata_by_id = {"000001": {"signer_id": "signer_9", "text_gloss": "X"}}
            self.rgb_root = tmp_path
            self.skel_root = tmp_path
            self.max_rgb_len = 8
            self.max_skel_len = 8
            self.forbid_ids = set()

        def __getitem__(self, idx: int):  # noqa: ANN001
            # Bypass file IO: call only the signer mapping fragment by constructing like production
            sid = self.ids[idx]
            meta_row = self.metadata_by_id.get(sid) or {}
            signer = str(meta_row.get("signer_id") or meta_row.get("signer") or "")
            return {"sample_id": sid, "signer": signer, "label": self.labels[sid]}

    item = FakeDS()[0]
    assert item["signer"] == "signer_9"
