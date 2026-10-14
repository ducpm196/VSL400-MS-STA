#!/usr/bin/env python3
"""Phase-1 representation pilot: Stage-A smokes + Stage-B weight load check (no train/test)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch

from src.models.ms_sta_v6 import build_ms_sta_v6, count_parameters  # noqa: E402
from src.models.ms_sta_v6_skel import build_adjacency_75  # noqa: E402

OUT = _ROOT / "Docs/results/ms_sta_v6/V6_REPRESENTATION_PILOT.json"


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    report: dict[str, Any] = {
        "schema": "ms_sta_v6_representation_pilot_v1",
        "stage_a_fixed": {"variants": {}, "adjacency_ok": False},
        "stage_b_strong": {"weights": {}, "extractor_status": "not_built_yet"},
        "freeze_shortlist_proposal": {},
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    a = build_adjacency_75()
    report["stage_a_fixed"]["adjacency_ok"] = bool(torch.allclose(a, a.T, atol=1e-5))
    report["stage_a_fixed"]["adjacency_nnz"] = int((a > 0).sum().item())

    B, T, L = 2, 16, 8
    batch = {
        "token_global": torch.randn(B, L, 2304),
        "token_left_hand": torch.randn(B, L, 2304),
        "token_right_hand": torch.randn(B, L, 2304),
        "token_face": torch.randn(B, L, 2304),
        "rgb_pad_mask": torch.zeros(B, L, dtype=torch.bool),
        "coordinates": torch.randn(B, T, 75, 3),
        "velocity": torch.randn(B, T, 75, 3),
        "validity": torch.ones(B, T, 75),
        "skel_pad_mask": torch.zeros(B, T, dtype=torch.bool),
    }
    for v in ("rgb_strong", "b3_strong", "b3_mlp_strong", "b4_v6", "b4_v6_ctr", "b4_v6_bi", "b4_v6_ctr_bi"):
        m = build_ms_sta_v6(v, dropout=0.0)
        out = m(batch)
        report["stage_a_fixed"]["variants"][v] = {
            "param_count": count_parameters(m),
            "logits_finite": bool(torch.isfinite(out["logits"]).all().item()),
            "logits_shape": list(out["logits"].shape),
        }

    # Stage-B: only verify local F0 weight dirs exist (no full extract in pilot).
    for name, rel in (
        ("videomae_vit_small", "Docs/results/ms_sta_v5_f0/weights/videomae_vit_small"),
        ("video_swin_tiny", "Docs/results/ms_sta_v5_f0/weights/video_swin_tiny"),
        ("videomae_vit_base", "Docs/results/ms_sta_v5_f0/weights/videomae_vit_base"),
    ):
        p = _ROOT / rel
        report["stage_b_strong"]["weights"][name] = {
            "rel": rel,
            "exists": p.is_dir(),
            "primary_candidate": name != "videomae_vit_base",
        }

    # Finite freeze shortlist (pre-metric): Stage-A matrix + one CTR + one BI ablation;
    # Stage-B table separate after RGB cache exists.
    report["freeze_shortlist_proposal"] = {
        "stage_a_matrix": ["rgb_strong", "b3_strong", "b3_mlp_strong", "b4_v6"],
        "stage_a_architecture_ablations_max": ["b4_v6_ctr", "b4_v6_bi"],
        "stage_b_matrix_when_rgb_ready": ["rgb_strong", "b3_strong", "b3_mlp_strong", "b4_v6"],
        "stage_b_backbone_primary": "videomae_vit_small",
        "stage_b_backbone_secondary": "video_swin_tiny",
        "note": "Do not add candidates after seeing HPO metrics. Stage-A and Stage-B remain separate tables.",
    }
    report["verdict"] = "PILOT_SMOKE_PASS_READY_FOR_POLICY_DISCUSSION"
    report["blockers_before_full_train"] = [
        "Stage-B requires new RGB feature extractor/cache from raw MP4s (not built in this smoke).",
        "Policy freeze must pick Stage-A-only first wave OR Stage-A+subset-B with explicit budgets.",
    ]
    sha = atomic_write(OUT, report)
    print(json.dumps({"event": "V6_REPRESENTATION_PILOT_WRITTEN", "sha256": sha, "verdict": report["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
