"""D4 candidate smoke on locked demo_set — does NOT set defense_ready/GO.

Requires E2-locked B4 checkpoint + registry e2_pass. Writes logs/gate_d4/D4_CANDIDATE_SMOKE.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.feature_versioning import load_json
from src.models.ms_sta_features import (
    FIXED_RGB_L,
    load_rgb_array,
    resolve_pooled_dir,
    resolve_skel_dir,
    resolve_temporal_dir,
    to_rgb_tensor,
    valid_mask_to_key_padding_mask,
)
from src.models.ms_sta_wordlevel import build_ms_sta

REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
DEMO = _ROOT / "data" / "processed" / "demo_set_ms_sta.json"
if not DEMO.is_file():
    DEMO = _ROOT / "data" / "processed" / "demo_set_option_b.json"
LABELS = _ROOT / "data" / "processed" / "vsl400_labels.json"
META = _ROOT / "metadata.csv"
OUT = _ROOT / "logs" / "gate_d4" / "D4_CANDIDATE_SMOKE.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    reg = load_json(REGISTRY)
    if reg.get("e2_pass") is not True or reg.get("evaluation_locked") is not True:
        print("[FAIL] D4 candidate requires e2_pass + evaluation_locked")
        return 2
    if reg.get("defense_ready") is True:
        print("[FAIL] refuse smoke while defense_ready=true (use defense path)")
        return 2
    if reg.get("status") != "NO-GO":
        print("[FAIL] candidate smoke expects status=NO-GO")
        return 2

    ckpt_dir = _ROOT / str(reg.get("ckpts_dir") or "data/processed/ms_sta_e2_lock")
    ckpt_p = ckpt_dir / Path(str(reg["checkpoint"])).name
    if _sha(ckpt_p) != reg.get("checkpoint_sha256"):
        print("[FAIL] checkpoint SHA mismatch")
        return 2
    payload = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    if payload.get("e1_selected") is not True or payload.get("evaluation_locked") is not True:
        print("[FAIL] payload not E2-locked")
        return 2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = load_json(LABELS)
    gloss_to_id = labels["gloss_to_id"]
    raw_id = labels["id_to_gloss"]
    if isinstance(raw_id, list):
        id_to_gloss = {i: g for i, g in enumerate(raw_id)}
    else:
        id_to_gloss = {int(k): v for k, v in raw_id.items()}

    variant = str(reg.get("active_variant") or "b4")
    model = build_ms_sta(variant, num_classes=int(labels["num_classes"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()

    import pandas as pd

    df = pd.read_csv(META)
    demo = load_json(DEMO)
    dirs = {
        "pooled": resolve_pooled_dir(_ROOT, reg),
        "temporal": resolve_temporal_dir(_ROOT, reg),
        "skel": resolve_skel_dir(_ROOT, reg),
    }

    rows = []
    with torch.no_grad():
        for vid in demo["video_ids"]:
            row = df[df["video_id"].astype(str) == str(vid)]
            if row.empty:
                rows.append({"video_id": vid, "error": "missing_metadata"})
                continue
            gloss = str(row.iloc[0]["text_gloss"])
            gold = gloss_to_id.get(gloss)
            rgb_p = dirs["temporal"] / f"{vid}.npy"
            skel_p = dirs["skel"] / f"{vid}.npy"
            if not rgb_p.is_file() or not skel_p.is_file() or gold is None:
                rows.append({"video_id": vid, "gold": gloss, "error": "missing_features_or_label"})
                continue
            rgb_t, valid = to_rgb_tensor(load_rgb_array(rgb_p), fixed_l=FIXED_RGB_L)
            rgb = rgb_t.unsqueeze(0).to(device)
            kpm = valid_mask_to_key_padding_mask(valid.unsqueeze(0)).to(device) if valid is not None else None
            sk = np.load(skel_p).astype(np.float32)
            if sk.shape[0] != 64:
                ix = np.linspace(0, sk.shape[0] - 1, 64).astype(int)
                sk = sk[ix]
            skel = torch.from_numpy(sk).unsqueeze(0).to(device)
            logits = model(rgb=rgb, skel=skel, rgb_key_padding_mask=kpm)
            pred = int(logits.argmax(dim=-1).item())
            top5 = [int(x) for x in logits.topk(5, dim=-1).indices.squeeze(0).tolist()]
            rows.append(
                {
                    "video_id": vid,
                    "gold": gloss,
                    "gold_id": gold,
                    "pred": id_to_gloss[pred],
                    "pred_id": pred,
                    "top5": [id_to_gloss[i] for i in top5],
                    "top1": pred == gold,
                    "top5_hit": gold in top5,
                }
            )

    ok = [r for r in rows if "error" not in r]
    out = {
        "schema": "option_b_d4_candidate_smoke_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "D4_CANDIDATE",
        "d4_pass": False,
        "defense_ready": False,
        "status": "NO-GO",
        "n_demo": len(demo["video_ids"]),
        "n_scored": len(ok),
        "top1": 100.0 * float(np.mean([r["top1"] for r in ok])) if ok else None,
        "top5": 100.0 * float(np.mean([r["top5_hit"] for r in ok])) if ok else None,
        "checkpoint": str(ckpt_p.relative_to(_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": reg.get("checkpoint_sha256"),
        "scenarios": demo.get("scenarios"),
        "rows": rows,
        "note": "Candidate smoke only. Not D4 PASS. Do not set defense_ready.",
    }
    out_path = args.out if args.out.is_absolute() else _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Docs mirror (logs may be gitignored)
    docs = _ROOT / "Docs" / "results" / "ms_sta_d4" / "D4_CANDIDATE_SMOKE.json"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "docs": str(docs), "top1": out["top1"], "top5": out["top5"], "d4_pass": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
