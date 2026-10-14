# -*- coding: utf-8 -*-
"""E4-R4C full-corpus quality scan (create-only report under Docs/results)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.e4_r4_joint_layouts import layout_indices
from src.data_prep.e4_r4_rgb_contract import require_rgb_candidate

POLICY = ROOT / "Docs/results/ms_sta_v6_e4_r4_policy/E4_R4C_POLICY.json"
OUT = ROOT / "Docs/results/ms_sta_v6_e4_r4"


def die(msg: str) -> None:
    print(f"E4_R4C_QUALITY_FAIL: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_stems(dir_path: Path) -> set[str]:
    # os.listdir is much faster than Path.glob on large Windows dirs.
    out: set[str] = set()
    for name in os.listdir(dir_path):
        if name.endswith(".npz"):
            out.add(name[:-4])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()
    root = args.feature_root if args.feature_root.is_absolute() else (ROOT / args.feature_root)
    print("quality_scan_start", root, flush=True)

    contract_path = root / "_run_contract.json"
    if not contract_path.is_file():
        die("missing_run_contract")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    eligible = json.loads((ROOT / policy["eligible_manifest_rel"]).read_text(encoding="utf-8"))
    ids = sorted(set(str(x) for x in eligible["eligible_ids"]))
    id_set = set(ids)
    if contract.get("n_ids") != len(ids):
        die(f"n_ids_mismatch:{contract.get('n_ids')}!={len(ids)}")
    ids_sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if contract.get("ids_sha256") != ids_sha:
        die("ids_sha_mismatch")

    rgb_dir = root / "rgb_R1"
    skel_dir = root / "skel_S_A"
    print("listing_rgb...", flush=True)
    t0 = time.perf_counter()
    rgb_ids = npz_stems(rgb_dir)
    print(f"listed_rgb n={len(rgb_ids)} sec={time.perf_counter()-t0:.1f}", flush=True)
    print("listing_skel...", flush=True)
    t0 = time.perf_counter()
    skel_ids = npz_stems(skel_dir)
    print(f"listed_skel n={len(skel_ids)} sec={time.perf_counter()-t0:.1f}", flush=True)

    if len(rgb_ids) != len(ids) or len(skel_ids) != len(ids):
        die(f"count_mismatch:rgb={len(rgb_ids)} skel={len(skel_ids)} expected={len(ids)}")
    if rgb_ids != id_set:
        miss = sorted(id_set - rgb_ids)[:5]
        extra = sorted(rgb_ids - id_set)[:5]
        die(f"rgb_id_set_mismatch:miss={miss}:extra={extra}")
    if skel_ids != id_set:
        miss = sorted(id_set - skel_ids)[:5]
        extra = sorted(skel_ids - id_set)[:5]
        die(f"skel_id_set_mismatch:miss={miss}:extra={extra}")

    spec = require_rgb_candidate("R1")
    j = len(layout_indices("S_A"))
    t_max = int(spec["frames_T_max"])
    ch, cw = spec["canvas_spatial"]
    errors = []
    t_scan = time.perf_counter()
    for i, sid in enumerate(ids, 1):
        try:
            with np.load(rgb_dir / f"{sid}.npz", allow_pickle=False) as z:
                ten = z["rgb_tensor"]
                mask = np.asarray(z["temporal_valid"], dtype=bool)
                if ten.dtype != np.float32 or ten.shape != (t_max, int(ch), int(cw), 3):
                    raise RuntimeError(f"rgb_shape_dtype:{ten.shape}:{ten.dtype}")
                if np.isnan(ten[mask]).any() or np.isinf(ten[mask]).any():
                    raise RuntimeError("rgb_nan_inf")
                if not mask.any():
                    raise RuntimeError("rgb_all_invalid")
                if "raw_video_sha256" not in z.files or "sealed_skel_sha256" not in z.files:
                    raise RuntimeError("missing_sha_binds")
            with np.load(skel_dir / f"{sid}.npz", allow_pickle=False) as z:
                coords = z["coordinates"]
                if coords.dtype != np.float32 or coords.ndim != 3 or coords.shape[1] != j or coords.shape[2] != 3:
                    raise RuntimeError(f"skel_shape:{coords.shape}")
                if np.isnan(coords).any() or np.isinf(coords).any():
                    raise RuntimeError("skel_nan_inf")
                if str(z["source_mode"]) != "sealed_75joint_read_only_compact_transform":
                    raise RuntimeError("bad_source_mode")
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": sid, "error": repr(exc)})
            if len(errors) >= 20:
                break
        if args.progress_every and i % args.progress_every == 0:
            rate = i / max(time.perf_counter() - t_scan, 1e-9)
            print(f"scan {i}/{len(ids)} rate={rate:.1f}/s errors={len(errors)}", flush=True)

    status = "PASS" if not errors else "FAIL"
    report = {
        "schema": "ms_sta_v6_e4_r4c_full_corpus_quality_v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_root_rel": str(root.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": sha256_file(contract_path),
        "n_ids": len(ids),
        "n_rgb": len(rgb_ids),
        "n_skel": len(skel_ids),
        "n_errors": len(errors),
        "errors": errors,
        "e4_ready": False,
        "forbid_train_if_fail": True,
    }
    out = OUT / "E4_R4C_FULL_CORPUS_QUALITY.json"
    if out.is_file():
        die(f"refuse_overwrite:{out}")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("QUALITY", status, "n", len(ids), "errors", len(errors), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
