# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 R0 FP32 online supplemental comparator — one-shot; no R1/S5 rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.e4_syskd_v1_s5_confirmation import build_models_for_seed  # noqa: E402
from scripts.e4_syskd_v1_s5_formal import (  # noqa: E402
    BIND_SHA,
    POL_SHA,
    append_jsonl,
    extract_arm,
    frames_sha,
    load_binding,
    load_meta,
    provenance_recheck,
    write_lf,
)
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.rgb_token_v3 import read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"
AUTH = ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_AUTHORIZATION.json"
MANIFEST = ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_MANIFEST.json"
JOURNAL = ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_JOURNAL.jsonl"
R0_PATH = REG / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_PREDICTIONS.jsonl"
CODE_CAND = "b52c23b0e3598bd78731c4e3e28c7dbb5e5eb8f2"
S5_SHA = "bca59f72f3ab505330a360fcd2f50d5b0920eeed8797d604e3360694b57578a6"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_auth() -> tuple[dict, dict]:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not auth.get("user_authorization_received"):
        raise SystemExit("R0_COMPARATOR_NOT_AUTHORIZED")
    if sha256_file(MANIFEST) != auth.get("manifest_sha256"):
        raise SystemExit("manifest_sha_mismatch")
    if auth.get("s5_predictions_sha256") != S5_SHA:
        raise SystemExit("s5_dump_sha_drift")
    s5 = REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_S5.jsonl"
    if sha256_file(s5) != S5_SHA:
        raise SystemExit("s5_dump_file_sha_mismatch")
    return auth, man


def completed_ids() -> set[str]:
    done: set[str] = set()
    if not JOURNAL.exists():
        return done
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "completed" and row.get("sample_id"):
            done.add(str(row["sample_id"]))
    return done


def run() -> int:
    auth, man = require_auth()
    provenance_recheck(when="before_r0_online_comparator")
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    split = json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))
    test_ids = list(split["test_ids"])
    if len(test_ids) != 6165:
        raise SystemExit("n_test")
    test_sha = hashlib.sha256(("\n".join(test_ids) + "\n").encode()).hexdigest()
    if test_sha != man["test_ids_sha256"]:
        raise SystemExit("TEST_INPUT_PROVENANCE_INVALID:test_ids")

    device = torch.device("cuda")
    ckpt = Path(man["checkpoint"]["path_absolute"])
    base, _r1, b4, pol = build_models_for_seed(data_root, device, ckpt, man["checkpoint"]["sha256"])
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"

    done = completed_ids()
    print(f"[r0-comparator] resume_completed={len(done)} remaining={len(test_ids) - len(done)}", flush=True)

    for i, sid in enumerate(test_ids):
        if sid in done:
            continue
        m = meta[sid]
        gloss = str(m.get("text_gloss") or "").strip()
        label = int(labels[gloss])
        signer = str(m.get("signer_id") or "")
        vpath = Path(resolver.resolve(m))
        sealed = sealed_root / f"{sid}.npz"
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        fsha = frames_sha(frames, frame_idx)
        # R0 = FrozenSlowFastBlock4 base, no AmpRoi, no ROI batching
        r0o = extract_arm(
            extractor=base,
            batch_roi=False,
            sid=sid,
            vpath=vpath,
            sealed_path=sealed,
            data_root=data_root,
            frames=frames,
            frame_idx=frame_idx,
            fps=fps,
            b4=b4,
            device=device,
            label=label,
            signer=signer,
            time_stages=False,
        )
        append_jsonl(
            R0_PATH,
            {
                "sample_id": sid,
                "target": label,
                "signer_id": signer,
                "frame_sha": fsha,
                "checkpoint_sha256": man["checkpoint"]["sha256"],
                "policy_sha256": POL_SHA,
                "input_binding_sha256": BIND_SHA,
                "code_sha": CODE_CAND,
                "role": "test",
                "arm": "R0",
                "precision": "FP32_online",
                "logits": r0o["logits"].numpy().astype(np.float32).tolist(),
                "prediction": r0o["prediction"],
                "top5": r0o["top5"],
                "y_true": label,
                "y_pred": r0o["prediction"],
                "nan_inf": bool(r0o["nan_inf"]),
                "auth_sha256": sha256_file(AUTH),
                "manifest_sha256": sha256_file(MANIFEST),
            },
        )
        append_jsonl(
            JOURNAL,
            {
                "sample_id": sid,
                "index": i,
                "status": "completed",
                "frame_sha": fsha,
                "r0_pred": r0o["prediction"],
                "nan_inf": bool(r0o["nan_inf"]),
                "ts": utc_now(),
            },
        )
        done.add(sid)
        if (len(done) % 50 == 0) or len(done) == 1 or len(done) == len(test_ids):
            print(f"[r0-comparator] {len(done)}/{len(test_ids)}", flush=True)

    if len(done) != len(test_ids):
        write_lf(
            ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_TERMINAL.json",
            {
                "terminal": "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_INCOMPLETE",
                "completed": len(done),
                "expected": len(test_ids),
                "selected": None,
                "e4_ready": False,
                "d4_demo_closed": True,
                "b4_retained": True,
            },
            overwrite=True,
        )
        raise SystemExit("E4_SYSKD_V1_R0_ONLINE_COMPARATOR_INCOMPLETE")

    pointer = {
        "schema": "r0_online_comparator_prediction_pointer_v1",
        "r0_path": str(R0_PATH.relative_to(CODE_ROOT)).replace("\\", "/"),
        "r0_sha256": sha256_file(R0_PATH),
        "r0_bytes": R0_PATH.stat().st_size,
        "n": len(test_ids),
        "checkpoint_sha256": man["checkpoint"]["sha256"],
        "s5_sha256_frozen": S5_SHA,
        "code_sha": CODE_CAND,
        "auth_sha256": sha256_file(AUTH),
        "manifest_sha256": sha256_file(MANIFEST),
    }
    write_lf(ART / "E4_SYSKD_V1_R0_ONLINE_COMPARATOR_PREDICTIONS.json", pointer, overwrite=True)
    provenance_recheck(when="after_r0_online_comparator")
    print(json.dumps(pointer, indent=2), flush=True)
    print("RUN_COMPLETE — invoke verify next", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["run"], default="run")
    args = ap.parse_args()
    if args.phase == "run":
        return run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
