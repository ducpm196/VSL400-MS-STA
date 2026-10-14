# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 S5 locked test — one-shot; resume only from verified journal rows."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
    macro_f1,
    provenance_recheck,
    write_lf,
)
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.rgb_token_v3 import read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_syskd.coverage_sets import paired_arm_coverage  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
REG = CODE_ROOT / "Docs/results/e4_system_kd_v1_registry"
AUTH = ART / "E4_SYSKD_V1_TEST_OPENING_AUTHORIZATION.json"
MANIFEST = ART / "E4_SYSKD_V1_S5_LOCKED_TEST_MANIFEST.json"
JOURNAL = ART / "E4_SYSKD_V1_S5_LOCKED_TEST_JOURNAL.jsonl"
R1_PATH = REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_R1.jsonl"
S5_PATH = REG / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS_S5.jsonl"
EXPECTED_TIP = "b52c23b0e3598bd78731c4e3e28c7dbb5e5eb8f2"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_auth() -> tuple[dict, dict]:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not auth.get("user_authorization_received"):
        raise SystemExit("TEST_NOT_AUTHORIZED")
    if auth.get("authorized_candidate_commit") != EXPECTED_TIP:
        raise SystemExit("auth_tip_mismatch")
    if sha256_file(MANIFEST) != auth.get("locked_test_manifest_sha256"):
        raise SystemExit("manifest_sha_mismatch")
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


def run_locked_test() -> dict:
    auth, man = require_auth()
    provenance_recheck(when="before_locked_test")
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    split = json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))
    test_ids = list(split["test_ids"])
    if len(test_ids) != int(man["n"]) or len(test_ids) != 6165:
        raise SystemExit("n_test")
    import hashlib

    test_sha = hashlib.sha256(("\n".join(test_ids) + "\n").encode()).hexdigest()
    if test_sha != man["test_ids_sha256"]:
        raise SystemExit("TEST_INPUT_PROVENANCE_INVALID:test_ids")

    device = torch.device("cuda")
    ckpt = Path(man["checkpoint"]["path_absolute"])
    base, r1, b4, pol = build_models_for_seed(data_root, device, ckpt, man["checkpoint"]["sha256"])
    meta = load_meta(data_root)
    labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
    resolver = RawVideoResolver(data_root)
    sealed_root = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"

    done = completed_ids()
    print(f"[locked-test] resume_completed={len(done)} remaining={len(test_ids) - len(done)}", flush=True)

    abs_max = 0.0
    abs_sum = 0.0
    abs_n = 0
    agree = top5_agree = nan_inf = 0

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
        r1o = extract_arm(
            extractor=r1,
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
        s5o = extract_arm(
            extractor=r1,
            batch_roi=True,
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
        diff = (r1o["logits"] - s5o["logits"]).abs()
        mad = float(diff.max())
        mean_ad = float(diff.mean())
        abs_max = max(abs_max, mad)
        abs_sum += float(diff.sum())
        abs_n += int(diff.numel())
        agree += int(r1o["prediction"] == s5o["prediction"])
        top5_agree += int(set(r1o["top5"]) == set(s5o["top5"]))
        nan_inf += int(bool(r1o["nan_inf"] or s5o["nan_inf"]))
        common = {
            "sample_id": sid,
            "target": label,
            "signer_id": signer,
            "frame_sha": fsha,
            "checkpoint_sha256": man["checkpoint"]["sha256"],
            "policy_sha256": POL_SHA,
            "input_binding_sha256": BIND_SHA,
            "code_sha": EXPECTED_TIP,
            "role": "test",
        }
        append_jsonl(
            R1_PATH,
            {
                **common,
                "arm": "R1",
                "logits": r1o["logits"].numpy().astype(np.float32).tolist(),
                "prediction": r1o["prediction"],
                "top5": r1o["top5"],
                "y_true": label,
                "y_pred": r1o["prediction"],
            },
        )
        append_jsonl(
            S5_PATH,
            {
                **common,
                "arm": "S5",
                "logits": s5o["logits"].numpy().astype(np.float32).tolist(),
                "prediction": s5o["prediction"],
                "top5": s5o["top5"],
                "y_true": label,
                "y_pred": s5o["prediction"],
                "logit_max_abs_vs_r1": mad,
                "logit_mean_abs_vs_r1": mean_ad,
            },
        )
        append_jsonl(
            JOURNAL,
            {
                "sample_id": sid,
                "index": i,
                "status": "completed",
                "frame_sha": fsha,
                "r1_pred": r1o["prediction"],
                "s5_pred": s5o["prediction"],
                "logit_max_abs": mad,
                "ts": utc_now(),
            },
        )
        done.add(sid)
        if (len(done) % 50 == 0) or len(done) == 1 or len(done) == len(test_ids):
            print(f"[locked-test] {len(done)}/{len(test_ids)} max_abs={abs_max:.3e}", flush=True)

    if len(done) != len(test_ids):
        write_lf(
            ART / "E4_SYSKD_V1_S5_LOCKED_TEST_TERMINAL.json",
            {
                "terminal": "E4_SYSKD_V1_S5_LOCKED_TEST_INCOMPLETE",
                "completed": len(done),
                "expected": len(test_ids),
                "selected": None,
                "e4_ready": False,
                "d4_demo_closed": True,
            },
            overwrite=True,
        )
        raise SystemExit("E4_SYSKD_V1_S5_LOCKED_TEST_INCOMPLETE")

    return aggregate(man, test_ids)


def aggregate(man: dict, test_ids: list[str]) -> dict:
    r1_rows = [json.loads(l) for l in R1_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    s5_rows = [json.loads(l) for l in S5_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    # de-dup by last write if resume duplicated (should not with skip)
    def dedup(rows: list[dict]) -> list[dict]:
        by = {}
        for r in rows:
            by[r["sample_id"]] = r
        return [by[sid] for sid in test_ids if sid in by]

    r1_rows, s5_rows = dedup(r1_rows), dedup(s5_rows)
    cov = paired_arm_coverage(test_ids, [r["sample_id"] for r in r1_rows], [r["sample_id"] for r in s5_rows])

    def acc(rows: list[dict]) -> dict:
        y = [int(r["y_true"]) for r in rows]
        p = [int(r["y_pred"]) for r in rows]
        top5 = [int(int(r["y_true"]) in set(r["top5"])) for r in rows]
        by_s: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_s[str(r["signer_id"])].append(int(r["y_pred"] == r["y_true"]))
        return {
            "n": len(rows),
            "top1": float(np.mean([int(a == b) for a, b in zip(y, p)])),
            "top5": float(np.mean(top5)),
            "macro_f1": macro_f1(y, p),
            "signer_top1": {s: float(np.mean(v)) for s, v in by_s.items() if s},
        }

    a_r1, a_s5 = acc(r1_rows), acc(s5_rows)
    s5_by = {r["sample_id"]: r for r in s5_rows}
    abs_max = 0.0
    abs_sum = 0.0
    abs_n = 0
    agree = top5_agree = nan_inf = 0
    for r in r1_rows:
        s = s5_by[r["sample_id"]]
        d = np.abs(np.asarray(r["logits"], np.float32) - np.asarray(s["logits"], np.float32))
        abs_max = max(abs_max, float(d.max()))
        abs_sum += float(d.sum())
        abs_n += int(d.size)
        agree += int(r["prediction"] == s["prediction"])
        top5_agree += int(set(r["top5"]) == set(s["top5"]))
        nan_inf += int(not np.isfinite(np.asarray(r["logits"])).all() or not np.isfinite(np.asarray(s["logits"])).all())
    mean_abs = abs_sum / max(abs_n, 1)
    exact = abs_max == 0.0 and mean_abs == 0.0
    fidelity_ok = (
        bool(cov["exact_coverage"])
        and abs_max <= 1e-4
        and mean_abs <= 1e-5
        and agree == len(test_ids)
        and top5_agree == len(test_ids)
        and nan_inf == 0
    )
    pointer = {
        "schema": "locked_test_prediction_pointer_v1",
        "r1_path": str(R1_PATH.relative_to(CODE_ROOT)).replace("\\", "/"),
        "s5_path": str(S5_PATH.relative_to(CODE_ROOT)).replace("\\", "/"),
        "r1_sha256": sha256_file(R1_PATH),
        "s5_sha256": sha256_file(S5_PATH),
        "r1_bytes": R1_PATH.stat().st_size,
        "s5_bytes": S5_PATH.stat().st_size,
        "n": len(test_ids),
        "checkpoint_sha256": man["checkpoint"]["sha256"],
        "code_sha": EXPECTED_TIP,
    }
    write_lf(ART / "E4_SYSKD_V1_S5_LOCKED_TEST_PREDICTIONS.json", pointer, overwrite=True)
    report = {
        "schema": "e4_syskd_v1_s5_locked_test_report_v1",
        "created_at": utc_now(),
        "n": len(test_ids),
        "coverage": cov,
        "nan_inf": nan_inf,
        "R1": {k: v for k, v in a_r1.items() if k != "signer_top1"},
        "S5": {k: v for k, v in a_s5.items() if k != "signer_top1"},
        "S5_signer_top1": a_s5["signer_top1"],
        "prediction_agreement": agree / len(test_ids),
        "top5_set_agreement": top5_agree / len(test_ids),
        "logit_max_abs": abs_max,
        "logit_mean_abs": mean_abs,
        "EXACT_LOGIT_PARITY": exact,
        "fidelity_pass": fidelity_ok,
        "pass": fidelity_ok and bool(cov["exact_coverage"]),
        "checkpoint_sha256": man["checkpoint"]["sha256"],
        "policy_sha256": POL_SHA,
        "binding_sha256": BIND_SHA,
        "code_sha": EXPECTED_TIP,
        "auth_sha256": sha256_file(AUTH),
        "manifest_sha256": sha256_file(MANIFEST),
        "r1_predictions_sha256": pointer["r1_sha256"],
        "s5_predictions_sha256": pointer["s5_sha256"],
        "selected": None,
        "e4_ready": False,
        "d4_demo_closed": True,
        "note": "One-shot locked test; no retune. selected/e4_ready follow promotion protocol only after independent verify.",
    }
    write_lf(ART / "E4_SYSKD_V1_S5_LOCKED_TEST_REPORT.json", report, overwrite=True)
    term = "E4_SYSKD_V1_S5_LOCKED_TEST_PASS" if report["pass"] else "E4_SYSKD_V1_S5_LOCKED_TEST_NO_GO"
    write_lf(
        ART / "E4_SYSKD_V1_S5_LOCKED_TEST_TERMINAL.json",
        {
            "terminal": term,
            "provisional_candidate": "S5",
            "selected": None,
            "e4_ready": False,
            "b4_retained": True,
            "d4_demo_closed": True,
            "pass": report["pass"],
            "created_at": utc_now(),
        },
        overwrite=True,
    )
    provenance_recheck(when="after_locked_test")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["run", "aggregate"], default="run")
    args = ap.parse_args()
    if args.phase == "run":
        print(json.dumps(run_locked_test(), indent=2)[:3000], flush=True)
    else:
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        binding = load_binding()
        data_root = Path(binding["canonical_data_root"])
        test_ids = json.loads((data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8"))[
            "test_ids"
        ]
        print(json.dumps(aggregate(man, list(test_ids)), indent=2)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
