# -*- coding: utf-8 -*-
"""E4_B4SC_V1 E2 full-val accuracy gate vs E0 (n=2629), create-only artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bench_ms_sta_v6_e4_r4b_stage_matched import sealed_landmarks  # noqa: E402
from scripts.e4_b4sc_e1_independent_verify import (  # noqa: E402
    acc_from_preds,
    paired_delta,
)
from scripts.e4_b4sc_train_e2_screen import (  # noqa: E402
    AmpRoiFp32SlowFastTrainable,
    build_b4_batch_from_projected,
    head_logits_eval,
    load_meta,
    project_tokens,
)
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import (  # noqa: E402
    REGIONS,
    FrozenSlowFastBlock4,
    extract_rgb_token_v3_for_video,
    read_video_frames_native,
)
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.optimization.e4_b4sc.narrow_slowfast import (  # noqa: E402
    E2_WIDTHS,
    NarrowSlowFastBlock4,
    RoiToHeadProjection,
)
from src.training.ms_sta_v5_data import load_split_role_ids  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_e4_b4sc"
POL_TRAIN = OUT / "E4_B4SC_V1_E2E3_TRAINING_POLICY.json"
POL_E4 = ROOT / "Docs/results/ms_sta_v6_e4_b4sc_policy/E4_B4SC_V1_POLICY.json"
LOCK_PATH = OUT / "E4_B4SC_V1_CHANNEL_SELECTION_LOCK.json"
CKPT = OUT / "checkpoints/e2_screen_seed42/best.pt"
PRED_PATH = OUT / "E4_B4SC_V1_E2_SCREEN_E2_PREDICTIONS.jsonl"
LOGITS_PATH = OUT / "E4_B4SC_V1_E2_SCREEN_E2_LOGITS.npz"
MANIFEST_PATH = OUT / "E4_B4SC_V1_E2_SCREEN_LOGITS_MANIFEST.json"
REPORT_PATH = OUT / "E4_B4SC_V1_E2_SCREEN_ACCURACY_REPORT.json"
E0_PRED = OUT / "E4_B4SC_V1_E1_E0_PREDICTIONS.jsonl"


def die(msg: str) -> None:
    raise SystemExit(f"E4_B4SC_E2_ACC_FAIL:{msg}")


def create_only(path: Path, obj: Any) -> None:
    if path.exists():
        die(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(str(obj), encoding="utf-8")


def ensure_train_cuda() -> None:
    venv_py = ROOT / "_venvs/train_cuda/Scripts/python.exe"
    if venv_py.is_file() and Path(sys.executable).resolve() != venv_py.resolve():
        os.execv(str(venv_py), [str(venv_py), *sys.argv])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def hard_subset_from_e0(e0_rows: list[dict]) -> dict[str, Any]:
    """Frozen report subset: E0 errors + top confusion classes (by E0 error count)."""
    err_ids = [r["sample_id"] for r in e0_rows if int(r["y_pred"]) != int(r["y_true"])]
    by_true: dict[int, int] = defaultdict(int)
    for r in e0_rows:
        if int(r["y_pred"]) != int(r["y_true"]):
            by_true[int(r["y_true"])] += 1
    top_classes = [c for c, _ in sorted(by_true.items(), key=lambda kv: (-kv[1], kv[0]))[:40]]
    class_ids = [r["sample_id"] for r in e0_rows if int(r["y_true"]) in set(top_classes)]
    return {
        "definition": "e0_errors_union_top40_error_classes",
        "e0_error_ids": err_ids,
        "top40_hard_classes": top_classes,
        "hard_class_sample_ids": class_ids,
        "union_ids": sorted(set(err_ids) | set(class_ids)),
    }


def subset_top1(rows: list[dict], ids: set[str]) -> float:
    sub = [r for r in rows if r["sample_id"] in ids]
    if not sub:
        return float("nan")
    return float(np.mean([int(r["y_pred"] == r["y_true"]) for r in sub]))


def main() -> int:
    ensure_train_cuda()
    ap = argparse.ArgumentParser(description="E2 screen full-val accuracy vs E0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    if not CKPT.is_file():
        die(f"missing_ckpt:{CKPT}")
    tpol = json.loads(POL_TRAIN.read_text(encoding="utf-8"))
    pol = json.loads(POL_E4.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if sha256_file(POL_TRAIN) != lock.get("training_policy_sha256"):
        die("training_policy_sha_mismatch")
    if not E0_PRED.is_file():
        die("missing_e0_predictions")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        die("cuda_required")
    gpu = torch.cuda.get_device_name(0)
    if gpu != pol["scientific_environment_lock"]["required_gpu_name"]:
        die(f"gpu_mismatch:{gpu}")

    split = json.loads((ROOT / tpol["data"]["split_manifest_rel"]).read_text(encoding="utf-8"))
    val_ids = load_split_role_ids(split, tpol["data"]["val_role"])
    if args.max_samples > 0:
        val_ids = val_ids[: args.max_samples]

    meta = load_meta()
    labels_obj = json.loads((ROOT / tpol["data"]["labels_rel"]).read_text(encoding="utf-8"))
    g2i = labels_obj["gloss_to_id"]
    resolver = RawVideoResolver(ROOT)

    def resolve(sid: str) -> tuple[Path, Path, int, str, dict[str, Any]]:
        row = meta[sid]
        gloss = str(row.get("text_gloss") or "").strip()
        if gloss not in g2i:
            die(f"missing_label:{sid}")
        vpath = resolver.resolve(row)
        if vpath is None:
            die(f"missing_video:{sid}")
        sealed = ROOT / tpol["data"]["skel_rel"] / "skeleton_v3" / f"{sid}.npz"
        if not sealed.is_file():
            sealed = ROOT / tpol["data"]["skel_rel"] / f"{sid}.npz"
        if not sealed.is_file():
            die(f"missing_sealed:{sid}")
        return Path(vpath), sealed, int(g2i[gloss]), str(row.get("signer_id") or ""), row

    plan_e2 = lock["E2"]
    teacher_raw = FrozenSlowFastBlock4(
        checkpoint_path=ROOT / pol["rgb_token_v3_invariants"]["slowfast_checkpoint_rel"],
        expected_sha256=pol["rgb_token_v3_invariants"]["slowfast_checkpoint_sha256"],
        device=device,
        repo_root=ROOT,
    )
    student_raw = NarrowSlowFastBlock4(teacher_raw, plan_e2, device=device, requires_grad_slow=False)
    student_amp = AmpRoiFp32SlowFastTrainable(student_raw)
    roi_proj = RoiToHeadProjection(E2_WIDTHS.token_dim, out_dim=2304).to(device)

    payload = torch.load(CKPT, map_location=device, weights_only=False)
    student_raw.load_state_dict(payload["student"])
    roi_proj.load_state_dict(payload["roi_to_head_projection"])
    student_raw.eval()
    roi_proj.eval()

    b4_head = build_ms_sta_v6("b4_v6_ctr").to(device)
    b4_ckpt_path = ROOT / tpol["checkpoints"]["b4_teacher_rel"]
    if sha256_file(b4_ckpt_path) != tpol["checkpoints"]["b4_teacher_sha256"]:
        die("b4_teacher_sha_mismatch")
    b4_ckpt = torch.load(b4_ckpt_path, map_location=device, weights_only=False)
    b4_head.load_state_dict(b4_ckpt["model"])
    b4_head.eval()

    if PRED_PATH.exists() or LOGITS_PATH.exists() or MANIFEST_PATH.exists() or REPORT_PATH.exists():
        die("accuracy_artifacts_already_exist")

    pred_f = PRED_PATH.open("w", encoding="utf-8")
    logits_list: list[np.ndarray] = []
    ids_out: list[str] = []
    nan_seen = False

    for i, sid in enumerate(val_ids):
        vpath, sealed_path, label, signer, row = resolve(sid)
        frames, frame_idx, fps = read_video_frames_native(str(vpath))
        landmarks_by_frame, missing_by_frame, sfi, sts, sealed = sealed_landmarks(sealed_path)
        with torch.no_grad():
            extract = extract_rgb_token_v3_for_video(
                video_path=str(vpath),
                extractor=student_amp,
                landmarks_by_frame=landmarks_by_frame,
                missing_by_frame=missing_by_frame,
                skel_frame_index=sfi,
                skel_timestamps=sts,
                sample_id=sid,
                enable_pose_wrist_prior=True,
                enable_ambiguity_aware_prior=True,
                ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
                repo_root=ROOT,
                frames_native=frames,
                frame_idx=frame_idx,
                fps=fps,
                pathway_window_batch_size=1,
            )
            student_tokens = {
                r: torch.from_numpy(np.asarray(extract["tokens"][r], dtype=np.float32)).to(device)
                for r in REGIONS
            }
            proj_tokens = project_tokens(roi_proj, student_tokens)
            batch = build_b4_batch_from_projected(
                sid=sid,
                label=label,
                signer=signer,
                proj_tokens=proj_tokens,
                sealed=sealed,
                device=device,
            )
            logits = head_logits_eval(b4_head, batch)
            if not bool(torch.isfinite(logits).all()):
                nan_seen = True
                die(f"nan_inf_logits:{sid}")
            logits_np = logits.detach().float().cpu().numpy().reshape(-1)
            pred = int(logits_np.argmax())
            top5 = [int(x) for x in np.argsort(-logits_np)[:5].tolist()]
        row_out = {
            "sample_id": sid,
            "signer_id": signer,
            "y_true": label,
            "y_pred": pred,
            "top5": top5,
            "arm": "E2",
            "seed": 42,
            "checkpoint_rel": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(CKPT),
            "channel_lock_sha256": sha256_file(LOCK_PATH),
            "training_policy_sha256": sha256_file(POL_TRAIN),
            "code_commit": payload.get("meta", {}).get("git_head") or "",
        }
        pred_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
        logits_list.append(logits_np.astype(np.float32))
        ids_out.append(sid)
        if (i + 1) % 50 == 0:
            print(f"E2_ACC {i+1}/{len(val_ids)}", flush=True)

    pred_f.close()
    logits_arr = np.stack(logits_list, axis=0)
    np.savez_compressed(LOGITS_PATH, sample_ids=np.asarray(ids_out), logits=logits_arr)
    create_only(
        MANIFEST_PATH,
        {
            "schema": "ms_sta_v6_e4_b4sc_v1_e2_screen_logits_manifest_v1",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "arm": "E2",
            "n": len(ids_out),
            "predictions_rel": str(PRED_PATH.relative_to(ROOT)).replace("\\", "/"),
            "predictions_sha256": sha256_file(PRED_PATH),
            "logits_rel": str(LOGITS_PATH.relative_to(ROOT)).replace("\\", "/"),
            "logits_sha256": sha256_file(LOGITS_PATH),
            "checkpoint_sha256": sha256_file(CKPT),
            "nan_inf": nan_seen,
        },
    )

    e2_rows = load_jsonl(PRED_PATH)
    e0_rows = load_jsonl(E0_PRED)
    if args.max_samples <= 0 and len(e2_rows) != 2629:
        die(f"pred_n:{len(e2_rows)}")
    if args.max_samples > 0:
        print(f"E2_ACC_SMOKE n={len(e2_rows)} (skipping gate)", flush=True)
        return 0
    e0_by = {r["sample_id"]: r for r in e0_rows}
    e2_align = [e0_by[r["sample_id"]] for r in e2_rows if r["sample_id"] in e0_by]
    if len(e2_align) != len(e2_rows):
        die("e0_alignment_gap")

    acc_e0 = acc_from_preds(e2_align)
    acc_e2 = acc_from_preds(e2_rows)
    delta = paired_delta(e2_align, e2_rows, n_boot=int(tpol["accuracy_gate"]["n_boot"]), seed=int(tpol["accuracy_gate"]["bootstrap_seed"]))
    top5_delta_pp = 100.0 * (acc_e2["top5"] - acc_e0["top5"])
    macro_delta_pp = 100.0 * (acc_e2["macro_f1"] - acc_e0["macro_f1"])
    s0, s2 = acc_e0["signer_top1"], acc_e2["signer_top1"]
    worst_signer = 0.0
    signer_deltas = {}
    if s0:
        for s in s0:
            dpp = 100.0 * (s2.get(s, 0.0) - s0[s])
            signer_deltas[s] = dpp
        worst_signer = min(signer_deltas.values()) if signer_deltas else 0.0

    hard = hard_subset_from_e0(e2_align)
    hard_ids = set(hard["union_ids"])
    hard_e0 = subset_top1(e2_align, hard_ids)
    hard_e2 = subset_top1(e2_rows, hard_ids)
    hard_delta_pp = 100.0 * (hard_e2 - hard_e0)

    gate = tpol["accuracy_gate"]
    coverage_pass = len(e2_rows) == len(val_ids) and len({r["sample_id"] for r in e2_rows}) == len(e2_rows)
    acc_pass = (
        delta["ci95_lo"] >= gate["hierarchical_ci95_lower_top1_pp_min"]
        and top5_delta_pp >= gate["delta_top5_pp_min"]
        and macro_delta_pp >= gate["delta_macro_f1_pp_min"]
        and worst_signer >= gate["per_signer_delta_top1_pp_min"]
        and coverage_pass
        and not nan_seen
        and hard_delta_pp >= -1.0  # material hard-subset: no worse than -1.0 pp
    )

    report = {
        "schema": "ms_sta_v6_e4_b4sc_v1_e2_screen_accuracy_report_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "arm": "E2",
        "seed": 42,
        "n": len(e2_rows),
        "e0": acc_e0,
        "e2": acc_e2,
        "paired_delta_top1": delta,
        "top5_delta_pp": top5_delta_pp,
        "macro_delta_pp": macro_delta_pp,
        "worst_signer_delta_pp": worst_signer,
        "signer_deltas_pp": signer_deltas,
        "hard_subset": {
            **hard,
            "e0_top1": hard_e0,
            "e2_top1": hard_e2,
            "delta_pp": hard_delta_pp,
        },
        "coverage_pass": coverage_pass,
        "nan_inf": nan_seen,
        "accuracy_gate_pass": acc_pass,
        "gate": gate,
        "checkpoint_sha256": sha256_file(CKPT),
        "predictions_sha256": sha256_file(PRED_PATH),
        "logits_sha256": sha256_file(LOGITS_PATH),
        "e4_ready": False,
        "note": "Screen seed42 only; confirmation seeds not opened in this artifact.",
    }
    create_only(REPORT_PATH, report)
    print(
        "E4_B4SC_E2_ACC_OK" if acc_pass else "E4_B4SC_E2_ACC_NO_GO",
        f"top1_e0={acc_e0['top1']:.4f}",
        f"top1_e2={acc_e2['top1']:.4f}",
        f"ci95_lo={delta['ci95_lo']:.3f}",
        f"pass={acc_pass}",
        flush=True,
    )
    return 0 if acc_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
