# -*- coding: utf-8 -*-
"""B4Q-V1 online prediction evidence recovery (no timing; create-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.b4q_run_gate_a_and_native_ladder import (  # noqa: E402
    b4_item_from_extract,
    hierarchical_boot_delta,
    load_meta,
    sealed_landmarks,
)
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.models.rgb_token_v3 import FrozenSlowFastBlock4, extract_rgb_token_v3_for_video, read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import load_policy, sha256_file, verify_b4_checkpoint, verify_slowfast_checkpoint  # noqa: E402
from src.optimization.b4q.precision_wrappers import AmpRoiFp32SlowFast, head_forward  # noqa: E402
from src.training.ms_sta_v5_data import collate_v5, load_split_role_ids  # noqa: E402
from scripts.ms_sta_v5_d0_lib import R7C_AMBIGUITY_ROI_LOCK  # noqa: E402

OUT = ROOT / "Docs/results/ms_sta_v6_b4q"
PROTO = OUT / "B4Q_V1_ONLINE_PREDICTION_RECOVERY_PROTOCOL.json"


def die(msg: str) -> None:
    raise SystemExit(f"B4Q_RECOVERY_FAIL:{msg}")


def create_only(path: Path) -> None:
    if path.is_file():
        die(f"exists:{path}")


def config_hash(pol: dict) -> str:
    blob = json.dumps(
        {
            "rgb": pol["rgb_token_v3"],
            "skel": pol["skeleton_v3"],
            "split": pol["split"]["manifest_sha256"],
            "b4": pol["b4"]["checkpoint_sha256"],
            "sf": pol["rgb_token_v3"]["slowfast_checkpoint_sha256"],
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def macro_f1(rows: list[dict]) -> float:
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)
    for r in rows:
        y, p = int(r["y_true"]), int(r["y_pred"])
        if y == p:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    f1s = []
    for c in set(list(tp) + list(fp) + list(fn)):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def mcnemar(base: list[dict], cand: list[dict]) -> dict:
    b = c = 0
    for x, y in zip(base, cand):
        bt, ct = int(x["top1"]), int(y["top1"])
        if bt == 1 and ct == 0:
            b += 1
        elif bt == 0 and ct == 1:
            c += 1
    return {"n01_base_correct_cand_wrong": b, "n10_base_wrong_cand_correct": c}


def run_arm(
    *,
    arm: str,
    extractor,
    amp_head: bool,
    val_ids: list[str],
    meta: dict,
    g2i: dict,
    resolver: RawVideoResolver,
    sealed_root: Path,
    b4,
    device,
    ckpt_sha: str,
    cfg_sha: str,
    pred_path: Path,
    logits_path: Path,
) -> list[dict]:
    create_only(pred_path)
    create_only(logits_path)
    logits_list = []
    rows = []
    with pred_path.open("w", encoding="utf-8") as fh:
        for i, sid in enumerate(val_ids):
            row = meta.get(sid)
            if row is None:
                continue
            gloss = str(row.get("text_gloss") or "").strip()
            if gloss not in g2i:
                continue
            vpath = resolver.resolve(row)
            if vpath is None or not Path(vpath).is_file():
                continue
            sealed = sealed_root / f"{sid}.npz"
            if not sealed.is_file():
                continue
            label = int(g2i[gloss])
            signer = str(row.get("signer_id") or "")
            vpath = Path(vpath)
            landmarks_by_frame, missing_by_frame, sfi, sts, sealed_blob = sealed_landmarks(sealed)
            frames, frame_idx, fps = read_video_frames_native(str(vpath))
            extract = extract_rgb_token_v3_for_video(
                video_path=str(vpath),
                extractor=extractor,
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
            )
            item = b4_item_from_extract(sid, extract, sealed_blob, label, signer)
            batch = collate_v5([item])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = head_forward(b4, batch, amp_fp16=amp_head, inference_mode=True)
            logits_np = logits.detach().float().cpu().numpy()[0]
            pred = int(logits_np.argmax())
            top5 = [int(x) for x in np.argsort(-logits_np)[:5].tolist()]
            hit1 = int(pred == label)
            hit5 = int(label in set(top5))
            rec = {
                "sample_id": sid,
                "signer_id": signer,
                "y_true": label,
                "y_pred": pred,
                "top5": top5,
                "top1": hit1,
                "top5_hit": hit5,
                "arm": arm,
                "config_sha256": cfg_sha,
                "checkpoint_sha256": ckpt_sha,
                "idx": len(rows),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
            logits_list.append(logits_np.astype(np.float16))
            if (i + 1) % 50 == 0:
                t1 = sum(r["top1"] for r in rows) / max(len(rows), 1)
                print(f"  {arm} {i+1}/{len(val_ids)} n={len(rows)} top1={t1:.4f}", flush=True)
    arr = np.stack(logits_list, axis=0)
    np.savez_compressed(logits_path, logits=arr, sample_ids=np.array([r["sample_id"] for r in rows]))
    return rows


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    top1 = sum(int(r["top1"]) for r in rows) / max(n, 1)
    top5 = sum(int(r["top5_hit"]) for r in rows) / max(n, 1)
    by_s: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_s[str(r["signer_id"])].append(int(r["top1"]))
    return {
        "n": n,
        "top1": top1,
        "top5": top5,
        "macro_f1": macro_f1(rows),
        "signer_top1": {s: float(np.mean(v)) for s, v in by_s.items()},
        "signer_counts": {s: len(v) for s, v in by_s.items()},
        "sample_ids": [r["sample_id"] for r in rows],
    }


def align_check(a: list[dict], b: list[dict]) -> dict:
    ids_a = [r["sample_id"] for r in a]
    ids_b = [r["sample_id"] for r in b]
    return {
        "same_length": len(a) == len(b),
        "same_order_ids": ids_a == ids_b,
        "duplicate_a": len(ids_a) - len(set(ids_a)),
        "duplicate_b": len(ids_b) - len(set(ids_b)),
        "missing_in_b": sorted(set(ids_a) - set(ids_b))[:20],
        "missing_in_a": sorted(set(ids_b) - set(ids_a))[:20],
        "label_alignment_ok": all(int(x["y_true"]) == int(y["y_true"]) for x, y in zip(a, b)),
        "pred_agreement_rate": sum(int(x["y_pred"] == y["y_pred"]) for x, y in zip(a, b)) / max(len(a), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    proto = json.loads(PROTO.read_text(encoding="utf-8"))
    pol = load_policy()
    verify_b4_checkpoint(pol)
    verify_slowfast_checkpoint(pol)

    pred_paths = {
        "Q0_ONLINE": OUT / "B4Q_V1_ONLINE_PRED_Q0.jsonl",
        "Q2_SLOWFAST_AMP_FP16_ROI_FP32": OUT / "B4Q_V1_ONLINE_PRED_Q2.jsonl",
        "Q3_FULL_B4_MIXED_PRECISION": OUT / "B4Q_V1_ONLINE_PRED_Q3.jsonl",
    }
    logits_paths = {
        "Q0_ONLINE": OUT / "B4Q_V1_ONLINE_LOGITS_Q0.npz",
        "Q2_SLOWFAST_AMP_FP16_ROI_FP32": OUT / "B4Q_V1_ONLINE_LOGITS_Q2.npz",
        "Q3_FULL_B4_MIXED_PRECISION": OUT / "B4Q_V1_ONLINE_LOGITS_Q3.npz",
    }
    verify_out = OUT / "B4Q_V1_ONLINE_ACCURACY_INDEPENDENT_VERIFY.json"
    mismatch_out = OUT / "B4Q_V1_ONLINE_RECOVERY_MISMATCH.json"

    if not args.verify_only:
        for p in list(pred_paths.values()) + list(logits_paths.values()) + [verify_out, mismatch_out]:
            create_only(p)

        device = torch.device("cuda")
        if torch.cuda.get_device_name(0) != pol["scientific_environment_lock"]["required_gpu_name"]:
            die("gpu_mismatch")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        cfg_sha = config_hash(pol)
        ckpt_sha = pol["b4"]["checkpoint_sha256"]

        meta = load_meta()
        labels_obj = json.loads((ROOT / pol["split"]["labels_rel"]).read_text(encoding="utf-8"))
        g2i = labels_obj["gloss_to_id"]
        split = json.loads((ROOT / pol["split"]["manifest_rel"]).read_text(encoding="utf-8"))
        val_ids = load_split_role_ids(split, "val")
        if len(val_ids) != proto["locks"]["expected_val_n"]:
            # not fatal yet — coverage checked later
            print("WARN val_ids_len", len(val_ids), flush=True)

        resolver = RawVideoResolver(ROOT)
        sealed_root = ROOT / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3"
        base_sf = FrozenSlowFastBlock4(
            checkpoint_path=ROOT / pol["rgb_token_v3"]["slowfast_checkpoint_rel"],
            expected_sha256=pol["rgb_token_v3"]["slowfast_checkpoint_sha256"],
            device=device,
            repo_root=ROOT,
        )
        amp_sf = AmpRoiFp32SlowFast(base_sf, channels_last=False)
        b4 = build_ms_sta_v6("b4_v6_ctr").to(device).eval()
        ckpt = torch.load(ROOT / pol["b4"]["checkpoint_rel"], map_location=device, weights_only=False)
        b4.load_state_dict(ckpt["model"])

        print("RECOVERY Q0_ONLINE...", flush=True)
        run_arm(
            arm="Q0_ONLINE",
            extractor=base_sf,
            amp_head=False,
            val_ids=val_ids,
            meta=meta,
            g2i=g2i,
            resolver=resolver,
            sealed_root=sealed_root,
            b4=b4,
            device=device,
            ckpt_sha=ckpt_sha,
            cfg_sha=cfg_sha,
            pred_path=pred_paths["Q0_ONLINE"],
            logits_path=logits_paths["Q0_ONLINE"],
        )
        print("RECOVERY Q2...", flush=True)
        run_arm(
            arm="Q2_SLOWFAST_AMP_FP16_ROI_FP32",
            extractor=amp_sf,
            amp_head=False,
            val_ids=val_ids,
            meta=meta,
            g2i=g2i,
            resolver=resolver,
            sealed_root=sealed_root,
            b4=b4,
            device=device,
            ckpt_sha=ckpt_sha,
            cfg_sha=cfg_sha,
            pred_path=pred_paths["Q2_SLOWFAST_AMP_FP16_ROI_FP32"],
            logits_path=logits_paths["Q2_SLOWFAST_AMP_FP16_ROI_FP32"],
        )
        print("RECOVERY Q3...", flush=True)
        run_arm(
            arm="Q3_FULL_B4_MIXED_PRECISION",
            extractor=amp_sf,
            amp_head=True,
            val_ids=val_ids,
            meta=meta,
            g2i=g2i,
            resolver=resolver,
            sealed_root=sealed_root,
            b4=b4,
            device=device,
            ckpt_sha=ckpt_sha,
            cfg_sha=cfg_sha,
            pred_path=pred_paths["Q3_FULL_B4_MIXED_PRECISION"],
            logits_path=logits_paths["Q3_FULL_B4_MIXED_PRECISION"],
        )
        run_meta = {"git_head_at_run": head, "config_sha256": cfg_sha, "checkpoint_sha256": ckpt_sha}
        (OUT / "B4Q_V1_ONLINE_RECOVERY_RUN_META.json").write_text(
            json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
        )

    # Independent verify from dumps
    def load_rows(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    q0 = load_rows(pred_paths["Q0_ONLINE"])
    q2 = load_rows(pred_paths["Q2_SLOWFAST_AMP_FP16_ROI_FP32"])
    q3 = load_rows(pred_paths["Q3_FULL_B4_MIXED_PRECISION"])
    m0, m2, m3 = metrics(q0), metrics(q2), metrics(q3)

    # rows for hierarchical boot need keys y/pred/top1/signer
    def as_boot(rows: list[dict]) -> list[dict]:
        return [
            {"y": r["y_true"], "pred": r["y_pred"], "top1": r["top1"], "top5": r["top5_hit"], "signer": r["signer_id"]}
            for r in rows
        ]

    d2 = hierarchical_boot_delta(as_boot(q0), as_boot(q2), n_boot=2000, seed=42)
    d3 = hierarchical_boot_delta(as_boot(q0), as_boot(q3), n_boot=2000, seed=42)
    a2 = align_check(q0, q2)
    a3 = align_check(q0, q3)

    orig = proto["locks"]["original_ladder_aggregates"]
    tol = proto["locks"]["aggregate_match_tolerance"]

    def match_arm(m: dict, key: str) -> bool:
        o = orig[key] if key != "Q0_ONLINE" else orig["Q0_ONLINE_implied_from_q2_q3_equality"]
        return (
            abs(m["top1"] - o["top1"]) <= tol["top1_abs"]
            and abs(m["top5"] - o["top5"]) <= tol["top5_abs"]
            and (m["n"] == o["n"] if tol["n_exact"] else True)
        )

    matches = {
        "Q0_ONLINE": match_arm(m0, "Q0_ONLINE"),
        "Q2": match_arm(m2, "Q2_SLOWFAST_AMP_FP16_ROI_FP32"),
        "Q3": match_arm(m3, "Q3_FULL_B4_MIXED_PRECISION"),
    }
    all_match = all(matches.values()) and a2["same_order_ids"] and a3["same_order_ids"] and a2["label_alignment_ok"]

    payload = {
        "schema": "ms_sta_v6_b4q_v1_online_accuracy_independent_verify_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_rel": str(PROTO.relative_to(ROOT)).replace("\\", "/"),
        "verdict_unchanged": "B4Q_V1_EFFICIENCY_NO_GO",
        "pred_sha256": {k: sha256_file(v) for k, v in pred_paths.items()},
        "logits_sha256": {k: sha256_file(v) for k, v in logits_paths.items()},
        "recomputed": {"Q0_ONLINE": {k: v for k, v in m0.items() if k != "sample_ids"}, "Q2": {k: v for k, v in m2.items() if k != "sample_ids"}, "Q3": {k: v for k, v in m3.items() if k != "sample_ids"}},
        "coverage": {
            "expected_n": proto["locks"]["expected_val_n"],
            "n_q0": m0["n"],
            "n_q2": m2["n"],
            "n_q3": m3["n"],
            "ids_equal_q0_q2": a2["same_order_ids"],
            "ids_equal_q0_q3": a3["same_order_ids"],
        },
        "paired": {
            "Q2_vs_Q0": {"delta": d2, "mcnemar": mcnemar(as_boot(q0), as_boot(q2)), "alignment": a2},
            "Q3_vs_Q0": {"delta": d3, "mcnemar": mcnemar(as_boot(q0), as_boot(q3)), "alignment": a3},
        },
        "aggregate_match_vs_ladder_report": matches,
        "pass": bool(all_match and m0["n"] == 2629),
        "source_gap_closed": bool(all_match and m0["n"] == 2629),
    }

    if verify_out.is_file() and not args.verify_only:
        die("verify_exists")
    if not verify_out.is_file():
        verify_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        # verify-only may overwrite? create-only forbids — only allow if missing
        pass

    if not all_match:
        if not mismatch_out.is_file():
            mismatch_out.write_text(
                json.dumps(
                    {
                        "schema": "ms_sta_v6_b4q_v1_online_recovery_mismatch_v1",
                        "created_at": payload["created_at"],
                        "matches": matches,
                        "recomputed_top1": {"Q0": m0["top1"], "Q2": m2["top1"], "Q3": m3["top1"]},
                        "original": orig,
                        "action": "KEEP_SOURCE_GAP_DO_NOT_OVERWRITE_LADDER_REPORT",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print("MISMATCH", matches, flush=True)
        return 2

    print("ONLINE_RECOVERY_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
