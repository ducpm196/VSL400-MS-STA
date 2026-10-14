#!/usr/bin/env python3
"""One-shot retrospective signer-disjoint test for frozen MS-STA v5 seed-42 models.

Requires sealed ONE_SHOT_TEST_AUTHORIZATION. Never loads B4-E.
Create-only artifacts under Docs/results/ms_sta_v5_oneshot/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import git_rev  # noqa: E402
from src.eval.ms_sta_v5_m0_strong_baseline import PredRow, hierarchical_bootstrap_ci  # noqa: E402
from src.models.ms_sta_v5 import build_ms_sta_v5  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json  # noqa: E402
from src.training.ms_sta_v5_train import evaluate, load_metadata_by_id  # noqa: E402

AUTH_REL = "Docs/results/ms_sta_v5_oneshot/ONE_SHOT_TEST_AUTHORIZATION.json"
FREEZE_REL = "Docs/results/ms_sta_v5_oneshot/PRETEST_FREEZE.json"
OUT_DIR = _ROOT / "Docs/results/ms_sta_v5_oneshot"
FORBIDDEN = frozenset({"b4_star_e_stride4"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, obj: Any) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _rows_to_pred(rows: list[dict[str, Any]]) -> dict[str, PredRow]:
    out: dict[str, PredRow] = {}
    for r in rows:
        sid = str(r["sample_id"])
        if sid in out:
            raise SystemExit(f"duplicate_sample_id_in_eval_rows:{sid}")
        out[sid] = PredRow(
            sample_id=sid,
            signer=str(r["signer"]),
            label=int(r["label"]),
            correct=bool(r["correct"]),
            top5_hit=bool(r["top5_hit"]),
        )
    return out


def _paired_deltas(
    primary: dict[str, PredRow], baseline: dict[str, PredRow]
) -> tuple[list[str], list[float], list[float], list[str], dict[str, float]]:
    ids = sorted(primary.keys())
    if set(ids) != set(baseline.keys()):
        raise RuntimeError("pred_id_mismatch")
    d1: list[float] = []
    d5: list[float] = []
    signers: list[str] = []
    by_signer: dict[str, list[float]] = {}
    for sid in ids:
        p = primary[sid]
        b = baseline[sid]
        if p.signer != b.signer or p.label != b.label:
            raise RuntimeError(f"alignment_fail:{sid}")
        delta = float(p.correct) - float(b.correct)
        d1.append(delta)
        d5.append(float(p.top5_hit) - float(b.top5_hit))
        signers.append(p.signer)
        by_signer.setdefault(p.signer, []).append(delta)
    signer_pp = {s: 100.0 * (sum(xs) / len(xs)) for s, xs in sorted(by_signer.items())}
    return ids, d1, d5, signers, signer_pp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm-retrospective-test", action="store_true", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    auth = load_json(_ROOT / AUTH_REL)
    freeze = load_json(_ROOT / FREEZE_REL)
    # Lifecycle: refuse re-entry once completion is sealed (immutable evidence).
    completion_path = OUT_DIR / "TEST_COMPLETION.json"
    report_path = OUT_DIR / "TEST_REPORT.json"
    if completion_path.is_file() or report_path.is_file():
        raise SystemExit("oneshot_already_complete_refuse_rerun")
    if auth.get("status") != "AUTHORIZED" or auth.get("allow_final_test") is not True:
        raise SystemExit("oneshot_auth_invalid")
    if auth.get("confirm_retrospective_test") is not True:
        raise SystemExit("confirm_flag_required_in_auth")
    if not args.confirm_retrospective_test:
        raise SystemExit("cli_confirm_required")
    if sha256_file(_ROOT / FREEZE_REL) != auth["pretest_freeze_sha256"]:
        raise SystemExit("freeze_sha_mismatch")
    if freeze.get("status") != "FROZEN":
        raise SystemExit("freeze_not_frozen")

    variants = list(auth["test_variants"])
    if any(v in FORBIDDEN or v in set(auth.get("forbidden_on_test") or []) for v in variants):
        raise SystemExit("forbidden_variant_in_auth")
    if "b4_star_e_stride4" in variants:
        raise SystemExit("b4_e_forbidden_on_test")

    policy = load_json(_ROOT / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json")
    split_path = _ROOT / policy["split_manifest_rel"]
    if sha256_file(split_path) != auth["split_manifest_sha256"]:
        raise SystemExit("split_sha_mismatch")
    split = load_json(split_path)
    test_ids = [str(x) for x in split["test_ids"]]
    if len(test_ids) != int(auth["expected_test_n"]):
        raise SystemExit(f"test_n_mismatch:{len(test_ids)}")

    meta = load_metadata_by_id(_ROOT / "metadata.csv")
    labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
    g2i = labels_obj["gloss_to_id"]
    labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}

    # Fail-closed: refuse if any train/val id sneaks into the test list.
    train_val = set()
    for key in ("hpo_train_ids", "tune_ids", "confirm_train_ids", "val_ids"):
        train_val.update(str(x) for x in (split.get(key) or []))
    leak = sorted(set(test_ids) & train_val)
    if leak:
        raise SystemExit(f"test_intersects_train_val:{leak[:5]}")

    rgb_root = _ROOT / policy["feature_roots"]["rgb_root_rel"]
    skel_root = _ROOT / policy["feature_roots"]["skeleton_root_rel"]
    ds = VSL400MSSTAv5Dataset(
        test_ids,
        rgb_root=rgb_root,
        skel_root=skel_root,
        labels=labels,
        metadata_by_id=meta,
        forbid_ids=set(),  # intentional test opening under sealed auth
        require_all=True,
    )
    if len(ds) != len(test_ids):
        raise SystemExit(f"dataset_n_mismatch:{len(ds)}!={len(test_ids)}")
    # Verify signers.
    signers = sorted({str(meta[sid]["signer"]) for sid in test_ids})
    expected = list(freeze["expected_test_signers"])
    if signers != expected:
        raise SystemExit(f"test_signers_mismatch:{signers}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type != "cuda":
        raise SystemExit("oneshot_test_requires_cuda")

    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_v5)
    seed = int(auth["primary_seed"])
    per_variant: dict[str, Any] = {}
    pred_by_variant: dict[str, dict[str, PredRow]] = {}

    for variant in variants:
        meta_ck = freeze["checkpoints_seed42"][variant]
        ckpt_path = _ROOT / meta_ck["checkpoint_rel"]
        if sha256_file(ckpt_path) != meta_ck["checkpoint_sha256"]:
            raise SystemExit(f"ckpt_sha_mismatch:{variant}")
        print(json.dumps({"event": "test_eval_start", "variant": variant, "seed": seed}), flush=True)
        model = build_ms_sta_v5(
            variant,
            num_classes=400,
            d_model=int(policy["model"]["d_model"]),
            nhead=int(policy["model"]["nhead"]),
            temporal_layers=int(policy["model"]["temporal_layers"]),
            dropout=float(meta_ck["dropout"]),
            alpha_max=float(policy["model"]["alpha_max"]),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        t0 = time.perf_counter()
        metrics = evaluate(model, loader, device, collect_rows=True)
        elapsed = time.perf_counter() - t0
        rows = metrics.pop("rows")
        pred_by_variant[variant] = _rows_to_pred(rows)
        per_variant[variant] = {
            "top1": metrics["top1"],
            "top5": metrics["top5"],
            "macro_f1": metrics["macro_f1"],
            "n": int(metrics["n"]),
            "elapsed_sec": elapsed,
            "checkpoint_rel": meta_ck["checkpoint_rel"],
            "checkpoint_sha256": meta_ck["checkpoint_sha256"],
        }
        # Persist per-sample preds create-only.
        atomic_write(
            OUT_DIR / f"TEST_PREDS_{variant}_seed{seed}.json",
            {"variant": variant, "seed": seed, "rows": rows},
        )
        print(
            json.dumps(
                {
                    "event": "test_eval_done",
                    "variant": variant,
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "macro_f1": metrics["macro_f1"],
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    _, d1, d5, signers_list, signer_deltas = _paired_deltas(
        pred_by_variant["b4_star_fixed"], pred_by_variant["b3_mlp_roi_fixed"]
    )
    mean_pp, ci_lo, ci_hi = hierarchical_bootstrap_ci(d1, signers_list, n_boot=10000, seed=42)
    mean5_pp, ci5_lo, ci5_hi = hierarchical_bootstrap_ci(d5, signers_list, n_boot=10000, seed=42)

    report = {
        "schema": "ms_sta_v5_oneshot_test_report_v1",
        "status": "TEST_COMPLETE",
        "retrospective_signer_disjoint": True,
        "pristine_blind_claim": False,
        "proposed_final_model": "b4_star_fixed",
        "primary_contrast": "b4_star_fixed_vs_b3_mlp_roi_fixed",
        "seed": seed,
        "head": git_rev(_ROOT),
        "auth_sha256": sha256_file(_ROOT / AUTH_REL),
        "freeze_sha256": sha256_file(_ROOT / FREEZE_REL),
        "split_manifest_sha256": auth["split_manifest_sha256"],
        "test_n": len(test_ids),
        "test_signers": signers,
        "variants": per_variant,
        "b4_minus_mlp": {
            "delta_top1": per_variant["b4_star_fixed"]["top1"] - per_variant["b3_mlp_roi_fixed"]["top1"],
            "delta_top5": per_variant["b4_star_fixed"]["top5"] - per_variant["b3_mlp_roi_fixed"]["top5"],
            "delta_macro_f1": per_variant["b4_star_fixed"]["macro_f1"] - per_variant["b3_mlp_roi_fixed"]["macro_f1"],
            "hierarchical_mean_delta_top1_pp": mean_pp,
            "hierarchical_ci95_top1_pp": [ci_lo, ci_hi],
            "hierarchical_mean_delta_top5_pp": mean5_pp,
            "hierarchical_ci95_top5_pp": [ci5_lo, ci5_hi],
            "per_signer_delta_top1_pp": signer_deltas,
        },
        "b4_e_evaluated": False,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_sha = atomic_write(OUT_DIR / "TEST_REPORT.json", report)
    print(json.dumps({"event": "ONESHOT_TEST_COMPLETE", "report_sha256": report_sha, "b4_top1": per_variant["b4_star_fixed"]["top1"], "mlp_top1": per_variant["b3_mlp_roi_fixed"]["top1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
