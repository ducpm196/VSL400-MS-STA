"""Aligned validation gates for retrain-v4 (exact ID contract; no silent drops).

Join is always by sample_id. Sets across seeds/variants must be identical to the
frozen val ID list (n_val from policy/manifest).

Primary Macro-F1 gate: per-seed F1 → mean ± sample std (not 3-seed majority vote).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.retrain_v4 import (
    CONFIRM_SEEDS,
    exact_id_list_sha256,
    evaluate_architecture_gates_v4,
    lineage_block,
    load_per_sample_manifest,
    load_policy,
    require_exact_str,
    require_frozen_v4_policy,
    require_hex64,
    require_per_sample_manifest_contract,
)
from src.training.retrain_v4_stats import (
    align_rows_by_sample_id,
    hierarchical_paired_bootstrap_delta_ci,
    seed_average_correctness,
)
from src.training.retrain_v3_stats import macro_f1, signer_metrics


def load_rows_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _require_label_range(value: object, *, label: str, num_classes: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} not int: {value!r}") from exc
    if v < 0 or v >= int(num_classes):
        raise ValueError(f"{label}={v} out of range [0,{num_classes - 1}]")
    return v


def verify_row_contract(
    row: dict[str, Any],
    *,
    seed: int,
    variant: str,
    num_classes: int,
    expected_policy_sha: str,
    expected_final_ckpt_sha: str | None,
    signer_by_id: dict[str, str] | None = None,
) -> None:
    """Fail-closed per-row evidence contract (exact SHA + recomputed correctness)."""
    require_exact_str(row.get("variant"), variant, label="row.variant")
    if int(row.get("seed", -1)) != int(seed):
        raise ValueError(f"row seed mismatch: {row.get('seed')!r} != {seed!r}")
    if "sample_id" not in row or "signer_id" not in row:
        raise ValueError("row missing sample_id/signer_id")
    if "y_pred" not in row:
        raise ValueError("row missing y_pred")
    if "correct_top5" not in row:
        raise ValueError("row missing correct_top5")
    if "top5" not in row or not isinstance(row.get("top5"), list) or len(row["top5"]) != 5:
        raise ValueError("row.top5 must be a length-5 class-id list")

    y_true = _require_label_range(row.get("y_true"), label="y_true", num_classes=num_classes)
    y_pred = _require_label_range(row.get("y_pred"), label="y_pred", num_classes=num_classes)
    top5 = [
        _require_label_range(x, label="top5[i]", num_classes=num_classes) for x in row["top5"]
    ]
    expect_top1 = int(y_pred == y_true)
    expect_top5 = int(y_true in top5)
    if int(row.get("correct_top1", -1)) != expect_top1:
        raise ValueError(
            f"correct_top1 mismatch for {row.get('sample_id')}: "
            f"{row.get('correct_top1')} != recomputed {expect_top1}"
        )
    if int(row.get("correct_top5", -1)) != expect_top5:
        raise ValueError(
            f"correct_top5 mismatch for {row.get('sample_id')}: "
            f"{row.get('correct_top5')} != recomputed {expect_top5}"
        )

    policy_sha = require_hex64(row.get("policy_sha256"), label="row.policy_sha256")
    require_exact_str(policy_sha, expected_policy_sha, label="row.policy_sha256")
    ckpt_sha = require_hex64(
        row.get("final_checkpoint_sha256"), label="row.final_checkpoint_sha256"
    )
    if expected_final_ckpt_sha is not None:
        require_exact_str(
            ckpt_sha, expected_final_ckpt_sha, label="row.final_checkpoint_sha256"
        )

    if signer_by_id is not None:
        sid = str(row["sample_id"])
        if sid not in signer_by_id:
            raise ValueError(f"sample_id {sid} missing from metadata signer map")
        require_exact_str(
            row.get("signer_id"), signer_by_id[sid], label="row.signer_id"
        )


def require_exact_id_set(
    rows: list[dict],
    *,
    expected_ids: list[str],
    seed: int,
    variant: str,
    num_classes: int = 400,
    expected_policy_sha: str | None = None,
    expected_final_ckpt_sha: str | None = None,
    signer_by_id: dict[str, str] | None = None,
) -> None:
    ids = [str(r["sample_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample_id in {variant}/seed{seed}")
    got = set(ids)
    exp = set(expected_ids)
    if got != exp:
        missing = sorted(exp - got)[:5]
        extra = sorted(got - exp)[:5]
        raise ValueError(
            f"{variant}/seed{seed} ID set mismatch "
            f"|got|={len(got)} |exp|={len(exp)} missing_ex={missing} extra_ex={extra}"
        )
    if len(ids) != len(expected_ids):
        raise ValueError(f"{variant}/seed{seed} n={len(ids)} != n_val={len(expected_ids)}")
    if expected_policy_sha is None:
        raise ValueError("expected_policy_sha required (None not allowed)")
    require_hex64(expected_policy_sha, label="expected_policy_sha")
    for r in rows:
        verify_row_contract(
            r,
            seed=seed,
            variant=variant,
            num_classes=num_classes,
            expected_policy_sha=expected_policy_sha,
            expected_final_ckpt_sha=expected_final_ckpt_sha,
            signer_by_id=signer_by_id,
        )


def _per_signer_seed_avg(
    *,
    sample_ids: list[str],
    signers: list[str],
    avg_b3: np.ndarray,
    avg_b4r: np.ndarray,
) -> dict[str, Any]:
    by: dict[str, dict[str, float | int]] = {}
    for s in sorted(set(signers)):
        idx = [i for i, ss in enumerate(signers) if ss == s]
        if not idx:
            continue
        b3 = float(np.mean(avg_b3[idx]) * 100.0)
        b4r = float(np.mean(avg_b4r[idx]) * 100.0)
        by[s] = {
            "n": int(len(idx)),
            "b3_top1": b3,
            "b4r_top1": b4r,
            "delta_top1_pp": float(b4r - b3),
        }
    worst = (
        min(by.items(), key=lambda kv: float(kv[1]["delta_top1_pp"])) if by else None
    )
    return {
        "unit": "seed_averaged_per_sample_correctness",
        "per_signer": by,
        "worst_signer_by_delta": (
            {"signer": worst[0], **worst[1]} if worst else None
        ),
    }


def build_gate_report(
    *,
    rows_b3_by_seed: dict[int, list[dict]],
    rows_b4r_by_seed: dict[int, list[dict]],
    expected_ids: list[str],
    expected_policy_sha: str,
    expected_final_ckpt_sha_by_variant_seed: dict[tuple[str, int], str],
    n_boot: int = 10_000,
    seed: int = 42,
    num_classes: int = 400,
    signer_by_id: dict[str, str] | None = None,
    accuracy_gates: dict | None = None,
) -> dict:
    if sorted(rows_b3_by_seed.keys()) != list(CONFIRM_SEEDS):
        raise ValueError(f"b3 seeds must be {CONFIRM_SEEDS}")
    if sorted(rows_b4r_by_seed.keys()) != list(CONFIRM_SEEDS):
        raise ValueError(f"b4r seeds must be {CONFIRM_SEEDS}")
    if len(expected_ids) <= 0:
        raise ValueError("expected_ids empty")
    require_hex64(expected_policy_sha, label="expected_policy_sha")
    if expected_final_ckpt_sha_by_variant_seed is None:
        raise ValueError("expected_final_ckpt_sha_by_variant_seed required (None not allowed)")
    expected_keys = {(v, s) for v in ("b3", "b4r") for s in CONFIRM_SEEDS}
    got_keys = set(expected_final_ckpt_sha_by_variant_seed)
    if got_keys != expected_keys:
        raise ValueError(
            f"expected_final_ckpt_sha_by_variant_seed keys {sorted(got_keys)} "
            f"!= {sorted(expected_keys)}"
        )
    for key, sha in expected_final_ckpt_sha_by_variant_seed.items():
        require_hex64(sha, label=f"expected_final_ckpt_sha[{key}]")

    for s in CONFIRM_SEEDS:
        ckpt_b3 = expected_final_ckpt_sha_by_variant_seed[("b3", s)]
        ckpt_b4r = expected_final_ckpt_sha_by_variant_seed[("b4r", s)]
        require_exact_id_set(
            rows_b3_by_seed[s],
            expected_ids=expected_ids,
            seed=s,
            variant="b3",
            num_classes=num_classes,
            expected_policy_sha=expected_policy_sha,
            expected_final_ckpt_sha=ckpt_b3,
            signer_by_id=signer_by_id,
        )
        require_exact_id_set(
            rows_b4r_by_seed[s],
            expected_ids=expected_ids,
            seed=s,
            variant="b4r",
            num_classes=num_classes,
            expected_policy_sha=expected_policy_sha,
            expected_final_ckpt_sha=ckpt_b4r,
            signer_by_id=signer_by_id,
        )

    ids_b3, signers_b3, corr_b3 = align_rows_by_sample_id(rows_b3_by_seed)
    ids_b4r, signers_b4r, corr_b4r = align_rows_by_sample_id(rows_b4r_by_seed)
    if ids_b3 != expected_ids or ids_b4r != expected_ids:
        if ids_b3 != sorted(expected_ids) or ids_b4r != sorted(expected_ids):
            raise ValueError("aligned IDs must equal sorted expected val IDs")
    if signers_b3 != signers_b4r:
        raise ValueError("signer vectors differ between b3 and b4r after alignment")

    _, avg_b3 = seed_average_correctness(ids_b3, corr_b3, seed_order=list(CONFIRM_SEEDS))
    _, avg_b4r = seed_average_correctness(ids_b4r, corr_b4r, seed_order=list(CONFIRM_SEEDS))
    paired = hierarchical_paired_bootstrap_delta_ci(
        avg_b3,
        avg_b4r,
        signers_b3,
        n_boot=n_boot,
        seed=seed,
        label_a="b3",
        label_b="b4r",
    )

    per_seed: dict[str, dict[str, float]] = {}
    top5_deltas: list[float] = []
    f1_b3_seeds: list[float] = []
    f1_b4r_seeds: list[float] = []
    y_true_ref = None
    pred_b3_stack: list[np.ndarray] = []
    pred_b4r_stack: list[np.ndarray] = []
    for s in CONFIRM_SEEDS:
        by_id_b3 = {str(r["sample_id"]): r for r in rows_b3_by_seed[s]}
        by_id_b4r = {str(r["sample_id"]): r for r in rows_b4r_by_seed[s]}
        ordered = ids_b3
        c3 = np.array([float(by_id_b3[i]["correct_top1"]) for i in ordered], dtype=float)
        c4 = np.array([float(by_id_b4r[i]["correct_top1"]) for i in ordered], dtype=float)
        t5_3 = np.array([float(by_id_b3[i]["correct_top5"]) for i in ordered], dtype=float)
        t5_4 = np.array([float(by_id_b4r[i]["correct_top5"]) for i in ordered], dtype=float)
        y = np.array([int(by_id_b3[i]["y_true"]) for i in ordered], dtype=np.int64)
        if y_true_ref is None:
            y_true_ref = y
        elif not np.array_equal(y_true_ref, y):
            raise ValueError(f"y_true mismatch across seeds at seed={s}")
        y4 = np.array([int(by_id_b4r[i]["y_true"]) for i in ordered], dtype=np.int64)
        if not np.array_equal(y, y4):
            raise ValueError(f"y_true mismatch b3 vs b4r at seed={s}")
        p3 = np.array([int(by_id_b3[i]["y_pred"]) for i in ordered], dtype=np.int64)
        p4 = np.array([int(by_id_b4r[i]["y_pred"]) for i in ordered], dtype=np.int64)
        pred_b3_stack.append(p3)
        pred_b4r_stack.append(p4)
        f1_3 = float(macro_f1(y, p3, num_classes))
        f1_4 = float(macro_f1(y, p4, num_classes))
        f1_b3_seeds.append(f1_3)
        f1_b4r_seeds.append(f1_4)
        per_seed[str(s)] = {
            "b3_top1": float(c3.mean() * 100.0),
            "b4r_top1": float(c4.mean() * 100.0),
            "b3_top5": float(t5_3.mean() * 100.0),
            "b4r_top5": float(t5_4.mean() * 100.0),
            "b3_macro_f1": f1_3,
            "b4r_macro_f1": f1_4,
            "delta_top1_pp": float((c4.mean() - c3.mean()) * 100.0),
            "delta_top5_pp": float((t5_4.mean() - t5_3.mean()) * 100.0),
            "delta_macro_f1_pp": float(f1_4 - f1_3),
        }
        top5_deltas.append(float((t5_4.mean() - t5_3.mean()) * 100.0))

    assert y_true_ref is not None
    mean_f1_b3 = float(np.mean(f1_b3_seeds))
    mean_f1_b4r = float(np.mean(f1_b4r_seeds))
    std_f1_b3 = float(np.std(f1_b3_seeds, ddof=1)) if len(f1_b3_seeds) > 1 else 0.0
    std_f1_b4r = float(np.std(f1_b4r_seeds, ddof=1)) if len(f1_b4r_seeds) > 1 else 0.0

    # Secondary analysis only (registered; not the primary gate).
    vote_b3 = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=num_classes).argmax(),
        0,
        np.stack(pred_b3_stack, axis=0),
    )
    vote_b4r = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=num_classes).argmax(),
        0,
        np.stack(pred_b4r_stack, axis=0),
    )
    vote_f1_b3 = float(macro_f1(y_true_ref, vote_b3, num_classes))
    vote_f1_b4r = float(macro_f1(y_true_ref, vote_b4r, num_classes))
    vote_signers = signer_metrics(y_true_ref, vote_b4r, signers_b3)

    per_signer = _per_signer_seed_avg(
        sample_ids=ids_b3,
        signers=signers_b3,
        avg_b3=avg_b3,
        avg_b4r=avg_b4r,
    )

    b3_vals = [per_seed[str(s)]["b3_top1"] for s in CONFIRM_SEEDS]
    b4r_vals = [per_seed[str(s)]["b4r_top1"] for s in CONFIRM_SEEDS]
    mean_b3 = float(np.mean(b3_vals))
    mean_b4r = float(np.mean(b4r_vals))
    std_b3 = float(np.std(b3_vals, ddof=1)) if len(b3_vals) > 1 else 0.0
    std_b4r = float(np.std(b4r_vals, ddof=1)) if len(b4r_vals) > 1 else 0.0
    mean_delta_top5 = float(np.mean(top5_deltas))

    gates_cfg = accuracy_gates or {}
    gates = evaluate_architecture_gates_v4(
        per_seed=per_seed,
        paired_ci=paired,
        mean_delta_top5_pp=mean_delta_top5,
        macro_f1_b3=mean_f1_b3,
        macro_f1_b4r=mean_f1_b4r,
        mean_delta_top1_ge_pp=float(gates_cfg.get("mean_delta_top1_ge_pp", 1.0)),
        mean_delta_top5_ge_pp=float(gates_cfg.get("mean_delta_top5_ge_pp", -1.0)),
    )
    bindings = {
        "policy_sha256": expected_policy_sha,
        "schema": "ms_sta_retrain_v4_paired_gate_v3",
        "evaluator": "scripts/evaluate_ms_sta_retrain_v4_validation.py",
        "canonical_producer": True,
        "final_checkpoint_sha256_by_variant_seed": {
            f"{v}/seed{s}": sha
            for (v, s), sha in sorted(
                expected_final_ckpt_sha_by_variant_seed.items(),
                key=lambda kv: (kv[0][0], kv[0][1]),
            )
        },
    }
    return {
        "schema": "ms_sta_retrain_v4_paired_gate_v3",
        "n_samples": len(ids_b3),
        "n_val_expected": len(expected_ids),
        "join_key": "sample_id",
        "exact_id_contract": True,
        "forbid_silent_intersection": True,
        "unit_of_analysis": "seed_averaged_per_sample_correctness",
        "macro_f1_primary": "per_seed_then_mean_pm_sample_std",
        "per_seed": per_seed,
        "summary": {
            "b3_top1_mean_pm_sample_std": [mean_b3, std_b3],
            "b4r_top1_mean_pm_sample_std": [mean_b4r, std_b4r],
            "mean_delta_top1_pp": gates["mean_delta_top1_pp"],
            "mean_delta_top5_pp": mean_delta_top5,
            "macro_f1_b3_mean_pm_sample_std": [mean_f1_b3, std_f1_b3],
            "macro_f1_b4r_mean_pm_sample_std": [mean_f1_b4r, std_f1_b4r],
            "macro_f1_b3": mean_f1_b3,
            "macro_f1_b4r": mean_f1_b4r,
        },
        "per_signer": per_signer.get("per_signer"),
        "worst_signer_by_delta": per_signer.get("worst_signer_by_delta"),
        "paired_bootstrap": paired,
        "architecture_gates": gates,
        "bindings": bindings,
        "secondary_analyses": {
            "seed_majority_vote": {
                "registered_as_primary": False,
                "macro_f1_b3": vote_f1_b3,
                "macro_f1_b4r": vote_f1_b4r,
                "per_signer_b4r_only": vote_signers.get("per_signer"),
                "worst_signer_b4r_only": vote_signers.get("worst_signer"),
                "note": "Diagnostic only; primary Macro-F1 uses per-seed mean+/-sample std.",
            }
        },
    }


def _norm_signer(value: object) -> str:
    s = str(value).strip()
    if s.startswith("signer_"):
        return s
    if s.isdigit() or s.lstrip("0").isdigit() or s == "0":
        return f"signer_{int(s)}"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument(
        "--per-sample-manifest",
        type=Path,
        required=True,
        help="Atomic PER_SAMPLE_MANIFEST.json from export_ms_sta_retrain_v4_per_sample.py",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    args = ap.parse_args()

    policy_path = args.policy if args.policy.is_absolute() else _ROOT / args.policy
    policy = load_policy(policy_path)
    from src.models.feature_versioning import sha256_file
    import hashlib
    import pandas as pd

    try:
        require_frozen_v4_policy(policy, root=_ROOT)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    policy_sha = sha256_file(policy_path)
    lineage = lineage_block(policy)
    split_rel = lineage.get("split_manifest")
    if not split_rel:
        print("[FAIL] policy missing split_manifest")
        return 2
    split_path = _ROOT / str(split_rel)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha = require_hex64(
        lineage.get("split_manifest_sha256"), label="lineage.split_manifest_sha256"
    )
    if sha256_file(split_path) != split_sha:
        print("[FAIL] split sha mismatch vs lineage")
        return 2
    expected_ids = sorted(str(x) for x in split.get("val_ids") or [])
    n_val = int(lineage.get("n_val") or split.get("n_val") or 0)
    if n_val and len(expected_ids) != n_val:
        print(f"[FAIL] val_ids n={len(expected_ids)} != n_val={n_val}")
        return 2

    man_path = (
        args.per_sample_manifest
        if args.per_sample_manifest.is_absolute()
        else _ROOT / args.per_sample_manifest
    )
    if not man_path.is_file():
        print(f"[FAIL] missing per-sample manifest: {man_path}")
        return 2
    try:
        manifest = load_per_sample_manifest(man_path)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    man_sha = sha256_file(man_path)
    git_rev = str(manifest.get("git_revision") or "")
    try:
        ckpt_map = require_per_sample_manifest_contract(
            manifest,
            expected_policy_sha=policy_sha,
            expected_split_sha=split_sha,
            expected_git_rev=git_rev,
        )
        require_exact_str(
            manifest.get("exact_id_sha256"),
            exact_id_list_sha256(expected_ids),
            label="manifest.exact_id_sha256",
        )
        if int(manifest.get("n_val") or -1) != len(expected_ids):
            raise ValueError("manifest.n_val mismatch vs split val_ids")
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    meta = pd.read_csv(_ROOT / "metadata.csv")
    labels_sha = sha256_file(_ROOT / "data/processed/vsl400_labels.json")
    meta_sha = sha256_file(_ROOT / "metadata.csv")
    expected_signer_meta = hashlib.sha256(
        (meta_sha + ":" + labels_sha).encode("utf-8")
    ).hexdigest()
    try:
        require_exact_str(
            manifest.get("signer_metadata_sha256"),
            expected_signer_meta,
            label="manifest.signer_metadata_sha256",
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    v = meta[meta["dataset"] == "VSL400"].copy()
    v["video_id"] = v["video_id"].astype(str)
    signer_by_id = {
        str(r.video_id): _norm_signer(r.signer_id) for r in v.itertuples(index=False)
    }

    rows_b3: dict[int, list[dict]] = {}
    rows_b4r: dict[int, list[dict]] = {}
    path_map = manifest.get("jsonl_paths_by_variant_seed") or {}
    jsonl_shas = manifest.get("jsonl_sha256_by_variant_seed") or {}
    for s in CONFIRM_SEEDS:
        for variant, bucket in (("b3", rows_b3), ("b4r", rows_b4r)):
            key = f"{variant}/seed{s}"
            rel = path_map.get(key)
            if not rel:
                print(f"[FAIL] manifest missing path for {key}")
                return 2
            path = _ROOT / str(rel)
            if not path.is_file():
                print(f"[FAIL] missing jsonl {path}")
                return 2
            actual = sha256_file(path)
            try:
                require_exact_str(actual, jsonl_shas[key], label=f"jsonl sha {key}")
            except ValueError as exc:
                print(f"[FAIL] {exc}")
                return 2
            bucket[s] = load_rows_jsonl(path)

    try:
        report = build_gate_report(
            rows_b3_by_seed=rows_b3,
            rows_b4r_by_seed=rows_b4r,
            expected_ids=expected_ids,
            n_boot=int(args.bootstrap),
            seed=int(args.bootstrap_seed),
            expected_policy_sha=policy_sha,
            expected_final_ckpt_sha_by_variant_seed=ckpt_map,
            signer_by_id=signer_by_id,
            accuracy_gates=policy.get("accuracy_gates") or {},
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    report["bindings"].update(
        {
            "split_manifest_sha256": split_sha,
            "git_revision": git_rev,
            "per_sample_manifest": str(man_path.relative_to(_ROOT)).replace("\\", "/"),
            "per_sample_manifest_sha256": man_sha,
            "exact_id_sha256": manifest.get("exact_id_sha256"),
            "signer_metadata_sha256": manifest.get("signer_metadata_sha256"),
            "jsonl_sha256_by_variant_seed": jsonl_shas,
        }
    )

    out = args.out if args.out.is_absolute() else _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out),
                "architecture_gate_pass": report["architecture_gates"]["architecture_gate_pass"],
                "n_samples": report["n_samples"],
                "macro_f1_primary": report["macro_f1_primary"],
                "per_sample_manifest_sha256": man_sha,
            },
            indent=2,
        )
    )
    return 0 if report["architecture_gates"]["architecture_gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
