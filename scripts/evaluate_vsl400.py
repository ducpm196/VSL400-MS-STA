"""Evaluate VSL400 word-level Top-1/Top-5 (front) + optional multi-view late fusion.

Canonical KPI: full SI test with ``--limit 0`` (default), locked checkpoint, once.
Training must not touch test; this evaluator is the only authorized test pass (Gate 3).

``--limit N`` is a diagnostic subset only — never claim multi-view gains from first-N iloc.
Default ``--cache-mode read_only`` keeps the benchmark from writing/mixing feature caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_prep.raw_video_resolve import RawVideoResolver
from src.load_env import load_project_env
from src.models.isolated_sign_classifier import DEFAULT_CKPT, DEFAULT_LABELS, IsolatedSignClassifier, IsolatedSignHead
from src.models.rgb_backbone import SlowFastFeatureExtractor
from src.training.train_vsl400_wordlevel import FEAT_CACHE, extract_clip_feat2304

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict:
    rev = dirty = None
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip() or None
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
    except Exception:
        pass
    return {"git_revision": rev, "git_dirty": dirty}


def _sample_test(df: pd.DataFrame, limit: int, mode: str, seed: int) -> pd.DataFrame:
    """Select a diagnostic subset. Full test = limit<=0 (caller)."""
    if limit <= 0 or limit >= len(df):
        return df.reset_index(drop=True)
    mode = mode.lower().strip()
    if mode == "first":
        return df.iloc[:limit].reset_index(drop=True)
    if mode == "random":
        return df.sample(n=limit, random_state=seed).reset_index(drop=True)
    if mode == "stratified":
        glosses = df["text_gloss"].astype(str)
        parts: list[pd.DataFrame] = []
        taken: set = set()
        for _, gdf in df.groupby(glosses, sort=True):
            pick = gdf.sample(n=1, random_state=seed)
            parts.append(pick)
            taken.update(pick.index.tolist())
        out = pd.concat(parts)
        if len(out) > limit:
            return out.sample(n=limit, random_state=seed).reset_index(drop=True)
        need = limit - len(out)
        leftover = df.loc[~df.index.isin(taken)]
        if need > 0 and len(leftover) > 0:
            out = pd.concat([out, leftover.sample(n=min(need, len(leftover)), random_state=seed)])
        return out.reset_index(drop=True)
    raise ValueError(f"Unknown --sample mode: {mode}")


def _load_feat(
    *,
    cache_p: Path,
    video_path: Path,
    extractor: SlowFastFeatureExtractor,
    cache_mode: str,
) -> np.ndarray:
    if cache_p.is_file():
        return np.load(cache_p).astype(np.float32)
    if cache_mode == "read_only":
        raise FileNotFoundError(f"cache miss (read_only): {cache_p.name}")
    feat = extract_clip_feat2304(extractor, video_path)
    if cache_mode == "write":
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_p, feat)
    return feat


@torch.no_grad()
def logits_from_feat(head: IsolatedSignHead, feat: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits, _ = head(feat.to(device).unsqueeze(0))
    return logits.squeeze(0)


def main() -> None:
    load_project_env()
    ap = argparse.ArgumentParser(
        description=(
            "VSL400 SI evaluation (authorized test pass). "
            "Default cache-mode=read_only. --limit is diagnostic only."
        )
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = full SI test (canonical)")
    ap.add_argument(
        "--sample",
        choices=["first", "random", "stratified"],
        default="first",
        help="How to pick --limit rows (first=iloc[:N], often single-signer — diagnostic only)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fusion", action="store_true", help="Late-fuse front+left+right logits")
    ap.add_argument(
        "--cache-mode",
        choices=["read_only", "write", "ephemeral"],
        default="read_only",
        help="read_only=Gate3 (no cache write); write=allow np.save; ephemeral=extract without save",
    )
    ap.add_argument("--ckpt", type=Path, default=_ROOT / DEFAULT_CKPT)
    ap.add_argument("--labels", type=Path, default=_ROOT / DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=_ROOT / "logs" / "eval_vsl400" / "eval.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt if args.ckpt.is_absolute() else _ROOT / args.ckpt
    labels = args.labels if args.labels.is_absolute() else _ROOT / args.labels
    clf = IsolatedSignClassifier(ckpt_path=ckpt, labels_path=labels, device=device)
    head = clf.head
    extractor = clf.extractor
    resolver = RawVideoResolver(_ROOT)
    meta_path = _ROOT / "metadata.csv"
    evaluator_path = Path(__file__).resolve()

    df = pd.read_csv(meta_path)
    test_full = df[(df["dataset"] == "VSL400") & (df["split"] == "test")].reset_index(drop=True)
    expected_n = len(test_full)
    diagnostic = bool(args.limit and args.limit > 0)
    test = _sample_test(test_full, args.limit, args.sample, args.seed) if diagnostic else test_full
    signers = sorted(test["signer_id"].astype(str).unique().tolist()) if "signer_id" in test.columns else []
    n_classes_in_slice = int(test["text_gloss"].nunique()) if len(test) else 0

    gloss_to_id = {g: i for i, g in enumerate(clf.id_to_gloss)}
    top1 = top5 = n = 0
    fuse_top1 = fuse_top5 = fuse_n = 0
    rows = []
    skip_reasons: Counter[str] = Counter()
    skipped: list[dict] = []

    print(
        f"[*] VSL400 eval expected_full={expected_n} n_slice={len(test)} "
        f"classes_model={clf.num_classes} classes_slice={n_classes_in_slice} "
        f"signers={signers} fusion={args.fusion} diagnostic={diagnostic} "
        f"sample={args.sample} cache_mode={args.cache_mode}"
    )
    if diagnostic and args.sample == "first":
        print(
            "[WARN] --limit with --sample first uses iloc[:N] (often one signer / partial classes). "
            "Report as single-signer diagnostic subset — not full SI multi-view KPI."
        )

    for i, row in test.iterrows():
        gold = str(row["text_gloss"])
        if gold not in gloss_to_id:
            skip_reasons["unknown_gloss"] += 1
            skipped.append({"video_id": str(row.get("video_id")), "reason": "unknown_gloss", "gold": gold})
            continue
        gold_id = gloss_to_id[gold]
        vid = str(row["video_id"])
        front_path = resolver.resolve(row, view="front")
        if front_path is None or not front_path.is_file():
            skip_reasons["missing_file"] += 1
            skipped.append({"video_id": vid, "reason": "missing_file"})
            continue

        cache_p = FEAT_CACHE / f"{vid}.npy"
        try:
            feat_np = _load_feat(
                cache_p=cache_p,
                video_path=front_path,
                extractor=extractor,
                cache_mode=args.cache_mode,
            )
        except Exception as exc:
            reason = "decode_or_extract_error" if "cache miss" not in str(exc) else "cache_miss"
            if "cache miss" in str(exc):
                reason = "cache_miss"
            skip_reasons[reason] += 1
            skipped.append({"video_id": vid, "reason": reason, "error": str(exc)[:240]})
            print(f"[WARN] skip {vid}: {exc}")
            continue

        feat = torch.from_numpy(feat_np)
        logits = logits_from_feat(head, feat, device)
        prob = F.softmax(logits, dim=-1)
        vals, idxs = torch.topk(prob, k=min(5, clf.num_classes))
        pred_ids = [int(x) for x in idxs.tolist()]
        hit1 = int(pred_ids[0] == gold_id)
        hit5 = int(gold_id in pred_ids)
        top1 += hit1
        top5 += hit5
        n += 1
        row_out = {
            "video_id": vid,
            "gold": gold,
            "pred": clf.id_to_gloss[pred_ids[0]],
            "top5": [clf.id_to_gloss[j] for j in pred_ids],
            "top1": bool(hit1),
            "top5_hit": bool(hit5),
            "prob": float(vals[0]),
            "signer_id": str(row.get("signer_id", "")),
            "views_used": ["front"],
        }

        if args.fusion and int(row.get("has_left", 0)) and int(row.get("has_right", 0)):
            acc_logits = [logits]
            views_used = ["front"]
            for view in ("left", "right"):
                p = resolver.resolve(row, view=view)
                if p is None or not p.is_file():
                    skip_reasons[f"fusion_{view}_missing"] += 1
                    continue
                stem = f"{vid}_{view}"
                cp = FEAT_CACHE / f"{stem}.npy"
                try:
                    f_np = _load_feat(
                        cache_p=cp,
                        video_path=p,
                        extractor=extractor,
                        cache_mode=args.cache_mode,
                    )
                except Exception as exc:
                    skip_reasons[f"fusion_{view}_error"] += 1
                    print(f"[WARN] fusion skip {vid}/{view}: {exc}")
                    continue
                acc_logits.append(logits_from_feat(head, torch.from_numpy(f_np), device))
                views_used.append(view)
            if len(acc_logits) >= 2:
                fused = torch.stack(acc_logits, dim=0).mean(dim=0)
                fp = F.softmax(fused, dim=-1)
                fvals, fidxs = torch.topk(fp, k=min(5, clf.num_classes))
                fids = [int(x) for x in fidxs.tolist()]
                f1 = int(fids[0] == gold_id)
                f5 = int(gold_id in fids)
                fuse_top1 += f1
                fuse_top5 += f5
                fuse_n += 1
                row_out["views_used"] = views_used
                row_out["fusion_pred"] = clf.id_to_gloss[fids[0]]
                row_out["fusion_top5"] = [clf.id_to_gloss[j] for j in fids]
                row_out["fusion_top1"] = bool(f1)
                row_out["fusion_top5_hit"] = bool(f5)
                row_out["fusion_n_views"] = len(acc_logits)

        rows.append(row_out)
        if (i + 1) % 100 == 0 or (i + 1) == len(test):
            print(f"  [{n}] top1={100*top1/max(n,1):.1f}% top5={100*top5/max(n,1):.1f}%")

    skipped_count = int(sum(skip_reasons.values()))
    git = _git_state()
    summary = {
        "n": n,
        "expected_n": expected_n,
        "slice_n": len(test),
        "skipped_count": skipped_count,
        "skip_reasons": dict(skip_reasons),
        "top1_acc": 100.0 * top1 / max(n, 1),
        "top5_acc": 100.0 * top5 / max(n, 1),
        "num_classes": clf.num_classes,
        "classes_in_slice": n_classes_in_slice,
        "signers_in_slice": signers,
        "dataset": "VSL400",
        "split": "test",
        "doi": "10.5281/zenodo.17943573",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256_file(ckpt),
        "labels_sha256": _sha256_file(labels),
        "metadata_sha256": _sha256_file(meta_path),
        "evaluator_sha256": _sha256_file(evaluator_path),
        "feat_cache_dir": str(FEAT_CACHE),
        "cache_mode": args.cache_mode,
        "diagnostic_subset": diagnostic,
        "sample_mode": args.sample if diagnostic else "full",
        "limit": args.limit,
        "product": "VSL400 signer-independent word-level recognition with human-assisted output",
        **git,
    }
    if fuse_n:
        summary["fusion_n"] = fuse_n
        summary["fusion_top1_acc"] = 100.0 * fuse_top1 / fuse_n
        summary["fusion_top5_acc"] = 100.0 * fuse_top5 / fuse_n
        summary["fusion_scope"] = (
            "diagnostic_subset_not_full_si"
            if diagnostic
            else "full_si_late_fusion"
        )
        summary["fusion_note"] = (
            "Naive late fusion of a front-only-trained head. "
            "Absence of SI gain does not imply multi-view learning is ineffective."
        )
        if diagnostic:
            summary["fusion_disclaimer"] = (
                "Fusion metrics on --limit slice are not full multi-signer SI evidence. "
                "Use --limit 0 --fusion for canonical multi-view KPI."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "skipped": skipped, "rows": rows}
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    args.out.write_text(raw, encoding="utf-8")
    summary["report_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # rewrite with report hash
    payload["summary"] = summary
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if skipped_count:
        print(f"[WARN] skipped_count={skipped_count} reasons={dict(skip_reasons)}")
    if summary["top1_acc"] < 5.0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
