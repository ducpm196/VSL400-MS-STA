"""VSL400 word-level recognition: SlowFast-2304 → 256 bottleneck → 400-class CE.

Primary thesis path (Zenodo VSL400). Front view train/eval; multi-view late fusion in evaluate_vsl400.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_prep.raw_video_resolve import RawVideoResolver
from src.load_env import load_project_env
from src.models.isolated_sign_classifier import IsolatedSignHead
from src.models.rgb_backbone import (
    INPUT_SIZE,
    RGB_DIM,
    WINDOW_SIZE,
    SlowFastFeatureExtractor,
    _read_video_frames,
    _window_tensors,
)
from src.models.utils import enforce_reproducibility_seed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

FEAT_CACHE = _ROOT / "data" / "features" / "isolated_sf2304"
OUT_CLS = _ROOT / "data" / "processed" / "vsl400_classifier.pth"
OUT_LABELS = _ROOT / "data" / "processed" / "vsl400_labels.json"
# Keep demo/legacy path in sync for IsolatedSignClassifier defaults after pivot.
OUT_CLS_LEGACY = _ROOT / "data" / "processed" / "isolated_sign_classifier.pth"
OUT_LABELS_LEGACY = _ROOT / "data" / "processed" / "isolated_sign_labels.json"
OUT_PROJ = _ROOT / "data" / "processed" / "slowfast_proj.pth"
STATUS = _ROOT / "logs" / "train_vsl400" / "status.json"


def _write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": datetime.now(timezone.utc).isoformat()}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_revision() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_video_frames_sparse(video_path: Path, num_frames: int = WINDOW_SIZE) -> np.ndarray:
    """Decode once; resize/cvt only uniformly sampled frames (faster than full-res pipeline)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    meta_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if meta_n <= 0:
        cap.release()
        return _read_video_frames(video_path)
    want = min(max(int(num_frames), 8), meta_n)
    keep = set(np.linspace(0, meta_n - 1, want).astype(int).tolist())
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in keep:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(resized.astype(np.float32) / 255.0)
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"Video rỗng: {video_path}")
    return np.stack(frames, axis=0)


@torch.no_grad()
def extract_clip_feat2304(
    extractor: SlowFastFeatureExtractor,
    video_path: Path | None = None,
    *,
    max_windows: int = 4,
    frames: np.ndarray | None = None,
) -> np.ndarray:
    # Sparse sample ≈ one SlowFast window worth of frames (word-level clips are short).
    if frames is None:
        if video_path is None:
            raise ValueError("video_path or frames required")
        frames = _read_video_frames_sparse(video_path, num_frames=WINDOW_SIZE)
    n = frames.shape[0]
    starts = list(range(0, max(n - WINDOW_SIZE + 1, 1), max(WINDOW_SIZE // 2, 8)))
    if not starts or starts[-1] + WINDOW_SIZE < n:
        starts.append(max(0, n - WINDOW_SIZE))
    if len(starts) > max_windows:
        picks = np.linspace(0, len(starts) - 1, max_windows).astype(int)
        starts = [starts[int(i)] for i in picks]
    feats = []
    device = extractor.device
    for start in starts:
        slow, fast = _window_tensors(frames, start, WINDOW_SIZE)
        slow = slow.to(device)
        fast = fast.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feat = extractor.backbone([slow, fast])
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            feats.append(feat.flatten(1).float().squeeze(0).cpu())
    return torch.stack(feats, dim=0).mean(dim=0).numpy().astype(np.float32)


class FeatDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, y: np.ndarray, cache_dir: Path) -> None:
        self.rows = rows.reset_index(drop=True)
        self.y = y.astype(np.int64)
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vid = str(self.rows.iloc[idx]["video_id"])
        feat = np.load(self.cache_dir / f"{vid}.npy").astype(np.float32)
        return torch.from_numpy(feat), torch.tensor(self.y[idx])


def main() -> None:
    load_project_env()
    enforce_reproducibility_seed(42)
    ap = argparse.ArgumentParser(description="VSL400 word-level SlowFast classifier")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--patience", type=int, default=12, help="Early-stop patience on val Top-5")
    ap.add_argument("--feat-noise", type=float, default=0.02, help="Train-time Gaussian noise on 2304 feats")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = all train samples/class")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--extract-limit", type=int, default=0)
    ap.add_argument(
        "--max-windows",
        type=int,
        default=2,
        help="Temporal windows averaged per clip (2=faster extract, still strong CE signal)",
    )
    ap.add_argument("--view", default="front", choices=["front", "left", "right"])
    ap.add_argument(
        "--final-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "FORBIDDEN for hyperparameter search. Default off: training never touches test. "
            "Canonical test is scripts/evaluate_vsl400.py after checkpoint lock (Gate 3)."
        ),
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("train_vsl400_wordlevel cần CUDA.")

    df = pd.read_csv(_ROOT / "metadata.csv")
    vsl = df[df["dataset"] == "VSL400"].copy()
    if vsl.empty:
        raise RuntimeError("metadata.csv thiếu VSL400 — chạy src/data_prep/build_metadata.py")

    glosses = sorted(vsl["text_gloss"].astype(str).unique().tolist())
    if len(glosses) != 400:
        print(f"[WARN] Expected 400 glosses, got {len(glosses)}")
    gloss_to_id = {g: i for i, g in enumerate(glosses)}
    viet = {}
    for g in glosses:
        vals = vsl.loc[vsl["text_gloss"].astype(str) == g, "text_vietnamese"].astype(str)
        viet[g] = vals.mode().iloc[0] if len(vals) else g
    label_meta = {
        "id_to_gloss": glosses,
        "id_to_vietnamese": [viet[g] for g in glosses],
        "gloss_to_id": gloss_to_id,
        "num_classes": len(glosses),
        "dataset": "VSL400",
        "view": args.view,
        "doi": "10.5281/zenodo.17943573",
    }
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.write_text(json.dumps(label_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_LABELS_LEGACY.write_text(json.dumps(label_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Cap train per class if requested
    parts = []
    for split_name in ("train", "val", "test"):
        sub = vsl[vsl["split"] == split_name]
        if split_name == "train" and args.max_per_class > 0:
            capped = []
            for _, gdf in sub.groupby(sub["text_gloss"].astype(str)):
                if len(gdf) > args.max_per_class:
                    gdf = gdf.sample(args.max_per_class, random_state=42)
                capped.append(gdf)
            sub = pd.concat(capped, ignore_index=True) if capped else sub
        parts.append(sub)
    pool = pd.concat(parts, ignore_index=True)

    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    resolver = RawVideoResolver(_ROOT)
    extractor = SlowFastFeatureExtractor(device=device)

    rows_need = pool
    if args.extract_limit and args.extract_limit > 0:
        rows_need = pool.iloc[: args.extract_limit].copy()

    if not args.skip_extract:
        from concurrent.futures import ThreadPoolExecutor

        print(
            f"[*] Extract SlowFast-2304 view={args.view} n={len(rows_need)} "
            f"max_windows={args.max_windows} → {FEAT_CACHE}"
        )
        t0 = time.perf_counter()
        done = 0
        cached = 0
        skipped = 0
        extracted = 0
        pending: list[tuple[str, Path, Path]] = []
        for _, row in rows_need.iterrows():
            vid = str(row["video_id"])
            cache_stem = vid if args.view == "front" else f"{vid}_{args.view}"
            out_p = FEAT_CACHE / f"{cache_stem}.npy"
            if out_p.is_file():
                done += 1
                cached += 1
                continue
            path = resolver.resolve(row, view=args.view)
            if path is None or not path.is_file():
                skipped += 1
                continue
            pending.append((vid, out_p, path))

        max_windows = max(1, int(args.max_windows))

        def _decode(path: Path) -> np.ndarray:
            return _read_video_frames_sparse(path, num_frames=WINDOW_SIZE)

        # Overlap CPU decode with GPU SlowFast (2-deep prefetch).
        with ThreadPoolExecutor(max_workers=2) as decode_pool:
            it = iter(pending)
            inflight: list = []
            for _ in range(2):
                try:
                    item = next(it)
                except StopIteration:
                    break
                inflight.append((item, decode_pool.submit(_decode, item[2])))
            while inflight:
                (vid, out_p, path), fut = inflight.pop(0)
                try:
                    frames = fut.result()
                    feat = extract_clip_feat2304(
                        extractor, path, max_windows=max_windows, frames=frames
                    )
                    np.save(out_p, feat)
                    done += 1
                    extracted += 1
                except Exception as exc:
                    skipped += 1
                    print(f"[WARN] {vid}: {exc}")
                try:
                    item = next(it)
                    inflight.append((item, decode_pool.submit(_decode, item[2])))
                except StopIteration:
                    pass
                if extracted > 0 and extracted % 25 == 0:
                    _write_status(
                        {
                            "phase": "extract",
                            "done": done,
                            "cached": cached,
                            "extracted": extracted,
                            "skipped": skipped,
                            "target": len(rows_need),
                            "elapsed_s": round(time.perf_counter() - t0, 1),
                        }
                    )
                    print(
                        f"  [{done}/{len(rows_need)}] cached={cached} extracted={extracted} "
                        f"skipped={skipped} elapsed={time.perf_counter()-t0:.0f}s"
                    )
        print(f"[OK] extract done={done} cached={cached} extracted={extracted} skipped={skipped}")

    def cache_ok(row: pd.Series) -> bool:
        vid = str(row["video_id"])
        stem = vid if args.view == "front" else f"{vid}_{args.view}"
        return (FEAT_CACHE / f"{stem}.npy").is_file()

    # Remap FeatDataset to use view-aware stems via temp column
    pool = pool[pool.apply(cache_ok, axis=1)].reset_index(drop=True)
    if args.view != "front":
        # Symlink-style: copy names expected by FeatDataset (video_id.npy) via wrapper
        # Instead: write a thin dataset that loads view stem
        class ViewFeatDataset(Dataset):
            def __init__(self, rows, y, cache_dir, view):
                self.rows = rows.reset_index(drop=True)
                self.y = y.astype(np.int64)
                self.cache_dir = cache_dir
                self.view = view

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                vid = str(self.rows.iloc[idx]["video_id"])
                stem = vid if self.view == "front" else f"{vid}_{self.view}"
                feat = np.load(self.cache_dir / f"{stem}.npy").astype(np.float32)
                return torch.from_numpy(feat), torch.tensor(self.y[idx])

        DS = lambda rows, y: ViewFeatDataset(rows, y, FEAT_CACHE, args.view)
    else:
        DS = lambda rows, y: FeatDataset(rows, y, FEAT_CACHE)

    y_all = pool["text_gloss"].astype(str).map(gloss_to_id).to_numpy()
    train_mask = (pool["split"] == "train").to_numpy()
    test_mask = (pool["split"] == "test").to_numpy()
    val_mask = (pool["split"] == "val").to_numpy()
    if int(train_mask.sum()) < 32:
        raise RuntimeError(f"Too few train feats: {int(train_mask.sum())}")

    train_ds = DS(pool[train_mask], y_all[train_mask])
    test_ds = DS(pool[test_mask], y_all[test_mask]) if test_mask.any() else None
    val_ds = DS(pool[val_mask], y_all[val_mask]) if val_mask.any() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds and len(test_ds) else None
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds and len(val_ds) else None

    head = IsolatedSignHead(
        num_classes=len(glosses), bottleneck=RGB_DIM, dropout=float(args.dropout)
    ).to(device)
    # Inverse-frequency class weights (mitigate long-tail gloss imbalance).
    train_y = y_all[train_mask]
    counts = np.bincount(train_y, minlength=len(glosses)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (len(glosses) * counts)).astype(np.float32)
    class_w = torch.from_numpy(weights).to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=float(args.weight_decay)
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    print(
        f"[*] Train VSL400 n_train={len(train_ds)} n_val={0 if not val_ds else len(val_ds)} "
        f"n_test={0 if not test_ds else len(test_ds)} classes={len(glosses)} epochs={args.epochs} "
        f"ls={args.label_smoothing} drop={args.dropout} noise={args.feat_noise}"
    )

    def _topk_acc(loader, k: int = 1) -> float | None:
        if loader is None:
            return None
        head.eval()
        tc = tn = 0
        with torch.no_grad():
            for x, label in loader:
                x = x.to(device)
                label = label.to(device)
                logits, _ = head(x)
                if k <= 1:
                    pred = logits.argmax(-1)
                    tc += int((pred == label).sum().item())
                else:
                    topk = logits.topk(min(k, logits.size(-1)), dim=-1).indices
                    tc += int((topk == label.unsqueeze(-1)).any(dim=-1).sum().item())
                tn += int(label.numel())
        return 100.0 * tc / max(tn, 1)

    best_score = -1.0
    best_state = None
    bad_epochs = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        n = 0
        correct = 0
        count = 0
        for x, label in train_loader:
            x = x.to(device)
            label = label.to(device)
            if args.feat_noise > 0:
                x = x + float(args.feat_noise) * torch.randn_like(x)
            logits, _ = head(x)
            loss = F.cross_entropy(
                logits,
                label,
                weight=class_w,
                label_smoothing=float(args.label_smoothing),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
            correct += int((logits.argmax(-1) == label).sum().item())
            count += int(label.numel())
        sched.step()
        train_acc = 100.0 * correct / max(count, 1)
        val_acc = _topk_acc(val_loader, 1)
        val_top5 = _topk_acc(val_loader, 5)
        # Development metrics only — never peek at test during epoch selection.
        print(
            f"    [epoch {epoch}] loss={total/max(n,1):.4f} train_acc={train_acc:.1f}%"
            + (f" val@1={val_acc:.1f}% val@5={val_top5:.1f}%" if val_acc is not None else "")
        )
        # Prefer Val Top-5 (better proxy for useful shortlist / SI robustness).
        score = (
            float(val_top5)
            if val_top5 is not None
            else float(val_acc if val_acc is not None else train_acc)
        )
        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "slowfast_proj": {
                    "weight": head.proj.weight.detach().cpu().clone(),
                    "bias": head.proj.bias.detach().cpu().clone(),
                },
                "num_classes": len(glosses),
                "val_acc": val_acc,
                "val_top5": val_top5,
                "train_acc": train_acc,
                "dataset": "VSL400",
                "selection": "val_top5",
                "hyperparams": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "dropout": args.dropout,
                    "feat_noise": args.feat_noise,
                    "patience": args.patience,
                    "view": args.view,
                    "max_per_class": args.max_per_class,
                },
                "label_sha256": _sha256_file(OUT_LABELS),
                "git_revision": _git_revision(),
                "git_dirty": _git_dirty(),
                "backbone": "slowfast_r50_frozen",
                "feat_dim": 2304,
                "schema_version": 2,
                "test_policy": "validation_only_during_train",
            }
        else:
            bad_epochs += 1
        _write_status(
            {
                "phase": "train",
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_top5": val_top5,
                "best_acc": best_score,
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "note": "test metrics deferred until checkpoint lock",
            }
        )
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[*] Early stop at epoch {epoch} (patience={args.patience})")
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    head.load_state_dict(best_state["head"])
    test_acc = test_top5 = None
    if args.final_test and test_loader is not None:
        # Opt-in only; prefer Gate-3 evaluate_vsl400.py so multi-trial HPO never peeks test.
        print(
            "[WARN] --final-test is discouraged: use scripts/evaluate_vsl400.py once after lock."
        )
        test_acc = _topk_acc(test_loader, 1)
        test_top5 = _topk_acc(test_loader, 5)
        best_state["test_acc"] = test_acc
        best_state["test_top5"] = test_top5
        best_state["test_eval"] = "once_after_lock_opt_in"
        print(f"[*] Opt-in test pass: @1={test_acc:.1f}% @5={test_top5:.1f}%")
    else:
        best_state.pop("test_acc", None)
        best_state.pop("test_top5", None)
        best_state["test_eval"] = "deferred_to_evaluate_vsl400"
        print("[*] Checkpoint locked on validation only — test deferred to Gate-3 evaluator.")

    torch.save(best_state, OUT_CLS)
    torch.save(best_state, OUT_CLS_LEGACY)
    torch.save(best_state["slowfast_proj"], OUT_PROJ)
    wstd = float(best_state["slowfast_proj"]["weight"].float().std())
    ckpt_hash = _sha256_file(OUT_CLS)
    _write_status(
        {
            "phase": "completed",
            "best_acc": best_score,
            "test_acc": test_acc,
            "test_top5": test_top5,
            "val_acc": best_state.get("val_acc"),
            "val_top5": best_state.get("val_top5"),
            "output_cls": str(OUT_CLS),
            "output_proj": str(OUT_PROJ),
            "checkpoint_sha256": ckpt_hash,
            "label_sha256": best_state.get("label_sha256"),
            "git_revision": best_state.get("git_revision"),
            "weight_std": wstd,
            "classes": len(glosses),
            "n_train": len(train_ds),
            "n_test": 0 if not test_ds else len(test_ds),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "selection": "val_top5",
            "test_eval": best_state.get("test_eval"),
        }
    )
    print(f"[OK] classifier → {OUT_CLS} best_val_top5={best_score:.1f}% sha256={ckpt_hash}")
    print(f"[OK] slowfast_proj → {OUT_PROJ} weight_std={wstd:.4f}")
    if best_score < 20.0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
