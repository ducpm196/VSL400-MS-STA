"""DEPRECATED for primary path — use train_vsl400_wordlevel.py.

Legacy Stage-1: SlowFast 2304→256 via VSL-Cropped top-k CE (pre-VSL400 pivot).
Kept only for appendix / ablation reproducibility.
"""

from __future__ import annotations

import argparse
import json
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
    RGB_DIM,
    WINDOW_SIZE,
    SlowFastFeatureExtractor,
    _read_video_frames,
    _window_tensors,
)
from src.models.utils import enforce_reproducibility_seed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

print(
    "[DEPRECATED] Primary train is src/training/train_vsl400_wordlevel.py "
    "(VSL400 400-class). This script is appendix-only.",
    file=sys.stderr,
)

FEAT_CACHE = _ROOT / "data" / "features" / "isolated_sf2304"
OUT_CLS = _ROOT / "data" / "processed" / "isolated_sign_classifier.pth"
OUT_LABELS = _ROOT / "data" / "processed" / "isolated_sign_labels.json"
OUT_PROJ = _ROOT / "data" / "processed" / "slowfast_proj.pth"
STATUS = _ROOT / "logs" / "train_stage1_isolated" / "status.json"


def _write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "updated_at": datetime.now(timezone.utc).isoformat()}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def extract_clip_feat2304(
    extractor: SlowFastFeatureExtractor,
    video_path: Path,
    *,
    max_windows: int = 4,
) -> np.ndarray:
    frames = _read_video_frames(video_path)
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


def build_label_maps(df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, dict]:
    counts = df["text_gloss"].astype(str).value_counts()
    keep = set(counts.head(top_k).index.tolist())
    sub = df[df["text_gloss"].astype(str).isin(keep)].copy().reset_index(drop=True)
    glosses = sorted(keep)
    # canonical vietnamese per gloss (mode)
    viet = {}
    for g in glosses:
        vals = sub.loc[sub["text_gloss"].astype(str) == g, "text_vietnamese"].astype(str)
        viet[g] = vals.mode().iloc[0] if len(vals) else g
    gloss_to_id = {g: i for i, g in enumerate(glosses)}
    meta = {
        "id_to_gloss": glosses,
        "id_to_vietnamese": [viet[g] for g in glosses],
        "gloss_to_id": gloss_to_id,
        "top_k": top_k,
        "n_rows": int(len(sub)),
    }
    return sub, meta


def main() -> None:
    load_project_env()
    enforce_reproducibility_seed(42)
    ap = argparse.ArgumentParser(description="Isolated VSL SlowFast domain adaptation")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-class", type=int, default=80, help="Cap train samples/class for speed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--extract-limit", type=int, default=0, help="Debug: max clips to extract")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("train_stage1_isolated_slowfast cần CUDA.")

    df = pd.read_csv(_ROOT / "metadata.csv")
    vc = df[df["dataset"] == "VSL-Cropped"].copy()
    sub, label_meta = build_label_maps(vc, args.top_k)
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.write_text(json.dumps(label_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    gloss_to_id = label_meta["gloss_to_id"]

    # Stratified cap for train speed on RTX 2060
    parts = []
    for g, gdf in sub.groupby(sub["text_gloss"].astype(str)):
        tr = gdf[gdf["split"] == "train"]
        te = gdf[gdf["split"] == "test"]
        if args.max_per_class > 0 and len(tr) > args.max_per_class:
            tr = tr.sample(args.max_per_class, random_state=42)
        parts.append(tr)
        parts.append(te)
    sub = pd.concat(parts, ignore_index=True)
    y = sub["text_gloss"].astype(str).map(gloss_to_id).to_numpy()

    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    resolver = RawVideoResolver(_ROOT)
    extractor = SlowFastFeatureExtractor(device=device)

    rows_need = sub
    if args.extract_limit and args.extract_limit > 0:
        rows_need = sub.iloc[: args.extract_limit].copy()

    if not args.skip_extract:
        print(f"[*] Extract SlowFast-2304 for {len(rows_need)} isolated clips → {FEAT_CACHE}")
        t0 = time.perf_counter()
        done = 0
        skipped = 0
        for i, row in rows_need.iterrows():
            vid = str(row["video_id"])
            out_p = FEAT_CACHE / f"{vid}.npy"
            if out_p.is_file():
                done += 1
                continue
            path = resolver.resolve(row)
            if path is None or not path.is_file():
                skipped += 1
                continue
            try:
                feat = extract_clip_feat2304(extractor, path)
                np.save(out_p, feat)
                done += 1
            except Exception as exc:
                skipped += 1
                print(f"[WARN] {vid}: {exc}")
            if done % 50 == 0:
                _write_status(
                    {
                        "phase": "extract",
                        "done": done,
                        "skipped": skipped,
                        "elapsed_s": round(time.perf_counter() - t0, 1),
                    }
                )
                print(f"  [{done}/{len(rows_need)}] skipped={skipped} elapsed={time.perf_counter()-t0:.0f}s")
        print(f"[OK] extract done={done} skipped={skipped}")

    # Keep only rows with cache
    mask = sub["video_id"].astype(str).map(lambda v: (FEAT_CACHE / f"{v}.npy").is_file())
    sub = sub[mask].reset_index(drop=True)
    y = sub["text_gloss"].astype(str).map(gloss_to_id).to_numpy()
    train_mask = (sub["split"] == "train").to_numpy()
    test_mask = ~train_mask
    if int(train_mask.sum()) < 32:
        raise RuntimeError(f"Too few train feats: {int(train_mask.sum())}")

    train_ds = FeatDataset(sub[train_mask], y[train_mask], FEAT_CACHE)
    test_ds = FeatDataset(sub[test_mask], y[test_mask], FEAT_CACHE) if test_mask.any() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds and len(test_ds) else None
    )

    head = IsolatedSignHead(num_classes=len(label_meta["id_to_gloss"]), bottleneck=RGB_DIM).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    print(
        f"[*] Train classifier n_train={len(train_ds)} n_test={0 if not test_ds else len(test_ds)} "
        f"classes={len(label_meta['id_to_gloss'])} epochs={args.epochs}"
    )

    best_acc = -1.0
    best_state = None
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
            logits, _ = head(x)
            loss = F.cross_entropy(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
            correct += int((logits.argmax(-1) == label).sum().item())
            count += int(label.numel())
        train_acc = 100.0 * correct / max(count, 1)
        test_acc = None
        if test_loader is not None:
            head.eval()
            tc = 0
            tn = 0
            with torch.no_grad():
                for x, label in test_loader:
                    x = x.to(device)
                    label = label.to(device)
                    logits, _ = head(x)
                    tc += int((logits.argmax(-1) == label).sum().item())
                    tn += int(label.numel())
            test_acc = 100.0 * tc / max(tn, 1)
        print(
            f"    [epoch {epoch}] loss={total/max(n,1):.4f} train_acc={train_acc:.1f}%"
            + (f" test_acc={test_acc:.1f}%" if test_acc is not None else "")
        )
        score = test_acc if test_acc is not None else train_acc
        if score > best_acc:
            best_acc = float(score)
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "slowfast_proj": {"weight": head.proj.weight.detach().cpu().clone(), "bias": head.proj.bias.detach().cpu().clone()},
                "num_classes": len(label_meta["id_to_gloss"]),
                "test_acc": test_acc,
                "train_acc": train_acc,
                "top_k": args.top_k,
            }
        _write_status(
            {
                "phase": "train",
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "best_acc": best_acc,
                "elapsed_s": round(time.perf_counter() - t0, 1),
            }
        )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    torch.save(best_state, OUT_CLS)
    # Export projection for Continuous_News RGB cache
    torch.save(best_state["slowfast_proj"], OUT_PROJ)
    wstd = float(best_state["slowfast_proj"]["weight"].float().std())
    _write_status(
        {
            "phase": "completed",
            "best_acc": best_acc,
            "output_cls": str(OUT_CLS),
            "output_proj": str(OUT_PROJ),
            "weight_std": wstd,
            "classes": len(label_meta["id_to_gloss"]),
            "n_train": len(train_ds),
            "n_test": 0 if not test_ds else len(test_ds),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    )
    print(f"[OK] classifier → {OUT_CLS} best_acc={best_acc:.1f}%")
    print(f"[OK] slowfast_proj → {OUT_PROJ} weight_std={wstd:.4f}")
    if best_acc < 15.0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
