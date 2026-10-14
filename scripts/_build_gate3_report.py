"""Build Docs/results/gate3 canonical report from gate3_v2 eval payload."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "logs" / "eval_vsl400" / "gate3_v2" / "eval_full_si_fusion.json"
data = json.loads(p.read_text(encoding="utf-8"))
s = data["summary"]

by: dict = defaultdict(lambda: {"n": 0, "t1": 0, "t5": 0, "f_n": 0, "f1": 0, "f5": 0})
a = b = 0
for r in data["rows"]:
    sid = str(r.get("signer_id") or "?")
    by[sid]["n"] += 1
    by[sid]["t1"] += int(bool(r.get("top1")))
    top5_hit = r.get("top5_hit")
    if top5_hit is None:
        top5_hit = r.get("gold") in (r.get("top5") or [])
    by[sid]["t5"] += int(bool(top5_hit))
    if "fusion_top1" in r:
        by[sid]["f_n"] += 1
        by[sid]["f1"] += int(bool(r.get("fusion_top1")))
        by[sid]["f5"] += int(bool(r.get("fusion_top5_hit")))
        ft = bool(r["top1"])
        fu = bool(r["fusion_top1"])
        if ft and not fu:
            a += 1
        if fu and not ft:
            b += 1

per = {
    sid: {
        "n": bb["n"],
        "front_top1": round(100 * bb["t1"] / bb["n"], 4),
        "front_top5": round(100 * bb["t5"] / bb["n"], 4),
        "fusion_n": bb["f_n"],
        "fusion_top1": round(100 * bb["f1"] / max(bb["f_n"], 1), 4) if bb["f_n"] else None,
        "fusion_top5": round(100 * bb["f5"] / max(bb["f_n"], 1), 4) if bb["f_n"] else None,
    }
    for sid, bb in sorted(by.items())
}

payload_sha = hashlib.sha256(p.read_bytes()).hexdigest()
report = {
    "gate": 3,
    "schema": "v2_validation_only",
    "product": s.get("product"),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "locked_checkpoint": {
        "path": "data/processed/vsl400_classifier.pth",
        "sha256": s.get("checkpoint_sha256"),
        "schema_version": 2,
        "selection": "val_top5",
        "test_eval": "deferred_to_evaluate_vsl400",
        "git_revision": s.get("git_revision"),
        "train_commit": "deb88f7a9c8b4ed2124951f01206862bbc403e8b",
    },
    "provenance": {
        "metadata_sha256": s.get("metadata_sha256"),
        "labels_sha256": s.get("labels_sha256"),
        "evaluator_sha256": s.get("evaluator_sha256"),
        "cache_mode": s.get("cache_mode"),
        "git_revision": s.get("git_revision"),
        "git_dirty": s.get("git_dirty"),
    },
    "canonical_front": {
        "n": s["n"],
        "expected_n": s["expected_n"],
        "skipped_count": s["skipped_count"],
        "skip_reasons": s["skip_reasons"],
        "top1_acc": s["top1_acc"],
        "top5_acc": s["top5_acc"],
    },
    "canonical_fusion": {
        "fusion_n": s.get("fusion_n"),
        "fusion_top1_acc": s.get("fusion_top1_acc"),
        "fusion_top5_acc": s.get("fusion_top5_acc"),
        "fusion_scope": s.get("fusion_scope"),
        "delta_top1_vs_front": (s.get("fusion_top1_acc") or 0) - s["top1_acc"],
        "delta_top5_vs_front": (s.get("fusion_top5_acc") or 0) - s["top5_acc"],
        "mcnemar_discordant_front_only_correct": a,
        "mcnemar_discordant_fusion_only_correct": b,
    },
    "per_signer": per,
    "skipped": data.get("skipped"),
    "artifacts": {
        "eval_json": "logs/eval_vsl400/gate3_v2/eval_full_si_fusion.json",
        "eval_payload_sha256": payload_sha,
        "eval_payload_bytes": p.stat().st_size,
    },
    "conclusions": [
        "Schema v2 checkpoint: validation-only during train; test deferred to this Gate 3 evaluator.",
        (
            f"Front Top-1 {s['top1_acc']:.4f}% / Top-5 {s['top5_acc']:.4f}% "
            f"(n={s['n']}, skip={s['skipped_count']})."
        ),
        (
            f"Naive late fusion Top-1 {s.get('fusion_top1_acc'):.4f}% / "
            f"Top-5 {s.get('fusion_top5_acc'):.4f}% — no observed SI improvement; "
            f"McNemar discordant {a} vs {b}."
        ),
        "Do not generalize to multi-view learning being ineffective.",
    ],
}

out_dir = ROOT / "Docs" / "results" / "gate3"
out_dir.mkdir(parents=True, exist_ok=True)
# Preserve prior pre-v2 report if present under alternate name
legacy = out_dir / "GATE3_REPORT.json"
if legacy.is_file():
    old = json.loads(legacy.read_text(encoding="utf-8"))
    if old.get("schema") != "v2_validation_only":
        (out_dir / "GATE3_REPORT_pre_schema_v2.json").write_text(
            json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8"
        )

(out_dir / "GATE3_REPORT.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)

md = f"""# Gate 3 Report — Schema v2 (validation-only) Full SI + Fusion

**Product:** VSL400 signer-independent word-level recognition with human-assisted output  
**Schema:** v2 — train never peeks test; this evaluator is the authorized test pass  
**Train commit:** `deb88f7`

## Locked artifact

| Field | Value |
|---|---|
| Checkpoint | `data/processed/vsl400_classifier.pth` |
| SHA256 | `{s.get('checkpoint_sha256')}` |
| Selection | `val_top5` = **75.03%** (val@1 ≈43.2%) |
| test_eval | `deferred_to_evaluate_vsl400` (no `test_acc` in ckpt) |
| cache_mode | `{s.get('cache_mode')}` |

## Canonical metrics (full SI)

| Metric | Front | Naive late fusion |
|---|---:|---:|
| n | **{s['n']}** / {s['expected_n']} | **{s.get('fusion_n')}** |
| Top-1 | **{s['top1_acc']:.4f}%** | **{s.get('fusion_top1_acc'):.4f}%** |
| Top-5 | **{s['top5_acc']:.4f}%** | **{s.get('fusion_top5_acc'):.4f}%** |
| Skip | {s['skipped_count']} (`019442`) | same |

McNemar discordant Top-1: front-only-correct={a}, fusion-only-correct={b}.

## Interpretation

Naive late fusion of a **front-only-trained** head: **no observed SI improvement**.  
Do **not** claim significant degradation without significance, and do **not** claim multi-view learning is ineffective.

## Artifacts

- Payload: `logs/eval_vsl400/gate3_v2/eval_full_si_fusion.json` (SHA256 `{payload_sha}`)
- Checksums: `Docs/results/gate3/CHECKSUMS.json`
- Prior pre-schema-v2 lock retained as `GATE3_REPORT_pre_schema_v2.json` when present
"""
(out_dir / "GATE3_REPORT.md").write_text(md, encoding="utf-8")

checksums = {
    "gate": 3,
    "status": "schema_v2_validation_only_canonical",
    "checkpoint_sha256": s.get("checkpoint_sha256"),
    "eval_payload_sha256": payload_sha,
    "eval_payload_bytes": p.stat().st_size,
    "eval_payload_path": "logs/eval_vsl400/gate3_v2/eval_full_si_fusion.json",
    "front_top1": s["top1_acc"],
    "front_top5": s["top5_acc"],
    "fusion_top1": s.get("fusion_top1_acc"),
    "fusion_top5": s.get("fusion_top5_acc"),
    "n": s["n"],
    "expected_n": s["expected_n"],
    "skipped": s["skipped_count"],
    "val_top5_selection": 75.02890173410404,
    "train_commit": "deb88f7a9c8b4ed2124951f01206862bbc403e8b",
    "mcnemar_discordant": {"front_only_correct": a, "fusion_only_correct": b},
    "fusion_interpretation": (
        "Naive late fusion of front-only-trained head: no observed SI improvement. "
        "Not a claim that multi-view learning is ineffective."
    ),
    "prior_pre_schema_v2": {
        "checkpoint_sha256": "6a32cefe04c2fe8798edefdc516e1a8305abd74ca2be0a73874c56a11ac2bc07",
        "eval_payload_sha256": "16d33dae43cff7cf31d1278d98aaff2a3588658d8b4add7aa1e1b2aa0509c43e",
        "note": "Historical reproducibility lock; not anti-leakage clean.",
    },
}
(out_dir / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

readme = """# Gate 3 — versioned evidence (canonical summary)

| File | Role |
|---|---|
| `CHECKSUMS.json` | SHA256 checkpoint + payload + headline metrics (**schema v2 canonical**) |
| `GATE3_REPORT.md` / `.json` | Canonical Gate 3 report |
| `GATE3_REPORT_pre_schema_v2.*` | Historical pre-hardening lock (if present) |

Large payloads live under `logs/eval_vsl400/gate3_v2/` (gitignored) and are referenced by SHA256.
"""
(out_dir / "README.md").write_text(readme, encoding="utf-8")

print("front", s["top1_acc"], s["top5_acc"])
print("fusion", s.get("fusion_top1_acc"), s.get("fusion_top5_acc"))
print("mcnemar", a, b)
print("payload_sha", payload_sha)
print("OK", out_dir)
