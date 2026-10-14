"""Defense registry for MS-STA demo loading. Default NO-GO until E1/E2/D4 PASS."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "data" / "processed" / "ms_sta_registry.json"

payload = {
    "status": "NO-GO",
    "defense_ready": False,
    "e1_pass": False,
    "e2_pass": False,
    "d4_pass": False,
    "d1_pass": False,
    "active_variant": None,
    "checkpoint": None,
    "canonical_b0": {
        "path": "data/processed/vsl400_classifier.pth",
        "sha256": "26d866c33676f921198c9ffc61710ff1c4671f6debe782c81c828124977cfe11",
        "note": "Gate3 schema-v2 SlowFast head — ONLY demo default until MS-STA defense_ready",
    },
    "pilot_artifacts": {
        "dir": "data/processed/ms_sta_pilot_nogo",
        "note": "Early B0-B5 .pth are technical pilots — not fair ablation, not defense-ready",
    },
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "reasons": [
        "D1 coverage < 95%; extractor bugs fixed — need clean re-extract",
        "RGB temporal cache [L,2304] not frozen",
        "Pilot B1-B5 checkpoints trained on unequal drifting sample sets",
        "B2-B5 collapsed (near-chance / few predicted classes)",
        "Val-800 was signer_21 subset via iloc; not full-val",
        "Test smoke opened before E1 lock (signer_24 leakage risk)",
        "Demo must default to Gate3 B0, not auto-load B4",
    ],
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
