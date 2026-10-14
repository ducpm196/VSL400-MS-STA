#!/usr/bin/env python3
"""E4-R2 confirm after remediated HPO selection. No test / D4."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.eval.ms_sta_v5_m0_strong_baseline import load_json  # noqa: E402
from src.training.ms_sta_e4_r2_provenance import (  # noqa: E402
    assert_no_drift,
    require_clean_code_lock,
    sha256_file,
    verify_sealed_code_lock,
)
from src.training.ms_sta_e4_r2_status import (  # noqa: E402
    assert_no_live_confirm_with_stale_closeout,
    build_campaign_status,
    load_json as load_status_json,
    write_campaign_status,
)


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _python() -> Path:
    for rel in ("_venvs/train_cuda/Scripts/python.exe", "env/Scripts/python.exe"):
        p = _ROOT / rel
        if p.is_file():
            return p
    return Path(sys.executable)


def main() -> int:
    rem = load_json(_ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_HPO_REMEDIATION_POLICY.json")
    selected = load_json(_ROOT / rem["remediation_selected_rel"])
    if selected.get("status") != "SELECTED" or not selected.get("selected"):
        raise SystemExit("no_remediated_selection")
    sel = selected["selected"]
    train_policy_path = _ROOT / "Docs/results/ms_sta_v6_e4_r2_policy/E4_R2_TRAIN_POLICY.json"
    train_policy = load_json(train_policy_path)

    code_lock = require_clean_code_lock(
        root=_ROOT, expect_branch=str(train_policy.get("branch") or "research/ms-sta-v6-e4-r2")
    )
    sealed = load_json(_ROOT / train_policy["code_lock_rel"])
    verify_sealed_code_lock(
        root=_ROOT,
        sealed=sealed,
        live=code_lock,
        policy_sha=sha256_file(train_policy_path),
    )

    runs_root = _ROOT / rem["runs_root_rel"]
    runs = runs_root / "confirm"
    runs.mkdir(parents=True, exist_ok=True)
    sealed_root = _ROOT / "Docs/results/ms_sta_v6_e4_r2"
    prev_top = load_status_json(runs_root / "STATUS.json") if (runs_root / "STATUS.json").is_file() else None
    # Supersede stale HPO_NO_SELECT before marking confirm RUNNING (observability fix).
    campaign = build_campaign_status(
        status="REMEDIATION_CONFIRM_RUNNING",
        phase="confirm_remediated",
        git_revision=code_lock["git_revision"],
        executable_tree_sha256=code_lock["executable_tree_sha256"],
        prev=prev_top,
        extra={
            "selected_config_id": sel["config_id"],
            "confirm_status_rel": "Docs/results/ms_sta_v6_e4_r2_runs_canonical/confirm/STATUS.json",
            "note": "Confirm SoT is confirm/STATUS.json; top-level is campaign lifecycle only.",
        },
    )
    seeds = [int(s) for s in rem["confirm"]["seeds"]]
    rows = []
    for i, seed in enumerate(seeds, 1):
        assert_no_drift(code_lock, root=_ROOT)
        tag = f"{sel['variant']}_lr{sel['lr']}_dw{sel['distill_weight']}_do{sel['dropout']}_wd{sel['weight_decay']}_s{seed}"
        out_dir = runs / tag
        if any((out_dir / n).is_file() for n in ("SUMMARY.json", "best.pt", "VAL_PREDS.json")):
            raise SystemExit(f"refuse_existing_confirm:{out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        confirm_live = {
            "campaign": "E4-R2",
            "phase": "confirm_remediated",
            "status": "RUNNING",
            "canonical": True,
            "trial_index": i,
            "trial_total": len(seeds),
            "current_tag": tag,
            "selected_config_id": sel["config_id"],
            "git_revision": code_lock["git_revision"],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        assert_no_live_confirm_with_stale_closeout(top_level=campaign, confirm=confirm_live)
        write_campaign_status(
            runs_root=runs_root,
            sealed_rel_root=sealed_root,
            payload=campaign,
            confirm=confirm_live,
        )
        atomic_write(runs / "STATUS.json", confirm_live)
        cmd = [
            str(_python()),
            str(_ROOT / "src/training/ms_sta_v5_train.py"),
            "--policy",
            str(train_policy_path),
            "--variant",
            sel["variant"],
            "--phase",
            "confirm",
            "--seed",
            str(seed),
            "--lr",
            str(sel["lr"]),
            "--dropout",
            str(sel["dropout"]),
            "--weight-decay",
            str(sel["weight_decay"]),
            "--distill-weight",
            str(sel["distill_weight"]),
            "--out-dir",
            str(out_dir),
            "--device",
            "cuda",
        ]
        print(json.dumps({"event": "confirm_start", "i": i, "n": len(seeds), "tag": tag}))
        with (out_dir / "train.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0 or not (out_dir / "SUMMARY.json").is_file():
            raise SystemExit(f"confirm_failed:{out_dir}:rc={proc.returncode}")
        summary = load_json(out_dir / "SUMMARY.json")
        if not summary.get("canonical"):
            raise SystemExit(f"confirm_not_canonical:{out_dir}")
        rows.append(
            {
                "seed": seed,
                "tag": tag,
                "best_val_top1": summary["best_val_top1"],
                "best_val_top5": summary.get("best_val_top5"),
                "best_val_macro_f1": summary.get("best_val_macro_f1"),
                "out_rel": str(out_dir.relative_to(_ROOT)).replace("\\", "/"),
                "git_revision": summary.get("git_revision"),
                "checkpoint_sha256": sha256_file(out_dir / "best.pt"),
            }
        )
        atomic_write(runs / "CONFIRM_ROWS.json", {"schema": "e4_r2_confirm_rows_v1", "canonical": True, "rows": rows})

    atomic_write(
        runs / "STATUS.json",
        {
            "campaign": "E4-R2",
            "phase": "confirm_remediated",
            "status": "CONFIRM_COMPLETE",
            "canonical": True,
            "n_seeds": len(rows),
            "selected_config_id": sel["config_id"],
            "git_revision": code_lock["git_revision"],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    write_campaign_status(
        runs_root=runs_root,
        sealed_rel_root=sealed_root,
        payload=build_campaign_status(
            status="CONFIRM_COMPLETE",
            phase="confirm_remediated",
            git_revision=code_lock["git_revision"],
            executable_tree_sha256=code_lock["executable_tree_sha256"],
            prev=campaign,
            extra={
                "selected_config_id": sel["config_id"],
                "n_seeds": len(rows),
                "confirm_status_rel": "Docs/results/ms_sta_v6_e4_r2_runs_canonical/confirm/STATUS.json",
                "note": "Confirm complete; accuracy gate not yet applied.",
            },
        ),
        confirm={"status": "CONFIRM_COMPLETE"},
    )
    print(json.dumps({"event": "confirm_complete", "n": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
