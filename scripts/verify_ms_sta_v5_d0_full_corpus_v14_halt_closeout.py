#!/usr/bin/env python3
"""Independent v14 halt close-out verifier (tracked). Recomputes skeleton counts/SHAs.

Does not mutate skeleton outputs, manifests, or FAILURE_LEDGER.
Writes/refreshes Docs evidence close-out and runtime HALT.json only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_full_corpus_lib import (  # noqa: E402
    CAMPAIGN_TOKEN,
    CORPUS_N,
    EVIDENCE_ROOT_REL,
    OUTPUT_VERSION,
    load_eligible_identity,
    load_json,
    sha256_file,
)

V14_BINARY_REL = "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5"
CLOSEOUT_REL = f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_V14_HALT_CLOSEOUT.json"
HALT_RUNTIME_REL = f"{V14_BINARY_REL}/runtime/HALT.json"
SNAPSHOT_DIR_REL = f"{EVIDENCE_ROOT_REL}/HALT_SNAPSHOT_20260816_213642"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_lf_json(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError(f"crlf_forbidden:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    with tmp.open("wb") as fh:
        fh.write(raw)
        fh.flush()
        import os

        os.fsync(fh.fileno())
    tmp.replace(path)
    return _sha_bytes(raw)


def verify_skeleton(*, root: Path, binary: Path, verify_all_shas: bool) -> dict[str, Any]:
    runtime = binary / "runtime"
    skel_dir = binary / "skeleton_v3"
    summary_path = runtime / "SKEL_PHASE_SUMMARY.json"
    index_path = runtime / "manifests" / "skeleton_v3" / "INDEX.json"
    ledger_path = runtime / "FAILURE_LEDGER.jsonl"
    contract_path = runtime / "_run_contract.json"

    errors: list[str] = []
    summary = load_json(summary_path)
    index = load_json(index_path)
    identity = load_eligible_identity(root=root)
    eligible_ids = list(identity["eligible_ids"])

    npz_count = sum(1 for _ in skel_dir.glob("*.npz"))
    recovery_count = sum(1 for _ in skel_dir.glob("*.recovery.json"))
    records = index.get("records") or {}
    if not isinstance(records, dict):
        errors.append("index_records_not_dict")
        records = {}

    if summary.get("n_ok") is None or int(summary["n_ok"]) != CORPUS_N:
        errors.append(f"summary_n_ok:{summary.get('n_ok')}")
    if summary.get("n_fail") is None or int(summary["n_fail"]) != 0:
        errors.append(f"summary_n_fail:{summary.get('n_fail')}")
    if summary.get("ok") is not True:
        errors.append("summary_ok_false")
    if index.get("n_records") is None or int(index["n_records"]) != CORPUS_N:
        errors.append(f"index_n_records:{index.get('n_records')}")
    if len(records) != CORPUS_N:
        errors.append(f"index_records_len:{len(records)}")
    if npz_count != CORPUS_N:
        errors.append(f"npz_count:{npz_count}")
    if recovery_count != CORPUS_N:
        errors.append(f"recovery_count:{recovery_count}")
    if len(eligible_ids) != CORPUS_N:
        errors.append(f"eligible_len:{len(eligible_ids)}")

    missing_npz = [sid for sid in eligible_ids if not (skel_dir / f"{sid}.npz").is_file()]
    missing_rec = [sid for sid in eligible_ids if not (skel_dir / f"{sid}.recovery.json").is_file()]
    if missing_npz:
        errors.append(f"missing_npz:{len(missing_npz)}")
    if missing_rec:
        errors.append(f"missing_recovery:{len(missing_rec)}")

    sha_mismatches = 0
    checked = 0
    sample_iter = eligible_ids if verify_all_shas else eligible_ids[:64]
    for sid in sample_iter:
        row = records.get(sid)
        if not isinstance(row, dict):
            errors.append(f"index_row_missing:{sid}")
            continue
        out_p = root / str(row.get("output_rel") or "")
        side_p = root / str(row.get("sidecar_rel") or "")
        if out_p.is_file() and row.get("output_sha256"):
            checked += 1
            if sha256_file(out_p) != row["output_sha256"]:
                sha_mismatches += 1
                errors.append(f"output_sha_mismatch:{sid}")
        if side_p.is_file() and row.get("sidecar_sha256"):
            if sha256_file(side_p) != row["sidecar_sha256"]:
                sha_mismatches += 1
                errors.append(f"sidecar_sha_mismatch:{sid}")
        if sha_mismatches >= 5:
            errors.append("sha_mismatch_cap_reached")
            break

    return {
        "ok": not errors,
        "errors": errors[:50],
        "npz_count": npz_count,
        "recovery_count": recovery_count,
        "index_n_records": int(index["n_records"]) if index.get("n_records") is not None else 0,
        "summary_n_ok": int(summary["n_ok"]) if summary.get("n_ok") is not None else 0,
        "summary_n_fail": int(summary["n_fail"]) if summary.get("n_fail") is not None else -1,
        "summary_ok": bool(summary.get("ok")),
        "skel_phase_summary_sha256": sha256_file(summary_path),
        "skel_index_sha256": sha256_file(index_path),
        "failure_ledger_sha256": sha256_file(ledger_path) if ledger_path.is_file() else None,
        "run_contract_sha256": sha256_file(contract_path) if contract_path.is_file() else None,
        "sha_checked": checked,
        "sha_mismatches": sha_mismatches,
        "verify_all_shas": verify_all_shas,
    }


def build_snapshot_manifest(*, root: Path, snap: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for p in sorted(snap.iterdir()):
        if p.is_file():
            files[p.name] = sha256_file(p)
    raw = (json.dumps({"files": files}, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return {
        "schema": "ms_sta_v5_d0_full_corpus_halt_snapshot_manifest_v1",
        "snapshot_rel": str(snap.relative_to(root)).replace("\\", "/"),
        "n_files": len(files),
        "files": files,
        "manifest_sha256": _sha_bytes(raw),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_ROOT)
    ap.add_argument("--verify-all-shas", action="store_true", help="Verify all 24752 output/sidecar SHAs")
    ap.add_argument("--write", action="store_true", help="Write close-out + HALT.json")
    args = ap.parse_args()
    root: Path = args.root.resolve()
    binary = root / V14_BINARY_REL
    snap = root / SNAPSHOT_DIR_REL

    skel = verify_skeleton(root=root, binary=binary, verify_all_shas=bool(args.verify_all_shas))
    snap_manifest = build_snapshot_manifest(root=root, snap=snap) if snap.is_dir() else None

    rgb_fail_ids = 0
    ledger = binary / "runtime" / "FAILURE_LEDGER.jsonl"
    if ledger.is_file():
        ids: set[str] = set()
        with ledger.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("phase") == "rgb_token_v3":
                    ids.add(str(obj.get("sample_id")))
        rgb_fail_ids = len(ids)

    close = {
        "schema": "ms_sta_v5_d0_full_corpus_halt_v1",
        "campaign_token": CAMPAIGN_TOKEN,
        "output_version": OUTPUT_VERSION,
        "decision": "HALT",
        "close_out": "SKELETON_PHASE_PASS_RGB_DEPENDENCY_FAIL",
        "halted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": "scripts/verify_ms_sta_v5_d0_full_corpus_v14_halt_closeout.py",
        "reason": (
            "RGB ModuleNotFoundError pandas under _venvs/f0_v5; "
            "do not install pandas into sealed f0 env; do not resume v14 RGB"
        ),
        "forbid": [
            "rgb_token_v3",
            "quality_scan",
            "completion",
            "M0",
            "train",
            "pip_install_pandas_f0_v5",
            "watchdog_relaunch_v14",
        ],
        "skeleton_verify": skel,
        "skeleton": {
            "committed_npz": skel["npz_count"],
            "recovery_json": skel["recovery_count"],
            "index_n_records": skel["index_n_records"],
            "skel_phase_summary_ok": skel["summary_ok"],
            "n_fail": skel["summary_n_fail"],
            "skel_phase_summary_sha256": skel["skel_phase_summary_sha256"],
            "skel_index_sha256": skel["skel_index_sha256"],
        },
        "rgb": {
            "committed": 0,
            "n_fail_ledger_unique_ids": rgb_fail_ids,
            "error": "No module named 'pandas'",
        },
        "binds": {
            "failure_ledger_sha256": skel["failure_ledger_sha256"],
            "run_contract_sha256": skel["run_contract_sha256"],
            "snapshot_manifest_sha256": (snap_manifest or {}).get("manifest_sha256"),
        },
        "snapshot_rel": SNAPSHOT_DIR_REL,
        "snapshot_manifest": snap_manifest,
        "process_stop": {
            "note": "operator_verified_zero_campaign_procs_at_halt",
            "halt_sentinel_rel": HALT_RUNTIME_REL,
        },
        "v15_next_rel": f"{EVIDENCE_ROOT_REL.replace('/v14', '/v15')}/D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL.json",
        "ok": bool(skel["ok"]) and rgb_fail_ids == CORPUS_N,
    }

    print(json.dumps({"ok": close["ok"], "skeleton_ok": skel["ok"], "rgb_fail_ids": rgb_fail_ids}, indent=2))
    if not skel["ok"]:
        print(json.dumps({"errors": skel["errors"]}, indent=2))
        return 2

    if args.write:
        close_sha = _write_lf_json(root / CLOSEOUT_REL, close)
        halt_obj = dict(close)
        halt_obj["closeout_sha256"] = close_sha
        _write_lf_json(root / HALT_RUNTIME_REL, halt_obj)
        if snap.is_dir() and snap_manifest is not None:
            _write_lf_json(snap / "SNAPSHOT_MANIFEST.json", snap_manifest)
            # refresh closeout copy in snapshot for provenance
            _write_lf_json(snap / "D0_FULL_CORPUS_V14_HALT_CLOSEOUT.json", close)
        print(json.dumps({"written": True, "closeout_sha256": close_sha}, indent=2))
    return 0 if close["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
