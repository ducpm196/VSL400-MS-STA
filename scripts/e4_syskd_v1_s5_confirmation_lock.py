# -*- coding: utf-8 -*-
"""Lock S5 confirmation schedule + checkpoints BEFORE confirmation metrics."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "Docs/results/e4_system_kd_v1"
POL = ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"
MAIN = Path(r"D:/TTNT/DNTA/VSL_Translation_Project")
EXPECTED_POLICY = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
BIND_SHA = "ed86157ddcbcb0dff8d1d454f9d572a409b9b5f50c4169b87fc16b34938a69eb"
CKPTS = {
    42: {
        "rel": "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt",
        "alt_confirm_rel": "Docs/results/ms_sta_v6_runs/confirm/b4_v6_ctr_seed42/best.pt",
        "expected_sha256": "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1",
    },
    123: {
        "rel": "Docs/results/ms_sta_v6_runs/confirm/b4_v6_ctr_seed123/best.pt",
        "expected_sha256": "0b0651935cb8cf2c21e043d44633ac4f5538facc910ccebc1293c02240899ba6",
    },
    456: {
        "rel": "Docs/results/ms_sta_v6_runs/confirm/b4_v6_ctr_seed456/best.pt",
        "expected_sha256": "b7bfa7a70e42be649e5408309c902b267d7040de4d4c7b18c1b3bb865889128b",
    },
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_lf(path: Path, obj: dict) -> str:
    if path.exists():
        raise SystemExit(f"exists:{path}")
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    pol = json.loads(POL.read_text(encoding="utf-8"))
    if sha(POL) != EXPECTED_POLICY:
        raise SystemExit("policy_sha")
    seeds = list(pol["confirmation_seeds"])
    if seeds != [42, 123, 456]:
        raise SystemExit(f"seeds:{seeds}")
    sched_lock = json.loads((ART / "E4_SYSKD_V1_S5_SCHEDULE_LOCK.json").read_text(encoding="utf-8"))
    binding = json.loads((ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json").read_text(encoding="utf-8"))
    if sha(ART / "E4_SYSKD_V1_CANONICAL_INPUT_BINDING.json") != BIND_SHA:
        raise SystemExit("binding")

    ckpt_rows = []
    for seed in seeds:
        meta = CKPTS[seed]
        path = MAIN / meta["rel"]
        if not path.is_file():
            raise SystemExit(f"S5_CONFIRMATION_CHECKPOINT_SOURCE_GAP:{seed}:{path}")
        digest = sha(path)
        if digest != meta["expected_sha256"]:
            raise SystemExit(f"ckpt_sha_mismatch:{seed}:{digest}")
        alt_ok = None
        if "alt_confirm_rel" in meta:
            alt = MAIN / meta["alt_confirm_rel"]
            alt_ok = alt.is_file() and sha(alt) == digest
        ckpt_rows.append(
            {
                "seed": seed,
                "path_absolute": str(path.resolve()),
                "path_rel_from_canonical_data_root": meta["rel"],
                "sha256": digest,
                "registry_source": "B4 V6 CTR confirmation / defense SoT",
                "restore_source": str(path.resolve()),
                "restore_sha_verification": True,
                "architecture": "b4_v6_ctr",
                "config_compatibility": "S5=R1+batch_roi_regions; head=b4_v6_ctr; SlowFast frozen shared",
                "alt_confirm_byte_identical": alt_ok,
            }
        )

    ckpt_lock = {
        "schema": "e4_syskd_v1_s5_confirmation_checkpoint_lock_v1",
        "status": "FROZEN_BEFORE_CONFIRMATION_METRICS",
        "created_at": now,
        "source_commit_sha": head,
        "policy_sha256": EXPECTED_POLICY,
        "binding_sha256": BIND_SHA,
        "canonical_data_root": binding["canonical_data_root"],
        "checkpoints": ckpt_rows,
        "require_all_three_seeds": True,
        "forbid_two_seed_conclusion": True,
        "selected": None,
        "e4_ready": False,
    }
    h_ckpt = write_lf(ART / "E4_SYSKD_V1_S5_CONFIRMATION_CHECKPOINT_LOCK.json", ckpt_lock)

    schedule = {
        "schema": "e4_syskd_v1_s5_confirmation_schedule_lock_v1",
        "status": "FROZEN_BEFORE_CONFIRMATION_METRICS",
        "created_at": now,
        "source_commit_sha": head,
        "checkpoint_lock_sha256": h_ckpt,
        "policy_sha256": EXPECTED_POLICY,
        "binding_sha256": BIND_SHA,
        "confirmation_seeds": seeds,
        "forbid_seed_change_after_metrics": True,
        "validation": {
            "role": "val",
            "n": 2629,
            "split_key": "val_ids",
            "arms_per_seed": ["R1", "S5"],
            "forbid_test": True,
            "comparator_parity": "R1",
            "accuracy_gates_from_policy": pol["accuracy_gate"],
            "parity_gates": {
                "logit_max_abs": 1e-4,
                "logit_mean_abs": 1e-5,
                "top1_agreement": 1.0,
                "top5_label_set_agreement": 1.0,
            },
        },
        "efficiency": {
            "mode": "per_seed_same_session",
            "rationale": "Frozen policy lists confirmation_seeds; absolute latency may differ by head seed — do not reuse seed42 latency for 123/456.",
            "arms": ["R0", "R1", "S5"],
            "sample_ids": sched_lock["sample_ids"],
            "measured_rounds": 5,
            "warmup_rounds_per_sample_per_arm": 1,
            "expected_rows_per_seed": 360,
            "triple_arm_order_by_round": sched_lock["triple_arm_order_by_round"],
            "gates_from_policy": pol["efficiency_gate"],
            "batch": 1,
        },
        "ci_bootstrap": {"n_boot": 10000, "seed": 42},
        "stop_rules": pol.get("stop_rules"),
        "test_opening_rule": pol.get("test_opening_rule"),
        "aggregation_errata_required_pass": True,
        "codegraph_campaign_audit_required_pass": True,
        "selected": None,
        "e4_ready": False,
        "note": "Locked before any confirmation metrics.",
    }
    h_sch = write_lf(ART / "E4_SYSKD_V1_S5_CONFIRMATION_SCHEDULE_LOCK.json", schedule)
    print(json.dumps({"checkpoint_lock_sha": h_ckpt, "schedule_lock_sha": h_sch, "head": head}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
