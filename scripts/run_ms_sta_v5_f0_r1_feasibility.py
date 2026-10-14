#!/usr/bin/env python3
"""F0-R1 feasibility runner — versioned artifacts; does not touch original F0 oneshot paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from f0_feasibility.campaign_paths import F0_R1_PATHS  # noqa: E402
from f0_feasibility.contracts import CANDIDATE_IDS, configure_feasibility_artifacts, feasibility_plan  # noqa: E402
from f0_feasibility.gpu_runner import implementation_surface, run_all_candidates_offline  # noqa: E402
from f0_feasibility.oneshot import (  # noqa: E402
    claim_execution_attempt,
    publish_oneshot_artifacts,
    refuse_if_seal_consumed,
    verify_oneshot_artifacts,
)
from f0_feasibility.preprocess import assert_layout_tensor_spec, synthetic_batch_for  # noqa: E402
from ms_sta_v5_f0_lib import (  # noqa: E402
    PROTOCOL_PLAN_REL,
    SNAPSHOT_CONTRACT_REL,
    WEIGHTS_DIR_REL,
    executable_tree_sha,
    gpu_run_authorization_gate,
    load_json,
    preflight,
    sha256_file,
)

configure_feasibility_artifacts(campaign="f0_r1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-plan", action="store_true", default=True)
    ap.add_argument("--assert-preprocess", action="store_true", default=True)
    ap.add_argument("--run-gpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    surface = implementation_surface()
    seal_rel = F0_R1_PATHS.execution_seal_rel

    if args.resume:
        try:
            if not (_ROOT / seal_rel).is_file():
                raise ValueError("resume_requires_execution_seal_file")
            seal_sha = sha256_file(_ROOT / seal_rel)
            tree_sha = executable_tree_sha(root=_ROOT)
            pf = preflight(
                root=_ROOT, action="dry-plan", python_exe=sys.executable, campaign="f0_r1"
            )
            if pf.get("executable_tree_sha256") != tree_sha:
                raise ValueError("resume_executable_tree_drift")
            out = verify_oneshot_artifacts(
                root=_ROOT,
                execution_seal_sha256=seal_sha,
                executable_tree_sha256=tree_sha,
            )
            print(
                json.dumps(
                    {"schema": "ms_sta_v5_f0_r1_feasibility_resume_v1", **out}, indent=2
                )
            )
            return 0
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "schema": "ms_sta_v5_f0_r1_feasibility_resume_v1",
                        "ok": False,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
            return 2

    if args.run_gpu:
        pf = preflight(root=_ROOT, action="gpu", python_exe=sys.executable, campaign="f0_r1")
        allowed, gate_reasons = gpu_run_authorization_gate(pf, device=args.device)
        consumed = refuse_if_seal_consumed(root=_ROOT)
        if consumed:
            gate_reasons = list(gate_reasons) + consumed
            allowed = False
        if not allowed:
            print(
                json.dumps(
                    {
                        "schema": "ms_sta_v5_f0_r1_feasibility_refuse_v1",
                        "ok": False,
                        "refused": True,
                        "gpu_authorized": False,
                        "side_effect_allowed": bool(pf.get("side_effect_allowed")),
                        "execution_ok": bool(pf.get("execution_ok")),
                        "device": args.device,
                        "implementation_surface": surface,
                        "gate_reasons": gate_reasons,
                        "reasons": list(pf.get("refuse_reasons") or []) + gate_reasons,
                    },
                    indent=2,
                )
            )
            return 2

        seal_sha = sha256_file(_ROOT / seal_rel)
        tree_sha = str(pf.get("executable_tree_sha256"))
        try:
            claim = claim_execution_attempt(
                root=_ROOT,
                execution_seal_sha256=seal_sha,
                executable_tree_sha256=tree_sha,
            )
        except RuntimeError as exc:
            print(
                json.dumps(
                    {
                        "schema": "ms_sta_v5_f0_r1_feasibility_refuse_v1",
                        "ok": False,
                        "refused": True,
                        "gate_reasons": [str(exc)],
                    },
                    indent=2,
                )
            )
            return 2

        snap = load_json(_ROOT / SNAPSHOT_CONTRACT_REL)
        plan = load_json(_ROOT / PROTOCOL_PLAN_REL)
        report = run_all_candidates_offline(
            weights_root=_ROOT / WEIGHTS_DIR_REL,
            snapshot_contract=snap,
            device=args.device,
            work_dir=_ROOT / F0_R1_PATHS.gpu_tmp_rel,
            plan=plan,
        )
        published = publish_oneshot_artifacts(
            root=_ROOT,
            report={**report, "campaign": "f0_r1", "implementation_surface": surface},
            execution_seal_sha256=seal_sha,
            executable_tree_sha256=tree_sha,
        )
        print(
            json.dumps(
                {
                    **report,
                    "campaign": "f0_r1",
                    "implementation_surface": surface,
                    "execution_attempt": claim,
                    "oneshot": published,
                },
                indent=2,
            )
        )
        return 0 if report.get("ok") else 1

    plan = feasibility_plan(allow_gpu=False)
    asserts = []
    if args.assert_preprocess:
        for cid in CANDIDATE_IDS:
            batch = synthetic_batch_for(cid)
            assert_layout_tensor_spec(cid, batch)
            asserts.append({"candidate_id": cid, "ok": True, "batch_keys": sorted(batch)})
    print(
        json.dumps(
            {
                **plan,
                "campaign": "f0_r1",
                "preprocess_assertions": asserts,
                "implementation_surface": surface,
                "executable": "scripts/run_ms_sta_v5_f0_r1_feasibility.py",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
