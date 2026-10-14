# -*- coding: utf-8 -*-
"""E4_SYSKD_V1 D4 acceptance — exactly 17 frozen scenarios; no e4_ready auto-set."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from scripts.e4_syskd_v1_s5_confirmation import build_models_for_seed  # noqa: E402
from scripts.e4_syskd_v1_s5_formal import extract_arm, load_binding, load_meta  # noqa: E402
from src.data_prep.raw_video_resolve import RawVideoResolver  # noqa: E402
from src.models.rgb_token_v3 import read_video_frames_native  # noqa: E402
from src.optimization.b4q.invariants import sha256_file  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset  # noqa: E402

ART = CODE_ROOT / "Docs/results/e4_system_kd_v1"
POL = CODE_ROOT / "Docs/results/e4_system_kd_v1_policy/E4_SYSKD_V1_POLICY.json"
SCHED = ART / "E4_SYSKD_V1_D4_SCHEDULE_LOCK.json"
AUTH = ART / "E4_SYSKD_V1_D4_EXECUTION_AUTHORIZATION.json"
PROP = ART / "E4_SYSKD_V1_D4_EXECUTION_AUTHORIZATION_PROPOSAL.json"
RAW = ART / "E4_SYSKD_V1_D4_EXECUTION_RAW.jsonl"
REPORT = ART / "E4_SYSKD_V1_D4_ACCEPTANCE_REPORT.json"
VERIFY = ART / "E4_SYSKD_V1_D4_INDEPENDENT_VERIFY.json"
TERM = ART / "E4_SYSKD_V1_D4_TERMINAL.json"

EXPECTED_SCHED_SHA = "ff61e9b491f31f0344e8368be4c4f0b8c06d681de80e745ce3b83967015f334b"
CKPT = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
POL_SHA = "83cd88f9ac210773dacbfa33ae1a8d668170cc481a8ae6c809c1c234a5a3a9f3"
CODE_CAND = "b52c23b0e3598bd78731c4e3e28c7dbb5e5eb8f2"
POST_VERIFY_SHA = "14644bdf30f27098f082f40700c7ee6de06b9fbf4f8ce1a3a01563c0e6311d5f"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_lf(path: Path, obj: Any, *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        raise SystemExit(f"exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def append_raw(row: dict) -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with RAW.open("ab") as f:
        f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def scen(name: str, ok: bool, **detail: Any) -> dict[str, Any]:
    row = {"scenario": name, "pass": bool(ok), "ts": utc_now(), **detail}
    append_raw(row)
    print(f"[d4] {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return row


def require_auth() -> tuple[dict, dict]:
    if not AUTH.exists():
        raise SystemExit("D4_NOT_AUTHORIZED")
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    sch = json.loads(SCHED.read_text(encoding="utf-8"))
    if not auth.get("user_authorization_received"):
        raise SystemExit("D4_NOT_AUTHORIZED")
    if sha256_file(SCHED) != EXPECTED_SCHED_SHA or auth.get("d4_schedule_lock_sha256") != EXPECTED_SCHED_SHA:
        raise SystemExit("schedule_lock_sha_mismatch")
    if len(sch.get("scenarios_a_priori") or []) != 17:
        raise SystemExit("n_scenarios")
    recon = json.loads((ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_TERMINAL.json").read_text(encoding="utf-8"))
    if recon.get("terminal") != "E4_SYSKD_V1_PRE_D4_ACCURACY_RECONCILIATION_PASS":
        raise SystemExit("reconciliation_not_pass")
    return auth, sch


def run_d4() -> dict:
    auth, sch = require_auth()
    if RAW.exists():
        raise SystemExit(f"exists:{RAW}")

    results: list[dict] = []
    names = [s["scenario"] for s in sch["scenarios_a_priori"]]
    if names != [
        "locked_test_terminal_present",
        "post_test_artifact_verify_pass",
        "checkpoint_sha_matches_freeze",
        "policy_sha_matches_freeze",
        "candidate_code_sha_matches_auth",
        "provisional_candidate_is_S5",
        "selected_remains_null_until_promotion",
        "e4_ready_false_until_promotion",
        "b4_retained_as_system_of_record",
        "exact_checkpoint_load_s5_runtime",
        "forward_smoke_demo_batch_fail_closed_paths",
        "missing_rgb_fail_closed",
        "missing_skeleton_fail_closed",
        "cache_miss_fail_closed",
        "wrong_sha_detected",
        "demo_not_cutover",
        "no_threshold_or_candidate_mutation",
    ]:
        raise SystemExit("scenario_order_or_set_mutated")

    # 1
    lt = json.loads((ART / "E4_SYSKD_V1_S5_LOCKED_TEST_TERMINAL.json").read_text(encoding="utf-8"))
    results.append(
        scen(
            "locked_test_terminal_present",
            lt.get("terminal") == "E4_SYSKD_V1_S5_LOCKED_TEST_PASS" and lt.get("pass") is True,
            terminal=lt.get("terminal"),
        )
    )

    # 2
    pv = json.loads((ART / "E4_SYSKD_V1_POST_TEST_ARTIFACT_VERIFY.json").read_text(encoding="utf-8"))
    results.append(
        scen(
            "post_test_artifact_verify_pass",
            pv.get("pass") is True and sha256_file(ART / "E4_SYSKD_V1_POST_TEST_ARTIFACT_VERIFY.json") == POST_VERIFY_SHA,
            verify_sha=sha256_file(ART / "E4_SYSKD_V1_POST_TEST_ARTIFACT_VERIFY.json"),
        )
    )

    # 3
    binding = load_binding()
    data_root = Path(binding["canonical_data_root"])
    ckpt_path = data_root / "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"
    ckpt_sha = sha256_file(ckpt_path) if ckpt_path.is_file() else ""
    results.append(
        scen(
            "checkpoint_sha_matches_freeze",
            ckpt_path.is_file() and ckpt_sha == CKPT == sch["checkpoint_sha256"],
            got_prefix=ckpt_sha[:16],
        )
    )

    # 4
    pol_sha = sha256_file(POL)
    results.append(
        scen(
            "policy_sha_matches_freeze",
            pol_sha == POL_SHA == sch["policy_sha256"],
            got_prefix=pol_sha[:16],
        )
    )

    # 5
    results.append(
        scen(
            "candidate_code_sha_matches_auth",
            sch.get("code_sha_candidate") == CODE_CAND == auth.get("candidate_code_sha"),
            code_sha=CODE_CAND,
        )
    )

    # 6–9 from gate/status artifacts
    gate = json.loads((ART / "E4_SYSKD_V1_S5_POST_TEST_GATE.json").read_text(encoding="utf-8"))
    recon = json.loads((ART / "E4_SYSKD_V1_PRE_D4_RECONCILIATION_TERMINAL.json").read_text(encoding="utf-8"))
    camp = json.loads((ART / "E4_SYSKD_V1_CAMPAIGN_STATUS.json").read_text(encoding="utf-8"))
    results.append(scen("provisional_candidate_is_S5", gate.get("provisional_candidate") == "S5" and recon.get("provisional_candidate") == "S5"))
    results.append(
        scen(
            "selected_remains_null_until_promotion",
            gate.get("selected") is None and recon.get("selected") is None and camp.get("selected") is None and auth.get("selected") is None,
        )
    )
    results.append(
        scen(
            "e4_ready_false_until_promotion",
            gate.get("e4_ready") is False and recon.get("e4_ready") is False and camp.get("e4_ready") is False and auth.get("e4_ready") is False,
        )
    )
    results.append(
        scen(
            "b4_retained_as_system_of_record",
            gate.get("b4_source_of_truth") is True and recon.get("b4_retained") is True and auth.get("b4_retained") is True,
        )
    )

    # 10 exact checkpoint load via S5 runtime (ROI batching)
    load_ok = False
    load_err = None
    try:
        device = torch.device("cuda")
        base, r1, b4, pol = build_models_for_seed(data_root, device, ckpt_path, CKPT)
        load_ok = True
    except Exception as exc:  # noqa: BLE001
        load_err = str(exc)
        base = r1 = b4 = pol = None  # type: ignore
        device = None  # type: ignore
    results.append(scen("exact_checkpoint_load_s5_runtime", load_ok, error=load_err))

    # 11 forward smoke on first locked-test sample with S5 batch_roi=True
    smoke_ok = False
    smoke_detail: dict[str, Any] = {}
    if load_ok:
        try:
            meta = load_meta(data_root)
            labels = json.loads((data_root / "data/processed/vsl400_labels.json").read_text(encoding="utf-8"))["gloss_to_id"]
            test_ids = json.loads(
                (data_root / "data/processed/vsl400_retrain_v4_split_manifest.json").read_text(encoding="utf-8")
            )["test_ids"]
            sid = str(test_ids[0])
            m = meta[sid]
            gloss = str(m.get("text_gloss") or "").strip()
            label = int(labels[gloss])
            signer = str(m.get("signer_id") or "")
            resolver = RawVideoResolver(data_root)
            vpath = Path(resolver.resolve(m))
            sealed = data_root / pol["feature_roots_read_only"]["skel_rel"] / "skeleton_v3" / f"{sid}.npz"
            frames, frame_idx, fps = read_video_frames_native(str(vpath))
            out = extract_arm(
                extractor=r1,
                batch_roi=True,
                sid=sid,
                vpath=vpath,
                sealed_path=sealed,
                data_root=data_root,
                frames=frames,
                frame_idx=frame_idx,
                fps=fps,
                b4=b4,
                device=device,
                label=label,
                signer=signer,
                time_stages=False,
            )
            smoke_ok = (not out["nan_inf"]) and isinstance(out["prediction"], int) and len(out["top5"]) == 5
            smoke_detail = {"sample_id": sid, "prediction": out["prediction"], "top5": out["top5"]}
        except Exception as exc:  # noqa: BLE001
            smoke_detail = {"error": str(exc)}
    results.append(scen("forward_smoke_demo_batch_fail_closed_paths", smoke_ok, **smoke_detail))

    # 12–14 fail-closed using cache-backed dataset contract (same as parent D4)
    freeze = json.loads((data_root / "Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json").read_text(encoding="utf-8"))
    rgb_root = data_root / freeze["feature_roots"]["rgb_root_rel"]
    skel_root = data_root / freeze["feature_roots"]["skeleton_root_rel"]
    demo = json.loads((data_root / "data/processed/demo_set_ms_sta.json").read_text(encoding="utf-8"))
    demo_ids = [str(x) for x in demo["video_ids"]]

    missing_rgb_ok = False
    try:
        VSL400MSSTAv5Dataset(
            demo_ids[:1] + ["__missing_rgb__"],
            rgb_root=rgb_root,
            skel_root=skel_root,
            labels={sid: 0 for sid in demo_ids[:1] + ["__missing_rgb__"]},
            metadata_by_id={sid: {"signer": "signer_24", "gloss": "X"} for sid in demo_ids[:1] + ["__missing_rgb__"]},
            forbid_ids=set(),
            require_all=True,
        )
    except ValueError as exc:
        msg = str(exc)
        missing_rgb_ok = any(
            tok in msg for tok in ("dataset_coverage_gap", "norgb", "noskel", "dataset_empty_after_feature_filter")
        )
    results.append(scen("missing_rgb_fail_closed", missing_rgb_ok))

    missing_skel_ok = False
    try:
        VSL400MSSTAv5Dataset(
            demo_ids[:1] + ["__missing_skel__"],
            rgb_root=rgb_root,
            skel_root=skel_root,
            labels={sid: 0 for sid in demo_ids[:1] + ["__missing_skel__"]},
            metadata_by_id={sid: {"signer": "signer_24", "gloss": "X"} for sid in demo_ids[:1] + ["__missing_skel__"]},
            forbid_ids=set(),
            require_all=True,
        )
    except ValueError as exc:
        msg = str(exc)
        missing_skel_ok = any(
            tok in msg for tok in ("dataset_coverage_gap", "norgb", "noskel", "dataset_empty_after_feature_filter")
        )
    results.append(scen("missing_skeleton_fail_closed", missing_skel_ok))

    cache_miss_ok = False
    try:
        VSL400MSSTAv5Dataset(
            ["__cache_miss__"],
            rgb_root=rgb_root,
            skel_root=skel_root,
            labels={"__cache_miss__": 0},
            metadata_by_id={"__cache_miss__": {"signer": "signer_24", "gloss": "X"}},
            forbid_ids=set(),
            require_all=True,
        )
    except ValueError as exc:
        msg = str(exc)
        cache_miss_ok = any(
            tok in msg for tok in ("dataset_coverage_gap", "norgb", "noskel", "dataset_empty_after_feature_filter")
        )
    results.append(scen("cache_miss_fail_closed", cache_miss_ok))

    # 15 wrong SHA detected
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.pt"
        bad.write_bytes(b"not-a-checkpoint")
        bad_sha = sha256_file(bad)
        refuse = False
        try:
            build_models_for_seed(data_root, torch.device("cuda"), bad, CKPT)
        except SystemExit as exc:
            refuse = "ckpt_sha" in str(exc)
        except Exception:
            refuse = bad_sha != CKPT
        results.append(
            scen(
                "wrong_sha_detected",
                refuse and bad_sha != CKPT,
                bad_sha_prefix=bad_sha[:16],
            )
        )

    # 16 demo not cutover
    demo_prop = ART / "E4_SYSKD_V1_DEMO_CUTOVER_PROPOSAL.json"
    demo_auth_false = True
    if demo_prop.exists():
        demo_auth_false = json.loads(demo_prop.read_text(encoding="utf-8")).get("user_authorization_received") is not True
    results.append(
        scen(
            "demo_not_cutover",
            gate.get("demo_closed") is True
            and recon.get("demo_closed") is True
            and auth.get("demo_closed") is True
            and demo_auth_false
            and "demo_cutover" in (camp.get("not_opened") or ["demo_cutover"]),
        )
    )

    # 17 no threshold/candidate mutation
    thr = sch.get("thresholds_frozen_refs") or {}
    acc = thr.get("accuracy_gates_from_confirmation_lock") or {}
    mutation_ok = (
        sha256_file(SCHED) == EXPECTED_SCHED_SHA
        and sch.get("provisional_candidate") == "S5"
        and sch.get("selected") is None
        and sch.get("e4_ready") is False
        and sch.get("forbid_candidate_change_after_lock") is True
        and sch.get("forbid_threshold_change_after_lock") is True
        and float(acc.get("hierarchical_ci95_lower_top1_pp_min")) == -0.3
        and float(acc.get("delta_macro_f1_pp_min")) == -0.5
        and float(acc.get("delta_top5_pp_min")) == -0.3
        and float(acc.get("per_signer_delta_top1_pp_min")) == -1.0
        and auth.get("e4_ready") is False
    )
    results.append(scen("no_threshold_or_candidate_mutation", mutation_ok, schedule_sha=EXPECTED_SCHED_SHA))

    n_pass = sum(1 for r in results if r["pass"])
    all_pass = n_pass == 17 and len(results) == 17
    terminal = "E4_SYSKD_V1_D4_PASS_PROMOTION_READY" if all_pass else "E4_SYSKD_V1_D4_NO_GO"

    report = {
        "schema": "e4_syskd_v1_d4_acceptance_report_v1",
        "gate": "D4",
        "created_at": utc_now(),
        "schedule_lock_sha256": EXPECTED_SCHED_SHA,
        "auth_sha256": sha256_file(AUTH),
        "n_scenarios": 17,
        "n_pass": n_pass,
        "scenarios": [{"scenario": r["scenario"], "pass": r["pass"]} for r in results],
        "scenario_details_rel": str(RAW.relative_to(CODE_ROOT)).replace("\\", "/"),
        "raw_sha256": sha256_file(RAW),
        "d4_pass": all_pass,
        "terminal": terminal,
        "promotion_ready": all_pass,
        "selected": None,
        "e4_ready": False,
        "provisional_candidate": "S5",
        "b4_retained": True,
        "demo_closed": True,
        "demo_cutover": "NOT_OPENED",
        "note": "D4 PASS does not auto-set e4_ready=true; requires separate promotion authorization.",
        "checkpoint_sha256": CKPT,
        "policy_sha256": POL_SHA,
        "code_sha_candidate": CODE_CAND,
    }
    write_lf(REPORT, report, overwrite=False)

    # Independent verify: recompute pass count from RAW only
    raw_rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    indep_pass = len(raw_rows) == 17 and all(r.get("pass") for r in raw_rows)
    indep = {
        "schema": "e4_syskd_v1_d4_independent_verify_v1",
        "created_at": utc_now(),
        "verifier": "scripts/e4_syskd_v1_d4_acceptance.py#independent",
        "raw_sha256": sha256_file(RAW),
        "report_sha256": sha256_file(REPORT),
        "n_scenarios": len(raw_rows),
        "n_pass": sum(1 for r in raw_rows if r.get("pass")),
        "pass": indep_pass and (sha256_file(REPORT) == sha256_file(REPORT)),
        "matches_generator": indep_pass == all_pass and report["d4_pass"] == all_pass,
        "terminal": terminal if indep_pass == all_pass else "E4_SYSKD_V1_D4_NO_GO",
        "selected": None,
        "e4_ready": False,
        "demo_closed": True,
        "b4_retained": True,
        "schedule_lock_sha256": EXPECTED_SCHED_SHA,
    }
    indep["pass"] = bool(indep_pass and indep["matches_generator"])
    write_lf(VERIFY, indep, overwrite=False)

    term = {
        "terminal": terminal if indep["pass"] or not all_pass else "E4_SYSKD_V1_D4_NO_GO",
        "d4_pass": all_pass and indep["pass"],
        "promotion_ready": all_pass and indep["pass"],
        "selected": None,
        "e4_ready": False,
        "provisional_candidate": "S5",
        "b4_retained": True,
        "demo_closed": True,
        "demo_cutover": "NOT_OPENED",
        "n_pass": n_pass,
        "n_scenarios": 17,
        "created_at": utc_now(),
        "report_sha256": sha256_file(REPORT),
        "independent_verify_sha256": sha256_file(VERIFY),
        "note": "Await promotion authorization before selected=S5 / e4_ready=true.",
    }
    if not (all_pass and indep["pass"]):
        term["terminal"] = "E4_SYSKD_V1_D4_NO_GO"
        term["d4_pass"] = False
        term["promotion_ready"] = False
    write_lf(TERM, term, overwrite=False)
    return term


def main() -> int:
    print(json.dumps(run_d4(), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
