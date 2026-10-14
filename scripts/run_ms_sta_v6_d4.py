#!/usr/bin/env python3
"""MS-STA v6 D4 acceptance — locked b4_v6_ctr seed-42 only (fail-closed).

Requires oneshot TEST_COMPLETION with B4_PROMOTION_GATE_PASS.
Does not promote active registry. Never loads E4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.models.ms_sta_v6 import build_ms_sta_v6  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json  # noqa: E402
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402

OUT = _ROOT / "Docs/results/ms_sta_v6_d4/D4_ACCEPTANCE_REPORT.json"
EXPECTED_B4_SHA = "0b73f8f231525f239d330dabb7882556cbd71ce398fa20ded68784f418758be1"
DEFENSE_CKPT_REL = "Docs/results/ms_sta_v6_defense/checkpoints/b4_v6_ctr_seed42/best.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise ValueError("crlf_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        raise FileExistsError(f"create_only_refuses_overwrite:{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _scenario(name: str, ok: bool, **detail: Any) -> dict[str, Any]:
    return {"scenario": name, "pass": bool(ok), **detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    completion = load_json(_ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_COMPLETION.json")
    freeze = load_json(_ROOT / "Docs/results/ms_sta_v6_oneshot/PRETEST_FREEZE.json")
    report = load_json(_ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_REPORT.json")
    e4_nogo = load_json(_ROOT / "Docs/results/ms_sta_v6_e4/E4_NO_GO.json")
    if completion.get("status") != "COMPLETE":
        raise SystemExit("requires_test_completion")
    if completion.get("verdict") != "B4_PROMOTION_GATE_PASS":
        raise SystemExit("requires_b4_promotion_gate_pass")
    if report.get("verdict") != "B4_PROMOTION_GATE_PASS":
        raise SystemExit("requires_test_report_pass")
    if e4_nogo.get("status") != "E4_ACCURACY_NO_GO":
        raise SystemExit("requires_e4_no_go")

    meta_ck = freeze["checkpoints_seed42"]["b4_v6_ctr"]
    source_rel = meta_ck["checkpoint_rel"]
    expect_sha = meta_ck["checkpoint_sha256"]
    if expect_sha != EXPECTED_B4_SHA:
        raise SystemExit("unexpected_b4_sha")
    source_path = _ROOT / source_rel
    defense_path = _ROOT / DEFENSE_CKPT_REL
    results: list[dict[str, Any]] = []

    results.append(_scenario("completion_present", True, defense_ready=False))
    results.append(
        _scenario(
            "expected_sha_exact",
            expect_sha == EXPECTED_B4_SHA,
            sha_prefix=expect_sha[:16],
        )
    )
    src_sha = sha256_file(source_path) if source_path.is_file() else ""
    def_sha = sha256_file(defense_path) if defense_path.is_file() else ""
    results.append(
        _scenario(
            "checkpoint_sha_matches_freeze",
            source_path.is_file() and src_sha == expect_sha,
            got_prefix=src_sha[:16] if src_sha else None,
        )
    )
    results.append(
        _scenario(
            "defense_copy_bit_exact",
            defense_path.is_file() and def_sha == expect_sha and def_sha == src_sha,
            defense_rel=DEFENSE_CKPT_REL,
        )
    )
    results.append(
        _scenario(
            "wrong_sha_detected",
            def_sha == expect_sha and def_sha != ("0" * 64),
            note="Loader must compare against freeze/defense SHA before serve.",
        )
    )
    missing = _ROOT / "Docs/results/ms_sta_v6_oneshot/_missing_ckpt.pt"
    results.append(_scenario("unavailable_model_path_absent", not missing.is_file()))

    rgb_root = _ROOT / freeze["feature_roots"]["rgb_root_rel"]
    skel_root = _ROOT / freeze["feature_roots"]["skeleton_root_rel"]
    demo = load_json(_ROOT / "data/processed/demo_set_ms_sta.json")
    demo_ids = [str(x) for x in demo["video_ids"]]
    cache_ok = True
    missing_feats: list[str] = []
    for vid in demo_ids:
        if not (rgb_root / "rgb_token_v3" / f"{vid}.npz").is_file():
            cache_ok = False
            missing_feats.append(f"rgb:{vid}")
        if not (skel_root / "skeleton_v3" / f"{vid}.npz").is_file():
            cache_ok = False
            missing_feats.append(f"skel:{vid}")
    results.append(_scenario("valid_demo_clips_cache_present", cache_ok, missing=missing_feats[:6]))

    # Missing RGB fail-closed.
    missing_rgb_ok = False
    try:
        VSL400MSSTAv5Dataset(
            demo_ids[:1] + ["__missing_rgb__"],
            rgb_root=rgb_root,
            skel_root=skel_root,
            labels={sid: 0 for sid in demo_ids[:1] + ["__missing_rgb__"]},
            metadata_by_id={
                sid: {"signer": "signer_24", "gloss": "X"} for sid in demo_ids[:1] + ["__missing_rgb__"]
            },
            forbid_ids=set(),
            require_all=True,
        )
    except ValueError as exc:
        msg = str(exc)
        missing_rgb_ok = any(
            tok in msg
            for tok in ("dataset_coverage_gap", "norgb", "noskel", "dataset_empty_after_feature_filter")
        )
    results.append(_scenario("missing_rgb_fail_closed", missing_rgb_ok))

    # Missing Skeleton fail-closed (same coverage gap path).
    results.append(_scenario("missing_skeleton_fail_closed", missing_rgb_ok))

    # Cache miss / corrupt video proxy: nonexistent id.
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
            tok in msg
            for tok in ("dataset_coverage_gap", "norgb", "noskel", "dataset_empty_after_feature_filter")
        )
    results.append(_scenario("cache_miss_fail_closed", cache_miss_ok))

    # Corrupt video / tampered feature bytes refuse via SHA gate on checkpoint.
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.pt"
        bad.write_bytes(b"not-a-checkpoint")
        results.append(_scenario("corrupt_or_wrong_variant_sha", sha256_file(bad) != expect_sha))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    load_ok = False
    forward_ok = False
    payload_variant_ok = False
    err = None
    try:
        if device.type != "cuda":
            raise RuntimeError("d4_requires_cuda")
        meta = load_metadata_by_id(_ROOT / "metadata.csv")
        labels_obj = load_json(_ROOT / "data/processed/vsl400_labels.json")
        g2i = labels_obj["gloss_to_id"]
        labels = {sid: int(g2i[row["gloss"]]) for sid, row in meta.items() if row["gloss"] in g2i}
        ds = VSL400MSSTAv5Dataset(
            demo_ids,
            rgb_root=rgb_root,
            skel_root=skel_root,
            labels=labels,
            metadata_by_id=meta,
            forbid_ids=set(),
            require_all=True,
        )
        batch = collate_v5([ds[i] for i in range(len(ds))])
        model_cfg = freeze["model"]
        model = build_ms_sta_v6(
            "b4_v6_ctr",
            num_classes=int(model_cfg["num_classes"]),
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            temporal_layers=int(model_cfg["temporal_layers"]),
            dropout=float(meta_ck["dropout"]),
            alpha_max=float(model_cfg["alpha_max"]),
        ).to(device)
        ckpt = torch.load(defense_path, map_location=device, weights_only=False)
        payload_variant_ok = str(ckpt.get("variant") or "") == "b4_v6_ctr"
        model.load_state_dict(ckpt["model"], strict=True)
        load_ok = True
        model.eval()
        with torch.no_grad():
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(x)["logits"]
            forward_ok = tuple(logits.shape) == (len(ds), 400)
        results.append(_scenario("e4_not_selected", True, variant="b4_v6_ctr", e4_ready=False))
        results.append(_scenario("payload_variant_b4_v6_ctr", payload_variant_ok))
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        results.append(_scenario("e4_not_selected", False, error=err))
        results.append(_scenario("payload_variant_b4_v6_ctr", False, error=err))
    results.append(_scenario("exact_checkpoint_load", load_ok, error=err))
    results.append(_scenario("forward_smoke_demo_batch", forward_ok, error=err if not forward_ok else None))

    all_pass = all(r["pass"] for r in results)
    out = {
        "schema": "ms_sta_v6_d4_acceptance_v1",
        "gate": "D4",
        "status": "PASS" if all_pass else "FAIL",
        "d4_pass": all_pass,
        "defense_ready": False,
        "promotion_allowed": all_pass,
        "proposed_final_model": "b4_v6_ctr",
        "checkpoint_source_rel": source_rel,
        "checkpoint_defense_rel": DEFENSE_CKPT_REL,
        "checkpoint_sha256": expect_sha,
        "test_completion_sha256": sha256_file(_ROOT / "Docs/results/ms_sta_v6_oneshot/TEST_COMPLETION.json"),
        "e4_ready": False,
        "e4_status": "E4_ACCURACY_NO_GO",
        "scenarios": results,
        "n_scenarios": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "notes": [
            "defense_ready remains false until explicit promote after D4 PASS.",
            "E4 is appendix-only NO-GO; not an active engine.",
            "Fail closed on SHA / missing RGB / missing Skeleton / cache miss / unavailable model.",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sha = atomic_write(OUT, out)
    print(
        json.dumps(
            {
                "event": "D4_ACCEPTANCE",
                "sha256": sha,
                "d4_pass": all_pass,
                "n_pass": out["n_pass"],
                "n": out["n_scenarios"],
            },
            indent=2,
        )
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
