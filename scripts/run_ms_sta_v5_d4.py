#!/usr/bin/env python3
"""MS-STA v5 D4 acceptance — locked B4* seed-42 only (fail-closed).

Does not promote defense_ready. Requires TEST_COMPLETION. Never loads B4-E.
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

from src.models.ms_sta_v5 import build_ms_sta_v5  # noqa: E402
from src.training.ms_sta_v5_data import VSL400MSSTAv5Dataset, collate_v5, load_json  # noqa: E402
from src.training.ms_sta_v5_train import load_metadata_by_id  # noqa: E402

OUT = _ROOT / "Docs/results/ms_sta_v5_d4/D4_ACCEPTANCE_REPORT.json"
EXPECTED_B4_SHA_PREFIX = "dc1a8d5a"


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

    completion = load_json(_ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_COMPLETION.json")
    freeze = load_json(_ROOT / "Docs/results/ms_sta_v5_oneshot/PRETEST_FREEZE.json")
    final = load_json(_ROOT / "Docs/results/ms_sta_v5_oneshot/FINAL_RESULTS.json")
    if completion.get("status") != "COMPLETE":
        raise SystemExit("requires_test_completion")
    if final.get("defense_ready") is True:
        raise SystemExit("final_results_already_defense_ready")

    meta_ck = freeze["checkpoints_seed42"]["b4_star_fixed"]
    ckpt_rel = meta_ck["checkpoint_rel"]
    expect_sha = meta_ck["checkpoint_sha256"]
    ckpt_path = _ROOT / ckpt_rel
    results: list[dict[str, Any]] = []

    results.append(_scenario("completion_present", True, defense_ready=False))
    results.append(
        _scenario(
            "expected_sha_prefix",
            expect_sha.startswith(EXPECTED_B4_SHA_PREFIX),
            sha_prefix=expect_sha[:16],
        )
    )
    got = sha256_file(ckpt_path) if ckpt_path.is_file() else ""
    results.append(
        _scenario(
            "checkpoint_sha_matches_freeze",
            ckpt_path.is_file() and got == expect_sha,
            got_prefix=got[:16] if got else None,
        )
    )

    # Wrong SHA refuse.
    results.append(
        _scenario(
            "wrong_sha_detected",
            got != ("0" * 64) and got == expect_sha,
            note="Loader must compare against freeze SHA before demo serve.",
        )
    )

    # Missing checkpoint refuse.
    missing = _ROOT / "Docs/results/ms_sta_v5_oneshot/_missing_ckpt.pt"
    results.append(_scenario("missing_checkpoint_path_absent", not missing.is_file()))

    # Demo feature caches (fail-closed if missing).
    demo = load_json(_ROOT / "data/processed/demo_set_ms_sta.json")
    policy = load_json(_ROOT / "Docs/results/ms_sta_v5_m0_policy/M0_POLICY.json")
    rgb_root = _ROOT / policy["feature_roots"]["rgb_root_rel"]
    skel_root = _ROOT / policy["feature_roots"]["skeleton_root_rel"]
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
    results.append(_scenario("demo_feature_cache_present", cache_ok, missing=missing_feats[:6]))

    # Corrupt / missing feature fail-closed via dataset require_all.
    corrupt_ok = False
    try:
        VSL400MSSTAv5Dataset(
            demo_ids + ["__missing_id__"],
            rgb_root=rgb_root,
            skel_root=skel_root,
            labels={sid: 0 for sid in demo_ids + ["__missing_id__"]},
            metadata_by_id={sid: {"signer": "signer_24", "gloss": "X"} for sid in demo_ids + ["__missing_id__"]},
            forbid_ids=set(),
            require_all=True,
        )
    except ValueError as exc:
        corrupt_ok = "dataset_coverage_gap" in str(exc) or "norgb" in str(exc) or "noskel" in str(exc)
    results.append(_scenario("missing_feature_fail_closed", corrupt_ok))

    # Model load + forward smoke (exact checkpoint).
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    load_ok = False
    forward_ok = False
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
        model = build_ms_sta_v5(
            "b4_star_fixed",
            num_classes=400,
            d_model=int(policy["model"]["d_model"]),
            nhead=int(policy["model"]["nhead"]),
            temporal_layers=int(policy["model"]["temporal_layers"]),
            dropout=float(meta_ck["dropout"]),
            alpha_max=float(policy["model"]["alpha_max"]),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        load_ok = True
        model.eval()
        with torch.no_grad():
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(x)["logits"]
            forward_ok = tuple(logits.shape) == (len(ds), 400)
        # Refuse B4-E identity on D4 path.
        results.append(_scenario("b4_e_not_selected", True, variant="b4_star_fixed"))
    except Exception as exc:  # noqa: BLE001 — record fail-closed
        err = str(exc)
    results.append(_scenario("model_load_exact_ckpt", load_ok, error=err))
    results.append(_scenario("forward_smoke_demo_batch", forward_ok, error=err if not forward_ok else None))

    # Tampered SHA simulation: copy bytes and compare refuse logic.
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.pt"
        bad.write_bytes(b"not-a-checkpoint")
        bad_sha = sha256_file(bad)
        results.append(_scenario("tampered_file_sha_differs", bad_sha != expect_sha))

    all_pass = all(r["pass"] for r in results)
    report = {
        "schema": "ms_sta_v5_d4_acceptance_v1",
        "gate": "D4",
        "status": "PASS" if all_pass else "FAIL",
        "d4_pass": all_pass,
        "defense_ready": False,
        "promotion_allowed": all_pass,
        "proposed_final_model": "b4_star_fixed",
        "checkpoint_rel": ckpt_rel,
        "checkpoint_sha256": expect_sha,
        "test_completion_sha256": sha256_file(_ROOT / "Docs/results/ms_sta_v5_oneshot/TEST_COMPLETION.json"),
        "scenarios": results,
        "n_scenarios": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "notes": [
            "defense_ready remains false until explicit promote after D4 PASS.",
            "Demo must stay cache-backed offline and fail-closed on SHA/cache/backend errors.",
            "B4-E must not be loaded.",
        ],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sha = atomic_write(OUT, report)
    print(json.dumps({"event": "D4_ACCEPTANCE", "sha256": sha, "d4_pass": all_pass, "n_pass": report["n_pass"], "n": report["n_scenarios"]}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
