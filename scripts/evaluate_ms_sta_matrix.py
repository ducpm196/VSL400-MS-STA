"""Gate E2 — evaluate locked MS-STA checkpoints on VSL400 test; McNemar vs B0."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp == fp == fn == 0:
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s) * 100.0) if f1s else 0.0

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.load_env import load_project_env
from src.models.feature_versioning import (
    ELIGIBLE_MANIFEST_REL,
    canonical_eligible_manifest_path,
    load_eligible_sample_manifest,
)
from src.models.isolated_sign_classifier import DEFAULT_CKPT, DEFAULT_LABELS, IsolatedSignHead
from src.models.ms_sta_features import (
    FIXED_RGB_L,
    load_rgb_array,
    resolve_pooled_dir,
    resolve_skel_dir,
    resolve_temporal_dir,
    to_rgb_tensor,
    valid_mask_to_key_padding_mask,
)
from src.models.ms_sta_wordlevel import build_ms_sta

REGISTRY = _ROOT / "data" / "processed" / "ms_sta_registry.json"
DEFENSE_CKPT_DIR = _ROOT / "data" / "processed" / "ms_sta_e2_lock"
GATE3_CKPT = _ROOT / DEFAULT_CKPT
PILOT_MARKERS = ("pilot", "nogo")
PRIMARY_SEED = 42
B0_E1_ROLE = "b0_e1"


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_ckpts_dir(reg: dict, cli: Path | None) -> Path:
    if cli is not None:
        p = cli if cli.is_absolute() else _ROOT / cli
        return p
    rel = reg.get("ckpts_dir")
    if rel:
        return _ROOT / str(rel)
    return DEFENSE_CKPT_DIR


def _validate_locked_payload(
    payload: dict,
    *,
    path: Path,
    expected_variant: str,
    expected_seed: int,
    expected_sha: str,
    elig_sha: str,
    b0_role: str | None = None,
) -> list[str]:
    """Return list of hard errors for a locked E2 payload."""
    errs: list[str] = []
    digest = _sha256_file(path)
    if digest != expected_sha:
        errs.append(f"SHA mismatch for {path.name}: {digest} != {expected_sha}")
    if str(payload.get("variant")) != expected_variant:
        errs.append(f"payload.variant={payload.get('variant')!r} != {expected_variant!r}")
    if int(payload.get("seed") or -1) != int(expected_seed):
        errs.append(f"payload.seed={payload.get('seed')!r} != {expected_seed}")
    if payload.get("e1_selected") is not True or payload.get("evaluation_locked") is not True:
        errs.append(f"{path.name} requires e1_selected=true and evaluation_locked=true")
    if int(payload.get("schema_version") or 0) < 2:
        errs.append(f"{path.name} requires schema_version>=2")
    ckpt_elig = payload.get("sample_manifest_sha256")
    if not ckpt_elig:
        errs.append(f"{path.name} missing sample_manifest_sha256")
    elif ckpt_elig != elig_sha:
        errs.append(f"{path.name} sample_manifest_sha256 mismatch vs eligible lock")
    if not payload.get("source_checkpoint_sha256"):
        errs.append(f"{path.name} missing source_checkpoint_sha256")
    if b0_role is not None and payload.get("b0_identity_role") != b0_role:
        errs.append(
            f"{path.name} b0_identity_role={payload.get('b0_identity_role')!r} != {b0_role!r}"
        )
    return errs


def _refuse_gate3_as_b0(path: Path) -> str | None:
    """Return error if path is Gate3 historical demo checkpoint."""
    try:
        if path.resolve() == GATE3_CKPT.resolve():
            return (
                "E2 refuses Gate3 DEFAULT_CKPT as B0 counterpart; "
                "require B0-E1 primary-seed locked checkpoint"
            )
    except Exception:
        pass
    name = path.name.lower()
    if name in {"vsl400_classifier.pth", "isolated_sign_classifier.pth"}:
        return f"E2 refuses Gate3/demo checkpoint basename as B0: {path.name}"
    return None


def _mcnemar(b0_correct: np.ndarray, other_correct: np.ndarray) -> dict:
    b0_only = int(np.sum(b0_correct & ~other_correct))
    other_only = int(np.sum(~b0_correct & other_correct))
    return {"b0_only_correct": b0_only, "other_only_correct": other_only, "n": int(len(b0_correct))}


def _feature_dirs():
    reg = {}
    if REGISTRY.is_file():
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {
        "pooled": resolve_pooled_dir(_ROOT, reg),
        "temporal": resolve_temporal_dir(_ROOT, reg),
        "skel": resolve_skel_dir(_ROOT, reg),
        "registry": reg,
    }


def _load_rgb(vid: str, *, temporal: bool, dirs: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    if temporal:
        path = dirs["temporal"] / f"{vid}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing temporal rgb {vid}")
        return to_rgb_tensor(load_rgb_array(path), fixed_l=FIXED_RGB_L)
    path = dirs["pooled"] / f"{vid}.npy"
    return to_rgb_tensor(load_rgb_array(path), fixed_l=None)


def _load_skel(vid: str, dirs: dict, target_t: int = 64) -> torch.Tensor | None:
    path = dirs["skel"] / f"{vid}.npy"
    if not path.is_file():
        return None
    sk = np.load(path).astype(np.float32)
    if sk.shape[0] != target_t:
        ix = np.linspace(0, sk.shape[0] - 1, target_t).astype(int)
        sk = sk[ix]
    return torch.from_numpy(sk).unsqueeze(0)


@torch.no_grad()
def _predict_ms(
    model,
    variant: str,
    rgb: torch.Tensor,
    skel: torch.Tensor | None,
    device: torch.device,
    *,
    rgb_kpm: torch.Tensor | None = None,
):
    rgb = rgb.to(device)
    skel_t = skel.to(device) if skel is not None else None
    kpm = rgb_kpm.to(device) if rgb_kpm is not None else None
    if variant in ("b0", "b1"):
        return model(rgb=rgb, rgb_key_padding_mask=kpm)
    if variant == "b2":
        return model(skel=skel_t)
    return model(rgb=rgb, skel=skel_t, rgb_key_padding_mask=kpm)


def _validate_lock_only(reg: dict, ckpt_dir: Path) -> int:
    """Validate E2 lock identity without computing test KPI."""
    errs: list[str] = []
    if reg.get("e1_selected") is not True:
        errs.append("registry e1_selected != true")
    if reg.get("evaluation_locked") is not True:
        errs.append("registry evaluation_locked != true")
    if reg.get("e1_pass") is not True:
        errs.append("registry e1_pass != true")
    if reg.get("defense_ready") is True:
        errs.append("registry defense_ready must remain false until E2")
    if reg.get("e2_pass") is True:
        errs.append("registry e2_pass must be false before E2 run")
    if str(reg.get("status") or "") != "NO-GO":
        errs.append("registry status must be NO-GO before E2")
    active = str(reg.get("active_variant") or "")
    if active != "b4":
        errs.append(f"active_variant={active!r} (expected b4 for this lock)")
    if not reg.get("checkpoint") or not reg.get("checkpoint_sha256"):
        errs.append("missing registry.checkpoint / checkpoint_sha256")
    if not reg.get("b0_checkpoint") or not reg.get("b0_checkpoint_sha256"):
        errs.append("missing registry.b0_checkpoint / b0_checkpoint_sha256")

    elig_path = canonical_eligible_manifest_path(_ROOT)
    try:
        elig_pack = load_eligible_sample_manifest(elig_path, root=_ROOT)
    except ValueError as exc:
        print(f"[FAIL] eligible manifest invalid: {exc}")
        return 2
    if reg.get("eligible_sample_manifest_sha256") != elig_pack["sha256"]:
        errs.append("registry.eligible_sample_manifest_sha256 mismatch")

    b0_path = ckpt_dir / Path(str(reg.get("b0_checkpoint") or "")).name
    act_path = ckpt_dir / Path(str(reg.get("checkpoint") or "")).name
    gate3_err = _refuse_gate3_as_b0(b0_path) if b0_path.is_file() else None
    if gate3_err:
        errs.append(gate3_err)
    if b0_path.is_file():
        b0_payload = torch.load(b0_path, map_location="cpu", weights_only=False)
        errs.extend(
            _validate_locked_payload(
                b0_payload,
                path=b0_path,
                expected_variant="b0",
                expected_seed=PRIMARY_SEED,
                expected_sha=str(reg["b0_checkpoint_sha256"]),
                elig_sha=str(elig_pack["sha256"]),
                b0_role=B0_E1_ROLE,
            )
        )
    else:
        errs.append(f"missing B0-E1 lock checkpoint: {b0_path}")
    if act_path.is_file():
        act_payload = torch.load(act_path, map_location="cpu", weights_only=False)
        errs.extend(
            _validate_locked_payload(
                act_payload,
                path=act_path,
                expected_variant=active,
                expected_seed=PRIMARY_SEED,
                expected_sha=str(reg["checkpoint_sha256"]),
                elig_sha=str(elig_pack["sha256"]),
                b0_role=None,
            )
        )
    else:
        errs.append(f"missing active lock checkpoint: {act_path}")

    out = {
        "gate": "VALIDATE_LOCK",
        "ok": not errs,
        "errors": errs,
        "ckpts_dir": str(ckpt_dir),
        "b0_checkpoint": str(reg.get("b0_checkpoint")),
        "checkpoint": str(reg.get("checkpoint")),
        "active_variant": active,
        "test_kpi_computed": False,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if not errs else 2


def _find_b0_e1_ckpt(ckpt_dir: Path, reg: dict) -> Path | None:
    b0_name = reg.get("b0_checkpoint")
    if b0_name:
        p = ckpt_dir / Path(str(b0_name)).name
        if p.is_file():
            return p
    cands = sorted(ckpt_dir.glob("b0_seed*.pth")) if ckpt_dir.is_dir() else []
    if not cands:
        return None
    return next((p for p in cands if f"seed{PRIMARY_SEED}" in p.name), cands[0])


def main() -> int:
    load_project_env()
    ap = argparse.ArgumentParser(description="Option B Gate E2 matrix eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--gate-label",
        default="",
        help="Override gate label. Default: E2 only for full test; else EVAL_<split>.",
    )
    ap.add_argument("--sample", choices=["first", "stratified"], default="first")
    ap.add_argument(
        "--ckpts-dir",
        type=Path,
        default=None,
        help="Checkpoint dir (default from registry or defense lock dir).",
    )
    ap.add_argument(
        "--allow-pilot-eval",
        action="store_true",
        help="Allow reading ms_sta_pilot_nogo (never labels E2)",
    )
    ap.add_argument(
        "--confirm-e2",
        action="store_true",
        help="Required for E2; needs e1_selected + evaluation_locked + active_variant checkpoint/SHA.",
    )
    ap.add_argument(
        "--validate-lock",
        action="store_true",
        help="Validate E2 lock identity only (no test KPI).",
    )
    ap.add_argument(
        "--allow-gate3-b0",
        action="store_true",
        help="EVAL_* only: use Gate3 IsolatedSignHead for B0 (forbidden on E2).",
    )
    ap.add_argument(
        "--variants",
        default="b0,b1,b2,b3,b4,b5",
        help="EVAL_* only. E2 always evaluates B0-E1 + registry.active_variant only.",
    )
    ap.add_argument("--out", type=Path, default=_ROOT / "logs" / "eval_ms_sta" / "matrix.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = json.loads((_ROOT / DEFAULT_LABELS).read_text(encoding="utf-8"))
    gloss_to_id = labels["gloss_to_id"]
    id_to_gloss = labels["id_to_gloss"]
    num_classes = int(labels["num_classes"])

    dirs = _feature_dirs()
    reg = dirs["registry"] if dirs["registry"] else (
        json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.is_file() else {}
    )
    ckpt_dir = _resolve_ckpts_dir(reg, args.ckpts_dir)

    if args.validate_lock:
        return _validate_lock_only(reg, ckpt_dir)

    ckpt_s = str(ckpt_dir).replace("\\", "/").lower()
    is_pilot = any(m in ckpt_s for m in PILOT_MARKERS)
    if is_pilot and not args.allow_pilot_eval:
        print(f"[FAIL] Refusing pilot ckpt dir: {ckpt_dir}. Use --allow-pilot-eval for diagnostic only.")
        return 2

    gate_label = args.gate_label.strip()
    wants_full_test = args.split == "test" and args.limit == 0
    if not gate_label:
        if wants_full_test and args.confirm_e2:
            gate_label = "E2"
        elif wants_full_test:
            print(
                "[FAIL] Full-test without --confirm-e2. "
                "Refusing to auto-label E2. Use --gate-label EVAL_test or unlock after E1."
            )
            return 2
        else:
            gate_label = f"EVAL_{args.split}" + (f"_limit{args.limit}" if args.limit else "")

    elig_pack: dict | None = None
    if gate_label == "E2":
        if args.allow_gate3_b0:
            print("[FAIL] E2 forbids --allow-gate3-b0 (Gate3 DEFAULT_CKPT cannot be B0 counterpart)")
            return 2
        if args.split != "test" or int(args.limit) != 0:
            print("[FAIL] E2 requires --split test --limit 0 (full holdout test only)")
            return 2
        if is_pilot:
            print("[FAIL] E2 cannot use pilot checkpoints")
            return 2
        if not args.confirm_e2:
            print("[FAIL] E2 requires --confirm-e2")
            return 2
        if reg.get("e1_selected") is not True:
            print("[FAIL] E2 requires registry e1_selected=true (canonical; e1_pass alone insufficient)")
            return 2
        if reg.get("evaluation_locked") is not True:
            print(
                "[FAIL] E2 requires registry evaluation_locked=true "
                "(canonical; e2_unlocked alone insufficient)"
            )
            return 2
        active = str(reg.get("active_variant") or "")
        if active not in {"b1", "b2", "b3", "b4", "b5"}:
            print("[FAIL] E2 requires registry.active_variant ∈ {b1,b2,b3,b4,b5}")
            return 2
        if not reg.get("checkpoint") or not reg.get("checkpoint_sha256"):
            print("[FAIL] E2 requires registry.checkpoint and checkpoint_sha256 (mandatory)")
            return 2
        if not reg.get("b0_checkpoint") or not reg.get("b0_checkpoint_sha256"):
            print("[FAIL] E2 requires registry.b0_checkpoint and b0_checkpoint_sha256 (mandatory)")
            return 2
        ckpt_name = Path(str(reg["checkpoint"])).name
        if str(reg["checkpoint"]) != ckpt_name:
            print("[FAIL] E2 registry.checkpoint must be a basename inside --ckpts-dir (no path separators)")
            return 2
        b0_ckpt_name = Path(str(reg["b0_checkpoint"])).name
        if str(reg["b0_checkpoint"]) != b0_ckpt_name:
            print(
                "[FAIL] E2 registry.b0_checkpoint must be a basename inside --ckpts-dir "
                "(no path separators)"
            )
            return 2
        elig_path = canonical_eligible_manifest_path(_ROOT)
        try:
            elig_pack = load_eligible_sample_manifest(elig_path, root=_ROOT)
        except ValueError as exc:
            print(f"[FAIL] E2 eligible manifest invalid: {exc}")
            return 2
        reg_elig_sha = reg.get("eligible_sample_manifest_sha256")
        if not reg_elig_sha:
            print("[FAIL] E2 requires registry.eligible_sample_manifest_sha256")
            return 2
        if reg_elig_sha != elig_pack["sha256"]:
            print(
                f"[FAIL] E2 registry.eligible_sample_manifest_sha256 mismatch: "
                f"{reg_elig_sha} != {elig_pack['sha256']}"
            )
            return 2

    gate3_head = False
    b0_model = None
    b0_head: IsolatedSignHead | None = None
    b0_counterpart: dict = {}

    if gate_label == "E2":
        assert elig_pack is not None
        b0_path = ckpt_dir / Path(str(reg["b0_checkpoint"])).name
        gate3_err = _refuse_gate3_as_b0(b0_path)
        if gate3_err:
            print(f"[FAIL] {gate3_err}")
            return 2
        if not b0_path.is_file():
            print(f"[FAIL] E2 B0-E1 checkpoint missing in defense dir: {b0_path}")
            return 2
        if b0_path.resolve().parent.resolve() != ckpt_dir.resolve():
            print(f"[FAIL] E2 B0 checkpoint must live under {ckpt_dir}")
            return 2
        b0_payload = torch.load(b0_path, map_location=device, weights_only=False)
        b0_errs = _validate_locked_payload(
            b0_payload,
            path=b0_path,
            expected_variant="b0",
            expected_seed=PRIMARY_SEED,
            expected_sha=str(reg["b0_checkpoint_sha256"]),
            elig_sha=str(elig_pack["sha256"]),
            b0_role=B0_E1_ROLE,
        )
        if b0_errs:
            for err in b0_errs:
                print(f"[FAIL] {err}")
            return 2
        b0_model = build_ms_sta("b0", num_classes=num_classes)
        b0_model.load_state_dict(b0_payload["model"], strict=True)
        b0_model.to(device).eval()
        b0_counterpart = {
            "role": B0_E1_ROLE,
            "path": str(b0_path),
            "sha256": str(reg["b0_checkpoint_sha256"]),
            "gate3_head": False,
            "gate3_forbidden_on_e2": True,
            "val": b0_payload.get("val"),
        }
    elif args.allow_gate3_b0:
        gate3_head = True
        gate3_path = GATE3_CKPT if GATE3_CKPT.is_absolute() else _ROOT / GATE3_CKPT
        if not gate3_path.is_file():
            print(f"[FAIL] Gate3 checkpoint missing: {gate3_path}")
            return 2
        b0_payload = torch.load(gate3_path, map_location=device, weights_only=False)
        b0_head = IsolatedSignHead(num_classes).to(device)
        b0_head.load_state_dict(b0_payload["head"])
        b0_head.eval()
        b0_counterpart = {
            "role": "b0_gate3_historical_demo",
            "path": str(gate3_path),
            "gate3_head": True,
            "gate3_forbidden_on_e2": True,
            "val": b0_payload.get("val"),
        }
    else:
        b0_path = _find_b0_e1_ckpt(ckpt_dir, reg)
        if b0_path is None:
            print(
                f"[FAIL] EVAL requires B0-E1 checkpoint in {ckpt_dir} "
                "(or use --allow-gate3-b0 for Gate3 demo head only)"
            )
            return 2
        gate3_err = _refuse_gate3_as_b0(b0_path)
        if gate3_err:
            print(f"[FAIL] {gate3_err}")
            return 2
        b0_payload = torch.load(b0_path, map_location=device, weights_only=False)
        if str(b0_payload.get("variant") or "") not in ("", "b0"):
            print(f"[FAIL] B0-E1 payload.variant={b0_payload.get('variant')!r} != 'b0'")
            return 2
        b0_model = build_ms_sta("b0", num_classes=num_classes)
        strict = b0_payload.get("evaluation_locked") is True
        b0_model.load_state_dict(b0_payload["model"], strict=strict)
        b0_model.to(device).eval()
        b0_counterpart = {
            "role": B0_E1_ROLE,
            "path": str(b0_path),
            "gate3_head": False,
            "gate3_forbidden_on_e2": True,
            "val": b0_payload.get("val"),
        }

    df = pd.read_csv(_ROOT / "metadata.csv")
    test = df[(df["dataset"] == "VSL400") & (df["split"] == args.split)].reset_index(drop=True)
    if gate_label == "E2":
        assert elig_pack is not None
        test = test[test["video_id"].astype(str).isin(elig_pack["eligible_set"])].reset_index(drop=True)
        expected_n = int(elig_pack["test_n"] or 6165)
        if len(test) != expected_n:
            print(
                f"[FAIL] E2 eligible test n={len(test)} != expected {expected_n} "
                f"(from {ELIGIBLE_MANIFEST_REL})"
            )
            return 2
        if "019442" in set(test["video_id"].astype(str)):
            print("[FAIL] E2 test set still contains excluded 019442")
            return 2
        print(f"[E2] eligible test locked n={len(test)} sample_manifest_sha256={elig_pack['sha256'][:12]}…")
    if args.limit > 0:
        if args.sample == "stratified" and "text_gloss" in test.columns:
            parts = []
            per = max(1, args.limit // max(int(test["text_gloss"].nunique()), 1))
            for _, g in test.groupby(test["text_gloss"].astype(str)):
                parts.append(g.sample(n=min(per, len(g)), random_state=PRIMARY_SEED))
            test = pd.concat(parts, ignore_index=True).iloc[: args.limit].reset_index(drop=True)
        else:
            test = test.iloc[: args.limit].reset_index(drop=True)
            if args.split != "test" or args.limit:
                print(
                    f"[WARN] {gate_label}: sample=first uses iloc[:N] — often single-signer; "
                    "not full-split KPI."
                )

    if gate_label == "E2":
        variants = ["b0", str(reg["active_variant"])]
        print(
            f"[E2] single-active contract: variants={variants} "
            f"b0={reg['b0_checkpoint']} active={reg['checkpoint']}"
        )
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    models: dict = {}
    for v in variants:
        if v == "b0":
            continue
        if gate_label == "E2":
            assert elig_pack is not None
            ckpt_name = Path(str(reg["checkpoint"])).name
            path = ckpt_dir / ckpt_name
            if not path.is_file():
                print(f"[FAIL] E2 checkpoint missing in defense dir: {path}")
                return 2
            if path.resolve().parent.resolve() != ckpt_dir.resolve():
                print(f"[FAIL] E2 checkpoint must live under {ckpt_dir}")
                return 2
            payload = torch.load(path, map_location=device, weights_only=False)
            act_errs = _validate_locked_payload(
                payload,
                path=path,
                expected_variant=v,
                expected_seed=PRIMARY_SEED,
                expected_sha=str(reg["checkpoint_sha256"]),
                elig_sha=str(elig_pack["sha256"]),
                b0_role=None,
            )
            if act_errs:
                for err in act_errs:
                    print(f"[FAIL] {err}")
                return 2
            if payload.get("sample_manifest_sha256") != reg.get("eligible_sample_manifest_sha256"):
                print("[FAIL] E2 ckpt sample_manifest_sha256 != registry")
                return 2
        else:
            ckpt_map = reg.get("checkpoints") if isinstance(reg.get("checkpoints"), dict) else {}
            entry = ckpt_map.get(v) if isinstance(ckpt_map.get(v), dict) else None
            if entry and entry.get("path"):
                path = Path(str(entry["path"]))
                if not path.is_absolute():
                    path = _ROOT / path
                if not path.is_file():
                    path = ckpt_dir / Path(str(entry["path"])).name
            else:
                cands = sorted(ckpt_dir.glob(f"{v}_seed*.pth")) if ckpt_dir.is_dir() else []
                if not cands:
                    print(f"[{gate_label}] skip {v}: no ckpt in {ckpt_dir}")
                    continue
                path = next((p for p in cands if f"seed{PRIMARY_SEED}" in p.name), cands[0])
            payload = torch.load(path, map_location=device, weights_only=False)
            if payload.get("variant") is not None and str(payload.get("variant")) != v:
                print(
                    f"[FAIL] {gate_label} payload.variant={payload.get('variant')!r} != requested {v!r}"
                )
                return 2
            if entry and entry.get("sha256"):
                if _sha256_file(path) != str(entry["sha256"]):
                    print(f"[FAIL] {v} SHA mismatch vs registry.checkpoints")
                    return 2

        m = build_ms_sta(v, num_classes=num_classes)  # type: ignore[arg-type]
        strict = gate_label == "E2" or (payload.get("evaluation_locked") is True)
        m.load_state_dict(payload["model"], strict=strict)
        m.to(device).eval()
        models[v] = {"model": m, "path": str(path), "val": payload.get("val"), "variant": v}

    rows = []
    y_true: list[int] = []
    preds: dict[str, list[int]] = {v: [] for v in ["b0", *models]}
    correct: dict[str, list[bool]] = {v: [] for v in ["b0", *models]}
    signer_rows: dict[str, list] = {}

    for _, r in test.iterrows():
        vid = str(r["video_id"])
        gloss = str(r["text_gloss"])
        if gloss not in gloss_to_id:
            if gate_label == "E2":
                print(f"[FAIL] E2 eligible id {vid} gloss={gloss!r} missing from labels")
                return 2
            continue
        has_pooled = (dirs["pooled"] / f"{vid}.npy").is_file()
        has_temporal = (dirs["temporal"] / f"{vid}.npy").is_file()
        if not has_pooled and not has_temporal:
            if gate_label == "E2":
                print(f"[FAIL] E2 eligible id {vid} missing pooled/temporal RGB features")
                return 2
            continue
        if gate_label == "E2" and not has_pooled:
            print(f"[FAIL] E2 eligible id {vid} missing B0 pooled feature (isolated_sf2304)")
            return 2
        gold = gloss_to_id[gloss]
        signer = str(r.get("signer_id", ""))
        try:
            rgb_b0, _ = _load_rgb(vid, temporal=False, dirs=dirs)
        except Exception as exc:
            if gate_label == "E2":
                print(f"[FAIL] E2 eligible id {vid} B0 RGB load error: {exc}")
                return 2
            continue
        skel = _load_skel(vid, dirs)

        if gate3_head:
            assert b0_head is not None
            if rgb_b0.ndim == 2:
                rgb_vec = rgb_b0 if rgb_b0.shape[0] == 1 else rgb_b0.mean(dim=0, keepdim=True)
            else:
                rgb_vec = rgb_b0.unsqueeze(0)
            logits_b0, _ = b0_head(rgb_vec.to(device))
        else:
            assert b0_model is not None
            if rgb_b0.ndim == 1:
                rgb_batch = rgb_b0.unsqueeze(0)
            elif rgb_b0.ndim == 2:
                rgb_batch = rgb_b0 if rgb_b0.shape[0] == 1 else rgb_b0.mean(dim=0, keepdim=True)
            else:
                rgb_batch = rgb_b0
            logits_b0 = _predict_ms(b0_model, "b0", rgb_batch, None, device)

        pred0 = int(logits_b0.argmax(dim=-1).item())
        top5_b0 = [
            int(x) for x in logits_b0.topk(k=min(5, num_classes), dim=-1).indices.squeeze(0).tolist()
        ]
        y_true.append(gold)
        preds["b0"].append(pred0)
        correct["b0"].append(pred0 == gold)

        row = {
            "video_id": vid,
            "signer_id": signer,
            "gold": gloss,
            "gold_id": gold,
            "b0_pred": id_to_gloss[pred0],
            "b0_top1": pred0 == gold,
            "b0_top5_hit": gold in top5_b0,
        }

        for v, pack in models.items():
            needs_skel = v in ("b2", "b3", "b4", "b5")
            needs_temporal_rgb = v in ("b1", "b3", "b4", "b5")
            if needs_skel and skel is None:
                if gate_label == "E2":
                    print(f"[FAIL] E2 eligible id {vid} missing skeleton for variant {v}")
                    return 2
                row[f"{v}_pred"] = None
                row[f"{v}_skipped"] = "missing_skel"
                preds[v].append(-1)
                correct[v].append(False)
                continue
            try:
                rgb, valid = _load_rgb(vid, temporal=needs_temporal_rgb, dirs=dirs)
            except Exception as exc:
                if gate_label == "E2":
                    print(f"[FAIL] E2 eligible id {vid} RGB load error for {v}: {exc}")
                    return 2
                row[f"{v}_pred"] = None
                row[f"{v}_skipped"] = f"rgb:{exc}"[:120]
                preds[v].append(-1)
                correct[v].append(False)
                continue
            rgb_batch = rgb.unsqueeze(0) if rgb.ndim == 2 else rgb
            kpm = None
            if valid is not None:
                kpm = valid_mask_to_key_padding_mask(valid.unsqueeze(0))
            try:
                logits = _predict_ms(pack["model"], v, rgb_batch, skel, device, rgb_kpm=kpm)
            except ValueError as exc:
                if gate_label == "E2":
                    print(f"[FAIL] E2 eligible id {vid} forward error for {v}: {exc}")
                    return 2
                row[f"{v}_pred"] = None
                row[f"{v}_skipped"] = str(exc)[:120]
                preds[v].append(-1)
                correct[v].append(False)
                continue
            pred = int(logits.argmax(dim=-1).item())
            top5 = [
                int(x) for x in logits.topk(k=min(5, num_classes), dim=-1).indices.squeeze(0).tolist()
            ]
            preds[v].append(pred)
            correct[v].append(pred == gold)
            row[f"{v}_pred"] = id_to_gloss[pred]
            row[f"{v}_top1"] = pred == gold
            row[f"{v}_top5_hit"] = gold in top5
            row[f"{v}_skipped"] = None

        rows.append(row)
        signer_rows.setdefault(signer, []).append(row)

    def _metrics(name: str) -> dict:
        y = np.array(y_true)
        p = np.array(preds[name])
        mask = p >= 0
        if int(mask.sum()) == 0:
            return {"n": 0, "top1": None, "top5": None, "macro_f1": None}
        y_m, p_m = y[mask], p[mask]
        top1 = float((y_m == p_m).mean() * 100.0)
        macro = _macro_f1(y_m, p_m, num_classes)
        if name == "b0":
            top5 = 100.0 * float(np.mean([r["b0_top5_hit"] for r in rows]))
        else:
            vals = [r[f"{name}_top5_hit"] for r in rows if r.get(f"{name}_pred") is not None]
            top5 = 100.0 * float(np.mean(vals)) if vals else None
        return {"n": int(mask.sum()), "top1": top1, "top5": top5, "macro_f1": macro}

    summary = {"b0": _metrics("b0")}
    if gate_label == "E2":
        assert elig_pack is not None
        expected_n = int(elig_pack["test_n"] or 6165)
        if summary["b0"]["n"] != expected_n:
            print(
                f"[FAIL] E2 scored n={summary['b0']['n']} != eligible test n={expected_n} "
                "(silent skips are forbidden)"
            )
            return 2
    mcnemar: dict = {}
    b0_c = np.array(correct["b0"], dtype=bool)
    p_arr = {v: np.array(preds[v]) for v in ["b0", *models]}
    for v in models:
        summary[v] = _metrics(v)
        pair_mask = p_arr[v] >= 0
        if int(pair_mask.sum()) == 0:
            mcnemar[v] = {"b0_only_correct": 0, "other_only_correct": 0, "n": 0, "note": "no_common_pairs"}
            continue
        other = np.array(correct[v], dtype=bool)
        mcnemar[v] = _mcnemar(b0_c[pair_mask], other[pair_mask])

    per_signer = {
        s: {"n": len(rs), "b0_top1": 100.0 * float(np.mean([r["b0_top1"] for r in rs]))}
        for s, rs in signer_rows.items()
    }
    worst = min(per_signer.items(), key=lambda kv: kv[1]["b0_top1"]) if per_signer else None

    out = {
        "gate": gate_label,
        "split": args.split,
        "limit": args.limit,
        "sample": args.sample,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "summary": summary,
        "mcnemar_vs_b0": mcnemar,
        "per_signer_b0": per_signer,
        "worst_signer_b0": {"signer": worst[0], **worst[1]} if worst else None,
        "models": {k: {"path": v["path"], "val": v["val"]} for k, v in models.items()},
        "b0_counterpart": b0_counterpart,
        "feature_dirs": {
            "pooled": str(dirs["pooled"]),
            "temporal": str(dirs["temporal"]),
            "skel": str(dirs["skel"]),
        },
        "e2_contract": "single_active_variant_b0_e1" if gate_label == "E2" else "eval_matrix",
        "note": (
            "B0 uses pooled RGB (isolated_sf2304). E2 evaluates B0-E1 + registry.active_variant only; "
            "Gate3 DEFAULT_CKPT is forbidden as E2 B0 counterpart. "
            "McNemar uses common-pair mask."
        ),
        "rows": rows,
    }
    if gate_label != "E2" and gate3_head:
        out["canonical_b0_ckpt"] = str(GATE3_CKPT if GATE3_CKPT.is_absolute() else _ROOT / GATE3_CKPT)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate_label, "n": len(rows), "summary": summary, "mcnemar": mcnemar}, indent=2))
    print(f"[{gate_label}] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

