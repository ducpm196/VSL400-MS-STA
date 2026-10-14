"""Shared helpers for metadata.csv generation (Signer-Independent Split)."""

import json
from pathlib import Path

import pandas as pd

TEST_SIGNERS = {"signer_12", "signer_15"}
VAL_SIGNERS = {"signer_08"}
HELD_OUT_SIGNERS = set(TEST_SIGNERS)

# VSL400 SI protocol: last 5 signers=test, preceding 3=val (sorted signer_id).
VSL400_TEST_N = 5
VSL400_VAL_N = 3
VSL400_SPLIT_PATH = Path("data/processed/vsl400_split_signers.json")


def assign_split(signer_id: str) -> str:
    if signer_id in HELD_OUT_SIGNERS:
        return "test"
    if signer_id in VAL_SIGNERS:
        return "val"
    return "train"


def build_vsl400_signer_splits(signer_ids: list[str]) -> dict:
    """Deterministic SI split for VSL400 signers. Persists audit JSON."""

    def _signer_key(sid: str) -> int:
        digits = "".join(ch for ch in sid if ch.isdigit())
        return int(digits) if digits else 0

    ordered = sorted({normalize_signer_id(s) for s in signer_ids}, key=_signer_key)
    if len(ordered) < VSL400_TEST_N + VSL400_VAL_N + 1:
        raise ValueError(
            f"VSL400 cần ≥{VSL400_TEST_N + VSL400_VAL_N + 1} signers, có {len(ordered)}"
        )
    test = ordered[-VSL400_TEST_N:]
    val = ordered[-(VSL400_TEST_N + VSL400_VAL_N) : -VSL400_TEST_N]
    train = ordered[: -(VSL400_TEST_N + VSL400_VAL_N)]
    payload = {
        "train": train,
        "val": val,
        "test": test,
        "n_signers": len(ordered),
        "rule": f"numeric sorted; last {VSL400_TEST_N}=test, prev {VSL400_VAL_N}=val",
    }
    return payload


def assign_vsl400_split(signer_id: str, split_map: dict[str, str]) -> str:
    sid = normalize_signer_id(signer_id)
    split = split_map.get(sid)
    if split is None:
        raise KeyError(f"Signer {sid} không có trong VSL400 split map")
    return split


def persist_vsl400_split(payload: dict, project_root: Path) -> Path:
    out = project_root / VSL400_SPLIT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def validate_split_integrity(metadata_df: pd.DataFrame) -> list[str]:
    """Fail-fast checks for signer-independent split policy."""
    errors: list[str] = []
    required_cols = {"signer_id", "split"}
    if not required_cols.issubset(metadata_df.columns):
        return [f"metadata.csv thiếu cột: {required_cols - set(metadata_df.columns)}"]

    for signer in sorted(HELD_OUT_SIGNERS):
        rows = metadata_df[
            (metadata_df["signer_id"] == signer) & (metadata_df["dataset"] != "VSL400")
        ]
        if rows.empty:
            continue
        leaked = rows[rows["split"] != "test"]
        if not leaked.empty:
            errors.append(
                f"Data leakage: {signer} có {len(leaked)} mẫu không thuộc split=test"
            )

    vsl = metadata_df[metadata_df["dataset"] == "VSL400"]
    if not vsl.empty:
        for split_name, other in (("test", "train"), ("val", "train"), ("test", "val")):
            a = set(vsl.loc[vsl["split"] == split_name, "signer_id"].astype(str))
            b = set(vsl.loc[vsl["split"] == other, "signer_id"].astype(str))
            leak = a & b
            if leak:
                errors.append(f"VSL400 SI leak {split_name}∩{other}: {sorted(leak)}")

    return errors


def load_voya_labels(voya_dir: Path) -> dict[str, str]:
    labels_path = voya_dir / "labels.json"
    merged_dir = voya_dir / "Merged"
    if not labels_path.exists():
        if merged_dir.exists() and any(merged_dir.glob("*.npz")):
            raise FileNotFoundError(f"VOYA_VSL có Merged/ nhưng thiếu {labels_path}")
        return {}
    with open(labels_path, encoding="utf-8") as f:
        return json.load(f)


def gloss_from_label_text(label_text: str) -> str:
    """Derive Stage-1 gloss token from Vietnamese label text."""
    return label_text.strip().upper().replace(" ", "_")


def normalize_signer_id(raw_signer: str) -> str:
    """Map dataset signer codes to metadata signer_id (e.g. '024' -> 'signer_24')."""
    digits = "".join(ch for ch in str(raw_signer) if ch.isdigit())
    if not digits:
        return f"signer_{raw_signer}"
    return f"signer_{int(digits)}"
