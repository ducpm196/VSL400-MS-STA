import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data_prep.metadata_utils import (
    assign_split,
    assign_vsl400_split,
    build_vsl400_signer_splits,
    gloss_from_label_text,
    load_voya_labels,
    normalize_signer_id,
    persist_vsl400_split,
    validate_split_integrity,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

class MetadataValidationError(ValueError):
    """Raised when metadata generation fails validation."""


def scan_vsl400(raw_dir: Path, project_root: Path) -> list[dict]:
    """Primary corpus: Zenodo/Kaggle VSL400 multi-view word-level (front rows + sibling views)."""
    vsl400_root = raw_dir / "VSL-Cropped" / "raw" / "raw" / "VSL400"
    if not vsl400_root.is_dir():
        print("    -> Bỏ qua VSL400: thư mục chưa tồn tại.")
        return []

    # Collect entries first to build SI split over all signers
    raw_entries: list[tuple[dict, Path, str]] = []
    for manifest in sorted(vsl400_root.rglob("front_view.json")):
        bundle = manifest.parent
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(entries, list):
            continue
        part = bundle.parent.name if bundle.parent else ""
        for entry in entries:
            video_id = str(entry.get("video_id", "")).strip()
            if not video_id:
                continue
            front = bundle / "front_view" / f"{video_id}.mp4"
            if not front.is_file():
                continue
            raw_entries.append((entry, bundle, part))

    if not raw_entries:
        print("    -> Bỏ qua VSL400: không có front_view mp4.")
        return []

    signers = [normalize_signer_id(e[0].get("signer_id", "unknown")) for e in raw_entries]
    split_payload = build_vsl400_signer_splits(signers)
    persist_vsl400_split(split_payload, project_root)
    split_map = {
        **{s: "train" for s in split_payload["train"]},
        **{s: "val" for s in split_payload["val"]},
        **{s: "test" for s in split_payload["test"]},
    }

    records: list[dict] = []
    seen: set[str] = set()
    for entry, bundle, part in raw_entries:
        video_id = str(entry.get("video_id", "")).strip()
        if video_id in seen:
            continue
        seen.add(video_id)
        signer_id = normalize_signer_id(entry.get("signer_id", "unknown"))
        gloss_text = str(entry.get("gloss", video_id))
        left = bundle / "left_view" / f"{video_id}.mp4"
        right = bundle / "right_view" / f"{video_id}.mp4"
        records.append(
            {
                "video_id": video_id,
                "dataset": "VSL400",
                "signer_id": signer_id,
                "text_gloss": gloss_from_label_text(gloss_text),
                "text_vietnamese": gloss_text,
                "view": "front",
                "has_left": int(left.is_file()),
                "has_right": int(right.is_file()),
                "part": part,
                "split": assign_vsl400_split(signer_id, split_map),
            }
        )

    n_gloss = len({r["text_gloss"] for r in records})
    print(
        f"    -> VSL400: {len(records)} front clips, {n_gloss} glosses, "
        f"test_signers={split_payload['test']}"
    )
    return records


def scan_vsl_cropped(raw_dir: Path) -> list[dict]:
    """Legacy online_sourced only (appendix). VSL400 is scanned separately as dataset=VSL400."""
    vsl_root = raw_dir / "VSL-Cropped"
    online_root = vsl_root / "raw" / "raw" / "online_sourced"
    if not online_root.exists():
        return []

    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for split_name in ("train", "test"):
        split_dir = online_root / split_name
        if not split_dir.exists():
            continue
        for mp4_path in sorted(split_dir.rglob("*.mp4")):
            video_id = mp4_path.stem
            gloss_text = mp4_path.parent.name if mp4_path.parent != split_dir else video_id
            key = (video_id, split_name)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "video_id": video_id,
                    "dataset": "VSL-Cropped",
                    "signer_id": "online_pool",
                    "text_gloss": gloss_from_label_text(gloss_text),
                    "text_vietnamese": gloss_text,
                    "view": "front",
                    "has_left": 0,
                    "has_right": 0,
                    "part": "",
                    "split": split_name,
                }
            )
    if records:
        print(f"    -> VSL-Cropped online_sourced (appendix): {len(records)}")
    return records


def scan_voya_vsl(raw_dir: Path) -> list[dict]:
    """VOYA_VSL: Merged/class_*.npz + labels.json (Stage-1)."""
    voya_dir = raw_dir / "VOYA_VSL"
    merged_dir = voya_dir / "Merged"
    if not merged_dir.exists():
        print("    -> Bỏ qua VOYA_VSL: chưa có thư mục Merged/.")
        return []

    label_map = load_voya_labels(voya_dir)
    records: list[dict] = []

    for npz_file in sorted(merged_dir.glob("*.npz")):
        class_id = npz_file.stem
        if class_id not in label_map:
            raise KeyError(f"Thiếu nhãn VOYA cho class_id={class_id} trong labels.json")
        label_text = label_map[class_id]
        records.append(
            {
                "video_id": class_id,
                "dataset": "VOYA_VSL",
                "signer_id": "voya_pool",
                "text_gloss": gloss_from_label_text(label_text),
                "text_vietnamese": label_text,
                "split": "train",
            }
        )

    return records


def generate_master_metadata():
    raw_dir = _PROJECT_ROOT / "data" / "raw"

    print("[*] Đang tiến hành quét không gian các thư mục dữ liệu thô...")

    metadata_records = (
        scan_vsl400(raw_dir, _PROJECT_ROOT)
        + scan_voya_vsl(raw_dir)
        + scan_vsl_cropped(raw_dir)
    )

    df = pd.DataFrame(metadata_records)
    output_csv = _PROJECT_ROOT / "metadata.csv"

    if df.empty:
        raise RuntimeError(
            "Không tìm thấy mẫu nào — cần VSL400 dưới data/raw/VSL-Cropped/raw/raw/VSL400."
        )

    # Align columns for mixed datasets
    for col, default in (
        ("view", "front"),
        ("has_left", 0),
        ("has_right", 0),
        ("part", ""),
    ):
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    leak = validate_split_integrity(df)
    if leak:
        raise MetadataValidationError("Split integrity failed:\n  " + "\n  ".join(leak))

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[OK] Đã khởi tạo thành công file quản lý nhãn tập trung tại: {output_csv}")
    print(f"     -> Tổng số lượng mẫu ghi nhận: {len(df)}")
    for name in ("VSL400", "VOYA_VSL", "VSL-Cropped"):
        n = int((df["dataset"] == name).sum())
        if n:
            print(f"     -> {name}: {n}")
    vsl = df[df["dataset"] == "VSL400"]
    if not vsl.empty:
        print(f"     -> VSL400 glosses: {vsl['text_gloss'].nunique()}")
        print(f"     -> VSL400 split: {vsl['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    generate_master_metadata()
