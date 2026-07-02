"""The `merge` command core: apply a class mapping and combine N datasets into one.

Given several YOLO datasets with differing class names and a mapping.yaml, this:
  1. builds a unified target class list,
  2. rewrites every box's class id to the target id (or drops it if mapped to null),
  3. copies images + rewritten labels into one output dataset,
  4. writes a fresh data.yaml.

Images are copied (not moved); filenames are prefixed with the source dataset name
to avoid collisions when two datasets both contain img_001.png.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from .yolo import Box, load_yolo_dataset, YoloDataset


@dataclass
class MergeReport:
    target_classes: list[str]
    images_written: dict[str, int]      # split -> count
    boxes_written: int
    boxes_dropped: int
    per_class_boxes: dict[str, int]


def _load_mapping(mapping_path: str | Path) -> tuple[dict[str, str | None], list[str]]:
    data = yaml.safe_load(Path(mapping_path).read_text())
    if not isinstance(data, dict) or "mapping" not in data:
        raise ValueError("mapping file must contain a 'mapping:' section")
    raw = data["mapping"]
    mapping: dict[str, str | None] = {str(k): (None if v is None else str(v)) for k, v in raw.items()}

    # target class list: explicit if provided, else derived from mapping values
    if data.get("target_classes"):
        targets = [str(t) for t in data["target_classes"]]
    else:
        targets = sorted({v for v in mapping.values() if v is not None})
    return mapping, targets


def merge_datasets(
    dataset_paths: list[str | Path],
    mapping_path: str | Path,
    output_path: str | Path,
) -> MergeReport:
    mapping, target_classes = _load_mapping(mapping_path)
    target_id = {name: i for i, name in enumerate(target_classes)}

    output = Path(output_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output folder {output} exists and is not empty")

    images_written: dict[str, int] = {}
    per_class_boxes: dict[str, int] = {name: 0 for name in target_classes}
    boxes_written = 0
    boxes_dropped = 0

    datasets = [load_yolo_dataset(p) for p in dataset_paths]

    for ds in datasets:
        prefix = ds.root.name
        for split, pairs in ds.splits.items():
            out_img_dir = output / split / "images"
            out_lbl_dir = output / split / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img, label in pairs:
                stem = f"{prefix}__{img.stem}"
                shutil.copy2(img, out_img_dir / f"{stem}{img.suffix}")
                images_written[split] = images_written.get(split, 0) + 1

                out_lines: list[str] = []
                if label.exists():
                    for line in label.read_text().splitlines():
                        box = Box.from_line(line)
                        if box is None:
                            continue
                        if not (0 <= box.class_id < len(ds.names)):
                            continue
                        src_name = ds.names[box.class_id]
                        tgt_name = mapping.get(src_name, src_name)
                        if tgt_name is None:            # explicitly dropped
                            boxes_dropped += 1
                            continue
                        if tgt_name not in target_id:   # mapped to a non-target
                            boxes_dropped += 1
                            continue
                        out_lines.append(box.to_line(target_id[tgt_name]))
                        per_class_boxes[tgt_name] += 1
                        boxes_written += 1
                (out_lbl_dir / f"{stem}.txt").write_text(
                    "\n".join(out_lines) + ("\n" if out_lines else "")
                )

    # write merged data.yaml
    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "train/images",
                "val": "valid/images",
                "nc": len(target_classes),
                "names": target_classes,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )

    return MergeReport(
        target_classes=target_classes,
        images_written=images_written,
        boxes_written=boxes_written,
        boxes_dropped=boxes_dropped,
        per_class_boxes=per_class_boxes,
    )
