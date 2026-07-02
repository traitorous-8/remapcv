"""The `scan` command: inspect one or more YOLO datasets and report their classes.

This is the first thing a user runs. Before you can merge datasets with different
class names, you need to SEE all the class names across them -- that's what scan does.
It also emits a starter mapping.yaml you can edit, so you don't write it from scratch.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from .yolo import load_yolo_dataset, YoloDataset


def scan_datasets(paths: list[str | Path]) -> list[YoloDataset]:
    return [load_yolo_dataset(p) for p in paths]


def build_class_index(datasets: list[YoloDataset]) -> dict[str, dict[str, int]]:
    """Map each class name -> {dataset_root_name: box_count}.

    Lets us show which dataset each class came from and how many boxes it has.
    """
    index: dict[str, dict[str, int]] = defaultdict(dict)
    for ds in datasets:
        counts = ds.class_counts
        for name, cnt in counts.items():
            index[name][ds.root.name] = cnt
    return dict(index)


def suggest_mapping_skeleton(datasets: list[YoloDataset]) -> dict:
    """Produce a starter mapping.yaml: every source class mapped to ITSELF.

    The user then hand-edits synonyms to collapse them, e.g. maps
    'hard_hat' and 'каска' both to 'helmet'. Mapping to null drops a class.
    """
    all_names: list[str] = []
    seen = set()
    for ds in datasets:
        for name in ds.names:
            if name not in seen:
                seen.add(name)
                all_names.append(name)
    return {
        "target_classes": sorted({n for n in all_names}),
        "mapping": {name: name for name in all_names},
    }


def write_mapping_skeleton(datasets: list[YoloDataset], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    skeleton = suggest_mapping_skeleton(datasets)
    header = (
        "# remapcv mapping file\n"
        "# Edit the 'mapping' below to merge classes across datasets.\n"
        "#   - point synonyms at one target name:  hard_hat: helmet\n"
        "#   - drop a class entirely:              background: null\n"
        "# 'target_classes' is auto-derived from the mapping when you run `merge`.\n\n"
    )
    out_path.write_text(header + yaml.safe_dump(skeleton, sort_keys=False, allow_unicode=True))
    return out_path
