"""YOLO dataset reading and writing for remapcv.

A YOLO dataset on disk looks like:

    dataset/
      data.yaml            # class names + split paths
      train/images/*.jpg
      train/labels/*.txt   # one line per box: "<class_id> cx cy w h" (normalized)
      valid/images/*.jpg
      valid/labels/*.txt

This module parses that into a simple in-memory representation so the rest of
remapcv never has to care about the on-disk layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Common folder names people use for the val split, in priority order.
_VAL_DIR_NAMES = ("valid", "val", "validation")
_SPLIT_DIR_NAMES = ("train", *_VAL_DIR_NAMES, "test")


@dataclass
class Box:
    """One bounding box: class id + normalized YOLO coords (cx, cy, w, h)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    @classmethod
    def from_line(cls, line: str) -> "Box | None":
        parts = line.split()
        if len(parts) != 5:
            return None
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            return None
        return cls(cid, cx, cy, w, h)

    def to_line(self, class_id: int | None = None) -> str:
        cid = self.class_id if class_id is None else class_id
        return f"{cid} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class YoloDataset:
    """A parsed YOLO dataset."""

    root: Path
    names: list[str]                       # class_id -> class name
    # split -> list of (image_path, label_path). label_path may not exist (background image).
    splits: dict[str, list[tuple[Path, Path]]] = field(default_factory=dict)

    @property
    def class_counts(self) -> dict[str, int]:
        """How many boxes exist per class name across all splits."""
        counts: dict[str, int] = {name: 0 for name in self.names}
        for pairs in self.splits.values():
            for _img, label in pairs:
                if not label.exists():
                    continue
                for line in label.read_text().splitlines():
                    box = Box.from_line(line)
                    if box is None:
                        continue
                    if 0 <= box.class_id < len(self.names):
                        counts[self.names[box.class_id]] += 1
        return counts

    @property
    def image_count(self) -> int:
        return sum(len(p) for p in self.splits.values())


def _resolve_names(data: dict) -> list[str]:
    """data.yaml 'names' can be a list or an id->name dict. Normalize to a list."""
    names = data.get("names")
    if names is None:
        raise ValueError("data.yaml has no 'names' field")
    if isinstance(names, dict):
        # keys may be ints or strings; sort by int key
        ordered = sorted(names.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _k, v in ordered]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError(f"unexpected 'names' type in data.yaml: {type(names)}")


def _find_label_for_image(img: Path) -> Path:
    """Given train/images/foo.jpg -> train/labels/foo.txt."""
    # replace the '/images/' segment with '/labels/' and extension with .txt
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    label = Path(*parts).with_suffix(".txt")
    return label


def _collect_split(root: Path, split_dir: str) -> list[tuple[Path, Path]]:
    images_dir = root / split_dir / "images"
    if not images_dir.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        pairs.append((img, _find_label_for_image(img)))
    return pairs


def load_yolo_dataset(root: str | Path) -> YoloDataset:
    """Read a YOLO dataset from disk."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {root}")

    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        # some exports call it dataset.yaml or data.yml
        for alt in ("dataset.yaml", "data.yml", "dataset.yml"):
            if (root / alt).exists():
                yaml_path = root / alt
                break
        else:
            raise FileNotFoundError(f"no data.yaml found in {root}")

    data = yaml.safe_load(yaml_path.read_text())
    names = _resolve_names(data)

    splits: dict[str, list[tuple[Path, Path]]] = {}
    for split_dir in _SPLIT_DIR_NAMES:
        pairs = _collect_split(root, split_dir)
        if pairs:
            # normalize val folder name to "valid" internally
            key = "valid" if split_dir in _VAL_DIR_NAMES else split_dir
            splits.setdefault(key, []).extend(pairs)

    return YoloDataset(root=root, names=names, splits=splits)
