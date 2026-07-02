"""Generate tiny fake YOLO datasets for testing remapcv without downloading anything.

Creates real folder structures with real (1x1) images and real label files, so the
parser and merge logic run against genuine files -- just small and synthetic.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

# Minimal valid 1x1 PNG (a single pixel) so we don't depend on Pillow.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8cfc0f01f0005010100a05f8c6c0000000049454e44ae426082"
)


def make_fake_dataset(
    root: str | Path,
    class_names: list[str],
    n_train: int = 6,
    n_valid: int = 2,
    seed: int | None = None,
) -> Path:
    """Create a small YOLO dataset with the given class names."""
    if seed is not None:
        random.seed(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # data.yaml
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "train/images",
                "val": "valid/images",
                "nc": len(class_names),
                "names": class_names,
            },
            sort_keys=False,
        )
    )

    for split, n in (("train", n_train), ("valid", n_valid)):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (img_dir / f"img_{i:03d}.png").write_bytes(_PNG_1x1)
            lines = []
            for _ in range(random.randint(1, 3)):
                cid = random.randrange(len(class_names))
                cx, cy = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
                w, h = random.uniform(0.05, 0.3), random.uniform(0.05, 0.3)
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / f"img_{i:03d}.txt").write_text("\n".join(lines) + "\n")

    return root
