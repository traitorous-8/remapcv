# remapcv

Merge computer-vision YOLO datasets with smart class remapping.

Combining object-detection datasets from different sources is painful because the same object is called `helmet` in one, `hard_hat` in another, `каска` in a third. `remapcv` solves this by intelligently collapsing synonyms into a single, clean schema and merging everything into one ready-to-train dataset.

## Installation

Install from source for now:

```bash
git clone https://github.com/yourusername/remapcv.git
cd remapcv
pip install -e .
```

## Quickstart

See how it works using the built-in demo.

**1. Generate sample data**
Create two sample datasets with mismatched class names:
```bash
remapcv demo
```

**2. Scan and auto-map**
Scan the datasets and automatically suggest synonym groups:
```bash
remapcv scan A B --auto mapping.yaml
```

*Example Output:*
```
Auto-suggested Class Merges:
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Target   ┃ Sources                      ┃ Confidence ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ helmet   │ helmet, hard_hat             │ 0.95       │
│ vest     │ vest, safety_vest            │ 0.95       │
│ person   │ person, worker               │ 0.95       │
└──────────┴──────────────────────────────┴────────────┘
```

**3. Review the mapping**
The command above generates a `mapping.yaml` file. You can review and edit it before merging.

*Example `mapping.yaml`:*
```yaml
target_classes:
  - name: helmet
    sources:
      - helmet
      - hard_hat
  - name: vest
    sources:
      - vest
      - safety_vest
  - name: person
    sources:
      - person
      - worker
```

**4. Merge datasets**
Produce one clean dataset using the mapping:
```bash
remapcv merge A B -c mapping.yaml -o merged
```

## How Auto-Mapping Works

The `--auto` flag takes the guesswork out of dataset reconciliation. It groups classes using:
- **String Similarity:** Fuzzy matching to catch typos and plurals.
- **Curated Domain Hints:** Pre-loaded knowledge for common domains (like PPE, where "hard_hat" and "helmet" are known synonyms).
- **Confidence Scores:** Each grouping is assigned a confidence score. Low-confidence groups are flagged for manual review, ensuring you remain in control of the final ontology.

## Why not just use X?

There are many excellent format converters and dataset management tools. However, most focus on format conversion or visualization. While they allow manual class renaming, none handle *intelligent class remapping* when merging disparate datasets together out-of-the-box. `remapcv` is a lightweight, dedicated CLI tool specifically for this pain point.

## Roadmap

- [ ] COCO and Pascal VOC input formats
- [ ] Dataset audit command to detect class imbalances and missing labels
- [ ] VLM-assisted auto-labeling for unmapped classes

## License

MIT License.
