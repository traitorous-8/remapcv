# remapcv

Merge computer-vision datasets with smart class remapping — one command.

Combining object-detection datasets from different sources is painful: the same
object is called `helmet` in one, `hard_hat` in another, `каска` in a third.
`remapcv` scans them, lets you collapse synonyms into one clean schema, and
merges everything into a single YOLO dataset.

## Install

```bash
pip install remapcv
```

## Quickstart

```bash
# 1. see what you're working with (and get a starter mapping file)
remapcv scan ./dataset_a ./dataset_b -w mapping.yaml

# 2. edit mapping.yaml — point synonyms at one target: hard_hat -> helmet

# 3. merge into one clean dataset
remapcv merge ./dataset_a ./dataset_b -c mapping.yaml -o ./merged
```

Try it with generated demo data:

```bash
remapcv demo
```

## License

MIT
