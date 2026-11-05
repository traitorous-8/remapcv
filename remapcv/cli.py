"""remapcv command-line interface.

Commands:
  remapcv scan   DATASET...            inspect datasets, list all classes
  remapcv merge  DATASET... -c map.yaml -o out/   remap classes and merge
  remapcv demo                         generate fake datasets to try it out
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .fake import make_fake_dataset
from .merge import merge_datasets
from .scan import build_class_index, scan_datasets, write_mapping_skeleton

app = typer.Typer(
    add_completion=False,
    help="Merge computer-vision datasets with smart class remapping.",
)
console = Console()


@app.command()
def scan(
    datasets: list[Path] = typer.Argument(..., help="One or more YOLO dataset folders."),
    write_mapping: Path = typer.Option(
        None, "--write-mapping", "-w", help="Write a starter mapping.yaml to this path."
    ),
):
    """Inspect datasets and list every class across all of them."""
    dss = scan_datasets(datasets)
    index = build_class_index(dss)

    root_names = [ds.root.name for ds in dss]
    table = Table(title="Classes across datasets")
    table.add_column("class", style="cyan", no_wrap=True)
    for rn in root_names:
        table.add_column(rn, justify="right")
    table.add_column("total", justify="right", style="bold")

    for cls in sorted(index):
        row = [cls]
        total = 0
        for rn in root_names:
            cnt = index[cls].get(rn, 0)
            total += cnt
            row.append(str(cnt) if cnt else "-")
        row.append(str(total))
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]{len(index)}[/bold] distinct class names across "
        f"[bold]{len(dss)}[/bold] dataset(s)."
    )

    # nudge toward the next step
    dupes = [c for c in index if len(index[c]) < len(dss)]
    if len(dss) > 1 and dupes:
        console.print(
            "\n[yellow]Tip:[/yellow] some classes appear in only some datasets — "
            "likely synonyms to merge (e.g. helmet / hard_hat)."
        )

    if write_mapping:
        path = write_mapping_skeleton(dss, write_mapping)
        console.print(f"\nStarter mapping written to [green]{path}[/green] — edit it, then run `remapcv merge`.")


@app.command()
def merge(
    datasets: list[Path] = typer.Argument(..., help="YOLO dataset folders to merge."),
    config: Path = typer.Option(..., "--config", "-c", help="mapping.yaml with class remapping."),
    output: Path = typer.Option(..., "--output", "-o", help="Output folder for the merged dataset."),
):
    """Apply a class mapping and merge datasets into one clean YOLO dataset."""
    report = merge_datasets(datasets, config, output)

    table = Table(title="Merged dataset")
    table.add_column("target class", style="cyan")
    table.add_column("boxes", justify="right", style="bold")
    for name in report.target_classes:
        table.add_row(name, str(report.per_class_boxes.get(name, 0)))
    console.print(table)

    imgs = ", ".join(f"{k}={v}" for k, v in report.images_written.items())
    console.print(
        f"\n[green]Done.[/green] {report.boxes_written} boxes written "
        f"({report.boxes_dropped} dropped), images: {imgs}"
    )
    console.print(f"Output: [green]{output}[/green]")


@app.command()
def demo(
    output: Path = typer.Option(Path("./remapcv_demo"), "--output", "-o", help="Where to put demo datasets."),
):
    """Generate two small fake datasets with mismatched class names to try remapcv."""
    d1 = make_fake_dataset(output / "site_a", ["helmet", "vest", "person"], seed=1)
    d2 = make_fake_dataset(output / "site_b", ["hard_hat", "safety_vest", "worker"], seed=2)
    console.print(f"Created two demo datasets in [green]{output}[/green]:")
    console.print(f"  • {d1.name}: helmet, vest, person")
    console.print(f"  • {d2.name}: hard_hat, safety_vest, worker  [dim](synonyms!)[/dim]")
    console.print("\nTry:")
    console.print(f"  [cyan]remapcv scan {d1} {d2} -w mapping.yaml[/cyan]")
    console.print("  [dim]# edit mapping.yaml to merge synonyms, then:[/dim]")
    console.print(f"  [cyan]remapcv merge {d1} {d2} -c mapping.yaml -o {output}/merged[/cyan]")


def main():
    app()


if __name__ == "__main__":
    main()
