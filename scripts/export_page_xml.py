"""Export each split JSON to per-page PAGE XML files (Étape 2.5).

Run from the project root after ``build_split.py`` has been executed::

    python scripts/export_page_xml.py \\
        --splits-dir data/processed/endp \\
        --output-dir data/processed/page_xml_export

Output layout::

    data/processed/page_xml_export/
    ├── train/    one *.xml per page
    ├── val/      one *.xml per page
    └── test/     one *.xml per page  (sealed — do not open during dev)

These files are the deliverable expected by step 2.5 of the brief and
will feed Kraken's recognition training in step 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.page_xml_export import export_split_to_page_xml


def main() -> None:
    """Entry point: iterate over the three splits, write XMLs, report counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/processed/endp"),
        help="Directory containing train.json / val.json / test.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/page_xml_export"),
        help="Destination directory. Subfolders train/val/test are created.",
    )
    args = parser.parse_args()

    totals = {"n_pages": 0, "n_lines": 0}
    for split_name in ("train", "val", "test"):
        split_json = args.splits_dir / f"{split_name}.json"
        if not split_json.exists():
            print(f"  ! {split_json} missing, skipping {split_name}.")
            continue
        out_dir = args.output_dir / split_name
        summary = export_split_to_page_xml(split_json, out_dir)
        print(
            f"  {split_name}: {summary['n_pages']:>3} pages, "
            f"{summary['n_lines']:>6} lines -> {out_dir}"
        )
        totals["n_pages"] += summary["n_pages"]
        totals["n_lines"] += summary["n_lines"]

    print(
        f"\nTotal exported: {totals['n_pages']} pages / {totals['n_lines']} lines."
    )


if __name__ == "__main__":
    main()
