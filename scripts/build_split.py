"""Build the train/val/test split of e-NDP and seal the test set.

Run from the project root::

    python scripts/build_split.py \\
        --corpus-root /path/to/HTR_Ground-Truth/page_xml \\
        --headers /path/to/doc_headers.txt

Outputs are written under ``data/processed/endp/``:

- ``train.json``, ``val.json``, ``test.json`` — the split files.
- ``stats.json`` — counts per register/year and SHA-256 digests of every
  split, including the sealed test-set hash demanded by the brief.

Default cutoff years implement the diachronic axis chosen for the project:

- 1320–1439 → ``train`` (early Middle French cursive, 14th + early 15th)
- 1440–1469 → ``val``   (mid 15th)
- 1470–1504 → ``test``  (late 15th + early 16th — sealed, used once)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.parser import iter_xml_files, parse_page_xml
from src.corpus.splitting import (
    build_folio_index,
    compute_sha256,
    save_split,
    split_temporally,
)

DEFAULT_TRAIN_MAX_YEAR = 1439
DEFAULT_VAL_MAX_YEAR = 1469


def main() -> None:
    """Entry point: parse, map, split, hash, save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/endp"),
    )
    parser.add_argument(
        "--train-max-year",
        type=int,
        default=DEFAULT_TRAIN_MAX_YEAR,
    )
    parser.add_argument(
        "--val-max-year",
        type=int,
        default=DEFAULT_VAL_MAX_YEAR,
    )
    args = parser.parse_args()

    # 1. Build the folio→(register, year) index from the benchmark headers.
    print(f"Building folio index from {args.headers} ...")
    folio_index = build_folio_index(args.headers)
    print(f"  -> {len(folio_index):,} folio entries indexed.")

    # 2. Walk the corpus, parse each XML, attach its register/year if known.
    print(f"\nParsing PAGE XML files under {args.corpus_root} ...")
    all_lines: list[dict] = []
    files_mapped = files_unmapped = 0
    for xml_path in iter_xml_files(args.corpus_root):
        info = folio_index.get(xml_path.stem)
        register, year = info if info else (None, None)
        lines = parse_page_xml(xml_path, register=register, year=year)
        all_lines.extend(lines)
        if info is not None:
            files_mapped += 1
        else:
            files_unmapped += 1
    print(
        f"  -> {files_mapped} files mapped, {files_unmapped} unmapped."
    )
    print(f"  -> {len(all_lines):,} text lines extracted in total.")

    # 3. Per-register and per-decade breakdowns (for reproducibility/audit).
    per_register = Counter(line["register"] for line in all_lines if line.get("register"))
    print("\nLines per register (mapped only):")
    for reg, n in sorted(per_register.items()):
        print(f"  {reg:<12} {n:>5}")

    per_decade: Counter[int] = Counter()
    for line in all_lines:
        yr = line.get("year")
        if yr is not None:
            per_decade[(yr // 10) * 10] += 1

    # 4. Apply the temporal split.
    splits = split_temporally(
        all_lines,
        train_max_year=args.train_max_year,
        val_max_year=args.val_max_year,
    )
    print(
        f"\nSplit sizes "
        f"(train ≤ {args.train_max_year}  /  val ≤ {args.val_max_year}  /  test > {args.val_max_year}):"
    )
    for name, items in splits.items():
        print(f"  {name:<6} {len(items):>6} lines")

    # 5. Save splits and compute their SHA-256 seals.
    digests = save_split(splits, args.output_dir)
    print("\nSHA-256 digests (record these in the article and README):")
    for split_name, digest in digests.items():
        print(f"  {split_name}: {digest}")

    # 6. Persist auxiliary metadata for reproducibility.
    metadata = {
        "corpus_root": str(args.corpus_root),
        "headers": str(args.headers),
        "n_files_total": files_mapped + files_unmapped,
        "n_files_mapped": files_mapped,
        "n_files_unmapped": files_unmapped,
        "n_lines_total": len(all_lines),
        "n_lines_per_register": dict(per_register),
        "n_lines_per_decade": dict(per_decade),
        "split_thresholds": {
            "train_max_year": args.train_max_year,
            "val_max_year": args.val_max_year,
        },
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "sha256_digests": digests,
    }
    metadata_path = args.output_dir / "stats.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nMetadata written to {metadata_path}")
    print(f"Metadata SHA-256: {compute_sha256(metadata)}")


if __name__ == "__main__":
    main()
