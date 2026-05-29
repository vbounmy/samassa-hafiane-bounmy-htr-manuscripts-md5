"""Build a folio→(register, year) index and apply a temporal split.

The e-NDP corpus does not natively expose the register source of each
PAGE XML file. The information lives in a separate vertical-format file
released with the benchmark repository
(``No_Sketch_Engine/endp/vertical/source``).

This module:

1. Parses the ``<doc>`` headers of that vertical file to build a mapping
   ``folio_identifier -> (register_id, year)``.
2. Applies a temporal split to a flat list of parsed lines: ``train`` for
   early years, ``val`` for middle years, ``test`` for late years (with
   the exact cut-off years chosen by the caller).
3. Saves each split to disk as JSON and returns the SHA-256 digest of
   each file, so that the test set can be sealed as the project brief
   requires.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

_FOLIO_RE = re.compile(r'folio="([^"]+)"')
_VOLUME_RE = re.compile(r'volume="([^"]+)"')
_YEAR_RE = re.compile(r'date="(\d{4})')


def build_folio_index(headers_path: Path) -> dict[str, tuple[str, int]]:
    """Parse vertical-source ``<doc>`` headers into a folio→(register, year) map.

    Args:
        headers_path: Path to a plain-text file containing one ``<doc>``
            header per line (typically produced by
            ``grep '^<doc ' source > headers.txt``).

    Returns:
        Mapping from folio identifier (the value of ``folio=``, e.g.
        ``"FRAN_0393_12418_L"``) to a tuple ``(register_id, year)``.

    Raises:
        FileNotFoundError: If ``headers_path`` does not exist.
    """
    index: dict[str, tuple[str, int]] = {}
    with headers_path.open(encoding="utf-8") as stream:
        for line in stream:
            folio_match = _FOLIO_RE.search(line)
            volume_match = _VOLUME_RE.search(line)
            year_match = _YEAR_RE.search(line)
            if folio_match and volume_match and year_match:
                index[folio_match.group(1)] = (
                    volume_match.group(1),
                    int(year_match.group(1)),
                )
    return index


def split_temporally(
    lines: Iterable[dict],
    *,
    train_max_year: int,
    val_max_year: int,
) -> dict[str, list[dict]]:
    """Partition a flat list of lines into train/val/test by ``year`` field.

    Args:
        lines: Iterable of line dicts. Each must carry an integer ``year``
            field (lines whose ``year`` is ``None`` are silently dropped).
        train_max_year: Inclusive upper bound of the training period.
            All lines with ``year <= train_max_year`` go to ``train``.
        val_max_year: Inclusive upper bound of the validation period.
            Lines with ``train_max_year < year <= val_max_year`` go to ``val``.
            All other lines (``year > val_max_year``) go to ``test``.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        lists of line dicts.

    Raises:
        ValueError: If ``train_max_year >= val_max_year``, which would
            create an empty validation period.
    """
    if train_max_year >= val_max_year:
        raise ValueError(
            "train_max_year must be strictly smaller than val_max_year, "
            f"got {train_max_year} >= {val_max_year}"
        )

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for line in lines:
        year = line.get("year")
        if year is None:
            continue
        if year <= train_max_year:
            splits["train"].append(line)
        elif year <= val_max_year:
            splits["val"].append(line)
        else:
            splits["test"].append(line)
    return splits


def compute_sha256(payload: object) -> str:
    """Compute a deterministic SHA-256 of any JSON-serializable payload.

    Sorting keys ensures the digest is independent of dict ordering,
    which matters for the sealed test-set hash required by the brief.

    Args:
        payload: Any JSON-serializable structure.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_split(
    splits: dict[str, list[dict]],
    output_dir: Path,
) -> dict[str, str]:
    """Write each split to a JSON file and return their SHA-256 digests.

    Polygons and baselines, stored internally as lists of tuples, are
    coerced to lists of lists during serialization (JSON has no tuple type).

    Args:
        splits: Output of :func:`split_temporally`.
        output_dir: Destination directory. Created if it does not exist.

    Returns:
        Mapping ``split_name -> sha256`` of the serialized JSON content.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for split_name, items in splits.items():
        serializable = [
            {
                **item,
                "polygon": [list(point) for point in item["polygon"]],
                "baseline": [list(point) for point in item["baseline"]],
            }
            for item in items
        ]
        path = output_dir / f"{split_name}.json"
        path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        digests[split_name] = compute_sha256(serializable)
    return digests
