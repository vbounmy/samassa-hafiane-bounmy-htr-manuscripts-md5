"""Parse PAGE XML files from the ANR e-NDP corpus.

The e-NDP ground-truth is distributed in PRImA PAGE XML 2019-07-15.
Each XML describes one page (or page half: Left/Right/Main_frame) and
contains the typology specific to capitular registers (block, liste,
Date, entrée, numérotation), encoded in the ``custom="structure
{type:...;}"`` attribute of every TextRegion.

This module exposes :func:`parse_page_xml`, which returns a flat list of
line dictionaries suitable for downstream serialization (JSON, HuggingFace
Dataset, ...). It is used by ``scripts/build_split.py`` to extract every
text line of the corpus before splitting them temporally.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_NS = {"p": PAGE_NS}

_STRUCTURE_TYPE_RE = re.compile(r"structure\s*\{[^}]*type:\s*([^;}]+)")


def _parse_points(points_str: str) -> list[tuple[int, int]]:
    """Convert a PAGE XML ``points`` attribute into a list of (x, y) tuples.

    PAGE XML separates coordinate pairs with spaces and the two members of
    a pair with a comma, e.g. ``"185,61 185,215 329,215"``.

    Args:
        points_str: Raw value of a ``points`` attribute.

    Returns:
        List of integer (x, y) pairs in the order they appear.
    """
    if not points_str:
        return []
    pairs: list[tuple[int, int]] = []
    for token in points_str.split():
        x_str, y_str = token.split(",")
        pairs.append((int(float(x_str)), int(float(y_str))))
    return pairs


def _extract_zone_type(custom_str: str) -> str | None:
    """Extract the e-NDP structure type from a ``custom`` attribute.

    Args:
        custom_str: Raw value of the ``custom`` attribute of a TextRegion.

    Returns:
        The structure type (``"block"``, ``"Date"``, ``"liste"``, ...) or
        ``None`` if the attribute does not declare one.
    """
    match = _STRUCTURE_TYPE_RE.search(custom_str or "")
    return match.group(1).strip() if match else None


def parse_page_xml(
    xml_path: Path,
    *,
    register: str | None = None,
    year: int | None = None,
) -> list[dict]:
    """Extract every transcribed line from one PAGE XML file.

    Args:
        xml_path: Path to the PAGE XML file to parse.
        register: Identifier of the source register (e.g. ``"LL119"``).
        year: Year associated with the source page, used by downstream
            temporal splitters.

    Returns:
        List of line dicts, one per transcribed line. Each dict contains:

        - ``line_id``: PAGE XML line identifier.
        - ``page``: ``imageFilename`` of the source page.
        - ``register``: register identifier (passed in).
        - ``year``: page year (passed in).
        - ``zone_type``: e-NDP zone type or ``None``.
        - ``content``: transcription text (always non-empty in the output).
        - ``polygon``: list of (x, y) pairs describing the line contour.
        - ``baseline``: list of (x, y) pairs describing the line baseline.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page_el = root.find("p:Page", _NS)
    page_filename = (
        page_el.get("imageFilename", xml_path.stem) if page_el is not None else xml_path.stem
    )

    lines: list[dict] = []
    for region in root.iter(f"{{{PAGE_NS}}}TextRegion"):
        zone_type = _extract_zone_type(region.get("custom", ""))

        for line in region.iter(f"{{{PAGE_NS}}}TextLine"):
            unicode_el = line.find("p:TextEquiv/p:Unicode", _NS)
            if unicode_el is None or not unicode_el.text or not unicode_el.text.strip():
                continue
            content = unicode_el.text

            coords_el = line.find("p:Coords", _NS)
            if coords_el is None:
                continue
            polygon = _parse_points(coords_el.get("points", ""))
            if not polygon:
                continue

            baseline_el = line.find("p:Baseline", _NS)
            baseline = _parse_points(baseline_el.get("points", "")) if baseline_el is not None else []

            lines.append({
                "line_id": line.get("id", ""),
                "page": page_filename,
                "register": register,
                "year": year,
                "zone_type": zone_type,
                "content": content,
                "polygon": polygon,
                "baseline": baseline,
            })

    return lines


def iter_xml_files(corpus_dir: Path) -> Iterator[Path]:
    """Yield every PAGE XML file inside a corpus directory, sorted by name.

    Args:
        corpus_dir: Directory holding the ``*.xml`` files (flat layout).

    Yields:
        Paths to the XML files in alphabetical order, for reproducibility.
    """
    yield from sorted(corpus_dir.glob("*.xml"))
