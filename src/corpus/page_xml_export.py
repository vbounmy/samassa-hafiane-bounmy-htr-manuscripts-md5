"""Re-emit PAGE XML files from the split JSONs.

This module rebuilds standard PAGE XML 2019-07-15 files from the train/
val/test JSON splits produced by ``build_split.py``. Each output XML
contains one TextRegion per page (we collapse the e-NDP fine-grained
typology because the downstream HTR engine — Kraken — only consumes
TextLine geometries, not region types).

Why we do this even though e-NDP already ships PAGE XML files:

1. **Consistency**: every line in the export carries the same
   ``register``/``year`` metadata that we use for the temporal split,
   making it trivial to re-derive train/val/test from the files alone.
2. **Split-aware loading**: by writing one XML directory per split, we
   can hand Kraken just the train and val sets at step 3 without ever
   touching the sealed test directory.
3. **Brief compliance**: the project brief (step 2.5) requires that
   the segmentation polygons be persisted as ``.page.xml`` or as
   dictionaries attached to each line. We do both.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

_PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_SCHEMA_LOC = (
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
)


def _points_to_str(points: list) -> str:
    """Render a list of [x, y] (or (x, y)) pairs as a PAGE ``points`` string.

    Args:
        points: List of two-element sequences (list or tuple).

    Returns:
        Space-separated ``"x1,y1 x2,y2 ..."`` PAGE-compatible string.
    """
    return " ".join(f"{int(p[0])},{int(p[1])}" for p in points)


def lines_to_page_xml(
    page_filename: str,
    lines: list[dict],
) -> str:
    """Render a list of line dicts as a PAGE XML 2019-07-15 document.

    Args:
        page_filename: Value to set as ``imageFilename`` on the ``Page``
            element (typically ``"FRAN_0393_12418_L.jpg"``).
        lines: Lines as returned by the parser (must include ``polygon``,
            ``baseline``, ``content``, ``line_id``).

    Returns:
        Serialized XML as a UTF-8 string, ready to write to disk.
    """
    # Compute a bounding box for the implicit "page" extent. We use the
    # max coordinate across all polygons rather than the original image
    # size, which isn't stored in the JSON. Kraken only needs a sane
    # upper bound for its coordinate system.
    max_x = max((p[0] for line in lines for p in line.get("polygon", [])), default=0)
    max_y = max((p[1] for line in lines for p in line.get("polygon", [])), default=0)

    ET.register_namespace("", _PAGE_NS)
    pc_gts = ET.Element(
        f"{{{_PAGE_NS}}}PcGts",
        {"{http://www.w3.org/2001/XMLSchema-instance}schemaLocation": _SCHEMA_LOC},
    )
    page = ET.SubElement(
        pc_gts,
        f"{{{_PAGE_NS}}}Page",
        {
            "imageFilename": page_filename,
            "imageWidth": str(max(max_x + 1, 1)),
            "imageHeight": str(max(max_y + 1, 1)),
        },
    )

    region = ET.SubElement(
        page,
        f"{{{_PAGE_NS}}}TextRegion",
        {"id": f"region_{Path(page_filename).stem}"},
    )
    # The bounding region's coords are the page's full rectangle.
    ET.SubElement(
        region,
        f"{{{_PAGE_NS}}}Coords",
        {"points": _points_to_str([(0, 0), (max_x, 0), (max_x, max_y), (0, max_y)])},
    )

    for line in lines:
        line_el = ET.SubElement(
            region,
            f"{{{_PAGE_NS}}}TextLine",
            {"id": line.get("line_id") or f"line_{len(region)}"},
        )
        ET.SubElement(
            line_el,
            f"{{{_PAGE_NS}}}Coords",
            {"points": _points_to_str(line.get("polygon", []))},
        )
        baseline = line.get("baseline", [])
        if baseline:
            ET.SubElement(
                line_el,
                f"{{{_PAGE_NS}}}Baseline",
                {"points": _points_to_str(baseline)},
            )
        text_equiv = ET.SubElement(line_el, f"{{{_PAGE_NS}}}TextEquiv")
        unicode_el = ET.SubElement(text_equiv, f"{{{_PAGE_NS}}}Unicode")
        unicode_el.text = line.get("content", "")

    return ET.tostring(pc_gts, encoding="unicode", xml_declaration=True)


def export_split_to_page_xml(
    split_json_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Write one PAGE XML file per page referenced in a split JSON.

    Args:
        split_json_path: Path to a ``train.json`` / ``val.json`` /
            ``test.json`` produced by ``build_split.py``.
        output_dir: Directory in which the XML files are written.
            Created if it does not exist.

    Returns:
        A summary dict: ``{"n_pages": ..., "n_lines": ...}``.
    """
    lines = json.loads(split_json_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    lines_by_page: dict[str, list[dict]] = defaultdict(list)
    for line in lines:
        lines_by_page[line["page"]].append(line)

    for page_filename, page_lines in lines_by_page.items():
        xml_text = lines_to_page_xml(page_filename, page_lines)
        out_path = output_dir / f"{Path(page_filename).stem}.xml"
        out_path.write_text(xml_text, encoding="utf-8")

    return {"n_pages": len(lines_by_page), "n_lines": len(lines)}
