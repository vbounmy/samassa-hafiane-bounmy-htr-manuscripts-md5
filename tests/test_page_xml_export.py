"""Unit tests for ``src.corpus.page_xml_export`` (Étape 2.5)."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.page_xml_export import (
    _points_to_str,
    export_split_to_page_xml,
    lines_to_page_xml,
)

_PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def test_points_to_str_formats_correctly() -> None:
    """A list of point pairs is rendered as the expected PAGE-style string."""
    assert _points_to_str([(10, 20), (30, 40)]) == "10,20 30,40"
    assert _points_to_str([[1.5, 2.5], [3.9, 4.1]]) == "1,2 3,4"  # ints only


def test_lines_to_page_xml_minimal_round_trip() -> None:
    """Generated XML is well-formed and includes our line content."""
    lines = [
        {
            "line_id": "line_1",
            "content": "Hello world",
            "polygon": [(10, 10), (200, 10), (200, 50), (10, 50)],
            "baseline": [(10, 40), (200, 40)],
        }
    ]
    xml_text = lines_to_page_xml("page.jpg", lines)
    root = ET.fromstring(xml_text)
    assert root.tag.endswith("PcGts")

    # Page filename is preserved.
    page = root.find(f"{{{_PAGE_NS}}}Page")
    assert page is not None
    assert page.get("imageFilename") == "page.jpg"

    # One TextLine with content and baseline.
    text_line = root.find(f".//{{{_PAGE_NS}}}TextLine")
    assert text_line is not None
    assert text_line.get("id") == "line_1"
    baseline = text_line.find(f"{{{_PAGE_NS}}}Baseline")
    assert baseline is not None and baseline.get("points") == "10,40 200,40"
    unicode_el = text_line.find(f".//{{{_PAGE_NS}}}Unicode")
    assert unicode_el is not None and unicode_el.text == "Hello world"


def test_export_split_writes_one_xml_per_page(tmp_path: Path) -> None:
    """A 2-page split produces exactly two XML files."""
    split = [
        {"line_id": "l1", "page": "p1.jpg", "content": "a",
         "polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "baseline": []},
        {"line_id": "l2", "page": "p1.jpg", "content": "b",
         "polygon": [(0, 20), (10, 20), (10, 30), (0, 30)], "baseline": []},
        {"line_id": "l3", "page": "p2.jpg", "content": "c",
         "polygon": [(5, 5), (15, 5), (15, 15), (5, 15)], "baseline": []},
    ]
    split_json = tmp_path / "train.json"
    split_json.write_text(json.dumps(split), encoding="utf-8")

    out_dir = tmp_path / "out"
    summary = export_split_to_page_xml(split_json, out_dir)

    assert summary == {"n_pages": 2, "n_lines": 3}
    assert (out_dir / "p1.xml").is_file()
    assert (out_dir / "p2.xml").is_file()

    # p1.xml must contain both line IDs.
    text = (out_dir / "p1.xml").read_text(encoding="utf-8")
    assert 'id="l1"' in text and 'id="l2"' in text
