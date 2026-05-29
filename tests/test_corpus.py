"""Unit tests for ``src.corpus`` modules (Étape 2.4)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus.parser import _extract_zone_type, _parse_points, parse_page_xml
from src.corpus.splitting import (
    build_folio_index,
    compute_sha256,
    save_split,
    split_temporally,
)

MINIMAL_PAGE_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
      <Page imageFilename="test_page.jpg" imageWidth="1000" imageHeight="1500">
        <TextRegion id="r1" custom="structure {type:block;}">
          <Coords points="50,50 950,50 950,1450 50,1450"/>
          <TextLine id="line_1">
            <Coords points="60,60 940,60 940,120 60,120"/>
            <Baseline points="60,100 940,100"/>
            <TextEquiv><Unicode>In capitulo parisiense</Unicode></TextEquiv>
          </TextLine>
          <TextLine id="line_2_no_text">
            <Coords points="60,130 940,130 940,190 60,190"/>
          </TextLine>
        </TextRegion>
      </Page>
    </PcGts>
""")


# -------- parser tests --------

def test_parse_points_simple() -> None:
    """Comma-separated coordinate pairs are split into integers."""
    assert _parse_points("10,20 30,40") == [(10, 20), (30, 40)]


def test_parse_points_empty() -> None:
    """Empty input returns an empty list, no exception."""
    assert _parse_points("") == []


def test_extract_zone_type_block() -> None:
    """The ``type:`` value is lifted out of the custom attribute."""
    assert _extract_zone_type("structure {type:block;}") == "block"


def test_extract_zone_type_missing() -> None:
    """Custom attributes without a structure declaration return None."""
    assert _extract_zone_type("readingDirection {direction:lr;}") is None


def test_parse_page_xml_returns_one_line(tmp_path: Path) -> None:
    """A page with one transcribed and one untranscribed line yields one entry."""
    xml_file = tmp_path / "0001_Main_frame.xml"
    xml_file.write_text(MINIMAL_PAGE_XML, encoding="utf-8")

    lines = parse_page_xml(xml_file, register="LL999", year=1400)

    assert len(lines) == 1
    line = lines[0]
    assert line["line_id"] == "line_1"
    assert line["content"] == "In capitulo parisiense"
    assert line["register"] == "LL999"
    assert line["year"] == 1400
    assert line["zone_type"] == "block"
    assert line["page"] == "test_page.jpg"
    assert line["polygon"][0] == (60, 60)
    assert line["baseline"] == [(60, 100), (940, 100)]


# -------- splitting tests --------

def test_build_folio_index_minimal(tmp_path: Path) -> None:
    """A single <doc> header is parsed into a (volume, year) tuple."""
    headers = tmp_path / "headers.txt"
    headers.write_text(
        '<doc folio="FRAN_0393_12418_L" volume="LL119" date="1458,1458::août" >\n',
        encoding="utf-8",
    )
    index = build_folio_index(headers)
    assert index == {"FRAN_0393_12418_L": ("LL119", 1458)}


def test_split_temporally_dispatches_correctly() -> None:
    """Lines are routed into train/val/test based on their year."""
    lines = [
        {"year": 1400, "content": "early"},
        {"year": 1450, "content": "middle"},
        {"year": 1500, "content": "late"},
        {"year": None, "content": "no year, dropped"},
    ]
    splits = split_temporally(lines, train_max_year=1439, val_max_year=1469)
    assert [item["content"] for item in splits["train"]] == ["early"]
    assert [item["content"] for item in splits["val"]] == ["middle"]
    assert [item["content"] for item in splits["test"]] == ["late"]


def test_split_temporally_rejects_invalid_bounds() -> None:
    """The validation period cannot be empty."""
    with pytest.raises(ValueError):
        split_temporally([], train_max_year=1500, val_max_year=1400)


def test_compute_sha256_is_deterministic() -> None:
    """Equal payloads (even with different dict ordering) yield the same digest."""
    payload_a = {"a": 1, "b": [1, 2]}
    payload_b = {"b": [1, 2], "a": 1}
    assert compute_sha256(payload_a) == compute_sha256(payload_b)


def test_save_split_writes_three_files_and_seal(tmp_path: Path) -> None:
    """Saving produces train/val/test JSON files and returns their digests."""
    splits = {
        "train": [{"content": "x", "polygon": [(0, 0), (1, 1)], "baseline": []}],
        "val": [],
        "test": [{"content": "y", "polygon": [(2, 2)], "baseline": [(0, 0)]}],
    }
    digests = save_split(splits, tmp_path)

    for name in ("train", "val", "test"):
        assert (tmp_path / f"{name}.json").is_file()
        assert digests[name] and len(digests[name]) == 64  # SHA-256 hex length
