"""Tests for HTR dataset preparation and training utilities."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare_htr_dataset import (
    crop_line_image,
    parse_page_xml,
    safe_filename,
    split_signature,
)
from src.train_kraken import generate_kraken_script
from src.train_trocr import log_to_journal


def test_parse_page_xml_extracts_line_text_and_polygon(tmp_path) -> None:
    """Check PAGE XML parsing for line coordinates, text, and review flags."""
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Page imageFilename="sample.png" imageWidth="100" imageHeight="40">
    <TextRegion id="r1">
      <Coords points="0,0 90,0 90,30 0,30"/>
      <TextLine id="l1">
        <Coords points="10,5 80,5 80,20 10,20"/>
        <TextEquiv><Unicode>ab&#39;c $sic$</Unicode></TextEquiv>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
        encoding="utf-8",
    )

    lines = parse_page_xml(str(xml_path))

    assert len(lines) == 1
    assert lines[0]["line_id"] == "l1"
    assert lines[0]["polygon"] == [(10, 5), (80, 5), (80, 20), (10, 20)]
    assert lines[0]["text"] == "ab'c $sic$"
    assert lines[0]["needs_review"] is True


def test_crop_line_image_respects_image_bounds() -> None:
    """Check that line crops are clipped to image dimensions."""
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_line_image(image, [(-5, -5), (12, 0), (12, 10), (0, 10)], padding=3)

    assert crop.shape[0] == 13
    assert crop.shape[1] == 15
    assert crop.dtype == np.uint8


def test_safe_filename_and_split_signature_are_deterministic() -> None:
    """Check filename normalization and stable metadata signatures."""
    assert safe_filename("page 1/r2:l3") == "page_1_r2_l3"

    records = [
        {"file_name": "b.png", "text": "beta", "page": "p2", "line_id": "l2"},
        {"file_name": "a.png", "text": "alpha", "page": "p1", "line_id": "l1"},
    ]
    assert split_signature(records) == split_signature(list(reversed(records)))


def test_log_to_journal_writes_jsonl(tmp_path) -> None:
    """Check that training runs are appended as valid JSON Lines records."""
    journal_path = tmp_path / "journal.jsonl"

    log_to_journal(
        str(journal_path),
        run_id="unit_run",
        model_name="microsoft/trocr-base-handwritten",
        parameters={"epochs": 1},
        metrics={"eval_cer": 0.1, "eval_wer": 0.2},
    )

    record = json.loads(journal_path.read_text(encoding="utf-8").strip())
    assert record["run_id"] == "unit_run"
    assert record["parameters"]["epochs"] == 1
    assert record["metrics"]["eval_cer"] == 0.1


def test_generate_kraken_script_contains_valid_shell_guard(tmp_path) -> None:
    """Check that the generated Kraken training script has a valid ketos guard."""
    script_path = tmp_path / "run_kraken_training.sh"

    generate_kraken_script(str(script_path), "data/kraken_dataset/train.txt", "data/kraken_dataset/validation.txt")

    script = script_path.read_text(encoding="utf-8")
    assert "if ! command -v ketos &> /dev/null\nthen" in script
    assert "clone" not in script
    assert "ketos train" in script
