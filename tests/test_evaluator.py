"""Unit tests for ``src.segmentation.evaluator`` (Étape 2.2)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation.evaluator import (
    best_iou_per_zone,
    iou,
    load_ground_truth_zones,
    summarize_by_zone_type,
)


def test_iou_of_identical_masks_is_one() -> None:
    """Two identical masks have an IoU of 1.0."""
    mask = np.array([[True, True], [False, False]])
    assert iou(mask, mask) == 1.0


def test_iou_of_disjoint_masks_is_zero() -> None:
    """Two non-overlapping masks have an IoU of 0.0."""
    a = np.array([[True, False], [False, False]])
    b = np.array([[False, False], [False, True]])
    assert iou(a, b) == 0.0


def test_iou_of_partial_overlap_is_correct() -> None:
    """Half-overlap of two 2-pixel masks yields IoU = 1/3."""
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [True, False]])
    assert abs(iou(a, b) - 1.0 / 3) < 1e-9


def test_iou_of_two_empty_masks_is_zero() -> None:
    """Two empty masks degenerate to 0 to avoid division by zero."""
    empty = np.zeros((2, 2), dtype=bool)
    assert iou(empty, empty) == 0.0


def test_load_ground_truth_zones_rasterizes_polygon(tmp_path: Path) -> None:
    """A simple block region is rasterized into the expected mask shape."""
    xml_text = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
          <Page imageFilename="page.jpg" imageWidth="10" imageHeight="10">
            <TextRegion id="r1" custom="structure {type:block;}">
              <Coords points="2,2 8,2 8,8 2,8"/>
            </TextRegion>
            <TextRegion id="r2_no_type" custom="readingDirection {direction:lr;}">
              <Coords points="0,0 1,0 1,1 0,1"/>
            </TextRegion>
          </Page>
        </PcGts>
    """)
    xml_file = tmp_path / "page.xml"
    xml_file.write_text(xml_text, encoding="utf-8")

    zones = load_ground_truth_zones(xml_file, image_shape=(10, 10))

    # The region without a structure type is dropped.
    assert len(zones) == 1
    assert zones[0]["zone_type"] == "block"
    # Roughly the centre of the polygon must be True.
    assert zones[0]["mask"][5, 5]
    # Corners outside the polygon are False.
    assert not zones[0]["mask"][0, 0]


def test_best_iou_per_zone_finds_matching_prediction() -> None:
    """The evaluator picks the predicted mask with the highest overlap."""
    gt_mask = np.zeros((10, 10), dtype=bool)
    gt_mask[2:8, 2:8] = True
    zones = [{"zone_type": "block", "mask": gt_mask}]

    far_mask = np.zeros((10, 10), dtype=bool)
    far_mask[8:10, 8:10] = True  # far from gt
    close_mask = gt_mask.copy()
    close_mask[7, 7] = False  # very close to gt

    results = best_iou_per_zone(zones, [far_mask, close_mask])
    assert results[0]["best_index"] == 1
    assert results[0]["best_iou"] > 0.9


def test_summarize_groups_by_zone_type() -> None:
    """Per-zone results are aggregated into per-type statistics."""
    per_zone = [
        {"zone_type": "block", "best_iou": 0.2, "best_index": 0},
        {"zone_type": "block", "best_iou": 0.4, "best_index": 1},
        {"zone_type": "liste", "best_iou": 0.1, "best_index": 0},
    ]
    summary = summarize_by_zone_type(per_zone)
    assert summary["block"]["n_zones"] == 2
    assert summary["block"]["mean_iou"] == 0.3
    assert summary["liste"]["n_zones"] == 1
