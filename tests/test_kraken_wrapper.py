"""Unit tests for ``src.segmentation.kraken_wrapper`` (Étape 2.3).

Tests focus on the rasterization helper, which is independent of Kraken's
heavy initialization. The Kraken inference path itself is exercised by
``scripts/evaluate_kraken_lines.py`` against real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation.kraken_wrapper import lines_to_masks


def test_lines_to_masks_rasterizes_polygon() -> None:
    """A 4-point polygon yields exactly one boolean mask of the right shape."""
    lines = [{
        "baseline": [(2, 5), (8, 5)],
        "polygon": [(2, 2), (8, 2), (8, 8), (2, 8)],
        "tags": {},
    }]
    masks = lines_to_masks(lines, image_shape=(10, 10))

    assert len(masks) == 1
    assert masks[0].shape == (10, 10)
    assert masks[0].dtype == bool
    # The polygon centre must be True; outside corners must be False.
    assert masks[0][5, 5]
    assert not masks[0][0, 0]


def test_lines_to_masks_skips_degenerate_polygons() -> None:
    """Lines whose polygon has fewer than 3 vertices are dropped."""
    lines = [
        {"baseline": [], "polygon": [(0, 0), (1, 1)], "tags": {}},          # too few
        {"baseline": [], "polygon": [(0, 0), (1, 1), (1, 0)], "tags": {}},  # OK
    ]
    masks = lines_to_masks(lines, image_shape=(5, 5))
    assert len(masks) == 1


def test_lines_to_masks_returns_empty_list_for_empty_input() -> None:
    """An empty input list produces an empty output list."""
    assert lines_to_masks([], image_shape=(10, 10)) == []
