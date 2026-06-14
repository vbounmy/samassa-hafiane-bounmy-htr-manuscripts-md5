"""Tests for confidence calibration module."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.calibration import binning_calibration


def test_binning_calibration_basic() -> None:
    records = [
        {"prediction": "a", "reference": "a", "confidence": 0.9},
        {"prediction": "b", "reference": "x", "confidence": 0.85},
        {"prediction": "c", "reference": "c", "confidence": 0.4},
        {"prediction": "d", "reference": "y", "confidence": 0.1},
    ]

    calibrated = binning_calibration(records, n_bins=4, smoothing=1.0)
    # should add confidence_calibrated field
    assert all("confidence_calibrated" in r for r in calibrated)
    # confident correct prediction should have higher calibrated score than low-confident wrong
    high = max(r["confidence_calibrated"] for r in calibrated if r["confidence"] >= 0.8)
    low = min(r["confidence_calibrated"] for r in calibrated if r["confidence"] <= 0.2)
    assert high >= low
