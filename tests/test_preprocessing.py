"""Unit tests for ``src.preprocessing.pipeline`` (Étape 2.1).

Tests use synthetic NumPy arrays so they remain fast and deterministic
and do not require the e-NDP corpus to be downloaded locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.pipeline import (
    apply_clahe,
    binarize_sauvola,
    deskew,
    estimate_skew_angle,
    preprocess_page,
    to_grayscale,
)


def _make_page(size: int = 256) -> np.ndarray:
    """Build a synthetic grayscale page with simulated ink strokes."""
    rng = np.random.default_rng(42)
    page = rng.integers(200, 230, size=(size, size), dtype=np.uint8)
    for y in range(40, size - 40, 20):
        page[y:y + 4, 20:size - 20] = rng.integers(30, 70, size=(4, size - 40))
    return page


def test_to_grayscale_returns_same_shape_when_already_gray() -> None:
    """Grayscale input is passed through untouched."""
    page = _make_page()
    assert to_grayscale(page).shape == page.shape


def test_to_grayscale_collapses_rgb_to_one_channel() -> None:
    """RGB input is reduced to a single channel."""
    page = _make_page()
    rgb = np.stack([page, page, page], axis=-1)
    out = to_grayscale(rgb)
    assert out.ndim == 2 and out.shape == page.shape


def test_estimate_skew_angle_returns_float_for_clean_image() -> None:
    """The estimator yields a finite angle on a non-empty image."""
    page = _make_page()
    angle = estimate_skew_angle(page)
    assert isinstance(angle, float)
    assert -10.0 <= angle <= 10.0


def test_deskew_preserves_shape_and_dtype() -> None:
    """Output shape and dtype match the input."""
    page = _make_page()
    out = deskew(page)
    assert out.shape == page.shape
    assert out.dtype == page.dtype


def test_apply_clahe_returns_valid_uint8() -> None:
    """CLAHE keeps the uint8 dtype and value range."""
    page = _make_page()
    out = apply_clahe(page)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_binarize_sauvola_produces_two_values_only() -> None:
    """Sauvola binarization yields strictly two pixel values: 0 and 255."""
    page = _make_page()
    out = binarize_sauvola(page)
    unique = set(np.unique(out).tolist())
    assert unique.issubset({0, 255})


def test_binarize_sauvola_forces_odd_window() -> None:
    """An even window size is silently incremented to the next odd integer."""
    page = _make_page()
    out_even = binarize_sauvola(page, window_size=24)
    out_odd = binarize_sauvola(page, window_size=25)
    assert np.array_equal(out_even, out_odd)


def test_preprocess_page_returns_binary_uint8() -> None:
    """The full pipeline produces a clean binary uint8 image."""
    page = _make_page(size=128)
    out = preprocess_page(page)
    assert out.dtype == np.uint8
    assert out.shape == page.shape
    assert set(np.unique(out).tolist()).issubset({0, 255})


def test_preprocess_page_accepts_rgb_input() -> None:
    """A 3-channel page is accepted and produces a 2-D binary output."""
    page = _make_page(size=128)
    rgb = np.stack([page, page, page], axis=-1)
    out = preprocess_page(rgb)
    assert out.ndim == 2


def test_preprocess_page_deskew_can_be_disabled() -> None:
    """The deskew step can be turned off for ablation studies."""
    page = _make_page(size=128)
    with_deskew = preprocess_page(page, do_deskew=True)
    without_deskew = preprocess_page(page, do_deskew=False)
    for variant in (with_deskew, without_deskew):
        assert set(np.unique(variant).tolist()).issubset({0, 255})
