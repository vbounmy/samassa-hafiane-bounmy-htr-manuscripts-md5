"""Thin wrapper around Kraken's Baseline Layout Analysis (BLLA).

Kraken's segmentation entry point ``kraken.blla.segment`` produces a
:class:`Segmentation` dataclass holding a list of :class:`BaselineLine`
objects. Each line carries:

- ``baseline``: two-point list defining the writing baseline.
- ``boundary``: closed polygon delimiting the line region.

This module exposes :func:`segment_lines`, a function that returns a
list of ``{baseline, polygon}`` dicts in a method-agnostic format. The
output is directly compatible with our :mod:`src.segmentation.evaluator`
IoU implementation, which is why we keep the dict structure flat and
plain rather than reusing Kraken's internal classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from kraken import blla
from PIL import Image


def segment_lines(image_path: Path) -> list[dict]:
    """Run Kraken BLLA on one page and return its detected lines.

    Args:
        image_path: Path to the page image (JPEG, PNG, or any format
            supported by Pillow).

    Returns:
        A list of dicts, one per detected line. Each dict contains:

        - ``baseline``: ``list[tuple[int, int]]`` of two points.
        - ``polygon``: ``list[tuple[int, int]]`` of N points
          delimiting the line region.
        - ``tags``: the original Kraken ``tags`` dict (useful for the
          region the line was attached to).

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
    """
    with Image.open(image_path) as img:
        img.load()  # decode pixels while the file handle is open
        result = blla.segment(img)

    lines: list[dict] = []
    for line in result.lines:
        lines.append({
            "baseline": [tuple(p) for p in line.baseline],
            "polygon": [tuple(p) for p in line.boundary],
            "tags": dict(line.tags) if line.tags else {},
        })
    return lines


def lines_to_masks(
    lines: list[dict],
    image_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Rasterize a list of Kraken lines into boolean masks.

    Args:
        lines: Output of :func:`segment_lines`.
        image_shape: ``(height, width)`` of the source image.

    Returns:
        List of ``(H, W)`` boolean arrays, one per line. ``True`` inside
        the line's polygon.
    """
    import cv2  # imported locally so the module stays importable without cv2

    height, width = image_shape
    masks: list[np.ndarray] = []
    for line in lines:
        polygon = line["polygon"]
        if len(polygon) < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], color=1)
        masks.append(mask.astype(bool))
    return masks
