"""Evaluate predicted page-segmentation masks against e-NDP ground truth.

For each ground-truth zone defined in the PAGE XML, we look up the
predicted mask whose IoU with that zone is highest and report it. Aggregated
over the sample, this gives a single score per zone type (``block``,
``liste``, ``Date`` …) that quantifies how well an automatic segmenter
recovers e-NDP's logical structure.

The evaluator is **method-agnostic**: it accepts any list of predicted
boolean masks, regardless of how they were produced (SAM, dhSegment,
Kraken, OpenCV, …). It is therefore reusable for every segmentation
experiment in the project.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

_PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_NS = {"p": _PAGE_NS}
_STRUCTURE_TYPE_RE = re.compile(r"structure\s*\{[^}]*type:\s*([^;}]+)")


def _parse_points(points_str: str) -> list[tuple[int, int]]:
    """Parse a PAGE XML ``points`` attribute into a list of (x, y) pairs."""
    pairs: list[tuple[int, int]] = []
    for token in (points_str or "").split():
        x_str, y_str = token.split(",")
        pairs.append((int(float(x_str)), int(float(y_str))))
    return pairs


def _extract_zone_type(custom: str) -> str | None:
    """Lift the structure type out of a PAGE XML ``custom`` attribute."""
    match = _STRUCTURE_TYPE_RE.search(custom or "")
    return match.group(1).strip() if match else None


def load_ground_truth_zones(
    xml_path: Path,
    image_shape: tuple[int, int],
) -> list[dict]:
    """Rasterize every annotated zone of a PAGE XML page into a boolean mask.

    Args:
        xml_path: Path to the PAGE XML file describing one e-NDP page.
        image_shape: ``(height, width)`` of the source image, used to
            initialize the masks at the correct resolution.

    Returns:
        List of dicts ``{"zone_type", "mask"}``, one per region whose
        ``custom`` attribute declares a structure type. ``mask`` is a
        boolean ``(H, W)`` array, True inside the polygon.
    """
    height, width = image_shape
    root = ET.parse(xml_path).getroot()

    zones: list[dict] = []
    for region in root.iter(f"{{{_PAGE_NS}}}TextRegion"):
        zone_type = _extract_zone_type(region.get("custom", ""))
        if zone_type is None:
            continue
        coords_el = region.find("p:Coords", _NS)
        if coords_el is None:
            continue
        polygon = _parse_points(coords_el.get("points", ""))
        if len(polygon) < 3:
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], color=1)
        zones.append({
            "zone_type": zone_type,
            "mask": mask.astype(bool),
        })
    return zones


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union between two boolean masks.

    Args:
        mask_a: ``(H, W)`` boolean array.
        mask_b: ``(H, W)`` boolean array of the same shape.

    Returns:
        IoU in [0, 1]. Returns 0.0 when both masks are entirely empty
        (degenerate case: there is nothing to match).
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def best_iou_per_zone(
    zones: list[dict],
    predicted_masks: list[np.ndarray],
) -> list[dict]:
    """For each ground-truth zone, find the predicted mask with highest IoU.

    Args:
        zones: Output of :func:`load_ground_truth_zones`.
        predicted_masks: List of boolean ``(H, W)`` arrays. Their shape
            must match the masks inside ``zones``.

    Returns:
        For each zone, a dict ``{"zone_type", "best_iou", "best_index"}``.
        ``best_index`` is ``None`` when ``predicted_masks`` is empty.
    """
    results: list[dict] = []
    for zone in zones:
        gt_mask = zone["mask"]
        best_iou_val = 0.0
        best_index: int | None = None
        for i, pred_mask in enumerate(predicted_masks):
            score = iou(gt_mask, pred_mask)
            if score > best_iou_val:
                best_iou_val = score
                best_index = i
        results.append({
            "zone_type": zone["zone_type"],
            "best_iou": best_iou_val,
            "best_index": best_index,
        })
    return results


def summarize_by_zone_type(per_zone_results: list[dict]) -> dict[str, dict]:
    """Aggregate per-zone IoU scores into per-zone-type statistics.

    Args:
        per_zone_results: Concatenation of outputs from
            :func:`best_iou_per_zone` across all evaluated pages.

    Returns:
        Mapping ``zone_type -> {"n_zones", "mean_iou", "median_iou"}``.
    """
    grouped: dict[str, list[float]] = {}
    for item in per_zone_results:
        grouped.setdefault(item["zone_type"], []).append(item["best_iou"])

    summary: dict[str, dict] = {}
    for zone_type, scores in grouped.items():
        arr = np.array(scores)
        summary[zone_type] = {
            "n_zones": len(scores),
            "mean_iou": round(float(arr.mean()), 3),
            "median_iou": round(float(np.median(arr)), 3),
        }
    return summary
