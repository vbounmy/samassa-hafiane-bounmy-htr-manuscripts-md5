"""Geometry validation utilities for PAGE XML line polygons."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Compute polygon area with the shoelace formula.

    Args:
        points (Sequence[Sequence[float]]): Polygon points.

    Returns:
        float: Absolute polygon area.
    """
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return abs(area) / 2.0


def bounding_box(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Return polygon bounding box.

    Args:
        points (Sequence[Sequence[float]]): Polygon points.

    Returns:
        Tuple[float, float, float, float]: xmin, ymin, xmax, ymax.
    """
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Compute bounding-box IoU for two polygons.

    Args:
        a (Sequence[Sequence[float]]): First polygon.
        b (Sequence[Sequence[float]]): Second polygon.

    Returns:
        float: Intersection over Union.
    """
    ax1, ay1, ax2, ay2 = bounding_box(a)
    bx1, by1, bx2, by2 = bounding_box(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def polygon_iou(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Compute polygon IoU, using Shapely when installed.

    Args:
        a (Sequence[Sequence[float]]): First polygon.
        b (Sequence[Sequence[float]]): Second polygon.

    Returns:
        float: Intersection over Union.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return bbox_iou(a, b)

    poly_a = Polygon(a)
    poly_b = Polygon(b)
    if not poly_a.is_valid or not poly_b.is_valid:
        return bbox_iou(a, b)
    union = poly_a.union(poly_b).area
    return poly_a.intersection(poly_b).area / union if union else 0.0


def polygon_in_bounds(points: Sequence[Sequence[float]], width: int, height: int) -> bool:
    """Check whether all polygon points are inside image bounds.

    Args:
        points (Sequence[Sequence[float]]): Polygon points.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        bool: True when all points are inside bounds.
    """
    return all(0 <= float(x) <= width and 0 <= float(y) <= height for x, y in points)


def validate_geometry_records(
    records: Iterable[Dict[str, Any]],
    image_sizes: Dict[str, Tuple[int, int]],
) -> Dict[str, Any]:
    """Validate polygons for final line records.

    Args:
        records (Iterable[Dict[str, Any]]): Records with page and polygon fields.
        image_sizes (Dict[str, Tuple[int, int]]): Mapping page -> (width, height).

    Returns:
        Dict[str, Any]: Validation summary.
    """
    rows = list(records)
    missing_polygon = 0
    out_of_bounds = 0
    zero_area = 0
    missing_size = 0

    for row in rows:
        polygon = row.get("polygon") or []
        page = row.get("page", "")
        if len(polygon) < 3:
            missing_polygon += 1
            continue
        if polygon_area(polygon) <= 0:
            zero_area += 1
        if page not in image_sizes:
            missing_size += 1
            continue
        width, height = image_sizes[page]
        if not polygon_in_bounds(polygon, width, height):
            out_of_bounds += 1

    total = len(rows)
    return {
        "num_lines": total,
        "missing_polygon": missing_polygon,
        "zero_area": zero_area,
        "out_of_bounds": out_of_bounds,
        "missing_image_size": missing_size,
        "valid_geometry_rate": 1.0 - ((missing_polygon + zero_area + out_of_bounds) / total if total else 0.0),
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSON Lines records.

    Args:
        path (str): JSONL file path.

    Returns:
        List[Dict[str, Any]]: Parsed records.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_image_sizes(images_dir: str) -> Dict[str, Tuple[int, int]]:
    """Load image sizes from a directory.

    Args:
        images_dir (str): Directory containing page images.

    Returns:
        Dict[str, Tuple[int, int]]: Mapping filename -> (width, height).
    """
    from PIL import Image

    sizes: Dict[str, Tuple[int, int]] = {}
    for name in os.listdir(images_dir):
        path = os.path.join(images_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as image:
                sizes[name] = image.size
        except Exception:
            continue
    return sizes


def main() -> None:
    """Validate final geometry from the command line."""
    parser = argparse.ArgumentParser(description="Validate final line polygons.")
    parser.add_argument("--records", required=True, help="Aggregated JSONL records.")
    parser.add_argument("--images_dir", required=True, help="Directory containing source page images.")
    parser.add_argument("--output", default="experiments/geometry_report.json", help="Report JSON path.")
    args = parser.parse_args()

    report = validate_geometry_records(load_jsonl(args.records), load_image_sizes(args.images_dir))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
