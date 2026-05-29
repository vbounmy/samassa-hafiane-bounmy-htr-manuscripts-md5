"""Evaluate Kraken BLLA line segmentation against e-NDP ground truth.

Runs Kraken BLLA (default segmentation model) on a few representative
pages of the stratified sample, rasterizes each detected line into a
boolean mask, and reports the IoU against the line polygons annotated
in e-NDP. The companion script of step 2.2's SAM baseline.

Example::

    python scripts/evaluate_kraken_lines.py \\
        --xml-dir <path>/HTR_Ground-Truth/page_xml \\
        --image-dir <path>/HTR_Ground-Truth/images \\
        --headers /tmp/doc_headers.txt \\
        --n-pages 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation.evaluator import (
    best_iou_per_zone,  # name kept generic, works on lines too
    load_ground_truth_lines,
    summarize_by_zone_type,
)
from src.segmentation.kraken_wrapper import lines_to_masks, segment_lines
from src.segmentation.sampler import build_folio_index, stratified_sample


def main() -> None:
    """Entry point: sample, segment with Kraken, evaluate, save report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--n-pages", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/kraken_lines_report.json"),
    )
    args = parser.parse_args()

    # 1. Pick a small slice of the stratified sample (different strata).
    folio_index = build_folio_index(args.headers)
    available = {p.stem for p in args.xml_dir.glob("*.xml")}
    sample = stratified_sample(folio_index, available, seed=42)
    seen: set[str] = set()
    selected: list[dict] = []
    for item in sample:
        if item["stratum"] in seen:
            continue
        selected.append(item)
        seen.add(item["stratum"])
        if len(selected) >= args.n_pages:
            break
    print(f"Pages selected: {[s['folio_id'] for s in selected]}\n")

    # 2. Run Kraken BLLA on each selected page.
    all_per_line: list[dict] = []
    per_page: list[dict] = []
    overall_start = time.time()

    for i, item in enumerate(selected, 1):
        folio = item["folio_id"]
        image_path = args.image_dir / f"{folio}.jpg"
        if not image_path.exists():
            image_path = args.image_dir / f"{folio}.jpeg"
        xml_path = args.xml_dir / f"{folio}.xml"

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"  [{i}/{len(selected)}] {folio}: cannot read image, skipping.")
            continue

        gt_lines = load_ground_truth_lines(xml_path, image_bgr.shape[:2])
        if not gt_lines:
            print(f"  [{i}/{len(selected)}] {folio}: no annotated lines, skipping.")
            continue

        t_page = time.time()
        kraken_lines = segment_lines(image_path)
        predicted_masks = lines_to_masks(kraken_lines, image_bgr.shape[:2])
        # Rename "zone_type" to a constant ("line") so we can reuse
        # best_iou_per_zone / summarize_by_zone_type unchanged.
        zones_as_lines = [{"zone_type": "line", "mask": gt["mask"]} for gt in gt_lines]
        results = best_iou_per_zone(zones_as_lines, predicted_masks)
        elapsed = time.time() - t_page

        mean_iou = sum(r["best_iou"] for r in results) / max(len(results), 1)
        page_summary = {
            "folio_id": folio,
            "stratum": item["stratum"],
            "year": item["year"],
            "n_ground_truth_lines": len(gt_lines),
            "n_kraken_lines": len(kraken_lines),
            "mean_iou": round(mean_iou, 3),
            "elapsed_seconds": round(elapsed, 1),
        }
        per_page.append(page_summary)
        all_per_line.extend(results)
        print(
            f"  [{i}/{len(selected)}] {folio} ({item['year']}, {item['stratum']}): "
            f"{len(gt_lines)} GT lines, {len(kraken_lines)} Kraken lines, "
            f"mean IoU = {mean_iou:.3f}, {elapsed:.0f}s"
        )

    # 3. Aggregate and save.
    overall_mean = (
        sum(item["best_iou"] for item in all_per_line) / max(len(all_per_line), 1)
    )
    report = {
        "n_pages_evaluated": len(per_page),
        "n_lines_evaluated": len(all_per_line),
        "overall_mean_iou": round(overall_mean, 3),
        "per_page": per_page,
        "total_elapsed_seconds": round(time.time() - overall_start, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Kraken BLLA line segmentation summary ===")
    print(f"Pages evaluated     : {len(per_page)}")
    print(f"Lines evaluated     : {len(all_per_line)}")
    print(f"Overall mean IoU    : {overall_mean:.3f}")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
