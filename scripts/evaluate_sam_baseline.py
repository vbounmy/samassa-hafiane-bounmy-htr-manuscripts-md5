"""Quantitative SAM baseline evaluation on the stratified e-NDP sample.

Runs SAM ViT-B (automatic mask generator) on a few representative pages,
compares the predicted masks to the ground-truth zones encoded in PAGE
XML, and reports the IoU per zone type.

This script materializes the "baseline that fails" experiment of step 2.2
of the project brief. Its expected outcome is a low average IoU, which
motivates our switch to Kraken in step 2.3.

Example::

    python scripts/evaluate_sam_baseline.py \\
        --xml-dir <path>/HTR_Ground-Truth/page_xml \\
        --image-dir <path>/HTR_Ground-Truth/images \\
        --headers /tmp/doc_headers.txt \\
        --sam-checkpoint <path>/sam_vit_b_01ec64.pth \\
        --n-pages 5
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
    best_iou_per_zone,
    load_ground_truth_zones,
    summarize_by_zone_type,
)
from src.segmentation.sam_wrapper import SamSegmenter
from src.segmentation.sampler import build_folio_index, stratified_sample


def main() -> None:
    """Entry point: sample, segment, evaluate, write report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--headers", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--n-pages", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sam_baseline_report.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # 1. Pick a small slice of the stratified sample (different strata).
    folio_index = build_folio_index(args.headers)
    available = {p.stem for p in args.xml_dir.glob("*.xml")}
    sample = stratified_sample(folio_index, available, seed=42)
    # Take one page per stratum until we hit n_pages.
    seen_strata: set[str] = set()
    selected: list[dict] = []
    for item in sample:
        if item["stratum"] in seen_strata:
            continue
        selected.append(item)
        seen_strata.add(item["stratum"])
        if len(selected) >= args.n_pages:
            break
    print(f"Pages selected: {[s['folio_id'] for s in selected]}")

    # 2. Load SAM once.
    print(f"\nLoading SAM ViT-B on {args.device} ...")
    t0 = time.time()
    segmenter = SamSegmenter(
        args.sam_checkpoint,
        model_type="vit_b",
        device=args.device,
        downscale=4,
        points_per_side=16,
    )
    print(f"  -> loaded in {time.time() - t0:.1f}s")

    # 3. Iterate over selected pages.
    all_per_zone: list[dict] = []
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

        zones = load_ground_truth_zones(xml_path, image_bgr.shape[:2])
        if not zones:
            print(f"  [{i}/{len(selected)}] {folio}: no annotated zones, skipping.")
            continue

        t_page = time.time()
        masks = segmenter.segment(image_bgr)
        predicted = [m["segmentation"] for m in masks]
        results = best_iou_per_zone(zones, predicted)
        elapsed = time.time() - t_page

        page_summary = {
            "folio_id": folio,
            "stratum": item["stratum"],
            "year": item["year"],
            "n_zones": len(zones),
            "n_predicted_masks": len(predicted),
            "results_per_zone": results,
            "elapsed_seconds": round(elapsed, 1),
        }
        per_page.append(page_summary)
        all_per_zone.extend(results)
        mean_iou = sum(r["best_iou"] for r in results) / max(len(results), 1)
        print(
            f"  [{i}/{len(selected)}] {folio} ({item['year']}, {item['stratum']}): "
            f"{len(zones)} zones, {len(predicted)} predicted, "
            f"mean IoU = {mean_iou:.3f}, took {elapsed:.0f}s"
        )

    # 4. Aggregate and persist.
    by_zone_type = summarize_by_zone_type(all_per_zone)
    overall_mean = (
        sum(item["best_iou"] for item in all_per_zone) / max(len(all_per_zone), 1)
    )
    report = {
        "n_pages_evaluated": len(per_page),
        "n_zones_evaluated": len(all_per_zone),
        "overall_mean_iou": round(overall_mean, 3),
        "by_zone_type": by_zone_type,
        "per_page": per_page,
        "total_elapsed_seconds": round(time.time() - overall_start, 1),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SAM ViT-B baseline summary ===")
    print(f"Pages evaluated     : {len(per_page)}")
    print(f"Zones evaluated     : {len(all_per_zone)}")
    print(f"Overall mean IoU    : {overall_mean:.3f}")
    for zt, stats in by_zone_type.items():
        print(f"  - {zt:<15}  n={stats['n_zones']:>3}  "
              f"mean IoU={stats['mean_iou']:.3f}  median={stats['median_iou']:.3f}")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
