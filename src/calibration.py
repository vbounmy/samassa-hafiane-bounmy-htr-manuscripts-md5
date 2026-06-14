"""Confidence calibration utilities for aggregated HTR predictions.

This module implements a simple histogram-binning calibration with Laplace
smoothing as a fallback when sklearn is not available. It annotates records
with `confidence_calibrated` and can write an updated JSONL file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def binning_calibration(records: Iterable[Dict[str, Any]], n_bins: int = 10, smoothing: float = 1.0) -> List[Dict[str, Any]]:
    """Calibrate confidences by histogram binning with Laplace smoothing.

    Args:
        records: Iterable of aggregated line records containing `confidence`,
            `prediction` and `reference` fields.
        n_bins: Number of equal-width bins in [0,1].
        smoothing: Laplace smoothing parameter (alpha).

    Returns:
        List of records with `confidence_calibrated` added.
    """
    rows = list(records)
    if not rows:
        return []

    # initialize bins
    bins = [0 for _ in range(n_bins)]
    correct = [0 for _ in range(n_bins)]

    def bin_index(p: float) -> int:
        if p >= 1.0:
            return n_bins - 1
        idx = int(math.floor(p * n_bins))
        return max(0, min(n_bins - 1, idx))

    total_correct = 0
    total = 0
    for row in rows:
        p = float(row.get("confidence", 0.0))
        idx = bin_index(p)
        bins[idx] += 1
        is_correct = (row.get("prediction", "") == row.get("reference", ""))
        if is_correct:
            correct[idx] += 1
            total_correct += 1
        total += 1

    prior = total_correct / total if total else 0.0

    calibrated_bin: List[float] = []
    for k, (b, c) in enumerate(zip(bins, correct)):
        if b == 0:
            calibrated_bin.append(prior)
        else:
            calibrated_bin.append((c + smoothing * prior) / (b + smoothing))

    out: List[Dict[str, Any]] = []
    for row in rows:
        p = float(row.get("confidence", 0.0))
        idx = bin_index(p)
        row = dict(row)
        row["confidence_calibrated"] = float(calibrated_bin[idx])
        out.append(row)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate aggregated prediction confidences.")
    parser.add_argument("--input", required=True, help="Aggregated JSONL records path.")
    parser.add_argument("--output", default="experiments/aggregated_calibrated.jsonl", help="Output JSONL path.")
    parser.add_argument("--bins", type=int, default=10, help="Number of histogram bins.")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace smoothing alpha.")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    calibrated = binning_calibration(records, n_bins=args.bins, smoothing=args.smoothing)
    write_jsonl(args.output, calibrated)
    print(f"Wrote {len(calibrated)} calibrated records to {args.output}")


if __name__ == "__main__":
    main()
