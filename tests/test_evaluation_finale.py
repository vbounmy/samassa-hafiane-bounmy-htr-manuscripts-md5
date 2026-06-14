"""Tests for final aggregation, evaluation, geometry, and data contract."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aggregation import aggregate_records, align_pair, weighted_vote
from src.calibration import binning_calibration
from src.data_contract import build_dataset, validate_dataset
from src.evaluation import bootstrap_cer_ci, corpus_error_rates, evaluate_records, levenshtein, mcnemar_chi2, mcnemar_counts
from src.validate_geometry import polygon_area, polygon_iou, validate_geometry_records


def test_levenshtein_and_corpus_error_rates() -> None:
    """Check CER/WER calculations on simple strings."""
    assert levenshtein("abc", "adc") == 1

    rates = corpus_error_rates(["abc", "hello world"], ["adc", "hello"])

    assert rates.char_distance == 7
    assert rates.char_total == 14
    assert rates.cer == 0.5
    assert rates.word_distance == 2
    assert rates.word_total == 3


def test_bootstrap_ci_is_reproducible() -> None:
    """Check deterministic bootstrap interval generation."""
    first = bootstrap_cer_ci(["abc", "def"], ["abc", "xef"], n_samples=20, seed=7)
    second = bootstrap_cer_ci(["abc", "def"], ["abc", "xef"], n_samples=20, seed=7)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_mcnemar_counts_and_statistic() -> None:
    """Check paired model comparison counts."""
    counts = mcnemar_counts(["a", "b", "c"], ["a", "x", "c"], ["z", "b", "c"])

    assert counts == {"both_correct": 1, "both_wrong": 0, "a_only": 1, "b_only": 1}
    assert mcnemar_chi2(counts) == 0.5


def test_weighted_vote_and_aggregation_flags() -> None:
    """Check weighted aggregation and needs_review flags."""
    assert align_pair("abc", "axbc") == (["a", "", "b", "c"], ["a", "x", "b", "c"])

    text, confidence, sources = weighted_vote(
        [
            {"prediction": "dominus", "confidence": 0.9, "model": "trocr"},
            {"prediction": "dominos", "confidence": 0.6, "model": "kraken"},
        ]
    )

    assert text == "dominus"
    assert confidence == 0.75
    assert sources == ["trocr", "kraken"]

    records = aggregate_records(
        metadata_records=[
            {"file_name": "l1.png", "page": "p.jpg", "line_id": "l1", "text": "dominus", "polygon": [[0, 0], [4, 0], [4, 2], [0, 2]]},
            {"file_name": "l2.png", "page": "p.jpg", "line_id": "l2", "text": "x", "polygon": [[0, 3], [4, 3], [4, 5], [0, 5]]},
        ],
        prediction_records=[{"file_name": "l1.png", "prediction": "dominus", "confidence": 0.95, "model": "trocr"}],
    )

    assert records[0]["needs_review"] is False
    assert records[1]["needs_review"] is True
    assert "missing_prediction" in records[1]["flags"]


def test_geometry_validation_and_iou() -> None:
    """Check polygon geometry helpers."""
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]

    assert polygon_area(polygon) == 100
    assert polygon_iou(polygon, polygon) == 1.0

    report = validate_geometry_records(
        [{"page": "p.jpg", "polygon": polygon}, {"page": "p.jpg", "polygon": [[-1, 0], [1, 0], [1, 1]]}],
        {"p.jpg": (20, 20)},
    )

    assert report["num_lines"] == 2
    assert report["out_of_bounds"] == 1


def test_data_contract_validation() -> None:
    """Check final NLP dataset construction and schema validation."""
    dataset = build_dataset(
        [
            {
                "page": "p.jpg",
                "line_id": "l1",
                "prediction": "dominus",
                "reference": "dominus",
                "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
                "confidence": 0.95,
                "needs_review": False,
                "source_models": ["trocr"],
                "flags": [],
            }
        ]
    )

    validate_dataset(dataset)
    assert dataset["documents"][0]["lines"][0]["text"] == "dominus"


def test_step4_pipeline_integration() -> None:
    """Check final step 4 integration from aggregation to dataset build."""
    metadata = [
        {
            "file_name": "l1.png",
            "page": "p.jpg",
            "line_id": "l1",
            "text": "dominus",
            "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "needs_review": False,
        },
        {
            "file_name": "l2.png",
            "page": "p.jpg",
            "line_id": "l2",
            "text": "hello",
            "polygon": [[0, 6], [10, 6], [10, 10], [0, 10]],
            "needs_review": True,
        },
    ]
    predictions = [
        {"file_name": "l1.png", "prediction": "dominus", "confidence": 0.9, "model": "trocr"},
        {"file_name": "l2.png", "prediction": "hello", "confidence": 0.4, "model": "kraken"},
    ]

    aggregated = aggregate_records(metadata, predictions, confidence_threshold=0.5)
    calibrated = binning_calibration(aggregated, n_bins=2, smoothing=1.0)
    dataset = build_dataset(calibrated)

    validate_dataset(dataset)
    metrics = evaluate_records(calibrated)

    assert all("confidence_calibrated" in rec for rec in calibrated)
    assert metrics["cer"] == 0.0
    assert dataset["documents"][0]["lines"][1]["needs_review"] is True
