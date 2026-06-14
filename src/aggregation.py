"""Aggregate HTR predictions into final line-level records."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSON Lines records.

    Args:
        path (str): JSONL file path.

    Returns:
        List[Dict[str, Any]]: Parsed records.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_key(record: Dict[str, Any]) -> str:
    """Return the stable line key used across metadata and prediction files.

    Args:
        record (Dict[str, Any]): Line metadata or prediction record.

    Returns:
        str: Stable key.
    """
    return str(record.get("file_name") or f"{record.get('page')}::{record.get('line_id')}")


def align_pair(reference: str, hypothesis: str, gap: str = "") -> Tuple[List[str], List[str]]:
    """Needleman-Wunsch alignment for two character strings.

    Args:
        reference (str): First string.
        hypothesis (str): Second string.
        gap (str): Gap symbol in the returned alignment.

    Returns:
        Tuple[List[str], List[str]]: Aligned character sequences.
    """
    ref = list(reference)
    hyp = list(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    scores = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        scores[i][0] = i
    for j in range(1, cols):
        scores[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution = scores[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            deletion = scores[i - 1][j] + 1
            insertion = scores[i][j - 1] + 1
            scores[i][j] = min(substitution, deletion, insertion)

    aligned_ref: List[str] = []
    aligned_hyp: List[str] = []
    i = len(ref)
    j = len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and scores[i][j] == scores[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and scores[i][j] == scores[i - 1][j] + 1:
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(gap)
            i -= 1
        else:
            aligned_ref.append(gap)
            aligned_hyp.append(hyp[j - 1])
            j -= 1

    return list(reversed(aligned_ref)), list(reversed(aligned_hyp))


def weighted_vote(predictions: Sequence[Dict[str, Any]]) -> Tuple[str, float, List[str]]:
    """Aggregate one line with confidence-weighted voting.

    With one prediction, the function returns it unchanged. With two or more
    predictions, all hypotheses are aligned to the highest-confidence prediction
    and a character-level weighted vote is performed.

    Args:
        predictions (Sequence[Dict[str, Any]]): Prediction records for one line.

    Returns:
        Tuple[str, float, List[str]]: Aggregated text, confidence, source model names.
    """
    if not predictions:
        return "", 0.0, []

    ordered = sorted(predictions, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    anchor = str(ordered[0].get("prediction", ""))
    sources = [str(item.get("model", "unknown")) for item in ordered]

    if len(ordered) == 1:
        return anchor, float(ordered[0].get("confidence", 0.0)), sources

    aligned_columns: List[List[str]] = [[char] for char in anchor]
    num_aligned_models = 1
    for prediction in ordered[1:]:
        text = str(prediction.get("prediction", ""))
        aligned_anchor, aligned_text = align_pair(anchor, text)
        new_columns: List[List[str]] = []
        anchor_index = 0
        for anchor_char, text_char in zip(aligned_anchor, aligned_text):
            if anchor_char:
                column = list(aligned_columns[anchor_index])
                column.append(text_char)
                new_columns.append(column)
                anchor_index += 1
            else:
                new_columns.append([""] * num_aligned_models + [text_char])
        while anchor_index < len(aligned_columns):
            column = list(aligned_columns[anchor_index])
            column.append("")
            new_columns.append(column)
            anchor_index += 1
        aligned_columns = new_columns
        num_aligned_models += 1

    chars: List[str] = []
    confidences = [float(item.get("confidence", 0.0)) for item in ordered]
    for column in aligned_columns:
        weights: Dict[str, float] = defaultdict(float)
        for char, confidence in zip(column, confidences):
            weights[char] += confidence
        winner = max(weights.items(), key=lambda item: item[1])[0]
        if winner:
            chars.append(winner)

    agreement = sum(confidences) / max(1, len(confidences))
    return "".join(chars), max(0.0, min(1.0, agreement)), sources


def aggregate_records(
    metadata_records: Sequence[Dict[str, Any]],
    prediction_records: Sequence[Dict[str, Any]],
    confidence_threshold: float = 0.70,
    short_line_threshold: int = 3,
) -> List[Dict[str, Any]]:
    """Merge metadata and model predictions into final line-level records.

    Args:
        metadata_records (Sequence[Dict[str, Any]]): Records from `metadata.jsonl`.
        prediction_records (Sequence[Dict[str, Any]]): Model prediction records.
        confidence_threshold (float): Minimum confidence before `needs_review`.
        short_line_threshold (int): Lines at or below this length are flagged.

    Returns:
        List[Dict[str, Any]]: Aggregated records ready for evaluation/data contract.
    """
    predictions_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for prediction in prediction_records:
        predictions_by_key[record_key(prediction)].append(prediction)

    output: List[Dict[str, Any]] = []
    for metadata in metadata_records:
        key = record_key(metadata)
        predictions = predictions_by_key.get(key, [])
        text, confidence, source_models = weighted_vote(predictions)

        flags = []
        if not predictions:
            flags.append("missing_prediction")
        if confidence < confidence_threshold:
            flags.append("low_confidence")
        if len(text.strip()) <= short_line_threshold:
            flags.append("short_line")
        if metadata.get("needs_review"):
            flags.append("source_needs_review")

        output.append(
            {
                "file_name": metadata.get("file_name", ""),
                "page": metadata.get("page", ""),
                "line_id": metadata.get("line_id", ""),
                "reference": metadata.get("text", ""),
                "prediction": text,
                "confidence": confidence,
                "needs_review": bool(flags),
                "flags": flags,
                "polygon": metadata.get("polygon", []),
                "source_models": source_models,
            }
        )
    return output


def parse_prediction_sources(values: Sequence[str]) -> List[Dict[str, Any]]:
    """Load prediction JSONL files declared as `model=path`.

    Args:
        values (Sequence[str]): CLI values such as `trocr=predictions/trocr.jsonl`.

    Returns:
        List[Dict[str, Any]]: Prediction records with a `model` field.
    """
    records: List[Dict[str, Any]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("prediction sources must use the form model=path")
        model_name, path = value.split("=", 1)
        for record in load_jsonl(path):
            item = dict(record)
            item.setdefault("model", model_name)
            records.append(item)
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Write JSON Lines records.

    Args:
        path (str): Output path.
        records (Iterable[Dict[str, Any]]): Records to write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Aggregate prediction files from the command line."""
    parser = argparse.ArgumentParser(description="Aggregate HTR predictions with weighted voting.")
    parser.add_argument("--metadata", required=True, help="Line metadata.jsonl from processed_lines split.")
    parser.add_argument("--prediction", action="append", default=[], help="Prediction source model=path.")
    parser.add_argument("--output", default="experiments/aggregated_predictions.jsonl", help="Output JSONL path.")
    parser.add_argument("--confidence_threshold", type=float, default=0.70, help="needs_review confidence threshold.")
    args = parser.parse_args()

    records = aggregate_records(
        metadata_records=load_jsonl(args.metadata),
        prediction_records=parse_prediction_sources(args.prediction),
        confidence_threshold=args.confidence_threshold,
    )
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} aggregated records to {args.output}")


if __name__ == "__main__":
    main()
