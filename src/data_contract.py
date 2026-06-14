"""Data contract for the final NLP-ready HTR transcription dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List


TRANSCRIPTION_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HTR manuscript transcription dataset",
    "type": "object",
    "required": ["metadata", "documents"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["dataset_name", "created_at", "coordinate_system"],
            "properties": {
                "dataset_name": {"type": "string"},
                "created_at": {"type": "string"},
                "source": {"type": "string"},
                "coordinate_system": {
                    "type": "object",
                    "required": ["origin", "unit"],
                    "properties": {
                        "origin": {"const": "top-left"},
                        "unit": {"const": "pixel"},
                    },
                },
            },
        },
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["page", "lines"],
                "properties": {
                    "page": {"type": "string"},
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "line_id",
                                "text",
                                "polygon",
                                "confidence",
                                "needs_review",
                                "source_models",
                            ],
                            "properties": {
                                "line_id": {"type": "string"},
                                "text": {"type": "string"},
                                "reference": {"type": "string"},
                                "polygon": {
                                    "type": "array",
                                    "minItems": 3,
                                    "items": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {"type": "number"},
                                    },
                                },
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "needs_review": {"type": "boolean"},
                                "source_models": {"type": "array", "items": {"type": "string"}},
                                "flags": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        },
    },
}


def build_dataset(records: Iterable[Dict[str, Any]], dataset_name: str = "anr-e-ndp-htr") -> Dict[str, Any]:
    """Build a page-grouped JSON dataset from line-level records.

    Args:
        records (Iterable[Dict[str, Any]]): Aggregated line records.
        dataset_name (str): Name included in dataset metadata.

    Returns:
        Dict[str, Any]: Dataset following :data:`TRANSCRIPTION_SCHEMA`.
    """
    pages: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        page = record["page"]
        pages.setdefault(page, []).append(
            {
                "line_id": record["line_id"],
                "text": record["prediction"],
                "reference": record.get("reference", ""),
                "polygon": record["polygon"],
                "confidence": float(record.get("confidence", 0.0)),
                "needs_review": bool(record.get("needs_review", False)),
                "source_models": list(record.get("source_models", [])),
                "flags": list(record.get("flags", [])),
            }
        )

    return {
        "metadata": {
            "dataset_name": dataset_name,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source": "ANR e-NDP Ground Truth",
            "coordinate_system": {"origin": "top-left", "unit": "pixel"},
        },
        "documents": [{"page": page, "lines": lines} for page, lines in sorted(pages.items())],
    }


def validate_dataset(dataset: Dict[str, Any]) -> None:
    """Validate a dataset against the JSON schema.

    Args:
        dataset (Dict[str, Any]): Dataset to validate.

    Raises:
        jsonschema.ValidationError: If jsonschema is installed and validation fails.
        ValueError: If required top-level keys are missing.
    """
    try:
        import jsonschema
    except ImportError:
        if "metadata" not in dataset or "documents" not in dataset:
            raise ValueError("dataset must contain metadata and documents")
        return

    jsonschema.validate(instance=dataset, schema=TRANSCRIPTION_SCHEMA)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSON Lines records.

    Args:
        path (str): JSONL file path.

    Returns:
        List[Dict[str, Any]]: Parsed records.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    """Build and validate `dataset_nlp/transcriptions.json`."""
    parser = argparse.ArgumentParser(description="Build final NLP-ready transcription JSON.")
    parser.add_argument("--records", required=True, help="Aggregated JSONL records.")
    parser.add_argument("--output", default="dataset_nlp/transcriptions.json", help="Output JSON path.")
    parser.add_argument("--dataset_name", default="anr-e-ndp-htr", help="Dataset name.")
    args = parser.parse_args()

    dataset = build_dataset(load_jsonl(args.records), dataset_name=args.dataset_name)
    validate_dataset(dataset)

    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
