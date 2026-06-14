"""Evaluation utilities for final HTR experiments.

This module contains dependency-light metrics used in step 4: global CER/WER,
bootstrap confidence intervals, and McNemar comparison for two model variants.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ErrorRates:
    """Container for corpus-level HTR error rates.

    Args:
        cer (float): Character error rate.
        wer (float): Word error rate.
        char_distance (int): Sum of character edit distances.
        char_total (int): Number of reference characters.
        word_distance (int): Sum of word edit distances.
        word_total (int): Number of reference words.
    """

    cer: float
    wer: float
    char_distance: int
    char_total: int
    word_distance: int
    word_total: int


def levenshtein(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Args:
        reference (Sequence[Any]): Ground-truth sequence.
        hypothesis (Sequence[Any]): Predicted sequence.

    Returns:
        int: Minimum number of insertions, deletions, and substitutions.
    """
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (ref_item != hyp_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def corpus_error_rates(references: Sequence[str], hypotheses: Sequence[str]) -> ErrorRates:
    """Compute global CER and WER over a corpus.

    Args:
        references (Sequence[str]): Reference transcriptions.
        hypotheses (Sequence[str]): Predicted transcriptions.

    Returns:
        ErrorRates: Corpus-level edit-distance metrics.

    Raises:
        ValueError: If the two sequences do not have the same length.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    char_distance = 0
    char_total = 0
    word_distance = 0
    word_total = 0

    for reference, hypothesis in zip(references, hypotheses):
        ref_text = reference or ""
        hyp_text = hypothesis or ""
        char_distance += levenshtein(ref_text, hyp_text)
        char_total += len(ref_text)

        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        word_distance += levenshtein(ref_words, hyp_words)
        word_total += len(ref_words)

    return ErrorRates(
        cer=char_distance / char_total if char_total else 0.0,
        wer=word_distance / word_total if word_total else 0.0,
        char_distance=char_distance,
        char_total=char_total,
        word_distance=word_distance,
        word_total=word_total,
    )


def bootstrap_cer_ci(
    references: Sequence[str],
    hypotheses: Sequence[str],
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Estimate a bootstrap confidence interval for CER.

    Args:
        references (Sequence[str]): Reference transcriptions.
        hypotheses (Sequence[str]): Predicted transcriptions.
        n_samples (int): Number of bootstrap resamples.
        confidence (float): Confidence level. Defaults to 0.95.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[float, float]: Lower and upper CER bounds.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        return 0.0, 0.0

    rng = random.Random(seed)
    values: List[float] = []
    n_items = len(references)
    for _ in range(n_samples):
        indices = [rng.randrange(n_items) for _ in range(n_items)]
        sampled_refs = [references[index] for index in indices]
        sampled_hyps = [hypotheses[index] for index in indices]
        values.append(corpus_error_rates(sampled_refs, sampled_hyps).cer)

    values.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, int((alpha / 2.0) * n_samples))
    upper_index = min(n_samples - 1, int((1.0 - alpha / 2.0) * n_samples) - 1)
    return values[lower_index], values[upper_index]


def mcnemar_counts(
    references: Sequence[str],
    hypotheses_a: Sequence[str],
    hypotheses_b: Sequence[str],
) -> Dict[str, int]:
    """Count paired correctness disagreements for McNemar comparison.

    A line is considered correct when its full transcription exactly matches the
    reference. This is intentionally strict and complements CER/WER.

    Args:
        references (Sequence[str]): Reference transcriptions.
        hypotheses_a (Sequence[str]): Predictions for model A.
        hypotheses_b (Sequence[str]): Predictions for model B.

    Returns:
        Dict[str, int]: Counts `both_correct`, `both_wrong`, `a_only`, `b_only`.
    """
    if not (len(references) == len(hypotheses_a) == len(hypotheses_b)):
        raise ValueError("all inputs must have the same length")

    counts = {"both_correct": 0, "both_wrong": 0, "a_only": 0, "b_only": 0}
    for reference, pred_a, pred_b in zip(references, hypotheses_a, hypotheses_b):
        a_ok = pred_a == reference
        b_ok = pred_b == reference
        if a_ok and b_ok:
            counts["both_correct"] += 1
        elif not a_ok and not b_ok:
            counts["both_wrong"] += 1
        elif a_ok:
            counts["a_only"] += 1
        else:
            counts["b_only"] += 1
    return counts


def mcnemar_chi2(counts: Dict[str, int]) -> float:
    """Compute continuity-corrected McNemar chi-square statistic.

    Args:
        counts (Dict[str, int]): Output of :func:`mcnemar_counts`.

    Returns:
        float: Chi-square statistic with one degree of freedom.
    """
    b = counts.get("a_only", 0)
    c = counts.get("b_only", 0)
    if b + c == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSON Lines records.

    Args:
        path (str): JSONL file path.

    Returns:
        List[Dict[str, Any]]: Parsed records.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate records containing `reference` and `prediction` fields.

    Args:
        records (Iterable[Dict[str, Any]]): Prediction records.

    Returns:
        Dict[str, Any]: CER/WER summary.
    """
    rows = list(records)
    references = [row["reference"] for row in rows]
    hypotheses = [row["prediction"] for row in rows]
    rates = corpus_error_rates(references, hypotheses)
    ci_low, ci_high = bootstrap_cer_ci(references, hypotheses)
    return {
        "num_lines": len(rows),
        "cer": rates.cer,
        "wer": rates.wer,
        "cer_bootstrap_95": [ci_low, ci_high],
        "char_distance": rates.char_distance,
        "char_total": rates.char_total,
        "word_distance": rates.word_distance,
        "word_total": rates.word_total,
    }


def main() -> None:
    """Run final evaluation from a JSONL predictions file."""
    parser = argparse.ArgumentParser(description="Evaluate HTR predictions with CER/WER.")
    parser.add_argument("--predictions", required=True, help="JSONL with reference and prediction fields.")
    parser.add_argument("--output", default="experiments/final_metrics.json", help="Metrics JSON output path.")
    args = parser.parse_args()

    metrics = evaluate_records(load_jsonl(args.predictions))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
