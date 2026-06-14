"""Generate figures and qualitative examples for the paper.

Outputs into `experiments/figures/` and `experiments/qualitative_examples.jsonl`.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def plot_metrics(metrics: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    labels = ["CER", "WER"]
    values = [metrics.get("cer", 0.0), metrics.get("wer", 0.0)]

    plt.figure(figsize=(4, 3))
    sns.barplot(x=labels, y=values, palette="muted")
    plt.ylim(0, max(values) * 1.1 + 0.01)
    plt.title("Corpus error rates")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cer_wer.png"), dpi=200)
    plt.close()


def plot_confidence_hist(records: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    confidences = [float(r.get("confidence_calibrated", r.get("confidence", 0.0))) for r in records]
    plt.figure(figsize=(5, 3))
    sns.histplot(confidences, bins=50, kde=False, color="tab:blue")
    plt.xlabel("Calibrated confidence")
    plt.title("Distribution of calibrated confidences")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_hist.png"), dpi=200)
    plt.close()


def save_qualitative_examples(records: List[Dict[str, Any]], out_dir: str, n_per_group: int = 5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # sort by confidence
    sorted_records = sorted(records, key=lambda r: float(r.get("confidence_calibrated", r.get("confidence", 0.0))))
    low = sorted_records[:n_per_group]
    high = sorted_records[-n_per_group:]
    examples = {"low_confidence": low, "high_confidence": list(reversed(high))}
    out_path = os.path.join(out_dir, "qualitative_examples.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for group, items in examples.items():
            for item in items:
                item_out = dict(item)
                item_out["group"] = group
                f.write(json.dumps(item_out, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate figures and qualitative examples.")
    parser.add_argument("--metrics", default="experiments/final_metrics.json")
    parser.add_argument("--records", default="experiments/aggregated_calibrated.jsonl")
    parser.add_argument("--out_dir", default="experiments/figures")
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    records = load_jsonl(args.records)

    plot_metrics(metrics, args.out_dir)
    plot_confidence_hist(records, args.out_dir)
    save_qualitative_examples(records, args.out_dir)
    print("Wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
