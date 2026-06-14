"""Generate TrOCR line predictions as JSONL from processed line crops."""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from peft import PeftModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_confidence(outputs, device: torch.device) -> float:
    if not hasattr(outputs, "scores") or outputs.scores is None:
        return 0.0

    sequence = outputs.sequences[0].tolist()
    token_ids = sequence[1:]
    if len(token_ids) == 0:
        return 0.0

    log_probs = []
    for step, score in enumerate(outputs.scores):
        if step >= len(token_ids):
            break
        token_id = token_ids[step]
        probs = torch.log_softmax(score, dim=-1)
        log_probs.append(probs[0, token_id])

    if not log_probs:
        return 0.0

    avg_log_prob = torch.stack(log_probs).mean().item()
    return float(torch.exp(torch.tensor(avg_log_prob, device=device)).clamp(0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TrOCR line predictions in JSONL format.")
    parser.add_argument(
        "--metadata",
        type=str,
        default=os.path.normpath("data/processed_lines/test/metadata.jsonl"),
        help="Path to the metadata.jsonl for the target split."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.normpath("data/processed_lines/test"),
        help="Directory containing the processed line crop images for the same split."
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=os.path.normpath("experiments/checkpoints/trocr_lora/final_lora_adapter"),
        help="Path to the LoRA adapter directory saved by train_trocr.py."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="Base TrOCR model name or local path."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.normpath("predictions/trocr.jsonl"),
        help="Output JSONL predictions path."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum generation length."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for generation."
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Limit inference to the first N lines from metadata."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU."
    )

    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info(f"[*] Using device: {device}")

    metadata = load_metadata(args.metadata)
    if not metadata:
        logger.error("No metadata records found.")
        return

    if args.max_lines is not None:
        metadata = metadata[: args.max_lines]
        logger.info(f"[*] Limiting inference to first {len(metadata)} lines.")

    processor = TrOCRProcessor.from_pretrained(args.base_model)
    base_model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()

    output_records: List[Dict[str, Any]] = []
    total = len(metadata)
    logger.info(f"[*] Generating predictions for {total} lines from {args.dataset_dir}")

    for idx, record in enumerate(metadata, start=1):
        file_name = record.get("file_name")
        if not file_name:
            logger.warning("Skipping metadata record without file_name.")
            continue

        image_path = os.path.join(args.dataset_dir, file_name)
        if not os.path.exists(image_path):
            logger.warning(f"Missing image file: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=args.max_length,
                num_beams=args.num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prediction = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        confidence = compute_confidence(outputs, device)

        output_records.append(
            {
                "file_name": file_name,
                "prediction": prediction,
                "confidence": confidence,
                "model": "trocr",
            }
        )

        if idx % max(1, args.batch_size * 5) == 0 or idx == total:
            logger.info(f"[*] Processed {idx}/{total} lines")

    write_jsonl(args.output, output_records)
    logger.info(f"[+] Wrote {len(output_records)} TrOCR prediction records to {args.output}")


if __name__ == "__main__":
    main()
