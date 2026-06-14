"""Module d'entraînement HTR de TrOCR par fine-tuning LoRA (PEFT).

Ce module implémente l'étape 3 du projet :
1. Chargement du dataset préparé par `src/prepare_htr_dataset.py`.
2. Évaluation de la baseline (modèle pré-entraîné sans fine-tuning).
3. Configuration de LoRA (PEFT) et entraînement reproductible (seed=42).
4. Suivi des performances avec les métriques CER et WER (jiwer).
5. Enregistrement des résultats de manière structurée dans `experiments/journal.jsonl`.
"""

import os
import argparse
import logging
import json
from datetime import UTC, datetime
import numpy as np
import random
from typing import Any, Dict

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fixe la graine aléatoire pour la reproductibilité complète.

    Args:
        seed (int): Graine aléatoire. Defaults to 42.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Rendre les opérations CUDA déterministes si applicables
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_to_journal(
    journal_path: str,
    run_id: str,
    model_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any]
) -> None:
    """Enregistre le résultat de l'exécution dans le journal en temps réel.

    Args:
        journal_path (str): Chemin d'accès au fichier journal.jsonl.
        run_id (str): Identifiant unique du run.
        model_name (str): Nom du modèle de base utilisé.
        parameters (Dict[str, Any]): Dictionnaire d'hyperparamètres.
        metrics (Dict[str, Any]): Métriques finales calculées.
    """
    os.makedirs(os.path.dirname(journal_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "model_name": model_name,
        "parameters": parameters,
        "metrics": metrics
    }
    
    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"[+] Run '{run_id}' journalisé avec succès dans {os.path.basename(journal_path)}.")


def average_error_rates(predictions: list[str], references: list[str]) -> Dict[str, float]:
    """Compute mean CER and WER while ignoring empty references.

    Args:
        predictions (list[str]): Model predictions.
        references (list[str]): Ground-truth transcriptions.

    Returns:
        Dict[str, float]: Mean character and word error rates.
    """
    import jiwer

    cer_scores = []
    wer_scores = []
    for pred, label in zip(predictions, references):
        pred_clean = pred.strip()
        label_clean = label.strip()
        if not label_clean:
            continue
        cer_scores.append(jiwer.cer(label_clean, pred_clean))
        wer_scores.append(jiwer.wer(label_clean, pred_clean))

    return {
        "cer": float(np.mean(cer_scores)) if cer_scores else 0.0,
        "wer": float(np.mean(wer_scores)) if wer_scores else 0.0,
    }


def main() -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator,
    )

    if not hasattr(VisionEncoderDecoderConfig, "vocab_size"):
        def _get_vocab_size(self):
            decoder = getattr(self, "decoder", None)
            return getattr(decoder, "vocab_size", None) if decoder is not None else self.__dict__.get("vocab_size")

        def _set_vocab_size(self, value):
            self.__dict__["vocab_size"] = value

        VisionEncoderDecoderConfig.vocab_size = property(_get_vocab_size, _set_vocab_size)

    parser = argparse.ArgumentParser(description="Fine-tuning de TrOCR avec LoRA.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.normpath("data/processed_lines"),
        help="Chemin vers le répertoire racine du dataset pré-segmenté."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.normpath("experiments/checkpoints/trocr_lora"),
        help="Répertoire d'enregistrement des checkpoints."
    )
    parser.add_argument(
        "--journal_path",
        type=str,
        default=os.path.normpath("experiments/journal.jsonl"),
        help="Chemin du journal d'entraînement."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="Modèle de base Hugging Face TrOCR."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Nombre d'époques d'entraînement."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Taille de batch (par GPU/CPU)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate initial."
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rang r de LoRA."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Facteur d'échelle alpha de LoRA."
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Taux de dropout LoRA."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité."
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Exécute un test rapide (1 époque sur un échantillon réduit sur CPU)."
    )

    args = parser.parse_args()

    # Fixer la seed
    set_seed(args.seed)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, args.data_dir)
    journal_path = os.path.join(root_dir, args.journal_path)

    if not os.path.exists(data_path):
        logger.error(f"[-] Dossier de données introuvable : {data_path}. Lancez d'abord src/prepare_htr_dataset.py.")
        return

    logger.info(f"[*] Chargement du dataset depuis {data_path}...")
    dataset = load_dataset("imagefolder", data_dir=data_path)

    # Réduction si smoke test (pour validation rapide CPU)
    if args.smoke_test:
        logger.info("[!] Mode SMOKE TEST activé : sous-échantillonnage à 10 exemples.")
        args.epochs = 1
        args.batch_size = 2
        for split in dataset.keys():
            # Limiter à 10 exemples par split
            dataset[split] = dataset[split].select(range(min(len(dataset[split]), 10)))

    # Chargement du processeur et du modèle TrOCR
    logger.info(f"[*] Chargement du modèle de base : {args.model_name}...")
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    # Configuration des jetons spéciaux du modèle
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Paramètres de génération à configurer via generation_config
    if getattr(model, "generation_config", None) is None:
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig()
    model.generation_config.max_length = 64
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    # Fonction de transformation à la volée (économise la mémoire vive et évite le cache disque volumineux)
    def transform(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        
        # Tokenisation des labels
        labels = processor.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=64,
            truncation=True
        ).input_ids
        
        # Remplacement du jeton de padding par -100 pour que la perte PyTorch l'ignore
        labels = [
            [(token if token != processor.tokenizer.pad_token_id else -100) for token in label]
            for label in labels
        ]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }

    dataset.set_transform(transform)

    # Configuration de la fonction de calcul des métriques
    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, label_ids = eval_pred
        
        # Décodage des prédictions (predict_with_generate génère des IDs)
        pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Reconstruction des vrais labels (-100 -> pad_token_id pour décoder)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        return average_error_rates(pred_str, label_str)

    # 1. Évaluation de la baseline sans aucun entraînement
    logger.info("[*] Évaluation de la baseline (Zero-shot)...")
    
    baseline_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(root_dir, "experiments/checkpoints/baseline_eval"),
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        use_cpu=True if args.smoke_test else False,
        remove_unused_columns=False,
        report_to="none"
    )
    
    baseline_trainer = Seq2SeqTrainer(
        model=model,
        args=baseline_args,
        eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    baseline_metrics = baseline_trainer.evaluate()
    logger.info(f"[+] Métriques Baseline - CER: {baseline_metrics['eval_cer']:.4f} | WER: {baseline_metrics['eval_wer']:.4f}")
    
    # Enregistrer la baseline dans le journal
    log_to_journal(
        journal_path=journal_path,
        run_id=f"baseline_zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_name=args.model_name,
        parameters={"zero_shot": True},
        metrics={
            "eval_loss": baseline_metrics.get("eval_loss", 0.0),
            "eval_cer": baseline_metrics["eval_cer"],
            "eval_wer": baseline_metrics["eval_wer"]
        }
    )

    # 2. Configuration de PEFT / LoRA
    logger.info("[*] Application de la configuration LoRA (PEFT)...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none"
    )

    # Convertir le modèle standard en modèle adaptateur LoRA
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()

    # 3. Entraînement LoRA
    logger.info("[*] Configuration du pipeline d'entraînement...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(root_dir, args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=10 if not args.smoke_test else 1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,  # Requis pour keep image/text originaux dans transform
        use_cpu=True if args.smoke_test else False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else dataset["test"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("[*] Lancement de l'entraînement LoRA...")
    trainer.train()

    # Évaluation finale avec le meilleur modèle LoRA chargé
    logger.info("[*] Évaluation finale sur l'ensemble de validation...")
    final_metrics = trainer.evaluate()
    logger.info(f"[+] Métriques Finales LoRA - CER: {final_metrics['eval_cer']:.4f} | WER: {final_metrics['eval_wer']:.4f}")

    # Enregistrer le run final dans le journal
    log_to_journal(
        journal_path=journal_path,
        run_id=f"trocr_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_name=args.model_name,
        parameters={
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "smoke_test": args.smoke_test
        },
        metrics={
            "eval_loss": final_metrics.get("eval_loss", 0.0),
            "eval_cer": final_metrics["eval_cer"],
            "eval_wer": final_metrics["eval_wer"]
        }
    )

    # Sauvegarde finale des poids de l'adaptateur LoRA uniquement
    final_save_dir = os.path.join(root_dir, args.output_dir, "final_lora_adapter")
    os.makedirs(final_save_dir, exist_ok=True)
    lora_model.save_pretrained(final_save_dir)
    logger.info(f"[+] Adaptateur LoRA sauvegardé avec succès dans : {final_save_dir}")


if __name__ == "__main__":
    main()
