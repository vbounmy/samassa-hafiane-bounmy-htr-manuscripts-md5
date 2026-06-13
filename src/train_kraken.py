"""Module de configuration et de génération de commandes pour le fine-tuning Kraken.

En raison de l'incompatibilité de kraken avec l'environnement Windows natif,
ce module permet d'exporter le dataset e-NDP dans le format requis par Kraken
(images de lignes + transcriptions .gt.txt) et génère le script d'entraînement 
Bash exécutable sous Linux / WSL ou Google Colab.
"""

import os
import argparse
import json
import logging
import cv2
import numpy as np
from tqdm import tqdm
try:
    from src.prepare_htr_dataset import parse_page_xml, crop_line_image, preprocess_crop, safe_filename
except ModuleNotFoundError:
    from prepare_htr_dataset import parse_page_xml, crop_line_image, preprocess_crop, safe_filename

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_kraken_script(output_script_path: str, train_list: str, val_list: str) -> None:
    """Génère un script Bash run_kraken_training.sh contenant les commandes de fine-tuning.

    Args:
        output_script_path (str): Chemin du script Bash de sortie.
        train_list (str): Chemin vers le fichier contenant la liste d'images train.
        val_list (str): Chemin vers le fichier contenant la liste d'images validation.
    """
    script_content = f"""#!/usr/bin/env bash
# ==============================================================================
# Script d'entraînement Kraken (Ketos) HTR
# À exécuter dans un environnement Linux (WSL, Docker, ou Colab) avec kraken.
# ==============================================================================

# S'assurer que kraken est installé
if ! command -v ketos &> /dev/null
then
    echo "[-] ketos (kraken) n'est pas installé dans le PATH de cet environnement."
    echo "    Installez-le avec : pip install kraken"
    exit 1
fi

echo "[*] Démarrage de l'entraînement Kraken..."

# Lancement de l'entraînement avec ketos train
# -t : fichier contenant la liste des images de lignes d'entraînement
# -e : fichier contenant la liste des images de lignes de validation
# -o : chemin de sortie du modèle
# --resize : comportement de redimensionnement de l'image
# -p : taux de dropout (recommandé pour éviter le surapprentissage)
# --device : matériel d'exécution (par défaut cuda si disponible)

ketos train \\
  --device cuda \\
  -t "{train_list}" \\
  -e "{val_list}" \\
  -o "experiments/checkpoints/kraken_model" \\
  --resize keep \\
  --epochs 15 \\
  --lrate 1e-4 \\
  -p 0.1

echo "[+] Entraînement Kraken terminé !"
"""

    with open(output_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script_content)
        
    # Essayer de donner les droits d'exécution (si supporté par l'OS hôte)
    try:
        os.chmod(output_script_path, 0o755)
    except Exception:
        pass
        
    logger.info(f"[+] Script Bash Kraken généré avec succès dans : {os.path.basename(output_script_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportation de données et script Kraken.")
    parser.add_argument(
        "--splits_json",
        type=str,
        default=os.path.normpath("experiments/splits.json"),
        help="Chemin vers le fichier JSON des splits."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.normpath("data/kraken_dataset"),
        help="Répertoire d'exportation pour Kraken."
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        choices=["none", "clahe", "binary"],
        default="binary",  # Kraken préfère généralement les images binarisées
        help="Type de prétraitement à appliquer aux images de lignes."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire."
    )
    parser.add_argument(
        "--max_pages_per_split",
        type=int,
        default=None,
        help="Limite optionnelle de pages par split pour valider rapidement l'export."
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_path = os.path.join(root_dir, args.splits_json)

    if not os.path.exists(splits_path):
        logger.error(f"[-] Fichier splits.json introuvable : {splits_path}")
        return

    with open(splits_path, "r", encoding="utf-8") as f:
        data_splits = json.load(f)

    # Répertoire de sortie
    out_base = os.path.join(root_dir, args.output_dir)
    os.makedirs(out_base, exist_ok=True)

    split_mapping = {
        "train": "train",
        "val": "validation",
        "test": "test"
    }

    logger.info(f"[*] Démarrage de l'exportation Kraken (Prétraitement : {args.preprocessing}).")

    for split_key, dir_name in split_mapping.items():
        if split_key not in data_splits["splits"]:
            continue

        split_dir = os.path.join(out_base, dir_name)
        os.makedirs(split_dir, exist_ok=True)

        list_file_path = os.path.join(out_base, f"{dir_name}.txt")
        image_paths = []

        logger.info(f"[*] Exportation du split '{split_key}'...")
        page_entries = data_splits["splits"][split_key]
        if args.max_pages_per_split is not None:
            page_entries = page_entries[: args.max_pages_per_split]

        for entry in tqdm(page_entries):
            img_rel = entry["image"]
            xml_rel = entry["xml"]

            img_abs = os.path.join(root_dir, img_rel)
            xml_abs = os.path.join(root_dir, xml_rel)

            if not os.path.exists(img_abs):
                continue

            page_image = cv2.imread(img_abs)
            if page_image is None:
                continue

            lines_data = parse_page_xml(xml_abs)
            page_name = os.path.splitext(os.path.basename(img_rel))[0]

            for idx, line in enumerate(lines_data):
                crop = crop_line_image(page_image, line["polygon"])
                
                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue

                processed_crop = preprocess_crop(crop, method=args.preprocessing)

                # Noms de fichiers Kraken standard (.png et .gt.txt)
                line_stem = safe_filename(str(line["line_id"]), fallback=f"line_{idx:04d}")
                base_stem = f"{safe_filename(page_name)}_{line_stem}_{idx}"
                crop_filename = f"{base_stem}.png"
                gt_filename = f"{base_stem}.gt.txt"

                crop_path = os.path.join(split_dir, crop_filename)
                gt_path = os.path.join(split_dir, gt_filename)

                # Écriture de l'image de ligne
                cv2.imwrite(crop_path, processed_crop)

                # Écriture de la transcription ground-truth en UTF-8
                with open(gt_path, "w", encoding="utf-8") as gt_f:
                    gt_f.write(line["text"])

                # Enregistrement du chemin relatif pour le fichier de liste ketos
                rel_path_to_list = os.path.join(args.output_dir, dir_name, crop_filename)
                # Normalisation en slashs Linux pour compatibilité WSL/Linux
                image_paths.append(rel_path_to_list.replace("\\", "/"))

        # Écriture du fichier de liste pour ketos train
        with open(list_file_path, "w", encoding="utf-8") as list_f:
            for path in image_paths:
                list_f.write(path + "\n")

        logger.info(f"[+] Split '{dir_name}' exporté : {len(image_paths)} paires d'images et ground-truth générées.")

    # Génération du script Bash run_kraken_training.sh
    script_path = os.path.join(root_dir, "run_kraken_training.sh")
    train_list_rel = os.path.join(args.output_dir, "train.txt").replace("\\", "/")
    val_list_rel = os.path.join(args.output_dir, "validation.txt").replace("\\", "/")
    generate_kraken_script(script_path, train_list_rel, val_list_rel)

    logger.info("[+] Exportation de données pour Kraken terminée avec succès !")


if __name__ == "__main__":
    main()
