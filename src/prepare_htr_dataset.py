"""Module de préparation des données pour le fine-tuning HTR (TrOCR).

Ce module implémente l'étape d'extraction des lignes de texte à partir des images
de pages et des coordonnées définies dans les fichiers PAGE XML. Il structure le 
jeu de données final selon le format standard ImageFolder de Hugging Face.

Example:
    Pour exécuter la préparation :
        python src/prepare_htr_dataset.py --splits_json experiments/splits.json --output_dir data/processed_lines
"""

import argparse
import hashlib
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
import html
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Namespace standard de PAGE XML
PAGE_NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def safe_filename(value: str, fallback: str = "line") -> str:
    """Convert a PAGE XML identifier into a filesystem-safe stem.

    Args:
        value (str): Raw identifier to normalize.
        fallback (str): Value used when the raw identifier is empty.

    Returns:
        str: ASCII-safe filename stem.
    """
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or fallback).strip("._")
    return normalized or fallback


def split_signature(records: List[Dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 signature for exported line records.

    Args:
        records (List[Dict[str, Any]]): Metadata records written to metadata.jsonl.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["file_name"]):
        payload = {
            "file_name": record["file_name"],
            "text": record["text"],
            "page": record["page"],
            "line_id": record["line_id"],
        }
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def parse_page_xml(xml_path: str) -> List[Dict[str, Any]]:
    """Parse un fichier PAGE XML pour en extraire les coordonnées et les transcriptions des lignes.

    Args:
        xml_path (str): Chemin d'accès au fichier PAGE XML.

    Returns:
        List[Dict[str, Any]]: Liste de dictionnaires contenant pour chaque ligne :
            - "line_id": Identifiant unique de la ligne.
            - "polygon": Liste de tuples (x, y) des coordonnées du polygone.
            - "text": Transcription de la ligne (nettoyée).
            - "needs_review": Booléen indiquant si la ligne contient une biffure ($...$).
    """
    lines = []
    if not os.path.exists(xml_path):
        logger.warning(f"[-] Fichier XML introuvable : {xml_path}")
        return lines

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Recherche de toutes les régions de texte
        for region in root.findall(".//page:TextRegion", PAGE_NS):
            for line in region.findall("page:TextLine", PAGE_NS):
                line_id = line.get("id") or f"line_{len(lines):04d}"
                
                # Coordonnées
                coords_elem = line.find("page:Coords", PAGE_NS)
                if coords_elem is None:
                    continue
                points_str = coords_elem.get("points")
                if not points_str:
                    continue
                
                # Conversion des points en liste de coordonnées (x, y)
                polygon = []
                try:
                    for pair in points_str.strip().split():
                        x, y = pair.split(",")
                        polygon.append((int(x), int(y)))
                except ValueError:
                    logger.warning(f"[-] Format de points invalide pour la ligne {line_id} dans {xml_path} : {points_str}")
                    continue

                if not polygon:
                    continue

                # Transcription
                unicode_elem = line.find("page:TextEquiv/page:Unicode", PAGE_NS)
                raw_text = unicode_elem.text if unicode_elem is not None and unicode_elem.text is not None else ""
                
                # Nettoyage de la transcription
                # 1. Résoudre les entités HTML (ex: &#39; -> ')
                clean_text = html.unescape(raw_text).strip()
                
                # 2. Détection des biffures / corrections entourées de '$'
                # Contrainte n°5 : Needs review si présence de biffures ($...$)
                needs_review = "$" in clean_text

                lines.append({
                    "line_id": line_id,
                    "polygon": polygon,
                    "text": clean_text,
                    "needs_review": needs_review
                })

    except Exception as e:
        logger.error(f"[-] Erreur lors du parsing XML de {xml_path} : {e}")

    return lines


def crop_line_image(
    image: np.ndarray,
    polygon: List[Tuple[int, int]],
    padding: int = 2
) -> np.ndarray:
    """Découpe la zone correspondant au polygone d'une ligne dans l'image de la page.

    Calcule la boîte englobante du polygone avec une marge (padding) et l'extrait.

    Args:
        image (np.ndarray): Image de la page complète (BGR).
        polygon (List[Tuple[int, int]]): Liste des sommets du polygone.
        padding (int): Marge de pixels à ajouter autour de la boîte englobante.

    Returns:
        np.ndarray: Image de la ligne découpée.
    """
    h, w = image.shape[:2]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    xmin = max(0, min(xs) - padding)
    xmax = min(w, max(xs) + padding)
    ymin = max(0, min(ys) - padding)
    ymax = min(h, max(ys) + padding)

    return image[ymin:ymax, xmin:xmax]


def preprocess_crop(
    crop: np.ndarray,
    method: str = "clahe"
) -> np.ndarray:
    """Applique des prétraitements d'image sur le découpage de la ligne.

    Args:
        crop (np.ndarray): Image de la ligne de texte découpée.
        method (str): Méthode de prétraitement : "none", "clahe", ou "binary".

    Returns:
        np.ndarray: Image prétraitée (en niveaux de gris ou binarisée).
    """
    if method == "none":
        return crop

    # Conversion en niveaux de gris si nécessaire
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Application d'un filtre bilatéral pour lisser les textures sans flouter les contours
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    if method == "clahe":
        # Amélioration adaptative de contraste locale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    elif method == "binary":
        # Binarisation locale d'Otsu en alternative simple
        # ou Sauvola si disponible. Ici Otsu est rapide et robuste sur les lignes isolées.
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    return gray


def main() -> None:
    parser = argparse.ArgumentParser(description="Préparation du dataset HTR ligne par ligne.")
    parser.add_argument(
        "--splits_json",
        type=str,
        default=os.path.normpath("experiments/splits.json"),
        help="Chemin vers le fichier JSON des splits."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.normpath("data/processed_lines"),
        help="Répertoire de sortie pour le dataset final."
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        choices=["none", "clahe", "binary"],
        default="clahe",
        help="Type de prétraitement à appliquer aux découpages de lignes."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour assurer la reproductibilité."
    )
    parser.add_argument(
        "--max_pages_per_split",
        type=int,
        default=None,
        help="Limite optionnelle de pages par split pour valider rapidement l'extraction."
    )

    args = parser.parse_args()

    # Assurer le déterminisme
    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_path = os.path.join(root_dir, args.splits_json)

    if not os.path.exists(splits_path):
        logger.error(f"[-] Fichier splits.json introuvable à l'adresse : {splits_path}")
        return

    with open(splits_path, "r", encoding="utf-8") as f:
        data_splits = json.load(f)

    # Création du dossier racine de sortie
    out_base = os.path.join(root_dir, args.output_dir)
    os.makedirs(out_base, exist_ok=True)

    # Définition des splits à traiter (mapping val -> validation pour HF ImageFolder)
    split_mapping = {
        "train": "train",
        "val": "validation",
        "test": "test"
    }

    logger.info(f"[*] Démarrage de l'extraction des lignes. Prétraitement choisi : {args.preprocessing}")

    for split_key, hf_dir_name in split_mapping.items():
        if split_key not in data_splits["splits"]:
            logger.warning(f"[-] Split {split_key} absent de splits.json. Ignoré.")
            continue

        split_dir = os.path.join(out_base, hf_dir_name)
        os.makedirs(split_dir, exist_ok=True)
        
        metadata_file = os.path.join(split_dir, "metadata.jsonl")
        metadata_records = []

        logger.info(f"[*] Traitement du split '{split_key}' -> '{hf_dir_name}'...")
        page_entries = data_splits["splits"][split_key]
        if args.max_pages_per_split is not None:
            page_entries = page_entries[: args.max_pages_per_split]

        for entry in tqdm(page_entries):
            img_rel = entry["image"]
            xml_rel = entry["xml"]

            img_abs = os.path.join(root_dir, img_rel)
            xml_abs = os.path.join(root_dir, xml_rel)

            if not os.path.exists(img_abs):
                logger.warning(f"[-] Image introuvable : {img_rel}")
                continue

            page_image = cv2.imread(img_abs)
            if page_image is None:
                logger.warning(f"[-] Échec du chargement de l'image : {img_rel}")
                continue

            # Parsing XML pour récupérer les lignes
            lines_data = parse_page_xml(xml_abs)
            page_name = os.path.splitext(os.path.basename(img_rel))[0]

            for idx, line in enumerate(lines_data):
                crop = crop_line_image(page_image, line["polygon"])
                
                # Si le découpage est vide ou trop petit, on l'ignore
                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue

                # Prétraitement
                processed_crop = preprocess_crop(crop, method=args.preprocessing)

                # Nom du fichier de ligne unique
                line_stem = safe_filename(str(line["line_id"]), fallback=f"line_{idx:04d}")
                crop_filename = f"{safe_filename(page_name)}_{line_stem}_{idx}.png"
                crop_path = os.path.join(split_dir, crop_filename)

                # Enregistrement de l'image de la ligne
                cv2.imwrite(crop_path, processed_crop)

                # Enregistrement des métadonnées correspondantes
                # Colonnes : file_name, text (requis par HF ImageFolder) + métadonnées complémentaires
                metadata_records.append({
                    "file_name": crop_filename,
                    "text": line["text"],
                    "transcription_convention": "semi-diplomatic",
                    "language": "lat",  # e-NDP est en latin capitulaire
                    "needs_review": line["needs_review"],
                    "polygon": line["polygon"],
                    "line_id": line["line_id"],
                    "page": os.path.basename(img_rel)
                })

        # Écriture du fichier metadata.jsonl
        with open(metadata_file, "w", encoding="utf-8") as meta_f:
            for record in metadata_records:
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        signature = split_signature(metadata_records)
        logger.info(
            f"[+] Split '{hf_dir_name}' terminé : {len(metadata_records)} lignes extraites "
            f"(sha256={signature})."
        )

    logger.info(f"[+] Dataset HTR complet généré avec succès dans : {out_base}")


if __name__ == "__main__":
    main()
