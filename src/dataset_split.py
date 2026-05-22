"""Module de découpage déterministe du corpus (Train / Val / Test) et scellement.

Ce module implémente l'étape de constitution du split (Étape 2 et Contrainte n°3) :
1. Recherche des paires (images, transcriptions PAGE XML) dans le dataset e-NDP.
2. Découpage déterministe (par ex. 80% train, 10% val, 10% test) avec une seed fixée.
3. Calcul et enregistrement du hachage SHA-256 des fichiers pour chaque ensemble.
4. Génération d'un fichier JSON de configuration scellé `experiments/splits.json`
   qui servira de référence immuable.

Example:
    Pour exécuter le split :
        python src/dataset_split.py --data_dir data/e-NDP_dataset
"""

import os
import argparse
import json
import random
import hashlib
import logging
from typing import List, Dict, Any, Tuple

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_file_sha256(filepath: str) -> str:
    """Calcule le hachage SHA-256 d'un fichier donné.

    Args:
        filepath (str): Chemin d'accès au fichier.

    Returns:
        str: Le hachage SHA-256 au format hexadécimal.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Lire par blocs de 64 ko pour ne pas surcharger la mémoire
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def calculate_split_signature(file_paths: List[str]) -> str:
    """Calcule la signature SHA-256 globale d'un ensemble de fichiers (split).

    Cette signature combine le nom et le hachage individuel de chaque fichier
    trié pour être indépendante de l'ordre, garantissant l'intégrité du split.

    Args:
        file_paths (List[str]): Liste des chemins absolus ou relatifs des fichiers.

    Returns:
        str: Signature globale du split.
    """
    combined_hashes = hashlib.sha256()
    # Trier les chemins pour assurer le déterminisme
    for path in sorted(file_paths):
        file_name = os.path.basename(path)
        file_hash = calculate_file_sha256(path)
        # Combiner le nom du fichier et son hash
        combined_hashes.update(f"{file_name}:{file_hash}".encode("utf-8"))
    return combined_hashes.hexdigest()


def discover_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """Découvre les paires d'images et de transcriptions PAGE XML dans le dossier du dataset.

    Args:
        data_dir (str): Dossier racine du dataset extrait.

    Returns:
        List[Tuple[str, str]]: Liste de paires (chemin_image, chemin_page_xml).
    """
    pairs = []
    
    # Structure typique d'e-NDP : HTR_ground_truth/Images et HTR_ground_truth/Transcriptions
    images_dir = os.path.join(data_dir, "HTR_ground_truth", "Images")
    trans_dir = os.path.join(data_dir, "HTR_ground_truth", "Transcriptions")
    
    # Fallback si la structure est à plat ou différente
    if not os.path.exists(images_dir) or not os.path.exists(trans_dir):
        logger.warning(f"Structure standard non trouvée dans {data_dir}. Recherche récursive...")
        # Recherche récursive de fichiers images et de fichiers .xml correspondants
        all_xmls = {}
        all_images = {}
        for root, _, files in os.walk(data_dir):
            for file in files:
                name, ext = os.path.splitext(file)
                full_path = os.path.join(root, file)
                if ext.lower() == ".xml":
                    all_xmls[name] = full_path
                elif ext.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    all_images[name] = full_path
                    
        # Apparier par nom de base
        for name, img_path in all_images.items():
            if name in all_xmls:
                pairs.append((img_path, all_xmls[name]))
                
        return pairs

    # Si structure standard
    img_files = sorted(os.listdir(images_dir))
    for img_file in img_files:
        img_name, img_ext = os.path.splitext(img_file)
        if img_ext.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            xml_file = f"{img_name}.xml"
            xml_path = os.path.join(trans_dir, xml_file)
            if os.path.exists(xml_path):
                pairs.append((
                    os.path.join(images_dir, img_file),
                    xml_path
                ))
            else:
                logger.warning(f"[-] Pas de transcription PAGE XML trouvée pour {img_file}")
                
    return pairs


def create_splits(
    pairs: List[Tuple[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Découpe de manière déterministe les paires en ensembles train, val et test.

    Args:
        pairs (List[Tuple[str, str]]): Liste de paires (image, xml).
        train_ratio (float): Proportion de données d'entraînement.
        val_ratio (float): Proportion de données de validation.
        test_ratio (float): Proportion de données de test.
        seed (int): Graine aléatoire pour le déterminisme.

    Returns:
        Tuple: (train_pairs, val_pairs, test_pairs)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "La somme des proportions doit valoir 1.0"
    
    # Fixer la seed
    rng = random.Random(seed)
    
    # Copier la liste et mélanger de manière reproductible
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)
    
    total = len(shuffled_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = shuffled_pairs[:train_end]
    val_pairs = shuffled_pairs[train_end:val_end]
    test_pairs = shuffled_pairs[val_end:]
    
    return train_pairs, val_pairs, test_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Découpage déterministe et scellement du corpus HTR.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Chemin vers le répertoire racine des données."
    )
    parser.add_argument(
        "--train", type=float, default=0.8, help="Ratio pour l'ensemble d'entraînement."
    )
    parser.add_argument(
        "--val", type=float, default=0.1, help="Ratio pour l'ensemble de validation."
    )
    parser.add_argument(
        "--test", type=float, default=0.1, help="Ratio pour l'ensemble de test."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed pour le mélange aléatoire."
    )
    
    args = parser.parse_args()
    
    # Découvrir les paires
    logger.info(f"[*] Analyse du dossier de données : {args.data_dir}")
    pairs = discover_pairs(args.data_dir)
    
    if not pairs:
        logger.error("[-] Aucune paire (Image, PAGE XML) trouvée. Veuillez télécharger et extraire le dataset au préalable.")
        return
        
    logger.info(f"[+] {len(pairs)} paires cohérentes d'images et de transcriptions détectées.")
    
    # Effectuer le découpage
    train_pairs, val_pairs, test_pairs = create_splits(
        pairs,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
    
    logger.info(f"    - Entraînement (Train) : {len(train_pairs)} paires")
    logger.info(f"    - Validation (Val)    : {len(val_pairs)} paires")
    logger.info(f"    - Test (Test scellé)  : {len(test_pairs)} paires")
    
    # Chemin relatif pour faciliter la réutilisation indépendamment du chemin absolu de l'hôte
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def to_relative(abs_path: str) -> str:
        return os.path.relpath(abs_path, root_dir)
        
    # Calculer les signatures SHA-256
    train_images = [p[0] for p in train_pairs]
    train_xmls = [p[1] for p in train_pairs]
    
    val_images = [p[0] for p in val_pairs]
    val_xmls = [p[1] for p in val_pairs]
    
    test_images = [p[0] for p in test_pairs]
    test_xmls = [p[1] for p in test_pairs]
    
    logger.info("[*] Calcul des hachages SHA-256 pour sceller les données...")
    
    train_images_hash = calculate_split_signature(train_images)
    val_images_hash = calculate_split_signature(val_images)
    test_images_hash = calculate_split_signature(test_images)
    
    # Préparer le fichier JSON final
    manifest = {
        "metadata": {
            "seed": args.seed,
            "train_ratio": args.train,
            "val_ratio": args.val,
            "test_ratio": args.test,
            "total_pairs": len(pairs),
            "sha256_signatures": {
                "train_images": train_images_hash,
                "val_images": val_images_hash,
                "test_images": test_images_hash
            }
        },
        "splits": {
            "train": [{"image": to_relative(p[0]), "xml": to_relative(p[1])} for p in train_pairs],
            "val": [{"image": to_relative(p[0]), "xml": to_relative(p[1])} for p in val_pairs],
            "test": [{"image": to_relative(p[0]), "xml": to_relative(p[1])} for p in test_pairs]
        }
    }
    
    # S'assurer de la présence du répertoire d'expériences
    exp_dir = os.path.join(root_dir, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    
    splits_file = os.path.join(exp_dir, "splits.json")
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    logger.info(f"[+] Splits déterministes enregistrés et scellés avec succès dans : {to_relative(splits_file)}")
    logger.info(f"[+] Hachage SHA-256 de test set (non-contamination) : {test_images_hash}")


if __name__ == "__main__":
    main()
