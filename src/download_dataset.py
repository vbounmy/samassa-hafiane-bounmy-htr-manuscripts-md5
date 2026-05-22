"""Script de téléchargement et d'extraction du dataset ANR e-NDP depuis Zenodo.

Ce module permet de télécharger automatiquement le fichier zip du dataset
ANR e-NDP Ground Truth (Zenodo ID: 7575693) et de l'extraire dans un répertoire
dédié pour le pipeline HTR.

Example:
    Pour lancer le téléchargement :
        python src/download_dataset.py
"""

import os
import zipfile
import hashlib
import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "7575693"
ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/e-NDP_dataset.zip?download=1"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ZIP_PATH = os.path.join(DATA_DIR, "e-NDP_dataset.zip")
EXPECTED_SHA256 = "c08502f64121bc3fb1bf0e6fdf21fdebeea36f73dbbe4a7d4ee7f3b55a90ca4a"  # Exemple fictif, recalculé à la volée


def ensure_directories() -> None:
    """Crée les répertoires nécessaires s'ils n'existent pas."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[*] Dossier de données configuré : {DATA_DIR}")


def download_file(url: str, dest_path: str) -> None:
    """Télécharge un fichier depuis une URL avec une barre de progression tqdm.

    Args:
        url (str): L'URL du fichier à télécharger.
        dest_path (str): Le chemin de destination locale.
    """
    print(f"[*] Début du téléchargement depuis : {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Téléchargement")
    
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
    progress_bar.close()
    print("[+] Téléchargement terminé avec succès !")


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extrait un fichier ZIP dans le répertoire spécifié.

    Args:
        zip_path (str): Chemin vers le fichier ZIP.
        extract_to (str): Dossier d'extraction de destination.
    """
    print(f"[*] Extraction du fichier {zip_path} vers {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Obtenir la liste des fichiers pour afficher la progression
        file_list = zip_ref.namelist()
        for file in tqdm(file_list, desc="Extraction"):
            zip_ref.extract(member=file, path=extract_to)
    print("[+] Extraction terminée !")


def main() -> None:
    """Fonction principale gérant le flux de téléchargement et d'extraction."""
    ensure_directories()
    
    if not os.path.exists(ZIP_PATH):
        try:
            download_file(ZIP_URL, ZIP_PATH)
        except Exception as e:
            print(f"[-] Erreur lors du téléchargement : {e}")
            return
    else:
        print("[*] Le fichier ZIP existe déjà localement. Passage à l'extraction.")
        
    extract_zip(ZIP_PATH, DATA_DIR)


if __name__ == "__main__":
    main()
