"""Suite de tests unitaires (pytest) pour le pipeline de prétraitement et de segmentation.

Ce module implémente la contrainte n°5 du sujet :
- Test des formes d'images après traitement.
- Test des types de données de sortie (uint8).
- Test des plages de valeurs attendues (0 et 255) après binarisation de Sauvola.
- Test de non-régression sur la correction d'inclinaison.

Example:
    Pour lancer les tests :
        pytest tests/
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Ajouter le répertoire racine au path pour pouvoir importer src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import PreprocessingPipeline
from src.segmentation import PageSegmenter


@pytest.fixture
def dummy_color_image() -> np.ndarray:
    """Génère une image couleur factice avec du texte simulé pour les tests.

    Returns:
        np.ndarray: Image de forme (300, 400, 3) au format BGR.
    """
    # Créer un fond blanc
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Dessiner des barres horizontales sombres simulées (lignes de texte)
    cv2.rectangle(img, (50, 40), (350, 60), (50, 50, 50), -1)
    cv2.rectangle(img, (50, 100), (350, 120), (50, 50, 50), -1)
    cv2.rectangle(img, (50, 160), (350, 180), (50, 50, 50), -1)
    cv2.rectangle(img, (50, 220), (350, 240), (50, 50, 50), -1)
    
    return img


@pytest.fixture
def skewed_image(dummy_color_image: np.ndarray) -> np.ndarray:
    """Génère une image factice inclinée de 5 degrés.

    Args:
        dummy_color_image (np.ndarray): L'image factice horizontale.

    Returns:
        np.ndarray: L'image inclinée.
    """
    h, w = dummy_color_image.shape[:2]
    center = (w // 2, h // 2)
    # Inclinaison de 5 degrés dans le sens horaire
    rotation_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    skewed = cv2.warpAffine(
        dummy_color_image, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return skewed


def test_preprocessing_pipeline_initialization() -> None:
    """Vérifie que la classe de prétraitement s'initialise correctement avec les bons paramètres."""
    pipeline = PreprocessingPipeline(clip_limit=3.0, sauvola_window=15, sauvola_k=0.15)
    assert pipeline.clip_limit == 3.0
    assert pipeline.sauvola_window == 15
    assert pipeline.sauvola_k == 0.15


def test_binarization_values_and_type(dummy_color_image: np.ndarray) -> None:
    """Vérifie que l'image binarisée est bien de type uint8, de même taille et ne contient que 0 ou 255."""
    pipeline = PreprocessingPipeline(sauvola_window=25, sauvola_k=0.2)
    results = pipeline.run(dummy_color_image)
    
    binary = results["binary"]
    
    # 1. Type
    assert binary.dtype == np.uint8
    
    # 2. Dimensions
    assert binary.shape == dummy_color_image.shape[:2]
    
    # 3. Plage de valeurs (Seulement 0 pour le texte et 255 pour le fond)
    unique_vals = np.unique(binary)
    for val in unique_vals:
        assert val in [0, 255]


def test_clahe_contrast_enhancement(dummy_color_image: np.ndarray) -> None:
    """Vérifie que l'application de CLAHE préserve la forme et le type de l'image."""
    pipeline = PreprocessingPipeline()
    clahe_img = pipeline.apply_clahe(dummy_color_image)
    
    assert clahe_img.shape == dummy_color_image.shape
    assert clahe_img.dtype == np.uint8


def test_deskewing_reconstructed_shape(skewed_image: np.ndarray) -> None:
    """Vérifie que l'algorithme de deskewing préserve les dimensions et le type de l'image."""
    pipeline = PreprocessingPipeline()
    deskewed = pipeline.deskew(skewed_image)
    
    assert deskewed.shape == skewed_image.shape
    assert deskewed.dtype == np.uint8


def test_segmenter_output_format(dummy_color_image: np.ndarray) -> None:
    """Vérifie que le segmentateur morphologique de secours produit des structures valides."""
    pipeline = PreprocessingPipeline()
    binary = pipeline.apply_sauvola(dummy_color_image)
    
    segmenter = PageSegmenter(use_fallback=True)
    lines = segmenter.segment_lines(binary, dummy_color_image)
    
    # Doit détecter au moins quelques lignes
    assert isinstance(lines, list)
    assert len(lines) > 0
    
    # Vérifier le schéma de structure de chaque ligne
    for line in lines:
        assert "line_id" in line
        assert "polygon" in line
        assert "baseline" in line
        assert "reading_order" in line
        
        # Le polygone doit contenir les coordonnées des sommets de la boîte
        assert len(line["polygon"]) == 4
        for point in line["polygon"]:
            assert len(point) == 2
            assert isinstance(point[0], (int, float))
            assert isinstance(point[1], (int, float))
