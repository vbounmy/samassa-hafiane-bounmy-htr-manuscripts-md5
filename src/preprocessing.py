"""Module de prétraitement d'images pour le pipeline HTR.

Ce module implémente les étapes de prétraitement documentaire requises pour
le projet HTR (Contrainte n°1 et Étape 2 du sujet) :
1. Correction d'inclinaison (deskew) via la transformée de Hough.
2. Amélioration de contraste via CLAHE (Contrast Limited Adaptive Histogram Equalization).
3. Binarisation adaptative (Sauvola).
4. Gestion des zones enluminées et lettrines.

Le pipeline est paramétrable et reproductible grâce à l'utilisation de seeds fixés.

Example:
    >>> import cv2
    >>> from src.preprocessing import PreprocessingPipeline
    >>> pipeline = PreprocessingPipeline(clip_limit=2.0, tile_grid_size=(8, 8), sauvola_window=25, sauvola_k=0.2)
    >>> image = cv2.imread("data/page.jpg")
    >>> preprocessed_img = pipeline.run(image)
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from typing import Tuple, Dict, Any, Union


class PreprocessingPipeline:
    """Chaîne de prétraitement d'images de manuscrits anciens.

    Cette classe regroupe les fonctions nécessaires pour nettoyer, redresser
    et binariser les scans bruts de manuscrits avant la segmentation et la transcription.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        sauvola_window: int = 25,
        sauvola_k: float = 0.2,
        seed: int = 42
    ) -> None:
        """Initialise la classe PreprocessingPipeline avec ses paramètres.

        Args:
            clip_limit (float): Limite de contraste pour l'égalisation CLAHE. Defaults to 2.0.
            tile_grid_size (Tuple[int, int]): Taille de la grille pour CLAHE. Defaults to (8, 8).
            sauvola_window (int): Taille de la fenêtre locale pour Sauvola (doit être un nombre impair). Defaults to 25.
            sauvola_k (float): Paramètre k de la binarisation de Sauvola. Defaults to 0.2.
            seed (int): Seed global pour la reproductibilité. Defaults to 42.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.sauvola_window = sauvola_window
        self.sauvola_k = sauvola_k
        self.seed = seed
        
        # Fixer la seed pour OpenCV et Numpy
        np.random.seed(self.seed)
        cv2.setRNGSeed(self.seed)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Corrige l'inclinaison de l'image (deskewing) à l'aide de la transformée de Hough.

        Cette méthode détecte l'angle d'inclinaison général des lignes de texte du manuscrit
        et applique une rotation inverse pour redresser le texte horizontalement.

        Args:
            image (np.ndarray): Image source en couleur (BGR) ou en niveaux de gris.

        Returns:
            np.ndarray: Image redressée de la même forme que l'image d'entrée.
        """
        # Conversion en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Floutage pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Détection des contours de Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Transformée de Hough pour trouver les lignes
        # On cherche à détecter des lignes horizontales proches (angles proches de 0 ou 180 degrés)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None:
            # Si aucune ligne n'est détectée, on retourne l'image originale
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            # On ne garde que les angles proches de l'horizontale (-45 à 45 degrés)
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return image

        # L'angle de rotation est la médiane des angles détectés
        median_angle = np.median(angles)

        # Si l'angle est trop faible, inutile de tourner l'image
        if abs(median_angle) < 0.1:
            return image

        # Effectuer la rotation autour du centre de l'image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Remplir les bords créés par la rotation avec du blanc (ou la couleur majoritaire)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )
        return rotated

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Applique l'égalisation adaptative de contraste CLAHE.

        Si l'image est en couleur BGR, la méthode convertit l'image dans l'espace LAB,
        applique CLAHE sur le canal L (Luminosité) pour éviter les distorsions de couleur,
        puis repasse en BGR. Si elle est en niveaux de gris, CLAHE est appliqué directement.

        Args:
            image (np.ndarray): Image source en couleur (BGR) ou niveaux de gris.

        Returns:
            np.ndarray: Image améliorée.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if len(image.shape) == 3:
            # Image couleur : traitement dans le domaine LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            # Image en niveaux de gris
            return clahe.apply(image)

    def apply_sauvola(self, image: np.ndarray) -> np.ndarray:
        """Applique la binarisation adaptative locale de Sauvola.

        Cette méthode est idéale pour les manuscrits anciens car elle s'adapte aux
        taches, dégradations de papier et variations d'encre locales.

        Args:
            image (np.ndarray): Image source (convertie en niveaux de gris si couleur).

        Returns:
            np.ndarray: Image binarisée (0 pour le texte, 255 pour le fond).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Calculer le seuil adaptatif de Sauvola
        thresh = threshold_sauvola(gray, window_size=self.sauvola_window, k=self.sauvola_k)
        
        # Binariser : True (1) pour le fond, False (0) pour le texte
        binary = gray > thresh
        
        # Convertir en format OpenCV standard (0 pour le noir/texte, 255 pour le blanc/fond)
        binary_image = np.zeros_like(gray, dtype=np.uint8)
        binary_image[binary] = 255  # Fond blanc
        binary_image[~binary] = 0   # Texte noir
        
        return binary_image

    def handle_illuminations(self, image: np.ndarray) -> np.ndarray:
        """Détecte et atténue le bruit des lettrines enluminées pour préserver la binarisation.

        Applique un filtre bilatéral pour lisser le fond coloré ou texturé tout en
        préservant les contours des caractères d'encre sombres.

        Args:
            image (np.ndarray): Image source BGR ou niveaux de gris.

        Returns:
            np.ndarray: Image filtrée.
        """
        # Un filtrage bilatéral est excellent pour lisser l'arrière-plan sans flouter les bords
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def run(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Exécute l'intégralité du pipeline de prétraitement.

        Args:
            image (np.ndarray): Scan brut du manuscrit (BGR).

        Returns:
            Dict[str, np.ndarray]: Dictionnaire contenant les images intermédiaires et finales :
                - "original": Image source brute.
                - "deskewed": Image après correction d'inclinaison.
                - "denoised": Image après traitement des enluminures/filtrage.
                - "clahe": Image couleur après égalisation locale.
                - "binary": Image binarisée finale (Sauvola, 0=texte, 255=fond).
        """
        # 1. Correction de l'inclinaison
        deskewed = self.deskew(image)

        # 2. Gestion des lettrines et filtrage du bruit
        denoised = self.handle_illuminations(deskewed)

        # 3. Amélioration du contraste (CLAHE)
        clahe_img = self.apply_clahe(denoised)

        # 4. Binarisation locale (Sauvola)
        binary_img = self.apply_sauvola(clahe_img)

        return {
            "original": image,
            "deskewed": deskewed,
            "denoised": denoised,
            "clahe": clahe_img,
            "binary": binary_img
        }
