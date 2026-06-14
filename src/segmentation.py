"""Module de segmentation de la structure de page (layout) et d'extraction de lignes.

Ce module implémente l'étape 3 et l'exigence n°6 du sujet :
1. Séparer les régions de texte, illustrations et marges.
2. Segmenter les lignes de texte sous forme de polygones ou boîtes orientées.
3. Assurer un ordre de lecture logique (gestion des pages multi-colonnes).
4. Fournir une solution modulaire : utilisation de Kraken BLLA si disponible,
   avec un fallback robuste par traitement d'images (morphologie mathématique) sous Windows.

Example:
    >>> from src.segmentation import PageSegmenter
    >>> segmenter = PageSegmenter(use_fallback=True)
    >>> lines = segmenter.segment_lines(binary_image, original_image)
    >>> # Chaque ligne contient son ID, son polygone, son ordre de lecture, etc.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class PageSegmenter:
    """Segmentateur de structure de page pour manuscrits anciens.

    Cette classe gère la détection des blocs de texte et l'extraction
    des polygones fins pour chaque ligne de texte, avec gestion de l'ordre de lecture.
    """

    def __init__(self, use_fallback: bool = True) -> None:
        """Initialise le segmentateur.

        Args:
            use_fallback (bool): Si True, autorise l'utilisation de l'algorithme
                morphologique en cas d'absence de Kraken. Defaults to True.
        """
        self.use_fallback = use_fallback
        self.has_kraken = False
        
        try:
            import kraken
            from kraken.lib import models
            self.has_kraken = True
            logger.info("[+] Kraken est disponible et sera utilisé pour la segmentation BLLA.")
        except ImportError:
            logger.warning("[-] Kraken n'est pas disponible. Utilisation du fallback morphologique sous Windows.")

    def segment_lines_kraken(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Segmente les lignes de l'image en utilisant Kraken BLLA.

        Args:
            image (np.ndarray): Image source en couleur ou noir et blanc.

        Returns:
            List[Dict[str, Any]]: Liste des lignes de texte segmentées.
        """
        if not self.has_kraken:
            raise RuntimeError("Kraken n'est pas installé dans cet environnement.")

        from PIL import Image
        from kraken import binarization
        from kraken.pageseg import segment

        # Convertir en image PIL
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Binariser avec le binariseur par défaut de Kraken
        binary_pil = binarization.nlbin(pil_img)
        
        # Lancer la segmentation de lignes (BLLA)
        res = segment(binary_pil)
        
        lines_data = []
        for i, line in enumerate(res.lines):
            # line.boundary contient le polygone (liste de points (x, y))
            # line.baseline contient la ligne de base
            lines_data.append({
                "line_id": f"kraken_line_{i:04d}",
                "polygon": line.boundary,
                "baseline": line.baseline,
                "reading_order": i
            })
            
        return lines_data

    def detect_columns(self, binary_image: np.ndarray) -> List[Tuple[int, int]]:
        """Détecte les colonnes de texte sur la page via le profil de projection verticale.

        Args:
            binary_image (np.ndarray): Image binarisée (0=texte, 255=fond).

        Returns:
            List[Tuple[int, int]]: Liste des plages de colonnes [(col1_start, col1_end), ...].
        """
        h, w = binary_image.shape
        # Inverser pour avoir le texte en blanc (1) et le fond en noir (0)
        inv_binary = (binary_image == 0).astype(np.uint8)
        
        # Somme des pixels sur l'axe vertical (profil de projection)
        vertical_projection = np.sum(inv_binary, axis=0)
        
        # Trouver les vallées (zones sans texte / espaces inter-colonnes)
        # Seuil adaptatif : une colonne doit avoir une certaine densité de texte
        threshold = np.max(vertical_projection) * 0.05
        
        text_cols = vertical_projection > threshold
        
        columns = []
        in_col = False
        start_idx = 0
        
        for idx, is_text in enumerate(text_cols):
            if is_text and not in_col:
                start_idx = idx
                in_col = True
            elif not is_text and in_col:
                # Filtrer les fausses colonnes trop étroites (bruit de bordure)
                if idx - start_idx > w * 0.1:
                    columns.append((start_idx, idx))
                in_col = False
                
        if in_col and (w - start_idx > w * 0.1):
            columns.append((start_idx, w))
            
        # Si aucune colonne nette n'est détectée, on considère la page entière comme une colonne
        if not columns:
            columns = [(0, w)]
            
        return columns

    def segment_lines_morphological(self, binary_image: np.ndarray, original_image: np.ndarray) -> List[Dict[str, Any]]:
        """Segmente les lignes de texte à l'aide de morphologie mathématique (solution de secours robuste).

        Cette méthode implémente un pipeline OpenCV :
        1. Inversion d'image.
        2. Dilation horizontale pour fusionner les caractères et mots en lignes d'un seul tenant.
        3. Détection des colonnes et tri selon l'ordre de lecture gauche-droite haut-bas.
        4. Détection des contours et extraction des polygones d'enveloppe.

        Args:
            binary_image (np.ndarray): Image binarisée (0=texte, 255=fond).
            original_image (np.ndarray): Image couleur originale (pour dimensions et debbuging).

        Returns:
            List[Dict[str, Any]]: Liste des lignes de texte segmentées avec leurs polygones.
        """
        h, w = binary_image.shape
        # Inverser pour avoir le texte en blanc (255)
        inv_binary = cv2.bitwise_not(binary_image)
        
        # Noyau rectangulaire large horizontalement pour fusionner les lignes de texte
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 4))
        dilated = cv2.dilate(inv_binary, kernel, iterations=2)
        
        # Trouver les contours des lignes dilatées
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Détection des colonnes
        columns = self.detect_columns(binary_image)
        
        lines_by_col: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(columns))}
        
        line_count = 0
        for contour in contours:
            # Filtrer les bruits (surfaces trop petites)
            area = cv2.contourArea(contour)
            if area < (h * w * 0.0001):  # < 0.01% de la page
                continue
                
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Éliminer les marges/bordures de page qui font toute la hauteur
            if ch > h * 0.8:
                continue
                
            # Calculer le polygone fin (boîte englobante orientée pour la robustesse à l'inclinaison)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Convertir le polygone en liste de points [[x1, y1], [x2, y2], ...]
            polygon = box.tolist()
            
            # Déterminer à quelle colonne appartient la ligne
            center_x = x + cw // 2
            col_assigned = 0
            for col_idx, (c_start, c_end) in enumerate(columns):
                if c_start <= center_x <= c_end:
                    col_assigned = col_idx
                    break
                    
            lines_by_col[col_assigned].append({
                "contour": contour,
                "polygon": polygon,
                "bbox": (x, y, cw, ch),
                "center_y": y + ch // 2
            })
            
        # Trier les colonnes de gauche à droite, puis les lignes du haut vers le bas
        sorted_lines: List[Dict[str, Any]] = []
        global_order = 0
        
        for col_idx in sorted(lines_by_col.keys()):
            # Trier les lignes au sein de cette colonne par coordonnée Y (haut vers bas)
            col_lines = sorted(lines_by_col[col_idx], key=lambda item: item["center_y"])
            for line in col_lines:
                # Créer une ligne de base approximative (bas du rectangle)
                x, y, cw, ch = line["bbox"]
                baseline = [[x, y + ch], [x + cw, y + ch]]
                
                sorted_lines.append({
                    "line_id": f"line_{global_order:04d}",
                    "polygon": line["polygon"],
                    "baseline": baseline,
                    "reading_order": global_order,
                    "column": col_idx
                })
                global_order += 1
                
        return sorted_lines

    def segment_lines(self, binary_image: np.ndarray, original_image: np.ndarray) -> List[Dict[str, Any]]:
        """Méthode principale de segmentation qui choisit automatiquement l'algorithme.

        Args:
            binary_image (np.ndarray): Image binarisée de la page.
            original_image (np.ndarray): Image couleur originale de la page.

        Returns:
            List[Dict[str, Any]]: Liste des lignes avec coordonnées de polygones,
                lignes de base et ordonnancement de lecture.
        """
        if self.has_kraken and not self.use_fallback:
            try:
                return self.segment_lines_kraken(original_image)
            except Exception as e:
                logger.error(f"[-] Erreur Kraken BLLA, bascule sur le fallback : {e}")
                
        return self.segment_lines_morphological(binary_image, original_image)
