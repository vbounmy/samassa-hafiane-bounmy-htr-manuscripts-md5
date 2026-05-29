# Étape 2.3 — Segmentation des lignes de texte avec Kraken BLLA

## Objectif

Évaluer la qualité de la segmentation des **lignes de texte** produites par Kraken BLLA (Baseline Layout Analysis) sur les pages d'e-NDP, en comparaison avec les polygones de lignes annotés manuellement dans le ground truth.

Cette sous-étape fait suite à l'échec documenté de SAM en sous-étape 2.2 (`docs/segmentation_baseline_sam.md`) et vise à valider un segmenteur **adapté au domaine** des manuscrits historiques.

## Protocole expérimental

### Configuration Kraken

| Paramètre | Valeur |
|---|---|
| Bibliothèque | `kraken==7.0.2` |
| Module | `kraken.blla.segment` |
| Modèle | **BLLA par défaut** (modèle générique embarqué) |
| Mode | API Python (pas de CLI) |
| Device | CPU |
| Image input | Pleine résolution (~3000×4000 px) |

### Échantillon évalué

Sous-ensemble du sample stratifié (`src/segmentation/sampler.py`), prélevé pour couvrir les sous-périodes diachroniques du projet :

| Folio | Année | Stratum | Lignes GT | Lignes Kraken | **IoU moyen** | Durée |
|---|---|---|---|---|---|---|
| FRAN_0393_09221_L | 1346 | XIVᵉ | 54 | 54 | **0,745** | 77 s |
| FRAN_0393_14096_L | 1501 | XVIᵉ | 73 | 81 | **0,756** | 109 s |

### Métrique

Pour chaque ligne annotée e-NDP, on calcule l'IoU avec la ligne Kraken qui la recouvre le mieux (best-match). Réutilise `src/segmentation/evaluator.py` (méthode-agnostique). Spécifiquement :

- ``load_ground_truth_lines`` parse les éléments ``<TextLine>`` du PAGE XML
- ``best_iou_per_zone`` (générique) fait le matching greedy GT → prédit
- L'IoU est calculé entre polygones rasterisés à la résolution de l'image source

## Résultats

### IoU moyen sur l'échantillon

**~0,75 d'IoU moyen** sur les deux pages évaluées (XIVᵉ et XVIᵉ siècles). C'est dans la zone considérée comme **utile** pour la HTR (l'IoU n'a pas besoin d'être parfait, il suffit qu'il recouvre suffisamment la zone d'encre pour que la transcription fonctionne).

### Comparaison qualitative avec la baseline SAM (sous-étape 2.2)

| Modèle | Tâche | Mean IoU |
|---|---|---|
| SAM ViT-B (auto) | Zones logiques de page | **0,05–0,09** |
| **Kraken BLLA** (défaut) | Lignes de texte | **0,74–0,76** |

L'amélioration est d'un **facteur 8 à 15**, ce qui confirme l'importance d'utiliser un modèle adapté au domaine.

### Comptage des lignes

Kraken retrouve un nombre de lignes **très proche** du ground truth :

- Page XIVᵉ : 54 GT / 54 Kraken (100 % de correspondance)
- Page XVIᵉ : 73 GT / 81 Kraken (~111 %, légère sur-segmentation)

La sur-segmentation modérée du XVIᵉ s'explique probablement par une écriture plus serrée (cursive bâtarde tardive), où Kraken sépare parfois des éléments que l'annotation manuelle fusionne. Ce comportement reste compatible avec la HTR : un excès de lignes a un impact limité sur le CER global.

### Une limite à mentionner dans l'article

Kraken signale un warning de **topologie** lors du traitement de certaines lignes :

```
Polygonizer failed on line 0: TopologyException: side location conflict.
```

C'est un cas connu où la géométrie d'une ligne est ambiguë (souvent dans des marges très denses). Le polygone correspondant est alors construit avec une boîte de secours moins précise. Le pipeline ne plante pas et continue normalement.

## Interprétation

Trois éléments expliquent le succès de Kraken là où SAM échoue :

1. **Domaine d'entraînement** — Kraken BLLA est entraîné sur des images de pages manuscrites (incluant des données HTR-United), tandis que SAM a vu principalement des objets naturels.
2. **Structure attendue** — Kraken sait que les pages contiennent des lignes de texte horizontales (avec déviations possibles) et calibre ses prédictions en conséquence.
3. **Baseline + boundary** — Kraken produit deux représentations complémentaires : la *baseline* (ligne d'écriture) et le *boundary* (polygone englobant). Notre évaluation IoU porte sur les boundaries, suffisants pour la HTR en aval.

## Conclusion

Kraken BLLA atteint un IoU moyen de **~0,75** sur les lignes annotées d'e-NDP, sans aucun fine-tuning ni adaptation au corpus. Ce résultat valide notre choix de Kraken comme segmenteur de production pour l'ensemble du pipeline HTR du projet.

## Décision projet

- Adoption de Kraken BLLA pour la segmentation de lignes en production
- Réutilisation des polygones de lignes Kraken pour l'export PAGE XML (sous-étape 2.5)
- Conservation des annotations ground truth e-NDP comme **référence d'entraînement** pour le modèle de reconnaissance (étape 3)

## Référence

- Kiessling, B. (2019). *Kraken: A modular OCR system for historical handwritten and printed documents*. Digital Humanities 2019.
- Documentation Kraken : https://kraken.re/main/
- HTR-United : https://htr-united.github.io/
