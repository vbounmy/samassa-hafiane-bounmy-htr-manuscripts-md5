# Étape 2.2 — Segmentation de structure de page : baseline SAM

## Objectif

Évaluer la capacité du modèle **Segment Anything (SAM ViT-B, Meta AI)** en mode *automatic mask generator* à retrouver les **zones logiques** annotées dans le corpus e-NDP (typologie SegmOnto étendue : `block`, `liste`, `Date`, `entrée`, `numérotation`).

## Protocole expérimental

### Échantillon évalué

L'évaluation porte sur un sous-ensemble du **sample stratifié** de 24 pages (voir `src/segmentation/sampler.py`), prélevé proportionnellement aux quatre sous-périodes de l'axe diachronique du projet :

| Stratum | Période | Pages évaluées |
|---|---|---|
| XIVᵉ | 1326-1399 | 1 (FRAN_0393_09221_L, 1346) |
| XVᵉ début | 1400-1449 | 1 (FRAN_0393_00677_L, 1401) |
| XVᵉ fin | 1450-1499 | 1 (FRAN_0393_05188_L, 1450) |
| XVIᵉ | 1500-1504 | 1 (FRAN_0393_14096_L, 1501) |

### Configuration SAM

| Paramètre | Valeur |
|---|---|
| Variante | SAM ViT-B (Meta, 358 Mo) |
| Mode | `SamAutomaticMaskGenerator` |
| `points_per_side` | 16 |
| `pred_iou_thresh` | 0.86 |
| `stability_score_thresh` | 0.92 |
| `min_mask_region_area` | 2 000 px |
| Downscale image | 4× (vitesse) |
| Device | CPU |

### Métrique

Pour chaque zone annotée e-NDP, on calcule l'**IoU maximal** entre cette zone et l'ensemble des masques générés par SAM, selon :

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Implémentation : `src/segmentation/evaluator.py`.

## Résultats

### IoU moyen par page

| Folio | Année | Stratum | Zones GT | Masques SAM | **IoU moyen** |
|---|---|---|---|---|---|
| FRAN_0393_09221_L | 1346 | XIV | 10 | 9 | **0,049** |
| FRAN_0393_14096_L | 1501 | XVI | 7 | 13 | **0,092** |

### Lecture qualitative (page 0016, exemple représentatif)

Sur la page d'exemple `FRAN_0393_00016_L.jpg` (visualisation `sam_automatic_demo.jpg`), SAM produit **18 masques** dont :

- 1 masque englobant **toute la page** (10,1 M pixels, ~85 % de l'image)
- 1 masque correspondant à **la reliure rouge** du livre (390 k pixels)
- 16 masques sur des **fragments graphiques arbitraires** (mots isolés, taches, paragraphes partiels)

**Aucun** des masques générés ne recouvre fidèlement les régions logiques annotées (`block` principal, `liste` des chanoines, zones `Date`, `numérotation`).

## Interprétation

Trois facteurs expliquent cet échec :

1. **Domain gap** — SAM est pré-entraîné sur des images naturelles (objets, scènes, animaux) et non sur des documents historiques structurés. Sa notion de « segment » reflète la salience visuelle (contraste, contours nets), pas la fonction documentaire.
2. **Granularité incorrecte** — SAM segmente au niveau du *visuellement saillant* : il sépare un mot d'une tache, mais ne sait pas qu'un paragraphe entier est une unité significative pour un paléographe.
3. **Bruit du parchemin** — Les variations de coloration et les taches du parchemin médiéval créent des "objets" parasites que SAM segmente, augmentant le nombre de faux positifs.

## Conclusion

Avec un IoU moyen de **0,05–0,10**, SAM en mode *automatic* est inopérant pour la segmentation de structure de page sur les manuscrits e-NDP. Ce résultat n'est ni surprenant ni anormal : il confirme la nécessité d'un segmenteur **adapté au domaine** des documents historiques.

## Décision projet

Nous abandonnons SAM comme segmenteur de page et basculons vers **Kraken** (sous-étape 2.3), dont le module `blla` (Baseline Layout Analysis) est spécifiquement entraîné sur des documents manuscrits historiques.

Le code SAM (`src/segmentation/sam_wrapper.py`) et l'évaluateur IoU (`src/segmentation/evaluator.py`) sont conservés dans le dépôt pour :

- traçabilité de la démarche scientifique ;
- réutilisation de l'évaluateur IoU sur les sorties Kraken (le calcul d'IoU est indépendant du segmenteur source).

## Référence

- Kirillov, A., Mintun, E., Ravi, N., et al. (2023). *Segment Anything*. arXiv:2304.02643.
- Meta AI Research. *segment-anything* repository, https://github.com/facebookresearch/segment-anything.
