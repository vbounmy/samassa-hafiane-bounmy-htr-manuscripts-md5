# CONVENTIONS_TRANSCRIPTION.md

Ce document spécifie les conventions éditoriales adoptées pour la constitution du corpus d'entraînement du projet **htr-medieval-manuscripts-2026**, conformément à la contrainte n°4 du brief (« un fichier `CONVENTIONS_TRANSCRIPTION.md` décrit les choix éditoriaux »).

Les conventions sont **héritées des corpus sources** et non imposées par l'équipe : nous avons choisi de conserver les conventions originales de chaque jeu de données, plutôt que de les harmoniser artificiellement. Ce document décrit donc à la fois ce que font les corpus retenus et la manière dont le projet gère leur cohabitation.

---

## 1. Niveau de transcription

| Corpus          | Niveau retenu              | Référence méthodologique                                                                                      |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| CREMMA Medieval | Diplomatique (graphémique) | Pinche, _Guide de transcription pour les manuscrits du Xᵉ au XVᵉ siècle_ (HAL 03697382), suivant D. Stutzmann |
| e-NDP           | Semi-diplomatique          | `transcription_guidelines.pdf` joint à la publication Zenodo                                                  |
| CREMMA MSS 16   | Diplomatique (graphémique) | Identique à CREMMA Medieval                                                                                   |

**Définitions retenues.**

- **Diplomatique (graphémique).** Reproduit le manuscrit signe pour signe : un signe dans l'image correspond à un signe dans le texte. Les abréviations sont conservées, les graphies anciennes (u/v, i/j non distingués) sont préservées.
- **Semi-diplomatique.** Préserve la graphie originale mais résout les abréviations et applique certaines régularisations (capitalisation des entités nommées, distinction u/v et i/j).

## 2. Traitement des abréviations

### CREMMA Medieval et CREMMA MSS 16

Abréviations **conservées** dans leur forme originale.

| Exemple               | Conservé sous la forme  |
| --------------------- | ----------------------- |
| Tilde de nasalisation | `tᷣ` (au lieu de `ter`) |
| Lettrines suspendues  | `ꝓ`, `ꝑ`, `⁊`           |
| Lettres en exposant   | `M^me`, `dud^t`         |

L'encodage suit la norme **MUFI** (Medieval Unicode Font Initiative), qui utilise la zone _Private Use Area_ d'Unicode pour les caractères propres aux écritures médiévales. Ces caractères apparaissent dans le corpus sous des codepoints comme `U+F158`, `U+F1A6`, etc. Voir https://mufi.info/ pour la documentation complète.

### e-NDP

Abréviations **résolues**, selon trois mécanismes :

| Type                  | Exemple original | Forme résolue |
| --------------------- | ---------------- | ------------- |
| Suspension            | `facimꝰ`         | `facimus`     |
| Contraction           | `dñi`            | `domini`      |
| Signes conventionnels | `⁊`              | `et`          |
| Signes conventionnels | `ꝓ`              | `pro`         |

### Conséquence pour le pipeline

Le modèle HTR doit apprendre **deux comportements** distincts : reconnaître l'abréviation telle qu'elle est dessinée (CREMMA / MSS-16), ou la lire et produire sa forme développée (e-NDP). Cette hétérogénéité est documentée dans le JSON livrable au Volet 2 par un champ `transcription_convention` au niveau de chaque ligne :

- `"diplomatic"` pour les lignes issues de CREMMA Medieval et CREMMA MSS 16
- `"semi-diplomatic"` pour les lignes issues d'e-NDP

## 3. Distinction u/v et i/j

| Corpus          | u/v                                        | i/j                                        |
| --------------- | ------------------------------------------ | ------------------------------------------ |
| CREMMA Medieval | **Non distingués** (u = u et v)            | **Non distingués** (i = i et j)            |
| e-NDP           | **Distingués** (`u` voyelle, `v` consonne) | **Distingués** (`i` voyelle, `j` consonne) |
| CREMMA MSS 16   | À vérifier (probablement non distingués)   | À vérifier                                 |

Cette divergence est traitée de la même manière que les abréviations : pas d'harmonisation, marquage par `transcription_convention`.

## 4. Casse et entités nommées

| Corpus          | Casse                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| CREMMA Medieval | Casse manuscrite respectée                                                                                               |
| e-NDP           | Entités nommées (personnes, lieux, institutions) **systématiquement capitalisées**, plus capitales originales du notaire |
| CREMMA MSS 16   | Casse manuscrite respectée                                                                                               |

## 5. Ponctuation

| Corpus          | Ponctuation                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------- |
| CREMMA Medieval | Ponctuation manuscrite uniquement                                                                  |
| e-NDP           | Marques `.` et `/` du manuscrit transcrites telles quelles, **aucune ponctuation moderne ajoutée** |
| CREMMA MSS 16   | Ponctuation manuscrite uniquement                                                                  |

## 6. Traitement des espaces

**Point critique.** La documentation officielle de CREMMA Medieval mentionne explicitement :

> _« The spaces in the dataset are not homogeneously represented, sometimes transcriptions reproduce the manuscript spacing while others use lexical spaces. It must be stressed that spaces are the most important source of error in medieval HTR models. »_

Notre pipeline ne tente pas de normaliser les espaces, mais documente ce fait comme une **limitation connue** du WER (Word Error Rate). Le CER global, qui est notre métrique principale, reste robuste à cette variation, ce qui justifie son choix dans le brief.

## 7. Corrections, biffures et passages illisibles

| Corpus          | Marquage                                                                     |
| --------------- | ---------------------------------------------------------------------------- |
| CREMMA Medieval | Pas de convention formelle (lacunes laissées vides ou notées au cas par cas) |
| e-NDP           | Texte barré ou corrigé encadré par `$` : `$mot biffé$`                       |
| CREMMA MSS 16   | Pas de convention formelle                                                   |

**Décision projet.** Les lignes contenant un marqueur de correction (`$...$`) ou présentant une lacune sont automatiquement marquées `needs_review = true` dans le JSON de sortie, conformément à l'exigence n°5 du brief.

## 8. Layout et segmentation

Les trois corpus suivent l'ontologie **SegmOnto** pour le marquage des zones de page :

| Type de zone       | Définition                 |
| ------------------ | -------------------------- |
| `MainZone`         | Bloc de texte principal    |
| `MarginTextZone`   | Notes marginales           |
| `NumberingZone`    | Pagination, foliotation    |
| `DropCapitalZone`  | Lettrines décorées         |
| `GraphicZone`      | Illustrations, enluminures |
| `RunningTitleZone` | Titre courant              |
| `TableZone`        | Tableaux                   |

Documentation officielle : https://github.com/SegmOnto

e-NDP utilise par ailleurs sa propre **typologie à 5 sections** spécifique aux registres capitulaires (`block`, `liste`, `date`, `entrée`, `numérotation`). Cette taxonomie spécifique est traitée comme une **extension** de SegmOnto et conservée telle quelle pour les lignes issues de ce corpus.

## 9. Caractères MUFI et alphabet effectif du modèle

Le vocabulaire de caractères présent dans le corpus est large (~190 codepoints distincts pour CREMMA Medieval seul). Il comprend :

- Les **lettres latines de base** (`a` à `z`, `A` à `Z`)
- Les **chiffres** (`0` à `9`)
- La **ponctuation manuscrite** (`.`, `/`, `,`, `:`, `;`, etc.)
- Les **caractères MUFI** dans la Private Use Area (`U+F100` à `U+F8FF`)
- Les **diacritiques combinants** (`U+0300` à `U+036F`)
- Le **s long** (`ſ`, `U+017F`)
- Les **tildes de nasalisation** et autres marques d'abréviation

L'alphabet exact utilisé pour entraîner le modèle Kraken est fourni dans `experiments/alphabet.txt`.

## 10. Synthèse pour le Volet 2 (NLP)

Le JSON livrable au module NLP contient, pour chaque ligne transcrite :

```json
{
  "line_id": "eSc_line_2d569888",
  "manuscript": "bnf_fr_24428-bestiaire",
  "page": "btv1b55005...f128.jpg",
  "content": "D ont se monstra diex a no gent",
  "transcription_convention": "diplomatic",
  "language": "fro",
  "polygon": [[357, 465], ...],
  "needs_review": false
}
```

Les champs `transcription_convention` et `language` permettent au pipeline NLP du Volet 2 d'adapter ses traitements (normalisation des abréviations, lemmatisation, etc.) en fonction de la provenance de chaque ligne.

---

## Références bibliographiques

- Pinche, A. (2022). _Guide de transcription pour les manuscrits du Xᵉ au XVᵉ siècle_. HAL Archives Ouvertes. https://hal.archives-ouvertes.fr/hal-03697382
- Stutzmann, D. (2011). « Paléographie statistique pour décrire, identifier, dater… Normaliser pour coopérer et aller plus loin ? » In F. Fischer et al. (dir.), _Kodikologie und Paläographie im digitalen Zeitalter 2_. BoD.
- Medieval Unicode Font Initiative. https://mufi.info/
- SegmOnto Ontology. https://github.com/SegmOnto
- Claustre, J., Smith, D., Torres Aguilar, S. et al. (2023). _e-NDP transcription guidelines._ In Zenodo dataset 10.5281/zenodo.7575693.

---
