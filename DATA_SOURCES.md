# DATA_SOURCES.md

Ce document recense l'ensemble des jeux de données utilisés pour entraîner et évaluer le pipeline HTR du projet **htr-medieval-manuscripts-2026**, dans le cadre du module *Vision par ordinateur* (Master Data/IA, HETIC, 2026).

Conformément à la contrainte n°7 du brief de projet, seuls des corpus diffusés sous licence libre (CC-BY, CC-BY-SA ou domaine public) sont utilisés. Chaque source est documentée ci-dessous avec sa licence, sa citation académique et son URL d'origine.

---

## 1. CREMMA Medieval

**Description.** Corpus de transcriptions de manuscrits médiévaux en ancien français, produit dans le cadre du projet CREMMA (Consortium pour la Reconnaissance d'Écritures Manuscrites des Matériaux Anciens) porté par l'École nationale des chartes (PSL) et financé par le DIM MAP (Région Île-de-France). Le corpus rassemble 14 manuscrits littéraires (chansons de geste, hagiographies, bestiaires, romans) couvrant les XIIᵉ–XVᵉ siècles.

| Métadonnée | Valeur |
|---|---|
| Période | 1100–1499 (XIIᵉ–XVᵉ siècle) |
| Langue(s) | Ancien français (`fro`), quelques passages en français (`fra`) |
| Type de documents | Textes littéraires médiévaux |
| Volume | 22 848 lignes / 612 000 caractères / 14 manuscrits / 279 pages |
| Convention de transcription | Diplomatique, abréviations conservées (suit Pinche, *Guide de transcription*, 2022) |
| Licence | **CC-BY 4.0** |
| Dépôt | https://github.com/HTR-United/cremma-medieval |
| DOI | [10.5281/zenodo.5235185](https://doi.org/10.5281/zenodo.5235185) |

**Citation (APA).**
Pinche, A. (2022). *Cremma Medieval* (Version Bicerin 1.1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.5235185

**Citation (BibTeX).**
```bibtex
@dataset{pinche_cremma_medieval_2022,
  author       = {Pinche, Ariane},
  title        = {Cremma Medieval},
  year         = {2022},
  publisher    = {Zenodo},
  version      = {Bicerin 1.1.0},
  doi          = {10.5281/zenodo.5235185},
  url          = {https://github.com/HTR-United/cremma-medieval}
}
```

---

## 2. ANR e-NDP Ground Truth

**Description.** Vérité terrain HTR issue du projet ANR e-NDP (« Notre-Dame de Paris et son cloître »), édition numérique collaborative des registres capitulaires de Notre-Dame de Paris. Porté par le LaMOP (Université Paris 1 Panthéon-Sorbonne) sous la direction de Julie Claustre et Darwin Smith, en partenariat avec les Archives nationales, la BnF, l'École nationale des chartes et la Bibliothèque Mazarine. Le corpus rassemble 512 pages représentatives des 26 registres du chapitre médiéval (cotes AN LL 105-128).

| Métadonnée | Valeur |
|---|---|
| Période | 1326–1504 (XIVᵉ–XVIᵉ siècle) |
| Langue(s) | Latin médiéval (≥ 98 %), français (≤ 2 %) |
| Type de documents | Registres administratifs (délibérations capitulaires) |
| Volume | 34 231 lignes / 3 320 407 caractères / 512 pages / 18 mains principales |
| Convention de transcription | Semi-diplomatique, abréviations résolues (`facimꝰ` → `facimus`) |
| Licence | **CC-BY 4.0** |
| Dépôt Zenodo | https://zenodo.org/records/7575693 |
| DOI | [10.5281/zenodo.7575693](https://doi.org/10.5281/zenodo.7575693) |
| Code des modèles | https://github.com/chartes/e-NDP_HTR |

**Justification d'inclusion malgré la dominance latine.** Le brief impose un périmètre « XIᵉ–XVIIᵉ siècle, vieux/moyen français ». L'inclusion d'e-NDP, dont le contenu est majoritairement latin, se justifie par la cohérence paléographique du corpus avec le reste du matériel retenu : le latin médiéval écrit en France et l'ancien français partagent la même tradition graphique (cursives, ligatures, systèmes d'abréviation). Un modèle HTR opère sur les formes visuelles des glyphes, indépendamment de la langue sous-jacente. Cette approche est suivie par des projets de référence tels que CREMMA-Medieval-LAT et CATMuS Medieval.

**Citation (APA).**
Claustre, J., Smith, D., Torres Aguilar, S., Bretthauer, I., Brochard, P., Canteaut, O., Cottereau, E., Delivré, F., Denglos, M., Jolivet, V., Julerot, V., Kouamé, T., Lusset, E., Massoni, A., Nadiras, S., Perreaux, N., Regazzi, H., & Treglia, M. (2023). *The e-NDP project: collaborative digital edition of the Chapter registers of Notre-Dame of Paris (1326-1504). Ground-truth for handwriting text recognition (HTR) on late medieval manuscripts* (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.7575693

**Citation (BibTeX).**
```bibtex
@dataset{claustre_endp_2023,
  author       = {Claustre, Julie and Smith, Darwin and Torres Aguilar, Sergio
                  and Bretthauer, Isabelle and Brochard, Pierre
                  and Canteaut, Olivier and Cottereau, Emilie and Delivré, Fabrice
                  and Denglos, Mathilde and Jolivet, Vincent and Julerot, Véronique
                  and Kouamé, Thierry and Lusset, Elisabeth and Massoni, Anne
                  and Nadiras, Sebastien and Perreaux, Nicolas and Regazzi, Hugo
                  and Treglia, Mathilde},
  title        = {The e-NDP project: collaborative digital edition of the Chapter
                  registers of Notre-Dame of Paris (1326-1504). Ground-truth for
                  handwriting text recognition (HTR) on late medieval manuscripts},
  year         = {2023},
  version      = {1.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7575693}
}
```

**Financement à mentionner dans les remerciements.** Agence Nationale de la Recherche, projet ANR-20-CE27-0012 (« e-NDP : Notre-Dame de Paris and its cloister »).

---

## 3. CREMMA MSS 16

**Description.** Vérité terrain HTR pour des manuscrits français du XVIᵉ siècle, sous-corpus de la famille CREMMA produit avec eScriptorium et Kraken. Première version (v0.0.1) publiée en mars 2024. Contient pour l'instant un seul sous-corpus, un *recueil de lettres originales*.

| Métadonnée | Valeur |
|---|---|
| Période | 1500–1599 (XVIᵉ siècle) |
| Langue | Français (`fra`) |
| Type de documents | Correspondance manuscrite (lettres) |
| Volume | 244 lignes / 10 911 caractères / 9 pages / 18 régions |
| Convention de transcription | Abréviations conservées (compatible CREMMA Medieval) |
| Licence | **CC-BY 4.0** |
| Dépôt | https://github.com/HTR-United/CREMMA-MSS-16 |
| Date de publication | 13 mars 2024 (v0.0.1) |

**Justification d'inclusion malgré le faible volume.** MSS 16 ne représente que 0,4 % du volume total du corpus de travail. Son inclusion vise à étendre la couverture temporelle jusqu'au XVIᵉ siècle, conformément à la consigne du brief de couvrir « plusieurs siècles, régions dialectales et types de documents », et à apporter un troisième type documentaire (épistolaire) distinct des deux corpus principaux.

**Citation (APA).**
Mazoue, A., Clérice, T., & Chagué, A. (2024). *CREMMA-MSS-16* (Version v0.0.1) [Dataset]. https://github.com/HTR-United/CREMMA-MSS-16

**Citation (BibTeX).**
```bibtex
@dataset{mazoue_cremma_mss16_2024,
  author       = {Mazoue, Anaïs and Clérice, Thibault and Chagué, Alix},
  title        = {CREMMA-MSS-16},
  year         = {2024},
  version      = {v0.0.1},
  url          = {https://github.com/HTR-United/CREMMA-MSS-16}
}
```

---

## Récapitulatif du corpus de travail

| Corpus | Période | Langue | Type | Lignes | Part du total | Licence |
|---|---|---|---|---:|---:|---|
| CREMMA Medieval | XIIᵉ–XVᵉ | ancien français | littéraire | 22 848 | 39,9 % | CC-BY 4.0 |
| e-NDP | XIVᵉ–XVIᵉ | latin + français | administratif | 34 231 | 59,7 % | CC-BY 4.0 |
| CREMMA MSS 16 | XVIᵉ | français | lettres | 244 | 0,4 % | CC-BY 4.0 |
| **Total** | **XIIᵉ–XVIᵉ** | — | — | **57 323** | **100 %** | — |

---

## Conformité licencielle

Les trois corpus sont diffusés sous **licence Creative Commons Attribution 4.0 International (CC-BY 4.0)**. Cette licence autorise la redistribution, la modification et l'usage commercial des données, sous réserve de mentionner correctement les auteurs originaux. Le présent projet respecte cette obligation par les citations ci-dessus et par le fichier `MODEL_CARD.md` du dépôt.

Aucun corpus protégé par le droit d'auteur ou diffusé sous licence restrictive (CC-BY-NC, CC-BY-ND, propriétaire) n'a été utilisé dans le pipeline.

## Modèles pré-entraînés réutilisés

(à compléter au fur et à mesure du projet, conformément à la contrainte n°7 du brief.)

| Modèle | Auteur | Licence | Compatibilité usage recherche |
|---|---|---|---|
| *à renseigner* | *à renseigner* | *à renseigner* | *à renseigner* |

---

## Attribution dans les publications

Toute publication issue du présent projet (article scientifique, modèles dérivés, jeux de données augmentés) doit mentionner explicitement :

1. **CREMMA Medieval** : Ariane Pinche et l'équipe CREMMALab (École nationale des chartes, PSL), financement DIM MAP.
2. **e-NDP** : Julie Claustre, Darwin Smith et l'ensemble de l'équipe e-NDP (LaMOP, Université Paris 1 Panthéon-Sorbonne), financement ANR-20-CE27-0012.
3. **CREMMA MSS 16** : Anaïs Mazoue, Thibault Clérice, Alix Chagué (CREMMA, HTR-United).

L'écosystème **HTR-United** (Chagué, Clérice, Chiffoleau, 2021) doit également être cité comme infrastructure ayant permis la mise à disposition de ces ressources.

---

*Document maintenu par l'équipe projet htr-medieval-manuscripts-2026 — HETIC Master Data/IA — module Vision par ordinateur, promotion 2026.*
