# Étape 2.5 — Export des polygones en PAGE XML

## Objectif

Persister les **géométries de lignes** (polygones et baselines) du corpus dans une structure intermédiaire standard, conformément à l'exigence du brief :

> *Après avoir extrait les lignes de base et les polygones via Kraken BLLA ou tout autre segmenteur, le pipeline doit immédiatement sauvegarder ces géométries dans une structure intermédiaire (par exemple un fichier `.page.xml` ou un dictionnaire associé à chaque ligne). Ne pas attendre la fin du projet pour produire les polygones : leur validation précoce évite les mauvaises surprises.*

## Notre choix : double persistance

Le projet implémente **les deux formats** suggérés par le brief, pour maximiser la traçabilité et l'interopérabilité :

| Format | Statut | Fichier(s) | Usage |
|---|---|---|---|
| Dictionnaire par ligne | ✅ Sous-étape 2.4 | `data/processed/endp/{train,val,test}.json` | Inspection rapide, analyses statistiques, code Python |
| PAGE XML par page | ✅ Sous-étape 2.5 | `data/processed/page_xml_export/{train,val,test}/*.xml` | Entrée Kraken pour la reconnaissance (étape 3) |

Les deux représentations sont strictement équivalentes en contenu : chaque ligne porte son `polygon`, sa `baseline`, son texte transcrit, et son identifiant.

## Géométrie des lignes : origine

Les polygones et baselines exportés proviennent du **ground truth e-NDP** (annotations manuelles). Le projet conserve également le code pour les générer via Kraken BLLA (`src/segmentation/kraken_wrapper.py`), évalué en sous-étape 2.3 (IoU 0.74-0.76). Pour l'étape 3 (fine-tuning du modèle de reconnaissance), on utilise les polygones du ground truth, plus précis.

## Structure des fichiers produits

```
data/processed/page_xml_export/
├── train/        253 fichiers *.xml   (16 013 lignes)
├── val/          106 fichiers *.xml   ( 5 816 lignes)
└── test/          98 fichiers *.xml   ( 7 829 lignes)  ← scellé, ne pas ouvrir
```

Format PAGE XML 2019-07-15 (standard PRImA Research Lab), schéma validable :

```xml
<?xml version='1.0' encoding='utf-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
       xsi:schemaLocation="..."
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Page imageFilename="FRAN_0393_00016_L.jpg" imageWidth="2808" imageHeight="3111">
    <TextRegion id="region_FRAN_0393_00016_L">
      <Coords points="0,0 2807,0 2807,3110 0,3110"/>
      <TextLine id="eSc_line_aa368492">
        <Coords points="528,416 528,316 627,329 627,416 ..."/>
        <Baseline points="528,416 627,416"/>
        <TextEquiv>
          <Unicode>328</Unicode>
        </TextEquiv>
      </TextLine>
      <!-- ... autres lignes ... -->
    </TextRegion>
  </Page>
</PcGts>
```

## Reproduire l'export

Une fois la sous-étape 2.4 exécutée (qui produit les JSON de split) :

```bash
python scripts/export_page_xml.py \
    --splits-dir data/processed/endp \
    --output-dir data/processed/page_xml_export
```

Coût : ~3 secondes pour les 457 pages.

## Validation précoce des polygones

L'exigence du brief de **valider tôt** est satisfaite par trois mécanismes complémentaires :

1. **Tests pytest** (`tests/test_page_xml_export.py`) — 3 tests vérifient la conformité du XML produit et la rasterisation correcte des polygones.
2. **Round-trip implicite** — tout polygone que `xml.etree.ElementTree` n'arrive pas à reparser cause un échec immédiat dans `lines_to_page_xml`.
3. **Comparaison avec Kraken BLLA** (sous-étape 2.3) — confirme que les polygones du ground truth sont géométriquement cohérents (matchent ce qu'un segmenteur moderne détecte indépendamment).

## Implémentation

- `src/corpus/page_xml_export.py` : module Python qui rebuild un PAGE XML par page (fonctions `lines_to_page_xml`, `export_split_to_page_xml`).
- `scripts/export_page_xml.py` : orchestrateur en ligne de commande.
- `tests/test_page_xml_export.py` : tests unitaires.

## Référence

- PRImA Research Lab. *PAGE XML Format 2019-07-15*. http://www.primaresearch.org/tools/PAGELibraries
