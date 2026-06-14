# HTR manuscrits MD5

Pipeline HTR pour le corpus ANR e-NDP Ground Truth, avec un axe d'analyse sur
l'evolution de la cursive sur deux siecles. Le depot est organise par branches :

- `main` : integration finale de toutes les etapes.
- `pretraitement_segmentation` : etape 2, pretraitement, segmentation et splits.
- `entrainement_fine-tuning` : etape 3, baselines et fine-tuning HTR.
- `agregation_evaluation_finale` : etape 4, aggregation, evaluation et JSON NLP.

## Etat actuel

Les étapes 1 à 3 sont déjà réalisées. La branche `agregation_evaluation_finale`
contient le workflow final d'agrégation, calibration et évaluation finale.

- `main` : intégration finale de toutes les étapes.
- `pretraitement_segmentation` : étape 2, prétraitement, segmentation et splits.
- `entrainement_fine-tuning` : étape 3, baselines et fine-tuning HTR.
- `agregation_evaluation_finale` : étape 4, agrégation, calibration, evaluation et JSON NLP.

Le test set reste scellé dans `experiments/splits.json`. Ne l'utilisez pas pour
choisir les hyperparamètres.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Kraken est optionnel sous Windows. Pour `ketos train`, privilegier Linux, WSL,
Docker ou Google Colab.

## Etape 3: preparation des lignes

Validation rapide sur quelques pages :

```powershell
python src/prepare_htr_dataset.py --preprocessing clahe --max_pages_per_split 2
```

Generation complete pour TrOCR :

```powershell
python src/prepare_htr_dataset.py --preprocessing clahe
```

Sortie attendue :

```text
data/processed_lines/
  train/metadata.jsonl
  validation/metadata.jsonl
  test/metadata.jsonl
```

Chaque entree `metadata.jsonl` contient l'image de ligne, la transcription, le
polygone PAGE XML, l'identifiant de ligne et le flag `needs_review`.

## Etape 3: TrOCR LoRA

Smoke test local CPU :

```powershell
python src/train_trocr.py --epochs 1 --batch_size 2 --smoke_test
```

Entrainement GPU recommande :

```powershell
python src/train_trocr.py --epochs 5 --batch_size 8 --lr 5e-4 --lora_r 16 --lora_alpha 32
```

Le script journalise deux lignes par execution : baseline zero-shot et run LoRA.
Les adaptateurs sont sauvegardes dans
`experiments/checkpoints/trocr_lora/final_lora_adapter`.

## Etape 3: Kraken/Ketos

Mini-export de validation :

```powershell
python src/train_kraken.py --preprocessing binary --max_pages_per_split 2
```

Export complet :

```powershell
python src/train_kraken.py --preprocessing binary
```

Le script genere :

- `data/kraken_dataset/train.txt`
- `data/kraken_dataset/validation.txt`
- `data/kraken_dataset/test.txt`
- `run_kraken_training.sh`

Sous Linux/WSL/Colab :

```bash
bash run_kraken_training.sh
```

## Etape 4 : agrégation et évaluation finale

Après génération des prédictions TrOCR et Kraken, utilisez le workflow final
suivant pour produire l'agrégé, calibrer les confiances, construire le dataset
NLP et valider les résultats.

### Agrégation

```powershell
python -m src.aggregation ^
  --metadata data/processed_lines/train/metadata.jsonl ^
  --prediction trocr=predictions/trocr.jsonl ^
  --prediction kraken=predictions/kraken.jsonl ^
  --output experiments/aggregated_predictions.jsonl
```

### Calibration des confiances

```powershell
python -m src.calibration ^
  --input experiments/aggregated_predictions.jsonl ^
  --output experiments/aggregated_calibrated.jsonl ^
  --bins 10 ^
  --smoothing 1.0
```

### Construction du dataset final

```powershell
python -m src.data_contract ^
  --records experiments/aggregated_calibrated.jsonl ^
  --output dataset_nlp/transcriptions.json
```

### Évaluation finale

```powershell
python -m src.evaluation ^
  --predictions experiments/aggregated_calibrated.jsonl ^
  --output experiments/final_metrics.json
```

### Validation géométrique

```powershell
python -m src.validate_geometry ^
  --records experiments/aggregated_calibrated.jsonl ^
  --images_dir data/HTR_Ground-Truth/images_docs ^
  --output experiments/geometry_report.json
```

Ajustez `--images_dir` selon le dossier qui contient les pages source si besoin.

## Tests

```powershell
pytest tests/ -p no:cacheprovider
```

Les tests actuels couvrent le pretraitement, la segmentation fallback, le parsing
PAGE XML, les crops de lignes, le journal JSONL et le script Kraken genere.

## Workflow Git recommande

Depuis `entrainement_fine-tuning` :

```powershell
git status
git add .gitignore README.md src/prepare_htr_dataset.py src/train_trocr.py src/train_kraken.py tests/test_training.py
git commit -m "Complete HTR fine-tuning workflow"
git checkout agregation_evaluation_finale
git merge entrainement_fine-tuning
```

Apres l'etape 4 :

```powershell
git checkout main
git merge agregation_evaluation_finale
```
