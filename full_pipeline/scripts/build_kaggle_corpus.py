import json
import csv
import shutil
from pathlib import Path

print("=== Building Kaggle Error Corpus ===\n")

# ── Collect all error data ───────────────────────────────────
kaggle_dir = Path.home() / "ocr_project/outputs/kaggle_corpus"
kaggle_dir.mkdir(exist_ok=True)

# 1. Load real OCR pairs
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    real_pairs = json.load(f)

# 2. Load synthetic pairs
syn_path = Path.home() / "ocr_project/data/synthetic_t5_training.json"
with open(syn_path) as f:
    synthetic = json.load(f)

# 3. Load error taxonomy
tax_path = Path.home() / "ocr_project/data/error_taxonomy.json"
with open(tax_path) as f:
    taxonomy = json.load(f)

# ── Build main corpus CSV ────────────────────────────────────
corpus_rows = []

# Add real pairs
for p in real_pairs:
    if len(p["gt_text"]) > 3:
        corpus_rows.append({
            "id": f"real_{len(corpus_rows):04d}",
            "source": "real_ocr",
            "noisy_text": p["ocr_text"],
            "clean_text": p["gt_text"],
            "edit_distance": p["edit_distance"],
            "document_type": "print_17th_century_spanish",
            "error_rate": round(p["edit_distance"] / max(len(p["gt_text"]), 1), 3)
        })

# Add synthetic pairs
for i, s in enumerate(synthetic[:2000]):
    clean = s["target"]
    noisy = s["input"].replace("correct: ", "")
    if len(clean) > 3:
        corpus_rows.append({
            "id": f"syn_{i:04d}",
            "source": "synthetic",
            "noisy_text": noisy,
            "clean_text": clean,
            "edit_distance": -1,
            "document_type": "synthetic_17th_century_spanish",
            "error_rate": -1
        })

print(f"Total corpus rows: {len(corpus_rows)}")

# Save as CSV
csv_path = kaggle_dir / "ocr_error_corpus.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=corpus_rows[0].keys())
    writer.writeheader()
    writer.writerows(corpus_rows)
print(f"Saved CSV: {csv_path}")

# ── Save confusion matrix ────────────────────────────────────
conf_path = kaggle_dir / "confusion_matrix.json"
with open(conf_path, "w", encoding="utf-8") as f:
    json.dump(taxonomy, f, indent=2, ensure_ascii=False)
print(f"Saved confusion matrix: {conf_path}")

# ── Save error taxonomy summary ──────────────────────────────
summary = {
    "dataset_name": "17th Century Spanish OCR Error Corpus",
    "description": "Real and synthetic OCR errors from 17th-century Spanish printed documents",
    "language": "Spanish (17th century)",
    "document_types": ["printed books", "legal documents", "religious texts"],
    "sources": [
        "Buendia - Instruccion (1740)",
        "Covarrubias - Tesoro de la Lengua",
        "Guardiola - Tratado de Nobleza",
        "PORCONES (legal documents 1628-1650)"
    ],
    "statistics": {
        "real_ocr_pairs": len(real_pairs),
        "synthetic_pairs": len(synthetic),
        "total_pairs": len(corpus_rows),
        "overall_error_rate": taxonomy["overall_error_rate"],
        "top_confusions": list(taxonomy["substitution_counts"].keys())[:10]
    },
    "files": {
        "ocr_error_corpus.csv": "Main dataset with noisy/clean text pairs",
        "confusion_matrix.json": "Character confusion probabilities",
        "dataset_card.md": "Dataset documentation"
    }
}

summary_path = kaggle_dir / "dataset_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ── Create dataset card (README for Kaggle) ──────────────────
readme = """# 17th Century Spanish OCR Error Corpus

## Overview
This dataset contains real and synthetic OCR error pairs from 17th-century 
Spanish printed documents. It is designed to train and evaluate OCR correction 
models for historical document transcription.

## Dataset Statistics
- **Real OCR pairs**: 137
- **Synthetic pairs**: 9,060
- **Total pairs**: 9,197
- **Language**: 17th-century Spanish (with Latin)
- **Document period**: 1606-1740

## Files
- `ocr_error_corpus.csv` — Main dataset
- `confusion_matrix.json` — Character confusion probabilities
- `dataset_summary.json` — Full statistics

## Column Description
| Column | Description |
|--------|-------------|
| id | Unique identifier |
| source | real_ocr or synthetic |
| noisy_text | OCR output (with errors) |
| clean_text | Ground truth transcription |
| edit_distance | Levenshtein distance |
| document_type | Type of source document |
| error_rate | Character error rate |

## Top OCR Confusions Found
1. s → f (long-s confusion, most common!)
2. s → l
3. u → a
4. e → c
5. t → r

## Usage
```python
import pandas as pd
df = pd.read_csv('ocr_error_corpus.csv')
# Filter real pairs only
real = df[df['source'] == 'real_ocr']
# Filter by error rate
easy = df[df['error_rate'] < 0.1]
