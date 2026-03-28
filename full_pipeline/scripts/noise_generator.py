import json
import random
import re
from pathlib import Path
from docx import Document

# Load error taxonomy
taxonomy_path = Path.home() / "ocr_project/data/error_taxonomy.json"
with open(taxonomy_path) as f:
    taxonomy = json.load(f)

# Build confusion matrix from real data
CONFUSION = {
    's': [('f', 0.40), ('l', 0.15), ('i', 0.05)],
    'e': [('c', 0.20), ('o', 0.10), ('é', 0.10)],
    'u': [('a', 0.15), ('n', 0.15), ('o', 0.10)],
    't': [('r', 0.15), ('l', 0.12), ('c', 0.08)],
    'a': [('á', 0.12), ('o', 0.08), ('e', 0.05)],
    'o': [('ó', 0.10), ('c', 0.08), ('e', 0.05)],
    'v': [('u', 0.15), ('b', 0.05)],
    'i': [('í', 0.10), ('l', 0.10), ('1', 0.05)],
    'n': [('u', 0.10), ('m', 0.05), ('ri', 0.05)],
    'r': [('t', 0.08), ('n', 0.05)],
    'c': [('e', 0.08), ('o', 0.05)],
    'd': [('cl', 0.05)],
    'm': [('rn', 0.08), ('ni', 0.03)],
    'g': [('e', 0.05), ('q', 0.03)],
    'h': [('b', 0.05), ('li', 0.03)],
    'p': [('q', 0.05), ('b', 0.03)],
    ' ': [('.', 0.03), ('', 0.08), ('-', 0.02)],
}

DELETION_PROBS = {
    ' ': 0.06, 's': 0.02, 'e': 0.015, ',': 0.01,
    'a': 0.01, '.': 0.01, 'r': 0.008, 't': 0.007,
    'i': 0.007, 'o': 0.007, 'n': 0.006
}

INSERTION_CHARS = [' ', '.', 'i', 'f', '-', 'r', 'a', ':', 'e', 'n']

def inject_noise(text, error_rate=0.15, seed=None):
    """
    Inject realistic OCR errors based on real confusion matrix.
    
    Args:
        text: clean ground truth text
        error_rate: probability of error per character (0.0 to 1.0)
        seed: random seed for reproducibility
    
    Returns:
        noisy text, list of changes made
    """
    if seed is not None:
        random.seed(seed)
    
    chars = list(text)
    result = []
    changes = []
    i = 0
    
    while i < len(chars):
        ch = chars[i]
        
        if random.random() < error_rate:
            action = random.choices(
                ['substitute', 'delete', 'insert', 'merge'],
                weights=[0.50, 0.20, 0.15, 0.15]
            )[0]
            
            if action == 'substitute' and ch.lower() in CONFUSION:
                candidates = CONFUSION[ch.lower()]
                choices, weights = zip(*candidates)
                replacement = random.choices(choices, weights=weights)[0]
                if ch.isupper() and len(replacement) == 1:
                    replacement = replacement.upper()
                result.append(replacement)
                changes.append(f"sub:{ch}->{replacement}")
                
            elif action == 'delete':
                changes.append(f"del:{ch}")
                # skip this character
                
            elif action == 'insert':
                result.append(ch)
                insert_char = random.choice(INSERTION_CHARS)
                result.append(insert_char)
                changes.append(f"ins:{insert_char}")
                
            elif action == 'merge' and ch == ' ':
                # Remove space (merge words)
                changes.append("merge")
                
            else:
                result.append(ch)
        else:
            result.append(ch)
        
        i += 1
    
    return ''.join(result), changes

def generate_dataset(clean_texts, error_rates=[0.05, 0.10, 0.15, 0.20],
                      samples_per_rate=3):
    """Generate synthetic noisy-clean pairs at multiple error rates."""
    dataset = []
    
    for text in clean_texts:
        if len(text) < 5:
            continue
        for rate in error_rates:
            for s in range(samples_per_rate):
                noisy, changes = inject_noise(text, error_rate=rate, seed=None)
                dataset.append({
                    "clean": text,
                    "noisy": noisy,
                    "error_rate": rate,
                    "num_changes": len(changes),
                    "changes": changes
                })
    
    return dataset

# ── Collect clean texts from transcriptions ──────────────────
print("Collecting clean texts from transcriptions...")
trans_dir = Path.home() / "ocr_project/data/transcriptions"
clean_texts = []

for docx_path in trans_dir.rglob("*.docx"):
    try:
        doc = Document(str(docx_path))
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text or text.startswith("PDF p") or text.startswith("NOTES:"):
                continue
            for line in text.split('\n'):
                line = line.strip()
                if len(line) > 10:
                    clean_texts.append(line)
    except:
        pass

print(f"Collected {len(clean_texts)} clean text lines")

# ── Generate synthetic dataset ───────────────────────────────
print("\nGenerating synthetic noisy-clean pairs...")
synthetic = generate_dataset(
    clean_texts,
    error_rates=[0.05, 0.10, 0.15, 0.20],
    samples_per_rate=3
)

print(f"Generated {len(synthetic)} synthetic pairs")

# ── Show samples at each error rate ──────────────────────────
print("\n=== SAMPLES AT DIFFERENT ERROR RATES ===")
for rate in [0.05, 0.10, 0.15, 0.20]:
    samples = [s for s in synthetic if s["error_rate"] == rate][:2]
    print(f"\n--- Error Rate: {rate*100:.0f}% ---")
    for s in samples:
        print(f"  Clean: {s['clean'][:60]}")
        print(f"  Noisy: {s['noisy'][:60]}")
        print(f"  Changes: {len(s['changes'])}")
        print()

# ── Save synthetic dataset ───────────────────────────────────
out_path = Path.home() / "ocr_project/data/synthetic_pairs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(synthetic, f, indent=2, ensure_ascii=False)

# Also save as T5 training format
t5_data = []
for s in synthetic:
    t5_data.append({
        "input": f"correct: {s['noisy']}",
        "target": s["clean"]
    })

t5_path = Path.home() / "ocr_project/data/synthetic_t5_training.json"
with open(t5_path, "w", encoding="utf-8") as f:
    json.dump(t5_data, f, indent=2, ensure_ascii=False)

# Stats
print("=== SYNTHETIC DATA SUMMARY ===")
print(f"Total pairs: {len(synthetic)}")
for rate in [0.05, 0.10, 0.15, 0.20]:
    count = sum(1 for s in synthetic if s["error_rate"] == rate)
    print(f"  {rate*100:.0f}% error rate: {count} pairs")
print(f"\nSaved to: {out_path}")
print(f"T5 format: {t5_path}")
print("\nStep 5.2 Complete!")
