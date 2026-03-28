import json
import re
from pathlib import Path
from spellchecker import SpellChecker
from docx import Document
from Levenshtein import distance as levenshtein_distance

# ── Build custom historical Spanish dictionary ───────────────
def extract_words_from_docx(docx_path):
    """Extract all unique words from ground truth transcriptions."""
    doc = Document(str(docx_path))
    words = set()
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text or text.startswith("PDF p") or text.startswith("NOTES:"):
            continue
        # Extract words
        for word in re.findall(r"[a-záéíóúñüçA-ZÁÉÍÓÚÑÜÇ]+", text):
            words.add(word.lower())
    return words

print("Building custom historical Spanish dictionary...")
trans_dir = Path.home() / "ocr_project/data/transcriptions"
custom_words = set()

for docx_path in trans_dir.rglob("*.docx"):
    words = extract_words_from_docx(docx_path)
    custom_words.update(words)
    print(f"  {docx_path.name}: {len(words)} unique words")

print(f"\nTotal custom historical words: {len(custom_words)}")

# Save custom dictionary
dict_path = Path.home() / "ocr_project/data/custom_dict.txt"
with open(dict_path, "w", encoding="utf-8") as f:
    for word in sorted(custom_words):
        f.write(word + "\n")
print(f"Saved custom dictionary to: {dict_path}")

# ── Initialize SpellChecker ──────────────────────────────────
spell = SpellChecker(language="es")
# Add custom historical words so they are not flagged as errors
spell.word_frequency.load_words(list(custom_words))
print(f"\nSpellChecker initialized with Spanish + {len(custom_words)} historical words")

# ── Correction function ──────────────────────────────────────
def correct_text_hunspell(text, spell, custom_words):
    """
    Correct OCR text using Hunspell.
    Returns corrected text + list of changes made.
    """
    words = text.split()
    corrected_words = []
    changes = []
    flagged = []

    for word in words:
        # Extract core word (remove punctuation)
        clean = re.sub(r"[^a-záéíóúñüçA-ZÁÉÍÓÚÑÜÇ]", "", word)

        if len(clean) < 2:
            corrected_words.append(word)
            continue

        # Check if word is correct
        if spell.known([clean.lower()]):
            corrected_words.append(word)
            continue

        # Get suggestions
        candidates = spell.candidates(clean.lower())

        if not candidates:
            corrected_words.append(word)
            flagged.append(word)
            continue

        # Find best candidate by edit distance
        best = min(candidates, key=lambda c: levenshtein_distance(clean.lower(), c))
        dist = levenshtein_distance(clean.lower(), best)

        if dist == 1:
            # Auto-correct: very confident
            corrected = word.replace(clean, best if clean.islower() else best.upper() if clean.isupper() else best.capitalize())
            corrected_words.append(corrected)
            changes.append({
                "original": word,
                "corrected": corrected,
                "distance": dist,
                "stage": "hunspell_auto"
            })
        elif dist == 2:
            # Auto-correct with lower confidence
            corrected = word.replace(clean, best)
            corrected_words.append(corrected)
            changes.append({
                "original": word,
                "corrected": corrected,
                "distance": dist,
                "stage": "hunspell_auto"
            })
        else:
            # Too ambiguous — flag for next stage
            corrected_words.append(word)
            flagged.append(word)

    corrected_text = " ".join(corrected_words)
    return corrected_text, changes, flagged

# ── Test on our OCR outputs ──────────────────────────────────
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

# Use pairs as noisy input
split = int(len(pairs) * 0.85)
test_pairs = pairs[split:]

print(f"\n=== Testing Hunspell on {len(test_pairs)} samples ===")

from jiwer import cer

results = []
for pair in test_pairs:
    raw_ocr = pair["ocr_text"]
    gt = pair["gt_text"]

    corrected, changes, flagged = correct_text_hunspell(raw_ocr, spell, custom_words)

    cer_before = cer(gt, raw_ocr)
    cer_after = cer(gt, corrected)

    results.append({
        "gt": gt,
        "raw_ocr": raw_ocr,
        "corrected": corrected,
        "changes": changes,
        "flagged": flagged,
        "cer_before": cer_before,
        "cer_after": cer_after
    })

# Compute average improvement
avg_cer_before = sum(r["cer_before"] for r in results) / len(results)
avg_cer_after = sum(r["cer_after"] for r in results) / len(results)

print(f"\nAvg CER before Hunspell: {avg_cer_before:.4f} ({avg_cer_before*100:.1f}%)")
print(f"Avg CER after Hunspell:  {avg_cer_after:.4f} ({avg_cer_after*100:.1f}%)")
print(f"Improvement: {(avg_cer_before - avg_cer_after)*100:.1f}%")

print(f"\n=== SAMPLE CORRECTIONS ===")
for r in results[:5]:
    print(f"GT:        {r['gt'][:60]}")
    print(f"Raw OCR:   {r['raw_ocr'][:60]}")
    print(f"Corrected: {r['corrected'][:60]}")
    if r['changes']:
        print(f"Changes:   {r['changes']}")
    if r['flagged']:
        print(f"Flagged:   {r['flagged']}")
    print()

# Save results
out_path = Path.home() / "ocr_project/outputs/stage1_hunspell_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "avg_cer_before": avg_cer_before,
        "avg_cer_after": avg_cer_after,
        "improvement": avg_cer_before - avg_cer_after,
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"Results saved to: {out_path}")
print("\nStage 1 (Hunspell) Complete!")
