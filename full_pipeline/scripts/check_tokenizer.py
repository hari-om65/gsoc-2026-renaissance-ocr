from transformers import TrOCRProcessor
import json
from pathlib import Path

print("Loading TrOCR processor...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

# Load our ground truth pairs
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

# Collect all unique characters from GT
all_chars = set()
for p in pairs:
    all_chars.update(p["gt_text"])

print(f"\nTotal unique characters in GT: {len(all_chars)}")
print(f"All characters: {''.join(sorted(all_chars))}")

# Check which characters are NOT in tokenizer
tokenizer = processor.tokenizer
missing_chars = []
for ch in sorted(all_chars):
    tokens = tokenizer.tokenize(ch)
    if tokens == ["<unk>"] or not tokens:
        missing_chars.append(ch)

print(f"\nCharacters missing from tokenizer: {len(missing_chars)}")
print(f"Missing: {missing_chars}")

# Check coverage
total_chars_in_gt = sum(len(p["gt_text"]) for p in pairs)
print(f"\nTotal characters in all GT texts: {total_chars_in_gt}")
print(f"\nTokenizer vocab size: {len(tokenizer)}")
print("\nConclusion:")
if len(missing_chars) == 0:
    print("All characters covered! No normalization needed.")
elif len(missing_chars) < 5:
    print(f"Only {len(missing_chars)} missing chars - simple normalization will work.")
else:
    print(f"{len(missing_chars)} missing chars - need normalization policy.")
