import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# Load our real OCR pairs
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

pairs = [p for p in pairs if len(p["gt_text"]) > 5 and len(p["ocr_text"]) > 5]
print(f"Analyzing {len(pairs)} real OCR error pairs...\n")

# ── Build confusion matrix ───────────────────────────────────
char_substitutions = Counter()
char_deletions = Counter()
char_insertions = Counter()
word_merges = 0
word_splits = 0
total_chars = 0
error_chars = 0

def align_strings(gt, ocr):
    """Simple character-level alignment using dynamic programming."""
    m, n = len(gt), len(ocr)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if gt[i-1] == ocr[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack to find operations
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i-1] == ocr[j-1]:
            ops.append(('match', gt[i-1], ocr[j-1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('substitute', gt[i-1], ocr[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('delete', gt[i-1], ''))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(('insert', '', ocr[j-1]))
            j -= 1
        else:
            break

    return list(reversed(ops))

for pair in pairs:
    gt = pair["gt_text"]
    ocr = pair["ocr_text"]
    total_chars += len(gt)

    ops = align_strings(gt, ocr)
    for op_type, gt_char, ocr_char in ops:
        if op_type == 'substitute':
            char_substitutions[(gt_char, ocr_char)] += 1
            error_chars += 1
        elif op_type == 'delete':
            char_deletions[gt_char] += 1
            error_chars += 1
        elif op_type == 'insert':
            char_insertions[ocr_char] += 1
            error_chars += 1

    # Detect word merges and splits
    gt_words = gt.split()
    ocr_words = ocr.split()
    if len(ocr_words) < len(gt_words) * 0.7:
        word_merges += 1
    if len(ocr_words) > len(gt_words) * 1.3:
        word_splits += 1

# ── Print results ────────────────────────────────────────────
print("=" * 60)
print("ERROR TAXONOMY FROM REAL OCR DATA")
print("=" * 60)

print(f"\nTotal characters analyzed: {total_chars}")
print(f"Total errors found: {error_chars}")
print(f"Overall error rate: {error_chars/total_chars*100:.1f}%")

print(f"\n--- CHARACTER SUBSTITUTIONS (top 25) ---")
print(f"{'GT':>5} → {'OCR':>5} | {'Count':>6} | {'Probability':>10}")
print("-" * 40)
total_subs = sum(char_substitutions.values())
for (gt_c, ocr_c), count in char_substitutions.most_common(25):
    gt_display = repr(gt_c)
    ocr_display = repr(ocr_c)
    prob = count / total_subs
    print(f"{gt_display:>5} → {ocr_display:>5} | {count:>6} | {prob:>10.4f}")

print(f"\n--- CHARACTER DELETIONS (top 15) ---")
for char, count in char_deletions.most_common(15):
    print(f"  '{char}' deleted {count} times")

print(f"\n--- CHARACTER INSERTIONS (top 15) ---")
for char, count in char_insertions.most_common(15):
    print(f"  '{char}' inserted {count} times")

print(f"\n--- WORD-LEVEL ERRORS ---")
print(f"  Word merges detected: {word_merges}")
print(f"  Word splits detected: {word_splits}")

# ── Build confusion matrix dictionary ────────────────────────
confusion_matrix = {}
for (gt_c, ocr_c), count in char_substitutions.most_common(50):
    if gt_c not in confusion_matrix:
        confusion_matrix[gt_c] = {}
    confusion_matrix[gt_c][ocr_c] = count

# Calculate probabilities
confusion_probs = {}
for gt_c, subs in confusion_matrix.items():
    total = sum(subs.values())
    confusion_probs[gt_c] = {
        ocr_c: round(count/total, 3)
        for ocr_c, count in subs.items()
    }

# Save
out_path = Path.home() / "ocr_project/data/error_taxonomy.json"
taxonomy = {
    "total_chars_analyzed": total_chars,
    "total_errors": error_chars,
    "overall_error_rate": error_chars / total_chars,
    "substitution_counts": {
        f"{gt}->{ocr}": count
        for (gt, ocr), count in char_substitutions.most_common(50)
    },
    "deletion_counts": dict(char_deletions.most_common(20)),
    "insertion_counts": dict(char_insertions.most_common(20)),
    "confusion_matrix_probabilities": confusion_probs,
    "word_merges": word_merges,
    "word_splits": word_splits
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(taxonomy, f, indent=2, ensure_ascii=False)

print(f"\nError taxonomy saved to: {out_path}")
print("Step 5.1 Complete!")
