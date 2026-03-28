import json
import random
import numpy as np
from pathlib import Path
from jiwer import cer

# Load synthetic data
syn_path = Path.home() / "ocr_project/data/synthetic_t5_training.json"
with open(syn_path) as f:
    synthetic = json.load(f)

# Load real OCR pairs
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    real_pairs = json.load(f)

real_pairs = [p for p in real_pairs if len(p["gt_text"]) > 5]

print("=== SYNTHETIC DATA QUALITY VALIDATION ===\n")

# ── 1. Compare error distributions ──────────────────────────
print("1. ERROR RATE COMPARISON")
print("-" * 40)

# Real error rates
real_cers = []
for p in real_pairs:
    real_cers.append(cer(p["gt_text"], p["ocr_text"]))

# Synthetic error rates
syn_cers = []
for s in synthetic:
    clean = s["target"]
    noisy = s["input"].replace("correct: ", "")
    syn_cers.append(cer(clean, noisy))

print(f"Real OCR errors:")
print(f"  Mean CER: {np.mean(real_cers):.3f}")
print(f"  Std CER:  {np.std(real_cers):.3f}")
print(f"  Min CER:  {np.min(real_cers):.3f}")
print(f"  Max CER:  {np.max(real_cers):.3f}")

print(f"\nSynthetic errors:")
print(f"  Mean CER: {np.mean(syn_cers):.3f}")
print(f"  Std CER:  {np.std(syn_cers):.3f}")
print(f"  Min CER:  {np.min(syn_cers):.3f}")
print(f"  Max CER:  {np.max(syn_cers):.3f}")

# ── 2. Compare error type distributions ──────────────────────
print(f"\n2. ERROR TYPE ANALYSIS")
print("-" * 40)

def count_error_types(clean, noisy):
    subs = 0
    dels = 0
    ins = 0
    for i, (c, n) in enumerate(zip(clean, noisy)):
        if c != n:
            subs += 1
    dels = max(0, len(clean) - len(noisy))
    ins = max(0, len(noisy) - len(clean))
    return subs, dels, ins

real_subs, real_dels, real_ins = 0, 0, 0
for p in real_pairs[:50]:
    s, d, i = count_error_types(p["gt_text"], p["ocr_text"])
    real_subs += s
    real_dels += d
    real_ins += i

syn_subs, syn_dels, syn_ins = 0, 0, 0
for s in random.sample(synthetic, min(50, len(synthetic))):
    clean = s["target"]
    noisy = s["input"].replace("correct: ", "")
    su, d, i = count_error_types(clean, noisy)
    syn_subs += su
    syn_dels += d
    syn_ins += i

real_total = real_subs + real_dels + real_ins
syn_total = syn_subs + syn_dels + syn_ins

print(f"{'Type':<15} {'Real %':>8} {'Synthetic %':>12}")
print(f"{'Substitutions':<15} {real_subs/max(real_total,1)*100:>7.1f}% {syn_subs/max(syn_total,1)*100:>11.1f}%")
print(f"{'Deletions':<15} {real_dels/max(real_total,1)*100:>7.1f}% {syn_dels/max(syn_total,1)*100:>11.1f}%")
print(f"{'Insertions':<15} {real_ins/max(real_total,1)*100:>7.1f}% {syn_ins/max(syn_total,1)*100:>11.1f}%")

# ── 3. Sample comparison ─────────────────────────────────────
print(f"\n3. SIDE-BY-SIDE SAMPLES")
print("-" * 40)

print("\nReal OCR errors:")
for p in random.sample(real_pairs, min(3, len(real_pairs))):
    print(f"  Clean: {p['gt_text'][:50]}")
    print(f"  Noisy: {p['ocr_text'][:50]}")
    print()

print("Synthetic OCR errors:")
for s in random.sample(synthetic, 3):
    clean = s["target"]
    noisy = s["input"].replace("correct: ", "")
    print(f"  Clean: {clean[:50]}")
    print(f"  Noisy: {noisy[:50]}")
    print()

# ── 4. Dataset size summary ──────────────────────────────────
print(f"4. DATASET SIZE SUMMARY")
print("-" * 40)
print(f"Real OCR pairs:       {len(real_pairs)}")
print(f"Synthetic pairs:      {len(synthetic)}")
print(f"Total training data:  {len(real_pairs) + len(synthetic)}")
print(f"Augmentation factor:  {len(synthetic)/len(real_pairs):.0f}x")

# ── 5. Save validation report ────────────────────────────────
report = {
    "real_mean_cer": float(np.mean(real_cers)),
    "synthetic_mean_cer": float(np.mean(syn_cers)),
    "real_pairs": len(real_pairs),
    "synthetic_pairs": len(synthetic),
    "augmentation_factor": len(synthetic) / len(real_pairs),
    "error_type_distribution": {
        "real": {"substitutions": real_subs, "deletions": real_dels, "insertions": real_ins},
        "synthetic": {"substitutions": syn_subs, "deletions": syn_dels, "insertions": syn_ins}
    }
}

out_path = Path.home() / "ocr_project/outputs/synthetic_validation.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nValidation report saved to: {out_path}")
print("\nPhase 5 Complete!")
