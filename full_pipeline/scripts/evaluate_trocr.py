import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Load model
print("Loading fine-tuned TrOCR...")
model_dir = Path.home() / "ocr_project/models/trocr_finetuned"
processor = TrOCRProcessor.from_pretrained(str(model_dir))
model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on {device}")

def run_trocr(image_path):
    """Run TrOCR on a single line image."""
    try:
        img = Image.open(str(image_path)).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated = model.generate(pixel_values)
        text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text.strip()
    except Exception as e:
        return ""

# Load pairs for evaluation
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

pairs = [p for p in pairs if len(p["gt_text"]) > 3]
split = int(len(pairs) * 0.85)
test_pairs = pairs[split:]
print(f"\nEvaluating on {len(test_pairs)} held-out pairs...")

# Run inference
results = []
for pair in test_pairs:
    pred = run_trocr(pair["image_path"])
    results.append({
        "gt": pair["gt_text"],
        "pred": pred,
        "image_path": pair["image_path"]
    })

# Compute metrics
gt_texts = [r["gt"] for r in results]
pred_texts = [r["pred"] for r in results]

overall_cer = cer(gt_texts, pred_texts)
overall_wer = wer(gt_texts, pred_texts)

print(f"\n=== EVALUATION RESULTS ===")
print(f"CER: {overall_cer:.4f} ({overall_cer*100:.1f}%)")
print(f"WER: {overall_wer:.4f} ({overall_wer*100:.1f}%)")

print(f"\n=== SAMPLE PREDICTIONS ===")
for r in results[:10]:
    print(f"GT:   {r['gt'][:70]}")
    print(f"PRED: {r['pred'][:70]}")
    print()

# Error analysis
print("=== ERROR ANALYSIS ===")
char_errors = {}
for r in results:
    gt = r["gt"]
    pred = r["pred"]
    for i, (gc, pc) in enumerate(zip(gt, pred)):
        if gc != pc:
            key = f"{gc}→{pc}"
            char_errors[key] = char_errors.get(key, 0) + 1

print("Top character confusions:")
for k, v in sorted(char_errors.items(), key=lambda x: -x[1])[:15]:
    print(f"  '{k}': {v} times")

# Save results
out_path = Path.home() / "ocr_project/outputs/evaluation_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "cer": overall_cer,
        "wer": overall_wer,
        "num_samples": len(results),
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {out_path}")
print("\nPhase 2 Complete!")
