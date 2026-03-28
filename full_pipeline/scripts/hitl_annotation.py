import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Create annotation review sheets ─────────────────────────
# These are PNG files a human can look at, mark corrections,
# and feed back into the training pipeline

def create_review_sheet(page_image_path, line_crops_dir,
                         ocr_results, out_path):
    """
    Create a review sheet showing:
    - Left: original line crop
    - Right: OCR prediction
    Human annotator writes corrections in the JSON file
    """
    crops = sorted(Path(line_crops_dir).glob("line_*.png"))[:15]
    if not crops:
        return

    n = len(crops)
    fig, axes = plt.subplots(n, 2, figsize=(20, n * 1.2))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Review Sheet: {Path(page_image_path).name}",
                 fontsize=14, fontweight='bold')

    for i, crop_path in enumerate(crops):
        crop_img = cv2.imread(str(crop_path))
        if crop_img is None:
            continue
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        # Left: line image
        axes[i][0].imshow(crop_rgb)
        axes[i][0].set_title(crop_path.name, fontsize=7)
        axes[i][0].axis('off')

        # Right: OCR text
        ocr_text = ocr_results.get(crop_path.name, "")
        axes[i][1].text(0.02, 0.5, f"OCR: {ocr_text}",
                        transform=axes[i][1].transAxes,
                        fontsize=9, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow'))
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

def create_annotation_json(line_crops_dir, ocr_results, out_path):
    """
    Create a JSON file for human annotators to fill in corrections.
    This is the HITL feedback mechanism.
    """
    crops = sorted(Path(line_crops_dir).glob("line_*.png"))
    annotation_template = {
        "source": str(line_crops_dir),
        "status": "pending_review",
        "annotator": "",
        "annotations": []
    }

    for crop_path in crops:
        ocr_text = ocr_results.get(crop_path.name, "")
        annotation_template["annotations"].append({
            "line_image": str(crop_path),
            "ocr_prediction": ocr_text,
            "human_correction": ocr_text,  # human fills this
            "is_correct": None,            # human marks True/False
            "confidence": "medium",        # high/medium/low
            "notes": ""
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation_template, f, indent=2, ensure_ascii=False)

    return annotation_template

# ── Run TrOCR on line crops ──────────────────────────────────
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

print("Loading TrOCR for HITL annotation...")
model_dir = Path.home() / "ocr_project/models/trocr_finetuned"
processor = TrOCRProcessor.from_pretrained(str(model_dir))
trocr = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trocr.to(device)
trocr.eval()

def run_trocr_on_crops(crops_dir):
    """Run TrOCR on all line crops in a directory."""
    results = {}
    crops = sorted(Path(crops_dir).glob("line_*.png"))
    for crop_path in crops:
        try:
            img = Image.open(str(crop_path)).convert("RGB")
            pixel_values = processor(
                img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated = trocr.generate(pixel_values)
            text = processor.tokenizer.decode(
                generated[0], skip_special_tokens=True)
            results[crop_path.name] = text.strip()
        except:
            results[crop_path.name] = ""
    return results

# ── Process pages for HITL ───────────────────────────────────
crops_base = Path.home() / "ocr_project/outputs/line_crops"
hitl_dir = Path.home() / "ocr_project/outputs/hitl_review"
hitl_dir.mkdir(exist_ok=True)

annotation_dir = Path.home() / "ocr_project/data/annotations/hitl"
annotation_dir.mkdir(exist_ok=True)

print("\nCreating HITL review sheets...")

processed = 0
for split in ["print"]:
    split_dir = crops_base / split
    if not split_dir.exists():
        continue
    for source in sorted(split_dir.iterdir()):
        if not source.is_dir():
            continue
        for page_dir in sorted(source.iterdir()):
            if not page_dir.is_dir():
                continue
            crops = list(page_dir.glob("line_*.png"))
            if len(crops) < 5:
                continue

            print(f"  Processing: {source.name}/{page_dir.name}")

            # Run OCR
            ocr_results = run_trocr_on_crops(page_dir)

            # Create review sheet
            sheet_path = hitl_dir / f"{source.name}_{page_dir.name}_review.png"
            create_review_sheet(
                str(page_dir),
                str(page_dir),
                ocr_results,
                str(sheet_path)
            )

            # Create annotation JSON
            ann_path = annotation_dir / f"{source.name}_{page_dir.name}_annotations.json"
            create_annotation_json(
                str(page_dir),
                ocr_results,
                str(ann_path)
            )

            print(f"    Review sheet: {sheet_path.name}")
            print(f"    Annotation JSON: {ann_path.name}")
            processed += 1

            if processed >= 3:
                break
        if processed >= 3:
            break

print(f"\nCreated {processed} review sheets")
print(f"Review sheets: {hitl_dir}")
print(f"Annotation JSONs: {annotation_dir}")
print("\nHITL Workflow:")
print("1. Open review sheet PNG - see line image + OCR prediction")
print("2. Edit annotation JSON - fill human_correction field")
print("3. Run feedback script to add corrected pairs to training data")
print("\nStep 4.1 Complete!")
