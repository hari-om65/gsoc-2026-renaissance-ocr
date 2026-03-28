import os
import json
import pytesseract
import cv2
import numpy as np
from pathlib import Path
from docx import Document
from Levenshtein import distance as levenshtein_distance
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

def read_docx_lines(docx_path):
    """Extract individual text lines from docx, skipping page markers."""
    doc = Document(str(docx_path))
    all_lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Skip page markers and notes
        if text.startswith("PDF p") or text.startswith("NOTES:"):
            continue
        # Split paragraph into individual lines
        sub_lines = text.split('\n')
        for line in sub_lines:
            line = line.strip()
            if len(line) > 3:
                all_lines.append(line)
    return all_lines

def ocr_line(image_path):
    """Run Tesseract OCR on a single line crop."""
    img = PIL.Image.open(str(image_path))
    config = '--psm 7 --oem 3 -l spa'
    try:
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except:
        return ""

def align_lines(ocr_results, gt_lines):
    """Match OCR output to ground truth using edit distance."""
    pairs = []
    used_gt = set()

    for ocr_text, img_path in ocr_results:
        if not ocr_text or len(ocr_text) < 3:
            continue

        best_score = float('inf')
        best_gt_idx = -1

        ocr_norm = ocr_text.lower().replace(" ", "")[:40]

        for gt_idx, gt_text in enumerate(gt_lines):
            if gt_idx in used_gt or not gt_text:
                continue
            gt_norm = gt_text.lower().replace(" ", "")[:40]
            if len(gt_norm) < 3:
                continue
            dist = levenshtein_distance(ocr_norm, gt_norm)
            if dist < best_score:
                best_score = dist
                best_gt_idx = gt_idx

        max_allowed = max(5, len(ocr_norm) // 3)
        if best_gt_idx >= 0 and best_score <= max_allowed:
            pairs.append({
                "image_path": str(img_path),
                "gt_text": gt_lines[best_gt_idx],
                "ocr_text": ocr_text,
                "edit_distance": best_score
            })
            used_gt.add(best_gt_idx)

    return pairs

# Paths
crops_dir = Path.home() / "ocr_project/outputs/line_crops"
trans_dir = Path.home() / "ocr_project/data/transcriptions"
out_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"

all_pairs = []

# Source name mapping
source_map = {
    "Buendia": "Buendia_-_Instruccion",
    "Covarrubias": "Covarrubias_-_Tesoro_lengua",
    "Guardiola": "Guardiola_-_Tratado_nobleza",
    "PORCONES.228": "PORCONES.228.38",
    "PORCONES.23": "PORCONES.23.5_-_1628",
    "PORCONES.748": "PORCONES.748.6",
}

print_trans_dir = trans_dir / "print/Print"
print_crops_dir = crops_dir / "print"

print("=== Processing Print Sources ===")
for docx_path in sorted(print_trans_dir.glob("*.docx")):
    print(f"\nReading: {docx_path.name}")

    gt_lines = read_docx_lines(docx_path)
    print(f"  GT lines extracted: {len(gt_lines)}")
    if gt_lines:
        print(f"  Sample GT: {gt_lines[0][:60]}")

    # Find matching crop folder
    matching_dir = None
    doc_stem = docx_path.stem
    for key, folder in source_map.items():
        if key in doc_stem:
            matching_dir = print_crops_dir / folder
            break

    if not matching_dir or not matching_dir.exists():
        print(f"  No crop folder found")
        continue

    # Get all crops
    all_crops = []
    for page_dir in sorted(matching_dir.iterdir()):
        if page_dir.is_dir():
            all_crops.extend(sorted(page_dir.glob("line_*.png")))

    print(f"  Found {len(all_crops)} crops")

    # OCR on crops
    print(f"  Running Tesseract...")
    ocr_results = []
    for crop_path in all_crops[:150]:
        ocr_text = ocr_line(crop_path)
        ocr_results.append((ocr_text, crop_path))

    pairs = align_lines(ocr_results, gt_lines)
    print(f"  Matched: {len(pairs)} pairs")
    all_pairs.extend(pairs)

print(f"\n=== TOTAL PAIRS: {len(all_pairs)} ===")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, indent=2, ensure_ascii=False)

print(f"Saved to: {out_path}")

print("\n=== SAMPLE PAIRS ===")
good_pairs = [p for p in all_pairs if p['edit_distance'] < 5]
for p in good_pairs[:8]:
    print(f"  GT:  {p['gt_text'][:70]}")
    print(f"  OCR: {p['ocr_text'][:70]}")
    print(f"  Edit dist: {p['edit_distance']}")
    print()
