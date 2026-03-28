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

def read_docx(docx_path):
    """Extract all text lines from a .docx file."""
    doc = Document(str(docx_path))
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)
    return lines

def ocr_line(image_path):
    """Run Tesseract OCR on a single line crop."""
    img = PIL.Image.open(str(image_path))
    config = '--psm 7 --oem 3 -l spa'
    try:
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except:
        return ""

def align_lines(ocr_lines, gt_lines):
    """Match OCR output to ground truth using edit distance."""
    pairs = []
    used_gt = set()
    
    for ocr_text, img_path in ocr_lines:
        if not ocr_text:
            continue
        
        best_score = float('inf')
        best_gt_idx = -1
        
        for gt_idx, gt_text in enumerate(gt_lines):
            if gt_idx in used_gt:
                continue
            if not gt_text:
                continue
            
            # Normalize both strings for comparison
            ocr_norm = ocr_text.lower().replace(" ", "")
            gt_norm = gt_text.lower().replace(" ", "")
            
            if len(ocr_norm) == 0 or len(gt_norm) == 0:
                continue
            
            dist = levenshtein_distance(ocr_norm[:30], gt_norm[:30])
            
            if dist < best_score:
                best_score = dist
                best_gt_idx = gt_idx
        
        # Only accept if edit distance is reasonable
        if best_gt_idx >= 0 and best_score < 15:
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

# Process print sources
print_trans_dir = trans_dir / "print/Print"
print_crops_dir = crops_dir / "print"

print("=== Processing Print Sources ===")
if print_trans_dir.exists() and print_crops_dir.exists():
    for docx_path in sorted(print_trans_dir.glob("*.docx")):
        source_name = docx_path.stem.replace(" transcription", "").replace(" Transcription", "").strip()
        print(f"\nProcessing: {source_name}")
        
        # Read ground truth
        gt_lines = read_docx(docx_path)
        print(f"  GT lines: {len(gt_lines)}")
        
        # Find matching crop folder
        matching_dir = None
        for d in print_crops_dir.iterdir():
            if d.is_dir() and any(part in d.name for part in source_name.split()[:2]):
                matching_dir = d
                break
        
        if not matching_dir:
            print(f"  No crop folder found for {source_name}")
            continue
        
        # Get all line crops from all pages
        all_crops = []
        for page_dir in sorted(matching_dir.iterdir()):
            if page_dir.is_dir():
                crops = sorted(page_dir.glob("line_*.png"))
                all_crops.extend(crops)
        
        print(f"  Found {len(all_crops)} line crops")
        
        # Run OCR on crops
        print(f"  Running Tesseract OCR...")
        ocr_results = []
        for crop_path in all_crops[:200]:
            ocr_text = ocr_line(crop_path)
            ocr_results.append((ocr_text, crop_path))
        
        # Align with GT
        pairs = align_lines(ocr_results, gt_lines)
        print(f"  Matched {len(pairs)} pairs")
        all_pairs.extend(pairs)

print(f"\n=== TOTAL PAIRS: {len(all_pairs)} ===")

# Save pairs
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, indent=2, ensure_ascii=False)

print(f"Saved to: {out_path}")

# Show sample pairs
print("\n=== SAMPLE PAIRS ===")
for p in all_pairs[:5]:
    print(f"  GT:  {p['gt_text'][:60]}")
    print(f"  OCR: {p['ocr_text'][:60]}")
    print(f"  Edit distance: {p['edit_distance']}")
    print()
