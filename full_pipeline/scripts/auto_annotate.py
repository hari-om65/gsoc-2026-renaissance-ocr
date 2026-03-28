import cv2
import json
import numpy as np
from pathlib import Path
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

def auto_annotate_page(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path}")
        return []
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 15
    )
    
    # Remove noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    
    # Connect text horizontally to form lines
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 2))
    connected = cv2.dilate(binary, kernel_line, iterations=2)
    
    # Find line contours
    contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    lines = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Filter: must be wide enough and reasonable height
        if cw > w * 0.15 and ch < h * 0.08 and ch > 5:
            # Add small padding
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            lines.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                "category_id": 1,
                "area": (x2 - x1) * (y2 - y1)
            })
    
    # Sort top to bottom
    lines.sort(key=lambda l: l["bbox"][1])
    return lines, w, h

# Build COCO dataset
coco = {
    "info": {"description": "17th century Spanish OCR line annotations"},
    "categories": [{"id": 1, "name": "text_line"}],
    "images": [],
    "annotations": []
}

img_id = 0
ann_id = 0

pages_dir = Path.home() / "ocr_project/data/pages"
selected_pages = []

for split in ["print", "handwriting"]:
    split_dir = pages_dir / split
    if not split_dir.exists():
        continue
    for source in sorted(split_dir.iterdir()):
        if source.is_dir():
            pages = sorted(source.glob("*.png"))[:5]
            selected_pages.extend(pages)

print(f"Auto-annotating {len(selected_pages)} pages...")

for page_path in selected_pages:
    result = auto_annotate_page(page_path)
    if not result:
        continue
    lines, w, h = result
    
    img_id += 1
    coco["images"].append({
        "id": img_id,
        "file_name": str(page_path),
        "width": w,
        "height": h
    })
    
    for line in lines:
        ann_id += 1
        line["id"] = ann_id
        line["image_id"] = img_id
        coco["annotations"].append(line)
    
    print(f"  {page_path.name}: {len(lines)} lines detected")

# Save
out_path = Path.home() / "ocr_project/data/annotations/coco_annotations.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(coco, f, indent=2)

print(f"\nTotal images: {len(coco['images'])}")
print(f"Total annotations: {len(coco['annotations'])}")
print(f"Saved to: {out_path}")
