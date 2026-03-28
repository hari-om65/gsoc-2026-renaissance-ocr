import cv2
import torch
import numpy as np
import json
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Load trained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = str(Path.home() / "ocr_project/models/segmentation/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)
print("Model loaded!")

def sort_lines_reading_order(boxes, page_width):
    """Sort lines top-to-bottom, detecting multi-column layouts."""
    if len(boxes) == 0:
        return []
    
    indices = list(range(len(boxes)))
    centers_x = [(boxes[i][0] + boxes[i][2]) / 2 for i in indices]
    centers_y = [(boxes[i][1] + boxes[i][3]) / 2 for i in indices]
    
    # Detect columns: if centers cluster in left/right halves
    mid_x = page_width / 2
    left_indices = [i for i in indices if centers_x[i] < mid_x * 0.95]
    right_indices = [i for i in indices if centers_x[i] >= mid_x * 0.95]
    
    # Check if truly two-column (significant lines on both sides)
    if len(left_indices) > 3 and len(right_indices) > 3:
        # Two-column: sort each column top-to-bottom, left column first
        left_sorted = sorted(left_indices, key=lambda i: centers_y[i])
        right_sorted = sorted(right_indices, key=lambda i: centers_y[i])
        return left_sorted + right_sorted
    else:
        # Single column: sort top-to-bottom
        return sorted(indices, key=lambda i: centers_y[i])

def filter_marginalia(boxes, page_width, page_height, margin_pct=0.12):
    """Remove detections in outer margins."""
    filtered = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        
        # Skip if too narrow (likely margin note)
        if w < page_width * 0.15:
            continue
        # Skip if centroid is in outer margin
        if cx < page_width * margin_pct or cx > page_width * (1 - margin_pct):
            if w < page_width * 0.3:
                continue
        filtered.append(i)
    return filtered

def crop_and_save_lines(image_path, output_dir, predictor):
    """Run model, crop lines, save in reading order."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    
    h, w = img.shape[:2]
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    
    if len(instances) == 0:
        return 0
    
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    
    # Filter marginalia
    valid_indices = filter_marginalia(boxes, w, h)
    if not valid_indices:
        return 0
    
    boxes_filtered = boxes[valid_indices]
    scores_filtered = scores[valid_indices]
    
    # Sort in reading order
    sorted_order = sort_lines_reading_order(boxes_filtered, w)
    
    # Crop and save each line
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    
    for line_num, idx in enumerate(sorted_order, start=1):
        x1, y1, x2, y2 = boxes_filtered[idx].astype(int)
        # Add small padding
        pad = 3
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        line_crop = img[y1:y2, x1:x2]
        if line_crop.size == 0:
            continue
        
        crop_name = f"line_{line_num:04d}.png"
        cv2.imwrite(str(output_dir / crop_name), line_crop)
        
        metadata.append({
            "line_num": line_num,
            "file": crop_name,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": float(scores_filtered[idx]),
            "source_image": str(image_path)
        })
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return len(metadata)

# Process all pages
pages_dir = Path.home() / "ocr_project/data/pages"
crops_dir = Path.home() / "ocr_project/outputs/line_crops"

total_lines = 0
total_pages = 0

for split in ["print", "handwriting"]:
    split_dir = pages_dir / split
    if not split_dir.exists():
        continue
    for source in sorted(split_dir.iterdir()):
        if not source.is_dir():
            continue
        pages = sorted(source.glob("*.png"))[:5]
        for page_path in pages:
            out_dir = crops_dir / split / source.name / page_path.stem
            n_lines = crop_and_save_lines(page_path, out_dir, predictor)
            total_lines += n_lines
            total_pages += 1
            print(f"  {source.name}/{page_path.name}: {n_lines} lines cropped")

print(f"\nTotal pages processed: {total_pages}")
print(f"Total lines cropped: {total_lines}")
print(f"Crops saved to: {crops_dir}")
