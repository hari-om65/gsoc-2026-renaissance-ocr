import cv2
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

crops_dir = Path.home() / "ocr_project/outputs/line_crops"
out_dir = Path.home() / "ocr_project/outputs/crop_validation"
out_dir.mkdir(exist_ok=True)

# Find sources with good number of crops
sources_to_check = []
for split in ["print", "handwriting"]:
    split_dir = crops_dir / split
    if not split_dir.exists():
        continue
    for source in sorted(split_dir.iterdir()):
        if not source.is_dir():
            continue
        for page_dir in sorted(source.iterdir()):
            if not page_dir.is_dir():
                continue
            crops = sorted(page_dir.glob("line_*.png"))
            if len(crops) >= 10:
                sources_to_check.append((page_dir, crops))

print(f"Found {len(sources_to_check)} pages with 10+ line crops")

# Visualize first 3 pages
for page_dir, crops in sources_to_check[:3]:
    # Show first 12 lines
    show_crops = crops[:12]
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    fig.suptitle(f"{page_dir.parent.name} / {page_dir.name} ({len(crops)} lines)", fontsize=12)
    
    for i, ax in enumerate(axes.flat):
        if i < len(show_crops):
            img = cv2.imread(str(show_crops[i]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(show_crops[i].name, fontsize=8)
        ax.axis('off')
    
    out_path = out_dir / f"validate_{page_dir.parent.name}_{page_dir.name}.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# Print summary statistics
print("\n=== CROP SUMMARY ===")
total = 0
for split in ["print", "handwriting"]:
    split_dir = crops_dir / split
    if not split_dir.exists():
        continue
    count = len(list(split_dir.rglob("line_*.png")))
    total += count
    print(f"{split}: {count} line crops")
print(f"TOTAL: {total} line crops")

# Check average crop dimensions
widths = []
heights = []
sample_crops = list(crops_dir.rglob("line_*.png"))[:100]
for crop_path in sample_crops:
    img = cv2.imread(str(crop_path))
    if img is not None:
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

if widths:
    print(f"\nAvg crop size: {np.mean(widths):.0f} x {np.mean(heights):.0f} pixels")
    print(f"Min crop size: {np.min(widths)} x {np.min(heights)} pixels")
    print(f"Max crop size: {np.max(widths)} x {np.max(heights)} pixels")

print("\nPhase 1 validation complete!")
