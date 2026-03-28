import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

with open(Path.home() / "ocr_project/data/annotations/coco_annotations.json") as f:
    coco = json.load(f)

# Build lookup
img_to_anns = {}
for ann in coco["annotations"]:
    iid = ann["image_id"]
    img_to_anns.setdefault(iid, []).append(ann)

out_dir = Path.home() / "ocr_project/outputs/annotation_viz"
out_dir.mkdir(exist_ok=True)

# Visualize first 6 images that have annotations
count = 0
for img_info in coco["images"]:
    iid = img_info["id"]
    if iid not in img_to_anns:
        continue
    anns = img_to_anns[iid]
    if len(anns) < 5:
        continue

    img = cv2.imread(img_info["file_name"])
    if img is None:
        continue

    for ann in anns:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 14))
    plt.imshow(img_rgb)
    plt.title(f"{Path(img_info['file_name']).parent.name} - {len(anns)} lines")
    plt.axis('off')
    out_path = out_dir / f"viz_{iid:03d}.png"
    plt.savefig(out_path, dpi=72, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path} ({len(anns)} lines)")

    count += 1
    if count >= 6:
        break

print("Done!")
