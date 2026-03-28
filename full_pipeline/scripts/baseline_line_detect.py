import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def detect_lines(image_path, out_path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    line_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > img.shape[1] * 0.2 and h < 100:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            line_boxes.append((x, y, w, h))
    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected {len(line_boxes)} lines")
    plt.axis('off')
    plt.savefig(out_path, dpi=72, bbox_inches='tight')
    plt.close()
    print(f"Detected {len(line_boxes)} lines -> {out_path}")
    return line_boxes

out_dir = Path.home() / "ocr_project/outputs/line_detection_baseline"
out_dir.mkdir(exist_ok=True)

test_pages = [
    Path.home() / "ocr_project/data/pages/print/Buendia_-_Instruccion/page_0002.png",
    Path.home() / "ocr_project/data/pages/handwriting/Pleito_entre_el_Marqués_de_Viana/page_0001.png",
]

for page in test_pages:
    if page.exists():
        detect_lines(page, out_dir / f"{page.parent.name}_{page.name}_lines.png")
    else:
        print(f"Not found: {page}")

print("Done!")
