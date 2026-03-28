import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer, wer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from docx import Document
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# ── Load Mask R-CNN ──────────────────────────────────────────
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = str(Path.home() / "ocr_project/models/segmentation/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
cfg.MODEL.DEVICE = "cuda"
segmentor = DefaultPredictor(cfg)
print("Segmentation model loaded!")

# ── Load TrOCR ───────────────────────────────────────────────
model_dir = Path.home() / "ocr_project/models/trocr_finetuned"
processor = TrOCRProcessor.from_pretrained(str(model_dir))
trocr = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
device = torch.device("cuda")
trocr.to(device)
trocr.eval()
print("TrOCR model loaded!")

def segment_and_sort(image_path):
    """Segment page into sorted line crops."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    outputs = segmentor(img)
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        return [], img
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()

    # Filter narrow boxes
    valid = [i for i, b in enumerate(boxes)
             if (b[2]-b[0]) > w * 0.15]
    if not valid:
        return [], img
    boxes = boxes[valid]
    scores = scores[valid]

    # Sort top-to-bottom
    centers_y = [(b[1]+b[3])/2 for b in boxes]
    sorted_idx = np.argsort(centers_y)
    return [(boxes[i], scores[i]) for i in sorted_idx], img

def ocr_line(img, box):
    """Run TrOCR on a single line crop."""
    x1, y1, x2, y2 = box.astype(int)
    pad = 3
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(img.shape[1], x2+pad)
    y2 = min(img.shape[0], y2+pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pixel_values = processor(
        crop_pil, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated = trocr.generate(pixel_values)
    text = processor.tokenizer.decode(
        generated[0], skip_special_tokens=True)
    return text.strip()

def read_docx_lines(docx_path):
    """Extract GT lines from docx."""
    doc = Document(str(docx_path))
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text or text.startswith("PDF p") or text.startswith("NOTES:"):
            continue
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 3:
                lines.append(line)
    return lines

# ── Test on unseen pages ─────────────────────────────────────
test_cases = [
    {
        "page": Path.home() / "ocr_project/data/pages/print/PORCONES.23.5_-_1628/page_0004.png",
        "docx": Path.home() / "ocr_project/data/transcriptions/print/Print/PORCONES.23.5 - 1628 transcription.docx",
        "name": "PORCONES.23.5 page 4"
    },
    {
        "page": Path.home() / "ocr_project/data/pages/print/Buendia_-_Instruccion/page_0004.png",
        "docx": Path.home() / "ocr_project/data/transcriptions/print/Print/Buendia - Instruccion transcription.docx",
        "name": "Buendia page 4"
    },
]

out_dir = Path.home() / "ocr_project/outputs/end_to_end_eval"
out_dir.mkdir(exist_ok=True)

all_results = []

for case in test_cases:
    if not case["page"].exists():
        print(f"Page not found: {case['page']}")
        continue

    print(f"\n=== Processing: {case['name']} ===")

    # Step 1: Segment
    line_data, img = segment_and_sort(case["page"])
    print(f"  Detected {len(line_data)} lines")

    # Step 2: OCR each line
    predicted_lines = []
    for box, score in line_data:
        text = ocr_line(img, box)
        if text:
            predicted_lines.append(text)

    predicted_text = " ".join(predicted_lines)
    print(f"  OCR output ({len(predicted_lines)} lines):")
    print(f"  {predicted_text[:200]}...")

    # Step 3: Load GT
    gt_lines = read_docx_lines(case["docx"])
    gt_text = " ".join(gt_lines)
    print(f"  GT ({len(gt_lines)} lines):")
    print(f"  {gt_text[:200]}...")

    # Step 4: Compute metrics
    if predicted_text and gt_text:
        page_cer = cer(gt_text, predicted_text)
        page_wer = wer(gt_text, predicted_text)
        print(f"  CER: {page_cer:.4f} ({page_cer*100:.1f}%)")
        print(f"  WER: {page_wer:.4f} ({page_wer*100:.1f}%)")
    else:
        page_cer = 1.0
        page_wer = 1.0
        print("  No text detected!")

    # Step 5: Save visualization
    result_img = img.copy()
    for box, score in line_data:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(result_img, (x1,y1), (x2,y2), (0,255,0), 2)

    plt.figure(figsize=(12,16))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{case['name']} | CER: {page_cer*100:.1f}% | WER: {page_wer*100:.1f}%")
    plt.axis('off')
    plt.savefig(out_dir / f"{case['name'].replace(' ','_')}.png",
                dpi=100, bbox_inches='tight')
    plt.close()

    all_results.append({
        "name": case["name"],
        "cer": page_cer,
        "wer": page_wer,
        "predicted_lines": predicted_lines,
        "num_lines_detected": len(line_data)
    })

# Final summary
print("\n=== FINAL END-TO-END SUMMARY ===")
for r in all_results:
    print(f"{r['name']}: CER={r['cer']*100:.1f}% | WER={r['wer']*100:.1f}% | Lines={r['num_lines_detected']}")

with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {out_dir}")
print("\nPhase 2 Step 2.5 COMPLETE!")
