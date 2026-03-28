import cv2
import torch
import numpy as np
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = str(Path.home() / "ocr_project/models/segmentation/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)
print("Model loaded!")

test_pages = [
    Path.home() / "ocr_project/data/pages/print/Buendia_-_Instruccion/page_0003.png",
    Path.home() / "ocr_project/data/pages/handwriting/Pleito_entre_el_Marqués_de_Viana/page_0002.png",
    Path.home() / "ocr_project/data/pages/print/Guardiola_-_Tratado_nobleza/page_0002.png",
]

out_dir = Path.home() / "ocr_project/outputs/model_predictions"
out_dir.mkdir(exist_ok=True)

for page_path in test_pages:
    if not page_path.exists():
        print(f"Not found: {page_path}")
        continue
    
    img = cv2.imread(str(page_path))
    outputs = predictor(img)
    
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    
    result = img.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"{score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    out_path = out_dir / f"pred_{page_path.parent.name}_{page_path.name}"
    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"{page_path.parent.name} - {len(boxes)} lines detected")
    plt.axis('off')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Detected {len(boxes)} lines (scores: {scores.min():.2f}-{scores.max():.2f}) -> {out_path}")

print("Done! Check outputs/model_predictions/")
