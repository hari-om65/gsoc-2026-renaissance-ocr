import os
import json
import torch
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.structures import BoxMode
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

def load_coco_to_detectron2(json_path):
    with open(json_path) as f:
        coco = json.load(f)
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
    dataset_dicts = []
    for img_info in coco["images"]:
        iid = img_info["id"]
        if iid not in ann_by_img:
            continue
        record = {
            "file_name": img_info["file_name"],
            "image_id": iid,
            "height": img_info["height"],
            "width": img_info["width"],
            "annotations": []
        }
        for ann in ann_by_img[iid]:
            x, y, w, h = ann["bbox"]
            obj = {
                "bbox": [x, y, x + w, y + h],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann["segmentation"],
                "category_id": 0,
            }
            record["annotations"].append(obj)
        if record["annotations"]:
            dataset_dicts.append(record)
    return dataset_dicts

ann_path = str(Path.home() / "ocr_project/data/annotations/coco_annotations.json")

for name in ["ocr_train", "ocr_val"]:
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)

def get_train_data():
    data = load_coco_to_detectron2(ann_path)
    split = int(len(data) * 0.85)
    return data[:split]

def get_val_data():
    data = load_coco_to_detectron2(ann_path)
    split = int(len(data) * 0.85)
    return data[split:]

DatasetCatalog.register("ocr_train", get_train_data)
MetadataCatalog.get("ocr_train").set(thing_classes=["text_line"])
DatasetCatalog.register("ocr_val", get_val_data)
MetadataCatalog.get("ocr_val").set(thing_classes=["text_line"])

print(f"Train: {len(get_train_data())} images")
print(f"Val: {len(get_val_data())} images")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ocr_train",)
cfg.DATASETS.TEST = ("ocr_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = (700, 900)
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = str(Path.home() / "ocr_project/models/segmentation")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("Starting Mask R-CNN training (1000 iters)...")
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
print("Training complete!")
