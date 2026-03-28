import os
import json
import cv2
import numpy as np
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
    img_lookup = {img["id"]: img for img in coco["images"]}
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

train_data = get_train_data()
val_data = get_val_data()
print(f"Train images: {len(train_data)}")
print(f"Val images: {len(val_data)}")
print(f"Train annotations: {sum(len(d['annotations']) for d in train_data)}")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("ocr_train",)
cfg.DATASETS.TEST = ("ocr_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0005
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = (2000, 2500)
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
cfg.OUTPUT_DIR = str(Path.home() / "ocr_project/models/segmentation")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("Starting Mask R-CNN training...")
print(f"Output dir: {cfg.OUTPUT_DIR}")
print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Training complete!")
print(f"Model saved to: {cfg.OUTPUT_DIR}")
