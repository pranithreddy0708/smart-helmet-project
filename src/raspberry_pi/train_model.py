"""
Smart Helmet Bike Ignition System
Model Training Script — YOLOv8 Fine-Tuning on Helmet Dataset
Author: Final Year Project Team

Dataset sources:
  1. Kaggle: Safety Helmet Detection Dataset
     https://www.kaggle.com/datasets/andrewmvd/hard-hat-workers
  2. Roboflow Universe: Helmet Detection
     https://universe.roboflow.com/joseph-nelson/helmet-detection
  3. Open Images Dataset (Google) - filtered for 'Helmet' class

Usage:
  python train_model.py --epochs 50 --batch 16 --imgsz 640
"""

import argparse
import os
import yaml
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DEFAULT TRAINING CONFIG
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model": "yolov8n.pt",          # nano (fastest) | yolov8s.pt | yolov8m.pt
    "data": "dataset/helmet.yaml",
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "device": "cpu",                # "0" for GPU, "cpu" for CPU
    "workers": 4,
    "patience": 10,                 # Early stopping patience
    "save_period": 10,
    "project": "runs/train",
    "name": "helmet_v1",
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "Adam",
    "lr0": 0.001,
    "augment": True,
    "degrees": 10,
    "flipud": 0.2,
    "fliplr": 0.5,
    "mosaic": 0.8,
}


# ─────────────────────────────────────────────
# DATASET YAML GENERATOR
# ─────────────────────────────────────────────
def create_dataset_yaml(dataset_root: str, output_path: str = "dataset/helmet.yaml"):
    """Generate YOLO-format dataset YAML configuration."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    yaml_content = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,                    # Number of classes
        "names": {
            0: "helmet",
            1: "no_helmet"
        }
    }

    with open(output_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Dataset YAML created: {output_path}")
    logger.info(f"  Classes: {yaml_content['names']}")
    return output_path


# ─────────────────────────────────────────────
# DATASET DIRECTORY SCAFFOLD
# ─────────────────────────────────────────────
def scaffold_dataset_structure(root: str = "dataset"):
    """Create the expected directory structure for the dataset."""
    dirs = [
        f"{root}/images/train",
        f"{root}/images/val",
        f"{root}/images/test",
        f"{root}/labels/train",
        f"{root}/labels/val",
        f"{root}/labels/test",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    readme = Path(f"{root}/README.md")
    readme.write_text("""# Helmet Detection Dataset

## Structure
```
dataset/
  images/
    train/    ← Training images (.jpg / .png)
    val/      ← Validation images
    test/     ← Test images
  labels/
    train/    ← YOLO label files (.txt, same name as image)
    val/
    test/
  helmet.yaml
```

## Label Format (YOLO)
Each .txt file contains one detection per line:
  <class_id> <cx> <cy> <width> <height>
  All values normalized 0.0–1.0.

  Class 0 = helmet
  Class 1 = no_helmet (bare head visible)

## Recommended Datasets
1. Kaggle Hard Hat Workers:
   https://www.kaggle.com/datasets/andrewmvd/hard-hat-workers

2. Roboflow Helmet Detection (4,500+ images):
   https://universe.roboflow.com/joseph-nelson/helmet-detection

3. Safety Helmet Detection (12,000+ images):
   https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset

## Download via Roboflow (recommended)
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace().project("helmet-detection")
dataset = project.version(1).download("yolov8")
```
""")
    logger.info(f"Dataset scaffold created at '{root}/'")


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────
def augment_dataset(source_dir: str, output_dir: str, multiplier: int = 3):
    """
    Augment training images using Albumentations.
    Run this if your dataset is small (< 500 images).
    """
    try:
        import albumentations as A
        import cv2
        import numpy as np
    except ImportError:
        logger.error("Install: pip install albumentations opencv-python")
        return

    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.4),
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.RandomRain(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    img_dir = Path(source_dir) / "images" / "train"
    lbl_dir = Path(source_dir) / "labels" / "train"
    out_img = Path(output_dir) / "images" / "train"
    out_lbl = Path(output_dir) / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in img_dir.glob("*.jpg"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        bboxes, classes = [], []
        for line in lbl_path.read_text().strip().split("\n"):
            parts = line.split()
            if len(parts) == 5:
                classes.append(int(parts[0]))
                bboxes.append([float(p) for p in parts[1:]])

        for i in range(multiplier):
            try:
                aug = transform(image=img, bboxes=bboxes, class_labels=classes)
                out_name = f"{img_path.stem}_aug{i}"
                cv2.imwrite(str(out_img / f"{out_name}.jpg"), aug["image"])
                with open(out_lbl / f"{out_name}.txt", "w") as f:
                    for cls, bb in zip(aug["class_labels"], aug["bboxes"]):
                        f.write(f"{cls} {' '.join(f'{v:.6f}' for v in bb)}\n")
                count += 1
            except Exception as e:
                logger.warning(f"Augmentation failed for {img_path.name}: {e}")

    logger.info(f"Augmentation complete: {count} new images generated")


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(config: dict):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Install YOLOv8: pip install ultralytics")
        return None

    logger.info("=" * 60)
    logger.info("  Starting YOLOv8 Helmet Detection Training")
    logger.info("=" * 60)
    logger.info(f"  Model : {config['model']}")
    logger.info(f"  Data  : {config['data']}")
    logger.info(f"  Epochs: {config['epochs']}")
    logger.info(f"  Batch : {config['batch']}")
    logger.info(f"  Image : {config['imgsz']}x{config['imgsz']}")
    logger.info(f"  Device: {config['device']}")
    logger.info("=" * 60)

    model = YOLO(config["model"])

    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        device=config["device"],
        workers=config["workers"],
        patience=config["patience"],
        save_period=config["save_period"],
        project=config["project"],
        name=config["name"],
        exist_ok=config["exist_ok"],
        pretrained=config["pretrained"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        augment=config["augment"],
        degrees=config["degrees"],
        flipud=config["flipud"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        verbose=True,
    )

    logger.info("Training complete!")
    best_model = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    logger.info(f"Best model saved: {best_model}")
    return results


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def validate(model_path: str, data_yaml: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Install: pip install ultralytics")
        return

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    logger.info("=" * 60)
    logger.info("  VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  mAP50    : {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95 : {metrics.box.map:.4f}")
    logger.info(f"  Precision: {metrics.box.p.mean():.4f}")
    logger.info(f"  Recall   : {metrics.box.r.mean():.4f}")
    logger.info("=" * 60)
    return metrics


# ─────────────────────────────────────────────
# EXPORT TO TFLite / ONNX
# ─────────────────────────────────────────────
def export_model(model_path: str, format: str = "tflite"):
    """
    Export trained YOLOv8 model to deployment format.
    Formats: tflite, onnx, openvino, coreml, engine (TensorRT)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Install: pip install ultralytics")
        return

    model = YOLO(model_path)
    exported = model.export(
        format=format,
        imgsz=640,
        int8=(format == "tflite"),   # INT8 quantization for TFLite
        data="dataset/helmet.yaml"
    )
    logger.info(f"Model exported to {format}: {exported}")
    return exported


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 Helmet Detection Model"
    )
    parser.add_argument("--action", choices=["train", "val", "export", "scaffold"],
                        default="train")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    parser.add_argument("--data", default=DEFAULT_CONFIG["data"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--export-format", default="tflite",
                        choices=["tflite", "onnx", "openvino"])
    parser.add_argument("--model-path", help="Path to trained model for val/export")
    return parser.parse_args()


def main():
    args = parse_args()
    config = {**DEFAULT_CONFIG}
    config.update({
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
    })

    if args.action == "scaffold":
        scaffold_dataset_structure()
        create_dataset_yaml("dataset")

    elif args.action == "train":
        if not Path(config["data"]).exists():
            logger.warning("Dataset YAML not found. Scaffolding structure...")
            scaffold_dataset_structure()
            create_dataset_yaml("dataset")
            logger.info("Please add your images/labels to dataset/ and re-run.")
            return
        train(config)

    elif args.action == "val":
        model_path = args.model_path or f"{config['project']}/{config['name']}/weights/best.pt"
        validate(model_path, config["data"])

    elif args.action == "export":
        model_path = args.model_path or f"{config['project']}/{config['name']}/weights/best.pt"
        export_model(model_path, args.export_format)


if __name__ == "__main__":
    main()
