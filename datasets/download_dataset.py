"""
Dataset Download & Preparation Helper
Smart Helmet Bike Ignition System

Supports:
  - Roboflow dataset download (recommended)
  - Manual Kaggle dataset preparation
  - Dataset statistics and visualization
"""

import os
import json
import shutil
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_roboflow(api_key: str, workspace: str, project: str,
                      version: int = 1, location: str = "dataset"):
    """
    Download helmet detection dataset from Roboflow.

    Steps to get free API key:
    1. Go to https://roboflow.com and create free account
    2. Create project OR find existing: universe.roboflow.com
    3. Go to Settings → API Keys → Copy API Key

    Recommended projects:
      - safety-helmet-detection-nnfnf  (7,035 images)
      - helmet-detection-bybhc         (4,546 images)
      - hardhats-2x6mk                 (5,269 images)
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("Install: pip install roboflow")
        return

    logger.info(f"Downloading from Roboflow: {workspace}/{project} v{version}")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=location)
    logger.info(f"Dataset downloaded to: {location}")
    return dataset


def prepare_kaggle_dataset(kaggle_dir: str, output_dir: str = "dataset",
                            val_split: float = 0.2, test_split: float = 0.1):
    """
    Prepare Kaggle Hard Hat Workers dataset for YOLOv8.

    Download manually from:
    https://www.kaggle.com/datasets/andrewmvd/hard-hat-workers
    Extract to kaggle_dir.
    """
    import xml.etree.ElementTree as ET
    import cv2

    CLASS_MAP = {"helmet": 0, "head": 1, "person": 2}
    HELMET_ONLY_CLASSES = {0}  # We only care about helmet class

    img_dir = Path(kaggle_dir) / "images"
    ann_dir = Path(kaggle_dir) / "annotations"

    if not img_dir.exists():
        logger.error(f"Images directory not found: {img_dir}")
        return

    images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    logger.info(f"Found {len(images)} images")

    # Convert Pascal VOC XML → YOLO TXT
    yolo_samples = []
    for img_path in images:
        ann_path = ann_dir / (img_path.stem + ".xml")
        if not ann_path.exists():
            continue

        tree = ET.parse(ann_path)
        root = tree.getroot()
        size = root.find("size")
        W = int(size.find("width").text)
        H = int(size.find("height").text)

        labels = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower()
            if cls_name not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls_name]
            bb = obj.find("bndbox")
            xmin = int(bb.find("xmin").text)
            ymin = int(bb.find("ymin").text)
            xmax = int(bb.find("xmax").text)
            ymax = int(bb.find("ymax").text)
            cx = (xmin + xmax) / 2 / W
            cy = (ymin + ymax) / 2 / H
            bw = (xmax - xmin) / W
            bh = (ymax - ymin) / H
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if labels:
            yolo_samples.append((img_path, "\n".join(labels)))

    logger.info(f"Converted {len(yolo_samples)} samples with annotations")

    # Shuffle & split
    random.shuffle(yolo_samples)
    n = len(yolo_samples)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    splits = {
        "test":  yolo_samples[:n_test],
        "val":   yolo_samples[n_test:n_test+n_val],
        "train": yolo_samples[n_test+n_val:],
    }

    # Write files
    for split, samples in splits.items():
        img_out = Path(output_dir) / "images" / split
        lbl_out = Path(output_dir) / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_text in samples:
            # Resize to 640×640 for faster training
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_resized = cv2.resize(img, (640, 640))
            out_name = img_path.stem
            cv2.imwrite(str(img_out / f"{out_name}.jpg"), img_resized)
            (lbl_out / f"{out_name}.txt").write_text(label_text)

        logger.info(f"  {split}: {len(samples)} samples")

    logger.info(f"Dataset prepared at: {output_dir}/")


def print_dataset_stats(dataset_dir: str = "dataset"):
    """Print statistics about the prepared dataset."""
    logger.info("=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 50)

    total_images = 0
    total_labels = {0: 0, 1: 0}  # 0=helmet, 1=no_helmet

    for split in ("train", "val", "test"):
        img_dir = Path(dataset_dir) / "images" / split
        lbl_dir = Path(dataset_dir) / "labels" / split

        if not img_dir.exists():
            continue

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        split_labels = {0: 0, 1: 0}

        for lbl_path in lbl_dir.glob("*.txt"):
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.split()
                if parts:
                    cls = int(parts[0])
                    split_labels[cls] = split_labels.get(cls, 0) + 1
                    total_labels[cls] = total_labels.get(cls, 0) + 1

        logger.info(f"  {split:6s}: {len(imgs):4d} images | "
                    f"helmet={split_labels.get(0,0)} "
                    f"no_helmet={split_labels.get(1,0)}")
        total_images += len(imgs)

    logger.info("-" * 50)
    logger.info(f"  {'TOTAL':6s}: {total_images:4d} images | "
                f"helmet={total_labels.get(0,0)} "
                f"no_helmet={total_labels.get(1,0)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action",
                        choices=["roboflow", "kaggle", "stats"],
                        default="stats")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--workspace", default="joseph-nelson")
    parser.add_argument("--project", default="helmet-detection")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--kaggle-dir", default="kaggle_raw")
    parser.add_argument("--output", default="dataset")
    args = parser.parse_args()

    if args.action == "roboflow":
        if not args.api_key:
            logger.error("Provide --api-key (get free key at roboflow.com)")
        else:
            download_roboflow(args.api_key, args.workspace,
                              args.project, args.version, args.output)
    elif args.action == "kaggle":
        prepare_kaggle_dataset(args.kaggle_dir, args.output)
    elif args.action == "stats":
        print_dataset_stats(args.output)
