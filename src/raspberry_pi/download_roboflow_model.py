"""
Download Helmet Model + Dataset from Roboflow
=============================================
Free account required: https://roboflow.com

This script downloads:
  1. A pre-trained YOLOv8 helmet detection model
  2. The helmet dataset (7000+ labelled images)

Usage:
  python download_roboflow_model.py --api-key YOUR_KEY

Get your free API key:
  1. Go to https://app.roboflow.com
  2. Sign up (free)
  3. Settings → Roboflow API → Copy key
"""

import argparse
import sys
import subprocess
from pathlib import Path


# ── Best helmet datasets on Roboflow Universe ──────────────────────────────
DATASETS = [
    {
        "workspace": "roboflow-100",
        "project":   "hard-hat-universe",
        "version":   2,
        "description": "Hard Hat Universe — 7,035 images, 3 classes (helmet/head/person)",
        "classes": {0: "helmet", 1: "head", 2: "person"}
    },
    {
        "workspace": "joseph-nelson",
        "project":   "helmet-detection",
        "version":   2,
        "description": "Helmet Detection — 4,546 images, 2 classes (helmet/no_helmet)",
        "classes": {0: "helmet", 1: "no_helmet"}
    },
    {
        "workspace": "team-roboflow",
        "project":   "safety-helmet-jvlfw",
        "version":   1,
        "description": "Safety Helmet — 5,000+ images",
        "classes": {0: "helmet", 1: "no_helmet"}
    },
]


def install_roboflow():
    try:
        import roboflow
    except ImportError:
        print("Installing roboflow...")
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "roboflow", "--quiet"])


def download_dataset(api_key: str, choice: int = 0):
    install_roboflow()
    from roboflow import Roboflow

    ds = DATASETS[choice]
    print(f"\nDownloading: {ds['description']}")
    print(f"  workspace : {ds['workspace']}")
    print(f"  project   : {ds['project']}")
    print(f"  version   : {ds['version']}")
    print(f"  classes   : {ds['classes']}\n")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(ds["workspace"]).project(ds["project"])
    dataset = proj.version(ds["version"]).download(
        "yolov8", location="./dataset"
    )

    print(f"\n✅ Dataset downloaded to: ./dataset")
    print(f"   Classes in this dataset:")
    for cls_id, cls_name in ds["classes"].items():
        print(f"     Class {cls_id} = '{cls_name}'")

    # Show what to put in CONFIG
    helmet_ids = {i for i, n in ds["classes"].items()
                  if "helmet" in n.lower() and "no" not in n.lower()}
    print(f"\n   In helmet_detection.py CONFIG, set:")
    print(f'     "helmet_class_ids": {helmet_ids}')

    return dataset, ds["classes"]


def download_trained_model(api_key: str, choice: int = 0):
    """Download the model weights (not just dataset)."""
    install_roboflow()
    from roboflow import Roboflow

    Path("models").mkdir(exist_ok=True)
    ds = DATASETS[choice]

    print(f"\nDownloading trained model weights from Roboflow...")
    rf = Roboflow(api_key=api_key)

    try:
        proj = rf.workspace(ds["workspace"]).project(ds["project"])
        version = proj.version(ds["version"])

        # Download model (if hosted on Roboflow)
        model = version.model
        if model:
            print("✅ Hosted model found. Use it via Roboflow Hosted API,")
            print("   or export to YOLOv8 format:")
            print("   version.export('yolov8')")
        else:
            print("ℹ No hosted model — you need to train from the dataset.")
            print("  Run: python train_model.py --action train --epochs 50")
    except Exception as e:
        print(f"Model download: {e}")
        print("→ Download dataset instead and train locally.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True,
                        help="Roboflow API key (free at roboflow.com)")
    parser.add_argument("--dataset", type=int, default=1,
                        choices=[0, 1, 2],
                        help="Dataset choice: 0=Hard Hat Universe, "
                             "1=Helmet Detection (recommended), 2=Safety Helmet")
    parser.add_argument("--model-only", action="store_true",
                        help="Try to download trained model weights")
    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════╗
║   Roboflow Helmet Dataset/Model Downloader         ║
╚════════════════════════════════════════════════════╝

Available datasets:
""")
    for i, ds in enumerate(DATASETS):
        print(f"  [{i}] {ds['description']}")

    print(f"\n  → Using dataset [{args.dataset}]")

    if args.model_only:
        download_trained_model(args.api_key, args.dataset)
    else:
        dataset, classes = download_dataset(args.api_key, args.dataset)
        print("""
NEXT STEPS:
  1. Train model:
       python train_model.py --action train --epochs 50 --device cpu
  2. Run detection:
       python helmet_detection.py
""")


if __name__ == "__main__":
    main()
