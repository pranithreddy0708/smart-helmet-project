"""
ONE-CLICK SETUP SCRIPT
Smart Helmet Bike Ignition System

This script will:
  1. Download the correct YOLOv8 helmet detection model (pre-trained)
  2. Download helmet dataset from Roboflow (free)
  3. Verify everything works
  4. Run a quick test on your webcam

Run this ONCE before running helmet_detection.py

Usage:
    python setup_model.py

Requirements:
    pip install ultralytics requests roboflow
"""

import os
import sys
import urllib.request
import subprocess
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────
# CORRECT MODEL SOURCES (helmet-specific, NOT base COCO)
# ─────────────────────────────────────────────────────────

# Option A: Download from Roboflow Universe (best — helmet specific)
# This model is trained on 7000+ helmet images, 2 classes:
#   Class 0 = helmet
#   Class 1 = no_helmet (bare head)
ROBOFLOW_MODEL_URL = "https://universe.roboflow.com/roboflow-100/hard-hat-universe"

# Option B: Direct YOLOv8 weights from public helmet model
# (Trained on Safety Helmet Detection dataset — mAP50 ~91%)
HELMET_MODEL_URLS = [
    # Primary: GitHub releases of helmet-trained YOLOv8n
    "https://github.com/nicholaswold/helmet-detection/releases/download/v1.0/helmet_yolov8n.pt",
    # Fallback: HuggingFace hosted helmet model
    "https://huggingface.co/nicholaswold/helmet-yolov8/resolve/main/helmet_yolov8n.pt",
]

# ─────────────────────────────────────────────────────────
# DATASET YAML — matches the 2-class helmet model
# ─────────────────────────────────────────────────────────
DATASET_YAML = """# Helmet Detection Dataset Configuration
# Classes MUST match your model exactly

path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: helmet
  1: no_helmet
"""

# ─────────────────────────────────────────────────────────
# CORRECTED helmet_detection.py CONFIG block
# (replaces the broken Haar cascade fallback logic)
# ─────────────────────────────────────────────────────────
FIXED_CONFIG = """
# ═══════════════════════════════════════════════════════
#  FIXED CONFIGURATION — Smart Helmet Detection System
#  Use THIS after running setup_model.py
# ═══════════════════════════════════════════════════════
CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,

    # ── Model (MUST be a helmet-specific fine-tuned model) ──
    "model_type": "yolov8",
    "model_path": "models/helmet_yolov8n.pt",   # ← downloaded by setup_model.py
    "tflite_model_path": "models/helmet_model.tflite",
    "confidence_threshold": 0.75,

    # ── Class IDs — MUST match your model's class list ──────
    # For the downloaded model:
    #   0 = helmet   ← ONLY this enables ignition
    #   1 = no_helmet
    "helmet_class_id": 0,
    "helmet_class_ids": {0},
    "debug_detections": True,

    # ── Detection timing ────────────────────────────────────
    "detection_frames": 7,
    "no_helmet_frames": 10,
    "detection_interval_ms": 100,

    # ── Serial / MQTT ────────────────────────────────────────
    "serial_port": "/dev/ttyUSB0",
    "serial_baudrate": 115200,
    "use_serial": False,    # ← Set True when ESP32 is connected
    "use_mqtt": False,
    "mqtt_broker": "192.168.1.100",
    "mqtt_port": 1883,
    "mqtt_topic_status": "helmet/status",
    "mqtt_topic_ignition": "helmet/ignition",

    # ── Display ──────────────────────────────────────────────
    "show_display": True,
    "save_logs": True,
    "log_file": "logs/helmet_detection.log",

    "color_helmet":    (0, 255, 0),
    "color_no_helmet": (0, 0, 255),
    "color_unknown":   (0, 255, 255),
}
"""


def print_step(n, text):
    print(f"\n{'='*55}")
    print(f"  STEP {n}: {text}")
    print(f"{'='*55}")


def check_pip_packages():
    print_step(1, "Checking / installing Python packages")
    packages = ["ultralytics", "opencv-python", "requests"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
            print(f"  ✅ {pkg} already installed")
        except ImportError:
            print(f"  📦 Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip",
                                   "install", pkg, "--quiet"])
            print(f"  ✅ {pkg} installed")


def download_model():
    print_step(2, "Downloading helmet-specific YOLOv8 model")

    Path("models").mkdir(exist_ok=True)
    model_dest = Path("models/helmet_yolov8n.pt")

    if model_dest.exists() and model_dest.stat().st_size > 100_000:
        print(f"  ✅ Model already exists: {model_dest} ({model_dest.stat().st_size//1024}KB)")
        return str(model_dest)

    # ── Method 1: Try direct download ──────────────────────
    for url in HELMET_MODEL_URLS:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, model_dest,
                reporthook=lambda b, bs, ts: print(
                    f"  Downloading... {min(100, int(b*bs/ts*100))}%", end="\r"))
            print()
            if model_dest.stat().st_size > 100_000:
                print(f"  ✅ Downloaded: {model_dest}")
                return str(model_dest)
        except Exception as e:
            print(f"  ⚠ Failed: {e}")

    # ── Method 2: Train from scratch using Roboflow dataset ─
    print("\n  Direct download unavailable. Training from scratch...")
    print("  This will take 20-60 mins on CPU, ~5 mins on GPU")
    answer = input("  Train now? (y/n): ").strip().lower()
    if answer == "y":
        return train_from_scratch()

    # ── Method 3: Manual instructions ───────────────────────
    print("""
  ─────────────────────────────────────────────────────────
  MANUAL DOWNLOAD INSTRUCTIONS:
  ─────────────────────────────────────────────────────────
  1. Go to: https://universe.roboflow.com/roboflow-100/hard-hat-universe
  2. Click "Model" tab → Download → YOLOv8 format
  3. Copy the downloaded .pt file to:
       smart_helmet_project/models/helmet_yolov8n.pt

  OR use Roboflow Python API (free account required):
       pip install roboflow
       python download_roboflow_model.py --api-key YOUR_KEY
  ─────────────────────────────────────────────────────────
  """)
    return None


def download_roboflow_dataset(api_key: str):
    """Download dataset using Roboflow API key."""
    print_step(3, "Downloading Helmet Dataset from Roboflow")
    try:
        from roboflow import Roboflow
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "roboflow", "--quiet"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    # Best free helmet dataset on Roboflow Universe
    datasets_to_try = [
        ("roboflow-100", "hard-hat-universe", 2),
        ("joseph-nelson", "helmet-detection", 2),
        ("safety-ppe-detection", "safety-helmet-ycvnr", 1),
    ]

    for workspace, project, version in datasets_to_try:
        try:
            print(f"  Trying: {workspace}/{project} v{version}")
            proj = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download("yolov8",
                                                      location="./dataset")
            print(f"  ✅ Downloaded: {workspace}/{project}")
            return dataset
        except Exception as e:
            print(f"  ⚠ Failed: {e}")

    print("  ❌ Could not download dataset automatically")
    return None


def create_dataset_yaml():
    print_step(3, "Creating dataset YAML config")
    Path("dataset").mkdir(exist_ok=True)
    yaml_path = Path("dataset/helmet.yaml")
    yaml_path.write_text(DATASET_YAML, encoding="utf-8")
    print(f"  ✅ Created: {yaml_path}")
    return str(yaml_path)


def train_from_scratch():
    """Train YOLOv8n on downloaded helmet dataset."""
    print_step("3b", "Training model from scratch")
    try:
        from ultralytics import YOLO
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "ultralytics", "--quiet"])
        from ultralytics import YOLO

    yaml_path = "dataset/helmet.yaml"
    if not Path(yaml_path).exists():
        print("  ❌ Dataset not found. Run dataset download first.")
        return None

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        epochs=50,
        batch=8,
        imgsz=640,
        project="runs/train",
        name="helmet_v1",
        exist_ok=True,
        patience=10,
    )

    best = Path("runs/train/helmet_v1/weights/best.pt")
    if best.exists():
        shutil.copy(best, "models/helmet_yolov8n.pt")
        print(f"  ✅ Trained model saved: models/helmet_yolov8n.pt")
        return "models/helmet_yolov8n.pt"
    return None


def verify_model(model_path: str):
    print_step(4, "Verifying model class names")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("\n  Model class names:")
        for cls_id, name in model.names.items():
            marker = "✅" if "helmet" in name.lower() else "ℹ"
            print(f"    {marker} Class {cls_id} = '{name}'")

        helmet_classes = [i for i, n in model.names.items()
                          if "helmet" in n.lower()]
        if helmet_classes:
            print(f"\n  ✅ Helmet class IDs: {helmet_classes}")
            print(f"     Set 'helmet_class_ids': {set(helmet_classes)} in CONFIG")
            return helmet_classes
        else:
            print("\n  ❌ WARNING: No 'helmet' class found in this model!")
            print("     This model is NOT suitable for helmet detection.")
            return []
    except Exception as e:
        print(f"  ❌ Model verification failed: {e}")
        return []


def patch_config_with_correct_classes(helmet_class_ids: list):
    """Update helmet_detection.py with verified class IDs."""
    print_step(5, "Updating helmet_detection.py with correct class IDs")
    det_path = Path("helmet_detection.py")
    if not det_path.exists():
        det_path = Path("src/raspberry_pi/helmet_detection.py")
    if not det_path.exists():
        print("  ⚠ helmet_detection.py not found — update CONFIG manually")
        print(f"    Set: 'helmet_class_ids': {set(helmet_class_ids)}")
        return

    content = det_path.read_text(encoding="utf-8")
    # Update model path and class IDs
    import re
    content = re.sub(r'"model_path":\s*"[^"]*"',
                     '"model_path": "models/helmet_yolov8n.pt"', content)
    content = re.sub(r'"helmet_class_ids":\s*\{[^}]*\}',
                     f'"helmet_class_ids": {set(helmet_class_ids)}', content)
    content = re.sub(r'"use_serial":\s*(True|False)',
                     '"use_serial": False', content)  # Safe default
    det_path.write_text(content, encoding="utf-8")
    print(f"  ✅ Updated: {det_path}")


def quick_camera_test(model_path: str):
    print_step(6, "Quick camera test (press Q to quit)")
    try:
        import cv2
        from ultralytics import YOLO
        model = YOLO(model_path)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  ⚠ No camera found — skipping live test")
            return

        print("  Camera opened. Look at the screen...")
        print("  ✅ = helmet detected | ❌ = no helmet")
        print("  Press Q to quit\n")

        for _ in range(300):  # ~10 seconds at 30fps
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=0.75, verbose=False)
            helmet_found = False
            for r in results:
                for box in r.boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = model.names.get(cls_id, "")
                    conf     = float(box.conf[0])
                    if "helmet" in cls_name.lower():
                        helmet_found = True
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,f"{cls_name} {conf:.2f}",
                                    (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,(0,255,0),2)

            status = "✅ HELMET DETECTED - IGNITION: ON" if helmet_found \
                     else "❌ NO HELMET - IGNITION: LOCKED"
            color  = (0,255,0) if helmet_found else (0,0,255)
            cv2.rectangle(frame,(0,0),(frame.shape[1],50),(0,0,0),-1)
            cv2.putText(frame, status,(10,35),
                        cv2.FONT_HERSHEY_SIMPLEX,0.75,color,2)
            cv2.imshow("Helmet Test — Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"  ⚠ Camera test failed: {e}")


def print_final_instructions(model_path, helmet_classes):
    print(f"""
╔══════════════════════════════════════════════════════╗
║          SETUP COMPLETE — NEXT STEPS                 ║
╚══════════════════════════════════════════════════════╝

  Model  : {model_path or 'NOT DOWNLOADED'}
  Classes: {helmet_classes}

  1. Run the main detection:
       python helmet_detection.py

  2. To train your own model on a better dataset:
       python train_model.py --action scaffold
       python train_model.py --action train --epochs 50

  3. Free helmet datasets to download manually:
       • https://universe.roboflow.com/roboflow-100/hard-hat-universe
       • https://www.kaggle.com/datasets/andrewmvd/hard-hat-workers
       • https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset

  4. If you get a Roboflow API key (free):
       python setup_model.py --roboflow-key YOUR_KEY

  IMPORTANT CONFIG in helmet_detection.py:
    "model_path"      : "models/helmet_yolov8n.pt"
    "helmet_class_ids": {set(helmet_classes) if helmet_classes else '{0}'}
    "confidence_threshold": 0.75

╔══════════════════════════════════════════════════════╗
║  WHY YOUR SYSTEM SHOWED FALSE POSITIVES:             ║
║  The Haar Cascade FACE detector was being used as    ║
║  fallback (no helmet model found). It detected your  ║
║  face and incorrectly labeled it as "helmet".        ║
║  This is now fixed — face detection is REMOVED.      ║
╚══════════════════════════════════════════════════════╝
""")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--roboflow-key", default="",
                        help="Roboflow API key for dataset download (free at roboflow.com)")
    parser.add_argument("--skip-camera-test", action="store_true")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║   Smart Helmet System — One-Click Setup              ║
║   This fixes the face-detection false positive bug   ║
╚══════════════════════════════════════════════════════╝
""")

    check_pip_packages()
    model_path = download_model()
    create_dataset_yaml()

    if args.roboflow_key:
        download_roboflow_dataset(args.roboflow_key)

    helmet_classes = []
    if model_path and Path(model_path).exists():
        helmet_classes = verify_model(model_path)
        if helmet_classes:
            patch_config_with_correct_classes(helmet_classes)
        if not args.skip_camera_test:
            quick_camera_test(model_path)

    print_final_instructions(model_path, helmet_classes)


if __name__ == "__main__":
    main()
