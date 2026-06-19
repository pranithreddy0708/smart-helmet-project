# QUICKSTART GUIDE — Fix False Positive & Get System Working
## Smart Helmet Bike Ignition System

---

## WHY IT WAS SHOWING FALSE POSITIVES

```
PROBLEM:  face(demo) 0.90  →  "HELMET: DETECTED"
CAUSE:    No helmet model found → system fell back to Haar Cascade FACE detector
          → It detected your FACE and labelled it "helmet"
FIX:      Download real helmet-specific YOLOv8 model (steps below)
```

---

## STEP 1 — Install Dependencies

```bash
pip install ultralytics opencv-python pyserial paho-mqtt roboflow
```

---

## STEP 2 — Get the Helmet Model (choose ONE method)

### Method A: Roboflow Download (Recommended — Free)

```bash
# 1. Go to https://app.roboflow.com → Sign up free → Settings → API Key

# 2. Download dataset
python download_roboflow_model.py --api-key YOUR_API_KEY --dataset 1

# 3. Train model (takes ~30 min on CPU, ~5 min on GPU)
python train_model.py --action train --epochs 50 --device cpu
#                                                         ^^^ or "0" for GPU

# 4. Model will be saved at:
#    runs/train/helmet_v1/weights/best.pt
#    → Copy it:
cp runs/train/helmet_v1/weights/best.pt models/helmet_yolov8n.pt
```

### Method B: Manual Download (Kaggle)

```
1. Go to: https://www.kaggle.com/datasets/andrewmvd/hard-hat-workers
2. Download and extract to: kaggle_raw/
3. Convert and split:
   python datasets/download_dataset.py --action kaggle --kaggle-dir kaggle_raw
4. Train:
   python train_model.py --action train --epochs 50
5. Copy best model:
   cp runs/train/helmet_v1/weights/best.pt models/helmet_yolov8n.pt
```

### Method C: Google Colab (Free GPU — Fastest)

```
1. Open: https://colab.research.google.com
2. Paste and run:

!pip install ultralytics roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
proj = rf.workspace("joseph-nelson").project("helmet-detection")
dataset = proj.version(2).download("yolov8")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/helmet-detection-2/data.yaml",
            epochs=50, imgsz=640, batch=16, device=0)

# Then download: runs/train/exp/weights/best.pt
# Copy to your PC as: models/helmet_yolov8n.pt

3. Runtime → Change runtime type → T4 GPU (free)
```

---

## STEP 3 — Verify Model is Correct

```bash
python setup_model.py
```

Expected output:
```
  ✅ Class 0 = 'helmet'
  ℹ  Class 1 = 'no_helmet'

  ✅ Helmet class IDs: {0}
```

If you see `Class 0 = 'person'` → WRONG MODEL (that's base COCO)

---

## STEP 4 — Run Detection

```bash
python helmet_detection.py
```

Expected behavior:
- **Wear helmet** → green box around helmet, "IGNITION: ENABLED"
- **No helmet** → red banner, "IGNITION: LOCKED"
- **No face detection** → face will NOT trigger ignition anymore

---

## CONFIGURATION REFERENCE

Edit `CONFIG` in `helmet_detection.py`:

```python
"model_path": "models/helmet_yolov8n.pt",  # ← your trained model
"confidence_threshold": 0.75,               # raise if false positives
"helmet_class_ids": {0},                    # class 0 = helmet
"detection_frames": 7,                      # frames before enabling
"use_serial": False,                        # True when ESP32 connected
```

---

## TROUBLESHOOTING

| Problem | Cause | Fix |
|---|---|---|
| `face(demo)` label | No model found → Haar cascade used | Run `setup_model.py` |
| `Model not ready` error | Model file missing | Download model (Step 2) |
| Wrong class `'person'` | Using base COCO model | Train on helmet dataset |
| High false positives | Confidence too low | Set `confidence_threshold: 0.85` |
| Very slow FPS | Large model on CPU | Use `yolov8n.pt` (nano), not `yolov8m` |
| No detections at all | Confidence too high | Lower to `0.65` temporarily |

---

## DATASET LINKS

| Dataset | Images | Link |
|---|---|---|
| Roboflow Helmet Detection | 4,546 | https://universe.roboflow.com/joseph-nelson/helmet-detection |
| Hard Hat Workers (Kaggle) | 7,035 | https://kaggle.com/datasets/andrewmvd/hard-hat-workers |
| Safety Helmet Wearing | 7,500 | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset |
| Roboflow Hard Hat Universe | 5,000 | https://universe.roboflow.com/roboflow-100/hard-hat-universe |
