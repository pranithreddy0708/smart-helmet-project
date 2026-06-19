# 🏍️ AI-Powered Smart Helmet Bike Ignition System

> **Final Year Engineering Project** | AI · IoT · Computer Vision · Embedded Systems

A motorcycle ignition control system that uses **YOLOv8 object detection** to verify helmet usage in real time. The bike **cannot start** unless the rider is wearing a helmet.

---

## 📁 Project Structure

```
smart_helmet_project/
├── src/
│   ├── raspberry_pi/
│   │   ├── helmet_detection.py     ← Main detection + ignition control
│   │   └── train_model.py          ← Model training (YOLOv8)
│   ├── esp32/
│   │   └── helmet_ignition_esp32.ino  ← ESP32 Arduino firmware
│   └── mobile_app/
│       └── App.js                  ← React Native dashboard
├── datasets/
│   └── download_dataset.py         ← Dataset download helper
├── tests/
│   └── test_system.py              ← Automated test suite
├── hardware/
│   └── circuit_diagram.txt         ← Full wiring diagrams
├── report/
│   └── PROJECT_REPORT.md           ← Complete project report
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install Dependencies (Raspberry Pi)
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Scaffold empty structure
python src/raspberry_pi/train_model.py --action scaffold

# Download from Roboflow (free account needed)
python datasets/download_dataset.py \
  --action roboflow \
  --api-key YOUR_KEY \
  --project safety-helmet-detection-nnfnf
```

### 3. Train Model
```bash
python src/raspberry_pi/train_model.py \
  --action train \
  --epochs 50 \
  --device 0    # GPU (use "cpu" if no GPU)
```

### 4. Export to TFLite
```bash
python src/raspberry_pi/train_model.py \
  --action export \
  --export-format tflite
```

### 5. Run Detection System
```bash
python src/raspberry_pi/helmet_detection.py
```

### 6. Flash ESP32
- Open `src/esp32/helmet_ignition_esp32.ino` in Arduino IDE
- Edit WiFi credentials
- Select Board: **ESP32 Dev Module**
- Upload

### 7. Run Tests
```bash
python tests/test_system.py --test all
```

---

## 🔧 Hardware Required

| Component | Cost (INR) |
|---|---|
| Raspberry Pi 4 (4GB) | ₹5,500 |
| ESP32 DevKit V1 | ₹350 |
| USB Webcam (720p) | ₹800 |
| 5V Relay Module | ₹80 |
| LCD 16x2 I2C | ₹120 |
| Power Bank 20000mAh | ₹1,200 |
| Misc (wires, LEDs, etc.) | ₹465 |
| **TOTAL** | **~₹8,515** |

---

## 📊 System Performance

| Metric | Value |
|---|---|
| Detection FPS (YOLOv8n, Pi4) | 10–15 FPS |
| Detection Accuracy (mAP50) | ~88–93% |
| Ignition Response Time | < 1 second |
| Power Consumption | ~5W |
| Offline Operation | ✅ Yes |

---

## 📚 Key Technologies

- **YOLOv8** — Real-time object detection
- **OpenCV** — Camera capture and frame processing
- **TensorFlow Lite** — Edge deployment optimization
- **MQTT (Mosquitto)** — IoT messaging
- **PySerial** — Raspberry Pi ↔ ESP32 communication
- **React Native (Expo)** — Cross-platform mobile app
- **Arduino/ESP32** — Hardware relay control

---

## 📄 Documents

- [`report/PROJECT_REPORT.md`](report/PROJECT_REPORT.md) — Full report with viva Q&A
- [`hardware/circuit_diagram.txt`](hardware/circuit_diagram.txt) — Wiring diagrams

---

## ⚠️ Safety Warning

When connecting to a real motorcycle:
- Always disconnect the battery before wiring
- Use a properly rated relay (10A+) for ignition circuits
- Add a blade fuse (5A) in line with relay power
- Test on a stationary bike before road use
- Never bypass existing safety systems

---

*Submitted for B.E. Final Year Project | Electronics/Computer Science Engineering*
