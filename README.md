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
│   │   ├── setup_model.py          ← One-click model downloader & setup
│   │   ├── live_server.py          ← Live Web Streaming Server (HTTP MJPEG)
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

### 1. Install Dependencies (Raspberry Pi / Local PC)
```bash
pip install -r requirements.txt
```

### 2. Run One-Click Model Setup
```bash
python src/raspberry_pi/setup_model.py --skip-camera-test
```

### 3. Run Live Web Server (HTTP Video Stream)
Start the real-time MJPEG live web streaming server to view detection in your browser or remote dashboard:
```bash
python src/raspberry_pi/live_server.py --port 5000
```
- Open browser locally: **`http://localhost:5000`**
- Open from mobile device / network: **`http://<RASPBERRY_PI_IP>:5000`**

### 4. Run Desktop Detection System
```bash
python src/raspberry_pi/helmet_detection.py
```

### 5. Train Custom Model (Optional)
```bash
# Download & train on helmet dataset
python src/raspberry_pi/train_model.py --action train --epochs 50 --device cpu
```

### 6. Export to TFLite
```bash
python src/raspberry_pi/train_model.py --action export --export-format tflite
```

### 7. Flash ESP32 Firmware
- Open `src/esp32/helmet_ignition_esp32.ino` in Arduino IDE
- Edit WiFi / Serial configuration
- Select Board: **ESP32 Dev Module**
- Upload to device

### 8. Run Automated Test Suite
```bash
python tests/test_system.py --test all
```

---

## 🌐 Live Web Server Features

The included Live Web Server (`live_server.py`) enables real-time monitoring over HTTP without needing an external GUI display:

- **MJPEG Live Stream**: Ultra-low latency camera feed with YOLOv8 helmet detection boxes.
- **Ignition Status Banner**: Visual overlay indicating `IGNITION: ENABLED` (Green) or `IGNITION: LOCKED` (Red).
- **Zero-Dependency Web Server**: Built using Python's standard library `http.server`, works on any platform.
- **Cross-Device Access**: Accessible from any laptop, tablet, or smartphone connected to the local WiFi network.

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
| Detection FPS (YOLOv8n, Pi4 / PC) | 15–17 FPS |
| Detection Accuracy (mAP50) | ~88–93% |
| Ignition Response Time | < 1 second |
| Power Consumption | ~5W |
| Offline Operation | ✅ Yes |
| Live Web Streaming | ✅ http://localhost:5000 |

---

## 📚 Key Technologies

- **YOLOv8** — Real-time object detection
- **OpenCV** — Frame capture and overlay processing
- **HTTP MJPEG Live Server** — Web video streaming engine
- **TensorFlow Lite** — Edge deployment optimization
- **MQTT (Mosquitto)** — IoT messaging
- **PySerial** — Raspberry Pi ↔ ESP32 communication
- **React Native (Expo)** — Cross-platform mobile app
- **Arduino/ESP32** — Hardware relay control

---

## 📄 Documents

- [`report/PROJECT_REPORT.md`](report/PROJECT_REPORT.md) — Full report with viva Q&A
- [`hardware/circuit_diagram.txt`](hardware/circuit_diagram.txt) — Wiring diagrams
- [`QUICKSTART.md`](QUICKSTART.md) — Step-by-step setup guide

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
