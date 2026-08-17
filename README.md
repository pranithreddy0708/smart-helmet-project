# 🏍️ AI-Powered Smart Helmet Bike Ignition System

> **Final Year Engineering Project** | AI · IoT · Computer Vision · Embedded Systems · Cloud & Edge Deployment

A motorcycle ignition control system that uses **YOLOv8 object detection** to verify helmet usage in real time. The bike **cannot start** unless the rider is wearing a helmet. Features a real-time web telemetry dashboard, Docker containerization, REST API, public cloud deployment, and GitHub CI/CD automation.

---

## 🌐 Live Deployed Server & Dashboard

- **Public Live Dashboard**: **[https://chilly-sites-knock.loca.lt](https://chilly-sites-knock.loca.lt)**
- **Public REST Telemetry API**: **[https://chilly-sites-knock.loca.lt/api/status](https://chilly-sites-knock.loca.lt/api/status)**
- **Local Dashboard**: `http://localhost:5050`

---

## 📁 Project Structure

```
smart_helmet_project/
├── .github/
│   └── workflows/
│       └── deploy.yml              ← GitHub Actions CI/CD Pipeline
├── Dockerfile                       ← Docker image specification
├── docker-compose.yml               ← Docker Compose deployment config
├── deploy.py                        ← Unified deployment runner
├── src/
│   ├── raspberry_pi/
│   │   ├── helmet_detection.py     ← Main detection + ignition control
│   │   ├── setup_model.py          ← One-click model downloader & setup
│   │   ├── live_server.py          ← Live Web Telemetry Dashboard (HTTP MJPEG)
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

## ⚡ Quick Start & Deployment Options

### Option 1: Direct Python Deployment
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Launch Live Telemetry Server & Web Dashboard
python deploy.py --mode local --port 5050
```
- Access Public Deployed Web Dashboard: **`https://chilly-sites-knock.loca.lt`**
- Access Local Web Dashboard: **`http://localhost:5050`**
- REST Telemetry API: **`http://localhost:5050/api/status`**

### Option 2: Docker Container Deployment
Deploy seamlessly on any cloud VM (AWS, Render, Azure, GCP) or local Linux/Windows host using Docker:
```bash
# Launch via Docker Compose
python deploy.py --mode docker
# OR directly:
docker compose up --build -d
```

### Option 3: Desktop Detection System
```bash
python src/raspberry_pi/helmet_detection.py
```

### Option 4: Run Automated Test Suite
```bash
python deploy.py --mode test
```

---

## 🌐 Live Web Server & Telemetry Dashboard Features

The included Live Web Telemetry Dashboard (`live_server.py`) enables real-time monitoring over HTTP without needing an external display monitor:

- **Glassmorphic UI**: Ultra-sleek dark mode interface with live status cards and pulse indicators.
- **MJPEG Live Stream**: Ultra-low latency camera feed with YOLOv8 helmet detection overlay boxes.
- **Ignition Status Banner**: Visual state indicator displaying `IGNITION: ENABLED` (Green) or `IGNITION: LOCKED` (Red).
- **REST Telemetry API**: Endpoint `/api/status` returns JSON telemetry (Ignition state, helmet detection, confidence score, FPS, uptime).
- **Simulation Mode**: Built-in fallback stream allowing headless testing on servers without attached physical USB webcams.
- **Cross-Device Access**: Accessible from any laptop, tablet, or mobile smartphone on the network.

---

## 📊 Telemetry API Specification

### GET `/api/status`
Returns real-time status of the helmet detection engine and ignition interlock state.

**Sample Response (`application/json`):**
```json
{
  "ignition": "ENABLED",
  "helmet_detected": true,
  "confidence": 0.94,
  "fps": 30.0,
  "active_clients": 1,
  "uptime_seconds": 182,
  "model_loaded": true
}
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
| Video Stream FPS | 30 FPS (Zero Motion Lag) |
| Detection Accuracy (mAP50) | ~88–93% |
| Ignition Response Time | < 1 second |
| Power Consumption | ~5W |
| Deployed Public URL | ✅ https://chilly-sites-knock.loca.lt |
| Telemetry API | ✅ https://chilly-sites-knock.loca.lt/api/status |

---

## 🚀 GitHub Actions CI/CD Pipeline

The project includes an automated GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically:
1. Runs python test suites on every push to `main` branch.
2. Validates Live Web Telemetry Server startup and REST API response.
3. Builds Docker container images automatically.

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
