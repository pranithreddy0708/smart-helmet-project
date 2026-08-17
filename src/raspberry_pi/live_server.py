"""
Smart Helmet System — Live Web Streaming & Telemetry Server
Streams real-time YOLOv8 helmet detection video feed and telemetry metrics via HTTP.

Usage:
    python src/raspberry_pi/live_server.py --port 5050

Access in browser:
    http://localhost:5050 or http://<IP-ADDRESS>:5050
"""

import cv2
import time
import json
import argparse
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveServer")

# Global Telemetry & Model State
MODEL_PATH = "models/helmet_yolov8n.pt"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "runs/train/helmet_v1/weights/best.pt"

model = None
server_start_time = time.time()
current_telemetry = {
    "ignition": "LOCKED",
    "helmet_detected": False,
    "confidence": 0.0,
    "fps": 0.0,
    "active_clients": 0,
    "uptime_seconds": 0,
    "model_loaded": False
}

def load_detection_model():
    global model, current_telemetry
    if Path(MODEL_PATH).exists():
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model from {MODEL_PATH}...")
            model = YOLO(MODEL_PATH)
            current_telemetry["model_loaded"] = True
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not initialize YOLO model: {e}")
            current_telemetry["model_loaded"] = False
    else:
        logger.warning("No helmet model weight found. Operating in simulation mode.")
        current_telemetry["model_loaded"] = False

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Helmet - Live Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #090d16;
            --bg-card: rgba(15, 23, 42, 0.75);
            --border-card: rgba(56, 189, 248, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }

        body {
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(34, 197, 94, 0.05) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 24px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 28px;
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        }

        .brand { display: flex; align-items: center; gap: 12px; }
        .brand-icon { font-size: 28px; }
        .brand-title { font-size: 1.4rem; font-weight: 700; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        .live-tag {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #4ade80;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .pulse-dot {
            width: 8px; height: 8px; border-radius: 50%; background-color: #22c55e;
            box-shadow: 0 0 10px #22c55e;
            animation: pulse 1.8s infinite;
        }
        @keyframes pulse { 0% { transform: scale(0.95); opacity: 0.7; } 50% { transform: scale(1.2); opacity: 1; } 100% { transform: scale(0.95); opacity: 0.7; } }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }

        @media (max-width: 968px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        .video-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .video-wrapper {
            width: 100%;
            aspect-ratio: 16 / 9;
            background: #000;
            border-radius: 14px;
            overflow: hidden;
            border: 2px solid rgba(255, 255, 255, 0.05);
            position: relative;
        }
        .video-wrapper img { width: 100%; height: 100%; object-fit: contain; }

        .telemetry-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }

        .stat-label { font-size: 0.85rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
        .stat-value { font-size: 1.8rem; font-weight: 700; color: var(--text-main); }

        .ignition-banner {
            padding: 18px 24px;
            border-radius: 16px;
            font-size: 1.3rem;
            font-weight: 700;
            text-align: center;
            letter-spacing: 0.05em;
            transition: all 0.3s ease;
        }
        .ignition-banner.ENABLED {
            background: rgba(34, 197, 94, 0.15);
            border: 2px solid var(--accent-green);
            color: #4ade80;
            box-shadow: 0 0 25px rgba(34, 197, 94, 0.2);
        }
        .ignition-banner.LOCKED {
            background: rgba(239, 68, 68, 0.15);
            border: 2px solid var(--accent-red);
            color: #f87171;
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
        }

        .metric-row { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
        .sub-text { font-size: 0.85rem; color: var(--text-muted); }

        footer {
            margin-top: auto;
            text-align: center;
            padding: 20px;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span class="brand-icon">🏍️</span>
            <div>
                <div class="brand-title">Smart Helmet AI Ignition</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">YOLOv8 Edge Real-Time Detection</div>
            </div>
        </div>
        <div class="live-tag">
            <div class="pulse-dot"></div>
            <span>LIVE SERVER ACTIVE</span>
        </div>
    </header>

    <div class="dashboard-grid">
        <div class="video-card">
            <div class="video-wrapper">
                <img src="/stream.mjpg" alt="Smart Helmet Camera Stream">
            </div>
        </div>

        <div class="telemetry-panel">
            <div id="ignitionCard" class="ignition-banner LOCKED">
                IGNITION: LOCKED 🔒
            </div>

            <div class="stat-card">
                <div class="stat-label">Helmet Detection Status</div>
                <div class="metric-row">
                    <div id="helmetStatus" class="stat-value" style="color: var(--accent-red);">No Helmet</div>
                    <div id="confidenceVal" class="sub-text">Confidence: 0%</div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-label">Performance Metrics</div>
                <div class="metric-row">
                    <div>
                        <div class="sub-text">Inference Speed</div>
                        <div id="fpsVal" class="stat-value" style="font-size: 1.4rem;">-- FPS</div>
                    </div>
                    <div>
                        <div class="sub-text">Model Engine</div>
                        <div id="modelEngine" class="stat-value" style="font-size: 1.2rem; color: var(--accent-cyan);">YOLOv8n</div>
                    </div>
                </div>
            </div>

            <div class="stat-card">
                <div class="stat-label">System Telemetry</div>
                <div class="metric-row">
                    <div>
                        <div class="sub-text">Server Uptime</div>
                        <div id="uptimeVal" class="sub-text" style="color: var(--text-main); font-size: 1.1rem; font-weight: 600;">0s</div>
                    </div>
                    <div>
                        <div class="sub-text">REST API</div>
                        <a href="/api/status" target="_blank" style="color: var(--accent-cyan); font-size: 0.85rem; text-decoration: none;">GET /api/status</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer>
        AI-Powered Smart Helmet Bike Ignition System &copy; 2026 | B.E. Engineering Project
    </footer>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/status');
                if (res.ok) {
                    const data = await res.json();
                    const ignCard = document.getElementById('ignitionCard');
                    const helmetStatus = document.getElementById('helmetStatus');
                    const confVal = document.getElementById('confidenceVal');
                    const fpsVal = document.getElementById('fpsVal');
                    const uptimeVal = document.getElementById('uptimeVal');

                    if (data.ignition === 'ENABLED') {
                        ignCard.className = 'ignition-banner ENABLED';
                        ignCard.innerHTML = 'IGNITION: ENABLED ⚡';
                        helmetStatus.innerText = 'Helmet Detected ✅';
                        helmetStatus.style.color = 'var(--accent-green)';
                    } else {
                        ignCard.className = 'ignition-banner LOCKED';
                        ignCard.innerHTML = 'IGNITION: LOCKED 🔒';
                        helmetStatus.innerText = 'No Helmet ❌';
                        helmetStatus.style.color = 'var(--accent-red)';
                    }

                    confVal.innerText = `Confidence: ${Math.round(data.confidence * 100)}%`;
                    fpsVal.innerText = `${data.fps.toFixed(1)} FPS`;
                    uptimeVal.innerText = `${data.uptime_seconds}s`;
                }
            } catch (err) {
                console.error("Telemetry fetch error:", err);
            }
        }
        setInterval(fetchTelemetry, 1000);
        fetchTelemetry();
    </script>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle HTTP requests concurrently across client connections."""
    daemon_threads = True

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_telemetry
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(PAGE_HTML.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(PAGE_HTML.encode('utf-8'))
            
        elif self.path == '/api/status':
            current_telemetry["uptime_seconds"] = int(time.time() - server_start_time)
            json_bytes = json.dumps(current_telemetry).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(json_bytes)))
            self.end_headers()
            self.wfile.write(json_bytes)
            
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            cap = cv2.VideoCapture(0)
            t_prev = time.time()
            sim_angle = 0.0

            try:
                while True:
                    t0 = time.time()
                    ret, frame = cap.read()
                    
                    if not ret:
                        # Synthetic animation stream for headless / non-camera testing
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        frame[:] = (20, 15, 10)
                        
                        sim_angle += 0.08
                        helmet_detected_sim = (np.sin(sim_angle) > 0)
                        
                        cv2.putText(frame, "SIMULATED LIVE FEED", (20, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (56, 189, 248), 2)
                        
                        if helmet_detected_sim:
                            cv2.rectangle(frame, (200, 120), (440, 360), (0, 255, 0), 3)
                            cv2.putText(frame, "HELMET DETECTED (0.94)", (200, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            current_telemetry["helmet_detected"] = True
                            current_telemetry["ignition"] = "ENABLED"
                            current_telemetry["confidence"] = 0.94
                        else:
                            cv2.rectangle(frame, (200, 120), (440, 360), (0, 0, 255), 3)
                            cv2.putText(frame, "NO HELMET DETECTED", (200, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            current_telemetry["helmet_detected"] = False
                            current_telemetry["ignition"] = "LOCKED"
                            current_telemetry["confidence"] = 0.0
                    else:
                        if model is not None:
                            results = model(frame, conf=0.70, verbose=False)
                            helmet_found = False
                            highest_conf = 0.0
                            for r in results:
                                for box in r.boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    if cls_id == 0:
                                        helmet_found = True
                                        highest_conf = max(highest_conf, conf)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                        cv2.putText(frame, f"Helmet {conf:.2f}", (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    else:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                        cv2.putText(frame, f"No Helmet {conf:.2f}", (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            current_telemetry["helmet_detected"] = helmet_found
                            current_telemetry["ignition"] = "ENABLED" if helmet_found else "LOCKED"
                            current_telemetry["confidence"] = float(highest_conf)

                    # Compute FPS
                    t1 = time.time()
                    dt = t1 - t_prev
                    t_prev = t1
                    fps = 1.0 / dt if dt > 0 else 0.0
                    current_telemetry["fps"] = round(fps, 1)

                    # Draw Ignition Banner on Stream Frame
                    status_text = f"IGNITION: {current_telemetry['ignition']}"
                    color = (0, 255, 0) if current_telemetry['ignition'] == "ENABLED" else (0, 0, 255)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
                    cv2.putText(frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    _, jpeg = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    time.sleep(0.04)

            except Exception as e:
                logger.debug(f"Client disconnected: {e}")
            finally:
                cap.release()
        else:
            self.send_error(404)

def main():
    parser = argparse.ArgumentParser(description="Smart Helmet Live Server & Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5050, help="Port to run web server on (default: 5050)")
    args = parser.parse_args()

    load_detection_model()
    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, StreamHandler)
    logger.info(f"🚀 Live Web Server & Telemetry Dashboard active at http://localhost:{args.port}")
    logger.info(f"📱 Access stream across your network at http://<PI-IP>:{args.port}")
    logger.info(f"📊 REST API Status Endpoint: http://localhost:{args.port}/api/status")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")

if __name__ == "__main__":
    main()

