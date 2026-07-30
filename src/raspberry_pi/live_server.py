"""
Smart Helmet System — Live Web Streaming Server
Streams real-time YOLOv8 helmet detection feed via HTTP MJPEG.

Usage:
    python src/raspberry_pi/live_server.py --port 5050

Access in browser:
    http://localhost:5050 or http://<IP-ADDRESS>:5050
"""

import cv2
import time
import argparse
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveServer")

# Model path resolution
MODEL_PATH = "models/helmet_yolov8n.pt"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "runs/train/helmet_v1/weights/best.pt"

model = None
camera = None

def load_detection_model():
    global model
    if Path(MODEL_PATH).exists():
        logger.info(f"Loading YOLOv8 model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully!")
    else:
        logger.warning("No helmet model found! Run setup_model.py first.")

PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Helmet - Live Web Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #38bdf8; margin-bottom: 10px; }
        p { color: #94a3b8; font-size: 1.1rem; }
        .video-container { margin: 20px auto; max-width: 720px; border: 3px solid #334155; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); background: #000; }
        img { width: 100%; display: block; }
        .badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: bold; background-color: #0284c7; color: white; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>🏍️ Smart Helmet Ignition System</h1>
    <p>Live Real-Time YOLOv8 Detection Stream</p>
    <div class="badge">Status: Live Feed Active</div>
    <div class="video-container">
        <img src="/stream.mjpg" alt="Live Camera Stream">
    </div>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle HTTP requests in separate threads."""
    daemon_threads = True

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(PAGE_HTML)))
            self.end_headers()
            self.wfile.write(PAGE_HTML.encode('utf-8'))
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            cap = cv2.VideoCapture(0)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        # Fallback synthetic frame if camera fail/missing
                        frame = cv2.putText(
                            cv2.imread("tests/sample_helmet.jpg") if Path("tests/sample_helmet.jpg").exists()
                            else (cv2.Laplacian(cv2.UMat(480, 640, cv2.CV_8UC3), cv2.CV_8U).get()),
                            "Camera feed unavailable", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                        )
                    
                    if model is not None:
                        results = model(frame, conf=0.70, verbose=False)
                        helmet_detected = False
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                if cls_id == 0:
                                    helmet_detected = True
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Helmet {conf:.2f}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                else:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(frame, f"No Helmet {conf:.2f}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        status_text = "IGNITION: ENABLED" if helmet_detected else "IGNITION: LOCKED"
                        color = (0, 255, 0) if helmet_detected else (0, 0, 255)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
                        cv2.putText(frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    _, jpeg = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    time.sleep(0.05)
            except Exception as e:
                logger.debug(f"Client disconnected: {e}")
            finally:
                cap.release()
        else:
            self.send_error(404)

def main():
    parser = argparse.ArgumentParser(description="Live Web Streaming Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind server to")
    parser.add_argument("--port", type=int, default=5050, help="Port to run server on")
    args = parser.parse_args()

    load_detection_model()
    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, StreamHandler)
    logger.info(f"🚀 Live Web Streaming Server active at http://localhost:{args.port}")
    logger.info(f"📱 Access stream from network devices at http://<PI-IP>:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")

if __name__ == "__main__":
    main()
