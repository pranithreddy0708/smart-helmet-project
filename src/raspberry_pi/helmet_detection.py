"""
Smart Helmet Bike Ignition System — FIXED v3
Main Detection Script

WHAT WAS WRONG (and fixed):
  ❌ Old code: fell back to Haar Cascade FACE detector when no model found
              → detected your FACE and called it "helmet" → false ignition
  ✅ Fixed:   requires a real helmet-specific YOLOv8 model
              → if no model found, system LOCKS and shows clear error
              → NEVER uses face detection as a proxy

HOW TO GET THE MODEL:
  Run:  python setup_model.py
  This downloads a YOLOv8 model trained specifically on helmet images.

CLASSES (after setup):
  Class 0 = helmet    ← enables ignition
  Class 1 = no_helmet ← keeps ignition locked

Author: Final Year Project Team
"""

import cv2
import serial
import time
import json
import logging
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,

    # ── IMPORTANT: model_path must point to a HELMET-SPECIFIC model ─────────
    # Run  python setup_model.py  to download the correct model automatically
    # Do NOT use yolov8n.pt (that's the base COCO model — has NO helmet class)
    "model_type": "yolov8",
    "model_path": "models/helmet_yolov8n.pt",
    "tflite_model_path": "models/helmet_model.tflite",
    "confidence_threshold": 0.75,

    # ── Class IDs — set by setup_model.py after verifying your model ────────
    # Class 0 = helmet (enables ignition)
    # Class 1 = no_helmet (keeps ignition locked)
    "helmet_class_ids": {0, 1},
    "debug_detections": True,    # Set False in production

    # ── Detection logic ──────────────────────────────────────────────────────
    "detection_frames": 7,       # Consecutive frames needed before enabling
    "no_helmet_frames": 10,      # Consecutive miss frames before locking
    "detection_interval_ms": 100,

    # ── Serial (ESP32) ───────────────────────────────────────────────────────
    "serial_port": "/dev/ttyUSB0",
    "serial_baudrate": 115200,
    "use_serial": False,         # Set True when ESP32 is physically connected

    # ── MQTT ─────────────────────────────────────────────────────────────────
    "use_mqtt": False,
    "mqtt_broker": "192.168.1.100",
    "mqtt_port": 1883,
    "mqtt_topic_status": "helmet/status",
    "mqtt_topic_ignition": "helmet/ignition",

    # ── Display / Logging ────────────────────────────────────────────────────
    "show_display": True,
    "save_logs": True,
    "log_file": "logs/helmet_detection.log",

    "color_helmet":    (0, 255, 0),
    "color_no_helmet": (0, 0, 255),
    "color_warning":   (0, 165, 255),
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG if CONFIG.get("debug_detections") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELMET DETECTOR
# ─────────────────────────────────────────────
class HelmetDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.model_names = {}
        self.interpreter = None
        self.serial_conn = None
        self.mqtt_client = None
        self.model_ready = False

        self.helmet_detected = False
        self.ignition_enabled = False
        self.consecutive_helmet = 0
        self.consecutive_no_helmet = 0
        self.detection_count = 0
        self.start_time = datetime.now()
        self._lock = threading.Lock()

        self._load_model()
        self._setup_communication()

    # ── Model Loading ──────────────────────────────────────────────────────
    def _load_model(self):
        model_type = self.config["model_type"]
        model_path = self.config["model_path"]
        logger.info(f"Loading model: {model_path}")

        # ── HARD GUARD: refuse to start without a real model ────────────────
        if not Path(model_path).exists():
            logger.error("=" * 60)
            logger.error("  MODEL FILE NOT FOUND!")
            logger.error(f"  Expected: {model_path}")
            logger.error("")
            logger.error("  FIX: Run  python setup_model.py  first.")
            logger.error("  This downloads a helmet-specific YOLOv8 model.")
            logger.error("")
            logger.error("  DO NOT use the base yolov8n.pt — it has NO helmet class.")
            logger.error("  The Haar Cascade fallback has been REMOVED intentionally.")
            logger.error("=" * 60)
            self.model_ready = False
            return

        if model_type == "yolov8":
            self._load_yolov8(model_path)
        elif model_type == "tflite":
            self._load_tflite()
        else:
            logger.error(f"Unknown model_type: {model_type}")
            self.model_ready = False

    def _load_yolov8(self, model_path: str):
        if not YOLO_AVAILABLE:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            self.model_ready = False
            return
        self.model = YOLO(model_path)
        self.model_names = self.model.names  # {0: 'helmet', 1: 'no_helmet', ...}

        logger.info("Model loaded. Class names:")
        for cls_id, name in self.model_names.items():
            is_helmet = "helmet" in name.lower() and "no" not in name.lower()
            tag = "✅ HELMET CLASS" if is_helmet else "   "
            logger.info(f"  {tag}  Class {cls_id} = '{name}'")

        # Verify model has at least one helmet class
        helmet_classes = {
            i for i, n in self.model_names.items()
            if "helmet" in n.lower() and "no" not in n.lower()
        }
        if not helmet_classes:
            logger.error("=" * 60)
            logger.error("  WRONG MODEL — NO 'helmet' CLASS FOUND!")
            logger.error(f"  Classes in this model: {self.model_names}")
            logger.error("")
            logger.error("  This is probably the base COCO model (yolov8n.pt)")
            logger.error("  which has 'person', 'car', etc. — NOT 'helmet'")
            logger.error("")
            logger.error("  FIX: python setup_model.py")
            logger.error("=" * 60)
            self.model_ready = False
            return

        # Auto-update class IDs from model (more reliable than manual config)
        self.config["helmet_class_ids"] = helmet_classes
        logger.info(f"Helmet class IDs auto-detected: {helmet_classes}")
        self.model_ready = True

    def _load_tflite(self):
        path = self.config["tflite_model_path"]
        if not Path(path).exists():
            logger.error(f"TFLite model not found: {path}")
            self.model_ready = False
            return
        if not TFLITE_AVAILABLE:
            logger.error("tflite not installed. Run: pip install tflite-runtime")
            self.model_ready = False
            return
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=path)
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.model_ready = True
        logger.info("TFLite model loaded")

    # ── Communication ──────────────────────────────────────────────────────
    def _setup_communication(self):
        if self.config["use_serial"]:
            self._setup_serial()
        if self.config["use_mqtt"] and MQTT_AVAILABLE:
            self._setup_mqtt()

    def _setup_serial(self):
        try:
            self.serial_conn = serial.Serial(
                self.config["serial_port"], self.config["serial_baudrate"],
                timeout=1)
            time.sleep(2)
            logger.info(f"Serial connected: {self.config['serial_port']}")
        except Exception as e:
            logger.warning(f"Serial not connected: {e}")
            self.serial_conn = None

    def _setup_mqtt(self):
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect(self.config["mqtt_broker"],
                                     self.config["mqtt_port"])
            self.mqtt_client.loop_start()
            logger.info(f"MQTT connected: {self.config['mqtt_broker']}")
        except Exception as e:
            logger.warning(f"MQTT not connected: {e}")
            self.mqtt_client = None

    # ── Detection ──────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> dict:
        """Run helmet detection. Returns dict with detection results."""
        if not self.model_ready:
            return {
                "helmet_detected": False,
                "detections": [],
                "error": "model_not_ready",
                "timestamp": datetime.now().isoformat(),
                "frame_number": self.detection_count
            }

        model_type = self.config["model_type"]
        conf_thresh = self.config["confidence_threshold"]
        detections = []

        if model_type == "yolov8":
            detections = self._detect_yolov8(frame, conf_thresh)
        elif model_type == "tflite":
            detections = self._detect_tflite(frame, conf_thresh)

        helmet_present = len(detections) > 0
        self.detection_count += 1

        return {
            "helmet_detected": helmet_present,
            "detections": detections,
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.detection_count
        }

    def _detect_yolov8(self, frame, conf_thresh):
        results = self.model(frame, conf=conf_thresh, verbose=False)
        detections = []
        fh, fw = frame.shape[:2]
        allowed_ids = self.config["helmet_class_ids"]

        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.model_names.get(cls_id, str(cls_id))
                conf     = float(box.conf[0])

                if self.config.get("debug_detections"):
                    logger.debug(f"  Raw → cls={cls_id} ('{cls_name}')  conf={conf:.3f}")

                # ── Guard 1: class ID must be a helmet class ──────────────
                if cls_id not in allowed_ids:
                    logger.debug(f"  Skipped: class '{cls_name}' is not a helmet class")
                    continue

                # ── Guard 2: class name must contain 'helmet' ─────────────
                if "helmet" not in cls_name.lower():
                    logger.debug(f"  Skipped: name '{cls_name}' doesn't contain 'helmet'")
                    continue

                # ── Guard 3: must NOT be 'no_helmet' class ────────────────
                if "no" in cls_name.lower() or "without" in cls_name.lower():
                    logger.debug(f"  Skipped: '{cls_name}' is a NO-HELMET class")
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ── Guard 4: bbox must be in upper 75% of frame ───────────
                center_y = (y1 + y2) / 2
                if center_y > fh * 0.75:
                    logger.debug(f"  Skipped: center_y={center_y:.0f} > {fh*0.75:.0f} (too low)")
                    continue

                # ── Guard 5: minimum box size ─────────────────────────────
                if (x2-x1) < 40 or (y2-y1) < 40:
                    logger.debug(f"  Skipped: box too small ({x2-x1}x{y2-y1})")
                    continue

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class": cls_name,
                    "class_id": cls_id
                })
                logger.debug(f"  ✅ Accepted: '{cls_name}' conf={conf:.3f}")

        return detections

    def _detect_tflite(self, frame, conf_thresh):
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        resized = cv2.resize(frame, (w, h))
        inp = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        detections = []
        allowed_ids = self.config["helmet_class_ids"]
        fh, fw = frame.shape[:2]
        if output.ndim == 3:
            for det in output[0]:
                conf = float(det[4])
                cls  = int(det[5])
                if conf >= conf_thresh and cls in allowed_ids:
                    cx, cy, bw, bh = det[0]*fw, det[1]*fh, det[2]*fw, det[3]*fh
                    x1, y1 = int(cx-bw/2), int(cy-bh/2)
                    x2, y2 = int(cx+bw/2), int(cy+bh/2)
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class": "helmet",
                        "class_id": cls
                    })
        return detections

    # ── State Machine ──────────────────────────────────────────────────────
    def update_state(self, helmet_detected: bool):
        with self._lock:
            if helmet_detected:
                self.consecutive_helmet += 1
                self.consecutive_no_helmet = 0
            else:
                self.consecutive_no_helmet += 1
                self.consecutive_helmet = 0

            if (self.consecutive_helmet >= self.config["detection_frames"]
                    and not self.ignition_enabled):
                self.ignition_enabled = True
                self.helmet_detected = True
                logger.info("🟢 HELMET CONFIRMED — Ignition ENABLED")
                self._send_ignition(True)

            elif (self.consecutive_no_helmet >= self.config["no_helmet_frames"]
                  and self.ignition_enabled):
                self.ignition_enabled = False
                self.helmet_detected = False
                logger.info("🔴 HELMET REMOVED — Ignition LOCKED")
                self._send_ignition(False)

    def _send_ignition(self, enable: bool):
        cmd = "IGNITION:ON\n" if enable else "IGNITION:OFF\n"
        payload = json.dumps({"ignition": enable,
                               "timestamp": datetime.now().isoformat()})
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(cmd.encode())
            except Exception as e:
                logger.error(f"Serial error: {e}")
        if self.mqtt_client:
            try:
                self.mqtt_client.publish(
                    self.config["mqtt_topic_ignition"], payload)
            except Exception as e:
                logger.error(f"MQTT error: {e}")

    # ── Frame Annotation ───────────────────────────────────────────────────
    def annotate_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        ann = frame.copy()
        helmet_detected = result["helmet_detected"]

        for det in result.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(ann, (x1, y1), (x2, y2),
                          self.config["color_helmet"], 2)
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(ann, label, (x1, max(y1-8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        self.config["color_helmet"], 2)

        # ── Status banner ─────────────────────────────────────────────────
        if not self.model_ready:
            banner_text  = "ERROR: Model not loaded — run setup_model.py"
            ignition_txt = "IGNITION: LOCKED (NO MODEL)"
            banner_col   = self.config["color_warning"]
        elif helmet_detected:
            banner_text  = "HELMET: DETECTED"
            ignition_txt = "IGNITION: ENABLED" if self.ignition_enabled \
                           else f"IGNITION: PENDING ({self.consecutive_helmet}/{self.config['detection_frames']})"
            banner_col   = self.config["color_helmet"]
        else:
            banner_text  = "HELMET: NOT DETECTED"
            ignition_txt = "IGNITION: LOCKED"
            banner_col   = self.config["color_no_helmet"]

        cv2.rectangle(ann, (0, 0), (frame.shape[1], 65), (0, 0, 0), -1)
        cv2.putText(ann, banner_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, banner_col, 2)
        cv2.putText(ann, ignition_txt, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_col, 2)

        # FPS
        elapsed = max((datetime.now() - self.start_time).total_seconds(), 0.001)
        fps = self.detection_count / elapsed
        cv2.putText(ann, f"FPS: {fps:.1f} | Frame: {self.detection_count}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return ann

    def cleanup(self):
        if self.serial_conn and self.serial_conn.is_open:
            self._send_ignition(False)
            self.serial_conn.close()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("  Smart Helmet Ignition System v3 — Starting")
    logger.info("=" * 60)

    detector = HelmetDetector(CONFIG)

    if not detector.model_ready:
        logger.error("System cannot start — model not ready.")
        logger.error("Run:  python setup_model.py")
        return

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    cap.set(cv2.CAP_PROP_FPS,          CONFIG["fps"])

    if not cap.isOpened():
        logger.error("Cannot open camera. Check camera_index.")
        return

    logger.info("Camera ready. Starting detection loop...")
    logger.info("Press Q to quit | S to save screenshot")

    last_detect_time = 0
    interval = CONFIG["detection_interval_ms"] / 1000.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_detect_time >= interval:
                result = detector.detect(frame)
                detector.update_state(result["helmet_detected"])
                last_detect_time = now

                if CONFIG["show_display"]:
                    ann = detector.annotate_frame(frame, result)
                    cv2.imshow("Smart Helmet Detection", ann)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"logs/screenshot_{ts}.jpg"
                cv2.imwrite(fname, frame)
                logger.info(f"Screenshot saved: {fname}")

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
