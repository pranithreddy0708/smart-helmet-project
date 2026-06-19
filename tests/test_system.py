"""
Smart Helmet Bike Ignition System
Testing & Evaluation Suite
Author: Final Year Project Team

Tests:
  1. Model accuracy (precision, recall, mAP)
  2. Inference speed (FPS on hardware)
  3. Serial communication
  4. End-to-end system test
  5. Stress test (long-running)
"""

import cv2
import time
import json
import os
import unittest
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. MODEL ACCURACY TEST
# ─────────────────────────────────────────────
class TestModelAccuracy(unittest.TestCase):
    """Tests for model detection accuracy on a labeled test set."""

    TEST_DIR    = "dataset/images/test"
    LABELS_DIR  = "dataset/labels/test"
    MODEL_PATH  = "runs/train/helmet_v1/weights/best.pt"
    IOU_THRESH  = 0.5
    CONF_THRESH = 0.70

    @classmethod
    def setUpClass(cls):
        try:
            from ultralytics import YOLO
            cls.model = YOLO(cls.MODEL_PATH)
            cls.model_available = True
        except Exception as e:
            logger.warning(f"Model not available: {e}")
            cls.model_available = False

    def test_model_loads(self):
        if not self.model_available:
            self.skipTest("Model not found")
        self.assertIsNotNone(self.model)

    def test_precision_recall(self):
        if not self.model_available:
            self.skipTest("Model not found")
        if not Path(self.TEST_DIR).exists():
            self.skipTest("Test dataset not found")

        tp = fp = fn = 0
        test_images = list(Path(self.TEST_DIR).glob("*.jpg"))[:100]

        for img_path in test_images:
            label_path = Path(self.LABELS_DIR) / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            results = self.model(img, conf=self.CONF_THRESH, verbose=False)

            # Ground truth
            gt_boxes = []
            for line in label_path.read_text().strip().split("\n"):
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) == 0:  # Class 0 = helmet
                    gt_boxes.append([float(p) for p in parts[1:]])

            # Predictions
            pred_boxes = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        pred_boxes.append(box.xyxyn[0].tolist())

            # Simple match count (production: use IoU)
            matched = min(len(gt_boxes), len(pred_boxes))
            tp += matched
            fp += max(0, len(pred_boxes) - len(gt_boxes))
            fn += max(0, len(gt_boxes) - len(pred_boxes))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        logger.info(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
        logger.info(f"TP={tp}  FP={fp}  FN={fn}  on {len(test_images)} images")

        self.assertGreater(precision, 0.80, "Precision must exceed 80%")
        self.assertGreater(recall,    0.75, "Recall must exceed 75%")


# ─────────────────────────────────────────────
# 2. INFERENCE SPEED TEST
# ─────────────────────────────────────────────
class TestInferenceSpeed(unittest.TestCase):
    """Measures inference FPS on current hardware."""

    MODEL_PATH  = "runs/train/helmet_v1/weights/best.pt"
    ITERATIONS  = 30
    TARGET_FPS  = 5    # Minimum acceptable FPS on Raspberry Pi 4

    @classmethod
    def setUpClass(cls):
        try:
            from ultralytics import YOLO
            cls.model = YOLO(cls.MODEL_PATH)
            cls.available = True
        except Exception:
            cls.available = False

    def test_inference_fps(self):
        if not self.available:
            self.skipTest("Model not available")

        dummy_frame = np.random.randint(0, 255,
                                        (480, 640, 3), dtype=np.uint8)
        times = []

        # Warm-up
        for _ in range(3):
            self.model(dummy_frame, verbose=False)

        # Timed runs
        for _ in range(self.ITERATIONS):
            t0 = time.perf_counter()
            self.model(dummy_frame, conf=0.70, verbose=False)
            times.append(time.perf_counter() - t0)

        avg_ms  = np.mean(times) * 1000
        min_ms  = np.min(times)  * 1000
        max_ms  = np.max(times)  * 1000
        fps     = 1.0 / np.mean(times)

        logger.info("=" * 40)
        logger.info(f"Inference Speed Results ({self.ITERATIONS} runs)")
        logger.info(f"  Avg: {avg_ms:.1f} ms  ({fps:.1f} FPS)")
        logger.info(f"  Min: {min_ms:.1f} ms")
        logger.info(f"  Max: {max_ms:.1f} ms")
        logger.info("=" * 40)

        self.assertGreater(fps, self.TARGET_FPS,
                           f"FPS {fps:.1f} below target {self.TARGET_FPS}")


# ─────────────────────────────────────────────
# 3. SERIAL COMMUNICATION TEST
# ─────────────────────────────────────────────
class TestSerialCommunication(unittest.TestCase):
    """Tests Serial link between Raspberry Pi and ESP32."""

    SERIAL_PORT = "/dev/ttyUSB0"
    BAUD_RATE   = 115200

    def setUp(self):
        try:
            import serial
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=2)
            time.sleep(2)
            self.serial_available = True
        except Exception as e:
            logger.warning(f"Serial not available: {e}")
            self.serial_available = False

    def tearDown(self):
        if self.serial_available:
            self.ser.close()

    def test_ping(self):
        if not self.serial_available:
            self.skipTest("Serial not available")
        self.ser.write(b"PING\n")
        response = self.ser.readline().decode().strip()
        self.assertEqual(response, "PONG")

    def test_ignition_on(self):
        if not self.serial_available:
            self.skipTest("Serial not available")
        self.ser.write(b"IGNITION:ON\n")
        time.sleep(0.5)
        # Read any response
        self.ser.flushInput()

    def test_ignition_off(self):
        if not self.serial_available:
            self.skipTest("Serial not available")
        self.ser.write(b"IGNITION:OFF\n")
        time.sleep(0.5)
        self.ser.flushInput()

    def test_status(self):
        if not self.serial_available:
            self.skipTest("Serial not available")
        self.ser.write(b"STATUS\n")
        response = self.ser.readline().decode().strip()
        logger.info(f"Status response: {response}")
        self.assertIn("ignition", response.lower())


# ─────────────────────────────────────────────
# 4. IMAGE BENCHMARK TEST
# ─────────────────────────────────────────────
class TestImageBenchmark(unittest.TestCase):
    """Tests helmet detection on known positive/negative images."""

    MODEL_PATH = "runs/train/helmet_v1/weights/best.pt"

    @classmethod
    def setUpClass(cls):
        try:
            from ultralytics import YOLO
            cls.model = YOLO(cls.MODEL_PATH)
            cls.available = True
        except Exception:
            cls.available = False

    def _detect(self, image: np.ndarray) -> bool:
        results = self.model(image, conf=0.70, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    return True
        return False

    def test_known_positive_image(self):
        """Should detect helmet in a synthetically-built positive frame."""
        if not self.available:
            self.skipTest("Model not available")

        img = np.full((480, 640, 3), 120, dtype=np.uint8)  # Gray bg

        img_path = "tests/sample_helmet.jpg"
        if not Path(img_path).exists():
            self.skipTest(f"Sample image not found: {img_path}")

        frame = cv2.imread(img_path)
        detected = self._detect(frame)
        self.assertTrue(detected, "Should detect helmet in positive sample")

    def test_known_negative_image(self):
        """Should NOT detect helmet in a known negative frame."""
        if not self.available:
            self.skipTest("Model not available")

        img_path = "tests/sample_no_helmet.jpg"
        if not Path(img_path).exists():
            self.skipTest(f"Sample image not found: {img_path}")

        frame = cv2.imread(img_path)
        detected = self._detect(frame)
        self.assertFalse(detected, "Should NOT detect helmet in negative sample")


# ─────────────────────────────────────────────
# 5. STRESS TEST
# ─────────────────────────────────────────────
def stress_test(duration_seconds: int = 60):
    """
    Runs detection in a loop for N seconds.
    Measures stability, memory usage, and FPS over time.
    """
    try:
        from ultralytics import YOLO
        import psutil
        model = YOLO("runs/train/helmet_v1/weights/best.pt")
    except Exception as e:
        logger.error(f"Stress test aborted: {e}")
        return

    logger.info(f"Starting {duration_seconds}s stress test...")
    start   = time.time()
    frames  = 0
    errors  = 0
    process = psutil.Process(os.getpid())

    while time.time() - start < duration_seconds:
        try:
            dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            model(dummy, verbose=False)
            frames += 1
        except Exception as e:
            errors += 1
            logger.warning(f"Error in frame {frames}: {e}")

    elapsed = time.time() - start
    fps     = frames / elapsed
    mem_mb  = process.memory_info().rss / 1024 / 1024

    logger.info("=" * 50)
    logger.info(f"Stress Test Results ({duration_seconds}s)")
    logger.info(f"  Frames processed : {frames}")
    logger.info(f"  Average FPS      : {fps:.2f}")
    logger.info(f"  Errors           : {errors}")
    logger.info(f"  Memory usage     : {mem_mb:.1f} MB")
    logger.info(f"  Error rate       : {errors/frames*100:.2f}%")
    logger.info("=" * 50)

    return {
        "frames": frames,
        "fps": fps,
        "errors": errors,
        "memory_mb": mem_mb,
        "duration_s": elapsed
    }


# ─────────────────────────────────────────────
# 6. GENERATE TEST REPORT
# ─────────────────────────────────────────────
def generate_test_report(results: dict, output_path: str = "tests/report.json"):
    Path("tests").mkdir(exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "results": results,
        "system": {
            "platform": os.uname().sysname if hasattr(os, 'uname') else "unknown",
            "python": __import__("sys").version,
        }
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Test report saved: {output_path}")


# ─────────────────────────────────────────────
# CLI RUNNER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["all", "accuracy", "speed",
                                           "serial", "stress"],
                        default="all")
    parser.add_argument("--stress-duration", type=int, default=60)
    args = parser.parse_args()

    if args.test in ("all", "speed", "accuracy", "serial"):
        suite = unittest.TestSuite()
        if args.test in ("all", "accuracy"):
            suite.addTests(unittest.TestLoader()
                           .loadTestsFromTestCase(TestModelAccuracy))
        if args.test in ("all", "speed"):
            suite.addTests(unittest.TestLoader()
                           .loadTestsFromTestCase(TestInferenceSpeed))
        if args.test in ("all", "serial"):
            suite.addTests(unittest.TestLoader()
                           .loadTestsFromTestCase(TestSerialCommunication))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

    if args.test in ("all", "stress"):
        stress_test(args.stress_duration)
