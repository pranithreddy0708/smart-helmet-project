"""
DEBUG TOOL — Smart Helmet System
Run this FIRST to diagnose false positives.

It shows you EVERY class your model detects in the camera feed,
so you can see exactly why the false positive is happening.

Usage:
    python debug_detection.py
    python debug_detection.py --image path/to/photo.jpg
    python debug_detection.py --model yolov8n.pt  # test with base COCO model
"""

import cv2
import sys
import argparse
import numpy as np

def run_debug(model_path: str, image_path: str = None, camera: int = 0):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)

    print("\n=== MODEL CLASS NAMES ===")
    for cls_id, name in model.names.items():
        print(f"  Class {cls_id:3d} = '{name}'")
    print("=========================\n")

    print("Press Q to quit | Press S to save screenshot\n")

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Cannot read: {image_path}")
            sys.exit(1)
        frames = [frame]
        use_camera = False
    else:
        cap = cv2.VideoCapture(camera)
        use_camera = True

    def process_frame(frame):
        results = model(frame, conf=0.30, verbose=False)  # Low threshold to catch everything
        annotated = frame.copy()
        found = []

        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_helmet = "helmet" in cls_name.lower()
                color = (0, 255, 0) if is_helmet else (0, 100, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"[{cls_id}] {cls_name}: {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(y1-8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                found.append((cls_id, cls_name, conf))
                print(f"  Detected: class={cls_id} ('{cls_name}')  conf={conf:.3f}  "
                      f"bbox=({x1},{y1},{x2},{y2})")

        # Status overlay
        h, w = annotated.shape[:2]
        helmet_found = any("helmet" in n.lower() for _, n, _ in found)
        status = "HELMET DETECTED" if helmet_found else "NO HELMET"
        color  = (0, 255, 0) if helmet_found else (0, 0, 255)

        cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(annotated, status, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Warning if non-helmet is triggering
        if found and not helmet_found:
            msg = f"FALSE POSITIVE RISK: detecting '{found[0][1]}' (not helmet!)"
            cv2.putText(annotated, msg, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            print(f"  ⚠ WARNING: {msg}")

        if not found:
            print("  (nothing detected above 0.30 confidence)")

        return annotated

    if not use_camera:
        annotated = process_frame(frames[0])
        cv2.imshow("DEBUG — All Detections", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:  # Process every 5th frame
            print(f"\n--- Frame {frame_count} ---")
            annotated = process_frame(frame)
        else:
            annotated = frame

        cv2.imshow("DEBUG — All Detections (threshold=0.30)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
            print(f"Saved: debug_frame_{frame_count}.jpg")

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== DIAGNOSIS GUIDE ===")
    print("If you see class='person' being detected instead of 'helmet':")
    print("  → You are using BASE YOLOv8 (COCO model) — it has no helmet class")
    print("  → You MUST train/use a helmet-specific fine-tuned model")
    print("  → Download fine-tuned model from: https://universe.roboflow.com")
    print()
    print("If you see class='helmet' but no actual helmet in frame:")
    print("  → Model confidence threshold needs to be raised (try 0.85)")
    print("  → Model needs more fine-tuning with negative examples")
    print("========================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="models/helmet_yolov8.pt")
    parser.add_argument("--image",  default=None)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_debug(args.model, args.image, args.camera)
