import os
import sys
import time
import argparse
import cv2
from ultralytics import YOLO

# ---- Argument Parsing ----
parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Detection with PiCamera or USB")
parser.add_argument('--model', required=True, help='Path to YOLO model (e.g. yolov8n.pt)')
parser.add_argument('--resolution', default="640x480", help='Resolution (e.g. 640x480)')
parser.add_argument('--record', action='store_true', help='Record the output video')
args = parser.parse_args()

# ---- Config ----
res_width, res_height = map(int, args.resolution.split('x'))
model_path = args.model

# ---- Load YOLO Model ----
try:
    model = YOLO(model_path)
    print("? YOLO model loaded successfully.")
except Exception as e:
    print(f"? Error loading model: {e}")
    sys.exit(1)

# ---- GStreamer Camera Pipeline ----
print("?? Opening PiCamera via GStreamer...")
gst_pipeline = (
    f"libcamerasrc ! video/x-raw,width={res_width},height={res_height},framerate=30/1 ! "
    f"videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("? Failed to open PiCamera using GStreamer.")
    sys.exit(1)

# ---- Video Recorder ----
if args.record:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 10, (res_width, res_height))
    print("?? Recording enabled: output.avi")

# ---- Display Window ----
cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", res_width, res_height)

print("?? Starting detection... Press 'q' to quit.")
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("?? Failed to grab frame.")
        break

    frame_count += 1

    # Run YOLOv8 inference
    try:
        results = model.predict(source=frame, conf=0.5, verbose=False)
    except Exception as e:
        print(f"?? YOLOv8 inference error: {e}")
        continue

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            if conf >= 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # FPS display
    fps = frame_count / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show detection frame
    cv2.imshow("YOLOv8 Detection", frame)

    if args.record:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Cleanup ----
print("?? Exiting...")
cap.release()
if args.record:
    out.release()
cv2.destroyAllWindows()
