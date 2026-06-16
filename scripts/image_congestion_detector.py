import cv2
import os
import sys
from ultralytics import YOLO

# =========================
# USAGE CHECK
# =========================
if len(sys.argv) != 2:
    print("Usage:")
    print("python image_congestion_detector.py <path_to_image>")
    print("Example:")
    print("python image_congestion_detector.py test_images/road1.jpg")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found -> {IMAGE_PATH}")
    sys.exit(1)


# =========================
# CONFIGURATION
# =========================
CONF_THRESHOLD = 0.4

VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike"}

LOW_CONGESTION = 0.15
HIGH_CONGESTION = 0.35

# =========================
# LOAD MODEL
# =========================
print("[INFO] Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
class_names = model.names

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread(IMAGE_PATH)
height, width, _ = img.shape
image_area = height * width

# =========================
# RUN INFERENCE
# =========================
print("[INFO] Running vehicle detection...")
results = model(img, conf=CONF_THRESHOLD)

vehicle_count = 0
vehicle_area = 0

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        class_name = class_names[cls_id]

        if class_name not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        vehicle_area += area
        vehicle_count += 1

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            class_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

# =========================
# CONGESTION ESTIMATION
# =========================
occupancy_ratio = vehicle_area / image_area

if occupancy_ratio < LOW_CONGESTION:
    congestion_level = "LOW"
elif occupancy_ratio < HIGH_CONGESTION:
    congestion_level = "MEDIUM"
else:
    congestion_level = "HIGH"

# =========================
# OUTPUT
# =========================
print("\n===== RESULT =====")
print(f"Image: {IMAGE_PATH}")
print(f"Vehicles detected: {vehicle_count}")
print(f"Road occupancy ratio: {round(occupancy_ratio, 3)}")
print(f"Congestion level: {congestion_level}")

label = f"Vehicles: {vehicle_count} | Congestion: {congestion_level}"
cv2.putText(
    img,
    label,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imshow("Road Congestion Estimation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
