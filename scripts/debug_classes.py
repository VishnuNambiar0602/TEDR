import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import TrafficAnalyzer
import os

analyzer = TrafficAnalyzer()
print("Model Classes:", analyzer.model.names)

# Test Image
test_image = "test_data/traffic_images/Traffic_1.jpg"
results = analyzer.model(test_image, conf=0.1)

for r in results:
    print(f"Detected {len(r.boxes)} boxes")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        print(f" - Class: {cls_id} ({analyzer.model.names[cls_id]}) Conf: {float(box.conf[0]):.2f}")
