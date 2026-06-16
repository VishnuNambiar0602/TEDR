import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import TrafficAnalyzer
import os
import cv2

# Initialize
print("Initializing TrafficAnalyzer...")
analyzer = TrafficAnalyzer()

# Test Image
test_image = "test_data/traffic_images/traffic_02.jpg"

if not os.path.exists(test_image):
    print(f"Image not found: {test_image}")
    exit()

print(f"Testing on {test_image}...")
results = analyzer.analyze(test_image)

print("\n--- Detection Results ---")
print(f"Vehicles Detected: {results['vehicle_count']}")
print(f"Occupancy Ratio: {results['occupancy_ratio']}")
print(f"Congestion Level: {results['congestion_level']}")

# Save output
output_path = "test_data/results/tested_traffic_02.jpg"
os.makedirs("test_data/results", exist_ok=True)
cv2.imwrite(output_path, results['processed_image'])
print(f"\nResult saved to {output_path}")
