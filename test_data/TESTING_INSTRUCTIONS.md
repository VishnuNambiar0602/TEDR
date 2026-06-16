
# TESTING GUIDE - Real Data

## Option 1: Test Image Congestion Detector

The sample traffic images have been downloaded to `test_data/traffic_images/`.

### Run detection on a single image:
```bash
python image_congestion_detector.py test_data/traffic_images/traffic_01.jpg
```

### Batch test all images:
```bash
python test_with_images.py
```

Expected output:
- Detection visualization with bounding boxes
- Vehicle count
- Congestion level (LOW/MEDIUM/HIGH)
- Results saved to test_data/results/

---

## Option 2: Test Traffic Preprocessing

Sample traffic CSV created: `test_data/traffic_csv/sample_traffic.csv`

### Preprocess the sample data:
```bash
python preprocess_traffic.py --file test_data/traffic_csv/sample_traffic.csv --freq H --lookback 24 --horizon 1 --output-dir test_data/processed
```

Expected output:
- Processed parquet files
- NumPy arrays (X_train, y_train, X_val, y_val, X_test, y_test)
- Statistics summary
- Results in test_data/processed/

---

## Option 3: Full Integration Test

Run the complete test suite:
```bash
python run_full_tests.py
```

This will:
1. Test image detection on all sample images
2. Preprocess traffic data
3. Generate reports
4. Validate outputs

---

## Dataset Recommendations

### For Image Detection:
1. **Kaggle Traffic Datasets:**
   - "Traffic and Road Safety Data" (kaggle.com/abhishekh5/road-traffic-signal-detection)
   - "Vehicle Detection Dataset" (kaggle.com/slamarc/vehicle-detection-coco-dataset)

2. **Public Sources:**
   - COCO Dataset: Has traffic/street scenes
   - BDD100K: Berkeley DeepDrive video dataset
   - KITTI: Autonomous driving dataset

### For Traffic Time-Series:
1. **Kaggle Traffic-Net:**
   - Use the built-in Kaggle integration in preprocess_traffic.py
   - Command: `python preprocess_traffic.py --file train.csv`

2. **Other Kaggle Datasets:**
   - "Traffic Volume" (UCI dataset on Kaggle)
   - "NYC Taxi Dataset"
   - "Beijing Air Quality" (includes traffic-related features)

---

## Interpreting Results

### Image Detection Output:
- **Vehicle Count:** Total vehicles detected
- **Vehicle Area %:** Percentage of image covered by vehicles
  - < 15% = LOW congestion (green)
  - 15-35% = MEDIUM congestion (yellow)
  - > 35% = HIGH congestion (red)

### Traffic Preprocessing Output:
- **X_train.npy:** Features for training (shape: samples × features)
- **y_train.npy:** Target values (traffic volume to predict)
- **metadata.json:** Feature names, scaler parameters, statistics
- **parquet files:** Full processed data in tabular format

