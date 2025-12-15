# Tennis Hit & Bounce Detection

This project implements two methods for detecting tennis ball hits and bounces from ball-tracking data extracted from the Roland-Garros 2025 Final.

## Project Structure

```
Tennis Hits & Bounces/
├── main.py                          # Production inference CLI
├── supervised.py                    # ML training & prediction
├── unsupervised.py                  # Physics-based baseline
├── utils.py                         # Feature engineering utilities
├── evaluate.py                      # Quick evaluation script
├── cross_validate.py                # K-fold cross-validation
├── diagnostics.py                   # Visualization & debugging
├── error_analysis.py                # False positive analysis
├── hit_bounce_model.pkl             # Trained XGBoost model (70 features)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── Data hit & bounce/
    └── per_point_v2/                # 313 labeled JSON files
        ├── ball_data_1.json
        ├── ball_data_2.json
        └── ...
```

**Note**: Temporary output files (diagnostics plots, CSV results) are gitignored and regenerated on demand.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The project provides two main functions as specified:

### 1. Unsupervised Detection

```python
from main import unsupervised_hit_bounce_detection

# Pass a file path
result = unsupervised_hit_bounce_detection("path/to/ball_data_X.json")

# Or pass a dict
with open("path/to/ball_data_X.json") as f:
    data = json.load(f)
result = unsupervised_hit_bounce_detection(data)
```

### 2. Supervised Detection

```python
from main import supervised_hit_bounce_detection

# Pass a file path
result = supervised_hit_bounce_detection("path/to/ball_data_X.json")

# Or pass a dict
with open("path/to/ball_data_X.json") as f:
    data = json.load(f)
result = supervised_hit_bounce_detection(data)
```

### Command Line Interface

```bash
# Unsupervised method
python main.py --input "Data hit & bounce/per_point_v2/ball_data_1.json" --method unsupervised --output output.json

# Supervised method
python main.py --input "Data hit & bounce/per_point_v2/ball_data_1.json" --method supervised --output output.json
```

## Methods

### Method 1: Unsupervised (Physics-Based)

The unsupervised method analyzes the ball's trajectory using physics principles:

- **Bounce Detection**: Identifies local maxima in the Y-coordinate (ball reaching ground) combined with high vertical acceleration
- **Hit Detection**: Detects sudden changes in horizontal velocity and acceleration spikes, filtered by speed increase (racket adds energy)

Key features analyzed:

- Vertical and horizontal velocity (vx, vy)
- Acceleration (ax, ay)
- Speed magnitude
- Direction changes

### Method 2: Supervised (Machine Learning)

The supervised method uses an **XGBoost Classifier** (or Random Forest if XGBoost is unavailable) trained on labeled data:

- **Model**: XGBoost (200 estimators) or Random Forest
- **Features**: Position (y), velocities (vx, vy, speed), accelerations (ax, ay, accel), jerk, angles, and rolling statistics
- **Train/Test Split**: Split by Point ID (80/20) to prevent data leakage
- **Performance**: ~99% weighted F1-score on test set

## Training the Supervised Model

The model is pre-trained and saved as `hit_bounce_model.pkl`. To retrain:

```python
from supervised import train_model

train_model("path/to/data/folder")
```

## Output Format

Both methods return a JSON structure identical to the input, with an additional `"pred_action"` key:

```json
{
  "56100": {
    "x": 894.0,
    "y": 395.0,
    "visible": true,
    "action": "air",
    "pred_action": "bounce"
  }
}
```

## Data Format

Input JSON files contain frame-indexed ball tracking data:

- **Frame Number** (key): Video frame number
- **x**: Horizontal pixel position (0-1920)
- **y**: Vertical pixel position (0-1080)
- **visible**: Boolean indicating if ball was detected
- **action**: Ground truth label ("air", "hit", "bounce")

## Implementation Details

### Feature Engineering

- **Interpolation**: Missing frames are interpolated linearly
- **Derivatives**: First and second derivatives computed for velocity and acceleration
- **Rolling Statistics**: Temporal context captured via rolling mean/std (windows: 3, 5 frames)

### Unsupervised Heuristics (Improved Phase 3)

- **Smoothing**: Savitzky-Golay filter applied to X/Y coordinates to reduce noise.
- **Bounce Detection**: Local Y maxima + strict vertical velocity reversal (V-shape) + acceleration spike.
- **Hit Detection**: High acceleration/jerk + trajectory reset (dot product check) + energy verification.
- **Deduplication**: Clustering-based Non-Maximum Suppression (NMS) to merge consecutive detections.

### Supervised Model

- Algorithm: Random Forest (robust to non-linear patterns, handles imbalanced classes)
- Cross-validation: Point-level split ensures no temporal leakage
- Features: 12 engineered features per frame

## Performance

### Cross-Validation Results (5-Fold on 313 Points)

**Supervised Model Performance**:

| Metric        | Air   | Bounce | Hit   | Macro Avg |
| ------------- | ----- | ------ | ----- | --------- |
| **Recall**    | 99.1% | 62.4%  | 70.5% | 77.3%     |
| **Precision** | 99.4% | 55.9%  | 55.8% | 70.4%     |
| **F1-Score**  | 99.2% | 58.9%  | 62.2% | 73.5%     |

**Overall Metrics**:

- **Accuracy**: 99.4%
- **Macro F1**: 0.735 (±0.007 std across folds)
- **Inference Time**: <100ms per point

**Key Achievements**:

- ✅ Detects 62% of bounces and 71% of hits in highly imbalanced data (99% air frames)
- ✅ Robust performance with low variance across folds
- ✅ Production-ready pipeline with physics-based validation

The unsupervised method provides a physics-based baseline for comparison and validation.

### Performance Reports

- See `FINAL_EVALUATION_REPORT.md` for comprehensive analysis
- See `cv_results.csv` for detailed per-fold metrics
- See `false_positive_report.csv` for error analysis

## Diagnostics

To analyze improved model performance and generate visualizations:

```bash
python diagnostics.py
```

This will create a `diagnostics/` folder with trajectory and physics plots for sample data.

## Author

EL BAHRAOUI HASSAN
