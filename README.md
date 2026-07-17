# Study-Focus Monitor

A computer vision system that monitors study sessions and tracks focus levels using traditional machine learning techniques.

## Project Overview

This project uses a webcam to watch you study and automatically classifies your state as:
- **Focused** - Looking at screen, engaged
- **Drowsy** - Eyes closing, falling asleep  
- **Distracted** - Looking away, checking phone
- **Absent** - Not at desk

Built for DSA4050 Computer Vision course using only traditional CV techniques (no deep learning).

## Prerequisites

- Python 3.11 or 3.12
- Webcam
- macOS, Windows, or Linux
- About 500MB free space

## Quick Start Guide

### Step 1: Clone the Repository
```bash
git clone [repository-url]
cd study-focus-monitor
```

### Step 2: Set Up Python Environment

**Option A: Using existing virtual environment**
```bash
# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

**Option B: Create new virtual environment**
```bash
# Create new virtual environment
python -m venv venv
source venv/bin/activate   # On macOS/Linux
# OR
venv\Scripts\activate       # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## Running the Application

### Quick Test (Using Pre-trained Model)

If a trained model already exists, you can run the monitor immediately:
```bash
python src/app.py
```

### Full Pipeline (Train Your Own Model)

1. **Collect Training Data**
   ```bash
   python src/collect_data.py
   ```
   - Record 30-second videos for each state
   - Follow on-screen prompts
   - Need at least 3 videos total

2. **Extract Frames**
   ```bash
   python src/extract_frames.py
   ```
   - Automatically processes all videos
   - Creates labeled frame dataset

3. **Build Feature Dataset**
   ```bash
   python src/build_dataset.py
   ```
   - Extracts features from frames
   - Creates training dataset

4. **Train Model**
   ```bash
   python src/train.py
   ```
   - Compares SVM, Random Forest, KNN
   - Saves best model automatically

5. **Run Live Monitor**
   ```bash
   python src/app.py
   ```

## Using the Live Monitor

### Controls
- **S** - Start/pause monitoring
- **Q** - Quit and save session  
- **ESC** - Exit without saving

### What You'll See
- Green box = Focused
- Orange box = Drowsy
- Red box = Distracted
- Gray box = Absent
- Live statistics showing focus percentage and current streak

### Output Files
After each session, find your results in:
- `outputs/logs/` - CSV files with detailed logs
- `outputs/figures/` - Charts showing your focus patterns

## Performance

- **Accuracy**: 97.4%
- **Speed**: 30+ FPS (real-time)
- **Features**: 777-dimensional vector using HOG, LBP, and geometric features
- **Best Classifier**: SVM with RBF kernel

## Troubleshooting

### Camera Not Found
- Check camera permissions (System Settings > Privacy > Camera)
- Try unplugging and reconnecting external webcam
- Close other apps using camera

### Low FPS or Lag
- Close unnecessary applications
- Ensure good lighting
- Check CPU usage

### Poor Detection
- Adjust lighting (avoid backlight)
- Position camera at eye level
- Remove reflective glasses if possible

### Module Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### OpenCV Version Issues
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.10.0.84
```

## Project Structure

```
study-focus-monitor/
├── src/                 # Source code
│   ├── app.py          # Main application
│   ├── detection.py    # Face/eye detection
│   ├── features.py     # Feature extraction
│   ├── train.py        # Model training
│   └── ...
├── data/               # Training data
├── models/             # Saved models
├── outputs/            # Session results
├── docs/               # Documentation
├── config.py           # Configuration
└── requirements.txt    # Dependencies
```

## For Assessors/Evaluators

### Quick Evaluation Setup

1. **Clone and setup** (5 minutes)
   ```bash
   git clone [repository-url]
   cd study-focus-monitor
   python -m venv test_env
   source test_env/bin/activate  # or test_env\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Test with pre-trained model** (2 minutes)
   ```bash
   python src/app.py
   # Press 'S' to start monitoring
   # Try different poses: looking at screen, closing eyes, looking away
   # Press 'Q' to see summary
   ```

3. **Review outputs**
   - Check `outputs/figures/` for session charts
   - Check `outputs/logs/` for detailed CSVs

### Key Files to Review

- `src/app.py` - Main application logic
- `src/features.py` - Feature extraction implementation
- `src/train.py` - Model training pipeline
- `docs/report/technical_report.md` - Full technical documentation
- `config.py` - All parameters and thresholds

### Evaluation Criteria Met

- Uses Haar cascades (mandatory requirement)  
- No deep learning (traditional CV only)  
- Complete pipeline from acquisition to classification  
- Real-time capable (30+ FPS)  
- Comprehensive evaluation metrics  
- Working prototype with GUI  

## Documentation

- [Technical Report](docs/report/technical_report.md) - Full project documentation
- [Requirements Compliance](docs/requirements_compliance.md) - Requirement verification
- [Project Plan](PROJECT_PLAN.md) - Development phases

## Author

Nathan Orang'o  
ID: 670637  
DSA4050 Computer Vision  
2024

## License

Educational project for DSA4050 Computer Vision course.

---

**Repository**: https://github.com/Nathan-Omenge/study-focus-monitor  
**Demo Video**: [Optional: Add link to demo video]