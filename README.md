# Study-Focus Monitor

A personal study-focus monitor built with traditional computer vision techniques. This system uses a webcam to observe study sessions and classifies the user's state as Focused, Drowsy, Distracted, or Absent. It tracks focus time throughout the session and generates summary statistics and visualizations at the end.

## Project Overview

This project is part of DSA4050 Computer Vision (Project 1) and demonstrates classical computer vision and machine learning techniques without deep learning. The system uses Haar cascade classifiers for face and eye detection, extracts hand-crafted features (HOG, LBP, geometric features), and employs classical machine learning algorithms (SVM, Random Forest, KNN) for state classification.

## Setup

1. Ensure Python 3.11 or 3.12 is installed
2. Clone this repository
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Grant camera permissions when prompted (macOS: System Settings > Privacy & Security > Camera)

## How to Run

### Complete Pipeline

1. **Data Collection**: Record training videos
   ```bash
   python src/collect_data.py
   ```
   Follow prompts to record videos for each state (focused, drowsy, distracted)

2. **Frame Extraction**: Extract frames from videos
   ```bash
   python src/extract_frames.py
   ```
   Automatically extracts frames at 5 FPS from collected videos

3. **Dataset Building**: Create feature dataset
   ```bash
   python src/build_dataset.py
   ```
   Processes frames to extract HOG, LBP, and geometric features

4. **Model Training**: Train and compare classifiers
   ```bash
   python src/train.py
   ```
   Trains SVM, Random Forest, and KNN; saves best model

5. **Evaluation** (optional): Detailed model evaluation
   ```bash
   python src/evaluate.py
   ```
   Generates confusion matrix, performance metrics, and ablation studies

### Running the Live Monitor

```bash
python src/app.py
```

**Controls:**
- `S`: Start/pause session monitoring
- `Q`: Quit and save session (generates summary and charts)
- `ESC`: Exit without saving

**Features:**
- Real-time state classification (Focused, Drowsy, Distracted, Absent)
- Live focus percentage and streak tracking
- Color-coded face detection boxes
- Session logging to CSV files
- End-of-session summary charts

**Output Files:**
- Session logs: `outputs/logs/session_YYYYMMDD_HHMMSS.csv`
- Session summary: `outputs/logs/summary_YYYYMMDD_HHMMSS.csv`
- Visualization charts: `outputs/figures/session_YYYYMMDD_HHMMSS.png`

## Project Structure

- `src/`: Source code modules for detection, features, training, and application
- `data/`: Dataset storage (raw videos and extracted frames)
- `models/`: Trained classifier and scaler files
- `notebooks/`: Jupyter notebooks for experimentation and analysis
- `outputs/`: Session logs and figures
- `docs/`: Technical report and documentation
- `config.py`: Central configuration for all parameters and paths

## Results Summary

- **Best Model**: SVM with RBF kernel
- **Test Accuracy**: 97.4% macro F1 score
- **Processing Speed**: 20,596 FPS (classification only), suitable for real-time
- **Key Features**: HOG features provide strongest signal for state classification

## License

This project is for educational purposes as part of DSA4050 Computer Vision course.