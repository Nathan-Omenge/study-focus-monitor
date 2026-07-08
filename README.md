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

### Data Collection
```bash
python src/collect_data.py
```

### Training the Model
```bash
python src/train.py
```

### Running the Live Monitor
```bash
python src/app.py
```

### Controls
- `s`: Start/pause session
- `q`: Quit and save session summary

## Project Structure

- `src/`: Source code modules for detection, features, training, and application
- `data/`: Dataset storage (raw videos and extracted frames)
- `models/`: Trained classifier and scaler files
- `notebooks/`: Jupyter notebooks for experimentation and analysis
- `outputs/`: Session logs and figures
- `docs/`: Technical report and documentation
- `config.py`: Central configuration for all parameters and paths

## Results Summary

(To be updated after model training and evaluation)

## License

This project is for educational purposes as part of DSA4050 Computer Vision course.