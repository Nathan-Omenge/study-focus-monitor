"""
Central configuration file for Study-Focus Monitor.
All paths, parameters, and thresholds are defined here.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
FOCUSED_FRAMES_DIR = FRAMES_DIR / "focused"
DROWSY_FRAMES_DIR = FRAMES_DIR / "drowsy"
DISTRACTED_FRAMES_DIR = FRAMES_DIR / "distracted"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Notebook paths
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Source code paths
SRC_DIR = PROJECT_ROOT / "src"

# Documentation paths
DOCS_DIR = PROJECT_ROOT / "docs"
REPORT_DIR = DOCS_DIR / "report"

# Test paths
TESTS_DIR = PROJECT_ROOT / "tests"

# Image processing parameters
STANDARD_FRAME_WIDTH = 640
STANDARD_FRAME_HEIGHT = 480
EYE_BAND_HEIGHT = 32
EYE_BAND_WIDTH = 64

# Face detection parameters (Haar cascade)
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (80, 80)

# Eye detection parameters (Haar cascade)
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 6
EYE_MIN_SIZE = (20, 20)
EYE_SEARCH_REGION_RATIO = 0.6  # Search for eyes in top 60% of face

# Feature extraction parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BINS = 10

# Temporal layer parameters
SMOOTHING_WINDOW_SIZE = 5  # frames
BLINK_MAX_DURATION_MS = 400  # milliseconds
DROWSY_MIN_DURATION_S = 2.0  # seconds

# Model training parameters
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5

# SVM parameters (for grid search)
SVM_C_RANGE = [0.1, 1, 10]
SVM_GAMMA_RANGE = ["scale", "auto", 0.001, 0.01]

# Random Forest parameters (for grid search)
RF_N_ESTIMATORS_RANGE = [50, 100, 200]
RF_MAX_DEPTH_RANGE = [10, 20, None]

# KNN parameters (for grid search)
KNN_N_NEIGHBORS_RANGE = [3, 5, 7, 9]

# Video capture parameters
CAMERA_INDEX = 0  # Default webcam
FPS = 30
FRAME_SKIP = 5  # Sample every 5th frame when extracting from video

# Application parameters
APP_WINDOW_NAME = "Study Focus Monitor"
APP_UPDATE_INTERVAL_MS = 33  # ~30 FPS
SESSION_LOG_INTERVAL_S = 1  # Log state every second

# Display colors (BGR format for OpenCV)
COLOR_FOCUSED = (0, 255, 0)  # Green
COLOR_DROWSY = (0, 0, 255)  # Red
COLOR_DISTRACTED = (0, 165, 255)  # Orange
COLOR_ABSENT = (128, 128, 128)  # Gray
COLOR_FACE_BOX = (255, 255, 0)  # Cyan
COLOR_EYE_BOX = (255, 0, 255)  # Magenta

# Font settings
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2