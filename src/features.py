#!/usr/bin/env python3
"""
Feature extraction module for Study-Focus Monitor.
Extracts HOG, LBP, geometric, and pupil position features from face images.
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from typing import List, Optional, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.detection import detect_face_and_eyes, get_eye_region


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients features.
    HOG captures edge structure that distinguishes open from closed eyes.
    
    Args:
        image: Grayscale image of eye region
    
    Returns:
        HOG feature vector
    """
    # Ensure image is the right size
    if image.shape != (config.EYE_BAND_HEIGHT, config.EYE_BAND_WIDTH):
        image = cv2.resize(image, (config.EYE_BAND_WIDTH, config.EYE_BAND_HEIGHT))
    
    # Extract HOG features
    features = hog(
        image,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm=config.HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True
    )
    
    return features


def extract_lbp_features(image: np.ndarray) -> np.ndarray:
    """
    Extract Local Binary Pattern histogram features.
    LBP captures fine texture differences between open and closed eyes.
    
    Args:
        image: Grayscale image of eye region
    
    Returns:
        LBP histogram feature vector
    """
    # Ensure image is the right size
    if image.shape != (config.EYE_BAND_HEIGHT, config.EYE_BAND_WIDTH):
        image = cv2.resize(image, (config.EYE_BAND_WIDTH, config.EYE_BAND_HEIGHT))
    
    # Compute LBP
    lbp = local_binary_pattern(
        image,
        P=config.LBP_POINTS,
        R=config.LBP_RADIUS,
        method=config.LBP_METHOD
    )
    
    # Create histogram
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, config.LBP_BINS + 1),
        density=True
    )
    
    return hist


def get_pupil_position(eye_region: np.ndarray) -> float:
    """
    Detect pupil position within eye region for gaze estimation.
    The pupil is typically the darkest blob in the eye.
    
    Args:
        eye_region: Grayscale eye region
    
    Returns:
        Horizontal position ratio (0=left, 0.5=center, 1=right)
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(eye_region, (7, 7), 0)
    
    # Threshold to find dark regions (pupil)
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.5  # Default to center if no pupil found
    
    # Get largest contour (likely the pupil)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get centroid
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return 0.5
    
    cx = M["m10"] / M["m00"]
    
    # Return normalized horizontal position
    return cx / eye_region.shape[1]


def extract_geometric_features(
    face_box: Tuple[int, int, int, int],
    eye_boxes: np.ndarray,
    frame_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Extract geometric features from face and eye positions.
    
    Args:
        face_box: Face bounding box (x, y, w, h)
        eye_boxes: Array of eye bounding boxes
        frame_shape: Shape of original frame (height, width)
    
    Returns:
        Geometric feature vector
    """
    features = []
    
    # Number of eyes detected (0, 1, or 2)
    num_eyes = len(eye_boxes) if len(eye_boxes) <= 2 else 2
    features.append(num_eyes)
    
    # Face position relative to frame (centered = looking straight)
    face_x, face_y, face_w, face_h = face_box
    face_center_x = (face_x + face_w / 2) / frame_shape[1]
    face_center_y = (face_y + face_h / 2) / frame_shape[0]
    features.extend([face_center_x, face_center_y])
    
    # Face size relative to frame
    face_size_ratio = (face_w * face_h) / (frame_shape[0] * frame_shape[1])
    features.append(face_size_ratio)
    
    # Eye positions if detected
    if num_eyes > 0:
        for i in range(min(2, num_eyes)):
            eye_x, eye_y, eye_w, eye_h = eye_boxes[i]
            # Eye center relative to face box
            eye_center_x = ((eye_x + eye_w / 2) - face_x) / face_w
            eye_center_y = ((eye_y + eye_h / 2) - face_y) / face_h
            features.extend([eye_center_x, eye_center_y])
        
        # Pad if only one eye
        if num_eyes == 1:
            features.extend([0.5, 0.5])  # Default position
    else:
        # No eyes detected, use default values
        features.extend([0.5, 0.5, 0.5, 0.5])
    
    # Inter-eye distance if both eyes detected
    if num_eyes == 2:
        eye1_x = eye_boxes[0][0] + eye_boxes[0][2] / 2
        eye2_x = eye_boxes[1][0] + eye_boxes[1][2] / 2
        inter_eye_distance = abs(eye1_x - eye2_x) / face_w
        features.append(inter_eye_distance)
    else:
        features.append(0.0)
    
    return np.array(features)


def extract_features_from_frame(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier
) -> Optional[np.ndarray]:
    """
    Extract all features from a single frame.
    
    Args:
        frame: Input frame (BGR)
        face_cascade: Face detection cascade
        eye_cascade: Eye detection cascade
    
    Returns:
        Combined feature vector or None if no face detected
    """
    # Detect face and eyes
    detections = detect_face_and_eyes(frame, face_cascade, eye_cascade)
    
    if not detections:
        return None
    
    # Get first detected face
    face_box, eye_boxes = detections[0]
    
    # Get eye region
    eye_region = get_eye_region(frame, face_box)
    if eye_region is None:
        return None
    
    # Convert to grayscale if needed
    if len(eye_region.shape) == 3:
        eye_region_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    else:
        eye_region_gray = eye_region
    
    # Extract HOG features
    hog_features = extract_hog_features(eye_region_gray)
    
    # Extract LBP features
    lbp_features = extract_lbp_features(eye_region_gray)
    
    # Extract geometric features
    geometric_features = extract_geometric_features(
        face_box, eye_boxes, frame.shape[:2]
    )
    
    # Extract pupil positions for detected eyes
    pupil_features = []
    if len(eye_boxes) > 0:
        for i in range(min(2, len(eye_boxes))):
            ex, ey, ew, eh = eye_boxes[i]
            eye_img = frame[ey:ey+eh, ex:ex+ew]
            if eye_img.size > 0:
                eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                pupil_x = get_pupil_position(eye_gray)
                pupil_features.append(pupil_x)
        
        # Pad if needed
        while len(pupil_features) < 2:
            pupil_features.append(0.5)
    else:
        pupil_features = [0.5, 0.5]
    
    # Combine all features
    features = np.concatenate([
        hog_features,
        lbp_features,
        geometric_features,
        pupil_features
    ])
    
    return features


def get_feature_names() -> List[str]:
    """
    Get names for all features in the feature vector.
    
    Returns:
        List of feature names
    """
    names = []
    
    # HOG features
    n_hog = len(extract_hog_features(np.zeros((config.EYE_BAND_HEIGHT, config.EYE_BAND_WIDTH))))
    names.extend([f'hog_{i}' for i in range(n_hog)])
    
    # LBP features
    names.extend([f'lbp_bin_{i}' for i in range(config.LBP_BINS)])
    
    # Geometric features
    names.extend([
        'num_eyes',
        'face_center_x',
        'face_center_y',
        'face_size_ratio',
        'eye1_center_x',
        'eye1_center_y',
        'eye2_center_x',
        'eye2_center_y',
        'inter_eye_distance'
    ])
    
    # Pupil features
    names.extend(['pupil1_x', 'pupil2_x'])
    
    return names


def main():
    """Test feature extraction on a sample frame."""
    print("Testing feature extraction...")
    
    # Load cascades
    from src.detection import load_cascades
    face_cascade, eye_cascade = load_cascades()
    
    # Test with a sample frame
    test_frames = list(Path('data/frames/focused').glob('*.jpg'))
    
    if test_frames:
        # Load first frame
        frame = cv2.imread(str(test_frames[0]))
        
        # Extract features
        features = extract_features_from_frame(frame, face_cascade, eye_cascade)
        
        if features is not None:
            print(f" Extracted {len(features)} features")
            print(f"  HOG features: {len([n for n in get_feature_names() if 'hog' in n])}")
            print(f"  LBP features: {len([n for n in get_feature_names() if 'lbp' in n])}")
            print(f"  Geometric features: 9")
            print(f"  Pupil features: 2")
            print(f"  Total: {len(features)}")
            
            # Check for NaN or Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print("Warning: Features contain NaN or Inf values")
            else:
                print(" All features are finite")
        else:
            print("Failed to extract features (no face detected)")
    else:
        print("No test frames found. Run data collection first.")


if __name__ == "__main__":
    main()