"""
Face and eye detection module using Haar cascades.
Core detection functionality for the Study-Focus Monitor.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import config


def load_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    """
    Load pre-trained Haar cascade classifiers for face and eye detection.
    
    Returns:
        Tuple of (face_cascade, eye_cascade) classifiers
    """
    cascade_path = cv2.data.haarcascades
    
    face_cascade = cv2.CascadeClassifier(
        cascade_path + 'haarcascade_frontalface_default.xml'
    )
    
    # Try eye_tree_eyeglasses first as it handles glasses better
    eye_cascade = cv2.CascadeClassifier(
        cascade_path + 'haarcascade_eye_tree_eyeglasses.xml'
    )
    
    # Verify cascades loaded successfully
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade classifier")
    if eye_cascade.empty():
        raise RuntimeError("Failed to load eye cascade classifier")
    
    return face_cascade, eye_cascade


def detect_face_and_eyes(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    eye_cascade: cv2.CascadeClassifier
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Detect faces and eyes in a frame.
    
    Key behaviors exploited for classification:
    - Eye cascade trained on OPEN eyes, fails when eyes closed (drowsiness cue)
    - Frontal face cascade fails on large head turns (absent/distracted cue)
    
    Args:
        frame: Input image frame (BGR)
        face_cascade: Face detection classifier
        eye_cascade: Eye detection classifier
    
    Returns:
        List of tuples: ((face_x, face_y, face_w, face_h), eye_detections_array)
    """
    # Convert to grayscale (Haar cascades need single channel)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=config.FACE_SCALE_FACTOR,
        minNeighbors=config.FACE_MIN_NEIGHBORS,
        minSize=config.FACE_MIN_SIZE
    )
    
    results = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y + h, x:x + w]
        
        # Search for eyes only in upper portion of face (reduces false positives)
        upper_face_height = int(h * config.EYE_SEARCH_REGION_RATIO)
        upper_face = face_roi[0:upper_face_height, :]
        
        # Detect eyes in upper face region
        eyes = eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=config.EYE_SCALE_FACTOR,
            minNeighbors=config.EYE_MIN_NEIGHBORS,
            minSize=config.EYE_MIN_SIZE
        )
        
        # Adjust eye coordinates to full frame coordinates
        eyes_adjusted = []
        for (ex, ey, ew, eh) in eyes:
            eyes_adjusted.append((x + ex, y + ey, ew, eh))
        
        results.append(((x, y, w, h), np.array(eyes_adjusted)))
    
    return results


def get_eye_region(
    frame: np.ndarray,
    face_box: Tuple[int, int, int, int],
    padding: float = 0.1
) -> Optional[np.ndarray]:
    """
    Extract the eye-band region from a detected face.
    
    Args:
        frame: Input frame
        face_box: Face bounding box (x, y, w, h)
        padding: Additional padding around eye region
    
    Returns:
        Cropped and resized eye-band region, or None if invalid
    """
    x, y, w, h = face_box
    
    # Define eye-band region (upper portion of face)
    eye_band_top = int(y + h * 0.2)
    eye_band_bottom = int(y + h * 0.6)
    eye_band_left = int(x - w * padding)
    eye_band_right = int(x + w + w * padding)
    
    # Clip to frame boundaries
    height, width = frame.shape[:2]
    eye_band_top = max(0, eye_band_top)
    eye_band_bottom = min(height, eye_band_bottom)
    eye_band_left = max(0, eye_band_left)
    eye_band_right = min(width, eye_band_right)
    
    # Extract region
    if eye_band_bottom > eye_band_top and eye_band_right > eye_band_left:
        eye_region = frame[eye_band_top:eye_band_bottom, eye_band_left:eye_band_right]
        
        # Resize to standard size for consistent feature extraction
        eye_region = cv2.resize(
            eye_region, 
            (config.EYE_BAND_WIDTH, config.EYE_BAND_HEIGHT)
        )
        
        return eye_region
    
    return None


def draw_detections(
    frame: np.ndarray,
    detections: List[Tuple[Tuple[int, int, int, int], np.ndarray]],
    state: str = "Unknown"
) -> np.ndarray:
    """
    Draw detection boxes and state on frame for visualization.
    
    Args:
        frame: Input frame to draw on
        detections: List of (face_box, eye_boxes) tuples
        state: Current detected state
    
    Returns:
        Frame with drawn detections
    """
    frame_copy = frame.copy()
    
    for (face_box, eyes) in detections:
        # Draw face box
        x, y, w, h = face_box
        cv2.rectangle(
            frame_copy,
            (x, y),
            (x + w, y + h),
            config.COLOR_FACE_BOX,
            2
        )
        
        # Draw eye boxes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                frame_copy,
                (ex, ey),
                (ex + ew, ey + eh),
                config.COLOR_EYE_BOX,
                1
            )
        
        # Draw eye count
        eye_count_text = f"Eyes: {len(eyes)}"
        cv2.putText(
            frame_copy,
            eye_count_text,
            (x, y - 10),
            config.FONT,
            config.FONT_SCALE,
            config.COLOR_FACE_BOX,
            config.FONT_THICKNESS
        )
    
    # Draw current state
    state_color = {
        "Focused": config.COLOR_FOCUSED,
        "Drowsy": config.COLOR_DROWSY,
        "Distracted": config.COLOR_DISTRACTED,
        "Absent": config.COLOR_ABSENT
    }.get(state, (255, 255, 255))
    
    cv2.putText(
        frame_copy,
        f"State: {state}",
        (10, 30),
        config.FONT,
        config.FONT_SCALE,
        state_color,
        config.FONT_THICKNESS
    )
    
    return frame_copy