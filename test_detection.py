#!/usr/bin/env python3
"""
Live detection viewer for testing and tuning face/eye detection.
Run this script to test webcam access and detection parameters.
"""

import cv2
import sys
import numpy as np
from src.detection import load_cascades, detect_face_and_eyes, draw_detections
import config


def determine_state(detections):
    """
    Simple rule-based state determination for testing.
    
    Args:
        detections: List of (face_box, eye_boxes) tuples
    
    Returns:
        String state: Absent, Drowsy, or Focused
    """
    if not detections:
        return "Absent"
    
    # Get the first face (assuming single user)
    face_box, eyes = detections[0]
    
    if len(eyes) == 0:
        # No eyes detected, likely closed or heavy-lidded
        return "Drowsy"
    elif len(eyes) >= 2:
        # Both eyes detected, likely focused
        return "Focused"
    else:
        # One eye detected, could be turning or partial occlusion
        return "Distracted"


def test_webcam():
    """Test basic webcam access."""
    print("Testing webcam access...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        print("Please check:")
        print("1. Camera is connected")
        print("2. Camera permissions are granted")
        print("3. No other application is using the camera")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"Webcam working. Frame shape: {frame.shape}")
        cap.release()
        return True
    else:
        print("Error: Cannot read from webcam")
        cap.release()
        return False


def main():
    """Run live detection viewer."""
    
    # Test webcam first
    if not test_webcam():
        sys.exit(1)
    
    print("\nLoading cascade classifiers...")
    try:
        face_cascade, eye_cascade = load_cascades()
        print("Cascades loaded successfully")
    except Exception as e:
        print(f"Error loading cascades: {e}")
        sys.exit(1)
    
    # Open webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.STANDARD_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.STANDARD_FRAME_HEIGHT)
    
    print("\nStarting live detection viewer...")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save current frame")
    print("  SPACE - Pause/Resume")
    print("\nObserve detection behaviors:")
    print("- Eyes open → 2 eye boxes detected")
    print("- Eyes closed → No eye boxes (Drowsy)")
    print("- Head turn → Face box lost (Absent)")
    print("- Look sideways → Possible Distracted state")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Run detection
            detections = detect_face_and_eyes(frame, face_cascade, eye_cascade)
            
            # Determine state
            state = determine_state(detections)
            
            # Draw visualization
            display_frame = draw_detections(frame, detections, state)
            
            # Add frame counter and FPS
            frame_count += 1
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, 60),
                config.FONT,
                config.FONT_SCALE * 0.7,
                (255, 255, 255),
                1
            )
            
            # Add detection stats
            if detections:
                face_box, eyes = detections[0]
                stats_text = f"Face: {face_box[2]}x{face_box[3]} | Eyes: {len(eyes)}"
                cv2.putText(
                    display_frame,
                    stats_text,
                    (10, 90),
                    config.FONT,
                    config.FONT_SCALE * 0.7,
                    (255, 255, 255),
                    1
                )
        else:
            # Show paused indicator
            cv2.putText(
                display_frame,
                "PAUSED",
                (10, 120),
                config.FONT,
                config.FONT_SCALE,
                (0, 0, 255),
                2
            )
        
        # Show frame
        cv2.imshow("Detection Test - Press 'q' to quit", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save frame for debugging
            filename = f"test_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("Paused")
            else:
                print("Resumed")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection test completed")


if __name__ == "__main__":
    main()