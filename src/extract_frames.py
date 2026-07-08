#!/usr/bin/env python3
"""
Frame extraction tool for processing recorded video clips.
Samples frames from clips, detects faces, and organizes by state.
"""

import cv2
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.detection import load_cascades, detect_face_and_eyes, get_eye_region
import config


class FrameExtractor:
    """Extract and organize frames from recorded clips."""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.stats = {
            'focused': 0,
            'drowsy': 0,
            'distracted': 0,
            'skipped_no_face': 0,
            'total_processed': 0
        }
        
        # Ensure frame directories exist
        for state_dir in [config.FOCUSED_FRAMES_DIR, 
                         config.DROWSY_FRAMES_DIR, 
                         config.DISTRACTED_FRAMES_DIR]:
            state_dir.mkdir(parents=True, exist_ok=True)
    
    def load_detectors(self):
        """Load face and eye cascade classifiers."""
        try:
            self.face_cascade, self.eye_cascade = load_cascades()
            print(" Detection cascades loaded")
            return True
        except Exception as e:
            print(f"Error loading cascades: {e}")
            return False
    
    def parse_filename(self, filename):
        """
        Parse filename to extract person, session, and state.
        Format: person_sXX_state_timestamp.mp4
        """
        parts = filename.stem.split('_')
        if len(parts) >= 4:
            person = parts[0]
            session = parts[1]  # Keep as string with 's' prefix
            state = parts[2]
            return person, session, state
        return None, None, None
    
    def extract_frames(self, video_path, skip_frames=None):
        """
        Extract frames from a video clip.
        
        Args:
            video_path: Path to video file
            skip_frames: Sample every Nth frame (default from config)
        """
        if skip_frames is None:
            skip_frames = config.FRAME_SKIP
        
        # Parse metadata from filename
        person, session, state = self.parse_filename(Path(video_path))
        
        if not person or not state:
            print(f"Warning: Cannot parse filename {video_path.name}")
            return
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path.name}")
            return
        
        print(f"\nProcessing: {video_path.name}")
        print(f"  Person: {person}, Session: {session}, State: {state}")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % skip_frames == 0:
                # Run face detection
                detections = detect_face_and_eyes(
                    frame, 
                    self.face_cascade, 
                    self.eye_cascade
                )
                
                # Only save frames with detected faces
                if detections:
                    # Get eye region from first detected face
                    face_box, eyes = detections[0]
                    eye_region = get_eye_region(frame, face_box)
                    
                    if eye_region is not None:
                        # Generate output filename preserving session info
                        output_name = f"{person}_{session}_{state}_{frame_count:06d}.jpg"
                        
                        # Determine output directory
                        if state == 'focused':
                            output_path = config.FOCUSED_FRAMES_DIR / output_name
                        elif state == 'drowsy':
                            output_path = config.DROWSY_FRAMES_DIR / output_name
                        elif state == 'distracted':
                            output_path = config.DISTRACTED_FRAMES_DIR / output_name
                        else:
                            print(f"Warning: Unknown state {state}")
                            continue
                        
                        # Save full frame (we'll extract features later)
                        cv2.imwrite(str(output_path), frame)
                        saved_count += 1
                        self.stats[state] += 1
                else:
                    self.stats['skipped_no_face'] += 1
            
            frame_count += 1
        
        cap.release()
        self.stats['total_processed'] += frame_count
        
        print(f"  Processed {frame_count} frames, saved {saved_count}")
    
    def process_all_clips(self):
        """Process all video clips in raw data directory."""
        video_files = list(config.RAW_DATA_DIR.glob("*.mp4"))
        
        if not video_files:
            print("No video files found in data/raw/")
            print("Please run collect_data.py first to record clips")
            return
        
        print(f"\nFound {len(video_files)} video clips to process")
        
        for video_path in sorted(video_files):
            self.extract_frames(video_path)
        
        self.print_summary()
    
    def balance_classes(self):
        """
        Balance the dataset by limiting each class to the minority class size.
        """
        # Count frames per class
        class_counts = {
            'focused': len(list(config.FOCUSED_FRAMES_DIR.glob("*.jpg"))),
            'drowsy': len(list(config.DROWSY_FRAMES_DIR.glob("*.jpg"))),
            'distracted': len(list(config.DISTRACTED_FRAMES_DIR.glob("*.jpg")))
        }
        
        if all(count == 0 for count in class_counts.values()):
            print("No frames to balance")
            return
        
        min_count = min(c for c in class_counts.values() if c > 0)
        
        print(f"\nBalancing classes to {min_count} frames each:")
        print(f"  Current counts: {class_counts}")
        
        # Trim excess frames from over-represented classes
        for state, state_dir in [
            ('focused', config.FOCUSED_FRAMES_DIR),
            ('drowsy', config.DROWSY_FRAMES_DIR),
            ('distracted', config.DISTRACTED_FRAMES_DIR)
        ]:
            if class_counts[state] > min_count:
                frames = sorted(state_dir.glob("*.jpg"))
                # Keep frames distributed across sessions
                step = len(frames) // min_count
                keep_frames = frames[::step][:min_count]
                keep_set = set(keep_frames)
                
                for frame_path in frames:
                    if frame_path not in keep_set:
                        frame_path.unlink()
                
                print(f"  Trimmed {state}: {class_counts[state]} -> {min_count}")
    
    def print_summary(self):
        """Print extraction summary statistics."""
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        
        print(f"Total frames processed: {self.stats['total_processed']}")
        print(f"Frames with no face (skipped): {self.stats['skipped_no_face']}")
        print()
        
        print("Frames saved by class:")
        for state in ['focused', 'drowsy', 'distracted']:
            print(f"  {state.capitalize()}: {self.stats[state]}")
        
        print()
        
        # Check actual frame counts on disk
        print("Frames on disk:")
        for state, state_dir in [
            ('focused', config.FOCUSED_FRAMES_DIR),
            ('drowsy', config.DROWSY_FRAMES_DIR),
            ('distracted', config.DISTRACTED_FRAMES_DIR)
        ]:
            count = len(list(state_dir.glob("*.jpg")))
            print(f"  {state.capitalize()}: {count}")
        
        print("\nFrames are organized in data/frames/")
        print("Session info preserved in filenames for train/test split")


def main():
    """Run frame extraction."""
    print("\n" + "="*60)
    print("FRAME EXTRACTION TOOL")
    print("="*60)
    
    extractor = FrameExtractor()
    
    # Load detection models
    if not extractor.load_detectors():
        return
    
    # Process all clips
    extractor.process_all_clips()
    
    # Optional: balance classes
    response = input("\nBalance classes to minority size? (y/n): ").strip().lower()
    if response == 'y':
        extractor.balance_classes()
        print("\nClasses balanced!")
    
    print("\nExtraction complete!")
    print("Next step: Run build_dataset.py to create feature vectors")


if __name__ == "__main__":
    main()