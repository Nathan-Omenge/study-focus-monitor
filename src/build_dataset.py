#!/usr/bin/env python3
"""
Dataset builder for Study-Focus Monitor.
Processes all labeled frames to create feature matrix and metadata.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.detection import load_cascades
from src.features import extract_features_from_frame, get_feature_names


class DatasetBuilder:
    """Build feature dataset from labeled frames."""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.X = []  # Feature matrix
        self.y = []  # Labels
        self.groups = []  # Session IDs for group-aware split
        self.frame_paths = []  # Original frame paths
        
    def load_detectors(self):
        """Load face and eye cascade classifiers."""
        try:
            self.face_cascade, self.eye_cascade = load_cascades()
            print(" Detection cascades loaded")
            return True
        except Exception as e:
            print(f"Error loading cascades: {e}")
            return False
    
    def process_class(self, state: str, label: int):
        """
        Process all frames for a single class.
        
        Args:
            state: Class name ('focused', 'drowsy', 'distracted')
            label: Numeric label for this class
        """
        # Get frame directory
        if state == 'focused':
            frame_dir = config.FOCUSED_FRAMES_DIR
        elif state == 'drowsy':
            frame_dir = config.DROWSY_FRAMES_DIR
        elif state == 'distracted':
            frame_dir = config.DISTRACTED_FRAMES_DIR
        else:
            raise ValueError(f"Unknown state: {state}")
        
        # Get all frames
        frame_files = sorted(frame_dir.glob("*.jpg"))
        
        if not frame_files:
            print(f"Warning: No frames found for {state}")
            return
        
        print(f"\nProcessing {len(frame_files)} {state} frames...")
        
        # Process each frame
        failed_count = 0
        for frame_path in tqdm(frame_files, desc=state):
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                failed_count += 1
                continue
            
            # Extract features
            features = extract_features_from_frame(
                frame, 
                self.face_cascade, 
                self.eye_cascade
            )
            
            if features is not None:
                # Extract session ID from filename
                # Format: person_session_state_frame.jpg
                parts = frame_path.stem.split('_')
                if len(parts) >= 2:
                    session_id = f"{parts[0]}_{parts[1]}"  # person_session
                else:
                    session_id = "unknown"
                
                # Add to dataset
                self.X.append(features)
                self.y.append(label)
                self.groups.append(session_id)
                self.frame_paths.append(str(frame_path))
            else:
                failed_count += 1
        
        if failed_count > 0:
            print(f"  Failed to extract features from {failed_count} frames")
    
    def build(self):
        """Build the complete dataset."""
        print("\n" + "="*60)
        print("BUILDING FEATURE DATASET")
        print("="*60)
        
        if not self.load_detectors():
            return None
        
        # Process each class
        # Labels: 0=Focused, 1=Drowsy, 2=Distracted
        self.process_class('focused', 0)
        self.process_class('drowsy', 1)
        self.process_class('distracted', 2)
        
        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.groups = np.array(self.groups)
        
        return self.X, self.y, self.groups
    
    def save_dataset(self, output_path: Path = None):
        """
        Save dataset to disk.
        
        Args:
            output_path: Where to save the dataset
        """
        if output_path is None:
            output_path = config.DATA_DIR / 'dataset.npz'
        
        # Save as compressed numpy archive
        np.savez_compressed(
            output_path,
            X=self.X,
            y=self.y,
            groups=self.groups,
            frame_paths=self.frame_paths,
            feature_names=get_feature_names()
        )
        
        print(f"\n Dataset saved to {output_path}")
        print(f"  Shape: {self.X.shape}")
        print(f"  Classes: {np.unique(self.y)}")
        print(f"  Sessions: {len(np.unique(self.groups))}")
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print(f"\nOverall:")
        print(f"  Total samples: {len(self.X)}")
        print(f"  Feature dimensions: {self.X.shape[1] if len(self.X) > 0 else 0}")
        print(f"  Unique sessions: {len(np.unique(self.groups))}")
        
        print(f"\nClass distribution:")
        class_names = ['Focused', 'Drowsy', 'Distracted']
        for i, name in enumerate(class_names):
            count = np.sum(self.y == i)
            percentage = count / len(self.y) * 100 if len(self.y) > 0 else 0
            print(f"  {name}: {count} ({percentage:.1f}%)")
        
        print(f"\nSession distribution:")
        for session in np.unique(self.groups):
            session_indices = self.groups == session
            session_labels = self.y[session_indices]
            print(f"  {session}: {len(session_labels)} frames", end=" ")
            print(f"(F:{np.sum(session_labels==0)}, ", end="")
            print(f"Dr:{np.sum(session_labels==1)}, ", end="")
            print(f"Di:{np.sum(session_labels==2)})")
        
        # Check for data issues
        print(f"\nData quality:")
        if len(self.X) > 0:
            n_nan = np.sum(np.isnan(self.X))
            n_inf = np.sum(np.isinf(self.X))
            print(f"  NaN values: {n_nan}")
            print(f"  Inf values: {n_inf}")
            print(f"  Min value: {np.min(self.X):.3f}")
            print(f"  Max value: {np.max(self.X):.3f}")
            print(f"  Mean value: {np.mean(self.X):.3f}")
            
            if n_nan > 0 or n_inf > 0:
                print("   Warning: Dataset contains invalid values!")
        else:
            print("  No data to analyze")


def main():
    """Build and save the dataset."""
    builder = DatasetBuilder()
    
    # Build dataset
    X, y, groups = builder.build()
    
    if X is not None and len(X) > 0:
        # Save dataset
        builder.save_dataset()
        
        # Print statistics
        builder.print_statistics()
        
        print("\n Dataset ready for training!")
        print("Next step: Run train.py to train classifiers")
    else:
        print("\nError: Failed to build dataset")
        print("Check that frames are properly extracted in data/frames/")


if __name__ == "__main__":
    main()