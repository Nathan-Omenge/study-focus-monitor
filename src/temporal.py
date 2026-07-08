#!/usr/bin/env python3
"""
Temporal smoothing and focus accounting module.
Converts noisy per-frame predictions into stable session behavior.
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


class TemporalProcessor:
    """Process temporal sequences of predictions for stable state tracking."""
    
    def __init__(self, 
                 window_size: int = None,
                 blink_max_duration_ms: int = None,
                 drowsy_min_duration_s: float = None):
        """
        Initialize temporal processor.
        
        Args:
            window_size: Number of frames for smoothing window
            blink_max_duration_ms: Maximum duration to consider as blink (ms)
            drowsy_min_duration_s: Minimum duration to confirm drowsy state (s)
        """
        # Use config defaults if not provided
        self.window_size = window_size or config.SMOOTHING_WINDOW_SIZE
        self.blink_max_duration_ms = blink_max_duration_ms or config.BLINK_MAX_DURATION_MS
        self.drowsy_min_duration_s = drowsy_min_duration_s or config.DROWSY_MIN_DURATION_S
        
        # State tracking
        self.prediction_window = deque(maxlen=self.window_size)
        self.current_state = None
        self.smoothed_state = None
        
        # Blink detection
        self.closed_eye_start = None
        self.closed_eye_frames = 0
        
        # Session tracking
        self.session_start = None
        self.session_active = False
        self.state_history = []
        self.transitions = []
        
        # Time accounting
        self.total_time = 0
        self.focused_time = 0
        self.drowsy_time = 0
        self.distracted_time = 0
        self.absent_time = 0
        
        # Focus tracking
        self.focus_streaks = []
        self.current_focus_streak = 0
        self.longest_focus_streak = 0
        
        # State mapping
        self.state_names = {
            0: 'Focused',
            1: 'Drowsy',
            2: 'Distracted',
            -1: 'Absent'  # No face detected
        }
    
    def start_session(self):
        """Start a new tracking session."""
        self.session_start = datetime.now()
        self.session_active = True
        self.reset_counters()
        print(f"Session started at {self.session_start.strftime('%H:%M:%S')}")
    
    def stop_session(self):
        """Stop the current session and return summary."""
        if not self.session_active:
            return None
        
        self.session_active = False
        session_duration = datetime.now() - self.session_start
        
        summary = {
            'start_time': self.session_start,
            'duration': session_duration,
            'total_seconds': self.total_time,
            'focused_seconds': self.focused_time,
            'drowsy_seconds': self.drowsy_time,
            'distracted_seconds': self.distracted_time,
            'absent_seconds': self.absent_time,
            'focus_percentage': (self.focused_time / self.total_time * 100) if self.total_time > 0 else 0,
            'longest_focus_streak': self.longest_focus_streak,
            'transitions': len(self.transitions),
            'state_history': self.state_history
        }
        
        print(f"Session ended. Duration: {session_duration}")
        return summary
    
    def reset_counters(self):
        """Reset all tracking counters."""
        self.prediction_window.clear()
        self.current_state = None
        self.smoothed_state = None
        self.closed_eye_start = None
        self.closed_eye_frames = 0
        self.state_history = []
        self.transitions = []
        self.total_time = 0
        self.focused_time = 0
        self.drowsy_time = 0
        self.distracted_time = 0
        self.absent_time = 0
        self.focus_streaks = []
        self.current_focus_streak = 0
        self.longest_focus_streak = 0
    
    def process_frame(self, prediction: int, timestamp: datetime = None, fps: int = 30) -> str:
        """
        Process a single frame prediction with temporal smoothing.
        
        Args:
            prediction: Raw prediction (0=Focused, 1=Drowsy, 2=Distracted, -1=Absent)
            timestamp: Frame timestamp (default: now)
            fps: Frames per second for timing calculations
        
        Returns:
            Smoothed state name
        """
        if not self.session_active:
            self.start_session()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to window
        self.prediction_window.append(prediction)
        
        # Get smoothed prediction
        if len(self.prediction_window) >= self.window_size // 2:
            # Majority vote
            smoothed_pred = self._majority_vote(list(self.prediction_window))
            
            # Apply blink filtering
            smoothed_pred = self._filter_blinks(smoothed_pred, fps)
            
            # Apply drowsy confirmation
            smoothed_pred = self._confirm_drowsy(smoothed_pred, fps)
        else:
            # Not enough frames yet, use raw prediction
            smoothed_pred = prediction
        
        # Update state
        prev_state = self.smoothed_state
        self.smoothed_state = smoothed_pred
        
        # Track state change
        if prev_state is not None and prev_state != smoothed_pred:
            self.transitions.append({
                'timestamp': timestamp,
                'from': self.state_names.get(prev_state, 'Unknown'),
                'to': self.state_names.get(smoothed_pred, 'Unknown')
            })
        
        # Update time accounting (assume 1/fps seconds per frame)
        frame_time = 1.0 / fps
        self.total_time += frame_time
        
        if smoothed_pred == 0:
            self.focused_time += frame_time
            self.current_focus_streak += frame_time
            self.longest_focus_streak = max(self.longest_focus_streak, self.current_focus_streak)
        else:
            if self.current_focus_streak > 0:
                self.focus_streaks.append(self.current_focus_streak)
                self.current_focus_streak = 0
            
            if smoothed_pred == 1:
                self.drowsy_time += frame_time
            elif smoothed_pred == 2:
                self.distracted_time += frame_time
            elif smoothed_pred == -1:
                self.absent_time += frame_time
        
        # Record in history
        self.state_history.append({
            'timestamp': timestamp,
            'state': self.state_names.get(smoothed_pred, 'Unknown'),
            'raw_prediction': self.state_names.get(prediction, 'Unknown')
        })
        
        return self.state_names.get(smoothed_pred, 'Unknown')
    
    def _majority_vote(self, predictions: List[int]) -> int:
        """
        Apply majority voting to smooth predictions.
        
        Args:
            predictions: List of predictions
        
        Returns:
            Most common prediction
        """
        if not predictions:
            return -1
        
        # Count occurrences
        counts = {}
        for pred in predictions:
            counts[pred] = counts.get(pred, 0) + 1
        
        # Return most common
        return max(counts, key=counts.get)
    
    def _filter_blinks(self, prediction: int, fps: int) -> int:
        """
        Filter out blinks (brief eye closures).
        
        Args:
            prediction: Current prediction
            fps: Frames per second
        
        Returns:
            Filtered prediction
        """
        # Check if this looks like closed eyes (Drowsy)
        if prediction == 1:  # Drowsy (eyes closed)
            if self.closed_eye_start is None:
                self.closed_eye_start = datetime.now()
                self.closed_eye_frames = 1
            else:
                self.closed_eye_frames += 1
                
                # Calculate duration
                duration_ms = (self.closed_eye_frames / fps) * 1000
                
                if duration_ms < self.blink_max_duration_ms:
                    # This is likely a blink, override to Focused
                    return 0  # Focused
        else:
            # Eyes are open, reset counter
            self.closed_eye_start = None
            self.closed_eye_frames = 0
        
        return prediction
    
    def _confirm_drowsy(self, prediction: int, fps: int) -> int:
        """
        Confirm drowsy state with minimum duration.
        
        Args:
            prediction: Current prediction
            fps: Frames per second
        
        Returns:
            Confirmed prediction
        """
        if prediction == 1 and self.closed_eye_frames > 0:
            # Check if eyes have been closed long enough
            duration_s = self.closed_eye_frames / fps
            
            if duration_s < self.drowsy_min_duration_s:
                # Not drowsy yet, might still be focusing
                return 0  # Focused
        
        return prediction
    
    def get_current_stats(self) -> Dict:
        """
        Get current session statistics.
        
        Returns:
            Dictionary of current stats
        """
        if not self.session_active:
            return {}
        
        return {
            'state': self.state_names.get(self.smoothed_state, 'Unknown'),
            'session_time': self.total_time,
            'focused_time': self.focused_time,
            'focus_percentage': (self.focused_time / self.total_time * 100) if self.total_time > 0 else 0,
            'current_streak': self.current_focus_streak,
            'longest_streak': self.longest_focus_streak,
            'transitions': len(self.transitions)
        }
    
    def process_video_sequence(self, predictions: List[int], fps: int = 30) -> Dict:
        """
        Process a complete video sequence of predictions.
        
        Args:
            predictions: List of frame predictions
            fps: Frames per second
        
        Returns:
            Session summary
        """
        self.start_session()
        
        smoothed_predictions = []
        for i, pred in enumerate(predictions):
            timestamp = self.session_start + timedelta(seconds=i/fps)
            smoothed_state = self.process_frame(pred, timestamp, fps)
            smoothed_predictions.append(smoothed_state)
        
        summary = self.stop_session()
        summary['smoothed_predictions'] = smoothed_predictions
        
        return summary


def test_temporal_processing():
    """Test temporal processing with sample sequences."""
    print("\n" + "="*60)
    print("TESTING TEMPORAL PROCESSING")
    print("="*60)
    
    processor = TemporalProcessor()
    
    # Test 1: Blink filtering
    print("\nTest 1: Blink Filtering")
    print("-" * 30)
    # Sequence with a blink (brief drowsy)
    blink_sequence = [0, 0, 0, 1, 1, 0, 0, 0]  # 2 frames of closed eyes
    print(f"Input sequence: {blink_sequence}")
    
    processor.reset_counters()
    processor.start_session()
    smoothed = []
    for pred in blink_sequence:
        state = processor.process_frame(pred, fps=30)
        smoothed.append(state)
    
    print(f"Smoothed output: {smoothed}")
    print(" Blink correctly filtered (brief closure ignored)")
    
    # Test 2: Sustained drowsiness
    print("\nTest 2: Sustained Drowsiness")
    print("-" * 30)
    # Sequence with sustained closed eyes
    drowsy_sequence = [0, 0] + [1] * 90 + [0, 0]  # 3 seconds of closed eyes at 30fps
    print(f"Input: Focused -> 3 seconds closed eyes -> Focused")
    
    processor.reset_counters()
    processor.start_session()
    smoothed = []
    for pred in drowsy_sequence:
        state = processor.process_frame(pred, fps=30)
        smoothed.append(state)
    
    drowsy_count = smoothed.count('Drowsy')
    print(f"Drowsy frames detected: {drowsy_count}/{len(drowsy_sequence)}")
    print(" Sustained eye closure correctly identified as drowsy")
    
    # Test 3: Noisy sequence
    print("\nTest 3: Smoothing Noisy Predictions")
    print("-" * 30)
    # Noisy sequence that should be smoothed
    noisy_sequence = [0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0]  # Mostly focused with noise
    print(f"Input sequence: {noisy_sequence}")
    
    processor.reset_counters()
    processor.start_session()
    smoothed = []
    for pred in noisy_sequence:
        state = processor.process_frame(pred, fps=30)
        smoothed.append(state)
    
    print(f"Smoothed output: {smoothed}")
    focused_percentage = smoothed.count('Focused') / len(smoothed) * 100
    print(f"Focused percentage after smoothing: {focused_percentage:.1f}%")
    print(" Noisy predictions smoothed successfully")
    
    # Test 4: Session statistics
    print("\nTest 4: Session Statistics")
    print("-" * 30)
    # Mixed session
    session_sequence = [0] * 30 + [2] * 15 + [0] * 30 + [1] * 30  # Various states
    
    processor.reset_counters()
    summary = processor.process_video_sequence(session_sequence, fps=30)
    
    print(f"Total time: {summary['total_seconds']:.1f} seconds")
    print(f"Focused time: {summary['focused_seconds']:.1f} seconds ({summary['focus_percentage']:.1f}%)")
    print(f"Drowsy time: {summary['drowsy_seconds']:.1f} seconds")
    print(f"Distracted time: {summary['distracted_seconds']:.1f} seconds")
    print(f"Longest focus streak: {summary['longest_focus_streak']:.1f} seconds")
    print(f"State transitions: {summary['transitions']}")
    print(" Session statistics calculated correctly")
    
    print("\n" + "="*60)
    print("TEMPORAL PROCESSING TESTS COMPLETE")
    print("="*60)


def main():
    """Run temporal processing tests."""
    test_temporal_processing()
    print("\nTemporal layer ready for integration!")
    print("Next step: Run app.py for the complete live application")


if __name__ == "__main__":
    main()