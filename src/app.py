#!/usr/bin/env python3
"""
Live Study-Focus Monitor Application.
Real-time webcam monitoring with focus tracking and session logging.
"""

import cv2
import joblib
import csv
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.detection import load_cascades, detect_face_and_eyes
from src.features import extract_features_from_frame
from src.temporal import TemporalProcessor


class StudyFocusApp:
    """Real-time study focus monitoring application."""
    
    def __init__(self):
        """Initialize the application components."""
        # Load detection models
        print("Loading detection models...")
        self.face_cascade, self.eye_cascade = load_cascades()
        
        # Load classification model and scaler
        print("Loading classifier...")
        self.classifier = joblib.load(config.CLASSIFIER_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        
        # Initialize temporal processor
        self.temporal = TemporalProcessor()
        
        # Session management
        self.session_active = False
        self.session_paused = False
        self.session_data = []
        self.frame_count = 0
        
        # Display settings
        self.window_name = "Study Focus Monitor"
        self.display_width = 800
        self.display_height = 600
        
        # State colors (BGR)
        self.state_colors = {
            'Focused': (0, 255, 0),      # Green
            'Drowsy': (0, 165, 255),      # Orange
            'Distracted': (0, 0, 255),    # Red
            'Absent': (128, 128, 128),    # Gray
            'Unknown': (255, 255, 255)    # White
        }
        
        print("Application initialized successfully!")
    
    def start_camera(self):
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Cannot access webcam")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # Default if cannot read
        
        print(f"Camera started at {self.fps:.1f} FPS")
        return True
    
    def process_frame(self, frame):
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame from webcam
        
        Returns:
            Processed state and detection results
        """
        # Detect face and eyes
        detections = detect_face_and_eyes(frame, self.face_cascade, self.eye_cascade)
        
        if not detections:
            # No face detected
            state = self.temporal.process_frame(-1, fps=self.fps)
            return state, None
        
        # Get first detection
        face_box, eye_boxes = detections[0]
        
        # Extract features
        features = extract_features_from_frame(
            frame, self.face_cascade, self.eye_cascade
        )
        
        if features is None:
            state = self.temporal.process_frame(-1, fps=self.fps)
            return state, None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict state
        prediction = self.classifier.predict(features_scaled)[0]
        
        # Apply temporal smoothing
        state = self.temporal.process_frame(prediction, fps=self.fps)
        
        return state, (face_box, eye_boxes)
    
    def draw_overlay(self, frame, state, detections):
        """
        Draw visualization overlay on frame.
        
        Args:
            frame: Frame to draw on
            state: Current state string
            detections: Face and eye detection results
        """
        # Draw face box if detected
        if detections:
            face_box, eye_boxes = detections
            x, y, w, h = face_box
            color = self.state_colors.get(state, self.state_colors['Unknown'])
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw eye boxes
            for ex, ey, ew, eh in eye_boxes:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 1)
        
        # Get current stats
        stats = self.temporal.get_current_stats()
        
        # Create info panel background
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw state indicator
        state_text = f"State: {state}"
        color = self.state_colors.get(state, self.state_colors['Unknown'])
        cv2.putText(frame, state_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw session info
        if self.session_active:
            if self.session_paused:
                cv2.putText(frame, "SESSION PAUSED", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Session time
                session_time = timedelta(seconds=int(stats.get('session_time', 0)))
                time_text = f"Session: {str(session_time)}"
                cv2.putText(frame, time_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Focus percentage
                focus_pct = stats.get('focus_percentage', 0)
                focus_text = f"Focus: {focus_pct:.1f}%"
                cv2.putText(frame, focus_text, (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Current streak
                streak = timedelta(seconds=int(stats.get('current_streak', 0)))
                streak_text = f"Streak: {str(streak)}"
                cv2.putText(frame, streak_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Draw controls reminder
        controls = "Controls: [S]tart/Pause | [Q]uit | [ESC] Exit"
        cv2.putText(frame, controls, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def save_session_log(self):
        """Save detailed session log to CSV."""
        if not self.session_data:
            return
        
        # Create logs directory if needed
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = config.LOGS_DIR / f"session_{timestamp}.csv"
        
        # Write CSV
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp', 'State', 'Raw_State'])
            
            for entry in self.session_data:
                writer.writerow([
                    entry['frame'],
                    entry['timestamp'].strftime('%H:%M:%S.%f')[:-3],
                    entry['state'],
                    entry['raw_state']
                ])
        
        print(f"Session log saved to {log_file}")
        return log_file
    
    def save_session_summary(self):
        """Save session summary and generate visualization."""
        summary = self.temporal.stop_session()
        if not summary:
            return
        
        # Create outputs directories if needed
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary CSV
        summary_file = config.LOGS_DIR / f"summary_{timestamp}.csv"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Start Time', summary['start_time'].strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Duration', str(summary['duration'])])
            writer.writerow(['Total Seconds', f"{summary['total_seconds']:.1f}"])
            writer.writerow(['Focused Seconds', f"{summary['focused_seconds']:.1f}"])
            writer.writerow(['Drowsy Seconds', f"{summary['drowsy_seconds']:.1f}"])
            writer.writerow(['Distracted Seconds', f"{summary['distracted_seconds']:.1f}"])
            writer.writerow(['Absent Seconds', f"{summary['absent_seconds']:.1f}"])
            writer.writerow(['Focus Percentage', f"{summary['focus_percentage']:.1f}%"])
            writer.writerow(['Longest Focus Streak', f"{summary['longest_focus_streak']:.1f}"])
            writer.writerow(['State Transitions', summary['transitions']])
        
        print(f"Summary saved to {summary_file}")
        
        # Generate visualization
        self.generate_session_chart(summary, timestamp)
    
    def generate_session_chart(self, summary, timestamp):
        """Generate session visualization chart."""
        # Prepare data for pie chart
        times = {
            'Focused': summary['focused_seconds'],
            'Drowsy': summary['drowsy_seconds'],
            'Distracted': summary['distracted_seconds'],
            'Absent': summary['absent_seconds']
        }
        
        # Filter out zero values
        times = {k: v for k, v in times.items() if v > 0}
        
        if not times:
            print("No data to visualize")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        colors = {
            'Focused': '#4CAF50',
            'Drowsy': '#FF9800',
            'Distracted': '#F44336',
            'Absent': '#9E9E9E'
        }
        
        pie_colors = [colors.get(k, '#000000') for k in times.keys()]
        ax1.pie(
            times.values(),
            labels=times.keys(),
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax1.set_title('Session Time Distribution')
        
        # Timeline (if we have state history)
        if self.session_data:
            # Sample data for timeline (every 10th frame to reduce density)
            sampled_data = self.session_data[::10]
            
            timestamps = [entry['timestamp'] for entry in sampled_data]
            states = [entry['state'] for entry in sampled_data]
            
            # Convert to numeric for plotting
            state_map = {'Focused': 3, 'Distracted': 2, 'Drowsy': 1, 'Absent': 0}
            state_values = [state_map.get(s, -1) for s in states]
            
            # Plot timeline
            ax2.plot(timestamps, state_values, color='blue', linewidth=1)
            ax2.fill_between(timestamps, 0, state_values, alpha=0.3)
            
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(['Absent', 'Drowsy', 'Distracted', 'Focused'])
            ax2.set_xlabel('Time')
            ax2.set_ylabel('State')
            ax2.set_title('State Timeline')
            ax2.grid(True, alpha=0.3)
            
            # Rotate x labels
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add summary text
        duration_str = str(summary['duration']).split('.')[0]
        focus_pct = summary['focus_percentage']
        fig.suptitle(
            f"Study Session Summary\n"
            f"Duration: {duration_str} | Focus: {focus_pct:.1f}% | "
            f"Longest Streak: {summary['longest_focus_streak']:.0f}s",
            fontsize=12, fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save figure
        chart_file = config.FIGURES_DIR / f"session_{timestamp}.png"
        plt.savefig(chart_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Session chart saved to {chart_file}")
    
    def run(self):
        """Main application loop."""
        if not self.start_camera():
            return
        
        print("\n" + "="*60)
        print("STUDY FOCUS MONITOR - LIVE")
        print("="*60)
        print("\nControls:")
        print("  S - Start/Pause session")
        print("  Q - Quit and save session")
        print("  ESC - Exit without saving")
        print("\nPress 'S' to start monitoring...")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Process frame if session is active and not paused
            if self.session_active and not self.session_paused:
                state, detections = self.process_frame(frame)
                
                # Log frame data
                self.session_data.append({
                    'frame': self.frame_count,
                    'timestamp': datetime.now(),
                    'state': state,
                    'raw_state': state  # Could track raw vs smoothed separately
                })
                
                self.frame_count += 1
            else:
                # Just detect face for display
                detections = detect_face_and_eyes(frame, self.face_cascade, self.eye_cascade)
                if detections:
                    detections = detections[0]
                else:
                    detections = None
                state = 'Unknown'
            
            # Draw overlay
            display_frame = self.draw_overlay(frame, state, detections)
            
            # Show frame
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                # Start/pause session
                if not self.session_active:
                    self.session_active = True
                    self.session_paused = False
                    self.temporal.start_session()
                    self.session_data = []
                    self.frame_count = 0
                    print("\nSession STARTED")
                else:
                    self.session_paused = not self.session_paused
                    if self.session_paused:
                        print("\nSession PAUSED")
                    else:
                        print("\nSession RESUMED")
            
            elif key == ord('q') or key == ord('Q'):
                # Quit and save session
                if self.session_active:
                    print("\nSaving session...")
                    self.save_session_log()
                    self.save_session_summary()
                    print("Session saved successfully!")
                break
            
            elif key == 27:  # ESC
                # Exit without saving
                print("\nExiting without saving...")
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nApplication closed.")


def main():
    """Run the Study Focus Monitor application."""
    app = StudyFocusApp()
    app.run()


if __name__ == "__main__":
    main()