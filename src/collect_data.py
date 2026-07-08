#!/usr/bin/env python3
"""
Data collection tool for recording labeled webcam clips.
Records short clips where the user maintains a single state throughout.
"""

import cv2
from datetime import datetime
import config


class DataCollector:
    """Webcam data collection interface."""
    
    def __init__(self):
        self.cap = None
        self.writer = None
        self.recording = False
        self.current_state = None
        self.current_person = None
        self.current_session = None
        self.fps = config.FPS
        
        # Ensure data directory exists
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    def start_camera(self):
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.STANDARD_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.STANDARD_FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            return False
        return True
    
    def generate_filename(self, person, session, state):
        """Generate filename encoding person, session, and state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person}_s{session:02d}_{state}_{timestamp}.mp4"
        return config.RAW_DATA_DIR / filename
    
    def start_recording(self, person, session, state):
        """Start recording a new clip."""
        if self.recording:
            self.stop_recording()
        
        self.current_person = person
        self.current_session = session
        self.current_state = state
        
        # Get output path
        output_path = self.generate_filename(person, session, state)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (config.STANDARD_FRAME_WIDTH, config.STANDARD_FRAME_HEIGHT)
        )
        
        if self.writer.isOpened():
            self.recording = True
            print(f"\n=4 Recording: {output_path.name}")
            print(f"   State: {state}")
            print(f"   Press SPACE to stop recording")
            return True
        else:
            print(f"Error: Failed to start recording")
            return False
    
    def stop_recording(self):
        """Stop current recording."""
        if self.recording and self.writer:
            self.writer.release()
            self.writer = None
            self.recording = False
            print("� Recording stopped")
    
    def draw_interface(self, frame):
        """Draw recording interface on frame."""
        display = frame.copy()
        
        # Recording indicator
        if self.recording:
            cv2.circle(display, (30, 30), 15, (0, 0, 255), -1)  # Red dot
            cv2.putText(
                display,
                f"Recording: {self.current_state}",
                (60, 35),
                config.FONT,
                config.FONT_SCALE,
                (0, 0, 255),
                config.FONT_THICKNESS
            )
            
            # Add timer (approximate)
            elapsed = getattr(self, 'frame_count', 0) / self.fps
            cv2.putText(
                display,
                f"Time: {elapsed:.1f}s",
                (60, 65),
                config.FONT,
                config.FONT_SCALE * 0.8,
                (255, 255, 255),
                1
            )
        else:
            cv2.putText(
                display,
                "Ready to record",
                (30, 35),
                config.FONT,
                config.FONT_SCALE,
                (0, 255, 0),
                config.FONT_THICKNESS
            )
        
        # Instructions
        instructions = [
            "Controls:",
            "1-3: Start recording (Focused/Drowsy/Distracted)",
            "SPACE: Stop recording",
            "Q: Quit",
        ]
        
        y_offset = display.shape[0] - 120
        for i, text in enumerate(instructions):
            cv2.putText(
                display,
                text,
                (10, y_offset + i * 25),
                config.FONT,
                config.FONT_SCALE * 0.6,
                (255, 255, 255),
                1
            )
        
        return display
    
    def run(self):
        """Main collection loop."""
        print("\n" + "="*60)
        print("DATA COLLECTION TOOL")
        print("="*60)
        
        # Get user info
        person = input("Enter your name (lowercase, no spaces): ").strip().lower()
        session_str = input("Enter session number (e.g., 1, 2, 3): ").strip()
        
        try:
            session = int(session_str)
        except ValueError:
            print("Invalid session number")
            return
        
        print("\nClass Definitions:")
        print("-" * 40)
        print("1. Focused: Eyes open, face toward screen")
        print("2. Drowsy: Eyes closed or heavy-lidded")
        print("3. Distracted: Looking away from screen")
        print("-" * 40)
        
        print("\nInstructions:")
        print("1. Hold each state for 30-60 seconds")
        print("2. Stay consistent within each recording")
        print("3. Vary your position slightly between recordings")
        print("4. Record multiple clips per state")
        
        if not self.start_camera():
            return
        
        print("\nCamera ready. Press 1, 2, or 3 to start recording.")
        
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Write frame if recording
                if self.recording and self.writer:
                    self.writer.write(frame)
                    self.frame_count += 1
                
                # Draw interface
                display = self.draw_interface(frame)
                
                # Show frame
                cv2.imshow("Data Collection - Press Q to quit", display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    if self.recording:
                        self.stop_recording()
                    break
                elif key == ord(' '):
                    if self.recording:
                        self.stop_recording()
                        self.frame_count = 0
                elif key == ord('1'):
                    self.start_recording(person, session, 'focused')
                    self.frame_count = 0
                elif key == ord('2'):
                    self.start_recording(person, session, 'drowsy')
                    self.frame_count = 0
                elif key == ord('3'):
                    self.start_recording(person, session, 'distracted')
                    self.frame_count = 0
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # List recorded files
        print("\n" + "="*60)
        print("Session Complete!")
        print("="*60)
        
        recordings = list(config.RAW_DATA_DIR.glob("*.mp4"))
        if recordings:
            print(f"\nRecorded {len(recordings)} clips:")
            for path in sorted(recordings)[-10:]:  # Show last 10
                print(f"  - {path.name}")
        
        print("\nNext steps:")
        print("1. Review recordings for quality")
        print("2. Run extract_frames.py to process clips")
        print("3. Record additional sessions for variety")


def main():
    """Run data collection."""
    collector = DataCollector()
    collector.run()


if __name__ == "__main__":
    main()