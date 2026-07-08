# Data Card for Study-Focus Monitor Dataset

## Dataset Description

This dataset contains webcam frames of study sessions, labeled with focus states for training a classical machine learning classifier.

## Collection Process

### Participants
(To be filled after data collection)
- Number of participants: TBD
- Demographics: University students (age 18-25)
- Consent type: Verbal consent for educational project use

### Consent
All participants provided verbal/written consent for data collection and use in this educational project. No personally identifiable information is stored beyond session identifiers.

### Recording Conditions
- Various lighting conditions (daylight, lamplight)
- Different backgrounds (office, bedroom, library)
- Multiple camera angles and distances (laptop webcam distance)
- With and without glasses (where applicable)
- Sessions recorded at different times of day

## Class Definitions

### Focused
- Eyes open and clearly visible
- Face directed toward the screen
- Small natural movements allowed
- Alert posture
- Typical duration: 30-60 seconds per clip

### Drowsy
- Eyes closed or clearly heavy-lidded
- Head may nod or droop
- Slower movements
- May include yawning
- Typical duration: 30-60 seconds per clip

### Distracted
- Gaze or head turned off-screen
- Looking at phone, to the side, or down
- Face may be partially visible
- Clear attention away from screen
- Typical duration: 30-60 seconds per clip

## Collection Protocol

1. **Setup**: Participant sits at normal study distance from webcam
2. **Calibration**: Test recording to ensure face detection works
3. **Recording**: 
   - Multiple short clips per state (30-60 seconds each)
   - Single state maintained throughout each clip
   - Brief breaks between recordings
4. **Variation**: Different sessions at different times/conditions
5. **Quality check**: Review clips for consistent labeling

## Dataset Statistics

### Class Distribution
(To be updated after data collection)
- Focused: X frames from Y clips
- Drowsy: X frames from Y clips  
- Distracted: X frames from Y clips
- Total frames: X (after face detection filtering)

### Session Information
(To be updated after data collection)
- Total sessions: X
- Participants: X
- Clips per session: X-Y
- Date range: YYYY-MM-DD to YYYY-MM-DD
- Frame extraction rate: Every 5th frame (6 FPS effective)

## Data Quality Measures

- **Face detection requirement**: Only frames with detected faces are kept
- **Eye region extraction**: Standardized eye-band region for all frames
- **Session tagging**: Every frame tagged with person_session_state
- **Balanced classes**: Option to balance to minority class size

## Train/Test Split Strategy

**Session-aware splitting** is used to prevent data leakage:
- All frames from a single recording session stay together
- Prevents near-duplicate consecutive frames from appearing in both sets
- Uses `GroupShuffleSplit` with session IDs as groups
- Typical split: 75% train, 25% test
- Ensures honest evaluation of generalization

## Privacy and Ethics

- **Consent**: Explicit verbal/written consent obtained
- **Use limitation**: Educational project only
- **Storage**: Local only, not uploaded to cloud
- **Identification**: No face recognition or identification performed
- **Deletion**: Data deleted after project completion
- **Access**: Limited to project participants

## File Naming Convention

Videos: `{person}_s{session:02d}_{state}_{timestamp}.mp4`
Frames: `{person}_s{session:02d}_{state}_{frame:06d}.jpg`

Where:
- person: Participant identifier (anonymized)
- session: Recording session number
- state: focused/drowsy/distracted
- timestamp/frame: Temporal identifier

## How to Reproduce

1. **Collect data**: 
   ```bash
   python src/collect_data.py
   ```
   Record 3-5 clips per state, varying conditions

2. **Extract frames**:
   ```bash
   python src/extract_frames.py
   ```
   Samples frames, detects faces, organizes by state

3. **Verify balance**:
   ```bash
   jupyter notebook notebooks/02_data_collection.ipynb
   ```
   Check class distribution and session coverage

## Known Limitations

- Limited to frontal face poses (Haar cascade constraint)
- Indoor lighting conditions only
- Single-person scenarios
- Laptop webcam quality and angle
- Potential ambiguity in borderline states (looking down: reading or distracted?)