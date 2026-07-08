# Data Card for Study-Focus Monitor Dataset

## Dataset Description

This dataset contains webcam frames of study sessions, labeled with focus states for training a classical machine learning classifier.

## Collection Process

### Participants
(To be filled after data collection)

### Consent
All participants provided verbal/written consent for data collection and use in this educational project.

### Recording Conditions
- Various lighting conditions (daylight, lamplight)
- Different backgrounds
- Multiple camera angles and distances
- With and without glasses (where applicable)

## Class Definitions

### Focused
Eyes open, face toward the screen, small natural movements allowed.

### Drowsy
Eyes closed or clearly heavy-lidded, head may nod.

### Distracted
Gaze or head turned off-screen (glancing at a phone, looking to the side or down away from the screen).

## Dataset Statistics

### Class Distribution
(To be updated after data collection)
- Focused: X frames
- Drowsy: X frames  
- Distracted: X frames

### Session Information
(To be updated after data collection)
- Total sessions: X
- Participants: X
- Date range: YYYY-MM-DD to YYYY-MM-DD

## Train/Test Split Strategy

Session-aware splitting is used to prevent data leakage. All frames from a single recording session stay together in either the training or test set. This prevents near-duplicate frames from appearing in both sets, ensuring honest evaluation.

## Privacy and Ethics

- Data collected with explicit consent
- Used only for this educational project
- Not shared publicly without anonymization
- Faces are detected but not identified

## How to Reproduce

1. Run `src/collect_data.py` to record webcam sessions
2. Run `src/extract_frames.py` to extract and label frames from recordings
3. Ensure balanced class distribution before training