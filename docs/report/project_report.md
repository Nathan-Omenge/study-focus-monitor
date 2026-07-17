# Study-Focus Monitor: Technical Report

**DSA4050 Computer Vision - Project 1**

**Author:** Nathan Orang'o Omenge  
**Student ID:** 670637   
**Repository:** https://github.com/Nathan-Omenge/study-focus-monitor
---

## Executive Summary

This project develops a Study-Focus Monitor that helps students track their concentration during study sessions. Using a webcam and computer vision techniques, the system automatically detects whether a student is focused, drowsy, distracted, or absent from their desk. The system achieves 97.4% accuracy and runs in real-time, providing immediate feedback and session summaries to help students improve their study habits.

---

## 1. Introduction

Staying focused while studying is a common challenge for students. We often lose track of time, get distracted by our phones, or start dozing off without realizing it. This project addresses this problem by creating an automated system that monitors study sessions and provides objective feedback about focus patterns.

The Study-Focus Monitor uses traditional computer vision techniques to analyze webcam footage in real-time. It detects faces and eyes using proven methods, extracts meaningful visual features, and uses machine learning to classify the student's current state. The system is designed to be privacy-conscious, running entirely on the user's computer without sending any data to the cloud.

This report describes the complete development process, from initial concept to working prototype, demonstrating how classical computer vision techniques can solve practical problems effectively.

## 2. Problem Statement

### The Challenge

Students face several challenges during self-study:
- **Lack of awareness**: We don't realize when we're losing focus
- **No objective feedback**: Self-assessment of focus time is often inaccurate
- **Gradual decline**: Focus deteriorates slowly without us noticing
- **Missing accountability**: Studying alone lacks the structure of a classroom

### Why This Matters

Research shows that the average student can only maintain full concentration for 10-15 minutes before experiencing attention lapses. These brief distractions accumulate, significantly reducing study effectiveness. Without awareness of these patterns, students cannot take corrective action like taking breaks or changing study strategies.

### Our Solution Approach

The Study-Focus Monitor provides:
- **Real-time monitoring** using a standard webcam
- **Automatic classification** into four states: Focused, Drowsy, Distracted, or Absent
- **Session tracking** with statistics and visualizations
- **Privacy protection** through local-only processing

We specifically chose traditional computer vision over deep learning because:
1. It requires less computational power (works on any laptop)
2. The system is more interpretable (we understand exactly how it makes decisions)
3. It needs less training data (hundreds of images vs. thousands)
4. It respects the course requirement to learn fundamental techniques

## 3. Objectives

### Primary Goals

1. **Build a working focus monitoring system** that runs on standard hardware
2. **Accurately classify student states** with at least 85% accuracy
3. **Provide real-time feedback** at minimum 15 frames per second
4. **Generate useful session summaries** to help students improve

### Learning Objectives

Through this project, we aimed to:
1. Master face detection using Haar cascades
2. Understand feature extraction techniques like HOG and LBP
3. Apply classical machine learning algorithms
4. Build a complete computer vision application
5. Evaluate system performance scientifically

### Technical Requirements

- Use Haar cascade classifiers for detection (mandatory)
- No deep learning allowed (course constraint)
- Implement the complete pipeline from image capture to classification
- Provide comprehensive performance evaluation
- Create a user-friendly application

## 4. Literature Review

### Face Detection: Haar Cascades

Haar cascade classifiers, developed by Viola and Jones in 2001, revolutionized real-time face detection. The technique works by:
1. Using simple rectangular features that calculate the difference between dark and light regions
2. Organizing these features in a cascade (series) of increasingly complex stages
3. Quickly rejecting non-face regions early in the cascade
4. Only applying all stages to likely face regions

This approach is extremely fast because it uses "integral images" that allow feature calculation in constant time, regardless of size. For our project, we use pre-trained cascades from OpenCV for both face and eye detection.

### Feature Extraction Techniques

**HOG (Histogram of Oriented Gradients):** This technique captures the shape and structure of objects by:
- Calculating gradients (edge directions) in small image regions
- Creating histograms showing which directions are most common
- Combining these histograms to describe the overall appearance

HOG works well for eye detection because open eyes have strong horizontal edges, while closed eyes are smoother.

**LBP (Local Binary Patterns):** This method describes texture by:
- Comparing each pixel with its neighbors
- Creating a binary code based on which neighbors are brighter/darker
- Building histograms of these patterns

LBP helps distinguish between smooth regions (closed eyes) and textured regions (open eyes with visible details).

### Machine Learning Classifiers

**Support Vector Machines (SVM):** Finds the best boundary between different classes by maximizing the margin (gap) between them. We use the RBF kernel which can handle non-linear patterns.

**Random Forest:** Combines many decision trees, each trained on different data subsets. The final prediction is based on voting across all trees.

**K-Nearest Neighbors (KNN):** Classifies new data based on the most common class among its K nearest neighbors in the feature space.

## 5. Methodology

### Data Collection Process

We developed a structured approach for collecting training data:

1. **Video Recording Protocol**
   - Participants record 30-second videos for each state
   - Clear instructions provided for each state:
     - Focused: Look at screen with engaged expression
     - Drowsy: Close eyes, yawn, head nodding
     - Distracted: Look away, check phone, glance around
   
2. **Frame Extraction**
   - Videos processed at 5 frames per second
   - This rate captures enough variation without redundancy
   - Each 30-second video yields about 150 frames

3. **Data Organization**
   - Frames labeled with session information
   - Prevents mixing frames from same video in training/test sets
   - Ensures honest evaluation without data leakage

### Feature Engineering

I extract four types of features from each detected face:

1. **HOG Features (756 dimensions)**
   - Applied to the eye region (200×60 pixels)
   - Captures edge patterns that distinguish open/closed eyes
   - Uses 9 orientation bins and 8×8 pixel cells

2. **LBP Features (10 dimensions)**
   - Describes texture patterns in the eye region
   - Helps identify subtle differences like squinting

3. **Geometric Features (9 dimensions)**
   - Face position in frame (centered vs. off-center)
   - Face size (closer vs. farther from camera)
   - Number of eyes detected (0, 1, or 2)
   - Eye positions within face

4. **Pupil Position (2 dimensions)**
   - Estimates gaze direction
   - Helps distinguish looking at screen vs. looking away

Total: 777 features per frame

### Model Training Strategy

1. **Data Splitting**
   - 75% training, 25% testing
   - Session-aware split (keeps video frames together)
   - Prevents overfitting on similar consecutive frames

2. **Feature Scaling**
   - Standardize all features to zero mean and unit variance
   - Ensures no single feature dominates due to scale

3. **Model Comparison**
   - Train three classifiers: SVM, Random Forest, KNN
   - Use cross-validation to tune parameters
   - Select best model based on F1 score

4. **Temporal Smoothing**
   - Apply 5-frame moving average to reduce noise
   - Filter out blinks (eye closures under 400ms)
   - Require 2+ seconds of eye closure for drowsy classification

## 6. Implementation

### System Architecture

The system consists of seven main modules:

```
Webcam → Detection → Feature Extraction → Classification → 
Temporal Smoothing → Display → Session Summary
```

### Key Modules

1. **detection.py**: Finds faces and eyes in video frames
   - Uses Haar cascades for detection
   - Searches for eyes only in upper face region (optimization)
   - Returns bounding boxes for visualization

2. **features.py**: Extracts the 777-dimensional feature vector
   - Computes HOG, LBP, geometric, and pupil features
   - Handles cases where eyes aren't detected
   - Normalizes features for consistency

3. **train.py**: Trains and compares machine learning models
   - Implements session-aware data splitting
   - Performs grid search for best parameters
   - Saves trained model for deployment

4. **temporal.py**: Smooths predictions over time
   - Reduces flickering between states
   - Filters out blinks vs. drowsiness
   - Tracks focus statistics

5. **app.py**: Main application with user interface
   - Captures webcam feed
   - Runs complete processing pipeline
   - Displays results with color-coded boxes
   - Logs session data and generates reports

### Configuration Parameters

Key settings in config.py:
- Face detection sensitivity: 5 neighbors minimum
- Eye search region: Top 60% of face
- Smoothing window: 5 frames
- Blink threshold: 400 milliseconds
- Drowsy threshold: 2 seconds

## 7. Experimental Results

### Classification Performance

We compared three machine learning algorithms:

| Algorithm | Accuracy | Training Time | Prediction Speed |
|-----------|----------|---------------|------------------|
| **SVM (chosen)** | **97.4%** | 0.32 seconds | Very Fast |
| Random Forest | 96.8% | 0.45 seconds | Fast |
| KNN | 94.2% | 0.08 seconds | Moderate |

### Detailed Results for SVM

| State | Precision | Recall | F1-Score | Description |
|-------|-----------|--------|----------|-------------|
| Focused | 98.4% | 97.5% | 98.0% | Correctly identifies focused students |
| Drowsy | 98.1% | 96.8% | 97.4% | Accurately detects drowsiness |
| Distracted | 95.6% | 97.9% | 96.7% | Good at catching distractions |
| **Average** | **97.4%** | **97.4%** | **97.4%** | Excellent overall performance |

### Confusion Matrix Analysis

The system's main confusions:
- 2.1% of Distracted states mistaken for Focused (looking at notes vs. phone)
- 1.5% of Drowsy mistaken for Focused (brief eye closures)
- Minimal confusion between Drowsy and Distracted (different visual patterns)

### Processing Speed

| Component | Time per Frame | Equivalent FPS |
|-----------|---------------|----------------|
| Face Detection | 28 ms | 35 FPS |
| Feature Extraction | 3 ms | 333 FPS |
| Classification | 0.05 ms | 20,000 FPS |
| **Total** | **31 ms** | **32 FPS** |

The system comfortably exceeds real-time requirements (15 FPS minimum).

### Feature Importance

We tested which features contribute most to accuracy:

| Feature Set | Accuracy | Impact |
|------------|----------|---------|
| All Features | 97.4% | Best |
| HOG Only | 94.8% | Strong alone |
| HOG + LBP | 96.1% | Good combination |
| LBP Only | 71.2% | Weak alone |
| Geometric Only | 68.5% | Weakest |

HOG features are most important, but combining all features gives best results.

## 8. Discussion

### Strengths of Our Approach

1. **High Accuracy**: 97.4% accuracy exceeds expectations for traditional methods
2. **Real-time Performance**: 32 FPS allows smooth, responsive monitoring
3. **Interpretable Features**: We understand exactly what the system looks for
4. **Privacy-Preserving**: All processing happens locally, no data leaves the computer
5. **Practical Utility**: Provides genuine value for student self-monitoring

### Limitations and Challenges

1. **Lighting Sensitivity**: Performance degrades in very dim or bright conditions
2. **Glasses Interference**: Reflective glasses can block eye detection
3. **Single-User Training**: Model may need retraining for different users
4. **Ambiguous States**: Looking down could be reading or using phone
5. **Fixed Camera Requirement**: User must stay within camera view

### Comparison with Deep Learning

While deep learning might achieve slightly higher accuracy, our traditional approach offers:
- **Efficiency**: Runs on CPU, no GPU needed
- **Interpretability**: Can explain decisions
- **Quick Training**: Minutes vs. hours
- **Small Dataset**: Hundreds of images sufficient

### Temporal Smoothing Impact

The temporal layer significantly improves user experience:
- Reduces state flickering from 18.7% to 2.3%
- Correctly ignores 100% of blinks
- Maintains responsiveness with 5-frame window

## 9. Conclusion

This project successfully demonstrates that traditional computer vision techniques can solve complex real-world problems effectively. We achieved all primary objectives:

- Built a functional real-time monitoring system  
- Achieved 97.4% accuracy (exceeded 85% requirement)  
- Maintained 32 FPS performance (exceeded 15 FPS requirement)  
- Created useful session summaries with visualizations  

### Key Achievements

1. **Technical Success**: The system works reliably and accurately
2. **Practical Value**: Provides genuine utility for students
3. **Educational Value**: Demonstrated mastery of traditional CV techniques
4. **Complete Implementation**: Full pipeline from capture to analysis

### Learning Outcomes

Through this project, we gained hands-on experience with:
- Face detection using Haar cascades
- Feature engineering with HOG and LBP
- Classical machine learning algorithms
- Real-time system development
- Performance evaluation and optimization

The Study-Focus Monitor proves that carefully engineered traditional methods remain valuable in the era of deep learning, especially for applications requiring efficiency, interpretability, and quick deployment.

## 10. Recommendations

### For Immediate Improvement

1. **User Calibration**: Add a setup wizard where users demonstrate their focused and distracted states for personalized thresholds

2. **Multi-Face Support**: Extend to monitor multiple students simultaneously for group study sessions

3. **Break Reminders**: Add smart notifications suggesting breaks based on declining focus trends

### For Future Development

1. **Mobile App**: Port to smartphones using front-facing camera for study-on-the-go

2. **Learning Analytics**: Integrate with study apps to correlate focus with learning outcomes

3. **Adaptive Thresholds**: Automatically adjust detection sensitivity based on lighting and user characteristics

4. **Extended States**: Add more nuanced states like "partially focused" or "taking notes"

### For Research Extensions

1. **Cross-Cultural Validation**: Test with diverse populations to ensure broad applicability

2. **Long-Term Studies**: Analyze how focus patterns change over weeks/months

3. **Correlation Analysis**: Study relationship between focus patterns and academic performance

4. **Alternative Features**: Explore additional features like head pose or blink rate

## 11. References

1. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *Conference on Computer Vision and Pattern Recognition*.

2. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *Conference on Computer Vision and Pattern Recognition*.

3. Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

4. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

5. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

6. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

8. Lienhart, R., & Maydt, J. (2002). An extended set of Haar-like features for rapid object detection. *International Conference on Image Processing*.

---

## Appendix A: Installation Guide

### System Requirements
- Python 3.11 or 3.12
- 4GB RAM minimum
- Webcam (built-in or external)
- 500MB disk space

### Setup Instructions

1. **Clone Repository**
   ```bash
   git clone [your-repository-url]
   cd study-focus-monitor
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   python src/app.py
   ```

### Quick Test
Press 'S' to start monitoring, try different poses (focused, drowsy, distracted), press 'Q' to see results.

---

## Appendix B: Code Repository

**GitHub Repository**: https://github.com/Nathan-Omenge/study-focus-monitor

**Key Files**:
- `src/app.py` - Main application
- `src/detection.py` - Face/eye detection
- `src/features.py` - Feature extraction
- `src/train.py` - Model training
- `config.py` - Configuration settings

**Documentation**:
- `README.md` - Setup and usage guide
- `requirements.txt` - Python dependencies


---

