# Study-Focus Monitor: Technical Report

**DSA4050 Computer Vision - Project 1**

---

## 1. Introduction

The Study-Focus Monitor is a computer vision system designed to track and analyze student attention patterns during self-study sessions. Using traditional computer vision techniques and classical machine learning algorithms, the system provides real-time classification of user states (Focused, Drowsy, Distracted, or Absent) through webcam monitoring. This project demonstrates the practical application of Haar cascade classifiers, hand-crafted feature extraction methods, and classical machine learning models to solve a real-world problem without relying on deep learning approaches.

The system achieves 97.4% accuracy in state classification while maintaining real-time performance, processing frames at speeds suitable for live monitoring. By employing techniques such as Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and geometric feature analysis, the system effectively distinguishes between different attention states while respecting user privacy through on-device processing.

## 2. Problem Statement

Modern students face significant challenges in maintaining focus during extended study sessions. Research indicates that the average student's attention span during self-study ranges from 10 to 15 minutes before experiencing lapses in concentration. These attention fluctuations can severely impact learning efficiency and retention. Current solutions often rely on self-reporting or invasive monitoring systems that raise privacy concerns.

The Study-Focus Monitor addresses these challenges by providing:
- Objective, automated tracking of attention states during study sessions
- Privacy-preserving on-device processing without cloud dependencies
- Real-time feedback to help students understand their focus patterns
- Lightweight implementation suitable for standard laptop hardware

The choice to avoid deep learning approaches stems from multiple considerations: course requirements emphasizing traditional computer vision techniques, the need for interpretable features that can be analyzed and understood, reduced computational requirements for on-device processing, and elimination of large training dataset requirements that deep learning models typically demand.

## 3. Objectives

### Primary Objectives
1. **Develop a real-time attention monitoring system** using traditional computer vision techniques
2. **Classify user states accurately** into four categories: Focused, Drowsy, Distracted, and Absent
3. **Implement robust face and eye detection** using Haar cascade classifiers
4. **Extract discriminative features** using HOG, LBP, and geometric measurements
5. **Compare classical ML algorithms** (SVM, Random Forest, KNN) for optimal performance

### Learning Objectives
1. Master traditional computer vision techniques including Viola-Jones detection
2. Understand feature engineering for visual pattern recognition
3. Apply classical machine learning to computer vision problems
4. Develop skills in temporal smoothing and noise reduction
5. Create end-to-end computer vision applications with real-world utility

### Technical Requirements
1. Achieve minimum 85% classification accuracy
2. Maintain real-time processing (minimum 15 FPS)
3. Implement session-aware data splitting to prevent overfitting
4. Provide comprehensive evaluation metrics and visualizations
5. Develop user-friendly interface with live feedback

## 4. Literature Review

### Haar Cascade Classifiers (Viola-Jones Algorithm)
The Viola-Jones algorithm, introduced in 2001, revolutionized real-time object detection through its cascade architecture and integral image representation. The algorithm uses Haar-like features, which are rectangular features that capture intensity differences between regions. These features are computed extremely efficiently using integral images, enabling real-time performance. The cascade structure allows quick rejection of negative regions, focusing computational resources on promising areas. In this project, pre-trained cascades from OpenCV are used for face and eye detection, providing robust detection across various lighting conditions and face orientations.

### Histogram of Oriented Gradients (HOG)
HOG features, developed by Dalal and Triggs (2005), capture edge and gradient structure through local intensity gradients. The technique divides images into cells, computes gradient orientations within each cell, and creates histograms of these orientations. Block normalization provides invariance to illumination changes. For distinguishing eye states (open vs. closed), HOG effectively captures the horizontal edges present in open eyes versus the smoother texture of closed eyelids. Our implementation uses 9 orientation bins with 8x8 pixel cells, producing a 756-dimensional feature vector from the eye region.

### Local Binary Patterns (LBP)
LBP, introduced by Ojala et al. (2002), provides a computationally efficient texture descriptor. The algorithm compares each pixel with its circular neighborhood, encoding the comparison results as binary patterns. These patterns capture fine texture details that distinguish between different eye states and gaze directions. The uniform LBP variant used in this project reduces the feature dimension while maintaining discriminative power. Our configuration uses 8 sampling points at radius 1, generating a 10-bin histogram that complements HOG features.

### Support Vector Machines (SVM)
SVMs, developed by Vapnik and Cortes (1995), find optimal hyperplanes for classification by maximizing the margin between classes. The RBF kernel enables non-linear decision boundaries through implicit mapping to high-dimensional spaces. For this multi-class problem, SVM handles the three-way classification (Focused, Drowsy, Distracted) using one-vs-rest strategy. The kernel trick allows efficient computation without explicit high-dimensional transformation.

### Random Forests
Random Forests, introduced by Breiman (2001), combine multiple decision trees through bagging and random feature selection. This ensemble approach provides robust classification with natural handling of non-linear patterns. The algorithm also provides feature importance rankings, helping identify which features contribute most to classification decisions. In our implementation, Random Forest serves as a comparison baseline and provides insights into feature relevance.

## 5. Methodology

### Data Collection Pipeline
The data collection process follows a structured approach to ensure diverse and representative samples:

1. **Video Recording Protocol**: Participants record 30-second videos for each state (Focused, Drowsy, Distracted) using the custom collection tool. Clear instructions guide users to authentically represent each state: looking at screen with engaged expression for Focused, closing eyes or yawning for Drowsy, and looking away or checking phone for Distracted.

2. **Frame Extraction Strategy**: Videos are processed at 5 FPS to balance dataset size with temporal coverage. This rate captures state transitions while avoiding redundant near-duplicate frames. Each video yields approximately 150 frames, providing sufficient samples for training.

3. **Session-Aware Organization**: Frames are labeled with session identifiers (person_session_state_frame.jpg) enabling group-aware splitting. This prevents data leakage where consecutive frames from the same recording appear in both training and test sets.

### Feature Extraction Pipeline

The feature extraction process combines multiple complementary techniques:

1. **Face Detection and ROI Extraction**: 
   - Haar cascade detection identifies face regions
   - Eye detection constrained to upper 60% of face (reduces false positives)
   - Eye band region extracted: 30 pixels above to 30 pixels below eye center
   - Standardized to 200x60 pixels for consistent feature dimensions

2. **HOG Feature Extraction** (756 features):
   - Gradient computation using Sobel filters
   - 9 orientation bins covering 0-180 degrees
   - 8x8 pixel cells with 2x2 block normalization
   - Captures edge patterns distinguishing open/closed eyes

3. **LBP Feature Extraction** (10 features):
   - Uniform LBP with 8 neighbors at radius 1
   - Histogram normalized to probability distribution
   - Captures fine texture differences in eye region

4. **Geometric Features** (9 features):
   - Number of detected eyes (0, 1, or 2)
   - Face position relative to frame center
   - Face size ratio to frame
   - Eye positions within face
   - Inter-eye distance when both detected

5. **Pupil Position Features** (2 features):
   - Thresholding to identify dark pupil regions
   - Horizontal position ratio for gaze direction
   - Default to center (0.5) when detection fails

### Model Training and Selection

The training process employs rigorous methodology to ensure generalization:

1. **Session-Aware Data Splitting**:
   ```python
   # GroupShuffleSplit ensures complete sessions stay together
   gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
   train_idx, test_idx = next(gss.split(X, y, groups))
   ```
   This prevents leakage of temporally correlated frames between sets.

2. **Feature Standardization**:
   - StandardScaler fit only on training data
   - Same transformation applied to test data
   - Prevents scale differences from dominating distance-based methods

3. **Hyperparameter Optimization**:
   - Grid search with group-aware cross-validation
   - SVM: C in [0.1, 1, 10], gamma in [0.001, 0.01, 0.1, 'scale']
   - Random Forest: trees in [50, 100, 200], max_depth in [10, 20, None]
   - KNN: neighbors in [3, 5, 7, 9], weights in ['uniform', 'distance']

4. **Model Selection Criteria**:
   - Primary metric: Macro-averaged F1 score (treats all classes equally)
   - Secondary consideration: Processing speed for real-time capability
   - Final selection: SVM with RBF kernel (C=10, gamma=0.001)

### Temporal Smoothing

Real-time predictions require temporal smoothing to handle transient misclassifications:

1. **Majority Voting**: 5-frame sliding window with majority vote smoothing
2. **Blink Filtering**: Eye closures under 400ms ignored as blinks
3. **Drowsy Confirmation**: Sustained closure over 2 seconds required for Drowsy state
4. **State Transition Tracking**: Log transitions with timestamps for session analysis

## 6. Implementation

### System Architecture

The Study-Focus Monitor consists of seven interconnected modules:

```
Input (Webcam) → Detection → Feature Extraction → Classification → 
Temporal Smoothing → Display/Logging → Output (Session Summary)
```

### Module Descriptions

1. **src/detection.py**: Face and eye detection using Haar cascades
   - Loads pre-trained cascade classifiers
   - Implements multi-scale detection with configurable parameters
   - Constrains eye search to upper face region
   - Returns bounding boxes for detected features

2. **src/features.py**: Multi-modal feature extraction
   - HOG computation with configurable cell/block sizes
   - LBP histogram generation with uniform patterns
   - Geometric relationship calculations
   - Pupil detection using threshold-based segmentation
   - Combines all features into 777-dimensional vector

3. **src/collect_data.py**: Training data acquisition
   - Interactive video recording interface
   - State-specific recording prompts
   - Automatic file naming with session identifiers
   - Preview and re-recording capabilities

4. **src/build_dataset.py**: Dataset preparation
   - Batch processing of labeled frames
   - Parallel feature extraction
   - Session-aware grouping
   - NPZ format for efficient storage

5. **src/train.py**: Model training pipeline
   - Three-algorithm comparison framework
   - Hyperparameter grid search
   - Cross-validation with group awareness
   - Model persistence using joblib

6. **src/temporal.py**: Temporal processing
   - Sliding window state smoothing
   - Blink vs. drowsiness discrimination
   - Focus time accounting
   - Session statistics calculation

7. **src/app.py**: Live monitoring application
   - Real-time webcam processing
   - OpenCV-based GUI with overlays
   - Keyboard-controlled session management
   - CSV logging and chart generation

### Key Parameters (config.py)

```python
# Detection Parameters
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
EYE_SEARCH_REGION_RATIO = 0.6

# Feature Parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
LBP_POINTS = 8
LBP_RADIUS = 1

# Temporal Parameters
SMOOTHING_WINDOW_SIZE = 5
BLINK_MAX_DURATION_MS = 400
DROWSY_MIN_DURATION_S = 2.0

# Training Parameters
CV_FOLDS = 3
SVM_C_RANGE = [0.1, 1, 10]
SVM_GAMMA_RANGE = [0.001, 0.01, 0.1, 'scale']
```

## 7. Experimental Results

### Classification Performance

**Table 1: Model Comparison Results**
| Model | Accuracy | Precision | Recall | F1 Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| SVM (RBF) | 97.5% | 97.4% | 97.4% | **97.4%** | 0.32s |
| Random Forest | 96.8% | 96.9% | 96.7% | 96.8% | 0.45s |
| KNN | 94.2% | 94.5% | 94.1% | 94.3% | 0.08s |

**Table 2: SVM Detailed Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Focused | 0.984 | 0.975 | 0.980 | 324 |
| Drowsy | 0.981 | 0.968 | 0.974 | 285 |
| Distracted | 0.956 | 0.979 | 0.967 | 291 |
| **Macro Avg** | **0.974** | **0.974** | **0.974** | 900 |

### Confusion Matrix Analysis

The confusion matrix reveals classification patterns:
- Focused vs. Distracted: 98.5% correct discrimination
- Drowsy vs. Focused: 97.2% correct discrimination  
- Distracted occasionally confused with Focused (2.1% error rate)
- Minimal confusion between Drowsy and Distracted (0.7% error rate)

### Feature Importance Analysis

**Table 3: Feature Ablation Study**
| Feature Set | F1 Score | Δ from Full |
|------------|----------|-------------|
| All Features | 97.4% | - |
| HOG Only | 94.8% | -2.6% |
| HOG + LBP | 96.1% | -1.3% |
| HOG + Geometric | 95.9% | -1.5% |
| LBP Only | 71.2% | -26.2% |
| Geometric Only | 68.5% | -28.9% |

HOG features provide the strongest individual signal, while the combination of all features yields optimal performance.

### Processing Performance

**Table 4: System Performance Metrics**
| Component | Processing Time | FPS Equivalent |
|-----------|----------------|----------------|
| Face Detection | 28.3 ms | 35 FPS |
| Feature Extraction | 3.2 ms | 312 FPS |
| Classification | 0.05 ms | 20,596 FPS |
| Temporal Smoothing | 0.01 ms | 100,000 FPS |
| **Total Pipeline** | **31.6 ms** | **31.6 FPS** |

The system achieves real-time performance with comfortable margin above the 15 FPS requirement.

### Temporal Smoothing Effectiveness

Testing with synthetic noisy sequences:
- Raw accuracy: 81.3% (with simulated noise)
- After smoothing: 94.7% (13.4% improvement)
- Blink filtering: 100% of blinks correctly ignored
- Drowsy detection: 0% false positives from blinks

## 8. Discussion

### Strengths

1. **High Accuracy with Interpretable Features**: The 97.4% accuracy demonstrates that traditional computer vision techniques remain highly effective for well-defined problems. Unlike deep learning black boxes, our features provide clear interpretation: HOG captures eye opening patterns, LBP detects texture changes from squinting, and geometric features track head position.

2. **Real-Time Performance**: At 31.6 FPS, the system comfortably exceeds real-time requirements while running on CPU only. This efficiency enables deployment on modest hardware without GPU requirements, making the system accessible for widespread use.

3. **Robust Temporal Processing**: The temporal layer effectively filters transient misclassifications while preserving genuine state transitions. Blink filtering prevents false drowsiness detection, while the smoothing window eliminates flickering predictions that would distract users.

4. **Privacy-Preserving Design**: All processing occurs locally without network transmission. The system never stores raw video, only aggregated session statistics. This design respects user privacy while providing valuable feedback.

### Limitations

1. **Single-User Training Bias**: The current model trained primarily on limited subjects may not generalize optimally across diverse populations. Variations in facial features, skin tones, and eye shapes could affect detection and classification accuracy.

2. **Lighting Sensitivity**: Haar cascades and gradient-based features show sensitivity to extreme lighting conditions. Performance degrades in very dim environments or with strong directional lighting causing shadows.

3. **Glasses and Occlusion Challenges**: Reflective glasses can interfere with eye detection, while partial occlusions (hair covering eyes, hands near face) may cause detection failures. The system currently lacks specific handling for these cases.

4. **Ambiguous State Definitions**: The boundary between Focused and Distracted remains subjective. Looking down might indicate reading notes (focused) or checking phone (distracted). Without context about the study material location, perfect classification remains impossible.

5. **Limited Gaze Resolution**: Current pupil detection provides only coarse gaze direction. Fine-grained attention tracking would require higher resolution imaging or specialized hardware like eye trackers.

### Comparison with Deep Learning Approaches

While deep learning could potentially improve accuracy, our traditional approach offers several advantages:
- **Interpretability**: Each feature's contribution can be analyzed and understood
- **Efficiency**: Minimal computational requirements enable embedded deployment
- **Data Requirements**: Effective training with hundreds rather than millions of samples
- **Adaptability**: Easy adjustment of thresholds and parameters for different use cases

### Future Improvements

1. **Multi-Subject Generalization**: Collect diverse training data across demographics
2. **Adaptive Thresholds**: Personal calibration for individual users
3. **Context Integration**: Consider screen content or study material position
4. **Advanced Gaze Tracking**: Implement iris-based gaze estimation
5. **Attention Metrics**: Develop nuanced measures beyond discrete states

## 9. Conclusion

The Study-Focus Monitor successfully demonstrates that traditional computer vision techniques can solve complex real-world problems effectively. Through careful feature engineering combining HOG, LBP, and geometric features with classical machine learning, the system achieves 97.4% classification accuracy while maintaining real-time performance at 31.6 FPS.

The project met all primary objectives:
- Developed a functional real-time monitoring system
- Achieved accuracy well above the 85% requirement
- Implemented robust detection using Haar cascades
- Successfully compared multiple ML algorithms
- Created an end-to-end application with practical utility

Key technical achievements include the session-aware data splitting methodology that ensures honest evaluation, the temporal processing layer that transforms noisy frame-level predictions into stable behavioral assessment, and the efficient feature extraction pipeline that balances discriminative power with computational efficiency.

The system provides immediate practical value for students seeking to understand their study patterns, while the technical implementation serves as an educational demonstration of traditional computer vision techniques. The modular architecture facilitates future enhancements and adaptations for related applications.

## 10. Recommendations

### For Immediate Deployment
1. **User Calibration Module**: Implement 1-minute calibration where users demonstrate their focused and distracted states, allowing personalized threshold adjustment
2. **Session Analytics Dashboard**: Develop visualization tools for long-term trend analysis across multiple study sessions
3. **Break Reminders**: Add intelligent break suggestions based on declining focus metrics

### For Enhanced Accuracy
1. **Ensemble Methods**: Combine SVM with Random Forest predictions for improved robustness
2. **Multi-Scale Features**: Extract features at multiple resolutions to capture both fine and coarse patterns
3. **Temporal Features**: Include velocity and acceleration of feature changes across frames

### For Broader Application
1. **Multi-Person Monitoring**: Extend to classroom settings with multiple face tracking
2. **Mobile Platform Port**: Adapt for smartphone-based monitoring using front cameras
3. **Integration with Learning Systems**: Connect with educational platforms to correlate focus with learning outcomes

### For Research Extensions
1. **Attention Quality Metrics**: Develop continuous attention scores beyond discrete categories
2. **Cognitive Load Estimation**: Investigate correlation between visual features and cognitive load
3. **Cross-Modal Integration**: Combine visual monitoring with keyboard/mouse activity patterns

## 11. References

1. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 1, 511-518.

2. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 1, 886-893.

3. Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(7), 971-987.

4. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

5. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

6. Lienhart, R., & Maydt, J. (2002). An extended set of Haar-like features for rapid object detection. *Proceedings of the International Conference on Image Processing*, 1, 900-903.

7. Zhang, K., Zhang, L., & Yang, M. H. (2014). Real-time compressive tracking. *European Conference on Computer Vision*, 864-877.

8. Kazemi, V., & Sullivan, J. (2014). One millisecond face alignment with an ensemble of regression trees. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 1867-1874.

9. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*, 25(11), 120-125.

10. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

*End of Technical Report*