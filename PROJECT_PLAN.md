# Study-Focus Monitor: Full Project Execution Plan

**Course:** DSA4050 Computer Vision, Project 1
**Project:** A personal study-focus monitor using traditional computer vision (Haar cascade detection plus classical machine-learning classification). No deep learning.
**Environment:** Local development on Mac (Apple Silicon), Python virtual environment, GitHub.

This document is the working guide for the whole build. Work top to bottom. Each phase has objectives, exhaustive steps, key code for the hard parts, an acceptance check ("Done when"), and a git checkpoint. Timings are relative so you can anchor them to your real deadline.

---

## 1. What we are building (system overview)

A webcam sits on while you study. Each frame flows through this pipeline:

1. **Detect** a face with a Haar cascade (this is the mandatory detection stage).
2. If no face is found, the frame is labelled **Absent** by rule (no classifier needed).
3. If a face is found, detect the eyes inside it, extract classical features from the eye region, and a trained classifier assigns one of three states:
   - **Focused** (eyes open, gaze toward the screen)
   - **Drowsy** (eyes closed or heavy over time)
   - **Distracted** (gaze off-screen, for example glancing at a phone)
4. A **temporal layer** smooths the per-frame predictions, ignores blinks, and accumulates focused time.
5. At session end, the app writes a CSV log and a summary chart (time per state, focus timeline).

So the graded pipeline is: Haar detection, classical feature extraction, multi-class ML classification, rule-based temporal logic, and a live prototype. That layered structure is exactly what keeps this out of the "too easy" band.

### Class design (important nuance)
The classifier only ever sees frames where a face was detected, so it distinguishes three classes: Focused, Drowsy, Distracted. "Absent" is decided by the detector, not the classifier. Keep this separation clean in the code and in the report.

---

## 2. Technology stack (and why)

| Tool | Role | Why this one |
|------|------|--------------|
| Python 3.11 or 3.12 | Language | Stable wheels on Apple Silicon for all libraries below |
| OpenCV (`opencv-python`) | Detection, webcam, image ops | Ships the Haar cascades; base package is enough (no `contrib` needed here) |
| scikit-image | HOG and LBP features | Cleaner feature APIs than raw OpenCV for this |
| scikit-learn | SVM, Random Forest, KNN, metrics, scaling | The traditional-ML core the brief asks for |
| NumPy | Arrays, math | Universal dependency |
| Matplotlib | Plots (confusion matrix, session charts) | Required for evaluation figures |
| pandas | Logging, dataset bookkeeping | Easy CSV handling for logs and results tables |
| joblib | Save and load trained models | Named in the brief |

No GPU is needed. Everything runs natively and fast on your machine, and local development is the right call because the prototype needs live webcam capture and an on-screen window, neither of which works in a hosted notebook.

---

## 3. Repository structure

Set this up once in Phase 0. Everything has a home from the start.

```
study-focus-monitor/
├── README.md                 # what it is, how to run, results summary
├── requirements.txt          # pinned dependencies
├── .gitignore                # excludes raw video, models, venv, caches
├── config.py                 # all paths, parameters, thresholds in one place
├── data/
│   ├── raw/                  # raw webcam clips (gitignored, large)
│   ├── frames/               # extracted labelled frames
│   │   ├── focused/
│   │   ├── drowsy/
│   │   └── distracted/
│   └── DATA_CARD.md          # how data was collected, consent, split logic
├── models/                   # saved classifier + scaler (.joblib)
├── notebooks/
│   ├── 01_detection_sandbox.ipynb
│   ├── 02_feature_check.ipynb
│   └── 03_train_evaluate.ipynb
├── src/
│   ├── __init__.py
│   ├── detection.py          # face + eye Haar detection helpers
│   ├── features.py           # HOG, LBP, geometric and pupil features
│   ├── collect_data.py       # webcam capture + labelling tool
│   ├── extract_frames.py     # video -> labelled frames
│   ├── build_dataset.py      # frames -> feature matrix (X, y) + metadata
│   ├── train.py              # train, tune, save model
│   ├── evaluate.py           # metrics, confusion matrix, ablations, plots
│   ├── temporal.py           # smoothing, blink filter, focus accounting
│   └── app.py                # live prototype
├── outputs/
│   ├── figures/              # evaluation and session figures
│   └── logs/                 # per-session CSV logs
├── docs/
│   └── report/               # technical report drafts and final
└── tests/                    # small sanity checks (optional but nice)
```

---

## 4. Phase-by-phase plan

### Phase 0: Environment and repository setup
**Goal:** a clean, reproducible project skeleton pushed to GitHub.
**Estimated time:** 1 to 2 hours.

1. Create the project folder and initialise git:
   ```bash
   mkdir study-focus-monitor && cd study-focus-monitor
   git init
   ```
2. Create and activate a virtual environment (keeps this isolated from system Python):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python --version   # confirm 3.11.x or 3.12.x
   ```
3. Install dependencies and freeze them:
   ```bash
   pip install --upgrade pip
   pip install opencv-python scikit-image scikit-learn numpy matplotlib pandas joblib jupyter
   pip freeze > requirements.txt
   ```
4. Create `.gitignore` with at least: `.venv/`, `__pycache__/`, `*.pyc`, `data/raw/`, `data/frames/`, `models/`, `outputs/logs/`, `.ipynb_checkpoints/`, `.DS_Store`.
5. Create the folder tree from section 3 (empty folders can hold a `.gitkeep` file so git tracks them).
6. Write a minimal `README.md`: one-paragraph description, setup commands, and a "how to run" placeholder you will fill in later.
7. Create `config.py` as the single source of truth for paths, image sizes, cascade choices, and thresholds. Everything else imports from here.
8. Grant camera permission: run a two-line webcam test (open `cv2.VideoCapture(0)`, read one frame, print its shape). On first run macOS asks for camera access; if it does not prompt, enable it under System Settings, Privacy and Security, Camera, for your terminal or editor.
9. First commit, then create the GitHub repo and push:
   ```bash
   git add .
   git commit -m "Phase 0: project skeleton and environment"
   ```

**Done when:** `source .venv/bin/activate` works, the webcam test prints a frame shape, and the skeleton is on GitHub.
**Checkpoint commit:** "Phase 0: project skeleton and environment".

---

### Phase 1: Detection foundation
**Goal:** reliable, tuned face and eye detection on your live webcam.
**Estimated time:** 3 to 4 hours.

1. Locate the bundled cascades. OpenCV exposes their folder:
   ```python
   import cv2
   base = cv2.data.haarcascades
   face_cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
   eye_cascade  = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
   ```
   Also note `haarcascade_eye_tree_eyeglasses.xml` for glasses cases; test both and keep whichever works better for you.
2. Build the detection routine in `src/detection.py`. Convert to grayscale first (Haar needs single-channel), then detect faces, then search for eyes **only in the upper half of each face box**. Constraining the eye search cuts false positives from nostrils and mouth dramatically:
   ```python
   def detect(gray, face_cascade, eye_cascade):
       faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=5, minSize=(80, 80))
       results = []
       for (x, y, w, h) in faces:
           roi = gray[y:y + h, x:x + w]
           upper = roi[0:int(h * 0.6), :]          # eyes live in the upper face
           eyes = eye_cascade.detectMultiScale(upper, scaleFactor=1.1,
                                               minNeighbors=6, minSize=(20, 20))
           results.append(((x, y, w, h), eyes))
       return results
   ```
3. Write a live viewer script that draws the face box (one colour) and eye boxes (another), and overlays the eye count. Watch how detection behaves as you: look straight, turn your head, close your eyes, glance sideways, leave the frame.
4. Note two behaviours you will exploit later, and write them down for the report:
   - The eye cascade is trained on **open** eyes, so it usually fails when eyes are closed. "Face detected but no eyes detected" is therefore a strong drowsiness cue.
   - The frontal face cascade fails on large head turns, so extreme "looking away" can read as "absent". Decide during labelling how to treat borderline turns.
5. Tune `scaleFactor`, `minNeighbors`, and `minSize` for your face, distance, and typical lighting. Record the values you settle on in `config.py`.

**Done when:** the live viewer holds a stable face box at your normal study distance, eye boxes appear when your eyes are open, and detection degrades in the predictable ways above.
**Checkpoint commit:** "Phase 1: face and eye detection with tuned parameters".

---

### Phase 2: Data collection
**Goal:** a labelled frame dataset across the three classes, from multiple sessions, with no printing and no external hardware.
**Estimated time:** 3 to 5 hours (including one or two friends).

This is the phase that makes or breaks the model, so treat it seriously.

1. **Consent and ethics.** Anyone you record gives verbal or written consent. Note it in `DATA_CARD.md`. This also gives you honest, substantive material for the report's discussion of ethics and data handling.
2. **Define each state precisely** so labels are consistent:
   - **Focused:** eyes open, face toward the screen, small natural movements allowed.
   - **Drowsy:** eyes closed or clearly heavy-lidded, head may nod.
   - **Distracted:** gaze or head turned off-screen (glancing at a phone, looking to the side or down away from the screen).
   Write these definitions into `DATA_CARD.md` and follow them exactly.
3. **Capture by session, one state per clip.** Record short clips (roughly 30 to 60 seconds each) where you hold a single state. Do several sessions across different times of day, lighting, and (if possible) with and without glasses. Aim for variety over volume. A rough target: at least 3 to 5 sessions, each covering all three states, from 2 to 3 people.
4. **Vary conditions deliberately** so the model does not overfit one lighting setup: daylight and lamplight, different backgrounds, slightly different camera angles and distances.
5. Build `src/collect_data.py`: a small tool that records webcam clips to `data/raw/` with filenames encoding person, session, and state (for example `ivy_s02_focused.mp4`). Encoding the label in the filename makes the next step trivial and preserves the session identity you need for the split.
6. Build `src/extract_frames.py`: sample frames from each clip (for example every 5th frame to reduce near-duplicates), run detection, keep only frames where a face is found, crop the standardised eye-band region, and save into `data/frames/<state>/` with the session id preserved in the filename. Discard frames with no face (those are the trivial Absent case and do not train the classifier).
7. **Balance the classes.** Check counts per class and trim or record more so no class dwarfs the others. A rough floor is a few hundred usable frames per class; more is better, but variety matters more than raw count.
8. Fill in `DATA_CARD.md`: who, when, conditions, consent, counts per class, and how sessions map to the future train and test split.

**Done when:** `data/frames/` holds balanced, face-verified, session-tagged frames for all three classes, and `DATA_CARD.md` documents them.
**Checkpoint commit:** "Phase 2: data collection tools and data card" (the frames themselves stay gitignored; commit the code and the card).

---

### Phase 3: Feature extraction
**Goal:** turn each labelled frame into a numeric feature vector using only classical descriptors from the brief.
**Estimated time:** 4 to 6 hours.

Build `src/features.py`. Each detected face frame becomes one feature vector combining texture and geometry.

1. **Standardise the region.** From the face box, crop the eye-band (upper portion) and resize to a fixed size (for example 64 by 32) so every HOG and LBP vector has the same length.
2. **HOG on the eye-band** captures the shape and edge structure that separates open from closed eyes:
   ```python
   from skimage.feature import hog
   hog_vec = hog(eye_band, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), block_norm="L2-Hys")
   ```
3. **LBP histogram on the eye-band** captures fine texture (open eyes and lids differ in texture, not only shape):
   ```python
   import numpy as np
   from skimage.feature import local_binary_pattern
   lbp = local_binary_pattern(eye_band, P=8, R=1, method="uniform")
   lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)
   ```
4. **Geometric and eye-presence features** (cheap and informative):
   - number of eyes detected (0, 1, 2): a strong drowsiness and occlusion signal.
   - for each detected eye, its centre position relative to the face box (captures head turn).
5. **Pupil position for gaze** (this is the neat traditional-CV part that powers the Distracted class without any deep learning). Inside each detected eye box, the pupil is the darkest blob. Threshold it, find the largest dark contour, take its centroid, and express its horizontal position as a ratio across the eye box. Centre means looking ahead; strongly left or right means gaze off to the side:
   ```python
   def pupil_x_ratio(eye_gray):
       _, th = cv2.threshold(eye_gray, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
       cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       if not cnts:
           return 0.5                       # default to centre if nothing found
       c = max(cnts, key=cv2.contourArea)
       M = cv2.moments(c)
       if M["m00"] == 0:
           return 0.5
       cx = M["m10"] / M["m00"]
       return cx / eye_gray.shape[1]         # 0 = far left, 1 = far right
   ```
6. **Assemble the vector:** concatenate HOG, LBP histogram, eye count, eye centre offsets, and mean pupil x-ratio into one array per frame.
7. Build `src/build_dataset.py`: walk `data/frames/`, run feature extraction on every frame, and produce `X` (feature matrix), `y` (labels), and a parallel `groups` array holding each frame's session id. Save all three (for example as a compressed `.npz`). The `groups` array is what makes an honest evaluation split possible in Phase 5, so do not skip it.

**Done when:** running `build_dataset.py` produces `X`, `y`, and `groups` with matching lengths, and a quick sanity notebook shows the feature vectors are finite and reasonably scaled.
**Checkpoint commit:** "Phase 3: feature extraction and dataset builder".

---

### Phase 4: Model training
**Goal:** a trained, tuned, saved classifier, chosen by fair comparison.
**Estimated time:** 3 to 5 hours.

Build `src/train.py`.

1. **Scale features.** Fit a `StandardScaler` on the training data and save it with the model (SVM and KNN need scaling; save it so the app applies the identical transform at run time).
2. **Split by session, never by random frame.** Use scikit-learn's group-aware tools so all frames from one recording session stay together in either train or test. This prevents near-duplicate frames from leaking across the split and inflating your score:
   ```python
   from sklearn.model_selection import GroupShuffleSplit
   gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
   train_idx, test_idx = next(gss.split(X, y, groups))
   ```
3. **Train three candidates** so you can justify your choice in the report: SVM (RBF kernel), Random Forest, and KNN.
4. **Tune with group-aware cross-validation** (`GroupKFold` inside `GridSearchCV`) over a small parameter grid (for SVM, `C` and `gamma`; for RF, number and depth of trees; for KNN, number of neighbours).
5. **Select** the best model by macro-averaged F1 (macro treats all three classes equally, which matters if one class is rarer).
6. **Save** the winning model and the scaler together with `joblib` into `models/`.

**Done when:** `models/` holds a saved model and scaler, and training prints cross-validated scores for all three candidates with a clear winner.
**Checkpoint commit:** "Phase 4: training, tuning, and model selection".

---

### Phase 5: Evaluation
**Goal:** a rigorous, honest results section that satisfies every metric in the brief.
**Estimated time:** 3 to 5 hours.

Build `src/evaluate.py`, working on the held-out test sessions from Phase 4.

1. **Core metrics:** overall accuracy, and per-class precision, recall, and F1 (use `classification_report`), plus macro-F1.
2. **Confusion matrix**, saved as a figure to `outputs/figures/`. Read it for the interesting confusions (for example, is Distracted being mistaken for Focused?).
3. **False positive rate** per class, derived from the confusion matrix, since the brief asks for it explicitly.
4. **Processing time:** measure average milliseconds per frame end to end and convert to frames per second, to argue real-time feasibility.
5. **Classifier comparison table:** SVM against Random Forest against KNN on the same test split. This directly earns marks in Methodology and Results.
6. **Feature ablation:** retrain with HOG only, LBP only, both, and both plus geometry. The resulting table shows which features carry the signal and gives you genuine analysis for the discussion.
7. Save every figure and table so the report can reference them directly.

**Done when:** you have a metrics table, a confusion-matrix figure, a classifier comparison, an ablation table, and an FPS number, all reproducible from one script.
**Checkpoint commit:** "Phase 5: evaluation, ablations, and figures".

---

### Phase 6: Temporal layer
**Goal:** turn noisy per-frame predictions into stable, meaningful session behaviour.
**Estimated time:** 2 to 3 hours.

Build `src/temporal.py`. Per-frame predictions flicker; real focus is about sustained states.

1. **Smoothing:** keep a short rolling window of recent predictions and output the majority vote, so a single misclassified frame does not flip the displayed state.
2. **Blink filter:** a closed-eye stretch shorter than a set duration (for example under about 400 milliseconds, so a few frames) is a blink and is ignored. A closed-eye stretch longer than a set threshold (for example 2 to 3 seconds) counts as Drowsy. Expose both thresholds in `config.py`.
3. **Focus accounting:** advance a "focused seconds" counter only while the smoothed state is Focused. Track total session time, focused time, and time in each state.
4. **Event log:** record transitions (for example "became drowsy at 00:14:32") with timestamps for the session summary.

**Done when:** feeding a recorded clip through the temporal layer yields sensible focused-time totals and correctly ignores blinks while catching sustained closures.
**Checkpoint commit:** "Phase 6: temporal smoothing and focus accounting".

---

### Phase 7: Prototype application
**Goal:** a working, demonstrable desktop app that ties everything together.
**Estimated time:** 4 to 6 hours.

Build `src/app.py`.

1. **Live loop:** open the webcam, and for each frame run detection, feature extraction, the scaler, the classifier, and the temporal layer.
2. **On-screen overlay:** draw the face box, the current smoothed state (colour-coded), a live focus timer, and elapsed session time.
3. **Controls:** simple keys, for example `s` to start or pause a session and `q` to end and save. This is the "simple interface" the brief asks for; a desktop window with keyboard control fully qualifies.
4. **Session logging:** write a per-interval CSV to `outputs/logs/` (timestamp and state), so every session is auditable.
5. **End-of-session summary:** on quit, write a summary CSV (totals and percentages per state, longest focus streak, number of drowsy and distracted events) and render a Matplotlib figure (a timeline of state across the session plus a breakdown of time per state).
6. **Robustness:** handle the camera failing to open, and handle stretches with no face gracefully (mark Absent, keep running).
7. **Fill in the README "how to run"** section now that the command exists.

**Done when:** you can start a session, study for a few minutes, quit, and find a CSV log, a summary CSV, and a chart written to disk, with the live labels having looked sensible throughout.
**Checkpoint commit:** "Phase 7: live prototype with session logging and summary".

---

### Phase 8: Technical report
**Goal:** the written report following the brief's exact 11-section structure.
**Estimated time:** 6 to 10 hours (spread it across the build, do not leave it all to the end).

Map each required section to your work:

1. **Introduction:** the study-focus problem and your traditional-CV approach in brief.
2. **Problem Statement:** attention and fatigue during self-study, and why an on-device, privacy-respecting, non-deep-learning tool is worthwhile.
3. **Objectives:** restated from the brief's learning objectives, made specific to this system.
4. **Literature Review:** Haar cascades (Viola-Jones), HOG, LBP, and classical classifiers (SVM, RF, KNN); briefly why you avoid deep learning here (constraint plus on-device simplicity).
5. **Methodology:** the full pipeline, the feature design, and crucially the session-aware split and why it prevents leakage. This is where methodological rigour shows.
6. **Implementation:** the modules, key parameters, and how the pieces connect (a pipeline diagram helps).
7. **Experimental Results:** the metrics table, confusion matrix, classifier comparison, ablation, and FPS from Phase 5.
8. **Discussion:** strengths and honest limitations (lighting sensitivity, glasses interfering with the eye cascade, single-user generalisation, and the labelling ambiguity where looking down could be phone-distraction or reading notes). Wrestling with that last point visibly is a plus.
9. **Conclusion:** what was achieved against the objectives.
10. **Recommendations:** concrete next steps (more subjects, calibration per user, adaptive thresholds).
11. **References:** every source, properly cited.

**Writing note:** no em dashes anywhere in the report. Use commas, colons, parentheses, and full stops.

**Done when:** all 11 sections are drafted, every figure and number traces back to a script output, and references are complete.
**Checkpoint commit:** "Phase 8: technical report draft".

---

### Phase 9: Demonstration and presentation
**Goal:** a smooth demo and slides.
**Estimated time:** 2 to 4 hours.

1. **Rehearse a clean two-minute live run** in good lighting: start a session, show Focused, close your eyes to trigger Drowsy, glance away to trigger Distracted, then quit to reveal the summary chart.
2. **Record a short screen capture** as a fallback in case live lighting or camera access misbehaves on the day.
3. **Build a short slide deck:** problem, pipeline diagram, one or two headline results (accuracy and FPS), a limitation, and the live demo.
4. **Prepare for likely questions:** why no deep learning, why the session-aware split, how gaze is estimated without landmarks, and where it fails.

**Done when:** you can run the demo confidently and have a recorded backup and slides ready.
**Checkpoint commit:** "Phase 9: demo assets and slides".

---

### Phase 10: Final packaging and submission
**Goal:** a clean, reproducible, submittable repository.
**Estimated time:** 1 to 2 hours.

1. Re-freeze dependencies: `pip freeze > requirements.txt`.
2. Confirm the README lets a stranger set up and run the project from scratch (setup, data note, how to run, results summary).
3. Check the `.gitignore` keeps large and private files out, and confirm the data card explains how to obtain or regenerate the data.
4. Tidy notebooks (restart and run top to bottom so they execute cleanly).
5. Tag a release and push.

**Done when:** a fresh clone plus the documented steps reproduces the pipeline, and all four deliverables (source code, dataset or its description, report, demo) are ready.
**Checkpoint commit and tag:** "Phase 10: final submission package".

---

## 5. Suggested timeline (relative)

Adjust to your real deadline; this assumes a focused pace alongside your other commitments.

| Week | Focus | Phases |
|------|-------|--------|
| Week 1 | Foundations and data | Phase 0, 1, 2, and start Phase 3 |
| Week 2 | Modelling and app | Finish Phase 3, then 4, 5, 6, 7 |
| Week 3 | Write-up and polish | Phase 8, 9, 10 (report started earlier in parallel) |

It compresses to about two weeks if you push, or relaxes to four with lighter days. If you tell me your actual submission date, I will convert this into a dated schedule.

---

## 6. Risk register

Each risk below is something to manage and then write up honestly, not a reason to change course.

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Poor detection in low light | Medium | Add histogram equalisation in preprocessing; collect data in varied lighting |
| Glasses break eye detection | Medium | Test the eyeglasses cascade; include glasses frames in training |
| Extreme head turns read as Absent, not Distracted | Medium | Define borderline turns during labelling; note the limit in the report |
| Model overfits to you | Medium | Record 2 to 3 people; keep test sessions subject-held-out where possible |
| Random split inflates accuracy | High if ignored | Session-aware split is built into Phase 4 and 5 |
| Class imbalance skews results | Low | Balance classes in Phase 2; report macro-F1 |
| Webcam access issues on macOS | Low | Handled in Phase 0 (camera permission) |

---

## 7. Deliverables checklist (from the brief)

- [ ] Source code (the `src/` modules and notebooks, on GitHub)
- [ ] Dataset (self-collected frames, described in `DATA_CARD.md` with any public supplement noted)
- [ ] Technical report (11 sections, no em dashes)
- [ ] Demonstration or presentation (live run plus recorded backup and slides)

---

## 8. How we work through this together

I will help build each phase as you reach it: writing and debugging the modules, tuning detection, designing the feature vector, running the evaluation and reading the confusion matrix with you, and drafting the report section by section. When you start a phase, we pick up at its first step. If you want to supplement your own data with a public set later (for example for closed-eye examples), tell me and I will verify a current, accessible source rather than assume one.
