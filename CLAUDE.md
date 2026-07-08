# CLAUDE.md

Project context for Claude Code. Read this and PROJECT_PLAN.md at the start of every session.

## Project: Study-Focus Monitor (DSA4050 Computer Vision, Project 1)

A personal study-focus monitor built with traditional computer vision. A webcam watches a study session and classifies the user's state (Focused, Drowsy, Distracted, or Absent), then logs how much of the session was spent focused and writes a summary chart at the end.

## Hard constraints (never violate)

- Traditional computer vision and classical machine learning only. No deep learning: no CNNs, no YOLO, no pretrained neural networks, no transfer learning.
- Object detection MUST use Haar cascade classifiers. This is a graded course requirement, not a preference.
- Classification uses classical methods only (SVM, Random Forest, KNN, and similar) on hand-crafted features (HOG, LBP, geometric features, pupil position).

## Authoritative plan

PROJECT_PLAN.md in the repo root is the source of truth for the whole build. Follow its phases in order. Each phase has a "Done when" acceptance check and a named git checkpoint commit. Do not skip the session-aware split (the `groups` array carried from Phase 3 through Phases 4 and 5); it is load-bearing for honest evaluation, because a random frame split leaks near-duplicate frames and inflates accuracy.

## Class design

The classifier only ever sees frames where a face was detected, so it distinguishes three classes: Focused, Drowsy, Distracted. "Absent" is decided by the detector (no face found), not by the classifier. Keep this separation clean in code and in the report.

## Tech stack

Python 3.11 or 3.12. Libraries: opencv-python (base package, not contrib), scikit-image (HOG and LBP), scikit-learn, numpy, matplotlib, pandas, joblib. Local virtual environment at .venv. No GPU is used or needed.

## Repository layout

See section 3 of PROJECT_PLAN.md for the full tree. In short: source modules in src/, data in data/ (raw video and extracted frames are gitignored), trained models in models/, figures and logs in outputs/, the report in docs/report/, all paths and parameters in config.py.

## Conventions

- Keep each pipeline stage in its own module: detection, features, temporal, app.
- Put every path, parameter, and threshold in config.py. Do not hardcode these elsewhere.
- Commit at each phase checkpoint using the commit message given in the plan.
- Written prose (the report, README, docstrings, comments): do not use em dashes. Use commas, colons, parentheses, and full stops.
- No emojis Professional, efficient and easy to read code only 
- Do not add author credits in any of your commits. Show all your commit messages to the user for apporval before commiting


## Working style
Work one phase at a time. At the start of a phase, restate its goal and its "Done when" check, then proceed step by step. Explain non-obvious computer-vision choices as you go, since this is a learning project. Ask before any large refactor or before pulling in a dependency not listed above.
