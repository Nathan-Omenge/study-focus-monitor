#!/usr/bin/env python3
"""
Model training script for Study-Focus Monitor.
Trains and compares SVM, Random Forest, and KNN classifiers.
Uses session-aware splitting to prevent data leakage.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class ModelTrainer:
    """Train and compare multiple classifiers."""
    
    def __init__(self, dataset_path: Path = None):
        """
        Initialize trainer with dataset.
        
        Args:
            dataset_path: Path to dataset.npz file
        """
        if dataset_path is None:
            dataset_path = config.DATA_DIR / 'dataset.npz'
        
        # Load dataset
        data = np.load(dataset_path)
        self.X = data['X']
        self.y = data['y']
        self.groups = data['groups']
        
        # Initialize placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.groups_train = None
        self.groups_test = None
        self.scaler = None
        self.best_model = None
        self.best_score = 0
        self.best_name = None
        
        print(f" Loaded dataset: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"  Classes: {np.unique(self.y)}")
        print(f"  Sessions: {np.unique(self.groups)}")
    
    def split_data(self, test_size=0.25, random_state=42):
        """
        Perform session-aware train/test split.
        
        This ensures all frames from a recording session stay together,
        preventing near-duplicate frames from leaking across the split.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        # Check if we have multiple sessions
        unique_groups = np.unique(self.groups)
        
        if len(unique_groups) == 1:
            print("\n Warning: Only one session found. Using random split instead.")
            print("  For better evaluation, collect data from multiple sessions.")
            
            # Fall back to random split
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
            self.groups_train = self.groups[:len(self.X_train)]
            self.groups_test = self.groups[:len(self.X_test)]
        else:
            # Use group-aware split
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(self.X, self.y, self.groups))
            
            self.X_train = self.X[train_idx]
            self.X_test = self.X[test_idx]
            self.y_train = self.y[train_idx]
            self.y_test = self.y[test_idx]
            self.groups_train = self.groups[train_idx]
            self.groups_test = self.groups[test_idx]
        
        print(f"\nTrain/test split:")
        print(f"  Training: {len(self.X_train)} samples")
        print(f"  Testing: {len(self.X_test)} samples")
        
        # Check class balance
        for split_name, y_split in [("Train", self.y_train), ("Test", self.y_test)]:
            unique, counts = np.unique(y_split, return_counts=True)
            print(f"  {split_name} distribution: ", end="")
            for cls, cnt in zip(unique, counts):
                print(f"Class {cls}: {cnt} ", end="")
            print()
    
    def scale_features(self):
        """
        Scale features using StandardScaler.
        Fit on training data only to prevent leakage.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("\n Features scaled")
    
    def train_svm(self):
        """Train SVM with RBF kernel."""
        print("\n" + "="*60)
        print("Training SVM (RBF kernel)")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'C': config.SVM_C_RANGE,
            'gamma': config.SVM_GAMMA_RANGE
        }
        
        # Setup cross-validation
        if len(np.unique(self.groups_train)) > 1:
            cv = GroupKFold(n_splits=min(config.CV_FOLDS, len(np.unique(self.groups_train))))
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
        
        # Grid search
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        
        # Handle groups for CV
        if len(np.unique(self.groups_train)) > 1:
            grid_search.fit(self.X_train, self.y_train, groups=self.groups_train)
        else:
            grid_search.fit(self.X_train, self.y_train)
        
        # Evaluate
        best_svm = grid_search.best_estimator_
        train_score = best_svm.score(self.X_train, self.y_train)
        test_score = best_svm.score(self.X_test, self.y_test)
        test_f1 = f1_score(self.y_test, best_svm.predict(self.X_test), average='macro')
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV score: {grid_search.best_score_:.3f}")
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        print(f"Test macro F1: {test_f1:.3f}")
        
        if test_f1 > self.best_score:
            self.best_model = best_svm
            self.best_score = test_f1
            self.best_name = 'SVM'
        
        return best_svm, test_f1
    
    def train_random_forest(self):
        """Train Random Forest classifier."""
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': config.RF_N_ESTIMATORS_RANGE,
            'max_depth': config.RF_MAX_DEPTH_RANGE,
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Setup cross-validation
        if len(np.unique(self.groups_train)) > 1:
            cv = GroupKFold(n_splits=min(config.CV_FOLDS, len(np.unique(self.groups_train))))
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
        
        # Grid search
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        
        # Handle groups for CV
        if len(np.unique(self.groups_train)) > 1:
            grid_search.fit(self.X_train, self.y_train, groups=self.groups_train)
        else:
            grid_search.fit(self.X_train, self.y_train)
        
        # Evaluate
        best_rf = grid_search.best_estimator_
        train_score = best_rf.score(self.X_train, self.y_train)
        test_score = best_rf.score(self.X_test, self.y_test)
        test_f1 = f1_score(self.y_test, best_rf.predict(self.X_test), average='macro')
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV score: {grid_search.best_score_:.3f}")
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        print(f"Test macro F1: {test_f1:.3f}")
        
        # Feature importance
        importances = best_rf.feature_importances_
        print(f"\nTop 5 feature importances:")
        top_indices = np.argsort(importances)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {importances[idx]:.3f}")
        
        if test_f1 > self.best_score:
            self.best_model = best_rf
            self.best_score = test_f1
            self.best_name = 'Random Forest'
        
        return best_rf, test_f1
    
    def train_knn(self):
        """Train K-Nearest Neighbors classifier."""
        print("\n" + "="*60)
        print("Training KNN")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': config.KNN_N_NEIGHBORS_RANGE,
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        # Setup cross-validation
        if len(np.unique(self.groups_train)) > 1:
            cv = GroupKFold(n_splits=min(config.CV_FOLDS, len(np.unique(self.groups_train))))
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
        
        # Grid search
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1
        )
        
        # Handle groups for CV
        if len(np.unique(self.groups_train)) > 1:
            grid_search.fit(self.X_train, self.y_train, groups=self.groups_train)
        else:
            grid_search.fit(self.X_train, self.y_train)
        
        # Evaluate
        best_knn = grid_search.best_estimator_
        train_score = best_knn.score(self.X_train, self.y_train)
        test_score = best_knn.score(self.X_test, self.y_test)
        test_f1 = f1_score(self.y_test, best_knn.predict(self.X_test), average='macro')
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"CV score: {grid_search.best_score_:.3f}")
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        print(f"Test macro F1: {test_f1:.3f}")
        
        if test_f1 > self.best_score:
            self.best_model = best_knn
            self.best_score = test_f1
            self.best_name = 'KNN'
        
        return best_knn, test_f1
    
    def compare_models(self):
        """Train all models and compare results."""
        results = {}
        
        # Train each model
        svm_model, svm_f1 = self.train_svm()
        results['SVM'] = svm_f1
        
        rf_model, rf_f1 = self.train_random_forest()
        results['Random Forest'] = rf_f1
        
        knn_model, knn_f1 = self.train_knn()
        results['KNN'] = knn_f1
        
        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print("\nTest Macro F1 Scores:")
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            marker = " BEST" if name == self.best_name else ""
            print(f"  {name}: {score:.3f} {marker}")
        
        # Detailed classification report for best model
        print(f"\n{self.best_name} Classification Report:")
        y_pred = self.best_model.predict(self.X_test)
        print(classification_report(
            self.y_test, y_pred,
            target_names=['Focused', 'Drowsy', 'Distracted'],
            digits=3
        ))
        
        return results
    
    def save_best_model(self):
        """Save the best model and scaler to disk."""
        # Ensure models directory exists
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = config.CLASSIFIER_PATH
        joblib.dump(self.best_model, model_path)
        print(f"\n Best model ({self.best_name}) saved to {model_path}")
        
        # Save scaler
        scaler_path = config.SCALER_PATH
        joblib.dump(self.scaler, scaler_path)
        print(f" Scaler saved to {scaler_path}")
        
        # Save metadata
        metadata = {
            'model_type': self.best_name,
            'test_f1': self.best_score,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'n_features': self.X.shape[1],
            'classes': ['Focused', 'Drowsy', 'Distracted']
        }
        
        metadata_path = config.MODELS_DIR / 'metadata.txt'
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f" Metadata saved to {metadata_path}")


def main():
    """Run model training pipeline."""
    print("\n" + "="*60)
    print("STUDY-FOCUS MONITOR: MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Split data
    trainer.split_data()
    
    # Scale features
    trainer.scale_features()
    
    # Train and compare models
    results = trainer.compare_models()
    
    # Save best model
    trainer.save_best_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: {trainer.best_name}")
    print(f"Test macro F1: {trainer.best_score:.3f}")
    print("\nNext step: Run evaluate.py for detailed evaluation")


if __name__ == "__main__":
    main()