#!/usr/bin/env python3
"""
Comprehensive evaluation script for Study-Focus Monitor.
Generates all metrics, figures, and analysis required by the brief.
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self):
        """Initialize evaluator with trained model and dataset."""
        # Load model and scaler
        self.model = joblib.load(config.CLASSIFIER_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        
        # Load dataset
        data = np.load(config.DATA_DIR / 'dataset.npz')
        self.X = data['X']
        self.y = data['y']
        self.groups = data['groups']
        self.feature_names = data['feature_names']
        
        # Class names
        self.class_names = ['Focused', 'Drowsy', 'Distracted']
        
        # Split data (same as training)
        self.split_data()
        
        print(f" Loaded model and dataset")
        print(f"  Model type: {type(self.model).__name__}")
        print(f"  Test samples: {len(self.X_test)}")
    
    def split_data(self, test_size=0.25, random_state=42):
        """Split data using same method as training."""
        # Use stratified split (matching training)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def evaluate_core_metrics(self):
        """Calculate and display core evaluation metrics."""
        print("\n" + "="*60)
        print("CORE METRICS")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Overall accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.3f}")
        
        # Per-class metrics
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1 = f1_score(self.y_test, y_pred, average=None)
        
        # Macro averages
        macro_precision = precision_score(self.y_test, y_pred, average='macro')
        macro_recall = recall_score(self.y_test, y_pred, average='macro')
        macro_f1 = f1_score(self.y_test, y_pred, average='macro')
        
        print(f"\nMacro Averages:")
        print(f"  Precision: {macro_precision:.3f}")
        print(f"  Recall: {macro_recall:.3f}")
        print(f"  F1-Score: {macro_f1:.3f}")
        
        # Per-class table
        print("\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 50)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15} {precision[i]:<12.3f} {recall[i]:<12.3f} {f1[i]:<12.3f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=self.class_names, digits=3))
        
        return y_pred
    
    def calculate_false_positive_rates(self, y_pred):
        """Calculate false positive rate for each class."""
        print("\n" + "="*60)
        print("FALSE POSITIVE RATES")
        print("="*60)
        
        cm = confusion_matrix(self.y_test, y_pred)
        fpr_per_class = []
        
        for i, class_name in enumerate(self.class_names):
            # False positives: predicted as class i but actually not
            fp = np.sum(cm[:, i]) - cm[i, i]
            # True negatives: not class i and not predicted as class i
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_per_class.append(fpr)
            
            print(f"{class_name}: {fpr:.3f} ({fp} false positives)")
        
        return fpr_per_class
    
    def plot_confusion_matrix(self, y_pred, save=True):
        """Generate and save confusion matrix figure."""
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Study Focus Monitor')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')
        
        plt.tight_layout()
        
        if save:
            output_path = config.FIGURES_DIR / 'confusion_matrix.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f" Confusion matrix saved to {output_path}")
        
        plt.show()
        
        # Print analysis
        print("\nConfusion Matrix Analysis:")
        for i, true_class in enumerate(self.class_names):
            for j, pred_class in enumerate(self.class_names):
                if i != j and cm[i, j] > 0:
                    print(f"  {cm[i, j]} {true_class} samples misclassified as {pred_class}")
    
    def measure_processing_time(self):
        """Measure average processing time per frame."""
        print("\n" + "="*60)
        print("PROCESSING TIME")
        print("="*60)
        
        # Test with 100 samples
        n_samples = min(100, len(self.X_test))
        X_sample = self.X_test[:n_samples]
        
        # Time feature scaling and prediction
        start_time = time.time()
        for _ in range(10):  # Run 10 times for better average
            X_scaled = self.scaler.transform(X_sample)
            _ = self.model.predict(X_scaled)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        time_per_iteration = total_time / 10
        time_per_frame = time_per_iteration / n_samples
        fps = 1 / time_per_frame if time_per_frame > 0 else 0
        
        print(f"Average processing time per frame: {time_per_frame*1000:.2f} ms")
        print(f"Frames per second (FPS): {fps:.1f}")
        print(f"Real-time capable: {'Yes' if fps >= 30 else 'No'}")
        
        # Note: This only measures classification time, not detection/feature extraction
        print("\nNote: This measures classification only.")
        print("Full pipeline (detection + features + classification) will be slower.")
        
        return fps
    
    def compare_classifiers(self):
        """Compare performance of different classifiers."""
        print("\n" + "="*60)
        print("CLASSIFIER COMPARISON")
        print("="*60)
        
        classifiers = {
            'SVM': SVC(C=10, gamma=0.001, kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, 
                                                   random_state=42, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=3, metric='manhattan')
        }
        
        results = []
        
        for name, clf in classifiers.items():
            # Train on same data
            clf.fit(self.X_train_scaled, self.y_train)
            
            # Predict
            y_pred = clf.predict(self.X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='macro')
            recall = recall_score(self.y_test, y_pred, average='macro')
            f1 = f1_score(self.y_test, y_pred, average='macro')
            
            results.append({
                'Classifier': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('F1-Score', ascending=False)
        
        # Display table
        print("\nClassifier Comparison Table:")
        print(df_results.to_string(index=False, float_format='%.3f'))
        
        # Save to file
        output_path = config.OUTPUTS_DIR / 'classifier_comparison.csv'
        df_results.to_csv(output_path, index=False)
        print(f"\n Comparison saved to {output_path}")
        
        return df_results
    
    def feature_ablation(self):
        """Perform feature ablation study."""
        print("\n" + "="*60)
        print("FEATURE ABLATION STUDY")
        print("="*60)
        
        # Define feature groups
        hog_indices = [i for i, name in enumerate(self.feature_names) if 'hog' in str(name)]
        lbp_indices = [i for i, name in enumerate(self.feature_names) if 'lbp' in str(name)]
        geometric_indices = [i for i, name in enumerate(self.feature_names) 
                          if any(x in str(name) for x in ['num_eyes', 'center', 'size', 'distance'])]
        pupil_indices = [i for i, name in enumerate(self.feature_names) if 'pupil' in str(name)]
        
        feature_groups = {
            'HOG only': hog_indices,
            'LBP only': lbp_indices,
            'Geometric only': geometric_indices,
            'HOG + LBP': hog_indices + lbp_indices,
            'HOG + Geometric': hog_indices + geometric_indices,
            'All features': list(range(len(self.feature_names)))
        }
        
        results = []
        
        for name, indices in feature_groups.items():
            if len(indices) == 0:
                continue
                
            # Select features
            X_train_subset = self.X_train[:, indices]
            X_test_subset = self.X_test[:, indices]
            
            # Scale
            scaler_subset = StandardScaler()
            X_train_scaled = scaler_subset.fit_transform(X_train_subset)
            X_test_scaled = scaler_subset.transform(X_test_subset)
            
            # Train SVM
            clf = SVC(C=10, gamma=0.001, kernel='rbf', random_state=42)
            clf.fit(X_train_scaled, self.y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            
            results.append({
                'Feature Set': name,
                'Num Features': len(indices),
                'Accuracy': accuracy,
                'Macro F1': f1
            })
            
            print(f"{name:<20} Features: {len(indices):<5} Accuracy: {accuracy:.3f}  F1: {f1:.3f}")
        
        # Create DataFrame
        df_ablation = pd.DataFrame(results)
        df_ablation = df_ablation.sort_values('Macro F1', ascending=False)
        
        # Save to file
        output_path = config.OUTPUTS_DIR / 'feature_ablation.csv'
        df_ablation.to_csv(output_path, index=False)
        print(f"\n Ablation results saved to {output_path}")
        
        return df_ablation
    
    def generate_summary_figure(self):
        """Generate a summary figure with key results."""
        print("\n" + "="*60)
        print("GENERATING SUMMARY FIGURE")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Per-class F1 scores
        ax = axes[0, 0]
        y_pred = self.model.predict(self.X_test_scaled)
        f1_scores = f1_score(self.y_test, y_pred, average=None)
        bars = ax.bar(self.class_names, f1_scores, color=['green', 'orange', 'red'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('F1-Score')
        ax.set_title('Per-Class F1 Scores')
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center')
        
        # 2. Confusion matrix heatmap
        ax = axes[0, 1]
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax, cbar=False)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # 3. Classifier comparison
        ax = axes[1, 0]
        df_comp = self.compare_classifiers()
        x_pos = np.arange(len(df_comp))
        ax.bar(x_pos, df_comp['F1-Score'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_comp['Classifier'])
        ax.set_ylabel('Macro F1-Score')
        ax.set_title('Classifier Comparison')
        ax.set_ylim([0, 1])
        for i, score in enumerate(df_comp['F1-Score']):
            ax.text(i, score + 0.01, f'{score:.3f}', ha='center')
        
        # 4. Feature importance (if Random Forest is available)
        ax = axes[1, 1]
        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            rf.fit(self.X_train_scaled, self.y_train)
            importances = rf.feature_importances_
            
            # Group importances
            hog_imp = np.mean([importances[i] for i, n in enumerate(self.feature_names) 
                              if 'hog' in str(n)])
            lbp_imp = np.mean([importances[i] for i, n in enumerate(self.feature_names) 
                              if 'lbp' in str(n)])
            geo_imp = np.mean([importances[i] for i, n in enumerate(self.feature_names) 
                              if any(x in str(n) for x in ['num_eyes', 'center', 'size', 'distance'])])
            pupil_imp = np.mean([importances[i] for i, n in enumerate(self.feature_names) 
                               if 'pupil' in str(n)])
            
            feature_groups = ['HOG', 'LBP', 'Geometric', 'Pupil']
            group_importances = [hog_imp, lbp_imp, geo_imp, pupil_imp]
            
            ax.bar(feature_groups, group_importances)
            ax.set_ylabel('Mean Importance')
            ax.set_title('Feature Group Importance')
            for i, imp in enumerate(group_importances):
                ax.text(i, imp + 0.001, f'{imp:.3f}', ha='center')
        except:
            ax.text(0.5, 0.5, 'Feature Importance\n(Random Forest)', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Study-Focus Monitor: Evaluation Summary', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_path = config.FIGURES_DIR / 'evaluation_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" Summary figure saved to {output_path}")
        
        plt.show()


def main():
    """Run comprehensive evaluation."""
    print("\n" + "="*60)
    print("STUDY-FOCUS MONITOR: COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Core metrics
    y_pred = evaluator.evaluate_core_metrics()
    
    # False positive rates
    fpr = evaluator.calculate_false_positive_rates(y_pred)
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(y_pred)
    
    # Processing time
    fps = evaluator.measure_processing_time()
    
    # Classifier comparison
    df_comparison = evaluator.compare_classifiers()
    
    # Feature ablation
    df_ablation = evaluator.feature_ablation()
    
    # Summary figure
    evaluator.generate_summary_figure()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nKey Results:")
    print(f"  Overall Accuracy: {accuracy_score(evaluator.y_test, y_pred):.3f}")
    print(f"  Macro F1-Score: {f1_score(evaluator.y_test, y_pred, average='macro'):.3f}")
    print(f"  Processing Speed: {fps:.1f} FPS (classification only)")
    print(f"\nAll figures and tables saved to {config.OUTPUTS_DIR}")
    print("\nNext step: Run temporal.py to add smoothing and blink filtering")


if __name__ == "__main__":
    main()