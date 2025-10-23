"""
Comprehensive Ensemble Model Evaluator
Provides detailed performance analysis and comparison with individual models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class EnsembleEvaluator:
    """
    Comprehensive evaluator for ensemble emotion recognition models
    """
    
    def __init__(self, models_path: str = "ensemble_learning/results"):
        self.models_path = Path(models_path)
        self.models = {}
        self.ensemble_weights = {}
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        self.results = {}
        
    def load_models(self):
        """Load trained ensemble models"""
        print("Loading trained models...")
        
        model_files = {
            'mini_xception': 'mini_xception_ensemble.h5',
            'mobilenetv2': 'mobilenetv2_ensemble.h5',
            'efficientnetb0': 'efficientnetb0_ensemble.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_path / filename
            if model_path.exists():
                self.models[model_name] = keras.models.load_model(model_path)
                print(f"✓ Loaded {model_name}")
            else:
                print(f"✗ Model not found: {model_name}")
        
        # Load ensemble weights
        weights_path = self.models_path / "ensemble_weights.json"
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.ensemble_weights = json.load(f)
            print("✓ Loaded ensemble weights")
        
        # Load previous results if available
        results_path = self.models_path / "evaluation_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.results = json.load(f)
            print("✓ Loaded previous results")
    
    def load_test_data(self, data_path: str = "../fer2013_project/data/fer2013.csv") -> Tuple[np.ndarray, np.ndarray]:
        """Load test data for evaluation"""
        print("Loading test data...")
        
        # Load CSV data
        df = pd.read_csv(data_path)
        
        # Convert pixel strings to arrays
        def pixels_to_array(pixel_str):
            return np.array([int(pixel) for pixel in pixel_str.split()]).reshape(48, 48)
        
        # Process images
        images = np.array([pixels_to_array(pixels) for pixels in df['pixels']])
        images = images.astype('float32') / 255.0
        images = np.expand_dims(images, axis=-1)  # Add channel dimension
        
        # Convert labels to categorical
        labels = keras.utils.to_categorical(df['emotion'], num_classes=7)
        
        # Get test data
        test_mask = df['Usage'] == 'PrivateTest'
        X_test = images[test_mask]
        y_test = labels[test_mask]
        
        print(f"Test samples: {len(X_test)}")
        return X_test, y_test
    
    def prepare_rgb_data(self, X_gray: np.ndarray) -> np.ndarray:
        """Convert grayscale to RGB for transfer learning models"""
        return np.repeat(X_gray, 3, axis=-1)
    
    def get_model_predictions(self, X_test) -> Dict[str, np.ndarray]:
        """Get predictions from all individual models"""
        predictions = {}
        
        # Mini-XCEPTION prediction
        if 'mini_xception' in self.models:
            predictions['mini_xception'] = self.models['mini_xception'].predict(X_test)
        
        # Transfer learning models predictions (RGB)
        X_test_rgb = self.prepare_rgb_data(X_test)
        
        if 'mobilenetv2' in self.models:
            predictions['mobilenetv2'] = self.models['mobilenetv2'].predict(X_test_rgb)
        
        if 'efficientnetb0' in self.models:
            predictions['efficientnetb0'] = self.models['efficientnetb0'].predict(X_test_rgb)
        
        return predictions
    
    def create_ensemble_prediction(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create ensemble prediction using weighted voting"""
        if not self.ensemble_weights:
            print("Warning: No ensemble weights found, using equal weights")
            weights = {k: 1.0/len(predictions) for k in predictions.keys()}
        else:
            weights = self.ensemble_weights
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                ensemble_pred += weights[model_name] * pred
        
        return ensemble_pred
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        
        metrics['f1_per_class'] = dict(zip(self.emotion_labels, f1_per_class))
        metrics['precision_per_class'] = dict(zip(self.emotion_labels, precision_per_class))
        metrics['recall_per_class'] = dict(zip(self.emotion_labels, recall_per_class))
        
        # ROC-AUC (one-vs-rest)
        try:
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='macro')
            metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_pred_proba, 
                                                      multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test) -> Dict:
        """Evaluate all models comprehensively"""
        print("Evaluating all models...")
        
        # Get predictions from all models
        individual_predictions = self.get_model_predictions(X_test)
        
        # Create ensemble prediction
        ensemble_pred_proba = self.create_ensemble_prediction(individual_predictions)
        
        # Convert to class predictions
        y_true = np.argmax(y_test, axis=1)
        
        results = {}
        
        # Evaluate individual models
        for model_name, pred_proba in individual_predictions.items():
            pred_classes = np.argmax(pred_proba, axis=1)
            
            metrics = self.calculate_comprehensive_metrics(y_true, pred_classes, pred_proba)
            
            # Add confusion matrix
            cm = confusion_matrix(y_true, pred_classes)
            metrics['confusion_matrix'] = cm.tolist()
            
            results[model_name] = metrics
            
            print(f"\n{model_name.upper()} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Micro F1: {metrics['micro_f1']:.4f}")
            print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"  ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        
        # Evaluate ensemble
        ensemble_pred_classes = np.argmax(ensemble_pred_proba, axis=1)
        
        ensemble_metrics = self.calculate_comprehensive_metrics(y_true, ensemble_pred_classes, ensemble_pred_proba)
        ensemble_metrics['confusion_matrix'] = confusion_matrix(y_true, ensemble_pred_classes).tolist()
        
        results['ensemble'] = ensemble_metrics
        
        print(f"\nENSEMBLE Results:")
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {ensemble_metrics['macro_f1']:.4f}")
        print(f"  Micro F1: {ensemble_metrics['micro_f1']:.4f}")
        print(f"  Weighted F1: {ensemble_metrics['weighted_f1']:.4f}")
        print(f"  ROC-AUC (Macro): {ensemble_metrics['roc_auc_macro']:.4f}")
        
        self.results = results
        return results
    
    def create_detailed_visualizations(self):
        """Create detailed performance visualizations"""
        if not self.results:
            print("No results available for visualization")
            return
        
        print("Creating detailed visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall Performance Comparison
        self._plot_overall_comparison()
        
        # 2. Per-Class Performance
        self._plot_per_class_performance()
        
        # 3. Confusion Matrices
        self._plot_confusion_matrices()
        
        # 4. ROC Curves
        self._plot_roc_curves()
        
        # 5. Performance Improvement Analysis
        self._plot_improvement_analysis()
        
        print("All visualizations saved!")
    
    def _plot_overall_comparison(self):
        """Plot overall performance comparison"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            values = [self.results[model][metric] for model in models]
            bars = ax.bar(models, values, color=sns.color_palette("husl", len(models)))
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_per_class_performance(self):
        """Plot per-class F1 scores"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        metrics = ['f1_per_class', 'precision_per_class', 'recall_per_class']
        titles = ['F1-Score per Class', 'Precision per Class', 'Recall per Class']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            # Prepare data
            data = []
            for model in self.results.keys():
                for emotion, score in self.results[model][metric].items():
                    data.append({'Model': model, 'Emotion': emotion, 'Score': score})
            
            df_plot = pd.DataFrame(data)
            
            # Create grouped bar plot
            sns.barplot(data=df_plot, x='Emotion', y='Score', hue='Model', ax=ax)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = np.array(results['confusion_matrix'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=self.emotion_labels, yticklabels=self.emotion_labels,
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name.upper()} Confusion Matrix', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curves(self):
        """Plot ROC curves comparison"""
        # This would require probability predictions for each class
        # For now, we'll create a simple comparison of ROC-AUC scores
        
        models = list(self.results.keys())
        roc_auc_scores = [self.results[model]['roc_auc_macro'] for model in models]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        bars = ax.bar(models, roc_auc_scores, color=sns.color_palette("husl", len(models)))
        ax.set_title('ROC-AUC (Macro) Comparison', fontweight='bold')
        ax.set_ylabel('ROC-AUC Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, roc_auc_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'roc_auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_improvement_analysis(self):
        """Plot improvement analysis over individual models"""
        if 'ensemble' not in self.results:
            return
        
        # Find best individual model
        individual_models = [k for k in self.results.keys() if k != 'ensemble']
        best_individual = max(individual_models, 
                             key=lambda x: self.results[x]['accuracy'])
        
        # Compare ensemble with best individual
        metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']
        
        ensemble_scores = [self.results['ensemble'][m] for m in metrics]
        best_individual_scores = [self.results[best_individual][m] for m in metrics]
        
        # Calculate improvements
        improvements = [(e - b) / b * 100 for e, b in zip(ensemble_scores, best_individual_scores)]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ensemble_scores, width, label='Ensemble', 
                      color='#2ca02c', alpha=0.8)
        bars2 = ax.bar(x + width/2, best_individual_scores, width, 
                      label=f'Best Individual ({best_individual})', 
                      color='#ff7f0e', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(f'Ensemble vs Best Individual Model ({best_individual})', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add improvement percentages
        for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            max_height = max(height1, height2)
            
            ax.text(i, max_height + 0.02, f'+{imp:.1f}%', 
                   ha='center', va='bottom', fontweight='bold', 
                   color='green' if imp > 0 else 'red')
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_comprehensive_report(self):
        """Save comprehensive evaluation report"""
        if not self.results:
            print("No results to save")
            return
        
        report_path = self.models_path / "comprehensive_evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE ENSEMBLE EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            if 'ensemble' in self.results:
                ensemble_acc = self.results['ensemble']['accuracy']
                f.write(f"Ensemble Model Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)\n")
            
            # Find best individual model
            individual_models = [k for k in self.results.keys() if k != 'ensemble']
            if individual_models:
                best_individual = max(individual_models, 
                                     key=lambda x: self.results[x]['accuracy'])
                best_acc = self.results[best_individual]['accuracy']
                f.write(f"Best Individual Model: {best_individual} ({best_acc:.4f} / {best_acc*100:.2f}%)\n")
                
                if 'ensemble' in self.results:
                    improvement = (ensemble_acc - best_acc) / best_acc * 100
                    f.write(f"Ensemble Improvement: {improvement:+.2f}%\n")
            
            f.write("\n\nDETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
                f.write(f"  Micro F1: {metrics['micro_f1']:.4f}\n")
                f.write(f"  Weighted F1: {metrics['weighted_f1']:.4f}\n")
                f.write(f"  Macro Precision: {metrics['macro_precision']:.4f}\n")
                f.write(f"  Macro Recall: {metrics['macro_recall']:.4f}\n")
                f.write(f"  ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}\n")
            
            f.write("\n\nPER-CLASS PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name.upper()} F1-Scores per Class:\n")
                for emotion, score in metrics['f1_per_class'].items():
                    f.write(f"  {emotion}: {score:.4f}\n")
        
        print(f"Comprehensive report saved to {report_path}")

def main():
    """Main evaluation pipeline"""
    print("=== Comprehensive Ensemble Evaluation ===")
    
    # Initialize evaluator
    evaluator = EnsembleEvaluator()
    
    # Load models
    evaluator.load_models()
    
    if not evaluator.models:
        print("No models found! Please train models first using ensemble_trainer.py")
        return
    
    # Load test data
    X_test, y_test = evaluator.load_test_data()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models(X_test, y_test)
    
    # Create visualizations
    evaluator.create_detailed_visualizations()
    
    # Save comprehensive report
    evaluator.save_comprehensive_report()
    
    print("\n=== EVALUATION COMPLETED ===")

if __name__ == "__main__":
    main()
