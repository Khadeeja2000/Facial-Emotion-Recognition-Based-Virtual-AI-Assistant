"""
Ensemble Model Comparison and Visualization System
Comprehensive comparison of ensemble vs individual models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnsembleComparison:
    """
    Comprehensive comparison system for ensemble vs individual models
    """
    
    def __init__(self, results_path: str = "ensemble_learning/results"):
        self.results_path = Path(results_path)
        self.results = {}
        self.emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        
    def load_evaluation_results(self):
        """Load evaluation results from JSON file"""
        results_file = self.results_path / "evaluation_results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            print("✓ Loaded evaluation results")
        else:
            print("✗ No evaluation results found")
            return False
        
        return True
    
    def create_comprehensive_comparison(self):
        """Create comprehensive comparison visualizations"""
        if not self.results:
            print("No results available for comparison")
            return
        
        print("Creating comprehensive comparison visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create multiple comparison plots
        self._create_metrics_comparison()
        self._create_per_class_comparison()
        self._create_improvement_analysis()
        self._create_confusion_matrix_comparison()
        self._create_performance_radar_chart()
        self._create_ensemble_contribution_analysis()
        
        print("All comparison visualizations created!")
    
    def _create_metrics_comparison(self):
        """Create comprehensive metrics comparison"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            values = [self.results[model][metric] for model in models]
            colors = ['#2ca02c' if model == 'ensemble' else '#1f77b4' for model in models]
            
            bars = ax.bar(models, values, color=colors, alpha=0.8)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight ensemble
            if 'ensemble' in models:
                ensemble_idx = models.index('ensemble')
                bars[ensemble_idx].set_edgecolor('red')
                bars[ensemble_idx].set_linewidth(3)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_per_class_comparison(self):
        """Create per-class performance comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        metrics = ['f1_per_class', 'precision_per_class', 'recall_per_class']
        titles = ['F1-Score per Class', 'Precision per Class', 'Recall per Class']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            # Prepare data for plotting
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
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', rotation=90, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'per_class_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_improvement_analysis(self):
        """Create improvement analysis over best individual model"""
        if 'ensemble' not in self.results:
            return
        
        # Find best individual model
        individual_models = [k for k in self.results.keys() if k != 'ensemble']
        if not individual_models:
            return
        
        best_individual = max(individual_models, 
                             key=lambda x: self.results[x]['accuracy'])
        
        # Compare ensemble with best individual
        metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        
        ensemble_scores = [self.results['ensemble'][m] for m in metrics]
        best_individual_scores = [self.results[best_individual][m] for m in metrics]
        
        # Calculate improvements
        improvements = [(e - b) / b * 100 for e, b in zip(ensemble_scores, best_individual_scores)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart comparison
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, ensemble_scores, width, label='Ensemble', 
                       color='#2ca02c', alpha=0.8)
        bars2 = ax1.bar(x + width/2, best_individual_scores, width, 
                       label=f'Best Individual ({best_individual})', 
                       color='#ff7f0e', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Ensemble vs Best Individual Model ({best_individual})', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add improvement percentages
        for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            max_height = max(height1, height2)
            
            ax1.text(i, max_height + 0.02, f'+{imp:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', 
                    color='green' if imp > 0 else 'red')
        
        # Improvement percentage chart
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(metrics, improvements, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Ensemble Improvement Over Best Individual', fontweight='bold')
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_confusion_matrix_comparison(self):
        """Create confusion matrix comparison"""
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
        plt.savefig(self.results_path / 'confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_performance_radar_chart(self):
        """Create radar chart comparison"""
        if len(self.results) < 2:
            return
        
        # Select key metrics for radar chart
        radar_metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
        
        # Set up radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            values = [results[metric] for metric in radar_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart Comparison', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'performance_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_ensemble_contribution_analysis(self):
        """Analyze individual model contributions to ensemble"""
        # Load ensemble weights if available
        weights_file = self.results_path / "ensemble_weights.json"
        if not weights_file.exists():
            print("No ensemble weights found for contribution analysis")
            return
        
        with open(weights_file, 'r') as f:
            ensemble_weights = json.load(f)
        
        # Get individual model performances
        individual_models = [k for k in self.results.keys() if k != 'ensemble']
        
        if not individual_models:
            return
        
        # Create contribution analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Weight vs Performance scatter plot
        weights = [ensemble_weights.get(model, 0) for model in individual_models]
        accuracies = [self.results[model]['accuracy'] for model in individual_models]
        
        scatter = ax1.scatter(weights, accuracies, s=200, alpha=0.7, c=range(len(individual_models)), cmap='viridis')
        
        for i, model in enumerate(individual_models):
            ax1.annotate(model, (weights[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('Ensemble Weight')
        ax1.set_ylabel('Individual Accuracy')
        ax1.set_title('Model Weight vs Individual Performance', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Pie chart of ensemble composition
        ax2.pie(weights, labels=individual_models, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Ensemble Model Composition', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'ensemble_contribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results available for report generation")
            return
        
        report_path = self.results_path / "ensemble_comparison_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ENSEMBLE MODEL COMPARISON REPORT\n")
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
                    
                    if improvement > 0:
                        f.write("✓ Ensemble outperforms best individual model\n")
                    else:
                        f.write("✗ Ensemble does not outperform best individual model\n")
            
            f.write("\n\nDETAILED MODEL COMPARISON\n")
            f.write("-" * 30 + "\n")
            
            # Create comparison table
            metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
            
            f.write(f"{'Model':<15} {'Accuracy':<10} {'Macro F1':<10} {'Micro F1':<10} {'Weighted F1':<12} {'Precision':<10} {'Recall':<10}\n")
            f.write("-" * 90 + "\n")
            
            for model_name, metrics_dict in self.results.items():
                f.write(f"{model_name:<15} {metrics_dict['accuracy']:<10.4f} {metrics_dict['macro_f1']:<10.4f} "
                       f"{metrics_dict['micro_f1']:<10.4f} {metrics_dict['weighted_f1']:<12.4f} "
                       f"{metrics_dict['macro_precision']:<10.4f} {metrics_dict['macro_recall']:<10.4f}\n")
            
            f.write("\n\nPER-CLASS PERFORMANCE ANALYSIS\n")
            f.write("-" * 35 + "\n")
            
            if 'ensemble' in self.results:
                f.write("\nEnsemble F1-Scores per Class:\n")
                for emotion, score in self.results['ensemble']['f1_per_class'].items():
                    f.write(f"  {emotion}: {score:.4f}\n")
                
                # Find best performing emotion class
                best_emotion = max(self.results['ensemble']['f1_per_class'].items(), key=lambda x: x[1])
                worst_emotion = min(self.results['ensemble']['f1_per_class'].items(), key=lambda x: x[1])
                
                f.write(f"\nBest performing emotion: {best_emotion[0]} (F1: {best_emotion[1]:.4f})\n")
                f.write(f"Worst performing emotion: {worst_emotion[0]} (F1: {worst_emotion[1]:.4f})\n")
            
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if 'ensemble' in self.results and individual_models:
                ensemble_acc = self.results['ensemble']['accuracy']
                best_individual_acc = max(self.results[model]['accuracy'] for model in individual_models)
                
                if ensemble_acc > best_individual_acc:
                    f.write("✓ Ensemble model is recommended for deployment\n")
                    f.write("  - Provides better accuracy than any individual model\n")
                    f.write("  - Combines strengths of multiple architectures\n")
                    f.write("  - More robust to different types of inputs\n")
                else:
                    f.write("⚠ Consider using best individual model instead\n")
                    f.write("  - Ensemble does not provide significant improvement\n")
                    f.write("  - Individual model may be more efficient\n")
                    f.write("  - Simpler deployment and maintenance\n")
            
            f.write("\n\nTECHNICAL DETAILS\n")
            f.write("-" * 18 + "\n")
            f.write("Models evaluated:\n")
            for model_name in self.results.keys():
                f.write(f"  - {model_name}\n")
            
            f.write(f"\nEmotion classes: {', '.join(self.emotion_labels)}\n")
            f.write(f"Total metrics evaluated: {len(metrics)}\n")
        
        print(f"Comparison report saved to {report_path}")

def main():
    """Main comparison pipeline"""
    print("=== Ensemble Model Comparison ===")
    
    # Initialize comparison system
    comparator = EnsembleComparison()
    
    # Load results
    if not comparator.load_evaluation_results():
        print("Please run ensemble_evaluator.py first to generate evaluation results")
        return
    
    # Create comprehensive comparison
    comparator.create_comprehensive_comparison()
    
    # Generate comparison report
    comparator.generate_comparison_report()
    
    print("\n=== COMPARISON COMPLETED ===")

if __name__ == "__main__":
    main()
