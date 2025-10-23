"""
Clean Visualization Script - Focus on Good Models Only
Creates professional plots excluding the broken EfficientNetB0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def load_results():
    """Load evaluation results"""
    results_path = Path("ensemble_learning/results/evaluation_results.json")
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    return None

def create_clean_performance_comparison():
    """Create clean performance comparison excluding EfficientNetB0"""
    results = load_results()
    if not results:
        print("No results found")
        return
    
    # Filter out EfficientNetB0 (broken model)
    good_models = {k: v for k, v in results.items() if k != 'efficientnetb0'}
    
    models = list(good_models.keys())
    metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Ensemble vs Individual Models Performance Comparison\n(Excluding Failed EfficientNetB0)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']  # Green for ensemble, Orange for Mini-XCEPTION, Blue for MobileNetV2
    
    for idx, metric in enumerate(metrics):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        values = [good_models[model][metric] for model in models]
        bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8)
        
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
    plt.savefig('ensemble_learning/results/CLEAN_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Clean performance comparison saved!")

def create_clean_confusion_matrices():
    """Create clean confusion matrices excluding EfficientNetB0"""
    results = load_results()
    if not results:
        print("No results found")
        return
    
    # Filter out EfficientNetB0
    good_models = {k: v for k, v in results.items() if k != 'efficientnetb0'}
    emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    
    n_models = len(good_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results_dict) in enumerate(good_models.items()):
        cm = np.array(results_dict['confusion_matrix'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=emotion_labels, yticklabels=emotion_labels,
                   ax=axes[idx], cbar_kws={'shrink': 0.8})
        
        # Set title with accuracy
        accuracy = results_dict['accuracy']
        axes[idx].set_title(f'{model_name.upper()}\nAccuracy: {accuracy:.3f}', 
                           fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('ensemble_learning/results/CLEAN_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Clean confusion matrices saved!")

def create_ensemble_benefit_analysis():
    """Create analysis showing ensemble benefits"""
    results = load_results()
    if not results:
        print("No results found")
        return
    
    # Compare ensemble with best individual model
    individual_models = {k: v for k, v in results.items() 
                        if k not in ['ensemble', 'efficientnetb0']}
    
    if not individual_models:
        print("No individual models to compare")
        return
    
    best_individual = max(individual_models.keys(), 
                         key=lambda x: individual_models[x]['accuracy'])
    
    metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
    
    ensemble_scores = [results['ensemble'][m] for m in metrics]
    best_individual_scores = [individual_models[best_individual][m] for m in metrics]
    
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
    ax1.set_title(f'Ensemble vs Best Individual Model\n({best_individual})', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add improvement percentages
    for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        max_height = max(height1, height2)
        
        ax1.text(i, max_height + 0.02, f'{imp:+.1f}%', 
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
    plt.savefig('ensemble_learning/results/CLEAN_ensemble_benefits.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Ensemble benefits analysis saved!")

def create_per_class_analysis():
    """Create per-class performance analysis"""
    results = load_results()
    if not results:
        print("No results found")
        return
    
    # Filter out EfficientNetB0
    good_models = {k: v for k, v in results.items() if k != 'efficientnetb0'}
    emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    metrics = ['f1_per_class', 'precision_per_class', 'recall_per_class']
    titles = ['F1-Score per Emotion', 'Precision per Emotion', 'Recall per Emotion']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Prepare data for plotting
        data = []
        for model in good_models.keys():
            for emotion, score in good_models[model][metric].items():
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
            ax.bar_label(container, fmt='%.2f', rotation=90, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ensemble_learning/results/CLEAN_per_class_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Per-class analysis saved!")

def create_summary_report():
    """Create a clean summary report"""
    results = load_results()
    if not results:
        print("No results found")
        return
    
    # Filter out EfficientNetB0
    good_models = {k: v for k, v in results.items() if k != 'efficientnetb0'}
    
    report_path = Path("ensemble_learning/results/CLEAN_ENSEMBLE_SUMMARY.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLEAN ENSEMBLE LEARNING SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        
        if 'ensemble' in good_models:
            ensemble_acc = good_models['ensemble']['accuracy']
            f.write(f"✓ Ensemble Model Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)\n")
        
        # Find best individual model
        individual_models = {k: v for k, v in good_models.items() if k != 'ensemble'}
        if individual_models:
            best_individual = max(individual_models.keys(), 
                                 key=lambda x: individual_models[x]['accuracy'])
            best_acc = individual_models[best_individual]['accuracy']
            f.write(f"✓ Best Individual Model: {best_individual} ({best_acc:.4f} / {best_acc*100:.2f}%)\n")
            
            if 'ensemble' in good_models:
                improvement = (ensemble_acc - best_acc) / best_acc * 100
                f.write(f"✓ Ensemble vs Best Individual: {improvement:+.2f}% improvement\n")
        
        f.write(f"✓ Models Successfully Trained: {len(good_models)}\n")
        f.write("✗ EfficientNetB0: Failed (predicts only 'happy' emotion)\n")
        
        f.write("\n\nDETAILED RESULTS\n")
        f.write("-" * 20 + "\n")
        
        for model_name, metrics in good_models.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"  Micro F1: {metrics['micro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {metrics['weighted_f1']:.4f}\n")
            f.write(f"  Macro Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"  Macro Recall: {metrics['macro_recall']:.4f}\n")
            f.write(f"  ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}\n")
        
        f.write("\n\nKEY FINDINGS\n")
        f.write("-" * 15 + "\n")
        f.write("1. Mini-XCEPTION performs best as individual model\n")
        f.write("2. Ensemble maintains competitive performance\n")
        f.write("3. Transfer learning models need better fine-tuning\n")
        f.write("4. EfficientNetB0 completely failed - domain mismatch\n")
        
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        f.write("1. Use Mini-XCEPTION for single model deployment\n")
        f.write("2. Use Ensemble for maximum robustness\n")
        f.write("3. Improve transfer learning fine-tuning\n")
        f.write("4. Focus on Mini-XCEPTION architecture optimization\n")
    
    print(f"✓ Clean summary report saved to {report_path}")

def main():
    """Create all clean visualizations"""
    print("Creating clean visualizations (excluding EfficientNetB0)...")
    
    create_clean_performance_comparison()
    create_clean_confusion_matrices()
    create_ensemble_benefit_analysis()
    create_per_class_analysis()
    create_summary_report()
    
    print("\n🎉 All clean visualizations created successfully!")
    print("Files saved with 'CLEAN_' prefix in results/ directory")

if __name__ == "__main__":
    main()
