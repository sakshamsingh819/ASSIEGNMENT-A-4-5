"""
Comprehensive evaluation and visualization module for Build-Failure Prediction.
Generates publication-quality charts and tables.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score
)

def set_plot_style():
    """Configure aesthetic matplotlib parameters."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
        'grid.color': '#eeeeee',
        'grid.linestyle': '--',
        'figure.autolayout': True
    })

def plot_class_distribution(y, output_path: str = 'plots/class_distribution.png'):
    """Plot distribution of build outcomes (Pass vs Fail)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = pd.Series(y).value_counts().sort_index()
    labels = ['Passed (0)', 'Failed (1)']
    colors = ['#2ecc71', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    bars = ax.bar(labels, counts.values, color=colors, width=0.55, edgecolor='#333333', linewidth=1.2)
    
    total = len(y)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., h + (total * 0.015),
                f"{h:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_title('Continuous Integration Build Outcome Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Builds', fontsize=12)
    ax.set_ylim(0, max(counts.values) * 1.18)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {output_path}")

def plot_roc_curves(y_test, prob_dict, output_path: str = 'plots/roc_curves.png'):
    """Plot ROC curves for all benchmarked models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    palette = ['#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#7f8c8d']
    
    for i, (name, probs) in enumerate(prob_dict.items()):
        clean_name = name.replace('prob_', '').replace('_', ' ')
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        color = palette[i % len(palette)]
        ax.plot(fpr, tpr, label=f"{clean_name} (AUC = {auc:.3f})", color=color, linewidth=2.2)
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {output_path}")

def plot_pr_curves(y_test, prob_dict, output_path: str = 'plots/precision_recall_curves.png'):
    """Plot Precision-Recall curves (critical for imbalanced defect prediction)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    palette = ['#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#7f8c8d']
    baseline_rate = np.mean(y_test)
    
    for i, (name, probs) in enumerate(prob_dict.items()):
        clean_name = name.replace('prob_', '').replace('_', ' ')
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        color = palette[i % len(palette)]
        ax.plot(rec, prec, label=f"{clean_name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2.2)
        
    ax.axhline(y=baseline_rate, color='r', linestyle='--', alpha=0.7, 
               label=f'No-Skill Baseline ({baseline_rate:.1%})')
    
    ax.set_title('Precision-Recall Curves (Failure Class)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {output_path}")

def plot_confusion_matrix(y_test, y_pred, model_name: str, output_path: str = 'plots/confusion_matrix_best.png'):
    """Plot annotated confusion matrix with counts and percentages."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False, ax=ax, linewidths=1.5, linecolor='white')
    
    # Custom annotations with count and recall per class
    classes = ['Pass (0)', 'Fail (1)']
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            label = f"{val:,}\n({pct:.1f}%)"
            color = 'white' if cm[i, j] > cm.max() * 0.45 else 'black'
            ax.text(j + 0.5, i + 0.5, label, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)
            
    clean_title = model_name.replace('_', ' ')
    ax.set_title(f'Confusion Matrix: {clean_title}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Outcome', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual Outcome', fontsize=12, labelpad=10)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11, rotation=0)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {output_path}")

def plot_feature_importance(model, feature_names: list, output_path: str = 'plots/feature_importance.png'):
    """Extract and plot top predictive features driving build failure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    importances = None
    metric_label = "Feature Importance"
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        metric_label = "Gini Importance"
    elif hasattr(model, 'coef_'):
        # For logistic regression: absolute value of standardized coefficients
        importances = np.abs(model.coef_[0])
        metric_label = "Absolute Logistic Coefficient |β|"
        
    if importances is None:
        return
        
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(12) # Top 12 features
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color='#2980b9', edgecolor='#1a5276', height=0.65)
    
    ax.set_title('Top Predictive Signals Driving Build Failures', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('CI Metric / Feature', fontsize=12)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {output_path}")

def plot_metrics_comparison(results_csv: str = 'models/benchmark_results.csv',
                            output_path: str = 'plots/metrics_comparison.png'):
    """Compare multiple metrics across all models side-by-side."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(results_csv)
    
    # Filter out baseline for clean visual comparison or keep as reference
    metrics = ['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    
    melted = df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
    melted['Model'] = melted['Model'].str.replace('_', ' ')
    melted['Metric'] = melted['Metric'].str.replace('_', ' ')
    
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette='tab10', ax=ax, edgecolor='#333333')
    
    ax.set_title('Comparative Benchmark Across Machine Learning Models', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Score [0.0 - 1.0]', fontsize=12)
    ax.set_xlabel('Evaluation Metric', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {output_path}")

def generate_all_evaluations():
    """Run complete evaluation and rendering suite."""
    set_plot_style()
    
    # Load cache
    cache = np.load('models/eval_cache.npz')
    y_test = cache['y_test']
    
    with open('models/metadata.json') as f:
        meta = json.load(f)
        
    best_model_name = meta['best_model_name']
    feature_names = meta['feature_names']
    best_model = joblib.load('models/best_model.joblib')
    
    prob_keys = [k for k in cache.files if k.startswith('prob_')]
    prob_dict = {k: cache[k] for k in prob_keys}
    
    best_pred_key = f"pred_{best_model_name}"
    y_pred_best = cache[best_pred_key] if best_pred_key in cache.files else best_model.predict(
        joblib.load('models/preprocessor.joblib').transform(pd.read_csv('data/ci_builds_dataset.csv').tail(len(y_test)))
    )
    
    print("\n=== Generating Evaluation Plots ===")
    plot_class_distribution(y_test)
    plot_roc_curves(y_test, prob_dict)
    plot_pr_curves(y_test, prob_dict)
    plot_confusion_matrix(y_test, y_pred_best, best_model_name)
    plot_feature_importance(best_model, feature_names)
    plot_metrics_comparison()
    print("All evaluation plots successfully generated in plots/\n")

if __name__ == '__main__':
    generate_all_evaluations()

