"""
Training and model comparison pipeline for Build-Failure Prediction.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import prepare_data
from models import get_models

def train_and_benchmark(data_path: str = 'data/ci_builds_dataset.csv',
                        output_dir: str = 'models',
                        split_strategy: str = 'temporal'):
    """
    Train multiple models, benchmark their performance on unseen future CI builds,
    and persist the top-performing artifact.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Step 1: Loading & Preprocessing CI Build Data ===")
    data = prepare_data(data_path=data_path, split_strategy=split_strategy)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    preprocessor = data['preprocessor']
    feature_names = data['feature_names']
    
    print(f"Training instances: {X_train.shape[0]} | Testing instances: {X_test.shape[0]}")
    print(f"Train Failure Rate: {y_train.mean():.2%} | Test Failure Rate: {y_test.mean():.2%}")
    
    models = get_models()
    results = []
    fitted_models = {}
    test_predictions = {}
    test_probabilities = {}
    
    print("\n=== Step 2: Training & Benchmarking Models ===")
    for name, model in models.items():
        print(f"--> Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        y_pred = model.predict(X_test)
        test_predictions[name] = y_pred
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = np.zeros_like(y_pred, dtype=float)
            
        test_probabilities[name] = y_prob
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
        except Exception:
            auc = 0.5
            pr_auc = 0.0
            
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Balanced_Accuracy': round(bal_acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(auc, 4),
            'PR_AUC': round(pr_auc, 4)
        })
        
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1_Score', ascending=False).reset_index(drop=True)
    
    print("\n=== Benchmark Results ===")
    print(results_df.to_string(index=False))
    
    # Save benchmark table
    results_df.to_csv(os.path.join(output_dir, 'benchmark_results.csv'), index=False)
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Select Best Model based on F1-Score on imbalanced failure class
    best_model_name = results_df.iloc[0]['Model']
    best_model = fitted_models[best_model_name]
    print(f"\n[+] Champion Model Selected: {best_model_name} (F1 = {results_df.iloc[0]['F1_Score']})")
    
    # Persist artifacts
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.joblib'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    
    metadata = {
        'best_model_name': best_model_name,
        'feature_names': feature_names,
        'metrics': results_df.iloc[0].to_dict(),
        'split_strategy': split_strategy,
        'num_train_samples': int(X_train.shape[0]),
        'num_test_samples': int(X_test.shape[0])
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Also save test predictions and probabilities for evaluation / plotting
    np.savez_compressed(
        os.path.join(output_dir, 'eval_cache.npz'),
        y_test=y_test,
        **{f"prob_{k}": test_probabilities[k] for k in test_probabilities},
        **{f"pred_{k}": test_predictions[k] for k in test_predictions}
    )
    
    return results_df, best_model, preprocessor

if __name__ == '__main__':
    train_and_benchmark()
