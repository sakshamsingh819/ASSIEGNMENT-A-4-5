"""
Inference CLI and Real-Time Build Failure Predictor for CI/CD Pipelines.
"""

import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np

# Ensure local imports work whether executed directly or as a package
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import engineer_features, NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES

def load_artifacts(model_dir: str = 'models'):
    """Load model, preprocessor, and metadata."""
    model_path = os.path.join(model_dir, 'best_model.joblib')
    prep_path = os.path.join(model_dir, 'preprocessor.joblib')
    meta_path = os.path.join(model_dir, 'metadata.json')
    
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        raise FileNotFoundError(f"Trained model artifacts not found in {model_dir}. Please run train.py first.")
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    
    with open(meta_path) as f:
        metadata = json.load(f)
        
    return model, preprocessor, metadata

def predict_single(input_dict: dict, model, preprocessor) -> dict:
    """Predict failure risk for a single build/commit."""
    df_raw = pd.DataFrame([input_dict])
    
    # Fill defaults if missing
    defaults = {
        'num_commits': 1,
        'files_changed': 2,
        'lines_added': 25,
        'lines_deleted': 10,
        'churn_total': input_dict.get('lines_added', 25) + input_dict.get('lines_deleted', 10),
        'src_churn': input_dict.get('lines_added', 25) + input_dict.get('lines_deleted', 10),
        'test_churn': 0,
        'test_to_src_ratio': 0.0,
        'time_since_last_build_min': 45,
        'committer_experience': 25,
        'committer_recent_failure_rate': 0.15,
        'team_size': 15,
        'consecutive_failures': 0,
        'is_pull_request': 1,
        'prev_build_status': 0,
        'git_branch': 'feature'
    }
    for k, v in defaults.items():
        if k not in df_raw.columns:
            df_raw[k] = v
            
    df_feat = engineer_features(df_raw)
    all_input_cols = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    
    X_trans = preprocessor.transform(df_feat[all_input_cols])
    
    if hasattr(model, 'predict_proba'):
        prob_fail = float(model.predict_proba(X_trans)[0, 1])
    else:
        prob_fail = float(model.predict(X_trans)[0])
        
    # Risk categorization
    if prob_fail < 0.20:
        risk_level = "LOW"
        color_code = "\033[92m" # Green
        rec = "Eligible for fast-track / parallelized test execution."
    elif prob_fail < 0.50:
        risk_level = "MODERATE"
        color_code = "\033[93m" # Yellow
        rec = "Execute standard test suite."
    elif prob_fail < 0.75:
        risk_level = "HIGH"
        color_code = "\033[91m" # Red
        rec = "Prioritize defect-prone test cases and flag for committer review."
    else:
        risk_level = "CRITICAL"
        color_code = "\033[95m" # Magenta
        rec = "Mandate full end-to-end regression suite; high likelihood of pipeline failure."
        
    # Risk factor diagnostics
    risk_factors = []
    if input_dict.get('prev_build_status', 0) == 1:
        risk_factors.append("Previous build failed (cascade failure risk)")
    if input_dict.get('test_to_src_ratio', 0.0) < 0.1 and input_dict.get('churn_total', 0) > 100:
        risk_factors.append("Substantial code churn with little or no accompanying test modifications")
    if input_dict.get('consecutive_failures', 0) >= 2:
        risk_factors.append(f"{input_dict.get('consecutive_failures')} consecutive failure streak on branch")
    if input_dict.get('committer_experience', 20) < 10:
        risk_factors.append("Committer has low repository tenure / history")
        
    return {
        'failure_probability': round(prob_fail, 4),
        'risk_level': risk_level,
        'recommendation': rec,
        'risk_factors': risk_factors
    }

def run_demo():
    """Run built-in benchmark demonstration cases."""
    print("=================================================================")
    print("      Continuous Integration Build-Failure Predictor (BFP)       ")
    print("=================================================================\n")
    
    model, preprocessor, metadata = load_artifacts()
    print(f"Loaded Champion Model: {metadata['best_model_name']}")
    print(f"Model Baseline ROC-AUC: {metadata['metrics'].get('ROC_AUC', 'N/A')}\n")
    
    cases = [
        {
            'name': "Scenario 1: High-Risk Hotfix (Prior Failure + No Tests)",
            'data': {
                'git_branch': 'hotfix',
                'is_pull_request': 1,
                'num_commits': 4,
                'files_changed': 9,
                'lines_added': 420,
                'lines_deleted': 110,
                'churn_total': 530,
                'src_churn': 530,
                'test_churn': 0,
                'test_to_src_ratio': 0.0,
                'time_since_last_build_min': 20,
                'committer_experience': 8,
                'committer_recent_failure_rate': 0.35,
                'team_size': 12,
                'prev_build_status': 1,
                'consecutive_failures': 2
            }
        },
        {
            'name': "Scenario 2: Low-Risk Feature with Comprehensive Tests",
            'data': {
                'git_branch': 'feature',
                'is_pull_request': 1,
                'num_commits': 1,
                'files_changed': 2,
                'lines_added': 45,
                'lines_deleted': 12,
                'churn_total': 57,
                'src_churn': 30,
                'test_churn': 27,
                'test_to_src_ratio': 0.90,
                'time_since_last_build_min': 90,
                'committer_experience': 85,
                'committer_recent_failure_rate': 0.08,
                'team_size': 22,
                'prev_build_status': 0,
                'consecutive_failures': 0
            }
        },
        {
            'name': "Scenario 3: Moderate Churn on Main Branch",
            'data': {
                'git_branch': 'main',
                'is_pull_request': 0,
                'num_commits': 2,
                'files_changed': 4,
                'lines_added': 140,
                'lines_deleted': 30,
                'churn_total': 170,
                'src_churn': 130,
                'test_churn': 40,
                'test_to_src_ratio': 0.31,
                'time_since_last_build_min': 45,
                'committer_experience': 40,
                'committer_recent_failure_rate': 0.14,
                'team_size': 20,
                'prev_build_status': 0,
                'consecutive_failures': 0
            }
        }
    ]
    
    for case in cases:
        print(f"--- {case['name']} ---")
        res = predict_single(case['data'], model, preprocessor)
        prob_pct = res['failure_probability'] * 100
        print(f"Predicted Failure Probability : {prob_pct:.1f}%")
        print(f"Assigned Risk Tier            : [{res['risk_level']}]")
        print(f"Actionable CI Guidance        : {res['recommendation']}")
        if res['risk_factors']:
            print("Flagged Risk Factors          :")
            for rf in res['risk_factors']:
                print(f"  • {rf}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Predict build failure risk for a commit/PR.")
    parser.add_argument('--demo', action='store_true', help="Run demonstration test cases")
    parser.add_argument('--files', type=int, default=2, help="Files changed")
    parser.add_argument('--churn', type=int, default=50, help="Total churn")
    parser.add_argument('--test-ratio', type=float, default=0.25, help="Test to source ratio")
    parser.add_argument('--prev-failed', type=int, choices=[0, 1], default=0, help="Previous build status (0=Pass, 1=Fail)")
    parser.add_argument('--consecutive-failures', type=int, default=0, help="Consecutive failures")
    parser.add_argument('--branch', type=str, default='feature', help="Git branch")
    parser.add_argument('--experience', type=int, default=25, help="Committer experience")
    
    args = parser.parse_args()
    
    if args.demo or len(sys.argv) == 1:
        run_demo()
    else:
        model, preprocessor, _ = load_artifacts()
        inp = {
            'files_changed': args.files,
            'churn_total': args.churn,
            'lines_added': int(args.churn * 0.7),
            'lines_deleted': int(args.churn * 0.3),
            'src_churn': int(args.churn / (1.0 + args.test_ratio)),
            'test_churn': int(args.churn * args.test_ratio / (1.0 + args.test_ratio)),
            'test_to_src_ratio': args.test_ratio,
            'prev_build_status': args.prev_failed,
            'consecutive_failures': args.consecutive_failures,
            'git_branch': args.branch,
            'committer_experience': args.experience
        }
        res = predict_single(inp, model, preprocessor)
        print("\n=== Prediction Result ===")
        print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
