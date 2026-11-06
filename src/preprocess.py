"""
Data preprocessing, feature engineering, and temporal/stratified splitting
for Build-Failure Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

NUMERICAL_FEATURES = [
    'num_commits',
    'files_changed',
    'lines_added',
    'lines_deleted',
    'churn_total',
    'src_churn',
    'test_churn',
    'test_to_src_ratio',
    'time_since_last_build_min',
    'committer_experience',
    'committer_recent_failure_rate',
    'team_size',
    'consecutive_failures',
    'churn_per_file'
]

BINARY_FEATURES = [
    'is_pull_request',
    'prev_build_status'
]

CATEGORICAL_FEATURES = [
    'git_branch'
]

TARGET = 'build_failed'

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed derived features."""
    df = df.copy()
    
    # Safe churn per file
    df['churn_per_file'] = df['churn_total'] / np.maximum(df['files_changed'], 1)
    
    # Ensure test_to_src_ratio is bounded and clean
    df['test_to_src_ratio'] = df['test_to_src_ratio'].fillna(0.0)
    
    return df

def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn preprocessing ColumnTransformer."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract ordered transformed feature names."""
    feature_names = []
    # Numerical
    feature_names.extend(NUMERICAL_FEATURES)
    # Binary
    feature_names.extend(BINARY_FEATURES)
    # Categorical
    cat_encoder = preprocessor.named_transformers_['cat']
    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names.extend(cat_names)
    return feature_names

def prepare_data(data_path: str = 'data/ci_builds_dataset.csv', 
                 split_strategy: str = 'temporal', 
                 test_size: float = 0.2,
                 random_state: int = 42):
    """
    Load, engineer features, split into train and test sets, and fit transformers.
    
    In CI/CD software engineering, 'temporal' split avoids future data leakage.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run data_generator.py first.")
        
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    if split_strategy == 'temporal':
        # Sort chronologically by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    elif split_strategy == 'stratified':
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET])
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
        
    all_input_cols = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    
    X_train_raw = train_df[all_input_cols]
    y_train = train_df[TARGET].values
    
    X_test_raw = test_df[all_input_cols]
    y_test = test_df[TARGET].values
    
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    feature_names = get_feature_names(preprocessor)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }

if __name__ == '__main__':
    data = prepare_data()
    print("Data Preparation Complete:")
    print(f"X_train shape: {data['X_train'].shape}, y_train failures: {np.mean(data['y_train']):.2%}")
    print(f"X_test shape: {data['X_test'].shape}, y_test failures: {np.mean(data['y_test']):.2%}")
    print(f"Features ({len(data['feature_names'])}): {data['feature_names']}")

