"""
Unit and integration tests for Build-Failure Prediction pipeline.
"""

import unittest
import os
import shutil
import numpy as np
import pandas as pd

from src.data_generator import generate_ci_dataset
from src.preprocess import engineer_features, prepare_data
from src.models import get_models
from src.predict import load_artifacts, predict_single

class TestBuildFailurePredictionPipeline(unittest.TestCase):
    
    def test_01_data_generator(self):
        """Test dataset generation and empirical schema."""
        df = generate_ci_dataset(num_samples=100, random_state=42)
        self.assertEqual(len(df), 100)
        self.assertIn('build_failed', df.columns)
        self.assertIn('test_to_src_ratio', df.columns)
        self.assertIn('prev_build_status', df.columns)
        # Failure rate should be between 5% and 40%
        fail_rate = df['build_failed'].mean()
        self.assertTrue(0.05 <= fail_rate <= 0.40, f"Unexpected fail rate: {fail_rate}")

    def test_02_feature_engineering(self):
        """Test feature engineering logic."""
        df = pd.DataFrame([{
            'files_changed': 5,
            'churn_total': 250,
            'test_to_src_ratio': np.nan
        }])
        df_feat = engineer_features(df)
        self.assertEqual(df_feat['churn_per_file'].iloc[0], 50.0)
        self.assertEqual(df_feat['test_to_src_ratio'].iloc[0], 0.0)

    def test_03_data_split_and_transforms(self):
        """Test temporal split and scaling transformation."""
        data = prepare_data(data_path='data/ci_builds_dataset.csv', split_strategy='temporal', test_size=0.2)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(len(X_train) + len(X_test), 6000)
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())

    def test_04_models_instantiation(self):
        """Verify candidate model dictionary."""
        models = get_models()
        self.assertIn('Logistic_Regression', models)
        self.assertIn('Random_Forest', models)
        self.assertIn('Gradient_Boosting', models)

    def test_05_inference_system(self):
        """Test trained model inference and risk evaluation."""
        model, preprocessor, metadata = load_artifacts(model_dir='models')
        sample_low_risk = {
            'files_changed': 1,
            'churn_total': 20,
            'test_to_src_ratio': 0.8,
            'prev_build_status': 0,
            'committer_experience': 50,
            'git_branch': 'main'
        }
        res_low = predict_single(sample_low_risk, model, preprocessor)
        self.assertIn('failure_probability', res_low)
        self.assertIn('risk_level', res_low)
        self.assertLess(res_low['failure_probability'], 0.5)

        sample_high_risk = {
            'files_changed': 15,
            'churn_total': 900,
            'test_to_src_ratio': 0.0,
            'prev_build_status': 1,
            'consecutive_failures': 3,
            'committer_experience': 2,
            'git_branch': 'hotfix'
        }
        res_high = predict_single(sample_high_risk, model, preprocessor)
        self.assertGreater(res_high['failure_probability'], 0.6)

if __name__ == '__main__':
    unittest.main()

