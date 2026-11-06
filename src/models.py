"""
Model definitions and factories for Build-Failure Prediction benchmarking.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def get_models(random_state: int = 42) -> dict:
    """
    Returns a dictionary of candidate models ranging from baseline to advanced ensembles.
    Includes class-weight handling where supported.
    """
    return {
        'Dummy_Baseline': DummyClassifier(
            strategy='most_frequent'
        ),
        'Logistic_Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            random_state=random_state
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=random_state
        ),
        'Multi_Layer_Perceptron': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=300,
            early_stopping=True,
            random_state=random_state
        )
    }

