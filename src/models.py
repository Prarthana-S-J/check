import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_model_definitions():
    """
    Returns a dictionary of the 5 candidate models from diverse algorithm families.
    """
    models = {
        'Logistic_Regression': {
            'family': 'Linear and Probabilistic',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs'))
            ]),
            'config': {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
            'objective': 'Binary Cross-Entropy (Logistic Loss)'
        },
        'Random_Forest': {
            'family': 'Ensemble (Bagging)',
            'pipeline': Pipeline([
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
            ]),
            'config': {'n_estimators': 100, 'max_depth': 12, 'criterion': 'gini'},
            'objective': 'Gini Impurity Minimization'
        },
        'LightGBM': {
            'family': 'Gradient Boosting',
            'pipeline': Pipeline([
                ('classifier', LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1))
            ]),
            'config': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 8, 'num_leaves': 31},
            'objective': 'Binary Logarithmic Loss'
        },
        'XGBoost': {
            'family': 'Extreme Gradient Boosting',
            'pipeline': Pipeline([
                ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss'))
            ]),
            'config': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            'objective': 'Second-Order Taylor Binary Logloss'
        },
        'Neural_Network_MLP': {
            'family': 'Deep Neural Network (MLP)',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, alpha=1e-4, random_state=42, early_stopping=True, n_iter_no_change=5))
            ]),
            'config': {'hidden_layers': '(64, 32)', 'activation': 'relu', 'optimizer': 'adam', 'alpha': 1e-4},
            'objective': 'Binary Cross-Entropy Loss with Adam Optimizer'
        }
    }
    return models

