"""
Traditional machine learning models for heart disease prediction.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import os

def train_logistic_regression(X_train, y_train):
    """Train baseline Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, tune=False):
    """Train Random Forest model with optional tuning."""
    if tune:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = RandomizedSearchCV(
            rf, param_grid, n_iter=5, cv=3, 
            scoring='roc_auc', random_state=42, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best Random Forest params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        return model

def train_svm(X_train, y_train, tune=False):
    """Train SVM model with optional tuning."""
    if tune:
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best SVM params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)
        return model

def train_xgboost(X_train, y_train, tune=False):
    """Train XGBoost model with optional tuning."""
    if tune:
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [50, 100]
        }
        
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        grid_search = RandomizedSearchCV(
            xgb, param_grid, n_iter=5, cv=3, 
            scoring='roc_auc', random_state=42, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best XGBoost params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        return model

def save_model(model, name):
    """Save trained model to file."""
    os.makedirs('models', exist_ok=True)
    filename = f'models/{name}.pkl'
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")