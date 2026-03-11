"""
Data preprocessing module for heart disease prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data(data_path='data/heart_disease.csv'):
    """
    Load and preprocess the heart disease dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"After dropping missing values: {df.shape}")
    
    # Convert target to binary (0 = no disease, 1 = disease)
    df['num'] = (df['num'] > 0).astype(int)
    
    # Separate features and target
    X = df.drop('num', axis=1)
    y = df['num']
    
    # One-hot encode categorical features
    categorical_features = ['cp', 'restecg', 'slope', 'thal']
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify numerical features for scaling
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Save scaler for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"\nPreprocessing complete:")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Features: {list(X_train_scaled.columns)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X_train_scaled.columns)

if __name__ == "__main__":
    # Test the preprocessing
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data()
    print("\nFirst 5 rows of training data:")
    print(X_train.head())