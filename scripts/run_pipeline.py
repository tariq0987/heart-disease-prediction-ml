#!/usr/bin/env python
"""
End-to-end pipeline for heart disease prediction project.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import local modules
from src.data_preprocessing import load_and_preprocess_data
from src.models.traditional_models import (
    train_logistic_regression, train_random_forest, 
    train_svm, train_xgboost, save_model
)
from src.models.neural_networks import (
    create_basic_nn, create_deep_nn, train_neural_network, save_keras_model
)
from src.evaluation import (
    evaluate_model, plot_confusion_matrix, save_results_table, plot_roc_curves, plot_feature_importance
)

def main():
    """Run the complete heart disease prediction pipeline."""
    
    print("=" * 60)
    print("HEART DISEASE PREDICTION PROJECT")
    print("=" * 60)
    print("Group Members: Tariq Ahmed, Rayyan Omer, Sameer Khan, Rafay Khan")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n[1/7] Loading and preprocessing data...")
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
        print("✓ Data preprocessing complete!")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Make sure you've run data/download_data.py first!")
        return
    
    # Step 2: Train traditional ML models
    print("\n[2/7] Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("✓ Logistic Regression complete!")
    
    print("\n[3/7] Training Random Forest (with tuning)...")
    rf_model = train_random_forest(X_train, y_train, tune=True)
    print("✓ Random Forest complete!")
    
    print("\n[4/7] Training SVM (with tuning)...")
    svm_model = train_svm(X_train, y_train, tune=True)
    print("✓ SVM complete!")
    
    print("\n[5/7] Training XGBoost (with tuning)...")
    xgb_model = train_xgboost(X_train, y_train, tune=True)
    print("✓ XGBoost complete!")
    
    # Step 3: Train neural networks
    print("\n[6/7] Training Basic Neural Network...")
    basic_nn = create_basic_nn(X_train.shape[1])
    basic_nn, history_basic = train_neural_network(
        basic_nn, X_train, y_train, X_test, y_test, epochs=30
    )
    print("✓ Basic Neural Network complete!")
    
    # Step 4: Evaluate all models
    print("\n[7/7] Evaluating all models...")
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model,
        'SVM': svm_model,
        'XGBoost': xgb_model,
        'Basic NN': basic_nn
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        is_nn = 'NN' in name
        metrics, y_pred, y_proba = evaluate_model(
            model, X_test, y_test, name, predict_proba=not is_nn
        )
        results[name] = metrics
        predictions[name] = (y_pred, y_proba)
    
    # Step 5: Create visualizations
    print("\nCreating visualizations...")
    
    # Confusion matrices for top models
    for name in ['XGBoost', 'Random Forest']:
        if name in predictions:
            plot_confusion_matrix(y_test, predictions[name][0], name)
    
    # ROC curves
    roc_models = {
        'Logistic Regression': (lr_model, 'blue'),
        'Random Forest': (rf_model, 'green'),
        'SVM': (svm_model, 'red'),
        'XGBoost': (xgb_model, 'purple'),
        'Basic NN': (basic_nn, 'orange')
    }
    plot_roc_curves(roc_models, X_test, y_test)
    
    # Feature importance
    plot_feature_importance(rf_model, feature_names)
    
    # Step 6: Save results
    print("\nSaving results...")
    results_df = save_results_table(results)
    
    # Save models
    save_model(lr_model, 'logistic_regression')
    save_model(rf_model, 'random_forest')
    save_model(svm_model, 'svm')
    save_model(xgb_model, 'xgboost')
    save_keras_model(basic_nn, 'basic_nn')
    
    # Step 7: Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Find best model
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\n🏆 BEST MODEL: {best_model.name}")
    print(f"   Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall:    {best_model['Recall']:.4f}")
    print(f"   F1-Score:  {best_model['F1-Score']:.4f}")
    if 'ROC-AUC' in best_model:
        print(f"   ROC-AUC:   {best_model['ROC-AUC']:.4f}")
    
    print("\n" + "=" * 60)
    print("All Models Comparison:")
    print("=" * 60)
    print(results_df.round(4))
    
    print("\n" + "=" * 60)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nResults saved in:")
    print("  - 'results/model_comparison.csv'")
    print("  - 'results/figures/' folder")
    print("\nModels saved in:")
    print("  - 'models/' folder")

if __name__ == "__main__":
    main()