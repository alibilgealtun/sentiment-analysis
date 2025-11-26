#!/usr/bin/env python
"""
Evaluate Models Script

This script loads trained models and evaluates them on the test data,
generating comprehensive evaluation reports and visualizations.

Usage:
    python scripts/evaluate_models.py

Author: Sentiment Analysis Team
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import project modules
from src.preprocessing import TextPreprocessor
from src.evaluation import (
    calculate_metrics,
    get_classification_report,
    plot_confusion_matrix,
    plot_roc_curves,
    compare_models,
    create_results_table,
    get_best_model
)
from src.utils import (
    load_data,
    load_model,
    set_random_seeds,
    print_section_header,
    ensure_directory
)


def main():
    """Main function to evaluate all models."""
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Ensure output directory exists
    ensure_directory('outputs')
    
    print_section_header("Sentiment Analysis - Model Evaluation")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    df, texts, labels = load_data('data/technician_feedback.csv')
    
    # Step 2: Preprocess text
    print("\nStep 2: Preprocessing text...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts)
    
    # Step 3: Load vectorizer and transform
    print("\nStep 3: Loading vectorizer and transforming data...")
    vectorizer = load_model('models/tfidf_vectorizer.joblib')
    X = vectorizer.transform(processed_texts)
    
    # Split data (same split as training)
    _, X_test, _, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Step 4: Load and evaluate models
    print("\nStep 4: Evaluating models...")
    
    model_files = {
        'Naive Bayes': 'models/naive_bayes_model.joblib',
        'SVM': 'models/svm_model.joblib',
        'Logistic Regression': 'models/logistic_regression_model.joblib',
        'Random Forest': 'models/random_forest_model.joblib'
    }
    
    results = {}
    all_predictions = {}
    all_probabilities = {}
    
    classes = ['negative', 'neutral', 'positive']
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            print(f"\nEvaluating {name}...")
            model_data = load_model(filepath)
            model = model_data['model']
            label_encoder = model_data['label_encoder']
            
            # Predict
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            y_proba = model.predict_proba(X_test)
            
            all_predictions[name] = y_pred
            all_probabilities[name] = y_proba
            
            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred)
            results[name] = metrics
            
            # Print classification report
            print(f"\n{name} Classification Report:")
            print(get_classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            fig = plot_confusion_matrix(
                y_test, y_pred,
                labels=classes,
                title=f'{name} - Confusion Matrix',
                save_path=f'outputs/{name.lower().replace(" ", "_")}_confusion_matrix.png'
            )
            plt.close(fig)
        else:
            print(f"Model file not found: {filepath}")
    
    # Step 5: Model comparison
    print_section_header("Model Comparison")
    
    # Create comparison table
    results_table = create_results_table(results)
    print(results_table.to_string(index=False))
    results_table.to_csv('outputs/model_comparison.csv', index=False)
    
    # Plot comparison
    if results:
        fig = compare_models(
            results,
            title='Model Performance Comparison',
            save_path='outputs/model_comparison.png'
        )
        plt.close(fig)
    
    # Plot ROC curves for best model
    if results:
        best_model_name, best_score = get_best_model(results)
        print(f"\nBest Model: {best_model_name} (F1-Score: {best_score:.4f})")
        
        if best_model_name in all_probabilities:
            # Transform y_test to match model's label encoding
            model_data = load_model(model_files[best_model_name])
            label_encoder = model_data['label_encoder']
            
            fig = plot_roc_curves(
                y_test,
                all_probabilities[best_model_name],
                classes=list(label_encoder.classes_),
                title=f'{best_model_name} - ROC Curves',
                save_path='outputs/best_model_roc_curves.png'
            )
            plt.close(fig)
    
    # Step 6: Summary
    print_section_header("Evaluation Complete!")
    
    print("Generated outputs:")
    print("  - outputs/model_comparison.csv")
    print("  - outputs/model_comparison.png")
    print("  - outputs/*_confusion_matrix.png")
    print("  - outputs/best_model_roc_curves.png")


if __name__ == "__main__":
    main()
