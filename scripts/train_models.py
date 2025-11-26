#!/usr/bin/env python
"""
Train Models Script

This script trains all sentiment classification models and saves them
to the models directory.

Usage:
    python scripts/train_models.py

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
import joblib
from sklearn.model_selection import train_test_split

# Import project modules
from src.data_generator import generate_technician_feedback
from src.preprocessing import TextPreprocessor
from src.feature_extraction import TFIDFExtractor
from src.models import (
    NaiveBayesClassifier,
    SVMClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifier
)
from src.evaluation import calculate_metrics, get_classification_report
from src.utils import (
    load_data,
    save_model,
    set_random_seeds,
    print_section_header,
    ensure_directory
)


def main():
    """Main function to train all models."""
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Ensure directories exist
    ensure_directory('models')
    ensure_directory('data')
    
    print_section_header("Sentiment Analysis - Model Training")
    
    # Step 1: Load or generate data
    print("Step 1: Loading data...")
    data_path = 'data/technician_feedback.csv'
    
    if not os.path.exists(data_path):
        print("Dataset not found. Generating new dataset...")
        df = generate_technician_feedback(n_samples=550, random_seed=42)
        df.to_csv(data_path, index=False)
    
    df, texts, labels = load_data(data_path)
    print(f"\nLoaded {len(texts)} samples")
    
    # Step 2: Preprocess text
    print("\nStep 2: Preprocessing text...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_stopwords=True,
        lemmatize=True
    )
    
    processed_texts = preprocessor.preprocess_batch(texts)
    print(f"Preprocessed {len(processed_texts)} texts")
    
    # Step 3: Feature extraction
    print("\nStep 3: Extracting features...")
    tfidf = TFIDFExtractor(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(processed_texts)
    print(f"Feature matrix shape: {X.shape}")
    
    # Save vectorizer
    save_model(tfidf.vectorizer, 'models/tfidf_vectorizer.joblib')
    
    # Step 4: Split data
    print("\nStep 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Train models
    print("\nStep 5: Training models...")
    models = {
        'Naive Bayes': NaiveBayesClassifier(alpha=1.0),
        'SVM': SVMClassifier(kernel='linear', C=1.0),
        'Logistic Regression': LogisticRegressionClassifier(C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        results[name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Save model
        model_filename = name.lower().replace(' ', '_')
        model.save(f'models/{model_filename}_model.joblib')
    
    # Step 6: Print results summary
    print_section_header("Results Summary")
    
    print("\nModel Performance Comparison:")
    print("-" * 50)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:.4f}       {metrics['f1_score']:.4f}")
    
    print("-" * 50)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/model_results.csv')
    print("\nResults saved to models/model_results.csv")
    
    print_section_header("Training Complete!")
    print("All models saved to models/ directory")
    print("\nTo use a model for prediction:")
    print("  from src.utils import load_model")
    print("  model = load_model('models/logistic_regression_model.joblib')")


if __name__ == "__main__":
    main()
