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
import argparse
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
from src.model_registry import ModelRegistry


def main():
    """Main function to train all models."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/technician_feedback.csv',
        help='Path to the CSV file containing the dataset'
    )
    parser.add_argument(
        '--train-data-path',
        type=str,
        help='Optional explicit training CSV path (overrides --data-path)'
    )
    parser.add_argument(
        '--test-data-path',
        type=str,
        help='Optional explicit test CSV path'
    )
    parser.add_argument(
        '--text-col',
        type=str,
        default='feedback_text',
        help='Name of the text column in the CSV file'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        default='sentiment',
        help='Name of the label column in the CSV file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Default',
        help='Name identifier for the trained models'
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate default dataset if file not found'
    )

    args = parser.parse_args()

    train_path = args.train_data_path or args.data_path
    test_path = args.test_data_path

    # Create safe filename from model name
    safe_model_name = args.model_name.replace(' ', '_').replace('-', '_').lower()

    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Ensure directories exist
    ensure_directory('models')
    ensure_directory('data')
    
    print_section_header("Sentiment Analysis - Model Training")
    
    # Step 1: Load or generate data
    print("Step 1: Loading data...")
    print(f"Dataset path: {train_path}")
    print(f"Text column: {args.text_col}")
    print(f"Label column: {args.label_col}")

    if not os.path.exists(train_path):
        if args.generate and train_path == 'data/technician_feedback.csv':
            print("Dataset not found. Generating new dataset...")
            df = generate_technician_feedback(n_samples=550, random_seed=42)
            df.to_csv(train_path, index=False)
        else:
            print(f"Error: Dataset not found at {train_path}")
            print("Use --generate flag to generate default dataset, or provide a valid path.")
            sys.exit(1)

    df, texts, labels = load_data(train_path, text_column=args.text_col, label_column=args.label_col)
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
    
    # Step 3: Extract features
    print("\nStep 3: Extracting features...")
    tfidf = TFIDFExtractor(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(processed_texts)
    print(f"Feature matrix shape: {X.shape}")
    
    # Save vectorizer with dataset name
    vectorizer_path = f'models/tfidf_vectorizer_{safe_model_name}.joblib'
    save_model(tfidf.vectorizer, vectorizer_path)

    if test_path:
        if not os.path.exists(test_path):
            print(f"Error: Test dataset not found at {test_path}")
            sys.exit(1)
        test_df, test_texts, test_labels = load_data(test_path, text_column=args.text_col, label_column=args.label_col)
        processed_test_texts = preprocessor.preprocess_batch(test_texts)
        X_train = X
        y_train = labels
        X_test = tfidf.transform(processed_test_texts)
        y_test = test_labels
        split_info = f"Using provided test dataset ({test_path})"
    else:
        print("\nStep 4: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        split_info = "Performing random 80/20 split"
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

    print(f"\nSplit strategy: {split_info}")

    if test_path:
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
    registry = ModelRegistry()

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        results[name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        if test_path:
            print("  Evaluation used external test dataset")

        # Save model with dataset name
        model_filename = name.lower().replace(' ', '_')
        model_path = f'models/{model_filename}_{safe_model_name}.joblib'
        model.save(model_path)

        # Register model in registry
        registry.register_model(
            model_type=name,
            dataset_name=args.model_name,
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            metrics=metrics,
            text_column=args.text_col,
            label_column=args.label_col,
            dataset_path=train_path
        )

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
    
    # Save results with dataset name
    results_df = pd.DataFrame(results).T
    results_csv_path = f'models/model_results_{safe_model_name}.csv'
    results_df.to_csv(results_csv_path)
    print(f"\nResults saved to {results_csv_path}")

    # Print registry summary
    print("\n" + "=" * 60)
    print(f"✅ All models trained and registered for: {args.model_name}")
    print("=" * 60)
    registry.print_summary()
    if test_path:
        print("Test metrics reflect external dataset inputs.")

    print_section_header("Training Complete!")
    print("All models saved to models/ directory")
    print("\nTo use a model for prediction:")
    print("  from src.utils import load_model")
    print("  model = load_model('models/logistic_regression_model.joblib')")


if __name__ == "__main__":
    main()
